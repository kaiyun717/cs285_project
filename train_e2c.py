import datetime
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import sys

from normal import *
from e2c_model import E2C
from datasets import *
import data.sample_planar as planar_sampler
import data.sample_pendulum_data as pendulum_sampler
import data.sample_cartpole_data as cartpole_sampler

torch.set_default_dtype(torch.float64)

device = torch.device("cuda")
datasets = {'planar': PlanarDataset, 'pendulum': GymPendulumDatasetV2}
settings = {'planar': (1600, 4, 2, 4), 'pendulum': (4608, 3, 1, 4)}
samplers = {'planar': planar_sampler, 'pendulum': pendulum_sampler, 'cartpole': cartpole_sampler}
num_eval = 10 # number of images evaluated on tensorboard

def compute_jac_loss(q_z, u_t, model):
    h_t = model.forward_h(q_z.mean)
    A_t, v, r = model.forward_A(h_t=h_t, return_vr=True)
    B_t = model.forward_B(h_t=h_t)

    G_t = model.forward_G(h_t=h_t)
    H_t = model.forward_H(h_t=h_t)

    device = q_z.mean.device

    # jacobian loss
    zbar = q_z.mean
    dz = torch.randn(zbar.shape, device=device) * 0.3
    du = torch.randn(u_t.shape, device=device) * 0.3
    zhat = zbar + dz
    uhat = u_t + du
    dz_next_jac = A_t.bmm(dz.unsqueeze(-1)).squeeze(-1) + B_t.bmm(du.unsqueeze(-1)).squeeze(-1)
    _, z_next_true, z_residual = model.forward(q_z.mean, q_z, uhat)
    _, zhat_next_true, zhat_residual = model.forward(zbar, NormalDistribution(zhat, q_z.logvar), uhat)
    zhat_next_jac = NormalDistribution(z_next_true.mean + dz_next_jac, q_z.logvar, A=A_t, v=v.squeeze(), r=r.squeeze())
    dresidual_jac = G_t.bmm(dz.unsqueeze(-1)).squeeze(-1) + H_t.bmm(du.unsqueeze(-1)).squeeze(-1)

    loss_dyn = NormalDistribution.KL_divergence(
        zhat_next_jac,
        zhat_next_true,
    )
    loss_res = F.mse_loss(z_residual + dresidual_jac, zhat_residual)
    return 10 * loss_dyn + loss_res, {'jac_dyn': loss_dyn.item(), 'jac_res': loss_res.item()}


def compute_loss(x, x_next, q_z_next, x_recon, x_next_pred, q_z, q_z_next_pred, lamda, cost, cost_pred, A_t=None, B_t=None, u_t=None, recon_loss='bce'):
    if recon_loss == 'bce':
        recon_loss = torch.nn.BCELoss(reduction='none')
    elif recon_loss == 'mse':
        recon_loss = torch.nn.MSELoss(reduction='none')
    elif recon_loss == 'l1':
        recon_loss = torch.nn.L1Loss(reduction='none')

    # lower-bound loss
    recon_term = recon_loss(x_recon, x).sum(dim=1).mean(dim=0)
    pred_loss = recon_loss(x_next_pred, x_next).sum(dim=1).mean(dim=0)

    kl_term = - 0.5 * torch.mean(torch.sum(1 + q_z.logvar - q_z.mean.pow(2) - q_z.logvar.exp(), dim = 1))

    lower_bound = recon_term + pred_loss + kl_term

    # consistency loss
    consis_term = NormalDistribution.KL_divergence(q_z_next_pred, q_z_next)

    cost_term = F.mse_loss(cost_pred, cost)

    return lower_bound + lamda * consis_term + cost_term, {'recon': recon_term.item(), 'pred': pred_loss.item(), 'kl': kl_term.item(), 'consis': consis_term.item(), 'cost': cost_term.item()}

def train(model, train_loader, lam, optimizer, writer, global_step):
    model.train()
    avg_loss = 0.0

    num_batches = len(train_loader)
    for i, (x, u, x_next) in enumerate(train_loader, 0):
        # TODO: Load cost in train_loader

        x = x.view(-1, model.obs_dim).to(device)
        u = u.to(device)
        cost = torch.zeros(x.shape[0]).to(device)
        x_next = x_next.view(-1, model.obs_dim).to(device)
        optimizer.zero_grad()

        x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next, cost_pred = model(x, u, x_next)

        loss, metrics = compute_loss(x, x_next, q_z_next, x_recon, x_next_pred, q_z, q_z_next_pred, lam, cost, cost_pred)
        loss_jac, metrics_jac = compute_jac_loss(q_z, u, model.trans)
        metrics = {**metrics, **metrics_jac}

        loss += loss_jac

        avg_loss += loss.item()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            for key, value in metrics.items():
                writer.add_scalar(f"train/{key}", value, global_step * num_batches + i)

    return avg_loss / num_batches

def compute_log_likelihood(x, x_recon, x_next, x_next_pred):
    loss_1 = torch.nn.BCELoss(reduction='none')(x_recon, x).sum(dim=1).mean(dim=0)
    loss_2 = torch.nn.BCELoss(reduction='none')(x_next_pred, x_next).sum(dim=1).mean(dim=0)
    return loss_1, loss_2

def evaluate(model, test_loader):
    model.eval()
    num_batches = len(test_loader)
    state_loss, next_state_loss = 0., 0.
    with torch.no_grad():
        for x, u, x_next in test_loader:
            x = x.view(-1, model.obs_dim).to(device)
            u = u.to(device)
            x_next = x_next.view(-1, model.obs_dim).to(device)

            x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next, cost_pred = model(x, u, x_next)
            loss_1, loss_2 = compute_log_likelihood(x, x_recon, x_next, x_next_pred)
            state_loss += loss_1
            next_state_loss += loss_2

    return state_loss.item() / num_batches, next_state_loss.item() / num_batches

# code for visualizing the training process
def predict_x_next(model, env, num_eval):
    # frist sample a true trajectory from the environment
    sampler = samplers[env]
    state_samples, sampled_data = sampler.sample(num_eval)

    # use the trained model to predict the next observation
    predicted = []
    for x, u, x_next in sampled_data:
        x_reshaped = x.reshape(-1)
        x_reshaped = torch.from_numpy(x_reshaped).unsqueeze(dim=0).to(device)
        u = torch.from_numpy(u).unsqueeze(dim=0).to(device)
        with torch.no_grad():
            x_next_pred, _ = model.predict(x_reshaped, u)
        predicted.append(x_next_pred.squeeze().cpu().numpy().reshape(sampler.width, sampler.height))
    true_x_next = [data[-1] for data in sampled_data]
    return true_x_next, predicted

def plot_preds(model, env, num_eval):
    true_x_next, pred_x_next = predict_x_next(model, env, num_eval)

    # plot the predicted and true observations
    fig, axes =plt.subplots(nrows=2, ncols=num_eval)
    plt.setp(axes, xticks=[], yticks=[])
    pad = 5
    axes[0, 0].annotate('True observations', xy=(0, 0.5), xytext=(-axes[0,0].yaxis.labelpad - pad, 0),
                   xycoords=axes[0,0].yaxis.label, textcoords='offset points',
                   size='large', ha='right', va='center')
    axes[1, 0].annotate('Predicted observations', xy=(0, 0.5), xytext=(-axes[1, 0].yaxis.labelpad - pad, 0),
                        xycoords=axes[1, 0].yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')

    for idx in np.arange(num_eval):
        axes[0, idx].imshow(true_x_next[idx], cmap='Greys')
        axes[1, idx].imshow(pred_x_next[idx], cmap='Greys')
    fig.tight_layout()
    return fig

def main(args):
    env_name = args.env
    assert env_name in ['planar', 'pendulum']
    propor = args.propor
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.decay
    lam = args.lam
    epoches = args.num_iter
    iter_save = args.iter_save
    log_dir = args.log_dir
    seed = args.seed

    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = datasets[env_name]('data/data/' + env_name)
    train_set, test_set = dataset[:int(len(dataset) * propor)], dataset[int(len(dataset) * propor):]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)

    obs_dim, z_dim, u_dim, r_dim = settings[env_name]
    model = E2C(obs_dim=obs_dim, z_dim=z_dim, u_dim=u_dim, r_dim=r_dim, env=env_name).to(device)

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=lr, weight_decay=weight_decay)

    writer = SummaryWriter('logs/' + env_name + '/' + log_dir + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    result_path = './result/' + env_name + '/' + log_dir
    if not path.exists(result_path):
        os.makedirs(result_path)
    with open(result_path + '/settings', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    for i in range(epoches):
        avg_loss = train(model, train_loader, lam, optimizer, writer, global_step=i)
        print('Epoch %d' % i)
        print("Training loss: %f" % (avg_loss))
        # evaluate on test set
        state_loss, next_state_loss = evaluate(model, test_loader)
        print('State loss: ' + str(state_loss))
        print('Next state loss: ' + str(next_state_loss))

        # ...log the running loss
        writer.add_scalar('eval/training loss', avg_loss, i)
        writer.add_scalar('eval/state loss', state_loss, i)
        writer.add_scalar('eval/next state loss', next_state_loss, i)

        # save model
        if (i + 1) % iter_save == 0:
            writer.add_figure('actual vs. predicted observations',
                              plot_preds(model, env_name, num_eval),
                              global_step=i)
            print('Saving the model.............')

            torch.save(model.state_dict(), result_path + '/model_' + str(i + 1))
            with open(result_path + '/loss_' + str(i + 1), 'w') as f:
                f.write('\n'.join([str(state_loss), str(next_state_loss)]))

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train e2c model')

    # the default value is used for the planar task
    parser.add_argument('--env', required=True, type=str, help='the environment used for training')
    parser.add_argument('--propor', default=3/4, type=float, help='the proportion of data used for training')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--lr', default=0.0005, type=float, help='the learning rate')
    parser.add_argument('--decay', default=0.001, type=float, help='the L2 regularization')
    parser.add_argument('--lam', default=0.25, type=float, help='the weight of the consistency term')
    parser.add_argument('--num_iter', default=5000, type=int, help='the number of epoches')
    parser.add_argument('--iter_save', default=1000, type=int, help='save model and result after this number of iterations')
    parser.add_argument('--log_dir', required=True, type=str, help='the directory to save training log')
    parser.add_argument('--seed', required=True, type=int, help='seed number')

    args = parser.parse_args()

    main(args)
