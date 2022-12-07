from collections import defaultdict
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
from tqdm import tqdm

torch.set_default_dtype(torch.float32)

device = torch.device("cuda")
datasets = {'planar': PlanarDataset, 'pendulum': GymPendulumDatasetV2}
settings = {'planar': (1600, 4, 2, 4), 'pendulum': (4608, 3, 1, 4)}
samplers = {'planar': planar_sampler, 'pendulum': pendulum_sampler, 'cartpole': cartpole_sampler}
num_eval = 10 # number of images evaluated on tensorboard


def compute_loss(x, u_t, x_next, q_z_next, x_recon, x_next_pred, q_z, q_z_next_pred, cost, cost_pred, transition_model, z_residual, dyn_mats, hparams, recon_loss='bce'):
    if recon_loss == 'bce':
        recon_loss = torch.nn.BCELoss(reduction='none')
    elif recon_loss == 'mse':
        recon_loss = torch.nn.MSELoss(reduction='none')
    elif recon_loss == 'l1':
        recon_loss = torch.nn.L1Loss(reduction='none')

    A_t, B_t, _, G_t, H_t, _ = dyn_mats

    # lower-bound loss
    recon_term = recon_loss(x_recon, x).sum(dim=1).mean(dim=0)
    pred_loss = recon_loss(x_next_pred, x_next).sum(dim=1).mean(dim=0)

    kl_term = - 0.5 * torch.mean(torch.sum(1 + q_z.logvar - q_z.mean.pow(2) - q_z.logvar.exp(), dim = 1))

    lower_bound = recon_term + pred_loss + kl_term

    # consistency loss
    consis_term = NormalDistribution.KL_divergence(q_z_next_pred, q_z_next).sum(dim=-1)

    cost_term = F.mse_loss(cost_pred, cost)

    # jacobian loss
    zbar = q_z.mean
    dz = torch.randn(zbar.shape, device=device) * 0.3
    du = torch.randn(u_t.shape, device=device) * 0.3
    zhat = zbar + dz
    uhat = u_t + du
    dz_next_jac = A_t.bmm(dz.unsqueeze(-1)).squeeze(-1) + B_t.bmm(du.unsqueeze(-1)).squeeze(-1)
    _, zhat_next_true, zhat_residual, _ = transition_model.forward(zbar, NormalDistribution(zhat, q_z.logvar), uhat)
    zhat_next_jac = NormalDistribution(q_z_next_pred.mean + dz_next_jac, q_z.logvar, A=A_t)
    dresidual_jac = G_t.bmm(dz.unsqueeze(-1)).squeeze(-1) + H_t.bmm(du.unsqueeze(-1)).squeeze(-1)

    jac_loss_dyn = NormalDistribution.KL_divergence(
        zhat_next_jac,
        zhat_next_true,
        # NormalDistribution(zhat_next_true.mean.detach(), zhat_next_true.logvar.detach(), A=zhat_next_true.A.detach()),
    ).sum(dim=-1)
    jac_loss_res = F.mse_loss(z_residual + dresidual_jac, zhat_residual)

    results = lower_bound + hparams.lam * consis_term + cost_term + hparams.jac_weight * (jac_loss_dyn + jac_loss_res), {
        'recon': recon_term.item(),
        'pred': pred_loss.item(),
        'kl': kl_term.item(),
        'consis': consis_term.item(),
        'cost': cost_term.item(),
        'jac_dyn': jac_loss_dyn.item(),
        'jac_cost': jac_loss_res.item(),
        'jac_dyn_err': (zhat_next_jac.mean - zhat_next_true.mean).pow(2).sum(-1).mean().sqrt().item(),
        'dyn_error': (q_z_next_pred.mean - q_z_next.mean).pow(2).sum(-1).mean().sqrt().item(),
        'z_std_rms': q_z.mean.pow(2).sum(-1).mean().sqrt().item(),
    }
    return results

def train(model, train_loader, optimizer, global_step, hparams):
    model.train()
    avg_loss = 0.0

    metrics = defaultdict(float)

    num_batches = len(train_loader)
    for i, (x, u, x_next) in tqdm(enumerate(train_loader, 0), total=num_batches):
        # TODO: Load cost in train_loader
        x = x.float()
        u = u.float()
        x_next = x_next.float()

        x = x.view(-1, model.obs_dim).to(device)
        u = u.to(device)
        cost = torch.zeros(x.shape[0]).to(device)
        x_next = x_next.view(-1, model.obs_dim).to(device)
        optimizer.zero_grad()

        x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next, cost_res_pred, dyn_mats = model(x, u, x_next)
        cost_pred = cost_res_pred.pow(2).sum(dim=1)
        loss, metrics_it = compute_loss(x, u, x_next, q_z_next, x_recon, x_next_pred, q_z, q_z_next_pred, cost, cost_pred, model.trans, cost_res_pred, dyn_mats, hparams)

        avg_loss += loss.item()
        loss.backward()
        optimizer.step()

        for key, value in metrics_it.items():
            metrics[key] += value

    return avg_loss / num_batches, {key: value / num_batches for key, value in metrics.items()}

def compute_log_likelihood(x, x_recon, x_next, x_next_pred):
    loss_1 = torch.nn.BCELoss(reduction='none')(x_recon, x).sum(dim=1).mean(dim=0)
    loss_2 = torch.nn.BCELoss(reduction='none')(x_next_pred, x_next).sum(dim=1).mean(dim=0)
    return loss_1, loss_2

def evaluate(model, test_loader, hparams):
    model.eval()
    num_batches = len(test_loader)
    metrics = defaultdict(float)
    with torch.no_grad():
        for x, u, x_next in test_loader:
            x = x.float()
            u = u.float()
            x_next = x_next.float()

            cost = torch.zeros(x.shape[0]).to(device)
            x = x.view(-1, model.obs_dim).to(device)
            u = u.to(device)
            x_next = x_next.view(-1, model.obs_dim).to(device)

            x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next, cost_res_pred, dyn_mats = model(x, u, x_next)
            cost_pred = cost_res_pred.pow(2).sum(dim=1)
            _, metrics_it = compute_loss(x, u, x_next, q_z_next, x_recon, x_next_pred, q_z, q_z_next_pred, cost, cost_pred, model.trans, cost_res_pred, dyn_mats, hparams)
            for key, value in metrics_it.items():
                metrics[key] += value

    return {key: value / num_batches for key, value in metrics.items()}

# code for visualizing the training process
def predict_x_next(model, env, num_eval):
    # frist sample a true trajectory from the environment
    sampler = samplers[env]
    state_samples, sampled_data = sampler.sample(num_eval)

    # use the trained model to predict the next observation
    predicted = []
    for x, u, x_next in sampled_data:
        x_reshaped = x.reshape(-1)
        x_reshaped = torch.from_numpy(x_reshaped).float().unsqueeze(dim=0).to(device)
        u = torch.from_numpy(u).float().unsqueeze(dim=0).to(device)
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

    if args.z_dim is None:
        args.z_dim = z_dim
    else:
        z_dim = args.z_dim

    if args.r_dim is None:
        args.r_dim = r_dim
    else:
        r_dim = args.r_dim

    model = E2C(obs_dim=obs_dim, z_dim=z_dim, u_dim=u_dim, r_dim=r_dim, env=env_name).to(device)

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=lr, weight_decay=weight_decay)

    writer = None

    log_dir = log_dir + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    result_path = './result/' + env_name + '/' + log_dir
    if not path.exists(result_path):
        os.makedirs(result_path)
    with open(result_path + '/settings', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    for i in range(epoches):
        avg_loss, train_metrics = train(model, train_loader, optimizer, global_step=i, hparams=args)
        print('Epoch %d' % i)
        print("Training loss: %f" % (avg_loss))
        # evaluate on test set
        eval_metrics = evaluate(model, test_loader, hparams=args)
        print('State loss: ' + str(eval_metrics['recon']))
        print('Next state loss: ' + str(eval_metrics['pred']))

        if writer is None:
            writer = SummaryWriter('logs/' + env_name + '/' + log_dir, comment=args.comment)
            writer.add_hparams(args.__dict__, {})

        # ...log the running loss
        writer.add_scalar('eval/training loss', avg_loss, i)
        for key, value in eval_metrics.items():
            writer.add_scalar('eval/' + key, value, i)
        for key, value in train_metrics.items():
            writer.add_scalar('train/' + key, value, i)

        # save model
        if (i + 1) % iter_save == 0:
            writer.add_figure('actual vs. predicted observations',
                              plot_preds(model, env_name, num_eval),
                              global_step=i)
            print('Saving the model.............')

            torch.save(model.state_dict(), result_path + '/model_' + str(i + 1))
            with open(result_path + '/loss_' + str(i + 1), 'w') as f:
                f.write('\n'.join([str(eval_metrics['recon']), str(eval_metrics['pred'])]))

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
    parser.add_argument('--jac_weight', default=1.0, type=float, help='the weight of the Jacobian terms')
    parser.add_argument('--num_iter', default=5000, type=int, help='the number of epoches')
    parser.add_argument('--iter_save', default=1000, type=int, help='save model and result after this number of iterations')
    parser.add_argument('--log_dir', required=True, type=str, help='the directory to save training log')
    parser.add_argument('--seed', required=True, type=int, help='seed number')
    parser.add_argument('--z_dim', type=int, help='latent space dimension (default to environment setting)')
    parser.add_argument('--r_dim', type=int, help='residual dimension (default to environment setting)')
    parser.add_argument('--comment', type=str, help='comment to append to logdir', default='')

    args = parser.parse_args()

    main(args)
