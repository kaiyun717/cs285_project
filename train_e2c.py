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

import wandb

torch.set_default_dtype(torch.float64)

device = torch.device("cuda")
datasets = {'planar': PlanarDataset, 'pendulum': GymPendulumDatasetV2}
settings = {'planar': (1600, 3, 2), 'pendulum': (4608, 5, 1)}
samplers = {'planar': planar_sampler, 'pendulum': pendulum_sampler, 'cartpole': cartpole_sampler}
num_eval = 10 # number of images evaluated on tensorboard

# dataset = datasets['planar']('./data/data/' + 'planar')
# x, u, x_next = dataset[0]
# imgplot = plt.imshow(x.squeeze(), cmap='gray')
# plt.show()
# print (np.array(u, dtype=float))
# imgplot = plt.imshow(x_next.squeeze(), cmap='gray')
# plt.show()

def compute_loss(x, x_next, q_z_next, x_recon, x_next_pred, q_z, q_z_next_pred, lamda):
    # lower-bound loss
    recon_term = -torch.mean(torch.sum(x * torch.log(1e-5 + x_recon)
                                       + (1 - x) * torch.log(1e-5 + 1 - x_recon), dim=1))
    pred_loss = -torch.mean(torch.sum(x_next * torch.log(1e-5 + x_next_pred)
                                      + (1 - x_next) * torch.log(1e-5 + 1 - x_next_pred), dim=1))

    kl_term = - 0.5 * torch.mean(torch.sum(1 + q_z.logvar - q_z.mean.pow(2) - q_z.logvar.exp(), dim = 1))

    lower_bound = recon_term + pred_loss + kl_term

    # consistency loss
    consis_term = NormalDistribution.KL_divergence(q_z_next_pred, q_z_next)
    return lower_bound + lamda * consis_term

def train(model, train_loader, lam, jac_loss_weight, cost_loss_weight, optimizer, cost_loss_type):
    model.train()
    avg_loss = 0.0

    num_batches = len(train_loader)
    for i, (x, u, cost, x_next) in enumerate(train_loader, 0):
        x = x.view(-1, model.obs_dim).double().to(device)
        u = u.double().to(device)
        cost = cost.double().to(device)
        x_next = x_next.view(-1, model.obs_dim).double().to(device)
        optimizer.zero_grad()

        x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next, cost_pred, residual = model(x, u, x_next)
        # Sample perturbed jac loss linearization
        zbar2 = model.reparam(q_z.mean, q_z.logvar)
        z_next_pred_zbar2, q_z_next_pred_zbar2, res_zbar2 = model.transition(zbar2, q_z, u)

        loss = compute_loss(x, x_next, q_z_next, x_recon, x_next_pred, q_z, q_z_next_pred, lam)
        jac_loss = NormalDistribution.KL_divergence(q_z_next_pred_zbar2, q_z_next_pred) + 10 * F.mse_loss(residual, res_zbar2)
        if cost_loss_type == 'mse':
            cost_loss = F.mse_loss(cost, cost_pred)
        elif cost_loss_type == 'l1':
            cost_loss = F.l1_loss(cost, cost_pred)
        loss += jac_loss_weight * jac_loss + cost_loss_weight * cost_loss

        avg_loss += loss.item()
        loss.backward()
        optimizer.step()

    return avg_loss / num_batches

def compute_log_likelihood(x, x_recon, x_next, x_next_pred):
    loss_1 = -torch.mean(torch.sum(x * torch.log(1e-5 + x_recon)
                                   + (1 - x) * torch.log(1e-5 + 1 - x_recon), dim=1))
    loss_2 = -torch.mean(torch.sum(x_next * torch.log(1e-5 + x_next_pred)
                                   + (1 - x_next) * torch.log(1e-5 + 1 - x_next_pred), dim=1))
    return loss_1, loss_2

def evaluate(model, test_loader, cost_loss_type):
    model.eval()
    num_batches = len(test_loader)
    state_loss, next_state_loss, consis_loss, jac_loss, cost_loss = 0., 0., 0., 0., 0.
    with torch.no_grad():
        for x, u, cost, x_next in test_loader:
            x = x.view(-1, model.obs_dim).double().to(device)
            u = u.double().to(device)
            cost = cost.double().to(device)
            x_next = x_next.view(-1, model.obs_dim).double().to(device)

            x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next, cost_pred, residual = model(x, u, x_next)
            zbar2 = model.reparam(q_z.mean, q_z.logvar)
            z_next_pred_zbar2, q_z_next_pred_zbar2, res_zbar2 = model.transition(zbar2, q_z, u)

            loss_1, loss_2 = compute_log_likelihood(x, x_recon, x_next, x_next_pred)

            state_loss += loss_1
            next_state_loss += loss_2
            consis_loss += NormalDistribution.KL_divergence(q_z_next_pred, q_z_next)
            jac_loss += NormalDistribution.KL_divergence(q_z_next_pred_zbar2, q_z_next_pred) + 10 * F.mse_loss(residual, res_zbar2)
            if cost_loss_type == 'mse':
                cost_loss += F.mse_loss(cost, cost_pred)
            elif cost_loss_type == 'l1':
                cost_loss += F.l1_loss(cost, cost_pred)

    return state_loss.item() / num_batches, next_state_loss.item() / num_batches, consis_loss.item() / num_batches, jac_loss.item() / num_batches, cost_loss.item() / num_batches

# code for visualizing the training process
def predict_x_next(model, env, num_eval):
    # frist sample a true trajectory from the environment
    sampler = samplers[env]
    state_samples, sampled_data = sampler.sample(num_eval)

    # use the trained model to predict the next observation
    predicted = []
    for x, u, x_next in sampled_data:
        x_reshaped = x.reshape(-1)
        x_reshaped = torch.from_numpy(x_reshaped).double().unsqueeze(dim=0).to(device)
        u = torch.from_numpy(u).double().unsqueeze(dim=0).to(device)
        with torch.no_grad():
            x_next_pred = model.predict(x_reshaped, u)
        predicted.append(x_next_pred.squeeze().cpu().numpy().reshape(x.shape))
    true_x_next = [data[-1] for data in sampled_data]
    return true_x_next, predicted

def plot_preds(model, env, num_eval):
    true_x_next, pred_x_next = predict_x_next(model, env, num_eval)

    # plot the predicted and true observations
    fig, axes =plt.subplots(nrows=2, ncols=num_eval, figsize=(20, 10))
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
    jac_loss_weight = args.jac_loss_weight
    cost_loss_weight = args.cost_loss_weight
    epoches = args.num_iter
    iter_save = args.iter_save
    log_dir = args.log_dir
    seed = args.seed
    dyn_rank = args.dyn_rank
    r_dim = args.r_dim
    cost_loss_type = args.cost_loss_type

    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = datasets[env_name]('./data/data/' + env_name)
    train_set, test_set = dataset[:int(len(dataset) * propor)], dataset[int(len(dataset) * propor):]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)

    obs_dim, z_dim, u_dim = settings[env_name]
    def model_name(env_name):
        if args.cnn:
            return f'{env_name}_cnn'
        else:
            return env_name

    model = E2C(obs_dim=obs_dim, z_dim=z_dim, u_dim=u_dim, r_dim=r_dim, dyn_rank=dyn_rank, env=model_name(env_name)).to(device)

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=lr, weight_decay=weight_decay)

    result_path = './result/' + env_name + '/' + log_dir
    if not path.exists(result_path):
        os.makedirs(result_path)

    for i in range(epoches):
        avg_loss = train(model, train_loader, lam, jac_loss_weight, cost_loss_weight, optimizer, cost_loss_type)
        print('Epoch %d' % i)
        print("Training loss: %f" % (avg_loss))
        # evaluate on test set
        state_loss, next_state_loss, consis_loss, jac_loss, cost_loss = evaluate(model, test_loader, cost_loss_type)
        print('State loss: ' + str(state_loss))
        print('Next state loss: ' + str(next_state_loss))

        if i == 0:
            wandb.init(project='e2c-original', config=args.__dict__)

        # save model
        if (i + 1) % iter_save == 0:
            print('Saving the model.............')

            torch.save(model.state_dict(), result_path + '/model_' + str(i + 1))
            with open(result_path + '/loss_' + str(i + 1), 'w') as f:
                f.write('\n'.join([str(state_loss), str(next_state_loss)]))

            wandb.log({'actual vs. predicted': wandb.Image(plot_preds(model, env_name, num_eval))}, step=i, commit=False)

        # ...log the running loss
        wandb.log({
            'train loss': avg_loss,
            'state loss': state_loss,
            'consis loss': consis_loss,
            'next state loss': next_state_loss,
            'jac loss': jac_loss,
            'cost loss': cost_loss,
        }, step=i, commit=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train e2c model')

    # the default value is used for the planar task
    parser.add_argument('--env', required=True, type=str, help='the environment used for training')
    parser.add_argument('--propor', default=3/4, type=float, help='the proportion of data used for training')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--lr', default=0.0005, type=float, help='the learning rate')
    parser.add_argument('--decay', default=0.001, type=float, help='the L2 regularization')
    parser.add_argument('--lam', default=0.25, type=float, help='the weight of the consistency term')
    parser.add_argument('--jac_loss_weight', default=0, type=float, help='the weight of the jacobian term')
    parser.add_argument('--cost_loss_weight', default=0, type=float, help='the weight of the cost term')
    parser.add_argument('--num_iter', default=5000, type=int, help='the number of epoches')
    parser.add_argument('--iter_save', default=1000, type=int, help='save model and result after this number of iterations')
    parser.add_argument('--log_dir', required=True, type=str, help='the directory to save training log')
    parser.add_argument('--seed', required=True, type=int, help='seed number')
    parser.add_argument('--cnn', action='store_true')

    parser.add_argument('--dyn_rank', type=int, default=1)
    parser.add_argument('--r_dim', type=int, default=3)
    parser.add_argument('--cost_loss_type', type=str, default='mse')

    args = parser.parse_args()

    main(args)
