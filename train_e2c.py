from collections import defaultdict
import datetime
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import sys
import numpy as np
import time
from utils import pytorch_utils as ptu

import os
import json
from normal import NormalDistribution
from e2c_model import E2C
from tqdm import tqdm
import datasets
from data.sample_eval import sample_eval_data

torch.set_default_dtype(torch.float32)

device = torch.device("cuda")
                                  # obs,    z,  u   r
settings = {'cartpole': {'image': (64*64,   8,  1,  3)},
            'planar':   {'image': (40*40,   8,  2,  3)},
            'pendulum': {'image': (48*48,   8,  1,  4)},
            'hopper':   {'image': (64*64,  16,  3,  8),
                        'serial': (11,     16,  3,  8)}}

step_sizes = {'cartpole': 4,
             'planar':    4,
             'pendulum':  4,
             'hopper':    4}

num_eval = 10 # number of images evaluated on tensorboard


def compute_loss(x, u_t, x_next, q_z_next, x_recon, x_next_pred, q_z, q_z_next_pred, cost, cost_pred, transition_model, z_residual, dyn_mats, args, recon_loss='bce'):
    if recon_loss == 'bce':
        recon_loss = torch.nn.BCELoss(reduction='none')
    elif recon_loss == 'mse':
        recon_loss = torch.nn.MSELoss(reduction='none')
    elif recon_loss == 'l1':
        recon_loss = torch.nn.L1Loss(reduction='none')

    A_t, B_t, _, G_t, H_t, _ = dyn_mats

    assert x_recon.min() >= 0 and x_recon.max() <= 1
    assert x_next_pred.min() >= 0 and x_next_pred.max() <= 1
    assert x.min() >= 0 and x.max() <= 1
    assert x_next.min() >= 0 and x_next.max() <= 1

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

    results = lower_bound + args.lam * consis_term + cost_term + args.jac_weight * (jac_loss_dyn + jac_loss_res), {
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

def train(model, train_loader, optimizer, global_step, args):
    model.train()
    avg_loss = 0.0

    metrics = defaultdict(float)

    num_batches = len(train_loader)

    print("Number of batches: ", num_batches)   # debug
    print("Train Loader `batch_size`: ", train_loader.batch_size)

    for _, (x, u, x_next, reward, done) in tqdm(enumerate(train_loader, 0), total=num_batches):
        # x: 'before' obs in batch & stack (num_batch, obs_dim)
        # u: 'action' in batch (num_batch, action)
        # x_next: 'after' obs in batch (num_batch, obs_dim)
        
        if args.cnn:
            x = x.float().to(ptu.device)
            x_next = x_next.float().to(ptu.device)
        else:
            x = x.view(-1, model.obs_dim).float().to(ptu.device)
            x_next = x_next.view(-1, model.obs_dim).float().to(ptu.device)
            
        u = u.float().to(ptu.device)
        optimizer.zero_grad()

        cost = -reward.float().to(ptu.device)

        x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next, cost_res_pred, dyn_mats = model(x, u, x_next)
        cost_pred = cost_res_pred.pow(2).sum(dim=1)

        if args.cnn:
            x = x.view(-1, model.obs_dim)
            x_next = x_next.view(-1, model.obs_dim)
            x_recon = x_recon.view(-1, model.obs_dim)
            x_next_pred = x_next_pred.view(-1, model.obs_dim)
            # NOTE: what is this?

        loss, metrics_it = compute_loss(
                    x=x,                            # 'before' obs
                    u_t=u,                          # action
                    x_next=x_next,                  # 'after' obs
                    q_z_next=q_z_next,              # q(z_next|x_next) distrubtion
                    x_recon=x_recon,                # decoded(z)
                    x_next_pred=x_next_pred,        # decoded(z_next)
                    q_z=q_z,                        # q(z|x) distribution
                    q_z_next_pred=q_z_next_pred,    # transition in latent space
                    cost=cost,
                    cost_pred=cost_pred,
                    transition_model=model.trans,
                    z_residual=cost_res_pred,
                    dyn_mats=dyn_mats,
                    args=args
                )

        avg_loss += loss.item()
        loss.backward()
        optimizer.step()

        for key, value in metrics_it.items():
            metrics[key] += value

    return avg_loss / num_batches, {key: value / num_batches for key, value in metrics.items()}


def evaluate(model, test_loader, args):
    model.eval()
    num_batches = len(test_loader)
    metrics = defaultdict(float)
    with torch.no_grad():
        for x, u, x_next, reward, d in test_loader:  # state, action, next_state, reward, done
            if args.cnn:
                x = x.float().to(ptu.device)
                x_next = x_next.float().to(ptu.device)
            else:
                x = x.view(-1, model.obs_dim).float().to(ptu.device)
                x_next = x_next.view(-1, model.obs_dim).float().to(ptu.device)
            u = u.float().to(ptu.device)

            x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next, cost_res_pred, dyn_mats = model(x, u, x_next)
            if args.cnn:
                x = x.view(-1, model.obs_dim)
                x_next = x_next.view(-1, model.obs_dim)
                x_recon = x_recon.view(-1, model.obs_dim)
                x_next_pred = x_next_pred.view(-1, model.obs_dim)

            x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next, cost_res_pred, dyn_mats = model(x, u, x_next)
            cost_pred = cost_res_pred.pow(2).sum(dim=1)
            cost = -reward.float().to(ptu.device)
            _, metrics_it = compute_loss(
                        x=x,                            # 'before' obs
                        u_t=u,                          # action
                        x_next=x_next,                  # 'after' obs
                        q_z_next=q_z_next,              # q(z_next|x_next) distrubtion
                        x_recon=x_recon,                # decoded(z)
                        x_next_pred=x_next_pred,        # decoded(z_next)
                        q_z=q_z,                        # q(z|x) distribution
                        q_z_next_pred=q_z_next_pred,    # transition in latent space
                        cost=cost,
                        cost_pred=cost_pred,
                        transition_model=model.trans,
                        z_residual=cost_res_pred,
                        dyn_mats=dyn_mats,
                        args=args
                    )
            for key, value in metrics_it.items():
                metrics[key] += value

    return {key: value / num_batches for key, value in metrics.items()}

# code for visualizing the training process
def predict_x_next(model, env_name, vis_loader, num_eval, stack):
    # first sample a true trajectory from the environment
    obs_res = int(np.sqrt(settings[env_name]['image'][0]))
    step_size = step_sizes[env_name]

    # use the trained model to predict the next observation
    predicted = []
    true_x = []
    for x, u, x_next, _, _ in vis_loader:
        x = x[0].cpu().numpy()
        u = u[0].cpu().numpy()
        x_next = x_next[0].cpu().numpy()
        x_reshaped = x.reshape(-1)
        x_reshaped = torch.from_numpy(x_reshaped).float().unsqueeze(dim=0).to(ptu.device)
        u = torch.from_numpy(u).float().unsqueeze(dim=0).to(ptu.device)
        with torch.no_grad():
            x_recon, x_next_pred, _, _, _, _, _ = model.forward(x_reshaped, u, x_reshaped)
        predicted.append(np.concatenate(
            tuple(x_recon.cpu().numpy().reshape(stack, obs_res, obs_res)) + tuple(x_next_pred.squeeze().cpu().numpy().reshape(stack, obs_res, obs_res))
        ))
        true_x.append(np.concatenate((tuple(x) + tuple(x_next))))

        if len(predicted) > num_eval:
            break

    return true_x, predicted

def plot_preds(model, env_name, vis_loader, num_eval, stack):
    true_x_next, pred_x_next = predict_x_next(model, env_name, vis_loader, num_eval, stack)

    # plot the predicted and true observations
    fig, axes =plt.subplots(nrows=2, ncols=len(true_x_next))
    plt.setp(axes, xticks=[], yticks=[])
    pad = 5
    axes[0, 0].annotate('True observations', xy=(0, 0.5), xytext=(-axes[0,0].yaxis.labelpad - pad, 0),
                   xycoords=axes[0,0].yaxis.label, textcoords='offset points',
                   size='large', ha='right', va='center')
    axes[1, 0].annotate('Predicted observations', xy=(0, 0.5), xytext=(-axes[1, 0].yaxis.labelpad - pad, 0),
                        xycoords=axes[1, 0].yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')

    for idx in np.arange(len(true_x_next)):
        axes[0, idx].imshow(true_x_next[idx], cmap='Greys', vmin=0, vmax=1)
        axes[1, idx].imshow(pred_x_next[idx], cmap='Greys', vmin=0, vmax=1)

    fig.tight_layout()
    return fig

def main(args):
    sample_path = args.sample_path
    env_name = args.env
    assert env_name in ['planar', 'pendulum', 'hopper', 'cartpole']
    propor = args.propor
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.decay
    lam = args.lam
    epoches = args.num_iter
    iter_save = args.iter_save
    exp_name = args.exp_name
    seed = args.seed
    stack = args.stack

    np.random.seed(seed)
    torch.manual_seed(seed)
    ptu.init_gpu(use_gpu=args.gpu)

    dataset = datasets.OfflineDataset(
        dir='./data/samples/' + env_name + '/' + sample_path,
        stack=stack
    )
    
    print('OFFLINE DATASET SIZE: ', len(dataset))

    train_set, test_set = dataset[:int(len(dataset) * propor)], dataset[int(len(dataset) * propor):]
    train_loader = DataLoader(train_set, batch_size=batch_size, 
                              shuffle=True, drop_last=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, 
                              shuffle=False, drop_last=False, num_workers=8)
    vis_loader = DataLoader(test_set, batch_size=1, 
                            shuffle=True, drop_last=False, num_workers=8)
    exp_name = f'{exp_name}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    # ##### Debugging ####
    # return
    # ####################
    if dataset.return_obs_image():  # True if observation is images
        obs_type = 'image'
    else:
        obs_type = 'serial'

    obs_dim, z_dim, u_dim, r_dim = settings[env_name][obs_type]
    obs_dim = obs_dim * stack

    if args.z_dim is None:
        args.z_dim = z_dim
    else:
        z_dim = args.z_dim

    if args.r_dim is None:
        args.r_dim = r_dim
    else:
        r_dim = args.r_dim
    
    model = E2C(obs_dim=obs_dim, z_dim=z_dim, u_dim=u_dim, r_dim=r_dim,
                env=env_name, stack=stack, use_cnn=args.cnn).to(ptu.device)

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), 
                           eps=1e-8, lr=lr, weight_decay=weight_decay)

    writer = None

    result_path = './result/' + env_name + '/' + exp_name
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(result_path + '/hyperparameters.json', 'w') as f:
        hparameter = {
            **args.__dict__,
            **{'obs_dim': obs_dim, 'z_dim': z_dim, 'u_dim': u_dim}              
        }
        json.dump(hparameter, f, indent=2)

    for i in range(epoches):
        avg_loss, train_metrics = train(model, train_loader, optimizer, global_step=i, args=args)
        print('Epoch %d' % i)
        print("Training loss: %f" % (avg_loss))
        # evaluate on test set
        eval_metrics = evaluate(model, test_loader, args=args)
        print('State loss: ' + str(eval_metrics['recon']))
        print('Next state loss: ' + str(eval_metrics['pred']))

        if writer is None:
            writer = SummaryWriter('logs/' + env_name + '/' + exp_name, comment=args.comment)
            writer.add_hparams(args.__dict__, {})

        # ...log the running loss
        writer.add_scalar('eval/training loss', avg_loss, i)
        for key, value in eval_metrics.items():
            writer.add_scalar('eval/' + key, value, i)
        for key, value in train_metrics.items():
            writer.add_scalar('train/' + key, value, i)

        # save model
        num_eval = 10 # number of images evaluated on tensorboard
        if (i + 1) % iter_save == 0:
            writer.add_figure('actual vs. predicted observations',
                              plot_preds(model, env_name, vis_loader, num_eval, stack),
                              global_step=i)
            print('Saving the model.............')

            torch.save(model.state_dict(), result_path + '/model_' + str(i + 1))
            with open(result_path + '/loss_' + str(i + 1), 'w') as f:
                f.write('\n'.join([str(eval_metrics['recon']), str(eval_metrics['pred'])]))

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train e2c model')

    # the default value is used for the planar task
    parser.add_argument('--sample_path', required=True, type=str, 
                        help='the "sample-M_D_Y_hms" folder for samples')
    parser.add_argument('--env', required=True, type=str, 
                        help='the environment used for training')
    parser.add_argument('--propor', default=0.75, type=float, 
                        help='the proportion of data used for training')
    parser.add_argument('--batch_size', default=128, type=int, 
                        help='batch size')
    parser.add_argument('--lr', default=0.0005, type=float, 
                        help='the learning rate')
    parser.add_argument('--decay', default=0.001, type=float, 
                        help='the L2 regularization')
    parser.add_argument('--lam', default=0.25, type=float, 
                        help='the weight of the consistency term')
    parser.add_argument('--jac_weight', default=1.0, type=float,
                        help='the weight of the Jacobian terms')
    parser.add_argument('--num_iter', default=5000, type=int, 
                        help='the number of epoches')
    parser.add_argument('--iter_save', default=1000, type=int, 
                        help='save model and result after this number of iterations')
    parser.add_argument('--exp_name', required=True, type=str, 
                        help='the directory to save training log (in `logs/env/exp_name`)')
    parser.add_argument('--comment', type=str, 
                        help='a comment describing the run')
    parser.add_argument('--seed', required=True, type=int, 
                        help='seed number')
    parser.add_argument('--cnn', action="store_true", 
                        help='use cnn as encoder and decoder')
    parser.add_argument('--stack', default=1, type=int, 
                        help='number of frames to stack when training')
    parser.add_argument('--gpu', action="store_true", default=True,
                        help='use gpu if available')
    parser.add_argument('--ngpu', action="store_false",
                        dest='gpu', help='do not use gpu')
    parser.add_argument('--z_dim', type=int,
                        help='latent space dimension (default to environment setting)')
    parser.add_argument('--r_dim', type=int,
                        help='residual dimension (default to environment setting)')
    
    args = parser.parse_args()

    main(args)