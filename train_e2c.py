from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import sys
import numpy as np
from utils import pytorch_utils as ptu

import os
import json
from normal import NormalDistribution
from e2c_model import E2C
import datasets
import data.sample_planar as planar_sampler
import data.sample_pendulum_data as pendulum_sampler
import data.sample_cartpole_data as cartpole_sampler
import data.sample_hopper_data as hopper_sampler

torch.set_default_dtype(torch.float64)

# if torch.cuda.is_available():
#   device = torch.device("cuda")
# else:
#   device = torch.device("cpu")

datasets = {'planar': datasets.PlanarDataset, 
            'pendulum': datasets.MujocoDataset,
            'hopper': datasets.MujocoDataset}
                                  # obs,  z,   u
settings = {'planar':   {'image': (1600,    2,   2) },
            'pendulum': {'image': (48*48,   3,   1) },
            'hopper':   {'image': (64*64, 512,   3),
                        'serial': (11,      2,   3) },
            }
samplers = {'planar': planar_sampler, 
            'pendulum': pendulum_sampler, 
            'cartpole': cartpole_sampler,
            'hopper': hopper_sampler}
num_eval = 10 # number of images evaluated on tensorboard

def compute_loss(x, x_next, q_z_next, x_recon, x_next_pred, q_z, q_z_next_pred, lamda, mse=False):
    # lower-bound loss
    if mse:
        recon_term = F.mse_loss(x_recon, x)
        pred_loss = F.mse_loss(x_next_pred, x_next)
    else:
        recon_term = -torch.mean(torch.sum(x * torch.log(1e-5 + x_recon)
                                          + (1 - x) * torch.log(1e-5 + 1 - x_recon), dim=1))
        pred_loss = -torch.mean(torch.sum(x_next * torch.log(1e-5 + x_next_pred)
                                      + (1 - x_next) * torch.log(1e-5 + 1 - x_next_pred), dim=1))

    kl_term = - 0.5 * torch.mean(torch.sum(1 + q_z.logvar - q_z.mean.pow(2) - q_z.logvar.exp(), dim = 1))

    lower_bound = recon_term + pred_loss + kl_term

    # consistency loss
    consis_term = NormalDistribution.KL_divergence(q_z_next_pred, q_z_next)
    return lower_bound + lamda * consis_term

def train(model, train_loader, lam, optimizer):
    model.train()
    avg_loss = 0.0

    num_batches = len(train_loader)

    print("Number of batches: ", num_batches)   # debug
    print("Train Loader `batch_size`: ", train_loader.batch_size)

    for _, (x, u, x_next, reward) in enumerate(train_loader, 0):
        # x: 'before' obs in batch & stack (num_batch, obs_dim)
        # u: 'action' in batch (num_batch, action)
        # x_next: 'after' obs in batch (num_batch, obs_dim)
        
        if args.cnn:
            x = x.double().to(ptu.device)
            x_next = x_next.double().to(ptu.device)
        else:
            x = x.view(-1, model.obs_dim).double().to(ptu.device)
            x_next = x_next.view(-1, model.obs_dim).double().to(ptu.device)
            
        u = u.double().to(ptu.device)
        optimizer.zero_grad()

        x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next = model(x, u, x_next)

        if args.cnn:
            x = x.view(-1, model.obs_dim)
            x_next = x_next.view(-1, model.obs_dim)
            x_recon = x_recon.view(-1, model.obs_dim)
            x_next_pred = x_next_pred.view(-1, model.obs_dim)
            # NOTE: what is this?

        loss = compute_loss(
                    x=x,                            # 'before' obs
                    x_next=x_next,                  # 'after' obs
                    q_z_next=q_z_next,              # q(z_next|x_next) distrubtion
                    x_recon=x_recon,                # decoded(z)
                    x_next_pred=x_next_pred,        # decoded(z_next)
                    q_z=q_z,                        # q(z|x) distribution
                    q_z_next_pred=q_z_next_pred,    # transition in latent space
                    lamda=lam,
                    mse=args.cnn                         
                )

        avg_loss += loss.item()
        loss.backward()
        optimizer.step()

    return avg_loss / num_batches

def compute_log_likelihood(x, x_recon, x_next, x_next_pred, mse=False):
    if mse:
        loss_1 = F.mse_loss(x_recon, x)
        loss_2 = F.mse_loss(x_next_pred, x_next)
    else:
        loss_1 = -torch.mean(torch.sum(x * torch.log(1e-5 + x_recon)
                                      + (1 - x) * torch.log(1e-5 + 1 - x_recon), dim=1))
        loss_2 = -torch.mean(torch.sum(x_next * torch.log(1e-5 + x_next_pred)
                                   + (1 - x_next) * torch.log(1e-5 + 1 - x_next_pred), dim=1))
    return loss_1, loss_2

def evaluate(model, test_loader):
    model.eval()
    num_batches = len(test_loader)
    state_loss, next_state_loss = 0., 0.
    with torch.no_grad():
        for x, u, x_next in test_loader:
            if args.cnn:
                x = x.double().to(ptu.device)
                x_next = x_next.double().to(ptu.device)
            else:
                x = x.view(-1, model.obs_dim).double().to(ptu.device)
                x_next = x_next.view(-1, model.obs_dim).double().to(ptu.device)
            u = u.double().to(ptu.device)

            x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next = model(x, u, x_next)
            if args.cnn:
                x = x.view(-1, model.obs_dim)
                x_next = x_next.view(-1, model.obs_dim)
                x_recon = x_recon.view(-1, model.obs_dim)
                x_next_pred = x_next_pred.view(-1, model.obs_dim)
            loss_1, loss_2 = compute_log_likelihood(x, x_recon, x_next, x_next_pred, mse=args.cnn)
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
        x_reshaped = torch.from_numpy(x_reshaped).double().unsqueeze(dim=0).to(ptu.device)
        u = torch.from_numpy(u).double().unsqueeze(dim=0).to(ptu.device)
        with torch.no_grad():
            x_next_pred = model.predict(x_reshaped, u)
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
    sample_path = args.sample_path
    env_name = args.env
    assert env_name in ['planar', 'pendulum', 'hopper']
    propor = args.propor
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.decay
    lam = args.lam
    epoches = args.num_iter
    iter_save = args.iter_save
    exp_name = args.log_dir
    seed = args.seed
    stack = args.stack 

    np.random.seed(seed)
    torch.manual_seed(seed)
    ptu.init_gpu(use_gpu=args.gpu)

    if env_name not in ['planar', 'cartpole']:  # MuJoCo Datasets & Pendulum
        dataset = datasets[env_name](
            dir='./data/samples/' + env_name + '/' + sample_path,
            stack=stack
        )
    else:
        dataset = datasets[env_name](
            dir='./data/samples/' + env_name + '/' + sample_path,
        )
    
    train_set, test_set = dataset[:int(len(dataset) * propor)], dataset[int(len(dataset) * propor):]
    train_loader = DataLoader(train_set, batch_size=batch_size, 
                              shuffle=True, drop_last=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, 
                              shuffle=False, drop_last=False, num_workers=8)

    # ##### Debugging ####
    # return
    # ####################
    if dataset.return_obs_image():  # True if observation is images
        obs_type = 'image'
    else:
        obs_type = 'serial'

    obs_dim, z_dim, u_dim = settings[env_name][obs_type]
    obs_dim = obs_dim * stack
    
    model = E2C(obs_dim=obs_dim, z_dim=z_dim, u_dim=u_dim, 
                env=env_name, stack=stack, use_cnn=args.cnn).to(ptu.device)

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), 
                           eps=1e-8, lr=lr, weight_decay=weight_decay)

    writer = SummaryWriter('logs/' + env_name + '/' + exp_name)

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
        avg_loss = train(model, train_loader, lam, optimizer)
        print('Epoch %d' % i)
        print("Training loss: %f" % (avg_loss))
        # evaluate on test set
        state_loss, next_state_loss = evaluate(model, test_loader)
        print('State loss: ' + str(state_loss))
        print('Next state loss: ' + str(next_state_loss))

        # ...log the running loss
        writer.add_scalar('training_loss', avg_loss, i)
        writer.add_scalar('state_loss', state_loss, i)
        writer.add_scalar('next_state_loss', next_state_loss, i)

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
    parser.add_argument('--num_iter', default=5000, type=int, 
                        help='the number of epoches')
    parser.add_argument('--iter_save', default=1000, type=int, 
                        help='save model and result after this number of iterations')
    parser.add_argument('--log_dir', required=True, type=str, 
                        help='the directory to save training log (in `logs/env/log_dir`)')
    parser.add_argument('--seed', required=True, type=int, 
                        help='seed number')
    parser.add_argument('--cnn', action="store_true", 
                        help='use cnn as encoder and decoder')
    parser.add_argument('--stack', default=1, type=int, 
                        help='number of frames to stack when training')
    parser.add_argument('--gpu', action="store_true",
                        help='use gpu if available')
    
    args = parser.parse_args()

    main(args)
