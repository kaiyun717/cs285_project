from functools import partial
import torch
from torch import nn
from normal import NormalDistribution
from utils import pytorch_utils as ptu
import numpy as np

torch.set_default_dtype(torch.float32)

def weights_init(m, gain=1):
    if type(m) in [nn.Conv2d, nn.Linear, nn.ConvTranspose2d]:
        torch.nn.init.orthogonal_(m.weight, gain=gain)

class Encoder(nn.Module):
    def __init__(self, net):
        super(Encoder, self).__init__()
        self.net = net
        self.net.apply(weights_init)

    def forward(self, x):
        """
        :param x: observation
        :return: the parameters of distribution q(z|x)
        """
        return self.net(x).chunk(2, dim = 1) # first half is mean, second half is logvar

class Decoder(nn.Module):
    def __init__(self, net):
        super(Decoder, self).__init__()
        self.net = net
        self.net.apply(weights_init)

    def forward(self, z):
        """
        :param z: sample from q(z|x)
        :return: reconstructed x
        """
        return self.net(z)


class Transition(nn.Module):
    def __init__(self, net, z_dim, u_dim, r_dim, use_vr, d_hidden):
        super(Transition, self).__init__()
        self.net = net  # network to output the last layer before predicting A_t, B_t and o_t
        self.net.apply(weights_init)
        self.h_dim = d_hidden
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.r_dim = r_dim
        self.use_vr = use_vr

        if use_vr:
            self.fc_vr = nn.Linear(self.h_dim, 2 * self.z_dim)
            torch.nn.init.orthogonal_(self.fc_vr.weight)
        else:
            self.fc_A = nn.Sequential(
                nn.Linear(self.h_dim, self.z_dim * self.z_dim),
                nn.Tanh()
            )
            self.fc_A.apply(partial(weights_init, gain=0.1))

        self.fc_B = nn.Linear(self.h_dim, self.z_dim * self.u_dim)
        torch.nn.init.orthogonal_(self.fc_B.weight)
        self.fc_o = nn.Linear(self.h_dim, self.z_dim)
        torch.nn.init.orthogonal_(self.fc_o.weight)

        self.fc_G = nn.Linear(self.h_dim, self.r_dim * self.z_dim)
        torch.nn.init.orthogonal_(self.fc_G.weight)
        self.fc_H = nn.Linear(self.h_dim, self.r_dim * self.u_dim)
        torch.nn.init.orthogonal_(self.fc_H.weight)
        self.fc_j = nn.Linear(self.h_dim, self.r_dim)
        torch.nn.init.orthogonal_(self.fc_j.weight)
        self.device = ptu.device

    def forward_h(self, z_bar_t):
        return self.net(z_bar_t)

    def forward_A(self, z_bar_t=None, h_t=None):
        if h_t is None:
            h_t = self.forward_h(z_bar_t)

        if self.use_vr:
            v, r = self.fc_vr(h_t).chunk(2, dim=1)
            return torch.eye(self.z_dim, device=h_t.device) + torch.bmm(v.view(-1, self.z_dim, 1), r.view(-1, 1, self.z_dim))
        else:
            A_t = torch.eye(self.z_dim, device=h_t.device) + self.fc_A(h_t).view(-1, self.z_dim, self.z_dim)
            return A_t

    def forward_B(self, z_bar_t=None, h_t=None):
        if h_t is None:
            h_t = self.forward_h(z_bar_t)

        B_t = self.fc_B(h_t)
        B_t = B_t.view(-1, self.z_dim, self.u_dim)

        return B_t

    def forward_o(self, z_bar_t=None, h_t=None):
        if h_t is None:
            h_t = self.forward_h(z_bar_t)

        return self.fc_o(h_t)
    
    def forward_G(self, z_bar_t=None, h_t=None):
        if h_t is None:
            h_t = self.forward_h(z_bar_t)

        G_t = self.fc_G(h_t)
        G_t = G_t.view(-1, self.r_dim, self.z_dim)

        return G_t

    def forward_H(self, z_bar_t=None, h_t=None):
        if h_t is None:
            h_t = self.forward_h(z_bar_t)

        H_t = self.fc_H(h_t)
        H_t = H_t.view(-1, self.r_dim, self.u_dim)

        return H_t

    def forward_j(self, z_bar_t=None, h_t=None):
        if h_t is None:
            h_t = self.forward_h(z_bar_t)

        return self.fc_j(h_t)

    def forward(self, z_bar_t, q_z_t, u_t):
        """
        :param z_bar_t: the reference point
        :param Q_z_t: the distribution q(z|x)
        :param u_t: the action taken
        :return: the predicted q(z^_t+1 | z_t, z_bar_t, u_t)
        """
        h_t = self.forward_h(z_bar_t)

        A_t = self.forward_A(h_t=h_t)
        B_t = self.forward_B(h_t=h_t)
        o_t = self.forward_o(h_t=h_t)

        G_t = self.forward_G(h_t=h_t)
        H_t = self.forward_H(h_t=h_t)
        j_t = self.forward_j(h_t=h_t)

        mu_t = q_z_t.mean

        mean = A_t.bmm(mu_t.unsqueeze(-1)).squeeze(-1) + B_t.bmm(u_t.unsqueeze(-1)).squeeze(-1) + o_t
        cost_residual = G_t.bmm(mu_t.unsqueeze(-1)).squeeze(-1) + H_t.bmm(u_t.unsqueeze(-1)).squeeze(-1) + j_t

        return mean, NormalDistribution(mean, logvar=q_z_t.logvar, A=A_t), cost_residual, (A_t, B_t, o_t, G_t, H_t, j_t)

################ IMAGE OBSERVATION ################
def make_cnn_encoder(obs_shape, z_dim, n_filters=32):
    channels, width, height = obs_shape
    return Encoder(nn.Sequential(
        nn.Conv2d(channels, n_filters, 5, stride=2, padding=2),
        nn.BatchNorm2d(n_filters),
        nn.ReLU(),

        nn.Conv2d(n_filters, n_filters, 5, stride=2, padding=2),
        nn.BatchNorm2d(n_filters),
        nn.ReLU(),

        nn.Conv2d(n_filters, n_filters, 5, stride=2, padding=2),
        nn.BatchNorm2d(n_filters),
        nn.ReLU(),

        nn.Flatten(),

        nn.Linear((n_filters * width*height)//64, (n_filters * width*height)//64),
        nn.ReLU(),
        nn.Linear((n_filters * width*height)//64, (n_filters * width*height)//64),
        nn.ReLU(),
        nn.Linear((n_filters * width*height)//64, z_dim * 2),
    ))

def make_cnn_decoder(obs_shape, z_dim, n_filters=32):
    channels, width, height = obs_shape
    return Decoder(nn.Sequential(
        nn.Linear(z_dim, (n_filters * height * width) // 64),
        nn.ReLU(),
        nn.Linear((n_filters * width*height)//64, (n_filters * width*height)//64),
        nn.ReLU(),
        nn.Linear((n_filters * width*height)//64, (n_filters * width*height)//64),
        nn.ReLU(),

        nn.Unflatten(-1, (n_filters, height // 8, width // 8)),

        nn.ConvTranspose2d(n_filters, n_filters, 4, stride=2, padding=1, output_padding=0),
        nn.BatchNorm2d(n_filters),
        nn.ReLU(),

        nn.ConvTranspose2d(n_filters, n_filters, 4, stride=2, padding=1, output_padding=0),
        nn.BatchNorm2d(n_filters),
        nn.ReLU(),

        nn.ConvTranspose2d(n_filters, n_filters, 4, stride=2, padding=1, output_padding=0),
        nn.BatchNorm2d(n_filters),
        nn.ReLU(),

        nn.ConvTranspose2d(n_filters, channels, 5, stride=1, padding=2, output_padding=0),
        nn.Sigmoid(),
    ))


################ SERIAL OBSERVATION ################
def make_serial_encoder(obs_shape, z_dim, d_hidden=800, n_layers=3, use_batchnorm=True):
    return Encoder(nn.Sequential(
        nn.Flatten(),

        nn.Linear(np.prod(obs_shape), d_hidden),
        nn.BatchNorm1d(d_hidden),
        nn.ReLU(),

        *[nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.BatchNorm1d(d_hidden) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
        ) for _ in range(n_layers)],

        nn.Linear(d_hidden, z_dim * 2)
    ))

def make_serial_decoder(obs_shape, z_dim, d_hidden=800, n_layers=3, use_batchnorm=True):
    return Decoder(nn.Sequential(
        nn.Linear(z_dim, d_hidden),
        nn.BatchNorm1d(d_hidden),
        nn.ReLU(),

        *[nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.BatchNorm1d(d_hidden) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
        ) for _ in range(n_layers)],

        nn.Linear(d_hidden, np.prod(obs_shape)),
        nn.Unflatten(1, obs_shape),
        nn.Sigmoid(),
    ))

################ MUJOCO TRANSITION ################
class MujocoTransition(Transition):
    def __init__(self, u_dim, z_dim, r_dim, d_hidden=256, n_layers=3, use_vr=False):   # NOTE: order is different!
        net = nn.Sequential(
            nn.Linear(z_dim, d_hidden),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(d_hidden, d_hidden),
                nn.ReLU(),
            ) for _ in range(n_layers)],
        )
        super().__init__(net, z_dim, u_dim, r_dim, use_vr=use_vr, d_hidden=d_hidden)

CONFIG = {
    'planar': (make_serial_encoder, make_serial_decoder, MujocoTransition),
    'pendulum': (make_serial_encoder, make_serial_decoder, MujocoTransition),
    'hopper': (make_serial_encoder, make_serial_decoder, MujocoTransition),
    'cartpole': (make_serial_encoder, make_serial_decoder, MujocoTransition)
}

def load_config(name):
    return CONFIG[name]

__all__ = ['load_config']
