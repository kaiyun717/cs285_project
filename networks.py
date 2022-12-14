import torch
from torch import nn
from normal import NormalDistribution

torch.set_default_dtype(torch.float64)

def weights_init(m):
    if type(m) in [nn.Conv2d, nn.Linear, nn.ConvTranspose2d]:
        torch.nn.init.orthogonal_(m.weight)

class Encoder(nn.Module):
    def __init__(self, net, obs_dim, z_dim):
        super(Encoder, self).__init__()
        self.net = net
        self.net.apply(weights_init)
        self.img_dim = obs_dim
        self.z_dim = z_dim

    def forward(self, x):
        """
        :param x: observation
        :return: the parameters of distribution q(z|x)
        """
        return self.net(x).chunk(2, dim = 1) # first half is mean, second half is logvar

class Decoder(nn.Module):
    def __init__(self, net, z_dim, obs_dim):
        super(Decoder, self).__init__()
        self.net = net
        self.net.apply(weights_init)
        self.z_dim = z_dim
        self.obs_dim = obs_dim

    def forward(self, z):
        """
        :param z: sample from q(z|x)
        :return: reconstructed x
        """
        return self.net(z)

class Transition(nn.Module):
    def __init__(self, net, z_dim, u_dim, r_dim, dyn_rank):
        super(Transition, self).__init__()
        self.net = net  # network to output the last layer before predicting A_t, B_t and o_t
        self.net.apply(weights_init)
        self.h_dim = self.net[-3].out_features
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.r_dim = r_dim
        self.dyn_rank = dyn_rank

        self.fc_A = nn.Sequential(
            nn.Linear(self.h_dim, self.dyn_rank * self.z_dim * 2), # v_t and r_t
            nn.Sigmoid()
        )
        self.fc_A.apply(weights_init)

        self.fc_B = nn.Linear(self.h_dim, self.z_dim * self.u_dim)
        torch.nn.init.orthogonal_(self.fc_B.weight)
        self.fc_o = nn.Linear(self.h_dim, self.z_dim)
        torch.nn.init.orthogonal_(self.fc_o.weight)

        self.fc_G = nn.Linear(self.h_dim, self.r_dim * self.z_dim)
        torch.nn.init.orthogonal_(self.fc_G.weight)
        self.fc_H = nn.Linear(self.h_dim, self.r_dim * self.u_dim)
        torch.nn.init.orthogonal_(self.fc_H.weight)
        self.fc_r0 = nn.Linear(self.h_dim, self.r_dim)
        torch.nn.init.orthogonal_(self.fc_r0.weight)

    def forward(self, z_bar_t, q_z_t, u_t):
        """
        :param z_bar_t: the reference point
        :param Q_z_t: the distribution q(z|x)
        :param u_t: the action taken
        :return: the predicted q(z^_t+1 | z_t, z_bar_t, u_t)
        """
        h_t = self.net(z_bar_t)
        B_t = self.fc_B(h_t)
        o_t = self.fc_o(h_t)

        G_t = self.fc_G(h_t).view(-1, self.r_dim, self.z_dim)
        H_t = self.fc_H(h_t).view(-1, self.r_dim, self.u_dim)
        r0_t = self.fc_r0(h_t).view(-1, self.r_dim)

        v_t, r_t = self.fc_A(h_t).chunk(2, dim=1)
        v_t = v_t.view(-1, self.z_dim, self.dyn_rank)
        r_t = r_t.view(-1, self.dyn_rank, self.z_dim)

        A_t = torch.eye(self.z_dim).repeat(z_bar_t.size(0), 1, 1).to(z_bar_t.device) + torch.bmm(v_t, r_t)

        B_t = B_t.view(-1, self.z_dim, self.u_dim)

        mu_t = q_z_t.mean

        mean = A_t.bmm(mu_t.unsqueeze(-1)).squeeze(-1) + B_t.bmm(u_t.unsqueeze(-1)).squeeze(-1) + o_t

        residual = G_t.bmm(mu_t.unsqueeze(-1)).squeeze(-1) + H_t.bmm(u_t.unsqueeze(-1)).squeeze(-1) + r0_t

        return mean, NormalDistribution(mean, logvar=q_z_t.logvar, A=A_t), residual

class PlanarEncoder(Encoder):
    def __init__(self, obs_dim = 1600, z_dim = 2):
        net = nn.Sequential(
            nn.Linear(obs_dim, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),

            nn.Linear(150, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),

            nn.Linear(150, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),

            nn.Linear(150, z_dim * 2)
        )
        super(PlanarEncoder, self).__init__(net, obs_dim, z_dim)

class PlanarDecoder(Decoder):
    def __init__(self, z_dim = 2, obs_dim = 1600):
        net = nn.Sequential(
            nn.Linear(z_dim, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),

            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),

            nn.Linear(200, 1600),
            nn.Sigmoid()
        )
        super(PlanarDecoder, self).__init__(net, z_dim, obs_dim)

class PlanarEncoderCNN(Encoder):
    def __init__(self, obs_dim = 1600, z_dim = 2):
        net = nn.Sequential(
            nn.Unflatten(-1, (1, 40, 40)),
            nn.Conv2d(1, 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),

            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(200, z_dim * 2)
        )
        super(PlanarEncoderCNN, self).__init__(net, obs_dim, z_dim)

class PlanarDecoderCNN(Decoder):
    def __init__(self, z_dim = 2, obs_dim = 1600):
        net = nn.Sequential(
            nn.Linear(z_dim, 200),
            nn.BatchNorm1d(200),
            nn.Unflatten(-1, (8, 5, 5)),
            nn.ReLU(),

            nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),

            nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1),

            nn.Flatten(),
            nn.Sigmoid()
        )
        super(PlanarDecoderCNN, self).__init__(net, z_dim, obs_dim)

class PlanarTransition(Transition):
    def __init__(self, z_dim = 2, u_dim = 2, r_dim=4, dyn_rank=1):
        net = nn.Sequential(
            nn.Linear(z_dim, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),

            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU()
        )
        super(PlanarTransition, self).__init__(net, z_dim, u_dim, r_dim, dyn_rank)

class PendulumEncoder(Encoder):
    def __init__(self, obs_dim = 4608, z_dim = 3):
        net = nn.Sequential(
            nn.Linear(obs_dim, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),

            nn.Linear(800, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),

            nn.Linear(800, z_dim * 2)
        )
        super(PendulumEncoder, self).__init__(net, obs_dim, z_dim)

class PendulumDecoder(Decoder):
    def __init__(self, z_dim = 3, obs_dim = 4608):
        net = nn.Sequential(
            nn.Linear(z_dim, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),

            nn.Linear(800, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),

            nn.Linear(800, obs_dim),
            nn.Sigmoid()
        )
        super(PendulumDecoder, self).__init__(net, z_dim, obs_dim)

class PendulumTransition(Transition):
    def __init__(self, z_dim = 3, u_dim = 1, r_dim=3, dyn_rank=1):
        net = nn.Sequential(
            nn.Linear(z_dim, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),

            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU()
        )
        super(PendulumTransition, self).__init__(net, z_dim, u_dim, r_dim, dyn_rank)

CONFIG = {
    'planar': (PlanarEncoder, PlanarDecoder, PlanarTransition),
    'planar_cnn': (PlanarEncoderCNN, PlanarDecoderCNN, PlanarTransition),
    'pendulum': (PendulumEncoder, PendulumDecoder, PendulumTransition)
}

def load_config(name):
    return CONFIG[name]

__all__ = ['load_config']

# enc = PendulumEncoder()
# dec = PendulumDecoder()
# trans = PendulumTransition()
#
# x = torch.randn(size=(10, 4608))
# # print (x.size())
# mean, logvar = enc(x)
# # print (logvar.size())
# x_recon = dec(mean)
# # print (x_recon.size())
#
# q_z_t = NormalDistribution(mean, logvar)
# print (q_z_t.mean.size())
# print (q_z_t.cov.size())
# u_t = torch.randn(size=(10, 1))
# z_t_1 = trans(mean, q_z_t, u_t)
# print (z_t_1[1].mean.size())
# print (z_t_1[1].cov.size())
#
# kl = NormalDistribution.KL_divergence(z_t_1[1], q_z_t)
# print (kl)