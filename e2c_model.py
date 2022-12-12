from torch import nn

from normal import *
from networks import *
import networks

torch.set_default_dtype(torch.float32)

class E2C(nn.Module):
    def __init__(self, obs_dim, z_dim, u_dim, r_dim, env, stack=1, use_cnn=False):
        """
        - stack (int): number of frames to concatenate (defaults to `1`)
        """
        super(E2C, self).__init__()
        enc, dec, trans = load_config(env)
        if use_cnn:
            enc = networks.CNNEncoder
            dec = networks.CNNDecoder

        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.r_dim = r_dim
        self.stack = stack

        self.encoder = enc(obs_dim=obs_dim, z_dim=z_dim, stack_num=self.stack)
        # self.encoder.apply(init_weights)
        self.decoder = dec(z_dim=z_dim, obs_dim=obs_dim, stack_num=self.stack)
        # self.decoder.apply(init_weights)
        self.trans = trans(z_dim=z_dim, u_dim=u_dim, r_dim=r_dim)
        # self.trans.apply(init_weights)

    def encode(self, x):
        """
        :param x:
        :return: mean and log variance of q(z | x)
        """
        return self.encoder(x)

    def decode(self, z):
        """
        :param z:
        :return: bernoulli distribution p(x | z)
        """
        return self.decoder(z)

    def transition(self, z_bar, q_z, u):
        """
        :param z_bar:
        :param q_z:
        :param u:
        :return: samples z_hat_next and Q(z_hat_next)
        """
        return self.trans(z_bar, q_z, u)

    def reparam(self, mean, logvar):
        sigma = (logvar / 2).exp()
        epsilon = torch.randn_like(sigma)
        return mean + torch.mul(epsilon, sigma)

    def forward(self, x, u, x_next):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        q_z = NormalDistribution(mu, logvar)

        x_recon = self.decode(z)

        z_next, q_z_next_pred, cost_residual, dyn_mats = self.transition(z, q_z, u)

        x_next_pred = self.decode(z_next)
        
        mu_next, logvar_next = self.encode(x_next)
        q_z_next = NormalDistribution(mean=mu_next, logvar=logvar_next, A=dyn_mats[0])

        return x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next, cost_residual, dyn_mats

    def predict(self, x, u):
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar)
        q_z = NormalDistribution(mu, logvar)

        z_next, q_z_next_pred, cost, _ = self.transition(z, q_z, u)

        x_next_pred = self.decode(z_next)
        return x_next_pred, cost