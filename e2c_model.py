from torch import nn

from normal import *
from networks import *
import networks

torch.set_default_dtype(torch.float32)

class E2C(nn.Module):
    def __init__(self, obs_shape, z_dim, u_dim, r_dim, env, stack, args):
        """
        - stack (int): number of frames to concatenate (defaults to `1`)
        """
        super(E2C, self).__init__()
        enc, dec, trans = load_config(env)

        self.obs_shape = obs_shape
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.r_dim = r_dim
        self.stack = stack

        if args.cnn:
            self.encoder = networks.make_cnn_encoder(obs_shape=obs_shape, z_dim=z_dim, n_filters=args.cnn_n_filters)
            self.decoder = networks.make_cnn_decoder(obs_shape=obs_shape, z_dim=z_dim, n_filters=args.cnn_n_filters)
        else:
            self.encoder = enc(obs_shape=obs_shape, z_dim=z_dim, use_batchnorm=args.mlp_use_batchnorm, n_layers=args.mlp_enc_n_layers, d_hidden=args.mlp_enc_d_hidden)
            self.decoder = dec(obs_shape=obs_shape, z_dim=z_dim, use_batchnorm=args.mlp_use_batchnorm, n_layers=args.mlp_dec_n_layers, d_hidden=args.mlp_dec_d_hidden)
        self.trans = trans(z_dim=z_dim, u_dim=u_dim, r_dim=r_dim, use_vr=args.use_vr, d_hidden=args.trans_d_hidden, n_layers=args.trans_n_layers, use_batchnorm=args.trans_batchnorm)

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

        if self.training:
            z = self.reparam(mu, logvar)
        else:
            z = mu

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