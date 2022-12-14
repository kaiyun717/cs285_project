import torch

torch.set_default_dtype(torch.float64)

class NormalDistribution:
    def __init__(self, mean, logvar, A=None, v=None, r=None):
        """
        :param mean: mu in the paper
        :param logvar: \Sigma in the paper
        :param v:
        :param r:
        if A is not None then covariance matrix = A \Sigma A^T, where A = I + v^T r
        else the covariance matrix is simply diag(logvar.exp())
        """
        self.mean = mean
        self.logvar = logvar
        batch_size, n = mean.shape

        sigma = torch.diag_embed(torch.exp(logvar))

        if v is not None and r is not None:
            self.v = v
            self.r = r
            assert v.shape == (batch_size, n), f"v.shape == {v.shape}"
            assert r.shape == (batch_size, n), f"r.shape == {r.shape}"
            A = torch.eye(mean.shape[-1], device=v.device)[None] + torch.bmm(self.v.unsqueeze(-1), self.r.unsqueeze(-2))
            self.cov = A.bmm(sigma.bmm(A.transpose(1, 2)))
            self.cov_inv = None
            self.A = A
        elif A is not None:
            self.cov = A.bmm(sigma.bmm(A.transpose(1, 2)))
            self.cov_inv = None
            self.A = A
        else:
            self.cov = sigma
            self.cov_inv = torch.diag_embed(1 / torch.exp(logvar))
            self.A = None


    def KL_divergence(q_z_next_pred, q_z_next):
        """
        :param q_z_next_pred: q(z_{t+1} | z_bar_t, q_z_t, u_t) using the transition
        :param q_z_next: q(z_t+1 | x_t+1) using the encoder
        :return: KL divergence between two distributions
        """
        mu_0 = q_z_next_pred.mean
        mu_1 = q_z_next.mean
        sigma_0 = q_z_next_pred.cov
        sigma_1 = q_z_next.cov

        if q_z_next_pred.A is None:
            logdet_0 = torch.sum(q_z_next_pred.logvar, dim=1)
        else:
            logdet_0 = torch.logdet(q_z_next_pred.cov)

        if q_z_next.A is None:
            logdet_1 = torch.sum(q_z_next.logvar, dim=1)
        else:
            logdet_1 = torch.logdet(q_z_next.cov)

        if q_z_next.cov_inv is None:
            q_z_next.cov_inv = torch.linalg.inv(sigma_1)
        sigma_1_inv = q_z_next.cov_inv
        k = q_z_next_pred.mean.size(1)
        batch_size = q_z_next_pred.mean.size(0)

        # Multivariate Gaussian KL divergence
        prod = torch.bmm(sigma_1_inv, sigma_0)
        assert prod.shape == (batch_size, k, k)
        trace = torch.diagonal(prod, offset=0, dim1=-1, dim2=-2).sum(-1)

        quadratic_term = torch.bmm(torch.bmm((mu_1 - mu_0).unsqueeze(1), sigma_1_inv), (mu_1 - mu_0).unsqueeze(2))
        assert quadratic_term.shape == (batch_size, 1, 1)

        result = 0.5 * (logdet_1 - logdet_0 - k + trace + quadratic_term.squeeze(-1).squeeze(-1))
        return torch.mean(result)