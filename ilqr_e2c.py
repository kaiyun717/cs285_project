import torch
import numpy as np
from e2c_model import E2C
from data import sample_planar
import matplotlib.pyplot as plt

device = 'cpu'
env_name = 'planar'

settings = {'planar': (1600, 3, 2), 'pendulum': (4608, 3, 1)}
obs_dim, z_dim, u_dim = settings[env_name]

model = E2C(obs_dim=obs_dim, z_dim=z_dim, u_dim=u_dim, env=f'{env_name}', dyn_rank=1).to(device)
model.load_state_dict(torch.load('result/planar/planar_mlp_z3_lam100/model_2190', map_location=device))
model.eval()

def plot_2d():
    # fig, axs = plt.subplots(3, 2)
    # vis_dims = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    fig, axs = plt.subplots(1, 1)
    vis_dims = [(0, 1)]

    points = []
    collide_points = []
    for i in range(0, 40):
        for j in range(0, 40):
            s = np.array([i, j])

            target_image = sample_planar.render(s)
            target_latent, _ = model.encode(torch.tensor(target_image, device=device).flatten()[None])

            if sample_planar.is_colliding(s):
                collide_points.append((target_latent[0, 0].item(), target_latent[0, 1].item()))
                # for b, dims in enumerate(vis_dims):
                #     axs.scatter(
                #         [target_latent[0, dims[0]].item()],
                #         [target_latent[0, dims[1]].item()],
                #         color='red')
                # ax.scatter([target_latent[0, 0].item()], [target_latent[0, 1].item()], [target_latent[0, 2].item()], color='red')
            else:
                points.append((target_latent[0, 0].item(), target_latent[0, 1].item()))
                # ax.scatter([target_latent[0, 0].item()], [target_latent[0, 1].item()], [target_latent[0, 2].item()], color=((i+j)/80, i/40, j/40))
                # ax.scatter([target_latent[0, 0].item()], [target_latent[0, 1].item()], color=((i+j)/80, i/40, j/40))
                for b, dims in enumerate(vis_dims):
                    axs.scatter(
                        [target_latent[0, dims[0]].item()],
                        [target_latent[0, dims[1]].item()],
                        color=((i+j)/80, i/40, j/40))

    # plt.scatter([p for p, q in points], [q for p, q in points])
    # plt.scatter([p for p, q in collide_points], [q for p, q in collide_points], color='red')
    plt.show()

def plot_3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs = []
    ys = []
    zs = []
    colors = []

    for i in range(0, 40):
        for j in range(0, 40):
            s = np.array([i, j])

            target_image = sample_planar.render(s)
            target_latent, _ = model.encode(torch.tensor(target_image, device=device).flatten()[None])

            xs.append(target_latent[0, 0].item())
            ys.append(target_latent[0, 1].item())
            zs.append(target_latent[0, 2].item())
            if sample_planar.is_colliding(s) == 1:
                colors.append([1, 1, 1])
            elif sample_planar.is_colliding(s) == 2:
                colors.append([0, 0, 0])
            else:
                colors.append([(i+j)/80, i/40, j/40])

    ax.scatter(xs, ys, zs, c=colors, s=15)
    plt.show()

plot_3d()
# print(target_latent)

def dynamics(z, u):
    trans = model.transition
    h = trans.net(z)
    v, r = trans.fc_A(h)
    Fx = torch.eye(trans.z_dim, device=device)[None] + torch.bmm(v.unsqueeze(-1), r.unsqueeze(-2))
    Fu = trans.fc_B(h).view(-1, trans.z_dim, trans.u_dim)
    f0 = trans.fc_o(h)
    z_next = Fx.bmm(z.unsqueeze(-1)).squeeze(-1) + Fu.bmm(u.unsqueeze(-1)).squeeze(-1) + f0

    Gx = torch.tensor([
        [1, 0],
        [0, 1],
    ], device=device)
    Gu = torch.tensor([

    ])

    return z_next, 
