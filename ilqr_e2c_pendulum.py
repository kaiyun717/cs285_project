import torch
import numpy as np
from e2c_model import E2C
from data import sample_pendulum_data
import matplotlib.pyplot as plt
from PIL import Image
import ilqr
from moviepy.editor import ImageSequenceClip
import sys

device = 'cpu'
env_name = 'pendulum'

settings = {'planar': (1600, 3, 2), 'pendulum': (4608, 5, 1)}
obs_dim, z_dim, u_dim = settings[env_name]

dyn_rank=2
model = E2C(obs_dim=obs_dim, z_dim=z_dim, u_dim=u_dim, env=f'{env_name}', dyn_rank=dyn_rank).to(device)
model.load_state_dict(torch.load('result/pendulum/pendulum_mlp_z5_lam100_rank2_jacloss_kl/model_30', map_location=device))
model.eval()

def process(arr):
    return np.asarray(Image.fromarray(arr).convert('L').resize((48, 48))) / 255

# s = np.array([3, 2.0])
# x, xn = sample_pendulum_data.render(s)
# x = np.hstack((process(x), process(xn)))
# z, _ = model.encode(torch.tensor(x, device=device).flatten()[None])
# x_recon = model.decode(z)
# plt.imshow(x)
# plt.show()
# print(x_recon.shape, x_recon.min(), x_recon.max())
# plt.imshow(x_recon.detach().cpu().numpy()[0].reshape(48, 96))
# plt.show()

def plot_2d():
    # fig, axs = plt.subplots(3, 2)
    # vis_dims = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    fig, axs = plt.subplots(1, 1)
    vis_dims = [(0, 1)]

    latents = []

    points = []
    collide_points = []
    colors = []
    for j in range(0, 40):
        for i in range(0, 40):
            th = i / 40 * 2 * np.pi
            thdot = 8 * (j / 40 - 0.5)
            s = np.array([th, thdot])
            target_image, target_image_next = sample_pendulum_data.render(s)
            target_image = np.hstack((process(target_image), process(target_image_next)))

            target_latent, _ = model.encode(torch.tensor(target_image, device=device).flatten()[None])

            latents.append(target_latent[0].detach().cpu().numpy())
            colors.append(((np.cos(th)+1)/2, (np.sin(th)+1)/2, j / 40))

    # plt.scatter([p for p, q in points], [q for p, q in points])
    # plt.scatter([p for p, q in collide_points], [q for p, q in collide_points], color='red')
    fig, axs = plt.subplots(z_dim, z_dim)
    for i in range(z_dim):
        for j in range(z_dim):
            axs[i, j].scatter([l[i] for l in latents], [l[j] for l in latents], c=colors)
    plt.show()

# plot_2d()

def state_to_latent(s):
    image, image_next = sample_pendulum_data.render(s)
    image = np.hstack((process(image), process(image_next)))
    z, _ = model.encode(torch.tensor(image, device=device).flatten()[None])
    z = z[0].detach().cpu().numpy()
    return z

def latent_to_obs(z):
    x_recon = model.decode(torch.tensor(z, device=device)[None])
    return x_recon.detach().cpu().numpy()[0].reshape(48, 96)

ztarget = state_to_latent(np.zeros(2))

def to_numpy(t):
    return t.detach().cpu().numpy()

def unbatchify(t):
    return to_numpy(t.squeeze(0))


def dynamics(z_np, u_np, t, is_last):
    z = torch.tensor(z_np[None], device=device)
    u = torch.tensor(u_np[None], device=device)

    trans = model.trans
    h = trans.net(z)
    v, r = trans.fc_A(h).chunk(2, dim=1)
    v = v.view(-1, z_dim, dyn_rank)
    r = r.view(-1, dyn_rank, z_dim)
    Fx = torch.eye(trans.z_dim, device=device)[None] + torch.bmm(v, r)
    Fu = trans.fc_B(h).view(-1, trans.z_dim, trans.u_dim)
    f0 = trans.fc_o(h)
    z_next = Fx.bmm(z.unsqueeze(-1)).squeeze(-1) + Fu.bmm(u.unsqueeze(-1)).squeeze(-1) + f0

    Fx = unbatchify(Fx)
    Fu = unbatchify(Fu)
    z_next = unbatchify(z_next)

    Gx = np.vstack((np.eye(z_dim), np.zeros((1, z_dim))))
    # if not is_last:
    #     Gx /= 40
    Gu = np.vstack((np.zeros((z_dim, 1)), np.eye(1))) * 0.01
    residuals = Gx @ (z_np - ztarget) + Gu @ u_np
    cost = (residuals ** 2).sum() / 2

    return z_next, cost, residuals, Fx, Fu, Gx, Gu, np.zeros(z_dim), np.zeros(1)

# z0 = state_to_latent(np.array([1, 0]))
# z = z0
# for _ in range(10):
#     plt.imshow(latent_to_obs(z))
#     plt.show()
#     for _ in range(5):
#         z = dynamics(z, np.ones(1))[0]

state = np.array([3.5, 1])

xbar, ubar, K, cost_total = ilqr.do_ilqr(
    dynamics,
    state_to_latent(state),
    z_dim, u_dim, z_dim + u_dim, horizon=15, num_iters=5, verbose=False)


images = []
for i in range(100):
    for j in range(5):
        x, xn = sample_pendulum_data.render(state)
        x = np.hstack((process(x), process(xn)))
        images.append(x[:, :, None].repeat(3, axis=2) * 255)

        z = state_to_latent(state)

        xbar, ubar, K, cost_total = ilqr.do_ilqr(
            dynamics,
            z,
            z_dim, u_dim, z_dim + u_dim, horizon=15, num_iters=5, verbose=False)

        u = ubar[0] # + K[0] @ (z - xbar[0])
        ubar = np.concatenate([ubar[1:], np.zeros_like(ubar[:1])], axis=0)
        xbar = np.concatenate([xbar[1:], np.zeros_like(xbar[:1])], axis=0)
        K = np.concatenate([K[1:], np.zeros_like(K[:1])], axis=0)

        state = sample_pendulum_data.step(sample_pendulum_data.env, state, u)

    # plt.imshow(x)
    # plt.show()
ImageSequenceClip(images, fps=20).write_videofile(f'/tmp/ilqr_rank{dyn_rank}.mp4')
