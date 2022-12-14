import os
from os import path
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import gym
import json
from datetime import datetime
import argparse
from functools import partial
from PIL import Image

env = gym.make('Pendulum-v1').env
width, height = 48 * 2, 48

def step(env, state, u):
    th, thdot = state  # th := theta

    g = env.g
    m = env.m
    l = env.l
    dt = env.dt

    u = np.clip(u, -env.max_torque, env.max_torque)[0]
    costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

    newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
    newthdot = np.clip(newthdot, -env.max_speed, env.max_speed)
    newth = th + newthdot * dt

    return np.array([newth, newthdot])

def render_state(env, state):
    import pygame
    from pygame import gfxdraw

    if env.screen is None:
        pygame.init()
        env.screen = pygame.Surface((env.screen_dim, env.screen_dim))
    if env.clock is None:
        env.clock = pygame.time.Clock()

    env.surf = pygame.Surface((env.screen_dim, env.screen_dim))
    env.surf.fill((255, 255, 255))

    bound = 2.2
    scale = env.screen_dim / (bound * 2)
    offset = env.screen_dim // 2

    rod_length = 1 * scale
    rod_width = 0.2 * scale
    l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
    coords = [(l, b), (l, t), (r, t), (r, b)]
    transformed_coords = []
    for c in coords:
        c = pygame.math.Vector2(c).rotate_rad(state[0] + np.pi / 2)
        c = (c[0] + offset, c[1] + offset)
        transformed_coords.append(c)
    gfxdraw.aapolygon(env.surf, transformed_coords, (204, 77, 77))
    gfxdraw.filled_polygon(env.surf, transformed_coords, (204, 77, 77))

    gfxdraw.aacircle(env.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
    gfxdraw.filled_circle(
        env.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
    )

    rod_end = (rod_length, 0)
    rod_end = pygame.math.Vector2(rod_end).rotate_rad(state[0] + np.pi / 2)
    rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
    gfxdraw.aacircle(
        env.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
    )
    gfxdraw.filled_circle(
        env.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
    )

    # drawing axle
    gfxdraw.aacircle(env.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
    gfxdraw.filled_circle(env.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

    env.surf = pygame.transform.flip(env.surf, False, True)
    env.screen.blit(env.surf, (0, 0))

    return np.transpose(
        np.array(pygame.surfarray.pixels3d(env.screen)), axes=(1, 0, 2)
    )

def render(state):
    # need two observations to restore the Markov property
    before1 = state
    before2 = step(env, state, np.array([0]))
    return map(lambda state: render_state(env, state), (before1, before2))

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def sample(sample_size):
    state_samples = []
    obs_samples = []
    for i in range(sample_size):
        th = np.random.uniform(0, np.pi * 2)
        thdot = np.random.uniform(-8, 8)

        state = np.array([th, thdot])

        initial_state = np.copy(state)
        before1, before2 = render(state)

        # apply the same control over a few timesteps
        u = np.random.uniform(-2, 2, size=(1,))

        state_next = state
        for _ in range(1):
            state_next = step(env, state_next, u)

        after_state = np.copy(state_next)
        after1, after2 = render(state_next)

        before = np.hstack((before1, before2))
        after = np.hstack((after1, after2))

        def process(arr):
            return np.asarray(Image.fromarray(arr).convert('L').resize((width, height))) / 255

        obs_samples.append((process(before), u, process(after)))
        state_samples.append((state, u, state_next))
    return state_samples, obs_samples



def sample_pendulum(sample_size, output_dir='data/pendulum_costs', step_size=1, apply_control=True, num_shards=10):
    assert sample_size % num_shards == 0

    samples = []

    if not path.exists(output_dir):
        os.makedirs(output_dir)

    for i in trange(sample_size):
        """
        for each sample:
        - draw a random state (theta, theta dot)
        - render x_t (including 2 images)
        - draw a random action u_t and apply
        - render x_t+1 after applying u_t
        """
        # th (theta) and thdot (theta dot) represent a state in Pendulum env
        th = np.random.uniform(0, np.pi * 2)
        thdot = np.random.uniform(-8, 8)

        state = np.array([th, thdot])

        initial_state = np.copy(state)
        before1, before2 = render(state)

        # apply the same control over a few timesteps
        if apply_control:
            u = np.random.uniform(-2, 2, size=(1,))
        else:
            u = np.zeros((1,))

        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        for _ in range(step_size):
            state = step(env, state, u)

        after_state = np.copy(state)
        after1, after2 = render(state)

        before = np.hstack((before1, before2))
        after = np.hstack((after1, after2))

        shard_no = i // (sample_size // num_shards)

        shard_path = path.join('{:03d}-of-{:03d}'.format(shard_no, num_shards))

        if not path.exists(path.join(output_dir, shard_path)):
            os.makedirs(path.join(output_dir, shard_path))

        before_file = path.join(shard_path, 'before-{:05d}.jpg'.format(i))
        plt.imsave(path.join(output_dir, before_file), before)

        after_file = path.join(shard_path, 'after-{:05d}.jpg'.format(i))
        plt.imsave(path.join(output_dir, after_file), after)

        samples.append({
            'before_state': initial_state.tolist(),
            'after_state': after_state.tolist(),
            'before': before_file,
            'after': after_file,
            'control': u.tolist(),
            'costs': costs.item(),
        })

    with open(path.join(output_dir, 'data.json'), 'wt') as outfile:
        json.dump(
            {
                'metadata': {
                    'num_samples': sample_size,
                    'step_size': step_size,
                    'apply_control': apply_control,
                    'time_created': str(datetime.now()),
                    'version': 1
                },
                'samples': samples
            }, outfile, indent=2)

    env.viewer.close()

def main(args):
    sample_size = args.sample_size

    sample_pendulum(sample_size=sample_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sample data')

    parser.add_argument('--sample_size', required=True, type=int, help='the number of samples')

    args = parser.parse_args()

    main(args)