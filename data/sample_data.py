from configparser import Interpolation
import os
from os import path
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
from utils.np_img_process import opencv_downsample, rgb2gray

import gym
from gym.envs.registration import register
from gym.envs.registration import registry

def sample_training_data(env_name, dataset_size, reset_steps):
    env = gym.make(env_name)
    obs_shape = env.observation_space.shape
    ac_shape = env.action_space.shape

    observations = np.zeros((dataset_size, *obs_shape), dtype=np.float32)
    next_observations = np.zeros((dataset_size, *obs_shape), dtype=np.float32)
    actions = np.zeros((dataset_size, *ac_shape), dtype=np.float32)
    rewards = np.zeros((dataset_size,), dtype=np.float32)
    dones = np.zeros((dataset_size,), dtype=np.float32)

    sample_num = 0
    current_ep_len = 0

    obs = env.reset()

    for _ in trange(dataset_size):
        current_ep_len += 1

        u = env.action_space.sample()
        next_obs, reward, done, _ = env.step(u)

        observations[sample_num, ...] = obs
        actions[sample_num, ...] = u
        rewards[sample_num] = reward
        next_observations[sample_num, ...] = next_obs
        dones[sample_num] = done

        if done or (current_ep_len >= reset_steps):
            obs = env.reset()
            current_ep_len = 0
        else:
            obs = next_obs

        sample_num += 1

    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        dones=dones,
    )


def main(args):
    env = args.env
    sample_size = args.sample_size
    reset_steps = args.reset_steps

    assert reset_steps >= 100, 'Minimum reset steps should be 100.'

    data = sample_training_data(env, sample_size, reset_steps)
    np.savez_compressed(f'data/{env}_sample_{sample_size}.npz', **data)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', required=True, type=str, help='Pendulum, Planar, Hopper, CartPole')
    parser.add_argument('--sample_size', required=True, type=int, help='the number of samples')
    parser.add_argument('--obs_type', required=True, type=str, help='type of obs to be saved')
    parser.add_argument('--obs_res', nargs='?', const=1, type=int)
    parser.add_argument('--step_size', default=4, type=int)
    parser.add_argument('--reset_steps', default=100, type=int, help='reset every this number of steps')
    # parser.add_argument('--seed', required=   True, default=1)    # Seed shouldn't matter since each reset is uniform

    args = parser.parse_args()

    main(args)