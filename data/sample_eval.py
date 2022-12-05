from configparser import Interpolation
import os
from os import path
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import cv2
from torchvision.transforms import ToTensor

import torch
import gym
from gym.envs.registration import register
from gym.envs.registration import registry

def opencv_downsample(image: np.ndarray, out_size: tuple):
    resized = cv2.resize(image, dsize=out_size, interpolation=cv2.INTER_CUBIC)
    return resized

def process_eval_data(eval_data, sample_size, stack):
    processed = []
    for i in range(stack-1, sample_size):
        before = []
        after = []

        for t in reversed(range(stack)):
            b_idx = i - t
            before.append(eval_data[b_idx][0])
            after.append(eval_data[b_idx][2])
            # before.append(process_image(temp_before))
            # after.append(process_image(temp_after))
        
        processed.append((torch.cat(tuple(before)),
                          np.array(eval_data[i][1]),
                          torch.cat(tuple(after))))
    
    return processed

def sample_eval_data(env_name,
                     sample_size, 
                     obs_res=64,
                     stack=1,
                     step_size=4):

    if env_name == 'hopper':
        ENV_NAME = 'Hopper-v4-uniform'
        if ENV_NAME not in registry.env_specs:
            register(
                id=ENV_NAME,
                entry_point='data.env.hopper_v4_uniform:HopperEnvUniform',
                max_episode_steps=1000,
                reward_threshold=3800.0,
            )
    
    elif env_name == 'pendulum':
        ENV_NAME = 'Pendulum-v1-uniform'
        if ENV_NAME not in registry.env_specs:
            register(
                id=ENV_NAME,
                entry_point='data.env.pendulum_v1_uniform:PendulumEnvUniform',
                max_episode_steps=200,
            )
    
    elif env_name == 'cartpole':
        ENV_NAME = 'CartPole-v0-uniform'
        if ENV_NAME not in registry.env_specs:
            register(
                id=ENV_NAME,
                entry_point='data.env.cartpole_v0_uniform:CartPoleEnvUniform',
                max_episode_steps=200,      # CartPole-v0 parameter
                reward_threshold=195.0,     # CartPole-v0 parameter
            )

    elif env_name == 'planar':
        ENV_NAME = 'Planar-v0-uniform'
        if ENV_NAME not in registry.env_specs:
            register(
                id=ENV_NAME,
                entry_point='data.env.planar_v0_uniform:PlanarEnvUniform',
            )
        assert step_size == 1

    env = gym.make(ENV_NAME)

    samples = []
    for _ in range(sample_size):
        state = env.reset() # Reset everystep
        
        before = env.render(mode='rgb_array')

        u = env.action_space.sample()
        
        if env_name == 'planar':    # Check if action is valid for planar env
            while True:
                if env.is_valid_action(state[-2:], u):
                    break
                else:
                    u = env.action_space.sample()

        for _ in range(step_size):
            state, reward, done, _ = env.step(u)
            if done:
                break            

        after = env.render(mode='rgb_array')

        print('RENDERED BEFORE: ', before.shape)

        before = before.convert('L')
        after = before.convert('L')

        before = ToTensor()((opencv_downsample(before, (obs_res, obs_res))))
        after = ToTensor()((opencv_downsample(after, (obs_res, obs_res))))
        samples.append((before, u, after))
    
    processed_samples = process_eval_data(samples, sample_size, stack)
    return processed_samples
