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

import gym
from gym.envs.registration import register
from gym.envs.registration import registry

def opencv_downsample(image: np.ndarray, out_size: tuple):
    resized = cv2.resize(image, dsize=out_size, interpolation=cv2.INTER_CUBIC)
    return resized
        
def sample_eval_data(env_name,
                     sample_size, 
                     obs_res=64,
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

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]

    samples = []
    for i in trange(sample_size):
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

        before = opencv_downsample(before, (obs_res, obs_res))
        after = opencv_downsample(after, (obs_res, obs_res))
        samples.append((before, u, after))
    
    return samples
