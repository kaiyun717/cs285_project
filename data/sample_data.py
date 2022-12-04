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

def maxpool_downsample(image: np.ndarray, out_size:int):
    """ https://scipython.com/blog/binning-a-2d-array-in-numpy/
        https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
    """
    in_size = image.shape[1]
    assert in_size > out_size
    
    last = image.shape[0] > image.shape[-1] # rgb info is in the last index
    if last:
        rgb = image.shape[-1]
        assert rgb == 3     # seems like rgb photos can only do 'last' ordering
        
        bin_size = in_size // out_size
        small_image = image.reshape((out_size, bin_size,
                                     out_size, bin_size, rgb)).max(3).max(1)
    else:
        rgb = image.shape[0]
        
        bin_size = in_size // out_size
        small_image = image.reshape((rgb, out_size, bin_size,
                                     out_size, bin_size)).max(4).max(2)

    return small_image

def opencv_downsample(image: np.ndarray, out_size: tuple):
    resized = cv2.resize(image, dsize=out_size, interpolation=cv2.INTER_CUBIC)
    return resized
        
def sample_hopper(sample_size, 
                  output_dir='data/samples/hopper',
                  obs_type='serial',
                  obs_res=None,
                  step_size=4, 
                  apply_control=True):

    assert obs_type in ['image', 'serial']
    assert obs_res == None if obs_type == 'serial' else type(obs_res) is int

    env = gym.make('Hopper-v4-uniform')

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]

    obs_serial = np.zeros((sample_size, 2, obs_size))
    act_serial = np.zeros((sample_size, act_size))
    rew_serial = np.zeros((sample_size,))
    done_serial = np.zeros((sample_size,))
    
    output_dir = output_dir \
        + '/hopper-' + (datetime.now()).strftime('%m_%d_%Y_%H%M%S')
    if not path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in trange(sample_size):
        state = env.reset() # Reset everystep
        
        before_state = state
        before = env.render(mode='rgb_array')

        if apply_control:
            u = env.action_space.sample()
        else:
            u = np.zeros((3,))

        for _ in range(step_size):
            state, reward, done, _ = env.step(u)
            if done:
                break            

        after_state = state
        after = env.render(mode='rgb_array')

        if obs_type == 'image': 
            before = opencv_downsample(before, (obs_res, obs_res))
            before_file = path.join(output_dir, 'before-{:06d}.jpg'.format(i))
            plt.imsave(before_file, before)

            after = opencv_downsample(after, (obs_res, obs_res))
            after_file = path.join(output_dir, 'after-{:06d}.jpg'.format(i))
            plt.imsave(after_file, after)

        else:
            obs_serial[i][0] = before_state
            obs_serial[i][-1] = after_state
        
        act_serial[i] = u
        rew_serial[i] = reward
        done_serial[i] = int(done)

    if obs_type == 'serial':
        df = pd.DataFrame(
            {'before': [list(obs_serial[i][0]) for i in range(sample_size)], 
             'after': [list(obs_serial[i][-1]) for i in range(sample_size)], 
             'action': [list(act_serial[i]) for i in range(sample_size)],
             'reward': rew_serial,
             'done': done_serial
             }
        )
    else:
        df = pd.DataFrame({'action': [list(act_serial[i]) for i in range(sample_size)],
                           'reward': rew_serial,
                           'done': done_serial})
    
    df_file_name = path.join(output_dir, 'dataframe.pkl')
    df.to_pickle(df_file_name)

    print(f"SAMPLING DONE! DATA SAVED AS {obs_type.upper()}-TYPE IN: {output_dir}")

##########################################################################################

def sample_pendulum(sample_size, 
                    output_dir='data/samples/pendulum', 
                    obs_type='serial',
                    obs_res=None,
                    step_size=4, 
                    apply_control=True):

    assert obs_type in ['image', 'serial']
    assert obs_res == None if obs_type == 'serial' else type(obs_res) is int
    
    env = gym.make('Pendulum-v1-uniform')

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]

    obs_serial = np.zeros((sample_size, 2, obs_size))
    act_serial = np.zeros((sample_size, act_size))
    rew_serial = np.zeros((sample_size,))
    done_serial = np.zeros((sample_size,))
    
    output_dir = output_dir \
        + '/pendulum-' + (datetime.now()).strftime('%m_%d_%Y_%H%M%S')
    if not path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in trange(sample_size):
        state = env.reset()
        
        before_state = state
        before = env.render(mode='rgb_array')

        if apply_control:
            u = env.action_space.sample()
        else:
            u = np.zeros((act_size,))

        for _ in range(step_size):
            state, reward, done, _ = env.step(u)
            if done:
                break            

        after_state = state
        after = env.render(mode='rgb_array')

        if obs_type == 'image': 
            before = opencv_downsample(before, (obs_res, obs_res))
            before_file = path.join(output_dir, 'before-{:06d}.jpg'.format(i))
            plt.imsave(before_file, before)

            after = opencv_downsample(after, (obs_res, obs_res))
            after_file = path.join(output_dir, 'after-{:06d}.jpg'.format(i))
            plt.imsave(after_file, after)

        else:
            obs_serial[i][0] = before_state
            obs_serial[i][-1] = after_state
        
        act_serial[i] = u
        rew_serial[i] = reward
        done_serial[i] = int(done)

    if obs_type == 'serial':
        df = pd.DataFrame(
            {'before': [list(obs_serial[i][0]) for i in range(sample_size)], 
             'after': [list(obs_serial[i][-1]) for i in range(sample_size)], 
             'action': [list(act_serial[i]) for i in range(sample_size)],
             'reward': rew_serial,
             'done': done_serial
             }
        )
    else:
        df = pd.DataFrame({'action': [list(act_serial[i]) for i in range(sample_size)],
                           'reward': rew_serial,
                           'done': done_serial})
    
    df_file_name = path.join(output_dir, 'dataframe.pkl')
    df.to_pickle(df_file_name)

    print(f"SAMPLING DONE! DATA SAVED AS {obs_type.upper()}-TYPE IN: {output_dir}")

##########################################################################################

def sample_cartpole(sample_size, 
                    output_dir='data/samples/cartpole', 
                    obs_type='serial',
                    obs_res=None,
                    step_size=4, 
                    apply_control=True):

    assert obs_type in ['image', 'serial']
    assert obs_res == None if obs_type == 'serial' else type(obs_res) is int
    
    env = gym.make('CartPole-v0-uniform')

    obs_size = env.observation_space.shape[0]
    act_size = 1

    obs_serial = np.zeros((sample_size, 2, obs_size))
    act_serial = np.zeros((sample_size, act_size))
    rew_serial = np.zeros((sample_size,))
    done_serial = np.zeros((sample_size,))
    
    output_dir = output_dir \
        + '/cartpole-' + (datetime.now()).strftime('%m_%d_%Y_%H%M%S')
    if not path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in trange(sample_size):
        state = env.reset()
        
        before_state = state
        before = env.render(mode='rgb_array')

        if apply_control:
            u = env.action_space.sample()
        else:
            u = np.zeros((act_size,))

        for _ in range(step_size):
            state, reward, done, _ = env.step(u)
            if done:
                break

        after_state = state
        after = env.render(mode='rgb_array')

        if obs_type == 'image': 
            before = opencv_downsample(before, (obs_res, obs_res))
            before_file = path.join(output_dir, 'before-{:06d}.jpg'.format(i))
            plt.imsave(before_file, before)

            after = opencv_downsample(after, (obs_res, obs_res))
            after_file = path.join(output_dir, 'after-{:06d}.jpg'.format(i))
            plt.imsave(after_file, after)

        else:
            obs_serial[i][0] = before_state
            obs_serial[i][-1] = after_state
        
        act_serial[i] = u
        rew_serial[i] = reward
        done_serial[i] = int(done)

    if obs_type == 'serial':
        df = pd.DataFrame(
            {'before': [list(obs_serial[i][0]) for i in range(sample_size)], 
             'after': [list(obs_serial[i][-1]) for i in range(sample_size)], 
             'action': [list(act_serial[i]) for i in range(sample_size)],
             'reward': rew_serial,
             'done': done_serial
             }
        )
    else:
        df = pd.DataFrame({'action': [list(act_serial[i]) for i in range(sample_size)],
                           'reward': rew_serial,
                           'done': done_serial})
    
    df_file_name = path.join(output_dir, 'dataframe.pkl')
    df.to_pickle(df_file_name)

    print(f"SAMPLING DONE! DATA SAVED AS {obs_type.upper()}-TYPE IN: {output_dir}")

##########################################################################################

def sample_planar(sample_size, 
                  output_dir='data/samples/planar', 
                  obs_type='serial',
                  obs_res=None,
                  # step_size=4, 
                  apply_control=True):

    assert obs_type in ['image', 'serial']
    assert obs_res == None if obs_type == 'serial' else obs_res == 40
    
    env = gym.make('Planar-v0-uniform')

    obs_size = env.observation_space.shape[0]   # 6*2 for obstacles, 1*2 for goal, 1*2 for state
    act_size = env.action_space.shape[0]

    obs_serial = np.zeros((sample_size, 2, obs_size))
    act_serial = np.zeros((sample_size, act_size))
    rew_serial = np.zeros((sample_size,))
    done_serial = np.zeros((sample_size,))
    
    output_dir = output_dir \
        + '/planar-' + (datetime.now()).strftime('%m_%d_%Y_%H%M%S')
    if not path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in trange(sample_size):
        state = env.reset()
        
        before_state = state
        before = env.render(mode='rgb_array')

        if apply_control:
            u = env.action_space.sample()
        else:
            u = np.zeros((act_size,))

        while True:
            if env.is_valid_action(state[-2:], u):
                break
            else:
                u = env.action_space.sample()

        state, reward, done, _ = env.step(u)
        
        after_state = state
        after = env.render(mode='rgb_array')

        if obs_type == 'image': 
            before = opencv_downsample(before, (obs_res, obs_res))
            before_file = path.join(output_dir, 'before-{:06d}.jpg'.format(i))
            plt.imsave(before_file, before)

            after = opencv_downsample(after, (obs_res, obs_res))
            after_file = path.join(output_dir, 'after-{:06d}.jpg'.format(i))
            plt.imsave(after_file, after)

        else:
            obs_serial[i][0] = before_state
            obs_serial[i][-1] = after_state
        
        act_serial[i] = u
        rew_serial[i] = reward
        done_serial[i] = int(done)

    if obs_type == 'serial':
        df = pd.DataFrame(
            {'before': [list(obs_serial[i][0]) for i in range(sample_size)], 
             'after': [list(obs_serial[i][-1]) for i in range(sample_size)], 
             'action': [list(act_serial[i]) for i in range(sample_size)],
             'reward': rew_serial,
             'done': done_serial
             }
        )
    else:
        df = pd.DataFrame({'action': [list(act_serial[i]) for i in range(sample_size)],
                           'reward': rew_serial,
                           'done': done_serial})
    
    df_file_name = path.join(output_dir, 'dataframe.pkl')
    df.to_pickle(df_file_name)

    print(f"SAMPLING DONE! DATA SAVED AS {obs_type.upper()}-TYPE IN: {output_dir}")

##########################################################################################

def sampler(env, sample_size, obs_type, obs_res, step_size):
    if env == 'Hopper':
        ENV_NAME = 'Hopper-v4-uniform'
        if ENV_NAME not in registry.env_specs:
            register(
                id=ENV_NAME,
                entry_point='data.env.hopper_v4_uniform:HopperEnvUniform',
                max_episode_steps=1000,
                reward_threshold=3800.0,
            )
    
        sample_hopper(sample_size=sample_size, obs_type=obs_type, obs_res=obs_res, step_size=step_size)
    
    elif env == 'Pendulum':
        ENV_NAME = 'Pendulum-v1-uniform'
        if ENV_NAME not in registry.env_specs:
            register(
                id=ENV_NAME,
                entry_point='data.env.pendulum_v1_uniform:PendulumEnvUniform',
                max_episode_steps=200,
            )

        sample_pendulum(sample_size=sample_size, obs_type=obs_type, obs_res=obs_res, step_size=step_size)
    
    elif env == 'CartPole':
        ENV_NAME = 'CartPole-v0-uniform'
        if ENV_NAME not in registry.env_specs:
            register(
                id=ENV_NAME,
                entry_point='data.env.cartpole_v0_uniform:CartPoleEnvUniform',
                max_episode_steps=200,      # CartPole-v0 parameter
                reward_threshold=195.0,     # CartPole-v0 parameter
            )

        sample_cartpole(sample_size=sample_size, obs_type=obs_type, obs_res=obs_res, step_size=step_size)
    
    elif env == 'Planar':
        ENV_NAME = 'Planar-v0-uniform'
        if ENV_NAME not in registry.env_specs:
            register(
                id=ENV_NAME,
                entry_point='data.env.planar_v0_uniform:PlanarEnvUniform',
            )

        sample_planar(sample_size=sample_size, obs_type=obs_type, obs_res=obs_res)

def main(args):
    env = args.env
    sample_size = args.sample_size
    obs_type = args.obs_type
    obs_res = args.obs_res
    step_size= args.step_size

    sampler(env, sample_size, obs_type, obs_res, step_size)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', required=True, type=str, help='Pendulum, Planar, Hopper')
    parser.add_argument('--sample_size', required=True, type=int, help='the number of samples')
    parser.add_argument('--obs_type', required=True, type=str, help='type of obs to be saved')
    parser.add_argument('--obs_res', nargs='?', const=1, type=int)
    parser.add_argument('--step_size', default=4, type=int)
    # parser.add_argument('--seed', required=   True, default=1)    # Seed shouldn't matter since each reset is uniform

    args = parser.parse_args()

    main(args)