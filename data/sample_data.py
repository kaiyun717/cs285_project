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

def sample_train_data(env_name,
                      start_num,
                      sample_size,
                      output_dir,
                      reset_steps=100,
                      obs_type='serial',
                      obs_res=None,
                      step_size=4, 
                      apply_control=True):

    assert obs_type in ['image', 'serial']
    assert (type(obs_res) is int) and (obs_res is not None)

    env = gym.make(env_name)

    obs_size = env.observation_space.shape[0]
    
    if (env_name.split('-')[0]).lower() == 'cartpole':
        act_size = 1
    else:
        act_size = env.action_space.shape[0]

    obs_serial = np.zeros((sample_size, 2, obs_res, obs_res))
    act_serial = np.zeros((sample_size, act_size))
    rew_serial = np.zeros((sample_size,))
    done_serial = np.zeros((sample_size,))
    
    pbar = tqdm(total=sample_size)

    resets = 0  # NOTE: debug
    dones = 0   # NOTE: debug

    sample_num = 0
    while sample_num < sample_size:
        if sample_num % reset_steps == 0:
            state = env.reset()
            resets += 1 # NOTE: debug
        
        before = env.render(mode='rgb_array')

        if apply_control:
            u = env.get_valid_random_action()
        else:
            u = np.zeros((act_size,))

        steps = 0
        for _ in range(step_size):
            state, reward, done, _ = env.step(u)
            steps += 1
            if done:
                break            
        
        after = env.render(mode='rgb_array')

        if done:
            state = env.reset()
            resets += 1 # NOTE: debug

        if steps != step_size: # Got done before frame skip completed.
            continue           # Then this step shouldn't be saved.

        if obs_type == 'image': 
            before = opencv_downsample(before, (obs_res, obs_res))
            before_file = path.join(output_dir, 'before-{:06d}.jpg'.format(start_num+sample_num))
            plt.imsave(before_file, before)

            after = opencv_downsample(after, (obs_res, obs_res))
            after_file = path.join(output_dir, 'after-{:06d}.jpg'.format(start_num+sample_num))
            plt.imsave(after_file, after)

        else:
            obs_serial[sample_num][0] = rgb2gray(opencv_downsample(before, (obs_res, obs_res)))
            obs_serial[sample_num][-1] = rgb2gray(opencv_downsample(after, (obs_res, obs_res)))
        
        act_serial[sample_num] = u
        rew_serial[sample_num] = reward
        done_serial[sample_num] = int(done)

        if done:
            dones += 1  # NOTE: debug

        sample_num += 1
        pbar.update(1)
    
    print('SAMPLE NUM: ', sample_num)
    print('SAMPLE SIZE: ', sample_size)
    print('# DONES: ', dones)
    print('# RESETS: ', resets)

    if obs_type == 'serial':
        df = pd.DataFrame(
            {'before': [list(obs_serial[i][0]) for i in range(sample_size)], 
             'after': [list(obs_serial[i][-1]) for i in range(sample_size)], 
             'action': [list(act_serial[i]) for i in range(sample_size)],
             'reward': rew_serial,
             'done': done_serial})
    else:
        df = pd.DataFrame({'action': [list(act_serial[i]) for i in range(sample_size)],
                           'reward': rew_serial,
                           'done': done_serial})
    
    pbar.close()
    return df

##########################################################################################

def sampler(env, sample_size, reset_steps, obs_type, obs_res, step_size):
    env = env.lower()

    if env == 'hopper':
        ENV_NAME = 'Hopper-v4-uniform'
        if ENV_NAME not in registry.env_specs:
            register(
                id=ENV_NAME,
                entry_point='data.env.hopper_v4_uniform:HopperEnvUniform',
                max_episode_steps=1000,
                reward_threshold=3800.0,
            )
    
    elif env == 'pendulum':
        ENV_NAME = 'Pendulum-v1-uniform'
        if ENV_NAME not in registry.env_specs:
            register(
                id=ENV_NAME,
                entry_point='data.env.pendulum_v1_uniform:PendulumEnvUniform',
                max_episode_steps=200,
            )
    
    elif env == 'cartpole':
        ENV_NAME = 'CartPole-v0-uniform'
        if ENV_NAME not in registry.env_specs:
            register(
                id=ENV_NAME,
                entry_point='data.env.cartpole_v0_uniform:CartPoleEnvUniform',
                max_episode_steps=200,      # CartPole-v0 parameter
                reward_threshold=195.0,     # CartPole-v0 parameter
            )
    
    elif env == 'planar':
        ENV_NAME = 'Planar-v0-uniform'
        if ENV_NAME not in registry.env_specs:
            register(
                id=ENV_NAME,
                entry_point='data.env.planar_v0_uniform:PlanarEnvUniform',
            )
    
    output_dir = 'data/samples/{}'.format(env) \
        + '/{}-{}'.format(env.lower(), (datetime.now()).strftime('%m_%d_%Y_%H%M%S'))
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    df = sample_train_data(env_name=ENV_NAME,
                           start_num=0,
                           sample_size=sample_size,
                           output_dir=output_dir,
                           reset_steps=reset_steps,
                           obs_type=obs_type,
                           obs_res=obs_res,
                           step_size=step_size,
                           apply_control=True)
                           
    df_file_name = path.join(output_dir, 'dataframe.pkl')
    df.to_pickle(df_file_name)

    print(f"SAMPLING DONE! DATA SAVED AS {obs_type.upper()}-TYPE IN: {output_dir}")


def main(args):
    env = args.env
    sample_size = args.sample_size
    obs_type = args.obs_type
    obs_res = args.obs_res
    step_size= args.step_size
    reset_steps = args.reset_steps

    assert reset_steps >= 100, 'Minimum reset steps should be 100.'

    sampler(env, sample_size, reset_steps, obs_type, obs_res, step_size)
    

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