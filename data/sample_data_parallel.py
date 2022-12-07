import os
from os import path
import pandas as pd
from datetime import datetime
import argparse

from data.sample_data import sample_train_data

from gym.envs.registration import register
from gym.envs.registration import registry

from multiprocessing import Pool

def parallel_sampler(ENV_NAME: str,
                     process_num: int, 
                     sample_size: int, 
                     reset_steps: int,
                     obs_type: str, 
                     obs_res, 
                     step_size):

    env = (ENV_NAME.split('-')[0]).lower()

    output_dir = 'data/samples/{}'.format(env) \
        + '/{}-{}'.format(env.lower(), (datetime.now()).strftime('%m_%d_%Y_%H%M%S'))
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    sample_size = sample_size // process_num
    parameters = []
    for i in range(process_num):
        parameters.append((ENV_NAME, sample_size*i, sample_size, output_dir, reset_steps, obs_type, obs_res, step_size, True))

    with Pool() as pool:
        df_s = pool.starmap(sample_train_data, parameters)
    
    df = pd.concat(list(df_s), ignore_index=True)

    #################### After all df's are returned #####################
    df_file_name = path.join(output_dir, 'dataframe.pkl')
    df.to_pickle(df_file_name)

    # print(df)

    print(f"SAMPLING DONE! DATA SAVED AS {obs_type.upper()}-TYPE IN: {output_dir}")

def main(args):
    env = (args.env).lower()
    sample_size = args.sample_size
    obs_type = args.obs_type
    obs_res = args.obs_res
    step_size= args.step_size
    process_num = args.process_num
    reset_steps = args.reset_steps

    assert sample_size % process_num == 0
    assert reset_steps >= 100, 'Minimum reset steps should be 100.'

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

    parallel_sampler(ENV_NAME, process_num, sample_size, reset_steps, obs_type, obs_res, step_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', required=True, type=str, help='Pendulum, Planar, Hopper')
    parser.add_argument('--sample_size', required=True, type=int, help='the number of samples')
    parser.add_argument('--process_num', required=True, type=int, default=1, help='number of cores')
    parser.add_argument('--obs_type', required=True, type=str, help='type of obs to be saved')
    parser.add_argument('--obs_res', nargs='?', const=1, type=int)
    parser.add_argument('--step_size', default=4, type=int)
    parser.add_argument('--reset_steps', default=100, type=int, help='reset every this number of steps')

    args = parser.parse_args()

    main(args)