import os
from os import path
import pandas as pd
from datetime import datetime
import argparse

from data.sample_data import sample_training_data

from gym.envs.registration import register
from gym.envs.registration import registry

from multiprocessing import Pool
import numpy as np

def parallel_sampler(env_name: str,
                     process_num: int, 
                     sample_size: int, 
                     reset_steps: int,
                     obs_type: str, 
                     obs_res, 
                     step_size):

    env = (env_name.split('-')[0]).lower()

    output_dir = f"data/samples/{env}/{env}-{datetime.now().strftime('%m_%d_%Y_%H%M%S')}"
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    sample_size = sample_size // process_num
    parameters = []
    for i in range(process_num):
        parameters.append((env_name, sample_size, reset_steps))

    with Pool() as pool:
        print(parameters)
        data_dicts = pool.starmap(sample_training_data, parameters)
    
    data_dict_combined = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if key not in data_dict_combined:
                data_dict_combined[key] = value
            else:
                data_dict_combined[key] = np.concatenate((data_dict_combined[key], value), axis=0)

    #################### After all df's are returned #####################
    df_file_name = path.join(output_dir, 'dataframe.npz')
    np.savez_compressed(df_file_name, **data_dict_combined)

    print(f"SAMPLING DONE! DATA SAVED AS {obs_type.upper()}-TYPE IN: {output_dir}")

def main(args):
    sample_size = args.sample_size
    obs_type = args.obs_type
    obs_res = args.obs_res
    step_size= args.step_size
    process_num = args.process_num
    reset_steps = args.reset_steps

    assert sample_size % process_num == 0
    assert reset_steps >= 100, 'Minimum reset steps should be 100.'

    parallel_sampler(args.env, process_num, sample_size, reset_steps, obs_type, obs_res, step_size)

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