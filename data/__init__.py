from . import env
import gym
import logging
from gym.envs.registration import register


logger = logging.getLogger(__name__)

env_dict = gym.envs.registration.registry.env_specs.copy()

for env in env_dict:
    if 'Hopper-v4-uniform' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]

register(
    id='Hopper-v4-uniform',
    entry_point='data.env.hopper_v4_uniform:HopperEnvUniform',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)