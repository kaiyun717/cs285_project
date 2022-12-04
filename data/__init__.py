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
    if 'Pendulum-v1-uniform' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
    if 'CartPole-v0-uniform' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
    if 'Planar-v0-uniform' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]

register(
    id='Hopper-v4-uniform',
    entry_point='data.env.hopper_v4_uniform:HopperEnvUniform',
    max_episode_steps=1000,     # Hopper-v4 parameter
    reward_threshold=3800.0,    # Hopper-v4 parameter
)

register(
    id='Pendulum-v1-uniform',
    entry_point='data.env.pendulum_v1_uniform:PendulumEnvUniform',
    max_episode_steps=200,      # Pendulum-v1 parameter
)

register(
    id='CartPole-v0-uniform',
    entry_point='data.env.cartpole_v0_uniform:CartPoleEnvUniform',
    max_episode_steps=200,      # CartPole-v0 parameter
    reward_threshold=195.0      # CartPole-v0 parameter
)

register(
    id='Planar-v0-uniform',
    entry_point='data.env.planar_v0_uniform:PlanarEnvUniform',
)