import gym
from gym.wrappers.pixel_observation import PixelObservationWrapper
import numpy as np
import gym.spaces

from data.env.hopper_v4_uniform import HopperEnvUniform
from data.env.pendulum_v1_uniform import PendulumEnvUniform
from data.env.cartpole_v0_uniform import CartPoleEnvUniform
from data.env.planar_v0_uniform import PlanarEnvUniform

class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            step_returns = self.env.step(action)
            if len(step_returns) == 4:
                obs, reward, done, info = step_returns
                truncated = False
            elif len(step_returns) == 5:
                obs, reward, done, truncated, info = step_returns
            total_reward += reward
            if done or truncated:
                break
        return obs, total_reward, done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class SelectKeyWrapper(gym.Wrapper):
    def __init__(self, env, key):
        super().__init__(env)
        self._key = key
        self.observation_space = env.observation_space[key]

    def step(self, action):
        step_returns = self.env.step(action)
        if len(step_returns) == 4:
            observation, reward, done, info = step_returns
            return self.observation(observation), reward, done, self.info(info, observation)
        if len(step_returns) == 5:
            observation, reward, done, truncated, info = step_returns
            return self.observation(observation), reward, done, truncated, self.info(info, observation)
        raise ValueError('step() should return 4 or 5 values')
    
    def reset(self, **kwargs):
        if kwargs.get("return_info", False):
            obs, info = self.env.reset(**kwargs)
            return self.observation(obs), self.info(info, obs)
        else:
            return self.observation(self.env.reset(**kwargs))
        
    def observation(self, observation):
        return observation[self._key]
    
    def info(self, info, observation):
        return {
            **info,
            **{k: observation[k] for k in observation.keys() if k != self._key}
        }


def wrap_env(env, grayscale=False, frame_skip=4, frame_stack=2, img_size=(48, 48), brightness_scale=(0, 1), squeeze_dim=None):
    env = FrameSkipWrapper(env, skip=frame_skip)
    env = PixelObservationWrapper(env, pixels_only=False)
    env = SelectKeyWrapper(env, key='pixels')
    env = gym.wrappers.ResizeObservation(env, shape=img_size)
    if grayscale:
        env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, frame_stack)

    def rescale(img):
        if squeeze_dim is not None:
            img = np.squeeze(img, axis=squeeze_dim)
        return np.clip((np.array(img) / 255.0 - brightness_scale[0]) / (brightness_scale[1] - brightness_scale[0]), a_min=0, a_max=1)

    env = gym.wrappers.TransformObservation(env, f=rescale)

    return env


def create_env_partial(env_cls, **kwargs_default):
    def create_env(*args, **kwargs):
        return wrap_env(env_cls(*args, **kwargs), **kwargs_default, **kwargs)

    return create_env


gym.register('Hopper-v4-uniform', entry_point=create_env_partial(HopperEnvUniform, grayscale=False, frame_stack=3))
gym.register('Pendulum-v1-uniform', entry_point=create_env_partial(PendulumEnvUniform, grayscale=True, frame_stack=2, brightness_scale=(0.329, 1.0)))
gym.register('Cartpole-v0-uniform', entry_point=create_env_partial(CartPoleEnvUniform, grayscale=True, frame_stack=2))
gym.register('Planar-v0-uniform', entry_point=create_env_partial(PlanarEnvUniform, grayscale=False, frame_stack=2, squeeze_dim=-1))