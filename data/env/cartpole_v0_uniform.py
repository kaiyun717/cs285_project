import numpy as np
from typing import Optional

from gym.envs.classic_control.cartpole import CartPoleEnv


class CartPoleEnvUniform(CartPoleEnv):
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__(render_mode)
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None):
        
        super().reset(seed=seed)
        
        x_thres = self.x_threshold - 0.4    # Safe range: (-2.4, +2.4). This range: (-2.0, +2.0)
        x_dot_thres = 1
        theta_thres = self.theta_threshold_radians * 4/5
        theta_dot_thres = 1
        
        high = np.array([x_thres, x_dot_thres, theta_thres, theta_dot_thres])
        low = -high

        self.state = np.random.uniform(low=low, high=high)
        self.steps_beyond_terminated = None
        self.renderer.reset()
        self.renderer.render_step()
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}


if __name__ == '__main__':
    import gym
    import numpy as np
    
    # env = gym.make('CartPole-v0')
    # env.reset()
    # count = np.zeros((4,))
    # dones = 0
    # for _ in range(10000):
    #     ac = env.action_space.sample()
    #     s, r, d, i = env.step(ac)

    #     if s[0] > 2.3 or s[0] < -2.3:
    #         count[0] += 1
    #     if s[1] > 10 or s[1] < -10:
    #         count[1] += 1
    #     if d:
    #         # print('DONE')
    #         dones += 1
    #         env.reset()

    # print(count)
    # print(dones)

    h = np.array([3,4,5,6])
    l = -h
    for _ in range(10):
        print(np.random.uniform(l, h))