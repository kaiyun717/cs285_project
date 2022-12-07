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

    def get_valid_random_action(self):
        return self.action_space.sample()