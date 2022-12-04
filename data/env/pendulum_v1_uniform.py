import numpy as np
from typing import Optional

from gym.envs.classic_control.pendulum import PendulumEnv


class PendulumEnvUniform(PendulumEnv):
    def __init__(self, render_mode: Optional[str] = None, g=10.0):
        super().__init__(render_mode=render_mode, g=g)
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None):
        
        super().reset(seed=seed)    # From `PendulumEnv`

        low = [0, -8]
        high = [np.pi * 2, 8]

        self.state = np.random.uniform(low=low, high=high)
        self.last_u = None
        self.renderer.reset()
        self.renderer.render_step()
        
        if not return_info:
            return self._get_obs()
        else:
            return self._get_obs(), {}
