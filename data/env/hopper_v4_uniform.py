import numpy as np

from gym.envs.mujoco.hopper_v4 import HopperEnv


class HopperEnvUniform(HopperEnv):
    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-3,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_state_range=(-100.0, 100.0),
        healthy_z_range=(0.7, float("inf")),
        healthy_angle_range=(-0.2, 0.2),
        reset_noise_scale=5e-3,
        exclude_current_positions_from_observation=True,
        **kwargs
    ):
        super().__init__(forward_reward_weight, ctrl_cost_weight, healthy_reward, terminate_when_unhealthy,
                        healthy_state_range, healthy_z_range, healthy_angle_range, reset_noise_scale,
                        exclude_current_positions_from_observation, **kwargs)

    def reset_model(self):
        qpos = np.array([
            # np.random.uniform(1, 5), # x-pos
            0,
            np.random.uniform(0.8, 1.5), # 0. [healthy_z_range] : rootz (pos)
            np.random.uniform(self._healthy_angle_range[0]/3, self._healthy_angle_range[1]/3), # 1. [healthy_angle_range] : rooty (ang)
            np.random.uniform(self._healthy_state_range[0]/3, self._healthy_state_range[1]/3), # 2. [healthy_state_range] : thigh_joint (ang)
            np.random.uniform(self._healthy_state_range[0]/3, self._healthy_state_range[1]/3), # 3. [healthy_state_range] : leg_joint (ang)
            np.random.uniform(self._healthy_state_range[0]/3, self._healthy_state_range[1]/3), # 4. [healthy_state_range] : foot_joint (ang)
        ])
        qvel = np.array([
            np.random.uniform(self._healthy_state_range[0]/3, self._healthy_state_range[1]/3), # 5. [healthy_state_range] : rootxz (vel)
            np.random.uniform(self._healthy_state_range[0]/3, self._healthy_state_range[1]/3), # 6. [healthy_state_range] : rootz (vel)
            np.random.uniform(self._healthy_state_range[0]/3, self._healthy_state_range[1]/3), # 7. [healthy_state_range] : rooty (ang vel)
            np.random.uniform(self._healthy_state_range[0]/3, self._healthy_state_range[1]/3), # 8. [healthy_state_range] : thigh_joint (ang vel)
            np.random.uniform(self._healthy_state_range[0]/3, self._healthy_state_range[1]/3), # 9. [healthy_state_range] : leg_joint (ang vel)
            np.random.uniform(self._healthy_state_range[0]/3, self._healthy_state_range[1]/3), #10. [healthy_state_range] :  foot_joint (ang vel)
        ])
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation