# Data Sampling Structure and Statistics

## ENV Design Choices
### Planar
- Action is considered valid even if the resulting next state is pushed out of boundary. This allows the agent to learn how to stay within the boundary.
- Reward is as follows:
    - Collision: -40
    - Out-of-bounds: -40
    - Else (regardless of reached or not): -dist
- Done conditions are as follow:
    - True: collision, out-of-bounds, reached
    - False: else
- If the state is out-of-bounds, the state is simply not rendered. The agent should learn that this state is undesirable from the reward.

### Hopper
- `healthy_angle_range` = [-0.2, 0.2]
- `healthy_state_range` = [-100.0, 100.0]

If given full range, Hopper dies basically after each reset. Then it's impossible to do frame-skipping. Thus, I tuned the reset state to be like the following:
(X-position shouldn't really matter since the objective is to have the Hopper standing.)

```python
qpos = np.array([
    # np.random.uniform(1, 5), # x-pos
    0,
    np.random.uniform(0.8, 1.5), # 0. [healthy_z_range] : rootz (pos: m)
    np.random.uniform(self._healthy_angle_range[0]/2, self._healthy_angle_range[1]/2), # 1. [healthy_angle_range] : rooty (ang: rad)
    np.random.uniform(self._healthy_state_range[0]/33, self._healthy_state_range[1]/33), # 2. [healthy_state_range] : thigh_joint (ang: rad)
    np.random.uniform(self._healthy_state_range[0]/33, self._healthy_state_range[1]/33), # 3. [healthy_state_range] : leg_joint (ang: rad)
    np.random.uniform(self._healthy_state_range[0]/33, self._healthy_state_range[1]/33), # 4. [healthy_state_range] : foot_joint (ang: rad)
])
qvel = np.array([
    np.random.uniform(self._healthy_state_range[0]/33, self._healthy_state_range[1]/33), # 5. [healthy_state_range] : rootxz (vel: m/s)
    np.random.uniform(self._healthy_state_range[0]/33, self._healthy_state_range[1]/33), # 6. [healthy_state_range] : rootz (vel: m/s)
    np.random.uniform(self._healthy_state_range[0]/33, self._healthy_state_range[1]/33), # 7. [healthy_state_range] : rooty (ang vel: rad/s)
    np.random.uniform(self._healthy_state_range[0]/33, self._healthy_state_range[1]/33), # 8. [healthy_state_range] : thigh_joint (ang vel: rad/s)
    np.random.uniform(self._healthy_state_range[0]/33, self._healthy_state_range[1]/33), # 9. [healthy_state_range] : leg_joint (ang vel: rad/s)
    np.random.uniform(self._healthy_state_range[0]/33, self._healthy_state_range[1]/33), #10. [healthy_state_range] :  foot_joint (ang vel: rad/s)
])
```

Random action is also sampled from a smaller state since too big of an action may instantly kill the Hopper agent.
```python
def get_valid_random_action(self):
    return self.action_space.sample()/3
```


## Sampling Structure
- Reset is done every `reset_steps`, which is a parameter passed into the sampler. Recommended is at least 200. If you reset too often between steps (smaller `reset_steps`), then it may be difficult to stack them.
- If the done condition is met while frame-skipping is performed, then that sample is discarded. I.e., the sample is saved only if frame-skipping is conducted without done condition is met. Env is reset in such situation.
- If the entirety of frame-skipping is performed, then that sample is saved. If done is met after frame-skipping completes, then env is reset. 
- `serial` option saves the rendered images as np.ndarray in dataframe. The saved np.ndarray is that of after `downsampling` and `rgb2gray` is performed. **This is the recommended method of sampling since it does not save image files, which can be quite heavy.**

- Recommended `step_size` for all the environments is 4.

## Statistics of Sampling:
- `#DONES`: number of dones that are logged as a sample. Steps that resulted in `done=True` when frame-skipping is not complete does not count to this.
- `#RESETS`: number of resets when `done=True` or `sample_num%reset_steps==0`. Here, even when frame-skipping is not complete, `#resets` is incremented. This metric is quite useless.
- `#DONES + SAMPLE_SIZE/RESET_STEPS` should be a good metric in terms of how much state-space is covered.

### Planar sampling statistics
```console
python data/sample_data.py --env planar --sample_size 10000 --obs_type serial --obs_res 40 --step_size 4 --reset_steps 100
```
- Output: # DONES:  996, # RESETS:  3932
- -> 1096 meaningful resets

### Pendulum sampling statistics
```console
python data/sample_data.py --env pendulum --sample_size 10000 --obs_type serial --obs_res 45 --step_size 4 --reset_steps 100
```
- Output: # DONES:  200, # RESETS:  300
- -> 300 meaningful resets

### CartPole sampling statistics
```console
python data/sample_data.py --env cartpole --sample_size 10000 --obs_type serial --obs_res 64 --step_size 4 --reset_steps 100
```
- Output: # DONES:  819, # RESETS:  3262
- -> 919 meaningful resets

### Hopper sampling statistics
```console
python data/sample_data.py --env hopper --sample_size 10000 --obs_type serial --obs_res 64 --step_size 4 --reset_steps 100
```
- Output: # DONES:  1132, # RESETS:  8253
- -> 1232 meaningful resets
- Time: 28 min 24 sec
