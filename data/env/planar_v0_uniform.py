import numpy as np
import os
from PIL import Image, ImageDraw

import gym
from gym import spaces

class PlanarEnvUniform(gym.Env):
    def __init__(self):
        np.random.seed(np.random.randint(1,100))
        
        self.width, self.height = 40, 40
        self.obstacles_center = np.array([
            [20.5, 5.5], 
            [20.5, 12.5], 
            [20.5, 27.5], 
            [20.5, 35.5], 
            [10.5, 20.5], 
            [30.5, 20.5]
        ])

        self.r_overlap = 0.5 # agent cannot be in any rectangular area with 
                             # obstacles as centers and half-width = 0.5
        self.r = 1           # radius of the obstacles when rendered
        self.rw = 3          # robot half-width
        self.rw_rendered = 2 # robot half-width when rendered
        self.max_step_len = 3
        
        num_obs = (self.obstacles_center.shape[0] + 2) * self.obstacles_center.shape[1]
        obs_low = np.zeros((num_obs,))
        obs_high = np.repeat(self.height, num_obs)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        
        act_low = np.repeat(-self.max_step_len, 2)
        act_high = np.repeat(self.max_step_len, 2)
        self.action_space = spaces.Box(act_low, act_high)

        self.goal_state = np.zeros((2,))

        self.state = np.zeros((2,))

        env_arr = np.zeros((self.width, self.height))
        env_img = Image.fromarray(env_arr)
        env_draw = ImageDraw.Draw(env_img)
        for y, x in self.obstacles_center:
            env_draw.ellipse((int(x)-int(self.r), 
                              int(y)-int(self.r), 
                              int(x)+int(self.r), 
                              int(y)+int(self.r)), fill=128)
        env_img = env_img.convert('L')
        self.env_arr = np.array(env_img) / 255.0

    def reset(self):
        reset_state = self._reset_state()
        while self._is_colliding(reset_state):
            reset_state = self._reset_state()
        self.state = reset_state        

        goal_state = np.array([25, 15]) # self._reset_state()
        # while self._is_colliding(goal_state, delta=self.r_overlap*5):
            # goal_state = self._reset_state()
        self.goal_state = goal_state

        aug_state = self._augument_state(np.vstack((self.goal_state, self.state)))
        return np.array(aug_state, dtype=np.float32)
        
    def _reset_state(self):
        x = np.random.uniform(low=self.rw, high=self.height-self.rw)
        y = np.random.uniform(low=self.rw, high=self.width-self.rw)
        state = np.array([x, y])
        return state

    def _augument_state(self, state):
        aug_state = np.vstack((self.obstacles_center, state))  # Return [obstacles, state]
        aug_state = np.reshape(aug_state, newshape=(aug_state.shape[0]*aug_state.shape[1],))
        return aug_state.astype(np.float32)

    def step(self, action):
        next_state = self.state + action
        self.state = next_state
        collision = self._is_colliding(self.state)
        reached = self._reached(self.state)

        if reached:
            reward = -self._distance_to_goal(self.state)
            done = True
        elif collision:
            reward = -40    # NOTE: design choice
            done = True
        else:
            reward = -self._distance_to_goal(self.state)
            done = False

        aug_next_state = self._augument_state(np.vstack((self.goal_state, self.state)))
        return aug_next_state, reward, done, {}

    def _distance_to_goal(self, state):
        return np.sqrt(np.sum((self.goal_state - state)**2))

    def render(self, mode="human"):
        # Obstacles and entire map
        env_array = np.copy(self.env_arr)
        
        # Current state
        # top, bottom, left, right = self._get_pixel_location(self.state)
        state_x = int(round(self.state[0]))
        state_y = int(round(self.state[1]))
        state_in_bound = (state_x in range(self.height)) and (state_y in range(self.height))
        if state_in_bound:
            env_array[max(state_x-1, 0):min(self.height-1, state_x+2), max(state_y-1, 0):min(self.height-1, state_y+2)] = 1

        # Goal state
        goal_x, goal_y = int(round(self.goal_state[0])), int(round(self.goal_state[1]))
        env_array[goal_x, goal_y] = 0.75
        env_array[goal_x+1, goal_y+1] = 0.75
        env_array[goal_x-1, goal_y-1] = 0.75
        env_array[goal_x+1, goal_y-1] = 0.75
        env_array[goal_x-1, goal_y+1] = 0.75

        return env_array

    def is_valid_action(self, state, action, epsilon=0.1):
        return self._is_valid_action(state, action, epsilon=epsilon)
    
    def get_valid_random_action(self, epsilon=0.1):
        action = self.action_space.sample()
        while True:
            if self._is_valid_action(self.state, action, epsilon):
                break
            else:
                action = self.action_space.sample()
        return action

    def _check_done(self, state):
        collision = self._is_colliding(state)
        reached = self._reached(state)
        return not collision or reached

    def _is_colliding(self, state, delta=None):
        # Out-of-bounds
        if np.any([state - self.rw < 0, state + self.rw > self.height]):
            return True
        
        if delta is None:
            delta = self.r_overlap

        x, y = state[0], state[1]
        for obst in self.obstacles_center:
            if (np.abs(obst[0] - x) <= delta) \
                and (np.abs(obst[1] - y) <= delta):
                return True
        
        return False

    def _reached(self, state):
        x, y = state[0], state[1]
        if (np.abs(self.goal_state[0] - x) <= self.r_overlap) \
            and (np.abs(self.goal_state[1] - y) <= self.r_overlap):
            return True
        return False
        
    def _is_valid_action(self, state, action, epsilon=0.1):
        # if the difference between the action and the actual distance 
        # between x and x_next are in range(0,epsilon)
        next_state = state + action
        top, bottom, left, right = self._get_pixel_location(state)
        top_next, bottom_next, left_next, right_next = self._get_pixel_location(next_state)
        x_diff = np.array([top_next - top, left_next - left], dtype=np.float32)
        
        ########## Valid action may still lead to out-of-bounds ##########
        # next_state_x = int(round(next_state[0]))
        # next_state_y = int(round(next_state[1]))
        # next_state_in_bounds = (next_state_x in range(self.height)) and (next_state_y in range(self.height))

        # return (not np.sqrt(np.sum((x_diff - action)**2)) > epsilon) and next_state_in_bounds
        ##################################################################
        
        return (not np.sqrt(np.sum((x_diff - action)**2)) > epsilon) 

    def _get_pixel_location(self, state):
        center_x, center_y = int(round(state[0])), int(round(state[1]))
        top = center_x - self.rw_rendered
        bottom = center_x + self.rw_rendered
        left = center_y - self.rw_rendered
        right = center_y + self.rw_rendered
        return top, bottom, left, right
