import os
import math
import sys
from collections import defaultdict

import gym
from gym import error, spaces, utils

from gym.utils import seeding

import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
    
class BaselEnv(gym.Env):

    def __init__(self):
        super(BaselEnv, self).__init__()
        print("Env initialized")
        # state space
        self.action_space = spaces.Discrete(3000)
        #self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([250, 11, 7]), dtype=int)
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([11, 250]), dtype=np.float32)
        self.viewer = None
        self.steps_beyond_done = None
        self.seed()
        
        self.observation = self.reset()
        
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        reward = 0
        done = False
        return self._get_obs(), reward, done, self._get_obs()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        return self._get_obs()
    
    def _get_obs(self):
        return np.array([1.0, 2.0])


from stable_baselines.common.vec_env import SubprocVecEnv
from multiprocessing import freeze_support
def make_env(seed):
    def _init():
        env = BaselEnv()
        env.seed(seed)
        return env
    return _init

if __name__ ==  "__main__":
    
    n_envs = 2

    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    obs = env.reset()

    for i in range(10):
        print(i)
        action = [env.action_space.sample() for _ in range(n_envs)]
        obs, reward, done, info = env.step(action)
    
    env.close()

    print("Done!")