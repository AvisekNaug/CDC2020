import os

import numpy as np
import pandas as pd

import gym
from gym import spaces
from gym.utils import seeding


class Env(gym.Env):

	def __init__(self, df,  *args, **kwargs):

		pass

	def seed(self,):
		"""sets the seed for the environment
		"""
		self.np_random, seed = seeding.np_random(seed)

	def reset(self,):
				
		raise NotImplementedError

	def step(self, action):
		
		raise NotImplementedError

	def rewardscaler(self, components: list, scalers : list):
		
		raise NotImplementedError

	def reward_calculation(self, s, a, s_next, **kwargs):

		raise NotImplementedError
