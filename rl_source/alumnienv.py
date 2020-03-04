import os

import numpy as np
import pandas as pd

import gym
from gym import spaces
from gym.utils import seeding


class Env(gym.Env):


	def __init__(self, df,
				 obs_space_vars : list, 
				 action_space_bounds: list(list,list), 
				 *args,
				 spacelb: np.ndarray = np.array([]), spaceub: np.ndarray = np.array([]),
				 episodelength = 2016,
				 slicepoint = 0.75,
				 **kwargs):


		# the iterable dataframe
		self.df = df
		self.rows, self.cols = df.shape
		# collect dataframe statistics in a dataframe
		self.stats = self.df.describe().loc[['mean', 'std', 'min', 'max'],:]

		# the observation space variables
		self.obs_space_vars = obs_space_vars


		'''Begin: standard requirements for interfacing with gym environments'''

		# the action space bounds
		assert len(action_space_bounds)==2, "Two elements: low and high bounds are needed"
		assert all([actlow == -1*acthigh for actlow, acthigh in zip(action_space_bounds[0], action_space_bounds[1])]), "Action Space bounds have to be symmetric"
		self.action_space_bounds = action_space_bounds
		self.action_space = spaces.Box(low = np.array(self.action_space_bounds[0]),
										high = np.array(self.action_space_bounds[1]),
										dtype = np.float32)

		# the observation space (it is assumed to be continuous)
		if spacelb.size & spaceub.size:
			self.observation_space = spaces.Box(low = spacelb, high = spaceub, dtype =  np.float32)
		else:  # in case the spaces are not provided
			spacelb = np.flatten(self.stats.loc['min', self.obs_space_vars].to_numpy())
			spaceub = np.flatten(self.stats.loc['max', self.obs_space_vars].to_numpy())
			self.observation_space = spaces.Box(low = spacelb, high = spaceub, dtype =  np.float32)

		# other base environment requirements
		self.seed()
		self.viewer = None

		'''End: standard requirements for interfacing with gym environments'''


		'''Begin: Requirements for our own environment'''

		# episode_length: dictates number of steps in an episode
		self.episodelength = episodelength
		# counter: counts the current step number in an episode
		self.counter = 0
		# testing: whether we env is in testing phase or not
		self.testing = False
		# dataptr: Steps through the entire available data in a cycle. 
		# Gets reset to the train data start index when entire train data is used up. 
		# Gets reset to the test data start index when entire test data is used up.
		self.dataPtr = 0
		# Steps for specifying train and test data indices
		self.slicepoint = slicepoint
		self.traindatalimit = int(self.slicepoint * self.rows)
		self.testdatalimit = self.rows

		'''End: Requirements for our own environment'''



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
