import os

import numpy as np
import pandas as pd

import gym
from gym import spaces
from gym.utils import seeding


class Env(gym.Env):


	def __init__(self, df,
				 obs_space_vars : list,
				 action_space_vars : list,
				 action_space_bounds: list, 
				 energy_model, model_input_shape, model_input_vars,
				 *args,
				#  spacelb: np.ndarray = np.array([]), spaceub: np.ndarray = np.array([]),
				 slicepoint = 0.75,
				 **kwargs):


		# the iterable dataframe
		self.df = df
		self.nrows, self.ncols = df.shape
		# collect dataframe statistics in a dataframe
		self.stats = self.df.describe().loc[['mean', 'std', 'min', 'max'],:]

		# make sure observation and action variables are part of the dataframe
		condition1 = all(name in self.df.columns for name in obs_space_vars)
		condition2 = all(name in self.df.columns for name in action_space_vars)
		assert condition1 & condition2 , "Both action and observation space variables must be in the input dataframe"
		# the observation space variable names list
		self.obs_space_vars = obs_space_vars
		# the action space variables names list
		self.action_space_vars = action_space_vars 


		'''Begin: standard requirements for interfacing with gym environments'''
		# the action space bounds
		assert len(action_space_bounds)==2, "Exactly two elements: low and high bounds are needed"
		assert all([actlow == -1*acthigh for actlow,
		 acthigh in zip(action_space_bounds[0], action_space_bounds[1])]), "Action Space bounds have to be symmetric"
		self.action_space_bounds = action_space_bounds
		self.action_space = spaces.Box(low = np.array(self.action_space_bounds[0]),
										high = np.array(self.action_space_bounds[1]),
										dtype = np.float32)

		# the observation space (it is assumed to be continuous)
		spacelb = self.stats.loc['min', self.obs_space_vars].to_numpy().flatten()
		spaceub = self.stats.loc['max', self.obs_space_vars].to_numpy().flatten()
		self.observation_space = spaces.Box(low = spacelb, high = spaceub, dtype =  np.float32)

		# other base environment requirements
		# self.seed()
		self.viewer = None

		# energy model
		self.model = energy_model
		self.model_input_shape = model_input_shape
		self.model_input_vars = model_input_vars
		'''End: standard requirements for interfacing with gym environments'''


		'''Begin: Requirements for our own environment'''
		# testing: whether we env is in testing phase or not
		self.testing = False
		# dataptr: Steps through the entire available data in a cycle. 
		# Gets reset to the train data start index when entire train data is used up. 
		# Gets reset to the test data start index when entire test data is used up.
		self.dataptr = 0
		# Steps for specifying train and test data indices
		self.slicepoint = slicepoint
		self.train_data_limit = int(self.slicepoint * self.nrows)
		self.test_data_limit = self.nrows
		# episode_length: dictates number of steps in an episode
		self.episode_length = self.train_data_limit
		'''End: Requirements for our own environment'''


	def seed(self,):
		"""sets the seed for the environment
		"""
		self.np_random, seed = seeding.np_random(seed)


	def reset(self,):

		# Helps iterate through the data file. This is not the actual observation
		self.row = self.df.iloc[[self.dataptr],:].copy()
		# Collect part of the observations for next time step
		self.s = self.row.loc[:, self.obs_space_vars]
		# step_counter: counts the current step number in an episode
		self.step_counter = 0
		'''Standard requirements for interfacing with gym environments'''
		self.steps_beyond_done = None

		return self.s.to_numpy().flatten()  # flatten them before feeding to the rl agent


	def step(self, a):

		# Advance datapointer to next row in the dataframe
		if not self.testing:
			self.dataptr = self.dataptr + 1 if self.dataptr < self.train_data_limit - 1 else 0
		else:
			self.dataptr = self.dataptr + 1 if self.dataptr < self.test_data_limit - 1 else self.train_data_limit

		# step to the next state and get new observation
		self.s_next, a = self.state_transition(self.s, a)

		# calculate reward and collect information for "info" dinctionary
		r, step_info = self.reward_calculation(self.s, a, self.s_next)
		
		# increment step counter
		self.step_counter += 1
		# check if episode has terminated
		done = False
		if self.step_counter > self.episode_length -1 :
			done = True  # episode has ended

		# proceed to the next state if not finished
		if not done:
			self.s = self.s_next

		return self.s_next.to_numpy().flatten(), float(r), done, step_info


	def state_transition(self, s, a):
		"""
		Custom method for state transition model
		"""

		# process control action
		a = self.process_action(a, s)

		# Helps iterate through the data file. This is not the actual observation
		self.row = self.df.iloc[[self.dataptr],:].copy()

		# Update historical action cells with actual agent action values
		self.row.loc[:,self.action_space_vars] = a

		# Collect observations for next time step and also return processed action
		return self.row.loc[:, self.obs_space_vars], a


	def process_action(self, a, s):
		"""Custom action processer
		"""
		# Action variable upper and lower bounds for the dataframe provided
		a_lowerbound = self.stats.loc['min', self.action_space_vars].to_numpy().flatten()
		a_upperbound = self.stats.loc['max', self.action_space_vars].to_numpy().flatten()

		#  Scale action differential from gym space to 0-1 scaled space
		a = a/(a_upperbound - a_lowerbound)

		# Supply Temperature at previous state
		previous_sat = s.loc[s.index[0], self.action_space_vars].to_numpy().flatten()

		# new action: previous state + new action differential
		a += previous_sat

		# clip action
		a = np.clip(a, a_lowerbound, a_upperbound)

		return a


	def reward_calculation(self, s, a, _, *args, **kwargs):
		"""Custom Reward Calculator
		"""

		# calculate energy for the next time step
		energy = 1*self.energy_cost(s, a)

		reward = -energy

		if self.testing:
			step_info = {'time': str(s.index[0]),
						'energy': energy,
						'oat': s.loc[s.index[0], 'oat'],
						'orh': s.loc[s.index[0], 'orh'],
						'rl_sat': s.loc[s.index[0], 'sat'],
						'avg_stpt': s.loc[s.index[0], 'avg_stpt']
						}
		else:
			step_info = {}

		return reward, step_info


	def energy_cost(self, s, a):
		"""Custom energy calculator
		"""

		# Since for our case, the action modifies the state sirectly, we replace action columns with corresponding rl action
		assert all(name in s.columns for name in self.action_space_vars), "Action space vars are not part of observation space vars"
		
		in_obs = s.copy()
		# change old actions to new actions
		in_obs.loc[:, self.action_space_vars] = a
		in_obs = in_obs.loc[:, self.model_input_vars]
		# convert to array and reshape
		in_obs = in_obs.to_numpy().reshape(self.model_input_shape)

		return float(self.model.predict(in_obs, batch_size=1).flatten())

	def testenv(self):
		self.testing = True
		self.dataptr = self.train_data_limit
