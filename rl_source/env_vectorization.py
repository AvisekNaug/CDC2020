"""
The scipt shows how to wrap custom gym environment with SubprocVecEnv for PPO multiprocessing
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
	sys.path.append(module_path)
import warnings

from dataprocess import dataprocessor as dp
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame

with warnings.catch_warnings():  

	warnings.filterwarnings("ignore",category=FutureWarning)

	# needed to prevent OOM error
	import tensorflow as tf
	config = tf.ConfigProto(log_device_placement=True)
	config.gpu_options.allow_growth = True  # pylint: disable=no-member
	session = tf.Session(config=config)
	
	from keras.models import load_model
	from nn_source import models as mp
	from rl_source import alumnienv
	from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
	from stable_baselines.common import set_global_seeds, make_vec_env

# wrap the environment in the vectorzied wrapper with SubprocVecEnv
# run the environment inside if __name__ == '__main__':

	if __name__ == '__main__':
	
		# read the pickled file for ahu data
		dfdata = dp.readfile('../data/processed/buildingdata.pkl')

		# return pickled df
		df = dfdata.return_df(processmethods=['file2df'])

		# 0-1 Scale the dataframe before sending it to the environment
		scaler = MinMaxScaler(feature_range=(0,1))
		dfscaler = scaler.fit(df)

		dfscaled = DataFrame(dfscaler.fit_transform(df), index=df.index, columns=df.columns)

		# load energy model
		energymodel = load_model('../results/lstm/LSTM_model_360_0.01')  # Specify the path where energy is stored

		# important parameters to scale the reward
		params = {
			'energy_saved': 1.0,
			'energy_savings_thresh': 0.0,
			'energy_penalty': -1.0,
			'energy_reward_weight': 1.0,
			'comfort': 1.0,
			'comfort_thresh': 0.10,
			'uncomfortable': -1.0,
			'comfort_reward_weight': 1.0,
			'action_minmax': [np.array([df.sat.min()]), np.array([df.sat.max()])]
		}

		# Arguments to be fed to the make_vec_env

		env_id = alumnienv.Env  # the environment ID or the environment class

		n_envs = 2  # can also be os.cpu_count()

		seed = 123  # the initial seed for the random number generator

		start_index = 0  # start rank index

		monitor_dir = '../log/'  # Path to a folder where the monitor files will be saved

		vec_env_cls = SubprocVecEnv  #  A custom `VecEnv` class constructor. Default: DummyVecEnv

		env_kwargs = dict(  #  Optional keyword argument to pass to the env constructor
			df=dfscaled,  # the file for iterating
			obs_space_vars=['oat', 'orh', 'ghi', 'sat', 'avg_stpt', 'flow'],  # state space variable
			action_space_vars=['sat'],  # action space variable
			action_space_bounds=[[-2.0], [2.0]],  # bounds for real world action space; is scaled internally using the params
			energy_model=energymodel,  # trained lstm model
			model_input_shape=(1, 1, 5),  # lstm model input data shape (no_samples, output_timestep, inputdim)
			model_input_vars=['oat', 'orh', 'sat', 'ghi', 'flow'],  # lstm model input variables
			**params  # the reward adjustment parameters
		)

		envmodel = make_vec_env(env_id = env_id,
						n_envs = n_envs,
						seed = seed,
						start_index = start_index,
						# monitor_dir = monitor_dir,
						vec_env_cls = vec_env_cls,
						env_kwargs = env_kwargs)

		# try the environment



		# make sure environment is in train mode
		print("setting train env mode..... \n")
		envmodel.env_method('trainenv')

		# resest environment
		print("Reset env in train mode... \n")
		out = envmodel.reset()  # env_method('reset')
		print('reset in train mode: {} \n'.format(out))

		# Step through the environment eg apply a change of 0.43*F in the up direction
		print("executing train step..... \n")
		obs, rewards, dones, info = envmodel.step(np.array([0.43]*n_envs))
		#out2a,out2b = envmodel.env_method( 'step', (np.array([0.43])) )
		print("In train mode obs:{}, \n rewards:{}, \n dones:{}, \n info:{} \n".format(obs, rewards, dones, info))



		# make sure environment is in test mode
		print("setting test env mode..... \n")
		envmodel.env_method('testenv')

		# resest environment
		print("Reset env in test mode.... \n")
		out = envmodel.reset()  # env_method('reset')
		print('reset in test mode: {} \n'.format(out))

		# Step through the environment eg apply a change of 0.43*F in the down direction
		print("executing test step..... \n")
		#out4a,out4b = envmodel.env_method( 'step', (np.array([0.43])) )  # step(np.array([0.43]))
		obs, rewards, dones, info = envmodel.step(np.array([0.43]*n_envs))
		print("In test mode obs:{}, \n rewards:{}, \n dones:{}, \n info:{} \n".format(obs, rewards, dones, info))
			
		envmodel.close()
