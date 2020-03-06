# including the project directory to the notebook level
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

	import tensorflow as tf
	config = tf.ConfigProto(log_device_placement=True)
	config.gpu_options.allow_growth = True
	#config = tf.ConfigProto(device_count={'CPU' : 4, 'GPU' : 0}, allow_soft_placement=False, log_device_placement=True)
	session = tf.Session(config=config)
	
	from keras.models import load_model
	from nn_source import models as mp
	from rl_source import alumnienv
	from stable_baselines.common.vec_env import SubprocVecEnv
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
	energymodel = load_model('../results/lstmtrain/LSTM_model_44_0.01')  # Specify the path where energy is stored
	
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
	
	n_envs = 4  # can also be os.cpu_count()
	
	seed = 123  # the initial seed for the random number generator
	
	start_index = 0  # start rank index
	
	monitor_dir = '../log/'  # Path to a folder where the monitor files will be saved
	
	vec_env_cls = SubprocVecEnv  #  A custom `VecEnv` class constructor. Default: DummyVecEnv
	
	env_kwargs = dict(  #  Optional keyword argument to pass to the env constructor
		df=dfscaled,  # the file for iterating
		obs_space_vars=['oat', 'orh', 'ghi', 'sat', 'avg_stpt'],  # state space variable
		action_space_vars=['sat'],  # action space variable
		action_space_bounds=[[-2.0], [2.0]],  # bounds for real world action space; is scaled internally using the params
		energy_model=energymodel,  # trained lstm model
		model_input_shape=(1, 1, 4),  # lstm model input data shape (no_samples, output_timestep, inputdim)
		model_input_vars=['oat', 'orh', 'ghi', 'sat'],  # lstm model input variables
		**params  # the reward adjustment parameters
	)
	
	envmodel = make_vec_env(env_id = env_id,
					  n_envs = n_envs,
					  seed = seed,
					  start_index = start_index,
					  # monitor_dir = monitor_dir,
					  vec_env_cls = vec_env_cls,
					  env_kwargs = env_kwargs)
	
	# test the environment
	
	# make sure environment is in train mode
	envmodel.env_method('trainenv')
	
	# resest environment
	out1 = envmodel.env_method('reset')
	
	# Step through the environment eg apply a change of 0.43*F in the up direction
	out2a,out2b,out2c,out2d  = envmodel.env_method( 'step', (np.array([0.43])) )
	
	# make sure environment is in test mode
	envmodel.env_method('testenv')
	
	# resest environment
	out3 = envmodel.env_method('reset')
	
	# Step through the environment eg apply a change of 0.43*F in the down direction
	out4a,out4b,out4c,out4d = envmodel.env_method( 'step', (np.array([0.43])) )
	
	for out in [out1,out2a,out2b,out2c,out2d,out3,out4a,out4b,out4c,out4d]:
		print(out)
		
	envmodel.close()

# import warnings
# with warnings.catch_warnings():  
# 	warnings.filterwarnings("ignore",category=FutureWarning)
# 	import tensorflow as tf

# a = tf.constant([x for x in range(0, 3)], dtype=tf.float32, shape=[2, 3], name='a')
# b = tf.constant([x for x in range(3, 6)], dtype=tf.float32, shape=[3, 2], name='b')
# c = tf.matmul(a, b)

# session_conf = tf.ConfigProto(
# 	device_count={'CPU' : 1, 'GPU' : 0},
# 	allow_soft_placement=False,
# 	log_device_placement=True
# )

# with tf.Session(config=session_conf) as sess:
# 	print(sess.run(c))