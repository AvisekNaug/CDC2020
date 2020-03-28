import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
	sys.path.append(module_path)
import warnings
from multiprocessing import freeze_support

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame, concat

with warnings.catch_warnings():

	warnings.filterwarnings("ignore",category=FutureWarning)

	# needed to prevent OOM error
	import tensorflow as tf
	config = tf.ConfigProto(log_device_placement=True)
	config.gpu_options.allow_growth = True  # pylint: disable=no-member
	session = tf.Session(config=config)

	from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
	from stable_baselines.common import set_global_seeds, make_vec_env
	
	from keras.models import load_model
	from dataprocess import dataprocessor as dp
	from nn_source import models as mp
	from rl_source import alumnienv

# return a new column which is the sum of previous window_size values
def windowsum(df, window_size: int, column_name: str):
	return df[[column_name]].rolling(window=window_size, min_periods=window_size).sum()

def quickmerge(listdf):
    return concat(listdf)

def prepare_dflist():

	period = 6  # the period to sample the data at. 1 period= 5 minutes

	dfdata = dp.readfile('../data/processed/buildingdata.pkl')  # read the pickled file for ahu data
	df = dfdata.return_df(processmethods=['file2df'])  # return pickled df

	df['30min_hwe'] = windowsum(df,window_size=period, column_name='hwe')  # Sum half hour energy data
	df['30min_cwe'] = windowsum(df,window_size=period, column_name='cwe')  # Sum half hour energy data
	df = dp.dropNaNrows(df)  # remove NaN rows created as a result
	
	order = 5  # order of the filter
	T = 300  # sampling_period in seconds
	cutoff = 0.0001  # desired cutoff frequency of the filter, Hz
	df_smoothed = dp.dfsmoothing(df=df, column_names=list(df.columns),
								order=order, Wn=cutoff, T=T)  # Smoothing the data
	df_smoothed = dp.sample_timeseries_df(df_smoothed, period=period)  # Sample the data at half hour intervals

	# Extract 0-1 Scaler for the entire dataframe before sending parts of it to the environment
	scaler = MinMaxScaler(feature_range=(0,1))
	dfscaler = scaler.fit(df_smoothed)
	#dfscaled = DataFrame(dfscaler.fit_transform(df_smoothed), index=df.index, columns=df.columns)

	# Creating a list of 7 day dataframes for training
	dflist = dp.df2dflist_alt(df_smoothed, subsequence=True, period=period, days=7, hours=0)

	return df, dflist, dfscaler

def dflist2overlap_dflist(dflist, data_weeks):
	"""Creates multiple overlapping dataframes of data_weeks length
	with an overlap of week_overlap
	
	Arguments:
		dflist {[list]} -- input list of pandas dataframes
		data_weeks {[int]} -- length of the dataframe in terms of number of weeks
	"""
	out_dflist = []  # create list of training, testing arrays

	datablock = dflist[:data_weeks]
	out_dflist.append(quickmerge(datablock))  # Initial Data Block for offline training

	for weekdata in dflist[data_weeks:]:
	
		datablock = datablock[1:]+[weekdata]  # remove 1st of data from initial_datablock
		out_dflist.append(quickmerge(datablock))

	return out_dflist

def overlap_dflist2array(out_dflist, input_vars, outut_vars,lag,splitvalue, X_scale=None, y_scale=None):

	weeklist = []  # create list of training, testing arrays

	for df in out_dflist:
		X_train, X_test, y_train, y_test, _, _ = dp.df2arrays(df,
		predictorcols=input_vars, outputcols=outut_vars, lag=lag, split=splitvalue, reshaping=True,
		scaling=True, feature_range=(0,1), X_scale= X_scale, y_scale=y_scale)

	weeklist.append({
        'Id':'Year-{}-Week-{}'.format(str(df.index[0].year), 
                                      str(df.index[0].week)),
        'X_train':X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    })

	return weeklist
	


def main():

	"""" Begin : Creating all the data requirements"""
	totaldf, dflist, dfscaler = prepare_dflist()  # break df to weekly dflist and return scaler for entire data
	data_weeks=52
	out_dflist = dflist2overlap_dflist(dflist, data_weeks=data_weeks)  # Create dflist with weekly overlaps

	cwe_input_vars=['oat', 'orh', 'sat', 'ghi', 'flow'],  # cwe lstm model input variables
	cwe_outut_vars = ['30min_cwe']  # cwe lstm model input variables
	hwe_input_vars=['oat','orh', 'sat', 'ghi', 'hw_sf', 'hw_st'],  # hwe lstm model input variables
	hwe_outut_vars = ['30min_hwe']  # hwe lstm model input variables

	_, _, _, _, cwe_X_scaler, cwe_y_scaler = dp.df2arrays(totaldf, predictorcols=cwe_input_vars, 
	outputcols=cwe_outut_vars, feature_range=(0,1), reshaping=False)  # extract cwe scaler for entire data

	_, _, _, _, hwe_X_scaler, hwe_y_scaler = dp.df2arrays(totaldf, predictorcols=hwe_input_vars,
	outputcols=hwe_outut_vars, feature_range=(0,1), reshaping=False)  # extract hwe scaler for entire data

	cwe_week_list = overlap_dflist2array(out_dflist, cwe_input_vars, cwe_outut_vars, lag=-1, 
										splitvalue=(data_weeks-1)/data_weeks, X_scale=cwe_X_scaler, y_scale=cwe_y_scaler)

	hwe_week_list = overlap_dflist2array(out_dflist, hwe_input_vars, hwe_outut_vars, lag=-1, 
										splitvalue=(data_weeks-1)/data_weeks, X_scale=hwe_X_scaler, y_scale=hwe_y_scaler)

	erromsg = "Unequal lists: len(out_dflist)={0:}, len(out_dflist)={1:}, len(hwe_week_list)={2:}".format(
		len(out_dflist), len(out_dflist), len(hwe_week_list))
	assert all([len(out_dflist)==len(cwe_week_list), len(out_dflist)==len(hwe_week_list)]), erromsg
	"""" End : Creating all the data requirements"""



	"""Begin : Prerequisites for the environment"""

	# important parameters to scale the reward
	params = {

		'energy_saved': 1.0, 'energy_savings_thresh': 0.0, 'energy_penalty': -1.0, 'energy_reward_weight': 0.5,
		'comfort': 1.0, 'comfort_thresh': 0.10, 'uncomfortable': -1.0, 'comfort_reward_weight': 0.5,
		'action_minmax': [np.array([totaldf.sat.min()]), np.array([totaldf.sat.max()])]  # required for clipping

	}

	env_id = alumnienv.Env  # the environment ID or the environment class
	n_envs = 2  # can also be os.cpu_count()
	"""always make sure that the number of environments is even"""
	seed = 123  # the initial seed for the random number generator
	start_index = 0  # start rank index
	vec_env_cls = SubprocVecEnv  #  A custom `VecEnv` class constructor. Default: DummyVecEnv
	"""End : Prerequisites for the environment"""
	
	# main iteration loop
	for out_df, cwe_week, hwe_week in zip(out_dflist, cwe_week_list,hwe_week_list):
		
		"""train lstm model on cwe"""

		"""train lstm model on hwe"""

		"""train rl model"""
		# create environment with new data
		monitor_dir = '../log/Trial_{}/Interval_{}/'.format(0,1)  # Path to a folder where the monitor files will be saved

		# create agent with new data or reuse old agent if in the loop for the first time

		# train the agent

		"""test rl model"""
		# load the agent file from previous best
	
	# Arguments to be fed to the custom environment inside make_vec_env
	env_kwargs = dict(  #  Optional keyword argument to pass to the env constructor

		df=dfscaler.transform(out_df),  # the file for iterating; it is scaled before being passed
		obs_space_vars=['oat', 'orh', 'ghi', 'sat', 'avg_stpt', 'flow'],  # state space variable
		action_space_vars=['sat'],  # action space variable
		action_space_bounds=[[-2.0], [2.0]],  # bounds for real world action space; is scaled internally using the params

		cwe_energy_model=cwe_model,  # trained lstm model
		cwe_input_vars=cwe_input_vars,  # lstm model input variables
		cwe_input_shape=(1, 1, len(cwe_input_vars)),  # lstm model input data shape (no_samples, output_timestep, inputdim)

		hwe_energy_model=hwe_model,  # trained lstm model
		hwe_input_vars=hwe_input_vars,  # lstm model input variables
		hwe_input_shape=(1, 1, len(hwe_input_vars)),  # lstm model input data shape (no_samples, output_timestep, inputdim)

		**params  # the reward adjustment parameters
	)
	#vec_env_kwargs = dict(start_method="fork")

	envmodel = make_vec_env(env_id = env_id,
					n_envs = n_envs,
					seed = seed,
					start_index = start_index,
					monitor_dir = monitor_dir,
					vec_env_cls = vec_env_cls,
					env_kwargs = env_kwargs)#,
					#vec_env_kwargs=vec_env_kwargs)

# wrap the environment in the vectorzied wrapper with SubprocVecEnv
# run the environment inside if __name__ == '__main__':

if __name__ == '__main__':
	main()
	print("Done!")