seed = 0  # the initial seed for the random number generator

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed)
import shutil
# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed)
# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed)

# Enable '0' or disable '' GPU use
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# including the project directory to the notebook level
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
	sys.path.append(module_path)

import warnings
from multiprocessing import freeze_support

from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame, concat

with warnings.catch_warnings():

	warnings.filterwarnings("ignore",category=FutureWarning)

	import tensorflow as tf
	# 4. Set the `tensorflow` pseudo-random generator at a fixed value
	tf.compat.v1.set_random_seed(seed)
	# Following Three lines needed to prevent OOM error
	config = tf.ConfigProto(log_device_placement=True)
	config.gpu_options.allow_growth = True  # pylint: disable=no-member
	session = tf.Session(config=config)
	from keras import backend as K

	from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
	from stable_baselines.common import set_global_seeds, make_vec_env
	from keras.models import load_model
	from keras.utils import to_categorical

	from nn_source import models as mp
	from rl_source import alumni_env, ppo_controller
	from rl_source import continual_learning_make_env as cme

# Would want to see the warnings
from dataprocess import dataprocessor as dp
from dataprocess import plotutils as pu
from dataprocess import logutils as lu
import alumni_env_utils as utils

import json

# create new directory if it does not exist; eles clear existing files in it
def make_dir(dir_path):
	# clear old files
	try:
		os.makedirs(dir_path)
	except IsADirectoryError:
		files = os.listdir(dir_path)
		for f in files:
			os.remove(dir_path + f)

def make_dir2(dir_path):
	try:
		os.makedirs(dir_path)
	except FileExistsError:
		files = os.listdir(dir_path)
		for f in files:
			try:
				shutil.rmtree(dir_path + f)
			except NotADirectoryError:
				os.remove(dir_path + f)

# return a new column which is the sum of previous window_size values
def windowsum(df, window_size: int, column_name: str):
	return df[[column_name]].rolling(window=window_size, min_periods=window_size).sum()

def quickmerge(listdf):
    return concat(listdf)

def df2operating_regions(df, column_names, thresholds):
    """
    Select from data frame the operating regions based on threshold
    """
    
    org_shape = df.shape[0]
    
    # select cells to be retained
    constraints = df.swifter.apply(
        lambda row: all([(cell > thresholds) for cell in row[column_names]]),
        axis=1)
    # Drop values set to be rejected
    df = df.drop(df.index[~constraints], axis = 0)
    
    print("Retaining {}% of the data".format(100*df.shape[0]/org_shape))
    
    return df

def getdf(exp_params):

	# read the data
	df1data = dp.readfile(exp_params['datapath'])  # read the pickled file for building data
	df1 = df1data.return_df(processmethods=['file2df'])  # return pickled df
	df2data = dp.readfile('../data/processed/interpolated/wetbulbtemp.pkl') # read the pickled file for wet bulb data
	df2 = df2data.return_df(processmethods=['file2df'])  # return pickled df
	df = dp.merge_df_columns([df1,df2])

	# create extra columns
	df['sat-oat']= df['sat']-df['oat']

	return df

def process_df(exp_params, df):
	"""Takes the raw data frame and performs operations common to 
	both energy modeling and reinforcement learning training
	i.e., smoothing, 0 threshold certain values, lag adjusting, aggregating,
	"""

	# smooth the data here as it we need to conditionally remove <0 values from certain columns
	if exp_params['smoothing']['smooth']:
		df = dp.dfsmoothing(df=df, column_names=list(df.columns), order=exp_params['smoothing']['order'],
							Wn=exp_params['smoothing']['cutoff'], T=exp_params['smoothing']['T'])

	# 0 thresholding values which may become negative due to smoothing
	for i in utils.POS_COLUMNS:
		df[i][df[i]<=0]=0.0001

	# lag adjusting
	if exp_params['lagging']['adjust_lag']:
		df = dp.createlag(df, exp_params['lagging']['lag_columns'], lag=exp_params['lagging']['data_lag'])

	# aggregating
	if exp_params['aggregation']['aggregate']:
		
		# rolling sum
		if exp_params['aggregation']['sum_aggregate']:
			df[exp_params['aggregation']['sum_aggregate']] =  utils.window_sum(df, 
			window_size=exp_params['period'], column_names=exp_params['aggregation']['sum_aggregate'])
		
		# rolling mean
		if exp_params['aggregation']['mean_aggregate']:
			df[exp_params['aggregation']['mean_aggregate']] =  utils.window_mean(df,
			 window_size=exp_params['period'], column_names=exp_params['aggregation']['mean_aggregate'])
		
		df = dp.dropNaNrows(df)
		
		# Sample the data at period intervals
		df = dp.sample_timeseries_df(df, period=exp_params['period'])

	return df

def dflist2rl_dflist(exp_params, dflist):
	"""Creates multiple overlapping dataframes of data_weeks length
	with an overlap of week_overlap. Needed only for RL part for creating the environment
	
	Arguments:
		dflist {[list]} -- input list of pandas dataframes
		data_weeks {[int]} -- length of the dataframe in terms of number of weeks
	"""
	weeklist = []
	num_of_elems = len(dflist)
	start_week = exp_params['df2xy']['start_week']
	end_week = exp_params['df2xy']['data_weeks']

	while end_week<exp_params['df2xy']['end_week']:  # exp_params['df2xy']['end_week']
		weeklist.append(quickmerge(dflist[start_week : end_week+1]))

		start_week += 1
		end_week += 1

	return weeklist

def dflist2array(exp_params, dflist, scaler, threshold_on_cols, threshold,
				inputs, outputs, input_timesteps, output_timesteps,scaleY=True):

	weeklist = []
	num_of_elems = len(dflist)
	start_week = exp_params['df2xy']['start_week']
	end_week = exp_params['df2xy']['data_weeks']

	# splitvalue
	splitvalue = dflist[end_week].shape[0]
	# year and week
	yearno = dflist[end_week].index[int(splitvalue/2)].year
	weekno = dflist[end_week].index[int(splitvalue/2)].week

	while end_week<exp_params['df2xy']['end_week']:  # exp_params['df2xy']['end_week']

		data_block_pre = quickmerge(dflist[start_week : end_week+1])
		if scaleY:
			data_block = df2operating_regions(data_block_pre, threshold_on_cols, threshold)
		else:
			data_block = data_block_pre

		# create numpy arrays
		X_train, X_test, y_train, y_test = dp.df_2_arrays(df = data_block,
		 predictorcols = inputs, outputcols = outputs, lag=exp_params['df2xy']['create_lag'],
		 scaling=exp_params['df2xy']['scaling'], scaler = scaler, scaleX = True, scaleY = scaleY,
		 split=splitvalue, shuffle=False,
		 reshaping=exp_params['df2xy']['reshaping'], input_timesteps=input_timesteps, output_timesteps = output_timesteps,)

		if not scaleY:  # if not scaling Y because its a binary class

			y_train = to_categorical(y_train)
			y_test = to_categorical(y_test)

		# splitvalue
		splitvalue = dflist[end_week].shape[0]
		# year and week
		weekno += 1
		weekno = weekno if weekno%53 != 0 else 1
		yearno = yearno if weekno!= 1 else yearno+1
		
		idx_end = -max(X_test.shape[1],y_test.shape[1])
		idx_start = idx_end - X_test.shape[0] + 1
		test_idx = data_block.index[[i for i in range(idx_start,idx_end+1,1)]]

		weeklist.append({
			'Id':'Year-{}-Week-{}'.format(str(yearno), 
										str(weekno)),
			'X_train':X_train,
			'y_train': y_train,
			'X_test': X_test,
			'y_test': y_test,
			'test_idx':test_idx,
		})

		start_week += 1
		end_week += 1

	return weeklist

def main(trial: int = 6, adaptive = True):

	exp_params = {}  # log experiment parameters

	# experiment number
	exp_params['trial'] = trial
	# whether we are testing adaptive or static control
	exp_params['adaptive'] = adaptive
	# Decide folder structure based on adaptive vs fixed controller
	exp_params['pathinsert'] = 'adaptive' if adaptive else 'fixed' 
	# period of aggregating 5 min data before starting experiments
	exp_params['period'] = 6
	# binary threshold below which values are considered 0
	exp_params['threshold_energy'] = 0.5  # kBTUS in half hour
	# track experiment seed for reproducability
	exp_params['seed'] = seed
	# data path
	exp_params['datapath'] = '../data/processed/buildingdata.pkl'

	# aggregate the data if needed
	exp_params['aggregation'] = {
		'aggregate': True, 'mean_aggregate' : utils.MEAN_AGG, 'sum_aggregate': utils.SUM_AGG,
	}
	# smooth the data if needed
	exp_params['smoothing'] = {
		'smooth': True, 'order' : 5, 'T' : 300, 'fs' : 1 / 300, 'cutoff' : 0.0001,
	}
	# lag certain columns if needed
	exp_params['lagging'] = {  # negative means shift column upwards
		'adjust_lag' : False, 'lag_columns' : utils.LAG_COLUMNS, 'data_lag' : -1,
	}
	# create temporal batches of data
	exp_params['df2dflist'] = {
		'days':7, 'hours': 0
	}
	# create numpy arrays from the data
	exp_params['df2xy'] = {
		'start_week' : 0, 'data_weeks' : 39, 'end_week' : 55,
		'create_lag' : 0, 'scaling' : True,
		 'reshaping' : True  # reshape data according to (batch_size, time_steps, features)
	}

	# cwe model configuration
	exp_params['cwe_model_config'] = {
		'inputs': ['flow', 'orh', 'wbt',  'sat-oat'], 'outputs' : ['cwe'], 'threshold': 0.5,
		'input_timesteps' : 1,  'output_timesteps' : 1,
		'lstm_hidden_units': 8, 'lstm_no_layers': 2, 'dense_hidden_units': 16, 'dense_no_layers': 4,
		'retrain_from_layers': 3, 'train_stateful': False, 'train_batchsize':32, 'train_epochs': 5000,
		'modeldesigndone' : False, 'initial_epoch' : 0, 'retain_prev_model' : True,
		'freeze_model' : True, 'reinitialize' : True, 'model_saved' : False, 'test_model_created' : False,
		'cwe_model_save_dir' : '../models/'+exp_params['pathinsert']+'/Trial_{}/cwe/'.format(exp_params['trial']),
	}
	make_dir2(exp_params['cwe_model_config']['cwe_model_save_dir'])  # create the folder if it does not exist
	os.mkdir(exp_params['cwe_model_config']['cwe_model_save_dir'] + 'loginfo')
	os.mkdir(exp_params['cwe_model_config']['cwe_model_save_dir'] + 'normalplots')
	os.mkdir(exp_params['cwe_model_config']['cwe_model_save_dir'] + 'detailedplots')

	# hwe model configuration
	exp_params['hwe_model_config'] = {
		'inputs': ['oat', 'orh', 'wbt', 'sat-oat'], 'outputs' : ['hwe'], 'threshold': 0.5,
		'input_timesteps' : 1,  'output_timesteps' : 1, 
		'lstm_hidden_units': 4, 'lstm_no_layers': 0, 'dense_hidden_units': 16, 'dense_no_layers': 6,
		'retrain_from_layers': 3, 'train_stateful': False, 'train_batchsize':32, 'train_epochs': 5000,
		'modeldesigndone' : False, 'initial_epoch' : 0, 'retain_prev_model' : True,
		'freeze_model' : True, 'reinitialize' : True, 'model_saved' : False, 'test_model_created' : False,
		'hwe_model_save_dir' : '../models/'+exp_params['pathinsert']+'/Trial_{}/hwe/'.format(exp_params['trial']),
	}
	make_dir2(exp_params['hwe_model_config']['hwe_model_save_dir'])  # create the folder if it does not exist
	os.mkdir(exp_params['hwe_model_config']['hwe_model_save_dir'] + 'loginfo')
	os.mkdir(exp_params['hwe_model_config']['hwe_model_save_dir'] + 'normalplots')
	os.mkdir(exp_params['hwe_model_config']['hwe_model_save_dir'] + 'detailedplots')
	

	# heating valve model configuration
	exp_params['vlv_model_config'] = {
		'inputs': ['oat', 'orh', 'wbt', 'sat-oat'], 'outputs' : ['valve_state'], 'threshold': 0.5,
		'input_timesteps' : 1,  'output_timesteps' : 1, 
		'lstm_hidden_units': 8, 'lstm_no_layers': 2, 'dense_hidden_units': 16, 'dense_no_layers': 4,
		'retrain_from_layers': 3, 'train_stateful': False, 'train_batchsize':32, 'train_epochs': 5000,
		'modeldesigndone' : False, 'initial_epoch' : 0, 'retain_prev_model' : True,
		'freeze_model' : True, 'reinitialize' : True, 'model_saved' : False, 'test_model_created' : False,
		'vlv_model_save_dir' : '../models/'+exp_params['pathinsert']+'/Trial_{}/vlv/'.format(exp_params['trial']),
	}
	make_dir2(exp_params['vlv_model_config']['vlv_model_save_dir'])  # create the folder if it does not exist
	os.mkdir(exp_params['vlv_model_config']['vlv_model_save_dir'] + 'loginfo')
	os.mkdir(exp_params['vlv_model_config']['vlv_model_save_dir'] + 'detailedplots')

	# steps to train the rl agent
	exp_params['num_rl_steps'] = 25000
	# always make sure that the number of environments is even; can also be os.cpu_count()
	exp_params['n_envs'] = 1
	# rl state space
	exp_params['obs_space_vars']=['oat', 'orh', 'wbt', 'avg_stpt', 'sat',]
	# rl action space
	exp_params['action_space_vars']=['sat']

	# save the model and rl agents here
	exp_params['rlmodel_save_dir'] ='../models/'+exp_params['pathinsert']+'/Trial_{}/rl/'.format(exp_params['trial'])
	make_dir2(exp_params['rlmodel_save_dir'])  # create the folder if it does not exist

	# save the rl performance output here
	exp_params['log_dir'] = exp_params['rlmodel_save_dir'] + '/performance/'
	make_dir2(exp_params['log_dir'])  # create the folder if it does not exist

	# path to save environment monitor logs
	exp_params['base_log_path'] = '../log/'+exp_params['pathinsert']+'/Trial_{}/'.format(exp_params['trial'])

	
	#######################         Begin : Creating all the data requirements     ##################################

	# get the raw dataframe
	df = getdf(exp_params)

	# process the raw df: smooth,0 adjust, adjust lags, aggregate
	df = process_df(exp_params, df)
	# created because non float columns are added later
	float_columns = df.columns
	
	# create data-scaler for all columns of entire the AGGREGATED data; can perform both minmax and normalize scaling
	scaler = dp.dataframescaler(df)

	# add binary classification column
	df['valve_state'] = 1.0
	df.loc[df['hwe']<= exp_params['threshold_energy'],['valve_state']] = 0.0
	# categorical_columns = ['valve_state']

	# create a list of of weekly dataframes
	dflist = dp.df2dflist_alt(df, subsequence=True, period=exp_params['period'], days=exp_params['df2dflist']['days'],
	 hours=exp_params['df2dflist']['hours'])
	
	# Create dflist with weekly overlaps for rl
	rl_dflist = dflist2rl_dflist(exp_params, dflist)

	# Create dflist with weekly overlaps for cwe model
	cwe_week_list = dflist2array(exp_params, dflist, scaler,
					exp_params['cwe_model_config']['outputs'], exp_params['cwe_model_config']['threshold'],
					exp_params['cwe_model_config']['inputs'], exp_params['cwe_model_config']['outputs'],
					exp_params['cwe_model_config']['input_timesteps'], exp_params['cwe_model_config']['output_timesteps'],)

	# Create dflist with weekly overlaps for hwe model
	hwe_week_list = dflist2array(exp_params, dflist, scaler,
					exp_params['hwe_model_config']['outputs'], exp_params['hwe_model_config']['threshold'],
					exp_params['hwe_model_config']['inputs'], exp_params['hwe_model_config']['outputs'],
					exp_params['hwe_model_config']['input_timesteps'], exp_params['hwe_model_config']['output_timesteps'],)

	# Create dflist with weekly overlaps for vlv model
	vlv_week_list = dflist2array(exp_params, dflist, scaler,
					exp_params['vlv_model_config']['outputs'], exp_params['vlv_model_config']['threshold'],
					exp_params['vlv_model_config']['inputs'], exp_params['vlv_model_config']['outputs'],
					exp_params['vlv_model_config']['input_timesteps'], exp_params['vlv_model_config']['output_timesteps'], scaleY=False)


	erromsg = "Unequal lists: len(rl_dflist)={0:}, len(cwe_week_list)={1:},\
	 len(hwe_week_list)={2:}, len(hwe_week_list)={3:}".format(
		len(rl_dflist), len(cwe_week_list), len(hwe_week_list), len(vlv_week_list))
	assert all([ len(rl_dflist)==len(cwe_week_list), len(rl_dflist)==len(hwe_week_list), len(rl_dflist)==len(vlv_week_list)]), erromsg
	#######################         End: Creating all the data requirements      ##################################


	#######################         Begin : Prerequisites for the environment     ##################################
	# important parameters to scale the reward
	reward_params = {
		'energy_saved': 100.0, 'energy_savings_thresh': 0.0, 'energy_penalty': -100.0, 'energy_reward_weight': 0.5,
		'comfort': 1.0, 'comfort_thresh': 0.10, 'uncomfortable': -1.0, 'comfort_reward_weight': 0.5,}

	exp_params['reward_params'] = reward_params

	scaled_df_stats = DataFrame(scaler.minmax_scale(df, float_columns, float_columns), index=df.index,
							columns=float_columns).describe().loc[['mean', 'std', 'min', 'max'],:]
	env_id = alumni_env.Env  # the environment ID or the environment class
	start_index = 0  # start rank index
	vec_env_cls = DummyVecEnv #  A custom `VecEnv` class constructor. Default: DummyVecEnv SubprocVecEnv 
	#######################         End : Prerequisites for the environment     ##################################

	# save the metadata
	with open('../models/'+exp_params['pathinsert']+'/Trial_{}/'.format(exp_params['trial'])+'experiemnt_params.json', 'w') as fp:
		json.dump(exp_params, fp, indent=4)
	reward_params['action_minmax'] = [np.array([df.sat_stpt.min()]), np.array([df.sat_stpt.max()])]  # required for clipping

	# main iteration loop

	env_created = False
	agent_created = False
	hwe_created = False
	cwe_created = False
	vlv_created = False
	initial_epoch_cwe, initial_epoch_hwe, initial_epoch_vlv = 0, 0, 0
	freeze_model = True
	reinitialize = True
	writeheader = True

	Week = 0


	for out_df, cwe_week, hwe_week, vlv_week in zip(rl_dflist, cwe_week_list, hwe_week_list, vlv_week_list):
		
		"""train cwe_model"""
		# load the data arrays
		X_train, y_train, X_test, y_test = cwe_week['X_train'], cwe_week['y_train'], cwe_week['X_test'], cwe_week['y_test']

		# create model for the first time
		if not cwe_created:
			#Instantiate learner model
			cwe_model = mp.regression_nn(exp_params['cwe_model_config']['cwe_model_save_dir'],
										inputdim = X_train.shape[-1],
										outputdim = y_train.shape[-1],
										input_timesteps = exp_params['cwe_model_config']['input_timesteps'],
										output_timesteps = exp_params['cwe_model_config']['output_timesteps'],
										period = exp_params['period'],
										stateful = exp_params['cwe_model_config']['train_stateful'],
										batch_size=exp_params['cwe_model_config']['train_batchsize'])
			# design the network
			cwe_model.design_network(
				lstmhiddenlayers=[exp_params['cwe_model_config']['lstm_hidden_units']] * exp_params['cwe_model_config']['lstm_no_layers'],
				densehiddenlayers=[exp_params['cwe_model_config']['dense_hidden_units']] * exp_params['cwe_model_config']['dense_no_layers'],
				dropoutlist=[[], []], batchnormalizelist=[[], []])
		# else load saved model and freeze layers and reinitialize head layers
		else:  
			cwe_model.model.load_weights(exp_params['cwe_model_config']['cwe_model_save_dir']+
											'LSTM_model_best')  # load best model weights in to cwe_model class
			if freeze_model:
				for layer in cwe_model.model.layers[:exp_params['cwe_model_config']['retrain_from_layers']]:
					layer.trainable = False
			if reinitialize:  
				for layer in cwe_model.model.layers[exp_params['cwe_model_config']['retrain_from_layers']:]:
					layer.kernel.initializer.run(session=K.get_session())
					layer.bias.initializer.run(session=K.get_session())
		# recompile model
		cwe_model.model_compile()


		# train the model
		cwe_history = cwe_model.train_model(X_train, y_train, X_test, y_test, epochs=exp_params['cwe_model_config']['train_epochs'],
									initial_epoch = initial_epoch_cwe)
		try:
			initial_epoch_cwe += len(cwe_history.history['loss'])
		except KeyError:
			pass

		# evaluate the model for metrics at this stage
		preds_test = cwe_model.evaluate_model( X_test, y_test, 
											scaler, save_plot_loc=exp_params['cwe_model_config']['cwe_model_save_dir']+'normalplots/',
											scaling=True, saveplot=False,
											Idx=cwe_week['Id'],
											outputdim_names=[utils.addl['names_abreviation'][exp_params['cwe_model_config']['outputs'][0]]],
											output_mean = [scaled_df_stats.loc['mean', exp_params['cwe_model_config']['outputs'][0]]])
		# log the outputs first in a csv file
		merged_log =  np.concatenate((scaler.minmax_inverse_scale(X_test[:,-1,:], exp_params['cwe_model_config']['inputs'], 
										exp_params['cwe_model_config']['inputs']), 
									 scaler.minmax_inverse_scale(y_test[:,-1,:], exp_params['cwe_model_config']['outputs'],
									 exp_params['cwe_model_config']['outputs']),
									 scaler.minmax_inverse_scale(preds_test[:,-1,:], exp_params['cwe_model_config']['outputs'],
									 exp_params['cwe_model_config']['outputs'])),
									 axis=-1)
		merged_log_df = DataFrame(data = merged_log, index = cwe_week['test_idx'], 
								columns = exp_params['cwe_model_config']['inputs'] + 
								[i+exp_params['cwe_model_config']['outputs'][0] for i in ['Actual ', 'Predicted ']])
		merged_log_df.to_csv(exp_params['cwe_model_config']['cwe_model_save_dir'] + 'detailedplots/' + cwe_week['Id']+'.csv')


		"""train hwe_model"""
		# load the data arrays
		X_train, y_train, X_test, y_test = hwe_week['X_train'], hwe_week['y_train'], hwe_week['X_test'], hwe_week['y_test']

		# create model for the first time
		if not hwe_created:
			#Instantiate learner model
			hwe_model = mp.regression_nn(exp_params['hwe_model_config']['hwe_model_save_dir'],
										inputdim = X_train.shape[-1],
										outputdim = y_train.shape[-1],
										input_timesteps = exp_params['hwe_model_config']['input_timesteps'],
										output_timesteps = exp_params['hwe_model_config']['output_timesteps'],
										period = exp_params['period'],
										stateful = exp_params['hwe_model_config']['train_stateful'],
										batch_size=exp_params['hwe_model_config']['train_batchsize'])
			# design the network
			hwe_model.design_network(
				lstmhiddenlayers=[exp_params['hwe_model_config']['lstm_hidden_units']] * exp_params['hwe_model_config']['lstm_no_layers'],
				densehiddenlayers=[exp_params['hwe_model_config']['dense_hidden_units']] * exp_params['hwe_model_config']['dense_no_layers'],
				dropoutlist=[[], []], batchnormalizelist=[[], []])
		# else load saved model and freeze layers and reinitialize head layers
		else:  
			hwe_model.model.load_weights(exp_params['hwe_model_config']['hwe_model_save_dir']+
											'LSTM_model_best')  # load best model weights in to hwe_model class
			if freeze_model:
				for layer in hwe_model.model.layers[:exp_params['hwe_model_config']['retrain_from_layers']]:
					layer.trainable = False
			if reinitialize:  
				for layer in hwe_model.model.layers[exp_params['hwe_model_config']['retrain_from_layers']:]:
					layer.kernel.initializer.run(session=K.get_session())
					layer.bias.initializer.run(session=K.get_session())
		# recompile model
		hwe_model.model_compile()


		# train the model
		hwe_history = hwe_model.train_model(X_train, y_train, X_test, y_test, epochs=exp_params['hwe_model_config']['train_epochs'],
									initial_epoch = initial_epoch_hwe)
		try:
			initial_epoch_hwe += len(hwe_history.history['loss'])
		except KeyError:
			pass

		# evaluate the model for metrics at this stage
		preds_test = hwe_model.evaluate_model( X_test, y_test, 
											scaler, save_plot_loc=exp_params['hwe_model_config']['hwe_model_save_dir']+'normalplots/',
											scaling=True, saveplot=False,
											Idx=hwe_week['Id'],
											outputdim_names=[utils.addl['names_abreviation'][exp_params['hwe_model_config']['outputs'][0]]],
											output_mean = [scaled_df_stats.loc['mean', exp_params['hwe_model_config']['outputs'][0]]])
		# log the outputs first in a csv file
		merged_log =  np.concatenate((scaler.minmax_inverse_scale(X_test[:,-1,:], exp_params['hwe_model_config']['inputs'],
										exp_params['hwe_model_config']['inputs']), 
									 scaler.minmax_inverse_scale(y_test[:,-1,:], exp_params['hwe_model_config']['outputs'],
									 exp_params['hwe_model_config']['outputs']),
									 scaler.minmax_inverse_scale(preds_test[:,-1,:], exp_params['hwe_model_config']['outputs'],
									 exp_params['hwe_model_config']['outputs'])),
									 axis=-1)
		merged_log_df = DataFrame(data = merged_log, index = hwe_week['test_idx'], 
									columns = exp_params['hwe_model_config']['inputs'] +
			 						[i+exp_params['hwe_model_config']['outputs'][0] for i in ['Actual ', 'Predicted ']])
		merged_log_df.to_csv(exp_params['hwe_model_config']['hwe_model_save_dir'] + 'detailedplots/' +hwe_week['Id']+'.csv')

		"""train vlv_state_model"""
		# load the data arrays
		X_train, y_train, X_test, y_test = vlv_week['X_train'], vlv_week['y_train'], vlv_week['X_test'], vlv_week['y_test']

		# create model for the first time
		if not vlv_created:
			#Instantiate learner model
			vlv_model = mp.classifier_nn(exp_params['vlv_model_config']['vlv_model_save_dir'],
											inputdim = X_train.shape[-1],
											outputdim = y_train.shape[-1],
											input_timesteps = exp_params['vlv_model_config']['input_timesteps'],
											period = exp_params['period'],
											stateful = exp_params['vlv_model_config']['train_stateful'],
											batch_size=exp_params['vlv_model_config']['train_batchsize'])
			# design the network
			vlv_model.design_network(
				lstmhiddenlayers=[exp_params['vlv_model_config']['lstm_hidden_units']] * exp_params['vlv_model_config']['lstm_no_layers'],
				densehiddenlayers=[exp_params['vlv_model_config']['dense_hidden_units']] * exp_params['vlv_model_config']['dense_no_layers'],
				dropoutlist=[[], []], batchnormalizelist=[[], []])
		# else load saved model and freeze layers and reinitialize head layers
		else:  
			vlv_model.model.load_weights(exp_params['vlv_model_config']['vlv_model_save_dir']+
										'LSTM_model_best')  # load best model weights in to vlv_model class
			if freeze_model:
				for layer in vlv_model.model.layers[:exp_params['vlv_model_config']['retrain_from_layers']]:
					layer.trainable = False
			if reinitialize:  
				for layer in vlv_model.model.layers[exp_params['vlv_model_config']['retrain_from_layers']:]:
					layer.kernel.initializer.run(session=K.get_session())
					layer.bias.initializer.run(session=K.get_session())
		# recompile model
		vlv_model.model_compile()

		# train the model
		vlv_history = vlv_model.train_model(X_train, y_train, X_test, y_test, epochs=exp_params['vlv_model_config']['train_epochs'],
									initial_epoch = initial_epoch_vlv)
		try:
			initial_epoch_vlv += len(vlv_history.history['loss'])
		except KeyError:
			pass

		# evaluate the model for metrics at this stage
		true_test, preds_test = vlv_model.evaluate_model( X_test, y_test, Idx=vlv_week['Id'])

		# log the outputs first in a csv file
		merged_log =  np.concatenate((scaler.minmax_inverse_scale(X_test[:,-1,:], exp_params['vlv_model_config']['inputs'],
										exp_params['vlv_model_config']['inputs']), 
									 true_test.reshape((-1,1)), preds_test.reshape((-1,1))), axis=-1)
		merged_log_df = DataFrame(data = merged_log, index = vlv_week['test_idx'], 
									columns = exp_params['vlv_model_config']['inputs'] +
			 						[i+exp_params['vlv_model_config']['outputs'][0] for i in ['Actual ', 'Predicted ']])
		merged_log_df.to_csv(exp_params['vlv_model_config']['vlv_model_save_dir'] + 'detailedplots/' +vlv_week['Id']+'.csv')

		

		"""create environment with new data"""
		# Path to a folder where the monitor files will be saved
		monitor_dir = exp_params['base_log_path']+'Interval_{}/'.format(Week)
		
		out_df[float_columns] = scaler.minmax_scale(out_df, float_columns, float_columns)  # the file for iterating; it is scaled before being passed
		# Arguments to be fed to the custom environment inside make_vec_env
		env_kwargs = dict(  #  Optional keyword argument to pass to the env constructor
			df = out_df, # the file for iterating
			totaldf_stats = scaled_df_stats,  # stats of the environment
			obs_space_vars=exp_params['obs_space_vars'],  # state space variable
			action_space_vars=exp_params['action_space_vars'],  # action space variable
			action_space_bounds=[[-2.0], [2.0]],  # bounds for real world action space; is scaled internally using the reward_params

			cwe_energy_model=load_model(exp_params['cwe_model_config']['cwe_model_save_dir']+'LSTM_model_best'),  # trained lstm model
			cwe_input_vars=exp_params['cwe_model_config']['inputs'],  # lstm model input variables
			cwe_input_shape=(1, 1, len(exp_params['cwe_model_config']['inputs'])),  # lstm model input data shape (no_samples, output_timestep, inputdim)

			hwe_energy_model=load_model(exp_params['hwe_model_config']['hwe_model_save_dir']+'LSTM_model_best'),  # trained lstm model
			hwe_input_vars=exp_params['hwe_model_config']['inputs'],  # lstm model input variables
			hwe_input_shape=(1, 1, len(exp_params['hwe_model_config']['inputs'])),  # lstm model input data shape (no_samples, output_timestep, inputdim)

			vlv_state_model=load_model(exp_params['vlv_model_config']['vlv_model_save_dir']+'LSTM_model_best'),  # trained lstm model
			vlv_input_vars=exp_params['vlv_model_config']['inputs'],  # lstm model input variables
			vlv_input_shape=(1, 1, len(exp_params['vlv_model_config']['inputs'])),  # lstm model input data shape (no_samples, output_timestep, inputdim)

			**reward_params  # the reward adjustment parameters
		)

		if not env_created:
			# make the environment
			envmodel =  cme.custom_make_vec_env(env_id = env_id, n_envs = exp_params['n_envs'],
							 seed = exp_params['seed'], start_index = start_index,
							monitor_dir = monitor_dir, vec_env_cls = vec_env_cls, env_kwargs = env_kwargs)
			env_created = True
		else:
			# change the monitor log directory
			envmodel.env_method('changelogpath', (monitor_dir))
			# reinitialize the environment
			envmodel.env_method('re_init_env', **env_kwargs)


		"""provide envrionment to the new or existing rl model"""
		# create agent with new data or reuse old agent if in the loop for the first time
		if not agent_created:
			# create new agent with the environment model
			agent = ppo_controller.get_agent(env=envmodel, model_save_dir=exp_params['rlmodel_save_dir'],
			 monitor_log_dir=exp_params['base_log_path'])
			


		"""train rl model"""
		# make sure environment is in train mode
		print("setting environment to train mode..... \n")
		envmodel.env_method('trainenv')

		if (not agent_created) | adaptive :
			# train the agent
			print("Training Started... \n")
			agent = ppo_controller.train_agent(agent, env = envmodel, steps=exp_params['num_rl_steps'],
			tb_log_name= exp_params['base_log_path']+'ppo2_event_folder')
		else:
			print('Retraining aborted. Fixed controller will be used... \n')


		"""test rl model"""
		# make sure environment is in test mode
		print("setting environment to test mode..... \n")
		envmodel.env_method('testenv')

		# provide path to the current best rl agent weights and test it
		best_model_path = exp_params['rlmodel_save_dir'] + 'best_model.pkl'
		test_perf_log = ppo_controller.test_agent(best_model_path, envmodel, num_episodes=1)

		# save the agent performance on test data
		lu.rl_perf_save(test_perf_log, exp_params['log_dir'], save_as='csv', header=writeheader)

		Week += 1  # shift to the next week

		agent_created = True  # flip the agent_created flag
		cwe_created = False  # flip the flag
		hwe_created = False  # flip the flag
		vlv_created = False  # flip the flag

		writeheader = False # flip the flag
	
	# plot the results
	pu.reward_agg_plot([trial], 0, Week-1, '../log/'+exp_params['pathinsert']+'/',
	 '../models/'+exp_params['pathinsert']+'/Trial_{}/'.format(trial), 0)

# wrap the environment in the vectorzied wrapper with SubprocVecEnv
# run the environment inside if __name__ == '__main__':

if __name__ == '__main__':
	main()
	print("Done!")