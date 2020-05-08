import os
# including the project directory to the notebook level
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
	sys.path.append(module_path)

from dataprocess import dataprocessor as dp
import alumni_env_utils as utils

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

if __name__ == '__main__':

	exp_params = {}  # log experiment parameters

	# period of aggregating 5 min data before starting experiments
	exp_params['period'] = 6

	# data path
	exp_params['datapath'] = '../data/processed/buildingdata.pkl'

	# smooth the data if needed
	exp_params['smoothing'] = {
		'smooth': True, 'order' : 5, 'T' : 300, 'fs' : 1 / 300, 'cutoff' : 0.0001,
	}

	# lag certain columns if needed
	exp_params['lagging'] = {  # negative means shift column upwards
		'adjust_lag' : False, 'lag_columns' : utils.LAG_COLUMNS, 'data_lag' : -1,
	}

	# aggregate the data if needed
	exp_params['aggregation'] = {
		'aggregate': True, 'mean_aggregate' : utils.MEAN_AGG, 'sum_aggregate': utils.SUM_AGG,
	}

	# get the raw dataframe
	df = getdf(exp_params)

	# process the raw df: smooth,0 adjust, adjust lags, aggregate
	df = process_df(exp_params, df)
	# created because non float columns are added later
	float_columns = df.columns

	# create data-scaler for all columns of entire the AGGREGATED data; can perform both minmax and normalize scaling
	scaler = dp.dataframescaler(df)

	# save the scaler information
	scaler.save_stats(path='../models/adaptive/Trial_0')

