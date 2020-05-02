"""
Utility methods that re being tested to simplify alumni environment handling
"""
import pandas as pd
import numpy as np

# Sum Aggregate these variables if used
# sum aggregate these variables
SUM_AGG = ['cwe', 'flow', 'hwe', 'hw_sf', 'ghi']
MEAN_AGG = ['oat', 'sat', 'sat_stpt', 'orh', 'avg_stpt', 'hw_rt',
			'hw_st', 'hw_s_stp', 'hx_vlv1', 'wbt', 'sat-oat']  # mean aggregate these variables

# Lag these columns if needed
LAG_COLUMNS = ['hwe', 'cwe']

# columns whose values should never be less than zero
POS_COLUMNS = ['cwe', 'flow', 'hwe', 'hw_sf', 'ghi', 'oat', 'sat', 'sat_stpt', 'orh', 'avg_stpt', 'hw_rt',
			   'hw_st', 'hw_s_stp', 'hx_vlv1', 'wbt']

# additional info
addl = {
	'metainfo': 'create a diff of sat and oat for hot water energy prediction as it is useful. See 1.0.8',
	'names_abreviation': {
		'oat': 'Outside Air Temperature',
		'orh': 'Outside Air Relative Humidity',
		'sat-oat': 'Difference of Supply Air and Outside Air Temps ',
		'ghi': 'Global Solar Irradiance',
		'hw_sf': 'Hot Water System Flow Rate',
		'hx_vlv1': 'Hot Water Valve %',
		'hw_st': 'Hot Water Supply Temperature',
		'hwe': 'Hot Water Energy Consumption',
		'cwe': 'Cooling Energy',
		'wbt': 'Wet Builb Temperature',
		'flow': 'Chilled Water Flow rate'
	}
}

# return a new column which is the sum of previous window_size values


def window_sum(df_, window_size: int, column_names: list):
	return df_[column_names].rolling(window=window_size, min_periods=window_size).sum()

# return a new column which is the average of previous window_size values


def window_mean(df_, window_size: int, column_names: list):
	return df_[column_names].rolling(window=window_size, min_periods=window_size).mean()

