"""
Utility methods that re being tested to simplify alumni environment handling
"""
import pandas as pd
import numpy as np

# Sum Aggregate these variables if used
SUM_AGG = ['cwe', 'flow', 'hwe', 'hw_sf', 'ghi']  # sum aggregate these variables
MEAN_AGG = ['oat', 'sat', 'sat_stpt', 'orh', 'avg_stpt' 'hw_rt',
 'hw_st', 'hw_s_stp', 'hx_vlv1', 'wbt', 'sat-oat']  # mean aggregate these variables

# Lag these columns if needed
LAG_COLUMNS = ['hwe', 'cwe']

# columns whose values should never be less than zero
POS_COLUMNS = ['cwe', 'flow', 'hwe', 'hw_sf', 'ghi', 'oat', 'sat', 'sat_stpt', 'orh', 'avg_stpt' 'hw_rt',
 'hw_st', 'hw_s_stp', 'hx_vlv1', 'wbt']

# return a new column which is the sum of previous window_size values
def window_sum(df_, window_size: int, column_names: list):
    return df_[column_names].rolling(window=window_size, min_periods=window_size).sum()

# return a new column which is the average of previous window_size values
def window_mean(df_, window_size: int, column_names: list):
    return df_[column_names].rolling(window=window_size, min_periods=window_size).mean()


