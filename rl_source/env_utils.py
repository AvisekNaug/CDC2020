"""
Utility methods that re being tested to simplify custom environment handling
"""
import pandas as pd
import numpy as np

from typing import Union


class dataframescaler():
	"""
	An attempt at creating a class that can scale or inverse scale any variable
	individually or collectively of a dataframe once, the original raw and cleaned values
	are given
	"""
	def __init__(self, df):
		
		self.df = df  # get the dataframe

		self.stats = df.describe()  # collect the statistics of a dataframe

		self.columns = df.columns

	def minmax_scale(self,
					 input_data : Union[pd.DataFrame, np.ndarray], 
					 input_columns: list):

		min = self.stats.loc['min',input_columns]
		max = self.stats.loc['max',input_columns]

		if isinstance(input_data,pd.DataFrame):
			return (input_data[input_columns]-min)/(max-min)
		else:
			return (input_data-min.to_numpy())/(max.to_numpy()-min.to_numpy())

	def minmax_inverse_scale(self,
					 input_data : Union[pd.DataFrame, np.ndarray], 
					 input_columns: list):

		min = self.stats.loc['min',input_columns]
		max = self.stats.loc['max',input_columns]

		if isinstance(input_data,pd.DataFrame):
			return input_data[input_columns]*(max-min) + min
		else:
			return input_data*(max.to_numpy()-min.to_numpy()) + min.to_numpy()
		
	def standard_scale(self, 
						input_data : Union[pd.DataFrame, np.ndarray], 
						input_columns: list):
		
		mean= self.stats.loc['mean',input_columns]
		std = self.stats.loc['std',input_columns]

		if isinstance(input_data,pd.DataFrame):
			return (input_data[input_columns]-mean)/std
		else:
			return (input_data-mean.to_numpy())/std.to_numpy()

	def standard_inverse_scale(self, 
						input_data : Union[pd.DataFrame, np.ndarray], 
						input_columns: list):
		
		mean= self.stats.loc['mean',input_columns]
		std = self.stats.loc['std',input_columns]

		if isinstance(input_data,pd.DataFrame):
			return input_data[input_columns]*std + mean
		else:
			return input_data*std.to_numpy() + mean.to_numpy()
