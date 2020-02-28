"""This script contains the skeleton Keras models
"""

from math import sqrt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from keras.models import Model
from keras.layers import Input, Dense, LSTM, RepeatVector, Reshape, Dropout, BatchNormalization, Activation
from keras.callbacks import TensorBoard
from keras.regularizers import L1L2
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

import matplotlib
from matplotlib import pyplot as plt

class lstm_model():
	
	
	def __init__(self, inputdim: int, outputdim: int = 1, input_timesteps: int = 1, output_timesteps: int = 1,
	batchsize = 32, reg_l1: float = 0.01, reg_l2: float = 0.02, period: int = 12, stateful: bool = True,
	modelerror = 'mse', optimizer = 'adam'):

		self.inputdim = inputdim
		self.outputdim = outputdim
		self.input_timesteps = input_timesteps
		self.output_timesteps = output_timesteps
		self.batchszie = batchsize
		self.l1, self.l2 = reg_l1, reg_l2
		self.period = period
		self.stateful = stateful
		self.modelerror = modelerror
		self.optimizer = optimizer

		# time gaps in minutes, needed only for human readable results in output file
		self.timegap = self.period*5

		# possible regularization strategies
		self.regularizers = L1L2(self.l1, self.l2)

		# logging error on each iteration subsequence
		self.train_plot = []  # each element has (samplesize, outputsequence=1, feature=1)
		self.test_plot = []  # each element has (samplesize, outputsequence=1, feature=1)


	# Create the network
	def design_model(self, lstmhiddenlayers: list = [64, 64], densehiddenlayers: list = [],
	 dropoutlist: list = [[],[]], batchnormalizelist : list = [[],[]]):

		# There will be one dense layer to output the targets
		densehiddenlayers += [self.outputdim]

		# Checking processors
		if not dropoutlist[0]:
			dropoutlist[0] = [False] * (len(lstmhiddenlayers))
		else:
			assert len(lstmhiddenlayers)==len(dropoutlist[0]), "lstmhiddenlayers and dropoutlist[0] must be of same length"

		if not dropoutlist[1]:
			dropoutlist[1] = [False] * (len(densehiddenlayers))
		else:
			assert len(densehiddenlayers)==len(dropoutlist[1]), "densehiddenlayers and dropoutlist[1] must be of same length"
		if not batchnormalizelist[0]:
			batchnormalizelist[0] = [False] * (len(lstmhiddenlayers))
		else:
			assert len(lstmhiddenlayers)==len(batchnormalizelist[0]), "lstmhiddenlayers and batchnormalizelist[0] must be of same length"

		if not batchnormalizelist[1]:
			batchnormalizelist[1] = [False] * (len(densehiddenlayers))
		else:
			assert len(densehiddenlayers)==len(batchnormalizelist[1]), "lstmhiddenlayers and batchnormalizelist[1] must be of same length"
		
		# Design the network
		self.input_layer = Input(batch_shape=(None, self.input_timesteps, self.inputdim), name='input_layer')
		self.reshape_layer = Reshape((self.input_timesteps*self.inputdim,),name='reshape_layer')(self.input_layer)
		self.num_op = self.output_timesteps
		self.input = RepeatVector(self.num_op, name='input_repeater')(self.reshape_layer)

		self.out = self.input

		# LSTM layers
		for no_units, dropout, normalize in zip(lstmhiddenlayers, dropoutlist[0], batchnormalizelist[0]):

			self.out = LSTM(no_units, return_sequences=True, recurrent_regularizer=self.regularizers)(self.out)

			if dropout:
				self.out = Dropout(0.2)(self.out)

			if normalize:
				self.out = BatchNormalization()(self.out)

		# Dense layers
		activationlist = ['relu']*(len(densehiddenlayers)-1) + ['linear']  # relu activation for all dense layers exept last
		for no_units, dropout, normalize, activation in zip(densehiddenlayers, dropoutlist[1], batchnormalizelist[1], activationlist):

			self.out = Dense(no_units, activation=activation)(self.out)

			if dropout:
				self.out = Dropout(0.2)(self.out)

			if normalize:
				self.out = BatchNormalization()(self.out)

		# compile model
		self.model = Model(inputs=self.input_layer, outputs=self.out)
		self.model.compile(loss=self.modelerror, optimizer=self.optimizer)

	
