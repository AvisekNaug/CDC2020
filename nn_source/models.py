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
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from dataprocess.plotutils import pred_v_target_plot

class lstm_model():
	
	
	def __init__(self, saveloc: str, inputdim: int, outputdim: int = 1, input_timesteps: int = 1, output_timesteps: int = 1,
	batch_size = 32, reg_l1: float = 0.01, reg_l2: float = 0.02, period: int = 12, stateful: bool = False,
	modelerror = 'mse', optimizer = 'adam'):

		self.saveloc = saveloc
		self.inputdim = inputdim
		self.outputdim = outputdim
		self.input_timesteps = input_timesteps
		self.output_timesteps = output_timesteps
		self.batch_size = batch_size
		self.l1, self.l2 = reg_l1, reg_l2
		self.period = period
		self.stateful = stateful
		self.modelerror = modelerror
		self.optimizer = optimizer

		# time gaps in minutes, needed only for human readable results in output file
		self.timegap = self.period*5
		self.epochs = 0

		# possible regularization strategies
		self.regularizers = L1L2(self.l1, self.l2)

		# logging error on each iteration subsequence
		# self.preds_train = []  # each element has (samplesize, outputsequence=1, feature=1)
		# self.preds_test = []  # each element has (samplesize, outputsequence=1, feature=1)

		# create a file to log the error
		if not self.saveloc.endswith('/'):  # attach forward slash if saveloc does not have one
			self.saveloc += '/'
		file = open(self.saveloc + str(self.timegap)+'min Results_File.txt','w')
		file.close()


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

			self.out = LSTM(no_units, return_sequences=True, recurrent_regularizer=self.regularizers, stateful = self.stateful)(self.out)

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


	def show_model(self,):
		print(self.model.summary())

	def model_callbacks(self,):

		self.modelchkpoint = ModelCheckpoint(self.saveloc+'LSTM_model_{epoch:02d}_{val_loss:.2f}',
		 monitor = 'val_loss', save_best_only = True, period=2)

		self.earlystopping = EarlyStopping(monitor = 'val_loss', patience=5, restore_best_weights=True)

		self.reduclronplateau = ReduceLROnPlateau(monitor = 'val_loss', patience=2, cooldown = 3)

	
	def train_model(self, X_train, y_train, X_val, y_val, epochs: int = 100, saveModel: bool = False):

		# Number of epochs to run
		self.epochs = epochs

		# train the model
		self.history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, \
			validation_data=(X_val, y_val) , verbose=2, shuffle=False, callbacks=[self.modelchkpoint, \
				self.earlystopping, self.reduclronplateau])

		if saveModel:
			self.save_model()


	def save_model(self,):

			self.model.save(self.saveloc+'LSTM_model_{:02d}epochs.hdf5'.format(self.epochs))

	def evaluate_model(self, X_train, y_train, X_test, y_test, y_sc, saveplot: bool = False, Week: int = 0,
	 lag: int = -1, outputdim_names = ['TotalEnergy']):
	

		# evaluate model on data. output -> (nsamples, output_timesteps, outputdim)
		self.preds_train = self.model.predict(X_train, batch_size=self.batch_size)
		self.preds_test = self.model.predict(X_test, batch_size=self.batch_size)

		for i in range(self.outputdim):
			for j in range(self.output_timesteps):

				# log error on training data
				rmse = sqrt(mean_squared_error(self.preds_train[:, j, i], y_train[:, j, i]))
				cvrmse = 100*(rmse/np.mean(y_train[:, j, i]))
				mae = mean_absolute_error(self.preds_train[:, j, i], y_train[:, j, i])
				file = open(self.saveloc + str(self.timegap)+'min Results_File.txt','a')
				file.write('Week No:{}-Time Step {}: Train RMSE={} |Train CVRMSE={} \
					|Train MAE={} \n'.format(Week,j+1, rmse, cvrmse, mae))
				file.close()

				# log error on test data
				rmse = sqrt(mean_squared_error(self.preds_test[:, j, i], y_test[:, j, i]))
				cvrmse = 100*(rmse/np.mean(y_test[:, j, i]))
				mae = mean_absolute_error(self.preds_test[:, j, i], y_test[:, j, i])
				file = open(self.saveloc + str(self.timegap)+'min Results_File.txt','a')
				file.write('Week No:{}-Time Step {}: Test RMSE={} |Test CVRMSE={} \
					|Test MAE={} \n'.format(Week,j+1, rmse, cvrmse, mae))
				file.close()

		if saveplot:

			pred_v_target_plot(self.timegap, self.outputdim, self.output_timesteps,
			 self.preds_train, y_train, self.saveloc, y_sc, lag = -1, outputdim_names = outputdim_names)

			pred_v_target_plot(self.timegap, self.outputdim, self.output_timesteps,
			 self.preds_test, y_test, self.saveloc, y_sc, lag)

		return [self.preds_train, self.preds_test]
