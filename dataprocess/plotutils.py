import matplotlib
from matplotlib import pyplot as plt


def pred_v_target_plot(timegap, outputdim, output_timesteps, preds, target,
 saveloc, scaling: bool, scaler, lag: int = -1, outputdim_names : list = [], typeofplot: str = 'train'):

	if not outputdim_names:
		outputdim_names = ['Output']*outputdim

	plt.rcParams["figure.figsize"] = (15, 5*outputdim*output_timesteps)
	font = {'size':16}
	plt.rc('font',**font)
	plt.rc('legend',**{'fontsize':14})

	# Inerse scaling the data for each time step
	if scaling:
		for j in  range(output_timesteps):
			preds[:,j,:] = scaler.inverse_transform(preds[:,j,:])
			target[:,j,:] = scaler.inverse_transform(target[:,j,:])


	# training output
	fig, axs = plt.subplots(nrows = outputdim*output_timesteps, squeeze=False)
	for i in range(outputdim):
		for j in range(output_timesteps):
			# plot predicted
			axs[i+j, 0].plot(preds[:, j, i], 'r--', label='Predicted'+outputdim_names[i])
			# plot target
			axs[i+j, 0].plot(target[:, j, i], 'g--', label='Actual'+outputdim_names[i])
			# Plot Properties
			axs[i+j, 0].set_title('Predicted vs Actual at time = t + {} for {}'.format(-1*lag+j, outputdim_names[i]))
			axs[i+j, 0].set_xlabel('Time points at {} minute(s) intervals'.format(timegap))
			axs[i+j, 0].set_ylabel('Actual Energy')
			axs[i+j, 0].grid(which='both',alpha=100)
			axs[i+j, 0].legend()
			axs[i+j, 0].minorticks_on()
	fig.savefig(saveloc+str(timegap)+'_LSTM_'+typeofplot+'prediction.pdf', bbox_inches='tight')
	plt.close(fig)