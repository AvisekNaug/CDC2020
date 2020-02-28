import matplotlib
from matplotlib import pyplot as plt


def pred_v_target_plot(timegap, outputdim, output_timesteps, preds, target,
 saveloc, scaler, lag: int = -1, outputdim_names : list = []):

	if not outputdim_names:
		outputdim_names = ['Output']*outputdim

	plt.rcParams["figure.figsize"] = (15, 5*outputdim*output_timesteps)
	font = {'family':'serif','size':16, 'serif': ['computer modern roman']}
	plt.rc('font',**font)
	plt.rc('legend',**{'fontsize':14})


	# training output
	fig, axs = plt.subplots(nrows = outputdim*output_timesteps, squeeze=False)
	for i in range(outputdim):
		for j in range(output_timesteps):
			# plot predicted
			axs[i+j, 1].plot(scaler.inverse_transform(preds[:, j, i]), 'r--', label='Predicted Energy')
			# plot target
			axs[i+j, 1].plot(scaler.inverse_transform(target[:, j, i]), 'g--', label='Actual Energy')
			# Plot Properties
			axs[i+j, 1].set_title('Predicted vs Actual at time = t + {} for {}'.format(-1*lag+j, outputdim_names[i]))
			axs[i+j, 1].set_xlabel('Time points at {} minute(s) intervals'.format(timegap))
			axs[i+j, 1].set_ylabel('Actual Energy')
			axs[i+j, 1].grid(which='both',alpha=100)
			axs[i+j, 1].legend()
			axs[i+j, 1].minorticks_on()
	fig.savefig(saveloc+str(timegap)+'_LSTM_Energyprediction.pdf', bbox_inches='tight')
	plt.close(fig)