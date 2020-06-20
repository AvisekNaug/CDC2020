import matplotlib
from matplotlib import pyplot as plt
import numpy as np

from scipy.interpolate import make_interp_spline, BSpline
from pandas import read_csv


def pred_v_target_plot(timegap, outputdim, output_timesteps, preds, target,
 saveloc, scaling: bool, scaler, lag: int = -1, outputdim_names : list = [], typeofplot: str = 'train', Idx: int = 0):

	if not outputdim_names:
		outputdim_names = ['Output']*outputdim

	plt.rcParams["figure.figsize"] = (15, 5*outputdim*output_timesteps)
	font = {'size':16}
	plt.rc('font',**font)
	plt.rc('legend',**{'fontsize':14})

	_preds = np.empty_like(preds)
	_target = np.empty_like(target)

	# Inerse scaling the data for each time step
	if scaling:
		for j in  range(output_timesteps):
			_preds[:,j,:] = scaler.inverse_transform(preds[:,j,:])
			_target[:,j,:] = scaler.inverse_transform(target[:,j,:])

	# attach forward slash if saveloc does not have one
	if not saveloc.endswith('/'):
			saveloc += '/'


	# training output
	fig, axs = plt.subplots(nrows = outputdim*output_timesteps, squeeze=False)
	for i in range(outputdim):
		for j in range(output_timesteps):
			# plot predicted
			axs[i+j, 0].plot(_preds[:, j, i], 'r--', label='Predicted '+outputdim_names[i])
			# plot target
			axs[i+j, 0].plot(_target[:, j, i], 'g--', label='Actual '+outputdim_names[i])
			# Plot Properties
			axs[i+j, 0].set_title('Predicted vs Actual at time = t + {} for {}'.format(-1*lag+j, outputdim_names[i]))
			axs[i+j, 0].set_xlabel('Time points at {} minute(s) intervals'.format(timegap))
			axs[i+j, 0].set_ylabel('Actual Energy')
			axs[i+j, 0].grid(which='both',alpha=100)
			axs[i+j, 0].legend()
			axs[i+j, 0].minorticks_on()
	fig.savefig(saveloc+str(timegap)+'min_LSTM_'+typeofplot+'_prediction-{}.pdf'.format(Idx), bbox_inches='tight')
	plt.close(fig)

  
def regression_plot(timegap, xs, outputdim, output_timesteps, input_timesteps, pred, target, X_var, x_loc, x_lab,
saveloc, scaling: bool, Xscaler, yscaler, lag: int = -1, outputdim_names : list = [], typeofplot: str = 'train', 
				Idx: str = '', extradata=None):
	
	if not outputdim_names:
		outputdim_names = ['Output']*outputdim
		
	_pred = np.empty_like(pred)
	_target = np.empty_like(target)
	_X_var = np.empty_like(X_var)

	plt.rcParams["figure.figsize"] = (15, 5*outputdim*output_timesteps)
	font = {'size':16}
	plt.rc('font',**font)
	plt.rc('legend',**{'fontsize':14})

	# Inerse scaling the data for each time step
	if scaling:
		for j in  range(output_timesteps):
			_pred[:,j,:] = yscaler.inverse_transform(pred[:,j,:])
			_target[:,j,:] = yscaler.inverse_transform(target[:,j,:])
		for j in  range(input_timesteps):
			_X_var[:,j,:] = Xscaler.inverse_transform(X_var[:,j,:])     

	# attach forward slash if saveloc does not have one
	if not saveloc.endswith('/'):
			saveloc += '/'

	sample_styles = ['b3-', 'c>-', 'y^-', 'm4-', 'k*-', 'm>-', 'c^-']
	# training output
	fig, axs = plt.subplots(nrows = outputdim*output_timesteps, squeeze=False)
	for i in range(outputdim):
		for j in range(output_timesteps):
			# plot predicted
			axs[i+j, 0].plot_date(xs, _pred[:, j, i], 'ro-', label='Predicted '+outputdim_names[i])
			# plot target
			axs[i+j, 0].plot_date(xs, _target[:, j, i], 'g*-', label='Actual '+outputdim_names[i])
			# plot other variables
			for idx, name, style in zip(x_loc, x_lab,sample_styles):
				axs[i+j, 0].plot_date(xs, _X_var[:, 0, idx], style, label=name)
			if extradata is not None:
				axs[i+j, 0].plot_date(xs, 100*extradata, 'k*-', label='On-Off-State(Predicted)')
			# Plot Properties
			axs[i+j, 0].set_title('Predicted vs Actual at time = t + {} for {}'.format(-1*lag+j, outputdim_names[i]))
			#axs[i+j, 0].set_xlabel('Time points at {} minute(s) intervals'.format(timegap))
			axs[i+j, 0].set_xlabel('DateTime')
			axs[i+j, 0].set_ylabel('Different Variables')
			axs[i+j, 0].grid(which='both',alpha=100)
			axs[i+j, 0].minorticks_on()
			axs[i+j, 0].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
	fig.savefig(saveloc+str(timegap)+'min_LSTM_detailed_'+typeofplot+'_plot-{}.pdf'.format(Idx), bbox_inches='tight')
	plt.gcf().autofmt_xdate()
	plt.close(fig)


def classification_plot(timegap, xs, input_timesteps, pred, target, X_var, x_loc, x_lab,
				saveloc, scaling: bool, Xscaler, lag: int = -1, outputdim_names : list = [],
				 typeofplot: str = 'train', 
				Idx: str = ''):

	if not outputdim_names:
		outputdim_names = ['Output']
		
	_pred = pred
	_target = target
	_X_var = np.empty_like(X_var)

	plt.rcParams["figure.figsize"] = (15, 5)
	font = {'size':16}
	plt.rc('font',**font)
	plt.rc('legend',**{'fontsize':14})

	# Inerse scaling the data for each time step
	if scaling:
		for j in  range(input_timesteps):
			_X_var[:,j,:] = Xscaler.inverse_transform(X_var[:,j,:])     

	# attach forward slash if saveloc does not have one
	if not saveloc.endswith('/'):
			saveloc += '/'

	sample_styles = ['b3-', 'c>-', 'y^-', 'm4-', 'k*-', 'm>-', 'c^-']
	# training output
	fig, axs = plt.subplots(nrows = 1, squeeze=False)
	for i in range(1):
		for j in range(1):
			# plot predicted
			axs[i+j, 0].plot_date(xs, 100*_pred, 'ro-', label='Predicted '+outputdim_names[i])
			# plot target
			axs[i+j, 0].plot_date(xs, 100*_target, 'g*-', label='Actual '+outputdim_names[i])
			# plot other variables
			for idx, name, style in zip(x_loc, x_lab,sample_styles):
				axs[i+j, 0].plot_date(xs, _X_var[:, 0, idx], style, label=name)
			# Plot Properties
			axs[i+j, 0].set_title('Predicted vs Actual at time = t + {} for {}'.format(-1*lag+j, outputdim_names[i]))
			#axs[i+j, 0].set_xlabel('Time points at {} minute(s) intervals'.format(timegap))
			axs[i+j, 0].set_xlabel('DateTime')
			axs[i+j, 0].set_ylabel('Different Variables')
			axs[i+j, 0].grid(which='both',alpha=100)
			axs[i+j, 0].minorticks_on()
			axs[i+j, 0].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
	fig.savefig(saveloc+str(timegap)+'min_LSTM_detailed_'+typeofplot+'_plot-{}.pdf'.format(Idx), bbox_inches='tight')
	plt.gcf().autofmt_xdate()
	plt.close(fig)	


def regression_bar_plot(bars: list, color, bar_label: str, saveloc: str, barwidth = 0.50, smoothcurve: bool = False,
 bar_annotate: bool = False, saveplot: bool = False, plot_name: str = 'BarPlot', xlabel: str = 'Xlabel', ylabel: str = 'ylabel',
 title: str = 'Title', xticktype: str = 'Bar', xticklist = None, plotwidth = None, plotheight = 15, fontsize = 16, avg_line = False,
 savetitle = 'Error Bar Plot.png'):

	if plotwidth is None:
		plt.rcParams["figure.figsize"] = (0.6*len(bars), plotheight)
	else:
		plt.rcParams["figure.figsize"] = (plotwidth,plotheight)
	font = {'size':fontsize, 'family': "Times New Roman"}
	plt.rc('font',**font)
	plt.rc('legend',**{'fontsize':fontsize})

	N = len(bars)
	ind = np.arange(N)

	plt.bar(ind,
	 bars, 
	 barwidth, 
	 color=color, )
	 #label=bar_label)

	# horizontal average line
	if avg_line:
		avg_ = sum(bars) / len(bars) 
		plt.axhline(avg_, c = 'k',ls='dashdot', label='Average CVRMSE :{0:.2f}%'.format(avg_),linewidth=3)

	if xticklist is None:
		plt.xticks(ind, [xticktype + str(i) for i in range(1, N + 1)], rotation = 45)
	else:
		plt.xticks(ind, xticklist, rotation = 90)
	plt.ylim((0,max(bars)+5))
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.legend(fontsize=18)
	#plt.title(title)

	if smoothcurve:
		T = np.array([i for i in range(len(bars))])
		xnew = np.linspace(T.min(), T.max(), 300)
		spl = make_interp_spline(T, bars, k=3)  # type: BSpline
		power_smooth = spl(xnew)
		plt.plot(xnew, power_smooth, color='k', alpha=0.8)

	if bar_annotate:
		for i, v in enumerate(bars):
			plt.text(i-0.3 , bars[i] + 1.5, '{0:.1f}%'.format(np.abs(v)), color='k',
			 fontweight='bold', fontsize=15, rotation= 90)
			plt.text(i-0.3 , 7, str(i+1), color='k',
			 fontweight='bold', fontsize=15, rotation= 0)

	if saveplot:
		# attach forward slash if saveloc does not have one
		if not saveloc.endswith('/'):
			saveloc += '/'
		plt.savefig(saveloc + savetitle, bbox_inches='tight', dpi=300)


def classification_bar_plot(bars: list, color, bar_label: str, saveloc: str, barwidth = 0.50, smoothcurve: bool = False,
 bar_annotate: bool = False, saveplot: bool = False, plot_name: str = 'BarPlot', xlabel: str = 'Xlabel', ylabel: str = 'ylabel',
 title: str = 'Title', xticktype: str = 'Bar', xticklist = None, plotwidth = None, plotheight = 15, fontsize = 16,  avg_line = False,
 metric_name = 'Accuracy', savetitle = 'Prediction Bar Plot.png'):

	if plotwidth is None:
		plt.rcParams["figure.figsize"] = (0.6*len(bars), plotheight)
	else:
		plt.rcParams["figure.figsize"] = (plotwidth,plotheight)
	font = {'size':fontsize, 'family': "Times New Roman"}
	plt.rc('font',**font)
	plt.rc('legend',**{'fontsize':fontsize})

	N = len(bars)
	ind = np.arange(N)

	plt.bar(ind,
	 bars, 
	 barwidth, 
	 color=color, )
	 #label=bar_label)

	# horizontal average line
	if avg_line:
		avg_ = sum(bars) / len(bars)
		plt.axhline(avg_, c = 'k',ls='dashdot', label='Average {} : {:.2f}'.format(metric_name,avg_),linewidth=3)

	if xticklist is None:
		plt.xticks(ind, [xticktype + str(i) for i in range(1, N + 1)], rotation = 45)
	else:
		plt.xticks(ind, xticklist, rotation = 90)
	plt.ylim((0.0,1.1))
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.legend(fontsize=18)
	#plt.title(title)

	if smoothcurve:
		T = np.array([i for i in range(len(bars))])
		xnew = np.linspace(T.min(), T.max(), 300)
		spl = make_interp_spline(T, bars, k=3)  # type: BSpline
		power_smooth = spl(xnew)
		plt.plot(xnew, power_smooth, color='k', alpha=0.8)

	if bar_annotate:
		for i, v in enumerate(bars):
			plt.text(i - 0.30, bars[i] + 0.05, '{0:.2f}'.format(np.abs(v)), color='k',
			 fontweight='bold', fontsize=15, rotation= 90)
			plt.text(i - 0.30, 0.3, str(i+1), color='k',
			 fontweight='bold', fontsize=15, rotation= 0)

	if saveplot:
		# attach forward slash if saveloc does not have one
		if not saveloc.endswith('/'):
			saveloc += '/'
		plt.savefig(saveloc + savetitle, bbox_inches='tight', dpi=300)


def reward_agg_plot(trial_list: list, 
					interval_start: int, 
					interval_end: int,
					readfrom : str, 
					saveto: str, 
					envid: int = 0):
	# since there will be virtually no variation between env_ids in terms of performance, 
	# we do not num_envs number of other environments and select just once env_id
	
	trialwise_ep_reward = []

	for trial in trial_list:

		ep_reward = []

		for interval in range(interval_start, interval_end+1):

			readpath = readfrom +'Trial_'+ str(trial) +'/Interval_' + str(interval)+'/' + str(envid) + '.monitor.csv'
			ep_reward = ep_reward + [float(j) for j in read_csv(readpath, header=1,index_col=False)['r']]
		
		trialwise_ep_reward.append(ep_reward)

	trialwise_ep_reward = np.array(trialwise_ep_reward)

	rewardmean, rewardstd = np.mean(trialwise_ep_reward, axis=0), np.std(trialwise_ep_reward, axis=0)
	updatedlb, updatedub = np.subtract(rewardmean, 2*rewardstd), np.add(rewardmean, 2*rewardstd)

	# Plot parameters
	width = 15.0
	height = width / 1.618
	plt.rcParams["figure.figsize"] = (width, height)
	plt.rc('font',**{'size':16})
	plt.rc('legend',**{'fontsize':14})

	fig, ax = plt.subplots()

	# plot the shaded range of the confidence intervals
	ax.fill_between(range(rewardmean.shape[0]), updatedub, updatedlb,
					color='g', alpha=0.6, hatch="\\", label='Two Standard Deviation \n Bounds for Cumulative Reward')
	# plot the mean on top
	ax.plot(rewardmean, 'lime', marker='*', label = 'Mean Reward')

	# Axis labeling
	ax.set_title('Progress of Cumulative Episode Reward \n as Training Progresses')
	ax.set_xlabel('Episode Number')
	ax.set_ylabel('Cumulative reward per episode')
	ax.legend(loc='upper left', bbox_to_anchor=(0, -0.10), prop={'size': 15})

	# Add grid for estimating
	plt.grid(which='both', linewidth=0.2)
	plt.show()
	fig.savefig(saveto + 'AverageReward.png', bbox_inches='tight')

