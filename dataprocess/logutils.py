"""
Contains custom methods to log information provided from different methods
"""
import pandas as pd


# save test performance of the rl agent
def rl_perf_save(test_perf_log_list: list, log_dir: str, save_as: str = 'csv'):

	# assert that perf metric has data from at least one episode
	assert all([len(i.metriclist) != 0 for i in test_perf_log_list]), 'Need metric data for at least one episode'

	# iterate throguh each environment in a single Trial
	for idx, test_perf_log in enumerate(test_perf_log_list):
	
		# performance metriclist is a list where each element has 
		# performance data for each episode in a dict
		perf_metric_list = test_perf_log.metriclist

		# iterating through the list to save the data
		for episode_dict in perf_metric_list:

			if save_as == 'csv':
				df = pd.DataFrame(data=episode_dict)
				df.to_csv(log_dir+'EnvId{}-results.csv'.format(idx), index=False, mode='a+')
			else:
				for key, value in episode_dict.items():
					f = open(log_dir + 'EnvId{}-'.format(idx) + key + '.txt', 'a+')
					f.writelines("%s\n" % j for j in value)
					f.close()