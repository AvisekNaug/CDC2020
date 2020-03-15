import numpy as np

import tensorflow as tf
from stable_baselines import PPO2
from stable_baselines.results_plotter import load_results, ts2xy

# current best mean reward
best_mean_reward = -np.inf
#steps completed
n_steps = 0


def get_agent(env, model_save_dir = '../models/controller/', monitor_logdir = '../log/'):
	"""
	The Proximal Policy Optimization algorithm combines ideas from A2C
	(having multiple workers) and TRPO (it uses a trust region to improve the actor)
	"""

	# Custom MLP policy
	policy_net_kwargs = dict(act_fun=tf.nn.tanh,
						net_arch=[dict(pi=[64, 64], 
						vf=[64, 64])])

	# Create the agent
	agent = PPO2("MlpPolicy", 
				env, 
				policy_kwargs=policy_net_kwargs, 
				verbose=1)

	agent.is_tb_set = False  # attribute for callback
	agent.model_save_dir = model_save_dir  # load or save model here
	agent.monitor_logdir = monitor_logdir  # logging directory

	return agent

def train_agent(agent, env=None, steps=30000, tb_log_name = "../log/ppo2_event_folder"):
	"""
	Train the agent on the environment
	"""
	agent.set_env(env)
	trained_model = agent.learn(total_timesteps=steps, callback=CustomCallBack, tb_log_name=tb_log_name)

	return trained_model

def CustomCallBack(_locals, _globals):
	"""
	Store neural network weights during training if the current episode's
	performance is better than the previous best performance.
	"""
	self_ = _locals['self']

	global best_mean_reward, n_steps

	if not self_.is_tb_set:
		# Do some initial logging setup

		"""Do some stuff here for setting up logging: eg log the weights"""

		# reverse the key
		self_is_tb_set = True

	# Print stats every 1000 calls, since for PPO it is called at every n_step
	if (n_steps + 1) % 100 == 0:
		# Evaluate policy training performance
		if np.any(_locals['masks']):  # if the current update step contains episode termination
			x, y = ts2xy(load_results(self_.monitor_logdir), 'episodes')
			#TODO: design different reader for reward data
			if len(x) > 0:
				mean_reward = np.mean(y[-5:])
				print(x[-1], 'timesteps')
				print("Best mean reward: {:.2f} - Latest 5 sample mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))
				# New best model, you could save the agent here
				if mean_reward > best_mean_reward:
					best_mean_reward = mean_reward
					# Example for saving best model
					print("Saving new best model")
					self_.save(self_.model_save_dir + 'best_model.pkl')
		n_steps += 1

	return True

def test_agent(agent_weight_path: str, env, num_episodes = 1):

	# load agent weights
	agent = PPO2.load(agent_weight_path, 
					env)
	
	"""
	num_envs is an attribute of any vectorized environment
	NB. Cannot use env.get_attr("num_envs") as get_attr return attributes of the base 
	environment which is being vectorized, not attributes of the VecEnv class itself.
	
	"""
	# create the list of classes which will help store the performance metrics
	perf_metrics_list = [performancemetrics()]*env.num_envs

	for _ in range(num_episodes):

		# issue episode begin command for all the environments
		for perf_metrics in perf_metrics_list: perf_metrics.on_episode_begin()
		# reset all the environments
		obslist = env.reset()
		# done_track contains information on which envs have completed the episode
		dones_trace = [False]*env.num_envs
		# set all_done to be true only when all envs have completed an episode
		all_done = all(dones_trace)

		# Step through the environment till all envs have finished current episode
		while not all_done:
			action, _ = agent.predict(obslist)
			obslist, _, doneslist, infotuple = env.step(action)

			# update dones_trace with new episode end information
			dones_trace = [i | j for i,j in zip(dones_trace,doneslist)]

			for idx, done in enumerate(dones_trace):
				if not done:
					perf_metrics_list[idx].on_step_end(infotuple[idx])  # log the info dictionary
			
		# end episode command issued for all environments
		for perf_metrics in perf_metrics_list: perf_metrics.on_episode_end()

	return perf_metrics_list

class performancemetrics():
	"""
	Store the history of performance metrics. Useful for evaluating the
	agent's performance:
	"""

	def __init__(self):
		self.metriclist = []  # store multiple performance metrics for multiple episodes
		self.metric = {}  # store performance metric for each episode

	def on_episode_begin(self):
		self.metric = {}  # flush existing metric data from previous episode

	def on_episode_end(self):
		self.metriclist.append(self.metric)

	def on_step_end(self, info = {}):
		for key, value in info.items():
			if key in self.metric:
				self.metric[key].append(value)
			else:
				self.metric[key] = [value]
