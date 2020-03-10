import numpy as np

import tensorflow as tf
from stable_baselines import PPO2
from stable_baselines.results_plotter import load_results, ts2xy

def get_agent(env):
	"""
	The Proximal Policy Optimization algorithm combines ideas from A2C
	(having multiple workers) and TRPO (it uses a trust region to improve the actor)
	"""

	# Custom MLP policy
	policy_net_kwargs = dict(act_fun=tf.nn.tanh,
						net_arch=[dict(pi=[16, 16, 16, 16], 
						vf=[16, 16, 16, 16])])

	# Create the agent
	agent = PPO2("MlpPolicy", 
				env, 
				policy_kwargs=policy_net_kwargs, 
				verbose=1)

	return agent

def train_agent(agent, env=None, steps=30000):
	"""
	Train the agent on the environment
	"""
	agent.set_env(env)
	trained_model = agent.learn(total_timesteps=steps, callback=CustomCallBack, tb_log_name="../log/ppo2_event_folder")

	return trained_model

# current best mean reward
best_mean_reward = -np.inf
#steps completed
n_steps = 0
# logging directory
monitor_logdir = '../results/rlagents/'

def CustomCallBack(_locals, _globals):
	"""
	Store neural network weights during training if the current episode's
	performance is better than the previous best performance.
	"""
	self_ = _locals['self']

	global best_mean_reward, n_steps, monitor_logdir

	if self_.is_tb_set:
		# Do some initial logging setup

		"""Do some stuff here for setting up logging: eg log the weights"""

		# reverse the key
		self_is_tb_set = True

	# Print stats every 1000 calls, since for PPO it is called at every n_step
	if (n_steps + 1) % 100 == 0:
		# Evaluate policy training performance
		if np.any(_locals['masks']):  # if the current update step contains episode termination
			x, y = ts2xy(load_results(monitor_logdir), 'timesteps')
			if len(x) > 0:
				mean_reward = np.mean(y[-100:])
				print(x[-1], 'timesteps')
				print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))
				# New best model, you could save the agent here
				if mean_reward > best_mean_reward:
					best_mean_reward = mean_reward
					# Example for saving best model
					print("Saving new best model")
					_locals['self'].save(monitor_logdir + 'best_model.pkl')
		n_steps += 1

	return True
