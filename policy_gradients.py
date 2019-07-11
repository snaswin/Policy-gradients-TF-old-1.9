# AIPC policy gradients- TF

import numpy as np
import tensorflow as tf
import gym
import os

class PGmodel:
	def __init__(self, num_states, num_actions, sess):
		self._num_states = num_states
		self._num_actions = num_actions
		
		self._sess = sess
		self._var_init = None
		self._optimizer = None
		self._loss = None
		
		self._define_model()
		
		
	def _define_model(self):
		
		self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
		fc1 = tf.layers.dense(self._states, 100, activation=tf.nn.relu)
		fc2 = tf.layers.dense(fc1, 50, activation= tf.nn.relu)
		logits = tf.layers.dense(fc2, self._num_actions)
		
		#sample an action
		self._sample = tf.reshape(tf.multinomial(logits=logits, num_samples=1) ,[])
		
		#get log probabilities
		log_prob = tf.log(tf.nn.softmax(logits) )
		
		#acts & advantages- Train components
		self._acts = tf.placeholder(tf.int32)
		self._advantages = tf.placeholder(tf.float32)
		
		#Log probs of actions from Episode
		##	create indices
		indices = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + self._acts
		act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices )
		
		#loss for Gradient Ascent
		self._loss = -tf.reduce_sum(tf.multiply(act_prob, self._advantages) )
		
		#update
		optimizer = tf.train.AdamOptimizer(learning_rate= 1e-2)
		self._optimizer = optimizer.minimize(self._loss)
		
		#init
		self._var_init = tf.global_variables_initializer()
		
		
	def act(self,state):
		#Get one action, by Multinomial sampling instead of E-greedy
		state = state.reshape((1,-1))
		return self._sess.run(self._sample, feed_dict={self._states: state} )
		
	def train_episode(self, states, acts, advantages):
		self._sess.run(self._optimizer, feed_dict={self._states:states, self._acts:acts, self._advantages: advantages} )

##=====================================================================##

def cpu_only():
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	
	if tf.test.gpu_device_name():
		print("GPU found")
	else:
		print("No GPU found")


#Running one episode
def policy_rollout(env, agent):
	state = env.reset()
	reward = 0
	done = False
	
	states, actions, rewards = [], [], []
	while not done:
		
		env.render()
		states.append(state)
		
		action = agent.act(state)
		next_state, reward, done, info = env.step(action)
		
		# ~ #custom reward
		# ~ if next_state[0] >= 0.1:
			# ~ reward += 10
		# ~ elif next_state[0] >= 0.25:
			# ~ reward += 20
		# ~ elif next_state[0] >= 0.5:
			# ~ reward += 100		
		
		actions.append(action)
		rewards.append(reward)
		
		state = next_state
		
	return states, actions, rewards

#Converting rewards to Advanatages for one episode
def process_rewards(rewards):
	#tot reward: length of episode
	return [len(rewards)] * len(rewards)
	
def main():
	env_name = "CartPole-v0"
	#env_name = "MountainCar-v0"
	#env_name = "SpaceInvaders-v0"
	env = gym.make(env_name)
	
	#monitor_dir = "./monitor/cp_exp1"
	#env.monitor.start(monitor_dir, force= True)
	
	
	num_states = env.observation_space.shape[0]
	num_actions = env.action_space.n
	
	with tf.Graph().as_default(), tf.Session() as sess:
		
		aipc = PGmodel(num_states, num_actions, sess)
		
		sess.run(aipc._var_init)
		
		for E in range(400):
			print("====\nEpisode {}\n====".format(E) )
			
			b_obs, b_acts, b_rews = [], [], []
			
			for _ in range(10):
				
				obs, acts, rews = policy_rollout(env, aipc)
				print("Episode steps: ", len(obs))
				
				b_obs.extend(obs)
				b_acts.extend(acts)
				advantages = process_rewards(rews)
				b_rews.extend(advantages)
			
				
			#Update policy
			#normalize rewards
			b_rews = (b_rews - np.mean(b_rews)) / ( np.std(b_rews) + 1e-10)
			
			aipc.train_episode(b_obs, b_acts, b_rews)
		
		env.env.close()
			

if __name__=="__main__":
	cpu_only()
	
	main()
				
				
				
		
	

	
		
	
	
	
	
