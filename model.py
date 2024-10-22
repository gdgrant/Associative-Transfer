### Authors: Nicolas Y. Masse, Gregory D. Grant

# Required packages
import tensorflow as tf
import numpy as np
import pickle
import os, sys, time

# Plotting suite
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model modules
from parameters import *
import stimulus
import AdamOpt

# Match GPU IDs to nvidia-smi command
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class Model:

	def __init__(self, stimulus, reward_data, mask, posterior, associative, gate_cost):

		print('Defining graph...')

		self.stimulus_data	= stimulus
		self.reward_data	= reward_data
		self.time_mask		= mask

		self.posterior_dist	= posterior 	# Should be of shape [batch_size x n_tasks]
		self.action_m		= associative 	# Should be of shape [n_assoc x n_assoc]

		self.gate_cost		= gate_cost

		self.declare_variables()
		self.run_model()
		self.optimize()

		print('Graph successfully defined.')


	def declare_variables(self):

		self.var_dict = {}
		lstm_var_prefixes = ['Wf', 'Wi', 'Wo', 'Wc', 'Uf', 'Ui', 'Uo', 'Uc', \
			'Pf', 'Pi', 'Po', 'Pc', 'bf', 'bi', 'bo', 'bc']
		gate_var_prefixes = ['W_pos_gate', 'W_cor_gate', 'W_hip_gate', \
			'W_cor_gate_val', 'W_hip_gate_val', 'b_act_gate', 'b_pos_gate']
		actm_var_prefixes = ['encoder']
		RL_var_prefixes = ['W_out', 'W_val', 'b_out', 'b_val']

		with tf.variable_scope('cortex'):
			for p in lstm_var_prefixes + RL_var_prefixes:
				self.var_dict[p] = tf.get_variable(p, initializer=par[p+'_init'])

		with tf.variable_scope('gate'):
			for p in gate_var_prefixes:
				self.var_dict[p] = tf.get_variable(p, initializer=par[p+'_init'])

		with tf.variable_scope('hippocampus'):
			for p in actm_var_prefixes:
				self.var_dict[p] = tf.get_variable(p, initializer=par[p+'_init'], trainable=False)


	def run_model(self):

		self.h = []
		self.c = []
		self.g = []
		self.pol_out = []
		self.val_out = []
		self.action  = []
		self.reward  = []
		self.mask    = []
		self.pol_out_raw    = []
		self.trial_encoding = []

		h = tf.zeros([par['batch_size'], par['n_hidden']])
		c = tf.zeros([par['batch_size'], par['n_hidden']])

		reward = tf.zeros([par['batch_size'], par['n_val']])
		mask   = tf.ones([par['batch_size'], 1])

		for t in range(par['num_time_steps']):

			# Make two possible actions and values for the network to pursue
			# by way of the LSTM-based cortex module and the associative
			# network hippocampus module
			ca, cv, h, c = self.cortex_lstm(self.stimulus_data[t], self.posterior_dist, h, c)
			ha, hv 		 = self.hippocampus_associative(self.stimulus_data[t], self.posterior_dist)

			# Decide between the two possible actions (policies) with a gate
			pol_out_raw, val_out, g	= self.gate_selector(ca, cv, ha, hv, self.posterior_dist)

			# Compute outputs for action and policy loss
			action_index	= tf.multinomial(pol_out_raw, 1)
			action 			= tf.one_hot(tf.squeeze(action_index), par['n_output'])
			pol_out			= tf.nn.softmax(pol_out_raw, -1) # Note softmax for entropy calculation

			# Check for trial continuation (ends if previous reward is non-zero)
			continue_trial	= tf.cast(tf.equal(reward, 0.), tf.float32)
			mask 		   *= continue_trial
			reward			= tf.reduce_sum(action*self.reward_data[t,...], axis=-1, keepdims=True) \
								* mask * self.time_mask[t,:,tf.newaxis]

			# Record outputs
			self.h.append(h)
			self.c.append(c)
			self.g.append(g)
			self.pol_out.append(pol_out)
			self.val_out.append(val_out)
			self.pol_out_raw.append(pol_out_raw)
			self.action.append(action)
			self.reward.append(reward)
			self.mask.append(mask)

		self.h = tf.stack(self.h, axis=0)
		self.c = tf.stack(self.c, axis=0)
		self.g = tf.stack(self.g, axis=0)
		self.pol_out = tf.stack(self.pol_out, axis=0)
		self.val_out = tf.stack(self.val_out, axis=0)
		self.pol_out_raw = tf.stack(self.pol_out_raw, axis=0)
		self.action = tf.stack(self.action, axis=0)
		self.reward = tf.stack(self.reward, axis=0)
		self.mask = tf.stack(self.mask, axis=0)
		self.trial_encoding = tf.stack(self.trial_encoding, axis=0)


	def cortex_lstm(self, x, p, h, c):
		""" Compute LSTM state from inputs and vars...
				f : forgetting gate
				i : input gate
				c : cell state
				o : output gate
			...and generate an action from that state. """

		# Iterate LSTM
		f  = tf.sigmoid(x @ self.var_dict['Wf'] + p @ self.var_dict['Pf'] + h @ self.var_dict['Uf'] + self.var_dict['bf'])
		i  = tf.sigmoid(x @ self.var_dict['Wi'] + p @ self.var_dict['Pi'] + h @ self.var_dict['Ui'] + self.var_dict['bi'])
		o  = tf.sigmoid(x @ self.var_dict['Wo'] + p @ self.var_dict['Po'] + h @ self.var_dict['Uo'] + self.var_dict['bo'])
		cn = tf.tanh(x @ self.var_dict['Wc'] + p @ self.var_dict['Pc'] + h @ self.var_dict['Uc'] + self.var_dict['bc'])
		c  = f * c + i * cn
		h  = o * tf.tanh(c)

		# Generate action vector
		a = h @ self.var_dict['W_out'] + self.var_dict['b_out']
		v = h @ self.var_dict['W_val'] + self.var_dict['b_val']

		# Return action, hidden state, and cell state
		return a, v, h, c


	def hippocampus_associative(self, x, p):
		""" Generate action associated with current stimulus and probability """

		enc_x = tf.nn.relu(x @ self.var_dict['encoder'])
		self.trial_encoding.append(enc_x)

		L_output = tf.concat([p, enc_x, tf.zeros([par['batch_size'],par['n_output']+par['num_reward_types']])], axis=-1)
		for i in range(par['associative_iters']):

			# Apply action memory to provided information
			L_output = L_output + par['train_beta']*(L_output @ self.action_m)

			# Logistic activation -- shift midpoint to 0.5 and
			# make curve steeper for faster decision convergence
			L_output = par['train_gamma']*tf.tanh(L_output)#1/(0.75*par['train_gamma']) * L_output)

		# Isolate suggested action vector
		a = L_output[:,-(par['n_output']+par['num_reward_types']):-par['num_reward_types']]

		# Isolate reward values and map one-hot to actual values
		v = L_output[:,-par['num_reward_types']:]
		v = v @ par['reward_map_matrix']

		# Return action
		return a, v


	def gate_selector(self, ca, cv, ha, hv, p):
		""" Select whether to use the cortex action or the hippocampus action
			based on the posterior distribution and the actions themselves """

		# Calculate the two parts of the gate and then multiply them together
		action_gate    = tf.nn.sigmoid(ca @ self.var_dict['W_cor_gate'] + ha @ self.var_dict['W_hip_gate'] \
				+ cv @ self.var_dict['W_cor_gate_val'] + hv @ self.var_dict['W_hip_gate_val'] + self.var_dict['b_act_gate'])
		posterior_gate = tf.nn.sigmoid(p @ self.var_dict['W_pos_gate'] + self.var_dict['b_pos_gate'])
		full_gate = 0. * action_gate * posterior_gate

		# Make a "composite" action for the network to use by weighting with the gate value
		action = full_gate*ca + (1-full_gate)*ha
		value  = full_gate*cv + (1-full_gate)*hv

		# Return the action to use and the gate value
		return action, value, full_gate


	def optimize(self):
		""" Calculate losses and apply corrections to model """

		# Set up optimizer and required constants
		epsilon = 1e-7
		cortex_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cortex')
		gate_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gate')
		cortex_optimizer = AdamOpt.AdamOpt(cortex_vars, learning_rate=par['learning_rate'])
		gate_optimizer = AdamOpt.AdamOpt(gate_vars, learning_rate=par['learning_rate'])

		# Spiking activity loss (penalty on high activation values in the hidden layer)
		self.spike_loss = par['spike_cost']*tf.reduce_mean(tf.stack([mask*time_mask*tf.reduce_mean(h) \
			for (h, mask, time_mask) in zip(tf.unstack(self.h), tf.unstack(self.mask), tf.unstack(self.time_mask))]))

		# Gate vlaue loss (penalty on indecisiveness)
		#self.gate_loss = par['gate_cost']*tf.reduce_mean(self.g*(1-self.g))
		self.gate_loss = self.gate_cost*tf.reduce_mean((0.5-tf.abs(0.5-self.g))**2)

		# Correct time mask shape
		self.time_mask = self.time_mask[...,tf.newaxis]

		# Get the value outputs of the network, and pad the last time step
		val_out = tf.concat([self.val_out, tf.zeros([1,par['batch_size'],par['n_val']])], axis=0)

		# Determine terminal state of the network
		terminal_state = tf.cast(tf.logical_not(tf.equal(self.reward, tf.constant(0.))), tf.float32)

		# Compute predicted value and the advantage for plugging into the policy loss
		pred_val = self.reward + par['discount_rate']*val_out[1:,:,:]*(1-terminal_state)
		advantage = pred_val - val_out[:-1,:,:]

		# Stop gradients back through action, advantage, and mask
		action_static    = tf.stop_gradient(self.action)
		advantage_static = tf.stop_gradient(advantage)
		mask_static      = tf.stop_gradient(self.mask)
		pred_val_static  = tf.stop_gradient(pred_val)

		# Multiply masks together
		self.full_mask   = mask_static*self.time_mask

		# Policy loss
		self.pol_loss = -tf.reduce_mean(self.full_mask*advantage_static*action_static*tf.log(epsilon+self.pol_out))

		# Value loss
		self.val_loss = 0.5*par['val_cost']*tf.reduce_mean(self.full_mask*tf.square(val_out[:-1,:,:]-pred_val_static))

		# Entropy loss
		self.ent_loss = -par['entropy_cost']*tf.reduce_mean(tf.reduce_sum(self.full_mask*self.pol_out*tf.log(epsilon+self.pol_out), axis=2))

		# Collect RL losses
		RL_loss = self.pol_loss + self.val_loss - self.ent_loss

		# Collect loss terms and compute gradients
		total_loss = RL_loss + self.spike_loss + self.gate_loss
		self.train_cortex = cortex_optimizer.compute_gradients(total_loss)
		self.train_gate = gate_optimizer.compute_gradients(total_loss)


def update_associative_memory(M, stimulus, action, reward, posterior, full_mask):

	def activation(x):
		# x = 10*(x - 0.5)
		# return 1/(1+np.exp(-x))
		return par['train_gamma']*np.tanh(x)#1/(0.75*par['train_gamma']) * x)

	# # Select appropriate events
	# inds = list(np.where(reward != 0.))
	# rew_zero_times = np.array([np.random.randint(ts) for ts in inds[0]])
	# inds[0] = np.array([np.random.choice([rt,zt], p=[0.75,0.25]) for rt, zt in zip(inds[0],rew_zero_times)])

	# # Collect those events
	# event_stimuli = stimulus[inds[0],inds[1],:]
	# event_actions = action[inds[0],inds[1],:]
	# event_rewards = reward[inds[0],inds[1],:]

	# Determine mask
	mask_inds = np.where(full_mask.reshape(-1, full_mask.shape[-1]) != 0.)[0]

	# Encode rewards as one-hot
	reward_encoding = np.zeros([par['num_time_steps'],par['batch_size'],par['num_reward_types']])

	reward_indices = np.zeros_like(np.squeeze(reward)).astype(np.int32)
	for rv, ri in par['reward_map'].items():
		reward_indices += np.where((reward[:,:,0]+1e-6).round(2) == rv, ri, 0)

	for t in range(par['num_time_steps']):
		reward_encoding[t,np.arange(par['batch_size']),reward_indices[t,...]] = 1.

	# Expand posterior data
	posterior = posterior[np.newaxis,...] * np.ones([par['num_time_steps'],1,1])

	# Reshape and cull event data
	posterior = posterior.reshape(-1, posterior.shape[-1])[mask_inds,:]
	event_stimuli = stimulus.reshape(-1, stimulus.shape[-1])[mask_inds,:]
	event_actions = action.reshape(-1, action.shape[-1])[mask_inds,:]
	event_rewards = reward_encoding.reshape(-1, reward_encoding.shape[-1])[mask_inds,:]
	reward_mult = (par['train_gamma']/2)*reward.reshape(-1, reward.shape[-1])[mask_inds,:,np.newaxis] + 0.1

	# Update M with the new event data
	event_data = np.concatenate([posterior, event_stimuli, event_actions, event_rewards], axis=1)

	M = M + par['train_gamma']*par['train_alpha']*np.mean(reward_mult*event_data[:,:,np.newaxis]*event_data[:,np.newaxis,:], axis=0)
	M = activation(M)

	return M


def main(gpu_id=None):

	if gpu_id is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'], par['n_input']], 'stim')
	r = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'], par['n_output']], 'reward')
	m = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size']], 'mask')
	p = tf.placeholder(tf.float32, [par['batch_size'], par['n_tasks']], 'posterior')
	a = tf.placeholder(tf.float32, [par['n_assoc'], par['n_assoc']], 'associative')
	g = tf.placeholder(tf.float32, [], 'gate_cost')

	stim = stimulus.Stimulus()
	M = np.zeros([par['n_assoc'], par['n_assoc']], dtype=np.float32)

	print('Loading no trials\' worth of data into M as initialization.\n')
	for i in range(0):
		name, trial_info = stim.generate_trial()
		encoding = np.stack([np.maximum(0, trial_info['neural_input'][t,...] @ par['encoder_init']) for t in range(par['num_time_steps'])], axis=0)
		action = trial_info['desired_output']
		reward = np.sum(trial_info['reward_data'] * action, axis=-1, keepdims=True)
		posterior_dist = np.zeros([par['batch_size'], par['n_tasks']], dtype=np.float32)
		posterior_dist[:,0] = 0.8
		posterior_dist[:,1] = 0.2
		full_mask = trial_info['train_mask'][...,np.newaxis]

		M = update_associative_memory(M, encoding, action, reward, posterior_dist, full_mask)

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8) if gpu_id == '0' else tf.GPUOptions()
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			model = Model(x, r, m, p, a, g)

		sess.run(tf.global_variables_initializer())

		print('\nGate value of 0 indicates using hippocampus (associative network).')
		print('Gate value of 1 indicates using cortex (LSTM).\n')


		for t in range(par['n_tasks']):
			print()
			if t == 1:
				quit('Complete.')
			for i in range(par['n_batches']):

				posterior_dist = np.zeros([par['batch_size'], par['n_tasks']], dtype=np.float32)
				posterior_dist[:,t] = 0.8
				posterior_dist[:,(t+1)%2] = 0.2

				name, trial_info = stim.generate_trial(t)

				# if t == 0:
				# 	feed_dict = {x:trial_info['neural_input'], r:trial_info['reward_data'],\
				# 		m:trial_info['train_mask'], p:posterior_dist, a:M, g:par['gate_cost']}

				# 	_, _, encoding, reward, pol_loss, gate, action = \
				# 		sess.run([model.train_cortex, model.train_gate, model.trial_encoding, \
				# 			model.reward, model.pol_loss, model.g, model.action], feed_dict=feed_dict)
				
				# elif t == 1:
				# 	feed_dict = {x:trial_info['neural_input'], r:trial_info['reward_data'],\
				# 		m:trial_info['train_mask'], p:posterior_dist, a:M, g:0.}

				# 	_, encoding, reward, pol_loss, gate, action = \
				# 		sess.run([model.train_gate, model.trial_encoding, \
				# 			model.reward, model.pol_loss, model.g, model.action], feed_dict=feed_dict)

				feed_dict = {x:trial_info['neural_input'], r:trial_info['reward_data'],\
					m:trial_info['train_mask'], p:posterior_dist, a:M, g:0.}

				encoding, reward, pol_loss, gate, action, pol_out, pol_out_raw, full_mask = \
					sess.run([model.trial_encoding, model.reward, model.pol_loss, model.g, \
						model.action, model.pol_out, model.pol_out_raw, model.full_mask], feed_dict=feed_dict)

				M = update_associative_memory(M, encoding, action, reward, posterior_dist, full_mask)

				if i%100 == 0:

					fig, ax = plt.subplots(1,3,figsize=(18,8))
					im = ax[0].imshow(M, aspect='auto')
					ax[0].set_title('Associative Matrix')
					fig.colorbar(im, ax=ax[0], orientation='horizontal')

					im = ax[1].imshow(np.mean(pol_out, axis=1).T, aspect='auto', clim=(0,np.mean(pol_out, axis=1).max()))
					ax[1].set_title('Mean Policy Over Time')
					fig.colorbar(im, ax=ax[1], orientation='horizontal')

					im = ax[2].imshow(np.mean(pol_out_raw, axis=1).T, aspect='auto')
					ax[2].set_title('Mean Raw Policy Over Time')
					fig.colorbar(im, ax=ax[2], orientation='horizontal')

					fig.suptitle('Iter {}'.format(i))
					plt.savefig('./plotdir/iter{}.png'.format(str(i).zfill(6)), bbox_inches='tight')
					plt.clf()
					plt.close()

					print('Task {:>2} | Iter {:>4} | Reward: {:6.3f} | Gate: {:5.3f} | Pol. Loss: {:6.3f} |'.format(\
						t, i, np.mean(np.sum(reward, axis=0)), np.mean(gate), pol_loss))

	print('Model complete.\n')


if __name__ == '__main__':
	try:
		if len(sys.argv) > 1:
			main(sys.argv[1])
		else:
			main()
	except KeyboardInterrupt:
		quit('Quit by KeyboardInterrupt.')