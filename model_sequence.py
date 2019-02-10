### Authors: Nicolas Y. Masse, Gregory D. Grant

# Required packages
import tensorflow as tf
import numpy as np
import pickle
import os, sys, time

# Plotting suite
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model modules
from parameters_sequence import *
import stimulus_sequence
import AdamOpt

# Match GPU IDs to nvidia-smi command
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class Model:

	def __init__(self, stimulus, reward_data, mask):

		print('Defining graph...')

		self.stimulus_data	= stimulus
		self.reward_data	= reward_data
		self.time_mask		= mask

		self.declare_variables()
		self.run_model()
		self.optimize()

		print('Graph successfully defined.')


	def declare_variables(self):

		self.var_dict = {}
		lstm_var_prefixes = ['Wf', 'Wi', 'Wo', 'Wc', 'Uf', 'Ui', 'Uo', 'Uc', \
			'Vf', 'Vi', 'Vo', 'Vc', 'bf', 'bi', 'bo', 'bc', 'W_write']
		RL_var_prefixes = ['W_pol', 'W_val', 'b_pol', 'b_val']

		with tf.variable_scope('cortex'):
			for p in lstm_var_prefixes + RL_var_prefixes:
				self.var_dict[p] = tf.get_variable(p, initializer=par[p+'_init'])


	def run_model(self):

		self.h = []
		self.c = []
		self.g = []
		self.pol_out = []
		self.val_out = []
		self.action  = []
		self.reward  = []
		self.mask    = []
		self.trial_encoding = []

		h = tf.zeros([par['batch_size'], par['n_hidden']], dtype = tf.float32)
		h_read = tf.zeros([par['batch_size'], par['n_hidden']], dtype = tf.float32)
		c = tf.zeros([par['batch_size'], par['n_hidden']], dtype = tf.float32)
		A = tf.zeros([par['batch_size'], par['n_hidden'], par['n_hidden']], dtype = tf.float32)

		for i in range(par['trials_per_seq']):

			reward = tf.zeros([par['batch_size'], par['n_val']])
			mask   = tf.ones([par['batch_size'], 1])

			for j in range(par['num_time_steps']):

				# Make two possible actions and values for the network to pursue
				# by way of the LSTM-based cortex module and the associative
				# network hippocampus module
				t = i*par['num_time_steps'] + j

				h, c = self.cortex_lstm(self.stimulus_data[t], h, h_read, c)

				h_read, A = self.fast_weights(h, A)

				pol_out = h @ self.var_dict['W_pol'] + self.var_dict['b_pol']
				val_out = h @ self.var_dict['W_val'] + self.var_dict['b_val']

				# Compute outputs for action and policy loss
				action_index	= tf.multinomial(pol_out, 1)
				action 			= tf.one_hot(tf.squeeze(action_index), par['n_output'])
				pol_out			= tf.nn.softmax(pol_out, -1) # Note softmax for entropy calculation

				# Check for trial continuation (ends if previous reward is non-zero)
				continue_trial	= tf.cast(tf.equal(reward, 0.), tf.float32)
				mask 		   *= continue_trial
				reward			= tf.reduce_sum(action*self.reward_data[t,...], axis=-1, keep_dims=True) \
									* mask * self.time_mask[t,:,tf.newaxis]

				# Record outputs
				self.h.append(h)
				self.c.append(c)
				self.pol_out.append(pol_out)
				self.val_out.append(val_out)
				self.action.append(action)
				self.reward.append(reward)
				self.mask.append(mask)

		self.h = tf.stack(self.h, axis=0)
		self.c = tf.stack(self.c, axis=0)
		self.pol_out = tf.stack(self.pol_out, axis=0)
		self.val_out = tf.stack(self.val_out, axis=0)
		self.action = tf.stack(self.action, axis=0)
		self.reward = tf.stack(self.reward, axis=0)
		self.mask = tf.stack(self.mask, axis=0)


	def cortex_lstm(self, x, h, h_read, c):
		""" Compute LSTM state from inputs and vars...
				f : forgetting gate
				i : input gate
				c : cell state
				o : output gate
			...and generate an action from that state. """

		# Iterate LSTM
		f  = tf.sigmoid(x @ self.var_dict['Wf'] + h @ self.var_dict['Uf'] + h_read @ self.var_dict['Vf'] + self.var_dict['bf'])
		i  = tf.sigmoid(x @ self.var_dict['Wi'] + h @ self.var_dict['Ui'] + h_read @ self.var_dict['Vi'] + self.var_dict['bi'])
		o  = tf.sigmoid(x @ self.var_dict['Wo'] + h @ self.var_dict['Uo'] + h_read @ self.var_dict['Vo'] + self.var_dict['bo'])
		cn = tf.tanh(x @ self.var_dict['Wc'] + h @ self.var_dict['Uc'] + h_read @ self.var_dict['Vc'] + self.var_dict['bc'])
		c  = f * c + i * cn
		h  = o * tf.tanh(c)

		# Return action, hidden state, and cell state
		return h, c

	def fast_weights(self, h, A):

		h_write = tf.nn.relu(tf.tensordot(h, self.var_dict['W_write'], axes = [[1], [0]]))
		h_read = h_write

		for i in range(par['inner_steps']):
			h_read = tf.nn.relu(tf.reduce_sum(A * h_read, axis = -1, keep_dims = True))

		A = par['A_alpha']*A + par['A_beta']*h_write*tf.transpose(h_write, [0, 2, 1])

		return tf.squeeze(h_read), A


	def optimize(self):
		""" Calculate losses and apply corrections to model """

		# Set up optimizer and required constants
		epsilon = 1e-7
		cortex_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cortex')
		cortex_optimizer = AdamOpt.AdamOpt(cortex_vars, learning_rate=par['learning_rate'])

		# Spiking activity loss (penalty on high activation values in the hidden layer)
		"""
		self.spike_loss = par['spike_cost']*tf.reduce_mean(tf.stack([mask*time_mask*tf.reduce_mean(h) \
			for (h, mask, time_mask) in zip(tf.unstack(self.h), tf.unstack(self.mask), tf.unstack(self.time_mask))]))
		"""

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
		full_mask        = mask_static*self.time_mask*par['sequence_mask']

		# Policy loss
		self.pol_loss = -tf.reduce_mean(full_mask*advantage_static*action_static*tf.log(epsilon+self.pol_out))

		# Value loss
		self.val_loss = 0.5*par['val_cost']*tf.reduce_mean(full_mask*tf.square(val_out[:-1,:,:]-pred_val_static))

		# Entropy loss
		self.ent_loss = -par['entropy_cost']*tf.reduce_mean(tf.reduce_sum(full_mask*self.pol_out*tf.log(epsilon+self.pol_out), axis=2))

		# Collect RL losses
		RL_loss = self.pol_loss + self.val_loss - self.ent_loss

		# Collect loss terms and compute gradients
		total_loss = RL_loss
		self.train_cortex = cortex_optimizer.compute_gradients(total_loss)




def main(gpu_id=None):

	if gpu_id is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['n_input']], 'stim')
	r = tf.placeholder(tf.float32, [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['n_output']], 'reward')
	m = tf.placeholder(tf.float32, [par['num_time_steps']*par['trials_per_seq'], par['batch_size']], 'mask')

	stim = stimulus_sequence.Stimulus()

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8) if gpu_id == '0' else tf.GPUOptions()
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			model = Model(x, r, m)

		sess.run(tf.global_variables_initializer())

		print('\nGate value of 0 indicates using hippocampus (associative network).')
		print('Gate value of 1 indicates using cortex (LSTM).\n')

		for t in range(par['n_tasks']):
			print()
			for i in range(par['n_batches']):


				name, trial_info = stim.generate_trial(t)
				"""
				fig, ax = plt.subplots(2,1,figsize=(12,8),)
				ax[0].imshow(trial_info['desired_output'][:, 0, :].T, aspect = 'auto')
				ax[1].imshow(trial_info['reward_data'][:, 0, :].T, aspect = 'auto')
				plt.show()

				fig, ax = plt.subplots(2,1,figsize=(12,8))
				ax[0].imshow(trial_info['desired_output'][:, 1, :].T, aspect = 'auto')
				ax[1].imshow(trial_info['reward_data'][:, 1, :].T, aspect = 'auto')
				plt.show()
				"""

				feed_dict = {x:trial_info['neural_input'], r:trial_info['reward_data'],\
					m:trial_info['train_mask']}

				_, reward, pol_loss, action = \
					sess.run([model.train_cortex, model.reward, model.pol_loss, model.action], feed_dict=feed_dict)


				if i%5 == 0:

					print('Task {:>2} | Iter {:>4} | Reward: {:6.3f} | Pol. Loss: {:6.3f} |'.format(\
						t, i, np.mean(np.sum(reward, axis=0)), pol_loss))

	print('Model complete.\n')


if __name__ == '__main__':
	try:
		if len(sys.argv) > 1:
			main(sys.argv[1])
		else:
			main()
	except KeyboardInterrupt:
		quit('Quit by KeyboardInterrupt.')
