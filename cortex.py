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


class Cortex:

	def __init__(self, input_data, target_data, mask):

		self.input_data = input_data
		self.target_data = target_data
		self.time_mask = mask

		self.declare_variables()
		self.run_model()
		self.optimize()

		print('Cortex graph successfully defined.')


	def declare_variables(self):

		lstm_var_prefixes = ['Wf', 'Wi', 'Wo', 'Wc', 'Uf', 'Ui', 'Uo', 'Uc', 'bf', 'bi', 'bo', 'bc']
		base_var_prefixes = ['W_out', 'b_out', 'W_val', 'b_val']

		# Current architecture is assumed to be LSTM
		prefix_list = base_var_prefixes + lstm_var_prefixes

		with tf.variable_scope('network'):
			self.var_dict = {p:tf.get_variable(p, initializer=par[p+'_init']) for p in prefix_list}


	def run_model(self):

		self.h = []
		self.c = []
		self.pol_out = []
		self.val_out = []
		self.action  = []
		self.reward  = []
		self.mask    = []

		h = tf.zeros([par['batch_size'], par['n_hidden']])
		c = tf.zeros([par['batch_size'], par['n_hidden']])

		reward = tf.zeros([par['batch_size'], par['n_val']])
		mask   = tf.ones([par['batch_size'], 1])

		for t in range(par['num_time_steps']):

			# Run recurrent cell
			h, c = self.lstm_cell(self.input_data[t], h, c)

			# Compute outputs for action
			pol_out			= h @ self.var_dict['W_out'] + self.var_dict['b_out']
			action_index	= tf.multinomial(pol_out, 1)
			action 			= tf.one_hot(tf.squeeze(action_index), par['n_output'])

			# Compute outputs for loss
			pol_out			= tf.nn.softmax(pol_out, -1) # Note softmax for entropy loss)
			val_out			= h @ self.var_dict['W_val'] + self.var_dict['b_val']

			# Check for trial continuation (ends if previous reward is non-zero)
			continue_trial	= tf.cast(tf.equal(reward, 0.), tf.float32)
			mask 		   *= continue_trial
			reward			= tf.reduce_sum(action*self.target_data[t,...], axis=-1, keepdims=True) \
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


	def lstm_cell(self, x, h, c):
		""" Compute LSTM state from inputs and vars
			f : forgetting gate
			i : input gate
			c : cell state
			o : output gate	"""

		f  = tf.sigmoid(x @ self.var_dict['Wf'] + h @ self.var_dict['Uf'] + self.var_dict['bf'])
		i  = tf.sigmoid(x @ self.var_dict['Wi'] + h @ self.var_dict['Ui'] + self.var_dict['bi'])
		cn = tf.tanh(x @ self.var_dict['Wc'] + h @ self.var_dict['Uc'] + self.var_dict['bc'])
		c  = f * c + i * cn
		o  = tf.sigmoid(x @ self.var_dict['Wo'] + h @ self.var_dict['Uo'] + self.var_dict['bo'])
		h  = o * tf.tanh(c)

		return h, c


	def optimize(self):
		""" Calculate losses and apply corrections to model """

		# Set up optimizer and required constants
		epsilon = 1e-7
		adam_optimizer = AdamOpt.AdamOpt(tf.trainable_variables(), learning_rate=par['learning_rate'])

		# Spiking activity loss (penalty on high activation values in the hidden layer)
		self.spike_loss = par['spike_cost']*tf.reduce_mean(tf.stack([mask*time_mask*tf.reduce_mean(h) \
			for (h, mask, time_mask) in zip(tf.unstack(self.h), tf.unstack(self.mask), tf.unstack(self.time_mask))]))

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
		full_mask        = mask_static*self.time_mask

		# Policy loss
		self.pol_loss = -tf.reduce_mean(full_mask*advantage_static*action_static*tf.log(epsilon+self.pol_out))

		# Value loss
		self.val_loss = 0.5*par['val_cost']*tf.reduce_mean(full_mask*tf.square(val_out[:-1,:,:]-pred_val_static))

		# Entropy loss
		self.ent_loss = -par['entropy_cost']*tf.reduce_mean(tf.reduce_sum(full_mask*self.pol_out*tf.log(epsilon+self.pol_out), axis=2))

		# Collect RL losses
		RL_loss = self.pol_loss + self.val_loss - self.ent_loss

		# Collect loss terms and compute gradients
		total_loss = RL_loss + self.spike_loss
		self.train_op = adam_optimizer.compute_gradients(total_loss)


def main(gpu_id=None):

	if gpu_id is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'], par['n_input']], 'stim')
	y = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'], par['n_output']], 'out')
	m = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size']], 'mask')
	
	stim = stimulus.Stimulus()

	with tf.Session() as sess:

		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			cortex = Cortex(x, y, m)

		sess.run(tf.global_variables_initializer())

		event_data = {'stimuli':[], 'actions':[], 'rewards':[]}
		print('\nTraining cortex for {} task:'.format(par['task']))
		for i in range(par['n_batches']):

			t0 = time.time()

			name, trial_info = stim.generate_trial()
			feed_dict = {x:trial_info['neural_input'], y:trial_info['reward_data'], m:trial_info['train_mask']}

			_, pol_loss, val_loss, spike_loss, ent_loss, h, reward, action = \
				sess.run([cortex.train_op, cortex.pol_loss, cortex.val_loss, cortex.spike_loss, \
					cortex.ent_loss, cortex.h, cortex.reward, cortex.action], feed_dict=feed_dict)

			# Select appropriate events
			inds_a = np.where(reward != 0.)
			inds_b = np.where(reward == 0.)
			print(len(inds_a), inds_a[0].shape)
			print(len(inds_b), inds_b[0].shape)
			quit()
			event_stimuli = trial_info['neural_input'][inds[0],inds[1],:]
			event_actions = action[inds[0],inds[1],:]
			event_rewards = reward[inds[0],inds[1],:]

			# Sample from events and save the event data
			event_data['stimuli'].append(event_stimuli[::par['sample_step'],:])
			event_data['actions'].append(event_actions[::par['sample_step'],:])
			event_data['rewards'].append(event_rewards[::par['sample_step'],:])

			if i%100 == 0:
				print('Iter: {:>4} | Rew: {:6.3f} | Pol. Loss: {:8.5f} | Val. Loss: {:8.5f} | Ent. Loss: {:8.5f} | Spiking: {:8.5f} |'.format(\
					i, np.mean(np.sum(reward, axis=0)), pol_loss, val_loss, ent_loss, np.mean(h)))

	for (key, val) in event_data.items(): event_data[key] = np.concatenate(val, axis=0)
	pickle.dump(event_data, open('./datadir/{}task_cortex_event_data.pkl'.format(par['task']), 'wb'))
	print('Event samples saved.  Model complete. \n')


if __name__ == '__main__':
	try:
		if len(sys.argv) > 1:
			main(sys.argv[1])
		else:
			main()
	except KeyboardInterrupt:
		quit('Quit by KeyboardInterrupt.')
