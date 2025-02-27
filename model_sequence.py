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
import AdamOpt_sequence as AdamOpt
import time

# Match GPU IDs to nvidia-smi command
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class Model:

	def __init__(self, stimulus, reward_data, target_out, mask):

		print('Defining graph...')

		self.stimulus_data	= stimulus
		self.reward_data	= reward_data
		self.target_out     = target_out
		self.time_mask		= mask

		self.declare_variables()
		self.run_model()
		self.optimize()

		print('Graph successfully defined.\n')


	def declare_variables(self):

		self.var_dict = {}
		lstm_var_prefixes = ['Wf', 'Wi', 'Wo', 'Wc', 'Uf', 'Ui', 'Uo', 'Uc', \
			'bf', 'bi', 'bo', 'bc', 'W_write', 'W_read']
		RL_var_prefixes = ['W_pol', 'W_val', 'b_pol', 'b_val']

		with tf.variable_scope('cortex'):
			for p in lstm_var_prefixes + RL_var_prefixes:
				self.var_dict[p] = tf.get_variable(p, initializer=par[p+'_init'])


	def run_model(self):

		#self.h = []
		self.h_write = []
		self.h_hat = []
		self.h_concat = []
		self.pol_out = []
		self.val_out = []
		self.action  = []
		self.reward  = []
		self.mask    = []
		self.pol_out_raw  = []
		self.target = []

		h = tf.zeros([par['batch_size'], par['n_hidden']], dtype = tf.float32)
		h_read = tf.zeros([par['batch_size'], par['n_latent']], dtype = tf.float32)
		h_write = tf.zeros([par['batch_size'], par['n_latent']], dtype = tf.float32)
		c = tf.zeros([par['batch_size'], par['n_hidden']], dtype = tf.float32)
		A = tf.zeros([par['batch_size'], par['n_latent'], par['n_latent']], dtype = tf.float32)
		action = tf.zeros([par['batch_size'], par['n_pol']])


		for i in range(par['trials_per_seq']):

			reward = tf.zeros([par['batch_size'], par['n_val']])
			mask   = tf.ones([par['batch_size'], 1])

			for j in range(par['num_time_steps']):

				# Make two possible actions and values for the network to pursue
				# by way of the LSTM-based cortex module and the associative
				# network hippocampus module
				t = i*par['num_time_steps'] + j

				input = tf.concat([mask*self.stimulus_data[t], h_read], axis = 1)

				#input = tf.concat([mask*self.stimulus_data[t], h_read], axis = 1)
				h, c = self.cortex_lstm(input, h, c)
				#h += tf.random_normal(tf.shape(h), 0, 0.1)
				h = tf.layers.dropout(h, 0.25, noise_shape = [par['batch_size'], par['n_hidden']], training = True)

				salient = tf.cast(tf.not_equal(reward, tf.constant(0.)), tf.float32)
				h_concat = tf.concat([self.stimulus_data[t], h, reward, action], axis = 1)
				#h_concat = tf.concat([ h, reward, action], axis = 1)
				h_write = tf.tensordot(h_concat, self.var_dict['W_write'], axes = [[1], [0]])
				#h_write += tf.random_normal(tf.shape(h_write), 0, 0.1)
				h_write = tf.nn.relu(h_write)
				h_hat = tf.tensordot(h_write, self.var_dict['W_read'], axes = [[1], [0]])

				h_write = tf.reshape(h_write,[par['batch_size'], par['n_latent'], 1])
				h_read,  A = self.fast_weights(h_write, A, salient)
				#h_read *= 0

				pol_out = h @ self.var_dict['W_pol'] + self.var_dict['b_pol']
				val_out = h @ self.var_dict['W_val'] + self.var_dict['b_val']

				# Compute outputs for action and policy loss
				action_index	= tf.multinomial(pol_out, 1)
				action 			= tf.one_hot(tf.squeeze(action_index), par['n_pol'])
				pol_out_sm		= tf.nn.softmax(pol_out, -1) # Note softmax for entropy calculation

				# Check for trial continuation (ends if previous reward is non-zero)
				continue_trial  = tf.cast(tf.equal(reward, 0.), tf.float32)
				mask 		   *= continue_trial
				reward			= tf.reduce_sum(action*self.reward_data[t,...], axis=-1, keep_dims=True) \
									* mask * self.time_mask[t,:,tf.newaxis]

				# Record outputs
				if i >= par['dead_trials']: # discard the first ~5 trials
					#self.h.append(h)
					self.h_write.append(h_write)
					self.h_hat.append(h_hat)
					self.h_concat.append(h_concat)
					self.pol_out.append(pol_out_sm)
					self.pol_out_raw.append(pol_out)
					self.val_out.append(val_out)
					self.action.append(action)
					self.reward.append(reward)
					self.target.append(self.target_out[t, ...])
					self.mask.append(mask * self.time_mask[t,:,tf.newaxis])


		#self.h = tf.stack(self.h, axis=0)
		self.h_write = tf.stack(self.h_write, axis=0)
		self.h_hat = tf.stack(self.h_hat, axis=0)
		self.h_concat = tf.stack(self.h_concat, axis=0)
		self.pol_out = tf.stack(self.pol_out, axis=0)
		self.pol_out_raw = tf.stack(self.pol_out_raw, axis=0)
		self.val_out = tf.stack(self.val_out, axis=0)
		self.action = tf.stack(self.action, axis=0)
		self.reward = tf.stack(self.reward, axis=0)
		self.target = tf.stack(self.target, axis=0)
		self.mask = tf.stack(self.mask, axis=0)


	def cortex_lstm(self, x, h, c):
		""" Compute LSTM state from inputs and vars...
				f : forgetting gate
				i : input gate
				c : cell state
				o : output gate
			...and generate an action from that state. """

		# Iterate LSTM
		f  = tf.sigmoid(x @ self.var_dict['Wf'] + h @ self.var_dict['Uf'] + self.var_dict['bf'])
		i  = tf.sigmoid(x @ self.var_dict['Wi'] + h @ self.var_dict['Ui'] + self.var_dict['bi'])
		o  = tf.sigmoid(x @ self.var_dict['Wo'] + h @ self.var_dict['Uo'] + self.var_dict['bo'])
		cn = tf.tanh(x @ self.var_dict['Wc'] + h @ self.var_dict['Uc'] + self.var_dict['bc'])
		c  = f * c + i * cn
		h  = o * tf.tanh(c)

		# Return action, hidden state, and cell state
		return h, c

	def fast_weights(self, h, A, salient):

		#A_new = par['A_alpha']*A + par['A_beta']*(tf.reshape(salient,[par['batch_size'],1,1])*h)*tf.transpose(h, [0, 2, 1])*par['A_mask']
		A_new = par['A_alpha']*A + par['A_beta']*(tf.reshape(salient,[par['batch_size'],1,1])*h)*tf.transpose(h, [0, 2, 1])*par['A_mask']


		for i in range(par['inner_steps']):
			h = tf.reduce_sum(A * h, axis = -1, keep_dims = True)
			# layer normalization
			if par['batch_norm_inner']:
				u, v = tf.nn.moments(h, axes = [1], keep_dims = True)
				h = tf.nn.relu((h-u)/tf.sqrt(1e-9+v))
			else:
				h = tf.nn.relu(h)

		return tf.squeeze(h), A_new


	def optimize(self):
		""" Calculate losses and apply corrections to model """

		# Set up optimizer and required constants
		epsilon = 1e-7
		cortex_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cortex')
		cortex_optimizer = AdamOpt.AdamOpt(cortex_vars, learning_rate=par['learning_rate'])

		# Spiking activity loss (penalty on high activation values in the hidden layer)
		h_write = tf.reduce_mean(self.h_write,axis = 2)
		self.spike_loss = par['spike_cost']*tf.reduce_mean(self.mask*h_write)

		self.weight_loss = par['weight_cost']*tf.reduce_mean([tf.reduce_sum(tf.abs(var)) \
			for var in tf.trainable_variables() if ('W_write' in var.op.name or 'W_read' in var.op.name)])

		self.reconstruction_loss = par['rec_cost']*tf.reduce_mean(self.mask*tf.square(self.h_concat-self.h_hat))

		# Collect loss terms and compute gradients
		if par['learning_method'] == 'RL':
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
			# Policy loss
			self.pol_loss = -tf.reduce_mean(mask_static*advantage_static*action_static*tf.log(epsilon+self.pol_out))
			# Value loss
			self.val_loss = 0.5*par['val_cost']*tf.reduce_mean(mask_static*tf.square(val_out[:-1,:,:]-pred_val_static))
			# Entropy loss
			self.ent_loss = -par['entropy_cost']*tf.reduce_mean(tf.reduce_sum(mask_static*self.pol_out*tf.log(epsilon+self.pol_out), axis=2))
			# Collect RL losses
			RL_loss = self.pol_loss + self.val_loss - self.ent_loss
			total_loss = RL_loss + self.spike_loss + self.reconstruction_loss + self.weight_loss

		elif par['learning_method'] == 'SL':
			self.task_loss = tf.reduce_mean(tf.squeeze(self.mask)*tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.pol_out_raw, \
				labels = self.target, dim = -1))
			total_loss = self.task_loss + self.spike_loss + self.reconstruction_loss + 1e-15*self.val_loss + self.weight_loss
		if par['train']:
			self.train_cortex = cortex_optimizer.compute_gradients(total_loss)
		else:
			self.train_cortex = tf.no_op()



def main(gpu_id=None):

	if gpu_id is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

	print_important_params()

	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['n_input']], 'stim')
	r = tf.placeholder(tf.float32, [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['n_pol']], 'reward')
	y = tf.placeholder(tf.float32, [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['n_pol']], 'target')
	m = tf.placeholder(tf.float32, [par['num_time_steps']*par['trials_per_seq'], par['batch_size']], 'mask')

	stim = stimulus_sequence.Stimulus()

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8) if gpu_id == '0' else tf.GPUOptions()
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			model = Model(x, r, y, m)

		sess.run(tf.global_variables_initializer())

		for t in range(par['n_tasks']):
			for i in range(par['n_batches']):

				name, trial_info = stim.generate_trial(t)
				feed_dict = {x:trial_info['neural_input'], r:trial_info['reward_data'],\
					y: trial_info['desired_output'], m:trial_info['train_mask']}

				_, reward, pol_loss, action, h, h_write, rec_loss = \
					sess.run([model.train_cortex, model.reward, model.pol_loss, \
					model.action, model.h, model.h_write, model.reconstruction_loss], feed_dict=feed_dict)

			if i%20 == 0:
				print('Iter {:>4} | Reward: {:6.3f} | Pol. Loss: {:6.3f} | Mean h_w: {:6.6f}  | Rec. loos: {:6.5f}  |'.format(\
					i, running_avg_reward, pol_loss, np.mean(h_write), np.mean(rec_loss)))

			if par['save_weights'] and i%500 == 0:
				t0 = time.time()
				print('Saving weights...')
				weights, = sess.run([model.var_dict])
				saved_data = {'weights':weights, 'par': par}
				pickle.dump(saved_data, open('./savedir/{}_model_weights.pkl'.format(par['save_fn']), 'wb'))
				print('Weights saved.\n')
				print('Time ', time.time() - t0)

	print('Model complete.\n')


def print_important_params():

	notes = ''

	keys = ['learning_method', 'n_hidden', 'n_latent', \
		'A_alpha', 'A_beta', 'inner_steps', 'batch_norm_inner', 'learning_rate', \
		'task_list', 'trials_per_seq', 'fix_break_penalty', 'wrong_choice_penalty', \
		'correct_choice_reward', 'discount_rate', 'num_motion_dirs', 'spike_cost', \
		'rec_cost', 'weight_cost', 'entropy_cost', 'val_cost', 'batch_size', 'n_batches']

	print('-'*60)
	[print('{:<24} : {}'.format(k, par[k])) for k in keys]
	print('{:<24} : {}'.format('notes', notes))
	print('-'*60 + '\n')



if __name__ == '__main__':
	try:
		if len(sys.argv) > 1:
			main(sys.argv[1])
		else:
			main()
	except KeyboardInterrupt:
		quit('Quit by KeyboardInterrupt.')
