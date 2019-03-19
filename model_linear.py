### Authors: Nicolas Y. Masse, Gregory D. Grant

# Required packages
import tensorflow as tf
import numpy as np
import pickle
import os, sys, time
from itertools import product

# Plotting suite
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model modules
from parameters_linear import *
import stimulus_sequence
import AdamOpt_sequence as AdamOpt
import time

# Match GPU IDs to nvidia-smi command
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class Model:

	def __init__(self, stimulus, reward_data, reward_matrix, target_out, mask):

		print('Defining graph...')

		self.stimulus_data	= stimulus
		self.reward_data	= reward_data
		self.reward_matrix	= reward_matrix
		self.target_out     = target_out
		self.time_mask		= mask

		self.declare_variables()
		self.run_model()
		self.optimize()

		print('Graph successfully defined.\n')


	def declare_variables(self):

		self.var_dict = {}
		#ff_prefixes   = ['W0', 'b0','W1','b1', 'W_td'] if par['top_down'] else \
		ff_prefixes   = ['W0', 'W1', 'b0'] if par['top_down'] else \
			['W0', 'b0','W1','b1', 'W_rnn']
		ff_prefixes2  = ['W2', 'b2']
		lstm_prefixes = ['Wf', 'Wi', 'Wo', 'Wc', 'Uf', 'Ui', 'Uo', 'Uc', \
			'bf', 'bi', 'bo', 'bc']
		module_prefixes = ['Y', 'bY', 'Xp']
		RL_prefixes = ['W_pol', 'W_val', 'b_pol', 'b_val']
		hyperparams = ['A_alpha', 'A_beta']

		with tf.variable_scope('FF'):
			for p in ff_prefixes:
				self.var_dict[p] = tf.get_variable(p, initializer = par[p + '_init'])
		with tf.variable_scope('FF2'):
			for p in ff_prefixes2:
				self.var_dict[p] = tf.get_variable(p, initializer = par[p + '_init'])
		with tf.variable_scope('LSTM'):
			for p in lstm_prefixes + RL_prefixes:
				self.var_dict[p] = tf.get_variable(p, initializer = par[p + '_init'])


	def run_model(self):

		self.y = []
		self.pol_out = []
		self.val_out = []
		self.action  = []
		self.reward  = []
		self.mask    = []
		self.pol_out_raw  = []
		self.target = []
		self.lstm_out = []
		self.lstm_action = []
		self.stim = []
		self.stim_hat = []

		self.A = []
		self.h_read = []

		x = tf.zeros([par['batch_size'], par['n_ff0']], dtype = tf.float32)
		h = tf.zeros([par['batch_size'], par['n_hidden']], dtype = tf.float32)
		c = tf.zeros([par['batch_size'], par['n_hidden']], dtype = tf.float32)
		A = tf.zeros([par['batch_size'], par['n_ff0'], par['n_pol'], par['n_val']], dtype = tf.float32)

		action = tf.zeros([par['batch_size'], par['n_pol']])

		self.stim_err = [[] for _ in range(par['n_modules'])]

		#W0 = self.var_dict['W0']

		for i in range(par['trials_per_seq']):

			reward			= tf.zeros([par['batch_size'], 1])
			reward_matrix 	= tf.zeros([par['batch_size'], par['n_val']])
			mask   			= tf.ones([par['batch_size'], 1])

			for k in range(par['num_time_steps']):

				t = i*par['num_time_steps'] + k
				stim = self.stimulus_data[t]
				y = tf.nn.relu(stim @ self.var_dict['W0'] + self.var_dict['b0'])
				stim_hat = y @ self.var_dict['W1']

				#z = tf.nn.relu(y @ self.var_dict['W2'] + self.var_dict['b2'])
				#z = tf.reshape(z, [par['batch_size'], par['n_ff0'], par['n_ff1']//par['n_ff0']])
				#z = tf.reduce_prod(z, axis = -1)
				#z = tf.min(z, axis = -1)


				h_read = self.read_fast_weights(A, y)
				h_read = tf.stop_gradient(h_read)

				ctl_input = tf.concat([stim, h_read, reward_matrix, action], axis = 1)
				#ctl_input = tf.concat([mask*self.stimulus_data[t], tf.stop_gradient(h_read), reward_matrix, action], axis = 1)

				h, c = self.cortex_lstm(ctl_input, h, c, '')
				h = tf.layers.dropout(h, rate = par['drop_rate'], training = True)

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
				reward_matrix	= tf.reduce_sum(action[...,tf.newaxis]*self.reward_matrix[t,...], axis=-2, keep_dims=False) \
									* mask * self.time_mask[t,:,tf.newaxis]

				A = self.write_fast_weights(A, mask*y, action, reward_matrix)


				# Record outputs
				if i >= par['dead_trials']: # discard the first ~5 trials
					self.pol_out.append(pol_out_sm)
					self.pol_out_raw.append(pol_out)
					self.val_out.append(val_out)
					self.action.append(action)
					self.reward.append(reward)
					self.target.append(self.target_out[t, ...])
					self.mask.append(mask * self.time_mask[t,:,tf.newaxis])
					self.h_read.append(h_read)
					self.y.append(y)
					self.stim.append(stim)
					self.stim_hat.append(stim_hat)



		self.pol_out = tf.stack(self.pol_out, axis=0)
		self.pol_out_raw = tf.stack(self.pol_out_raw, axis=0)
		self.val_out = tf.stack(self.val_out, axis=0)
		self.action = tf.stack(self.action, axis=0)
		self.reward = tf.stack(self.reward, axis=0)
		self.target = tf.stack(self.target, axis=0)
		self.mask = tf.stack(self.mask, axis=0)
		self.h_read  = tf.stack(self.h_read, axis=1)
		self.y = tf.stack(self.y, axis=0)
		self.stim = tf.stack(self.stim, axis=0)
		self.stim_hat = tf.stack(self.stim_hat, axis=0)




	def cortex_lstm(self, x, h, c, suffix):
		""" Compute LSTM state from inputs and vars...
				f : forgetting gate
				i : input gate
				c : cell state
				o : output gate
			...and generate an action from that state. """

		#print('x', x)
		#print('Wf', self.var_dict['Wf'+suffix])

		# Iterate LSTM
		f  = tf.sigmoid(x @ self.var_dict['Wf'+suffix] + h @ self.var_dict['Uf'+suffix] + self.var_dict['bf'+suffix])
		i  = tf.sigmoid(x @ self.var_dict['Wi'+suffix] + h @ self.var_dict['Ui'+suffix] + self.var_dict['bi'+suffix])
		o  = tf.sigmoid(x @ self.var_dict['Wo'+suffix] + h @ self.var_dict['Uo'+suffix] + self.var_dict['bo'+suffix])
		cn = tf.tanh(x @ self.var_dict['Wc'+suffix] + h @ self.var_dict['Uc'+suffix] + self.var_dict['bc'+suffix])
		c  = f * c + i * cn
		h  = o * tf.tanh(c)

		# Return action, hidden state, and cell state
		return h, c

	def read_fast_weights(self, A, h):

		# can we think of h as a probability over states?
		value_probe = tf.einsum('ijkm,ij->ikm', A, h)
		return tf.reshape(value_probe, [par['batch_size'], par['n_pol']*par['n_val']])


	def write_fast_weights(self, A, h, a, r):

		return par['A_alpha_init']*A + par['A_beta_init']* \
			tf.einsum('im,ijk->ijkm', r, tf.einsum('ij,ik->ijk', h, a))


	def optimize(self):
		""" Calculate losses and apply corrections to model """

		# Set up optimizer and required constants
		epsilon = 1e-5
		lstm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='LSTM')
		ff_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='FF')

		adam_optimizer = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])

		#spike_loss = tf.reduce_sum(tf.square(self.lstm_out))

		# Collect loss terms and compute gradients
		if par['learning_method'] == 'RL':
			# Get the value outputs of the network, and pad the last time step
			val_out = tf.concat([self.val_out, tf.zeros([1,par['batch_size'],1])], axis=0)
			# Determine terminal state of the network
			terminal_state = tf.cast(tf.logical_not(tf.equal(self.reward, tf.constant(0.))), tf.float32)
			# Compute predicted value and the advantage for plugging into the policy loss
			pred_val = self.reward + par['discount_rate']*val_out[1:,:,:]*(1-terminal_state)
			advantage = pred_val - val_out[:-1,:,:]
			# Stop gradients back through action, advantage, and mask

			advantage_static = tf.stop_gradient(advantage)
			mask_static      = tf.stop_gradient(self.mask)
			pred_val_static  = tf.stop_gradient(pred_val)
			# Policy loss
			action_static    = tf.stop_gradient(self.action)
			self.pol_loss = -tf.reduce_mean(mask_static*advantage_static*action_static*tf.log(epsilon+self.pol_out))
			# Value loss
			self.val_loss = 0.5*tf.reduce_mean(mask_static*tf.square(val_out[:-1,:,:]-pred_val_static))
			# Entropy loss
			self.ent_loss = -tf.reduce_mean(tf.reduce_sum(mask_static*self.pol_out*tf.log(epsilon+\
				self.pol_out), axis=2))


		elif par['learning_method'] == 'SL':
			self.task_loss = tf.reduce_mean(tf.squeeze(self.mask)*tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.pol_out_raw, \
				labels = self.target, dim = -1))
			total_loss = self.task_loss

		if par['train']:
			train_ops = []
			print('HR ', self.h_read)
			print('Y ', self.y)
			print('STIM ', self.stim)
			print('MASK ', self.mask)

			"""
			M = par['trials_per_seq'] - par['dead_trials']
			h_read = tf.reshape(self.h_read,[par['batch_size'], par['num_time_steps']*M, par['n_pol'], par['n_val']])
			h_read = tf.transpose(h_read, [1,0,2,3])
			h_read += epsilon
			state_count = tf.reduce_sum(h_read, axis = -1, keepdims = True)
			h_read /= state_count
			self.entropy_loss = -tf.reduce_mean(tf.reduce_sum(h_read*tf.log(h_read), axis = -1), axis = -1, keepdims = True)
			"""




			y = tf.reshape(self.y, [-1, par['n_ff0']])
			self.reconstruction_loss = tf.reduce_mean(tf.square(self.stim - self.stim_hat))
			self.weight_loss = tf.reduce_mean(tf.abs(self.var_dict['W0'])) + tf.reduce_mean(tf.abs(self.var_dict['W1']))
			self.sparsity_loss = tf.reduce_mean(tf.transpose(y) @ y)
			ff_loss = self.reconstruction_loss + par['sparsity_cost']*self.sparsity_loss \
				+ par['weight_cost']*self.weight_loss
			train_ops.append(adam_optimizer.minimize(ff_loss, var_list = ff_vars))

			train_ops.append(adam_optimizer.minimize(self.pol_loss + par['val_cost']*self.val_loss \
				- par['entropy_cost']*self.ent_loss, var_list = lstm_vars))

			self.train_cortex = tf.group(*train_ops)

			#normalize_weights = tf.assign(self.var_dict['W0'], self.var_dict['W0']/0./ \
			#	tf.reduce_sum(self.var_dict['W0']**2, axis = 0, keepdims = True))
			#normalize_weights = tf.assign(ff_vars[0], ff_vars[0]/ \
			#	tf.reduce_sum(tf.abs(ff_vars[0]), axis = 0, keepdims = True))
			#self.normalize_weights = tf.group([normalize_weights])
		else:
			self.train_cortex = tf.no_op()



def main(gpu_id=None):

	if gpu_id is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

	print_important_params()
	t0 = time.time()

	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['n_filters']*par['n_input']], 'stim')
	#x = tf.placeholder(tf.float32, [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], 33], 'stim')
	r = tf.placeholder(tf.float32, [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['n_pol']], 'reward')
	rm = tf.placeholder(tf.float32, [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['n_pol'], par['n_val']], 'reward_matrix')
	y = tf.placeholder(tf.float32, [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['n_pol']], 'target')
	m = tf.placeholder(tf.float32, [par['num_time_steps']*par['trials_per_seq'], par['batch_size']], 'mask')

	stim = stimulus_sequence.Stimulus()

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8) if gpu_id == '0' else tf.GPUOptions()
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			model = Model(x, r, rm, y, m)

		sess.run(tf.global_variables_initializer())

		for i in range(par['n_batches']):

			name, trial_info = stim.generate_trial()
			"""
			input0 = trial_info['neural_input'][:,:,-1][...,np.newaxis]*trial_info['neural_input'][:,:,:16]
			input1 = (1 - trial_info['neural_input'][:,:,-1][...,np.newaxis])*trial_info['neural_input'][:,:,:16]
			input2 = trial_info['neural_input'][:,:,-1][...,np.newaxis]*(1-(np.sum(trial_info['neural_input'][:,:,:16],axis=2,keepdims=True)>0))
			n_input = np.concatenate([input0, input1, input2], axis = -1)

			plt.imshow(n_input[:, 0, :], aspect = 'auto')
			plt.show()
			plt.imshow(np.sum(n_input,axis=2), aspect = 'auto')
			plt.colorbar()
			plt.show()
			"""

			feed_dict = {x:trial_info['neural_input_filtered'], r:trial_info['reward_data'],\
				rm:trial_info['reward_matrix'], y: trial_info['desired_output'], \
				m:trial_info['train_mask']}


			_, reward, action, h, recon_loss, weight_loss, sparsity_loss = \
				sess.run([model.train_cortex, model.reward, \
				model.action, model.y, model.reconstruction_loss, model.weight_loss, \
				model.sparsity_loss], feed_dict=feed_dict)


			reward = np.mean(np.sum(reward, axis=0))
			stim_loss = 0.

			if i%20 == 0:
				print('Iter {:>4} | Reward: {:6.3f} | Rec Loss: {:6.6f} | Weight Loss: {:6.6f} | Sparsity Loss: {:6.6f}  | Mean h^2: {:6.6f}'.format(\
					i, reward, np.mean(recon_loss), np.mean(weight_loss), np.mean(sparsity_loss), np.mean(h**2)))
				w = sess.run([model.var_dict])
				print('Mean W) ', np.mean(w[0]['W0']**2))

				fig, ax = plt.subplots(2, 2, figsize=[6,6])
				ax[0,0].imshow(h[:,0,:].T, aspect = 'auto')
				ax[0,1].imshow(h[:,1,:].T, aspect = 'auto')
				ax[1,0].imshow(h[:,2,:].T, aspect = 'auto')
				ax[1,1].imshow(h[:,3,:].T, aspect = 'auto')

				plt.savefig('./savedir/summary_comp3_iter{}.png'.format(i))
				plt.clf()
				plt.close()

			if par['save_weights'] and i%100 == 0:

				print('Saving weights...')
				weights, = sess.run([model.var_dict])
				saved_data = {'weights':weights, 'par': par}
				pickle.dump(saved_data, open('./savedir/{}_model_weights.pkl'.format(par['save_fn']), 'wb'))
				print('Weights saved.\n')
				print('Time ', time.time() - t0)
				t0 = time.time()

	print('Model complete.\n')


def print_important_params():

	notes = ''

	keys = ['learning_method', 'n_hidden', 'n_latent', 'noise_in','noise_rnn','top_down',\
		'A_alpha_init', 'A_beta_init', 'inner_steps', 'batch_norm_inner', 'learning_rate', \
		'task_list', 'trials_per_seq', 'fix_break_penalty', 'wrong_choice_penalty', \
		'correct_choice_reward', 'discount_rate', 'num_motion_dirs', 'sparsity_cost', 'n_filters', \
		'rec_cost', 'weight_cost', 'entropy_cost', 'val_cost', 'drop_rate', 'batch_size', 'n_batches', 'save_fn']

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
