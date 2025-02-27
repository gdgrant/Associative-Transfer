### Authors: Nicolas Y. Masse, Gregory D. Grant

import numpy as np
import pickle

print('\n--> Loading parameters...')

##############################
### Independent parameters ###
##############################

par = {
	# Setup parameters
	'savedir'				: './savedir/',
	'LSTM_init'				: 0.05,
	'w_init'				: 0.05,

	# Network shape
	'num_motion_tuned'		: 64,
	'num_fix_tuned'			: 4,
	'num_rule_tuned'		: 0,
	'n_hidden'				: 50,
	'n_linear'				: 512,
	'n_val'					: 1,

	# Encoder configuration
	'n_latent'				: 50,
	'enc_activity_cost'		: 0.1,
	'enc_weight_cost'		: 0.05,
	'internal_sampling'		: False,

	# Cortex configuration
	'sample_step'			: 5,

	# Hippocampus configuration
	'test_sample_prop'		: 0.2,
	'train_alpha'			: 1.,
	'train_beta'			: 0.1,
	'train_gamma'			: 4.,
	'associative_iters'		: 5,

	# Timings and rates
	'dt'					: 20,
	'learning_rate'			: 5e-4,

	# Variance values
	'input_mean'			: 0.0,
	'noise_in'				: 0.05,
	'noise_rnn'				: 0.05,

	# Task specs
	'task'					: 'go',
	'n_tasks'				: 2,
	'trial_length'			: 2000,
	'mask_duration'			: 0,
	'dead_time'				: 200,

	# RL parameters
	'fix_break_penalty'     : -1.,
	'wrong_choice_penalty'  : -0.01,
	'correct_choice_reward' : 1.,
	'discount_rate'         : 0.,

	# Tuning function data
	'num_motion_dirs'		: 8,
	'tuning_height'			: 4.0,

	# Cost values
	'gate_cost'				: 1e-3,
	'spike_cost'            : 0.,
	'weight_cost'           : 0.,
	'entropy_cost'          : 0.001,
	'val_cost'              : 0.01,

	# Training specs
	'batch_size'			: 256,
	'n_batches'				: 1000000,		# 1500 to train straight cortex

}


############################
### Dependent parameters ###
############################

def update_parameters(updates, verbose=True, update_deps=True):
	""" Updates parameters based on a provided
		dictionary, then updates dependencies """

	par.update(updates)
	if verbose:
		print('Updating parameters:')
		for (key, val) in updates.items():
			print('{:<24} --> {}'.format(key, val))

	if update_deps:
		update_dependencies()


def update_dependencies():
	""" Updates all parameter dependencies """

	# Reward map, for hippocampus reward one-hot conversion
	par['reward_map'] = {
		par['fix_break_penalty']		: 0,
		par['wrong_choice_penalty']		: 1,
		0.								: 2,
		par['correct_choice_reward']	: 3
	}

	par['num_reward_types'] = len(par['reward_map'].keys())
	par['reward_map_matrix'] = np.zeros([par['num_reward_types'],1]).astype(np.float32)
	for key, val in par['reward_map'].items():
		par['reward_map_matrix'][val,:] = key

	# Set input and output sizes
	par['n_input']  = par['num_motion_tuned'] + par['num_fix_tuned'] + par['num_rule_tuned']
	par['n_output'] = par['num_motion_dirs'] + 1
	par['n_assoc']  = par['n_tasks'] + par['n_latent'] +  par['n_output'] + par['num_reward_types']

	# Set trial step length
	par['num_time_steps'] = par['trial_length']//par['dt']

	# Set up standard LSTM weights and biases
	LSTM_weight = lambda size : np.random.uniform(-par['LSTM_init'], par['LSTM_init'], size=size).astype(np.float32)
	for p in ['Wf', 'Wi', 'Wo', 'Wc']: par[p+'_init'] = LSTM_weight([par['n_input'], par['n_hidden']])
	for p in ['Uf', 'Ui', 'Uo', 'Uc']: par[p+'_init'] = LSTM_weight([par['n_hidden'], par['n_hidden']])
	for p in ['bf', 'bi', 'bo', 'bc']: par[p+'_init'] = np.zeros([1, par['n_hidden']], dtype=np.float32)

	# LSTM posterior distribution weights
	for p in ['Pf', 'Pi', 'Po', 'Pc']: par[p+'_init'] = LSTM_weight([par['n_tasks'], par['n_hidden']])
	
	# Cortex RL weights and biases
	par['W_out_init'] = np.random.uniform(-par['w_init'], par['w_init'], size=[par['n_hidden'], par['n_output']]).astype(np.float32)
	par['b_out_init'] = np.zeros([1,par['n_output']], dtype=np.float32)

	par['W_val_init'] = np.random.uniform(-par['w_init'], par['w_init'], size=[par['n_hidden'], par['n_val']]).astype(np.float32)
	par['b_val_init'] = np.zeros([1,par['n_val']], dtype=np.float32)

	# Gate weights and biases
	par['W_pos_gate_init'] = np.random.uniform(-par['w_init'], par['w_init'], size=[par['n_tasks'],1]).astype(np.float32)
	par['W_cor_gate_init'] = np.random.uniform(-par['w_init'], par['w_init'], size=[par['n_output'],1]).astype(np.float32)
	par['W_hip_gate_init'] = np.random.uniform(-par['w_init'], par['w_init'], size=[par['n_output'],1]).astype(np.float32)
	par['W_cor_gate_val_init'] = np.random.uniform(-par['w_init'], par['w_init'], size=[par['n_val'],1]).astype(np.float32)
	par['W_hip_gate_val_init'] = np.random.uniform(-par['w_init'], par['w_init'], size=[par['n_val'],1]).astype(np.float32)
	par['b_act_gate_init'] = np.ones([1,1], dtype=np.float32)
	par['b_pos_gate_init'] = np.ones([1,1], dtype=np.float32)

	par['encoder_weight_file'] = './datadir/gotask_50unit_input_encoder_weights.pkl'
	print('--> Loading encoder from {}.'.format(par['encoder_weight_file']))
	par['encoder_init'] = pickle.load(open(par['encoder_weight_file'], 'rb'))['weights']['W']

update_dependencies()
print('--> Parameters successfully loaded.\n')