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
	'save_fn'				: '80_tasks_v1',
	'train'					: True,
	'save_weights'			: True,
	'learning_method'		: 'RL', # 'RL' or 'SL'
	'LSTM_init'				: 0.02,
	'w_init'				: 0.02,

	# Network shape
	'num_motion_tuned'		: 32,
	'num_fix_tuned'			: 1,
	'num_rule_tuned'		: 0,
	'n_hidden'				: 250,
	'n_val'					: 4,
	'n_modules'				: 4,
	'n_module_out'			: 10,
	'n_ff0'					: 200,
	'n_ff1'					: 200,
	'top_down'				: True,


	# Encoder configuration
	'n_latent'				: 200,
	'enc_activity_cost'		: 0.1,
	'enc_weight_cost'		: 0.05,
	'internal_sampling'		: True,

	# Cortex configuration
	'sample_step'			: 5,

	# read/write configuration
	'A_alpha_init'			: 0.99995,
	'A_beta_init'			: 1.5,
	'inner_steps'			: 1,
	'batch_norm_inner'		: False,

	# Timings and rates
	'learning_rate'			: 5e-4,
	'drop_rate'				: 0.5,

	# Variance values
	'input_mean'			: 0.0,
	'noise_in'				: 0.1,
	'noise_rnn'				: 0.05,
	'n_filters'				: 1,

	# Task specs
	'task'					: 'multistim',
	'trial_length'			: 2000,
	'mask_duration'			: 0,
	'dead_time'				: 200,
	'dt'					: 100,
	'trials_per_seq'		: 70,
	'task_list'				: [a for a in range(1,81)], #[a for a in range(49)],
	'dead_trials'			: 60,

	# RL parameters
	'fix_break_penalty'     : -1.,
	'wrong_choice_penalty'  : -0.01,
	'correct_choice_reward' : 1.,
	'discount_rate'         : 0.9,

	# Tuning function data
	'num_motion_dirs'		: 8,
	'tuning_height'			: 4.0,

	# Cost values
	'sparsity_cost'         : 1e-2, # was 1e-2
	'rec_cost'				: 1e-3,  # was 1e-2
	'weight_cost'           : 1e-5,  # was 1e-6
	'entropy_cost'          : 0.01,
	'val_cost'              : 0.01,
	'stim_cost'				: 1e-1,

	# Training specs
	'batch_size'			: 128,
	'n_batches'				: 3000000,		# 1500 to train straight cortex

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



def load_encoder_weights():

	fn = './savedir/80_tasks_v0_model_weights.pkl'
	results = pickle.load(open(fn, 'rb'))
	print('Weight keys ', results['weights'].keys())
	par['W0_init'] = results['weights']['W0']
	par['W1_init'] = results['weights']['W1']


def update_weights(var_dict):

	print('Setting weight values manually; disabling training and weight saving.')
	par['train'] = False
	par['save_weights'] = False
	for key, val in var_dict['weights'].items():
		print(key, val.shape)
		if not 'A_' in key:
			par[key+'_init'] = val


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
	par['n_pol'] = par['num_motion_dirs'] + 1

	# Set trial step length
	par['num_time_steps'] = par['trial_length']//par['dt']

	# Set up standard LSTM weights and biases
	LSTM_weight = lambda size : np.random.uniform(-par['LSTM_init'], par['LSTM_init'], size=size).astype(np.float32)

	# option 1
	#for p in ['Wf', 'Wi', 'Wo', 'Wc']: par[p+'_init'] = LSTM_weight([par['n_input']+par['n_hidden'], par['n_hidden']])
	#par['W_write_init'] = LSTM_weight([par['n_input']+par['n_val']+par['n_pol'], par['n_latent']])
	# option 2
	#for p in ['Wf', 'Wi', 'Wo', 'Wc']: par[p+'_init'] = LSTM_weight([par['n_input'], par['n_hidden'][0]])
	#for j in range(par['n_modules']):
	#	for p in ['Wf', 'Wi', 'Wo', 'Wc']: par[p+str(j)+'_init'] = LSTM_weight([par['n_input'], par['n_hidden'][j]])
	#	for p in ['Uf', 'Ui', 'Uo', 'Uc']: par[p+str(j)+'_init'] = LSTM_weight([par['n_hidden'][0], par['n_hidden'][j]])
	#	for p in ['bf', 'bi', 'bo', 'bc']: par[p+str(j)+'_init'] = np.zeros([1, par['n_hidden'][j]], dtype=np.float32)
	par['W0_init'] = np.random.uniform(-0.02, 0.02, size=[par['n_input']*par['n_filters'], par['n_ff0']]).astype(np.float32)
	#par['W1_init'] = np.random.uniform(-1., 1., size=[par['n_ff0'], par['n_ff1']]).astype(np.float32)
	par['W1_init'] = np.random.uniform(-0.02, 0.02, size=[par['n_ff0'], par['n_input']*par['n_filters']]).astype(np.float32)
	par['W2_init'] = np.random.uniform(-0.02, 0.02, size=[par['n_ff0'], par['n_ff1']]).astype(np.float32)
	par['W_td_init'] = np.random.uniform(-0.02, 0.02, size=[par['n_hidden'], par['n_ff0']]).astype(np.float32)
	par['b0_init'] = np.zeros([1, par['n_ff0']], dtype=np.float32)
	par['b2_init'] = np.zeros([1, par['n_ff1']], dtype=np.float32)

	# V0
	n_input_ctl = par['n_pol']*par['n_val'] + par['n_pol'] + par['n_val'] + par['n_input']*par['n_filters']
	#n_input_ctl = 33 + par['n_pol'] + par['n_val'] + par['n_pol']*par['n_val']
	# V1
	#n_input_ctl = par['n_module_out']*par['n_modules'] + par['n_pol'] + par['n_val'] + par['n_pol']*par['n_val']
	#n_input_ctl = par['n_input'] + par['n_module_out']*par['n_modules'] + par['n_pol']*par['n_val']

	for p in ['Wf', 'Wi', 'Wo', 'Wc']: par[p+'_init'] = LSTM_weight([n_input_ctl, par['n_hidden']])
	for p in ['Uf', 'Ui', 'Uo', 'Uc']: par[p+'_init'] = LSTM_weight([par['n_hidden'], par['n_hidden']])
	for p in ['bf', 'bi', 'bo', 'bc']: par[p+'_init'] = np.zeros([1, par['n_hidden']], dtype=np.float32)


	# LSTM posterior distribution weights
	#for p in ['Pf', 'Pi', 'Po', 'Pc']: par[p+'_init'] = LSTM_weight([par['n_tasks'], par['n_hidden']])

	# Cortex RL weights and biases
	par['W_pol_init'] = np.random.uniform(-par['w_init'], par['w_init'], size=[par['n_hidden'], par['n_pol']]).astype(np.float32)
	par['b_pol_init'] = np.zeros([1,par['n_pol']], dtype=np.float32)
	par['W_val_init'] = np.random.uniform(-par['w_init'], par['w_init'], size=[par['n_hidden'], 1]).astype(np.float32)
	par['b_val_init'] = np.zeros([1,1], dtype=np.float32)

	par['W_norm'] = np.zeros((par['n_ff1'], par['n_ff1']), dtype=np.float32)
	for i in range(par['n_ff1']):
		u = np.arange(i, i + 50)%par['n_ff1']
		par['W_norm'][i, u] = 1.



	# Gate weights and biases
	"""
	par['W_pos_gate_init'] = np.random.uniform(-par['w_init'], par['w_init'], size=[par['n_tasks'],1]).astype(np.float32)
	par['W_cor_gate_init'] = np.random.uniform(-par['w_init'], par['w_init'], size=[par['n_pol'],1]).astype(np.float32)
	par['W_hip_gate_init'] = np.random.uniform(-par['w_init'], par['w_init'], size=[par['n_pol'],1]).astype(np.float32)
	par['W_cor_gate_val_init'] = np.random.uniform(-par['w_init'], par['w_init'], size=[par['n_val'],1]).astype(np.float32)
	par['W_hip_gate_val_init'] = np.random.uniform(-par['w_init'], par['w_init'], size=[par['n_val'],1]).astype(np.float32)
	par['b_act_gate_init'] = np.ones([1,1], dtype=np.float32)
	par['b_pos_gate_init'] = np.ones([1,1], dtype=np.float32)

	par['encoder_weight_file'] = './datadir/gotask_50unit_input_encoder_weights.pkl'
	print('--> Loading encoder from {}.'.format(par['encoder_weight_file']))
	par['encoder_init'] = pickle.load(open(par['encoder_weight_file'], 'rb'))['weights']['W']
	"""

	load_encoder_weights()

update_dependencies()
print('--> Parameters successfully loaded.\n')
