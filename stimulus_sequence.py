### Authors: Nicolas Y. Masse, Gregory D. Grant

import numpy as np
from parameters_sequence import par

class Stimulus:

	def __init__(self, analysis=False):

		# Stimulus shapes
		self.input_shape    = [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['n_input'] ]
		self.output_shape   = [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['n_pol'] ]
		self.stimulus_shape = [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['num_motion_tuned'] ]
		self.response_shape = [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['num_motion_dirs'] ]
		self.fixation_shape = [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['num_fix_tuned'] ]
		self.rule_shape		= [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['num_rule_tuned'] ]
		self.mask_shape     = [par['num_time_steps']*par['trials_per_seq'], par['batch_size']]


		# Motion information
		self.modality_size    	= par['num_motion_tuned']//2
		self.motion_dirs	  	= np.arange(0,2*np.pi,2*np.pi/par['num_motion_dirs'])
		self.stimulus_dirs	  	= np.arange(0,2*np.pi,2*np.pi/self.modality_size)
		self.pref_motion_dirs 	= self.stimulus_dirs[:,np.newaxis]

		# Configurations
		self.dm_c_set 	  = np.array([-0.4, -0.2, -0.1, 0.1, 0.2, 0.4])
		self.dm_dly_c_set = np.array([-0.4, -0.2, -0.1, 0.1, 0.2, 0.4])

		# Timings (ms)
		self.fix_time = 400

		if not analysis:
			self.go_delay 		 	= np.array([200, 400, 800])//par['dt']
			self.dm_stim_lengths	= np.array([200, 400, 800])//par['dt']
			self.dm_dly_delay 	 	= np.array([200, 400, 800])//par['dt']
			self.match_delay 	 	= np.array([200, 400, 800])//par['dt']
		else:
			self.go_delay 		 	= np.array([400])//par['dt']
			self.dm_stim_lengths 	= np.array([400])//par['dt']
			self.dm_dly_delay 	 	= np.array([400])//par['dt']
			self.match_delay 	 	= np.array([400])//par['dt']

		# Initialize task interface
		self.get_tasks()


	def circ_tuning(self, theta):
		ang_dist = np.angle(np.exp(1j*theta - 1j*self.pref_motion_dirs))
		return par['tuning_height']*np.exp(-0.5*(8*ang_dist/np.pi)**2)


	def rule_tuning(self, task):
		if par['num_rule_tuned'] >= len(self.task_types):
			tuning = np.zeros(self.input_shape, dtype=np.float32)
			tuning[:,:,-par['num_rule_tuned']+task] = par['tuning_height']
			return tuning
		else:
			raise Exception('Use more rule neurons than task types.')


	def get_tasks(self):

		if par['task'] == 'go':
			self.task_types = [
				[self.task_go, 'go', 0],
				[self.task_go, 'go', np.pi]
			]

		elif par['task'] == 'dms':
			self.task_types = [
				[self.task_matching, 'dms'],
			]

		elif par['task'] == 'multistim':
			self.task_types = [
				[self.task_go, 'go', 0],
				[self.task_go, 'rt_go', 0],
				[self.task_go, 'dly_go', 0],

				[self.task_go, 'go', np.pi],
				[self.task_go, 'rt_go', np.pi],
				[self.task_go, 'dly_go', np.pi],

				[self.task_dm, 'dm1'],
				[self.task_dm, 'dm2'],
				[self.task_dm, 'ctx_dm1'],
				[self.task_dm, 'ctx_dm2'],
				[self.task_dm, 'multsen_dm'],

				[self.task_dm_dly, 'dm1_dly'],
				[self.task_dm_dly, 'dm2_dly'],
				[self.task_dm_dly, 'ctx_dm1_dly'],
				[self.task_dm_dly, 'ctx_dm2_dly'],
				[self.task_dm_dly, 'multsen_dm_dly'],

				[self.task_matching, 'dms'],
				[self.task_matching, 'dmc'],
				[self.task_matching, 'dnms'],
				[self.task_matching, 'dnmc']
			]

		elif par['task'] == 'twelvestim':
			self.task_types = [
				[self.task_go, 'go', 0],
				[self.task_go, 'dly_go', 0],

				[self.task_dm, 'dm1'],
				[self.task_dm, 'dm2'],
				[self.task_dm, 'ctx_dm1'],
				[self.task_dm, 'ctx_dm2'],
				[self.task_dm, 'multsen_dm'],

				[self.task_dm_dly, 'dm1_dly'],
				[self.task_dm_dly, 'dm2_dly'],
				[self.task_dm_dly, 'ctx_dm1_dly'],
				[self.task_dm_dly, 'ctx_dm2_dly'],
				[self.task_dm_dly, 'multsen_dm_dly']
			]


	def generate_trial(self, task_num=0):

		# Create blank trial info
		self.trial_info = {
			'neural_input'   : np.random.normal(par['input_mean'], par['noise_in'], size=self.input_shape),
			'desired_output' : np.zeros(self.output_shape, dtype=np.float32),
			'reward_data'    : np.zeros(self.output_shape, dtype=np.float32),
			'train_mask'     : np.ones(self.mask_shape, dtype=np.float32)
		}

		# Populate trial info with basic information
		self.trial_info['train_mask'][:par['dead_time']//par['dt'],:] = 0.
		if par['num_rule_tuned'] > 0:
			self.trial_info['neural_input'] += self.rule_tuning(task_num)


		# Apply reinforcement learning task specifications
		for self.trial_num in range(par['batch_size']):

			task_num = np.random.choice(par['task_list'])
			current_task = self.task_types[task_num]
			current_task[0](*current_task[1:])

			resp_vect = np.sum(self.trial_info['desired_output'][:,self.trial_num,:-1], axis=1)

			for b in range(par['trials_per_seq']):

				# Designate timings
				t0 = par['num_time_steps']*b
				t1 = par['num_time_steps']*(b+1)
				respond_time    = np.where(resp_vect[t0:t1] > 0)[0]
				fix_time        = list(range(t0,t0+respond_time[0])) if len(respond_time) > 0 else [t1]
				respond_time    = t0+respond_time if len(respond_time) > 0 else [t1]

				# Designate responses
				correct_response    = np.where(self.trial_info['desired_output'][respond_time[0],self.trial_num,:]==1)[0]
				incorrect_response  = np.where(self.trial_info['desired_output'][respond_time[0],self.trial_num,:-1]==0)[0]

				# Build reward data
				self.trial_info['reward_data'][fix_time,self.trial_num,:-1] = par['fix_break_penalty']
				self.trial_info['reward_data'][respond_time,self.trial_num,correct_response] = par['correct_choice_reward']
				for i in incorrect_response:
					self.trial_info['reward_data'][respond_time,self.trial_num,i] = par['wrong_choice_penalty']

					# Penalize fixating throughout entire trial if response was required
					if not self.trial_info['desired_output'][t1-1,self.trial_num,-1] == 1:
						self.trial_info['reward_data'][t1-1,self.trial_num,-1] = par['fix_break_penalty']
					else:
						self.trial_info['reward_data'][t1-1,self.trial_num,-1] = par['correct_choice_reward']

		# Make required corrections
		self.trial_info['neural_input'] = np.maximum(0., self.trial_info['neural_input'])

		# Returns the task name and trial info
		return current_task[1], self.trial_info


	def task_go(self, variant='go', offset=0):

		# Task parameters
		if variant == 'go':
			stim_onset = np.random.randint(self.fix_time, self.fix_time+1000, par['trials_per_seq'])//par['dt']
			stim_off = par['num_time_steps']
			fixation_end = np.ones(par['trials_per_seq'], dtype=np.int16)*(self.fix_time+1000)//par['dt']
			resp_onset = fixation_end
		elif variant == 'rt_go':
			stim_onset = np.random.randint(self.fix_time, self.fix_time+1000, par['trials_per_seq'])//par['dt']
			stim_off = par['num_time_steps']
			fixation_end = np.ones(par['trials_per_seq'],dtype=np.int16)*par['num_time_steps']
			resp_onset = stim_onset
		elif variant == 'dly_go':
			stim_onset = self.fix_time//par['dt']*np.ones((par['trials_per_seq']),dtype=np.int16)
			stim_off = (self.fix_time+300)//par['dt']
			fixation_end = stim_off + np.random.choice(self.go_delay, size=par['trials_per_seq'])
			resp_onset = fixation_end
		else:
			raise Exception('Bad task variant.')

		# Need dead time
		self.trial_info['train_mask'][:par['dead_time']//par['dt'], :] = 0


		for b in range(par['trials_per_seq']):

			t0 = par['num_time_steps']*b
			t1 = par['num_time_steps']*(b+1)

			# Input neurons index above par['num_motion_tuned'] encode fixation
			self.trial_info['neural_input'][t0:t0+fixation_end[b], self.trial_num, par['num_motion_tuned']:par['num_motion_tuned']+par['num_fix_tuned']] \
				+= par['tuning_height']
			"""
			self.trial_info['neural_input'][:fixation_end[b], self.trial_num, par['num_motion_tuned']:par['num_motion_tuned']+par['num_fix_tuned']] \
				+= par['tuning_height']
			"""
			modality   = np.random.randint(2)
			neuron_ind = range(self.modality_size*modality, self.modality_size*(1+modality))
			stim_dir   = np.random.choice(self.motion_dirs)
			target_ind = int(np.round(par['num_motion_dirs']*(stim_dir+offset)/(2*np.pi))%par['num_motion_dirs'])

			#print('stim_dir', stim_dir)

			self.trial_info['neural_input'][t0+stim_onset[b]:t0+stim_off, self.trial_num, neuron_ind] += np.reshape(self.circ_tuning(stim_dir),(1,-1))
			self.trial_info['desired_output'][t0+resp_onset[b]:t1, self.trial_num, target_ind] = 1
			self.trial_info['desired_output'][t0:t0+resp_onset[b], self.trial_num, -1] = 1

			self.trial_info['train_mask'][t0+resp_onset[b]:t0+resp_onset[b]+par['mask_duration']//par['dt'], self.trial_num] = 0

		return self.trial_info


	def task_dm(self, variant='dm1'):

		# Create trial stimuli
		stim_dir1 = np.random.choice(self.motion_dirs, [1, par['trials_per_seq']])
		stim_dir2 = (stim_dir1 + np.pi/2 + np.random.choice(self.motion_dirs[::2], [1, par['trials_per_seq']])/2)%(2*np.pi)

		stim1 = self.circ_tuning(stim_dir1)
		stim2 = self.circ_tuning(stim_dir2)

		# Determine the strengths of the stimuli in each modality
		c_mod1 = np.random.choice(self.dm_c_set, [1, par['trials_per_seq']])
		c_mod2 = np.random.choice(self.dm_c_set, [1, par['trials_per_seq']])
		mean_gamma = 0.8 + 0.4*np.random.rand(1, par['trials_per_seq'])
		gamma_s1_m1 = mean_gamma + c_mod1
		gamma_s2_m1 = mean_gamma - c_mod1
		gamma_s1_m2 = mean_gamma + c_mod2
		gamma_s2_m2 = mean_gamma - c_mod2

		# Determine response directions and convert to output indices
		resp_dir_mod1 = np.where(gamma_s1_m1 > gamma_s2_m1, stim_dir1, stim_dir2)
		resp_dir_mod2 = np.where(gamma_s1_m2 > gamma_s2_m2, stim_dir1, stim_dir2)
		resp_dir_sum  = np.where(gamma_s1_m1 + gamma_s1_m2 > gamma_s2_m1 + gamma_s2_m2, stim_dir1, stim_dir2)

		resp_dir_mod1 = np.round(par['num_motion_dirs']*resp_dir_mod1/(2*np.pi))
		resp_dir_mod2 = np.round(par['num_motion_dirs']*resp_dir_mod2/(2*np.pi))
		resp_dir_sum  = np.round(par['num_motion_dirs']*resp_dir_sum/(2*np.pi))

		# Apply stimuli to modalities and build appropriate response
		if variant == 'dm1':
			modality1 = gamma_s1_m1*stim1 + gamma_s2_m1*stim2
			modality2 = np.zeros_like(stim1)
			resp_dirs = resp_dir_mod1
		elif variant == 'dm2':
			modality1 = np.zeros_like(stim1)
			modality2 = gamma_s1_m2*stim1 + gamma_s2_m2*stim2
			resp_dirs = resp_dir_mod2
		elif variant == 'ctx_dm1':
			modality1 = gamma_s1_m1*stim1 + gamma_s2_m1*stim2
			modality2 = gamma_s1_m2*stim1 + gamma_s2_m2*stim2
			resp_dirs = resp_dir_mod1
		elif variant == 'ctx_dm2':
			modality1 = gamma_s1_m1*stim1 + gamma_s2_m1*stim2
			modality2 = gamma_s1_m2*stim1 + gamma_s2_m2*stim2
			resp_dirs = resp_dir_mod2
		elif variant == 'multsen_dm':
			modality1 = gamma_s1_m1*stim1 + gamma_s2_m1*stim2
			modality2 = gamma_s1_m2*stim1 + gamma_s2_m2*stim2
			resp_dirs = resp_dir_sum
		else:
			raise Exception('Bad task variant.')

		resp = np.zeros([par['num_motion_dirs'], par['trials_per_seq']])
		for b in range(par['trials_per_seq']):
			resp[np.int16(resp_dirs[0,b]%par['num_motion_dirs']),b] = 1

		# Setting up arrays
		fixation = np.zeros(self.fixation_shape)
		response = np.zeros(self.response_shape)
		stimulus = np.zeros(self.stimulus_shape)
		mask     = np.ones(self.mask_shape)
		mask[:par['dead_time']//par['dt'],:] = 0
		resp_fix  = np.copy(fixation[:,:,0:1])

		# Identify stimulus onset for each trial and build each trial from there
		stim_onset = self.fix_time//par['dt']
		stim_off   = stim_onset + np.random.choice(self.dm_stim_lengths, par['trials_per_seq'])

		for b in range(par['trials_per_seq']):
			t0 = par['num_time_steps']*b
			t1 = par['num_time_steps']*(b+1)

			fixation[t0:t0+stim_off[b],self.trial_num,:] = par['tuning_height']
			resp_fix[t0:t0+stim_off[b],self.trial_num] = 1
			stimulus[t0+stim_onset:t0+stim_off[b],self.trial_num,:] = np.transpose(np.concatenate([modality1[:,b], modality2[:,b]], axis=0)[:,np.newaxis])
			response[t0+stim_off[b]:t1,self.trial_num,:] = np.transpose(resp[:,b,np.newaxis])
			mask[t0+stim_off[b]:t0+stim_off[b]+par['mask_duration']//par['dt'],self.trial_num] = 0

		# Merge activies and fixations into single vector
		stimulus = np.concatenate([stimulus, fixation], axis=2)
		response = np.concatenate([response, resp_fix], axis=2)

		self.trial_info['neural_input'][:,:,:par['num_motion_tuned']+par['num_fix_tuned']] += stimulus
		self.trial_info['desired_output'] = response
		self.trial_info['train_mask'] = mask

		return self.trial_info


	def task_dm_dly(self, variant='dm1'):

		# Create trial stimuli
		stim_dir1 = 2*np.pi*np.random.rand(1, par['batch_size'])
		stim_dir2 = (stim_dir1 + np.pi/2 + np.pi*np.random.rand(1, par['batch_size']))%(2*np.pi)
		stim1 = self.circ_tuning(stim_dir1)
		stim2 = self.circ_tuning(stim_dir2)

		# Determine the strengths of the stimuli in each modality
		c_mod1 = np.random.choice(self.dm_dly_c_set, [1, par['batch_size']])
		c_mod2 = np.random.choice(self.dm_dly_c_set, [1, par['batch_size']])
		mean_gamma = 0.8 + 0.4*np.random.rand(1, par['batch_size'])
		gamma_s1_m1 = mean_gamma + c_mod1
		gamma_s2_m1 = mean_gamma - c_mod1
		gamma_s1_m2 = mean_gamma + c_mod2
		gamma_s2_m2 = mean_gamma - c_mod2

		# Determine the delay for each trial
		delay = np.random.choice(self.dm_dly_delay, [1, par['batch_size']])

		# Determine response directions and convert to output indices
		resp_dir_mod1 = np.where(gamma_s1_m1 > gamma_s2_m1, stim_dir1, stim_dir2)
		resp_dir_mod2 = np.where(gamma_s1_m2 > gamma_s2_m2, stim_dir1, stim_dir2)
		resp_dir_sum  = np.where(gamma_s1_m1 + gamma_s1_m2 > gamma_s2_m1 + gamma_s2_m2, stim_dir1, stim_dir2)

		resp_dir_mod1 = np.round(par['num_motion_dirs']*resp_dir_mod1/(2*np.pi))
		resp_dir_mod2 = np.round(par['num_motion_dirs']*resp_dir_mod2/(2*np.pi))
		resp_dir_sum  = np.round(par['num_motion_dirs']*resp_dir_sum/(2*np.pi))

		# Apply stimuli to modalities and build appropriate response
		if variant == 'dm1_dly':
			modality1_t1 = gamma_s1_m1*stim1
			modality2_t1 = np.zeros_like(stim1)
			modality1_t2 = gamma_s2_m1*stim2
			modality2_t2 = np.zeros_like(stim2)
			resp_dirs = resp_dir_mod1
		elif variant == 'dm2_dly':
			modality1_t1 = np.zeros_like(stim1)
			modality2_t1 = gamma_s1_m2*stim1
			modality1_t2 = np.zeros_like(stim2)
			modality2_t2 = gamma_s2_m2*stim2
			resp_dirs = resp_dir_mod2
		elif variant == 'ctx_dm1_dly':
			modality1_t1 = gamma_s1_m1*stim1
			modality2_t1 = gamma_s1_m2*stim1
			modality1_t2 = gamma_s2_m1*stim2
			modality2_t2 = gamma_s2_m2*stim2
			resp_dirs = resp_dir_mod1
		elif variant == 'ctx_dm2_dly':
			modality1_t1 = gamma_s1_m1*stim1
			modality2_t1 = gamma_s1_m2*stim1
			modality1_t2 = gamma_s2_m1*stim2
			modality2_t2 = gamma_s2_m2*stim2
			resp_dirs = resp_dir_mod2
		elif variant == 'multsen_dm_dly':
			modality1_t1 = gamma_s1_m1*stim1
			modality2_t1 = gamma_s1_m2*stim1
			modality1_t2 = gamma_s2_m1*stim2
			modality2_t2 = gamma_s2_m2*stim2
			resp_dirs = resp_dir_sum
		else:
			raise Exception('Bad task variant.')

		resp = np.zeros([par['num_motion_dirs'], par['batch_size']])
		for b in range(par['batch_size']):
			resp[np.int16(resp_dirs[0,b]%par['num_motion_dirs']),b] = 1

		# Setting up arrays
		fixation = np.zeros(self.fixation_shape)
		response = np.zeros(self.response_shape)
		stimulus = np.zeros(self.stimulus_shape)
		mask     = np.ones(self.mask_shape)
		mask[:par['dead_time']//par['dt'],:] = 0
		resp_fix  = np.copy(fixation[:,:,0:1])

		# Identify stimulus onset for each trial and build each trial from there
		stim_on1   = self.fix_time//par['dt']
		stim_off1  = (self.fix_time+300)//par['dt']
		stim_on2   = delay + stim_off1
		stim_off2  = stim_on2 + 300//par['dt']
		resp_time  = stim_off2 + 0//par['dt']
		for b in range(par['batch_size']):
			fixation[:resp_time[0,b],b,:] = par['tuning_height']
			resp_fix[:resp_time[0,b],b] = 1
			stimulus[stim_on1:stim_off1,b,:] = np.concatenate([modality1_t1[:,b], modality2_t1[:,b]], axis=0)[np.newaxis,:]
			stimulus[stim_on2[0,b]:stim_off2[0,b],b] = np.concatenate([modality1_t2[:,b], modality2_t2[:,b]], axis=0)[np.newaxis,:]
			response[resp_time[0,b]:,b,:] = resp[np.newaxis,:,b]
			mask[resp_time[0,b]:resp_time[0,b]+par['mask_duration'],b] = 0

		# Merge activies and fixations into single vectors
		stimulus = np.concatenate([stimulus, fixation], axis=2)
		response = np.concatenate([response, resp_fix], axis=2)    # Duplicates starting fixation on output

		self.trial_info['neural_input'][:,:,:par['num_motion_tuned']+par['num_fix_tuned']] += stimulus
		self.trial_info['desired_output'] = response
		self.trial_info['train_mask'] = mask

		return self.trial_info


	def task_matching(self, variant='dms'):

		# Determine matches, and get stimuli
		if variant in ['dms', 'dnms']:
			stim1 = np.random.choice(self.motion_dirs, par['batch_size'])
			nonmatch = (stim1 + np.random.choice(self.motion_dirs[1:], par['batch_size']))%(2*np.pi)

			match = np.random.choice(np.array([True, False]), par['batch_size'])
			stim2 = np.where(match, stim1, nonmatch)

		elif variant in ['dmc', 'dnmc']:
			stim1 = np.random.choice(self.motion_dirs, par['batch_size'])
			stim2 = np.random.choice(self.motion_dirs, par['batch_size'])

			stim1_cat = np.logical_and(np.less(-1e-3, stim1), np.less(stim1, np.pi))
			stim2_cat = np.logical_and(np.less(-1e-3, stim2), np.less(stim2, np.pi))
			match = np.logical_not(np.logical_xor(stim1_cat, stim2_cat))
		else:
			raise Exception('Bad variant.')

		# Establishing stimuli
		stimulus1 = self.circ_tuning(stim1)
		stimulus2 = self.circ_tuning(stim2)

		# Convert to response
		stim1_int = np.round(par['num_motion_dirs']*stim1/(2*np.pi))
		stim2_int = np.round(par['num_motion_dirs']*stim2/(2*np.pi))

		if variant in ['dms', 'dmc']:
			resp = np.where(match, 4, 0)#stim1_int, -1)
		elif variant in ['dnms', 'dnmc']:
			resp = np.where(match, 0, 4)#-1, stim2_int)
		else:
			raise Exception('Bad variant.')

		# Setting up arrays
		modality_choice = np.random.choice(np.array([0,1], dtype=np.int16), [2, par['batch_size']])
		modalities = np.zeros([2, par['num_time_steps'], par['batch_size'], par['num_motion_tuned']//2])
		fixation = np.zeros(self.fixation_shape)
		response = np.zeros(self.response_shape)
		stimulus = np.zeros(self.stimulus_shape)
		mask     = np.ones(self.mask_shape)
		mask[:par['dead_time']//par['dt'],:] = 0

		# Decide timings and build each trial
		stim1_on  = self.fix_time//par['dt']
		stim1_off = (self.fix_time+300)//par['dt']
		stim2_on  = stim1_off + np.random.choice(self.match_delay, par['batch_size'])
		stim2_off = stim2_on + 300//par['dt']
		resp_time = stim2_off
		resp_fix  = np.copy(fixation[:,:,0:1])

		for b in range(par['batch_size']):
			fixation[:resp_time[b],b,:] = par['tuning_height']

			# Ensuring that sample and test stimuli are in same modality (RF)
			modalities[modality_choice[0,b],stim1_on:stim1_off,b,:] = stimulus1[np.newaxis,:,b]
			modalities[modality_choice[0,b],stim2_on[b]:stim2_off[b],b,:] = stimulus2[np.newaxis,:,b]

			mask[resp_time[b]:resp_time[b]+par['mask_duration']//par['dt'],b] = 0
			if not resp[b] == -1:
				response[resp_time[b]:,b,int(resp[b])] = 1
				resp_fix[:resp_time[b],b] = 1
			else:
				resp_fix[:,b,:] = 1

		# Merge activies and fixations into single vectors)
		stimulus = np.concatenate([modalities[0], modalities[1], fixation], axis=2)
		response = np.concatenate([response, resp_fix], axis=2)

		self.trial_info['neural_input'][:,:,:par['num_motion_tuned']+par['num_fix_tuned']] += stimulus
		self.trial_info['desired_output'] = response
		self.trial_info['train_mask'] = mask

		return self.trial_info


if __name__ == '__main__':
	print('Testing stimulus generation.')
	s = Stimulus()
	s.generate_trial()
