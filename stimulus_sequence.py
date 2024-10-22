### Authors: Nicolas Y. Masse, Gregory D. Grant

import numpy as np
from parameters_v6 import par
from math import gamma
import scipy.signal
import matplotlib.pyplot as plt

class Stimulus:

	def __init__(self, analysis=False):

		# Stimulus shapes
		self.input_shape    = [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['n_input'] ]
		self.filtered_input_shape    = [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['n_filters']*par['n_input'] ]
		self.output_shape   = [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['n_pol'] ]
		self.stimulus_shape = [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['num_motion_tuned'] ]
		self.response_shape = [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['num_motion_dirs'] ]
		self.fixation_shape = [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['num_fix_tuned'] ]
		self.rule_shape		= [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['num_rule_tuned'] ]
		self.mask_shape     = [par['num_time_steps']*par['trials_per_seq'], par['batch_size']]

		# Gamma filters
		self.gamma_filters = self.make_gamma_filters()

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

	def make_gamma_filters(self):

		c = 0.08
		t = np.arange(1000//par['dt'])/(1000/par['dt'])
		f = np.zeros((1, par['n_filters'], len(t)), dtype = np.float32)
		for i in range(par['n_filters']):
			k = 2**i
			f[0, i, :] = t**(k-1)*np.exp(-t/c)/(gamma(k)*c**k)
			f[0, i, :] /= np.sum(f[0, i, :])

		return np.real(f)


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

			self.task_types = []
			for offset in [0, np.pi, np.pi/2, -np.pi/2, np.pi/4, -np.pi/4, 3*np.pi/4, -3*np.pi/4]:


				self.task_types.append([self.task_go, 'go', offset])
				self.task_types.append([self.task_go, 'rt_go', offset])


				self.task_types.append([self.task_go, 'go_OIC', offset, 0])
				self.task_types.append([self.task_go, 'go_OIC', offset, 2])
				self.task_types.append([self.task_go, 'rt_OIC', offset, 0])
				self.task_types.append([self.task_go, 'rt_OIC', offset, 2])

				#self.task_types.append([self.task_go, 'go_OIC', offset, 5])
				#self.task_types.append([self.task_go, 'dly_go', offset])

				self.task_types.append([self.task_dm, 'dm1',offset])
				self.task_types.append([self.task_dm, 'dm2',offset])

				self.task_types.append([self.task_dm, 'ctx_dm1',offset])
				self.task_types.append([self.task_dm, 'ctx_dm2',offset])
				self.task_types.append([self.task_dm, 'multsen_dm',offset])
				"""

				self.task_types.append([self.task_dm_dly, 'dm1_dly',offset])
				self.task_types.append([self.task_dm_dly, 'dm2_dly',offset])
				self.task_types.append([self.task_dm_dly, 'ctx_dm1_dly',offset])
				self.task_types.append([self.task_dm_dly, 'ctx_dm2_dly',offset])
				self.task_types.append([self.task_dm_dly, 'multsen_dm_dly',offset])
				# the above trials are used for the 100_tasks_XXX

				self.task_types.append([self.task_matching, 'dms',offset])
				self.task_types.append([self.task_matching, 'dmc',offset])
				"""




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


	def generate_trial(self, fixed_task_num = None):

		# Create blank trial info
		self.trial_info = {
			'neural_input'   : np.random.normal(par['input_mean'], par['noise_in'], size=self.input_shape),
			'neural_input_filtered'   : np.random.normal(par['input_mean'], par['noise_in'], size=self.filtered_input_shape),
			'desired_output' : np.zeros(self.output_shape, dtype=np.float32),
			'reward_data'    : np.zeros(self.output_shape, dtype=np.float32),
			'reward_matrix'  : np.zeros([*self.output_shape, par['num_reward_types']], dtype=np.float32),
			'train_mask'     : np.ones(self.mask_shape, dtype=np.float32)
		}

		# Populate trial info with basic information
		self.trial_info['train_mask'][:par['dead_time']//par['dt'],:] = 0.
		if par['num_rule_tuned'] > 0:
			self.trial_info['neural_input'] += self.rule_tuning(task_num)


		# Apply reinforcement learning task specifications
		for self.trial_num in range(par['batch_size']):

			if fixed_task_num is None:
				task_num = np.random.choice(par['task_list'])
			else:
				task_num = fixed_task_num
			#print('task num ', task_num, self.task_types[task_num][1])
			current_task = self.task_types[task_num]
			current_task[0](*current_task[1:])

			resp_vect = np.sum(self.trial_info['desired_output'][:,self.trial_num,:-1], axis=1)


			for b in range(par['trials_per_seq']):

				# Designate timings
				t0 = par['num_time_steps']*b
				t1 = par['num_time_steps']*(b+1)
				respond_time    = np.where(resp_vect[t0:t1] > 0)[0]
				fix_time        = list(range(t0,t0+respond_time[0])) if len(respond_time) > 0 else [t1-1]
				respond_time    = t0+respond_time if len(respond_time) > 0 else [t1-1]

				# Designate responses
				correct_response    = np.where(self.trial_info['desired_output'][respond_time[0],self.trial_num,:]==1)[0]
				incorrect_response  = np.where(self.trial_info['desired_output'][respond_time[0],self.trial_num,:-1]==0)[0]

				# Build reward data
				self.trial_info['reward_data'][fix_time,self.trial_num,:-1] = par['fix_break_penalty']
				self.trial_info['reward_matrix'][fix_time,self.trial_num,:-1, 0] = 1.
				self.trial_info['reward_data'][respond_time,self.trial_num,correct_response] = par['correct_choice_reward']
				self.trial_info['reward_matrix'][respond_time,self.trial_num,correct_response, 3] = 1.
				self.trial_info['reward_matrix'][fix_time,self.trial_num,-1, 2] = 1.


				for i in incorrect_response:
					self.trial_info['reward_data'][respond_time,self.trial_num,i] = par['wrong_choice_penalty']
					self.trial_info['reward_matrix'][respond_time,self.trial_num,i,1] = 1.

					# Penalize fixating throughout entire trial if response was required
					if not self.trial_info['desired_output'][t1-1,self.trial_num,-1] == 1:
						pass
						#self.trial_info['reward_data'][t1-1,self.trial_num,-1] = par['fix_break_penalty']
						#self.trial_info['reward_matrix'][t1-1,self.trial_num,-1, 0] = 1.
					else:
						self.trial_info['reward_data'][t1-1,self.trial_num,-1] = par['correct_choice_reward']
						self.trial_info['reward_matrix'][t1-1,self.trial_num,-1, 3] = 1.

		# Make required corrections
		self.trial_info['neural_input'] = np.maximum(0., self.trial_info['neural_input'])

		# SIMPLIFYING STIM; TESTING ONLY
		#self.trial_info['neural_input'] = np.float32(self.trial_info['neural_input'] > par['tuning_height']-0.001)

		s = [scipy.signal.convolve(self.trial_info['neural_input'], np.transpose(self.gamma_filters[0:1,i:i+1,:],[2,1,0]), 'full') \
			for i in range(par['n_filters'])]
		"""
		for i in range(5):
			plt.imshow(s[i][:, 0, :], aspect = 'auto')
			plt.show()
		"""
		s_concat = np.concatenate([*s], axis = -1)
		s_concat = s_concat[:par['num_time_steps']*par['trials_per_seq'], :,:]
		self.trial_info['neural_input_filtered'] = s_concat


		# Returns the task name and trial info
		return current_task[1], self.trial_info


	def task_go(self, variant='go', offset=0, cat_boundary = 0):

		# Task parameters
		if variant == 'go' or variant == 'go_OIC':
			stim_onset = np.random.randint(self.fix_time, self.fix_time+800, par['trials_per_seq'])//par['dt']
			stim_off = par['num_time_steps']
			fixation_end = np.ones(par['trials_per_seq'], dtype=np.int16)*(self.fix_time+800)//par['dt']
			resp_onset = fixation_end
		elif variant == 'rt_go' or variant == 'rt_OIC':
			stim_onset = np.random.randint(self.fix_time, self.fix_time+800, par['trials_per_seq'])//par['dt']
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
			if 'OIC' in variant:
				stim_index = int(np.round(par['num_motion_dirs']*(stim_dir+offset)/(2*np.pi)))
				stimulus_category_one = (stim_index in \
					list(np.arange(cat_boundary,cat_boundary+par['num_motion_dirs']//2)%par['num_motion_dirs']))
				if stimulus_category_one:
					target_ind = int(np.round(par['num_motion_dirs']*(offset)/(2*np.pi))%par['num_motion_dirs'])
				else:
					target_ind = int(np.round(par['num_motion_dirs']*(offset+np.pi)/(2*np.pi))%par['num_motion_dirs'])
			else:
				target_ind = int(np.round(par['num_motion_dirs']*(stim_dir+offset)/(2*np.pi))%par['num_motion_dirs'])

			#print(b, stim_dir, stim_index, stimulus_category_one, target_ind)
			#print('stim_dir', stim_dir)

			self.trial_info['neural_input'][t0+stim_onset[b]:t0+stim_off, self.trial_num, neuron_ind] += np.reshape(self.circ_tuning(stim_dir),(1,-1))
			self.trial_info['desired_output'][t0+resp_onset[b]:t1, self.trial_num, target_ind] = 1
			self.trial_info['desired_output'][t0:t0+resp_onset[b], self.trial_num, -1] = 1


			self.trial_info['train_mask'][t0+resp_onset[b]:t0+resp_onset[b]+par['mask_duration']//par['dt'], self.trial_num] = 0

		return self.trial_info


	def task_dm(self, variant='dm1', offset = 0):

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

		resp_dirs = np.int8(np.round(par['num_motion_dirs']*(resp_dirs+offset)/(2*np.pi))%par['num_motion_dirs'])


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
			#stimulus[t0+stim_onset:t0+stim_off[b],self.trial_num,:] = np.transpose(np.concatenate([modality1[:,b], modality2[:,b]], axis=0)[:,np.newaxis])

			# TEMP CHANGE SO NO WORKING MEMORY REQUIRED
			stimulus[t0+stim_onset:t1,self.trial_num,:] = np.transpose(np.concatenate([modality1[:,b], modality2[:,b]], axis=0)[:,np.newaxis])

			response[t0+stim_off[b]:t1,self.trial_num,:] = np.transpose(resp[:,b,np.newaxis])
			mask[t0+stim_off[b]:t0+stim_off[b]+par['mask_duration']//par['dt'],self.trial_num] = 0

		# Merge activies and fixations into single vector
		stimulus = np.concatenate([stimulus, fixation], axis=2)
		response = np.concatenate([response, resp_fix], axis=2)

		self.trial_info['neural_input'][:,:,:par['num_motion_tuned']+par['num_fix_tuned']] += stimulus
		self.trial_info['desired_output'] = response
		self.trial_info['train_mask'] = mask

		return self.trial_info


	def task_dm_dly(self, variant='dm1', offset = 0):

		# Create trial stimuli
		stim_dir1 = 2*np.pi*np.random.rand(1, par['trials_per_seq'])
		stim_dir2 = (stim_dir1 + np.pi/2 + np.pi*np.random.rand(1, par['trials_per_seq']))%(2*np.pi)
		stim1 = self.circ_tuning(stim_dir1)
		stim2 = self.circ_tuning(stim_dir2)

		# Determine the strengths of the stimuli in each modality
		c_mod1 = np.random.choice(self.dm_dly_c_set, [1, par['trials_per_seq']])
		c_mod2 = np.random.choice(self.dm_dly_c_set, [1, par['trials_per_seq']])
		mean_gamma = 0.8 + 0.4*np.random.rand(1, par['trials_per_seq'])
		gamma_s1_m1 = mean_gamma + c_mod1
		gamma_s2_m1 = mean_gamma - c_mod1
		gamma_s1_m2 = mean_gamma + c_mod2
		gamma_s2_m2 = mean_gamma - c_mod2

		# Determine the delay for each trial
		delay = np.random.choice(self.dm_dly_delay, [1, par['trials_per_seq']])

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

		resp_dirs = np.int8(np.round(par['num_motion_dirs']*(resp_dirs+offset)/(2*np.pi))%par['num_motion_dirs'])

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
		stim_on1   = self.fix_time//par['dt']
		stim_off1  = (self.fix_time+300)//par['dt']
		stim_on2   = delay + stim_off1
		stim_off2  = par['num_time_steps']
		resp_time  = stim_on2
		for b in range(par['trials_per_seq']):
			t0 = par['num_time_steps']*b
			t1 = par['num_time_steps']*(b+1)

			fixation[t0:t0+resp_time[0,b],self.trial_num,:] = par['tuning_height']
			resp_fix[t0:t0+resp_time[0,b],self.trial_num] = 1
			stimulus[t0+stim_on1:t0+stim_off1,self.trial_num,:] = np.concatenate([modality1_t1[:,b], modality2_t1[:,b]], axis=0)[np.newaxis,:]
			stimulus[t0+stim_on2[0,b]:t1,self.trial_num,:] = np.concatenate([modality1_t2[:,b], modality2_t2[:,b]], axis=0)[np.newaxis,:]
			response[t0+resp_time[0,b]:t1,self.trial_num,:] = resp[np.newaxis,:,b]
			mask[t0+resp_time[0,b]:t0+resp_time[0,b]+par['mask_duration'],self.trial_num] = 0

		# Merge activies and fixations into single vectors
		stimulus = np.concatenate([stimulus, fixation], axis=2)
		response = np.concatenate([response, resp_fix], axis=2)    # Duplicates starting fixation on output

		self.trial_info['neural_input'][:,:,:par['num_motion_tuned']+par['num_fix_tuned']] += stimulus
		self.trial_info['desired_output'] = response
		self.trial_info['train_mask'] = mask

		return self.trial_info


	def task_matching(self, variant='dms', offset = 0):

		# Determine matches, and get stimuli
		if variant in ['dms', 'dnms']:
			stim1 = np.random.choice(self.motion_dirs, par['trials_per_seq'])
			nonmatch = (stim1 + np.random.choice(self.motion_dirs[1:], par['trials_per_seq']))%(2*np.pi)

			match = np.random.choice(np.array([True, False]), par['trials_per_seq'])
			stim2 = np.where(match, stim1, nonmatch)

		elif variant in ['dmc', 'dnmc']:
			stim1 = np.random.choice(self.motion_dirs, par['trials_per_seq'])
			stim2 = np.random.choice(self.motion_dirs, par['trials_per_seq'])

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

		# Saccade targets for match and non-match
		r0 = np.int8(np.round(par['num_motion_dirs']*offset/(2*np.pi))%par['num_motion_dirs'])
		r1 = (r0+par['num_motion_dirs']//2)%par['num_motion_dirs']

		if variant in ['dms', 'dmc']:
			resp = np.where(match, r0, r1)#stim1_int, -1)
		else:
			raise Exception('Bad variant.')

		# Setting up arrays
		modality_choice = np.random.choice(np.array([0,1], dtype=np.int16), [2, par['trials_per_seq']])
		modalities = np.zeros([2, par['trials_per_seq']*par['num_time_steps'], par['batch_size'], par['num_motion_tuned']//2])
		fixation = np.zeros(self.fixation_shape)
		response = np.zeros(self.response_shape)
		stimulus = np.zeros(self.stimulus_shape)
		mask     = np.ones(self.mask_shape)
		mask[:par['dead_time']//par['dt'],:] = 0

		# Decide timings and build each trial
		stim1_on  = self.fix_time//par['dt']
		stim1_off = (self.fix_time+300)//par['dt']
		stim2_on  = stim1_off + np.random.choice(self.match_delay, par['trials_per_seq'])
		stim2_off = par['num_time_steps']
		resp_time = stim2_on
		resp_fix  = np.copy(fixation[:,:,0:1])

		for b in range(par['trials_per_seq']):
			t0 = par['num_time_steps']*b
			t1 = par['num_time_steps']*(b+1)

			fixation[t0:t0+resp_time[b],self.trial_num,:] = par['tuning_height']

			# Ensuring that sample and test stimuli are in same modality (RF)
			modalities[modality_choice[0,b],t0+stim1_on:t0+stim1_off,self.trial_num,:] = stimulus1[np.newaxis,:,b]
			modalities[modality_choice[0,b],t0+stim2_on[b]:t1,self.trial_num,:] = stimulus2[np.newaxis,:,b]

			mask[t0+resp_time[b]:t0+resp_time[b]+par['mask_duration']//par['dt'],self.trial_num] = 0
			if not resp[b] == -1:
				response[t0+resp_time[b]:t1,self.trial_num,int(resp[b])] = 1
				resp_fix[t0:t0+resp_time[b],self.trial_num] = 1
			else:
				resp_fix[t0:t1,b,:] = 1

		# Merge activies and fixations into single vectors)
		#print(modalities.shape, fixation.shape)
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
