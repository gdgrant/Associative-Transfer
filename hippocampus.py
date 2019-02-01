### Authors: Nicolas Y. Masse, Gregory D. Grant

# Required packages
import tensorflow as tf
import numpy as np
import pickle
import os, sys, time

# Plotting suite
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model modules
from parameters import *
import stimulus
import AdamOpt

# Match GPU IDs to nvidia-smi command
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def sigmoid(x):
	x = 10*(x - 0.5)
	return 1/(1+np.exp(-x))


class Hippocampus:

	def __init__(self, vector_size, num_actions, num_rewards):
		self.M = np.zeros([vector_size, vector_size], dtype=np.float32)
		self.vector_size = vector_size

		self.num_actions = num_actions
		self.num_rewards = num_rewards
		self.action_reward_size = num_actions + num_rewards
		self.batch_size = 200


	def train_events(self, event_data):

		# Event data has shape [trials x neurons]
		trials = event_data.shape[0]
		num_batches = trials//self.batch_size

		for b in range(num_batches):

			trial_inds = np.arange(b*self.batch_size, (b+1)*self.batch_size).astype(np.int32)
			self.M = self.M + par['train_alpha']*np.mean(event_data[trial_inds,:,np.newaxis]*event_data[trial_inds,np.newaxis,:], axis=0)
			self.M = sigmoid(self.M)

			print('{} of {} batches complete.'.format(b+1, num_batches), end='\r')

		print('\nTraining complete.\n')


	def test_events(self, event_data):

		trials = event_data.shape[0]
		num_batches = trials//self.batch_size

		event_data = np.copy(event_data)
		event_data[:,-self.action_reward_size:] = 0.

		L = []
		for b in range(num_batches):

			trial_inds = np.arange(b*self.batch_size, (b+1)*self.batch_size).astype(np.int32)
			L_batch = np.zeros([self.batch_size, self.vector_size], dtype=np.float32)

			L_batch = event_data[trial_inds,:]

			for i in range(par['associative_iters']):
				L_batch = L_batch + par['train_beta']*(L_batch @ self.M)
				L_batch = sigmoid(L_batch)

			L.append(L_batch)

			print('{} of {} batches complete.'.format(b+1, num_batches), end='\r')

		self.L = np.concatenate(L, axis=0)
		print('\nTesting complete.\n')

	def eval_events(self, event_data):

		event_data = sigmoid(event_data)

		loss = np.abs(self.L - event_data)

		ar_arg_E_action = np.argmax(event_data[:,-self.action_reward_size:-self.num_rewards], axis=-1)
		ar_arg_L_action = np.argmax(self.L[:,-self.action_reward_size:-self.num_rewards], axis=-1)

		ar_arg_E_reward = np.argmax(event_data[:,-self.num_rewards:], axis=-1)
		ar_arg_L_reward = np.argmax(self.L[:,-self.num_rewards:], axis=-1)

		print('\n' + '-'*40 + '\n')
		print('Action Association Accuracy: {:5.3f}'.format(np.mean(ar_arg_E_action==ar_arg_L_action)))
		print('Reward Association Accuracy: {:5.3f}'.format(np.mean(ar_arg_E_reward==ar_arg_L_reward)))

		act_inds = np.unique([ar_arg_E_action, ar_arg_L_action])
		rew_inds = np.unique([ar_arg_E_reward, ar_arg_L_reward])

		print('\nAction Index Accuracy Breakdown:')
		for a in act_inds:
			print('Index {} --> Acc. {:5.3f}'.format(a, np.mean((ar_arg_E_action==ar_arg_L_action)[np.where(ar_arg_E_action==a)])))

		print('\nReward Index Accuracy Breakdown:')
		for r in rew_inds:
			print('Index {} --> Acc. {:5.3f}'.format(r, np.mean((ar_arg_E_reward==ar_arg_L_reward)[np.where(ar_arg_E_reward==r)])))


		fig, ax = plt.subplots(1,3,figsize=(12,8), sharex=True, sharey=True)
		ax[0].imshow(event_data, aspect='auto', clim=(0,1))
		ax[1].imshow(self.L, aspect='auto', clim=(0,1))
		ax[2].imshow(loss, aspect='auto', clim=(-1,1))

		ax[0].set_ylabel('Trials')
		ax[1].set_xlabel('Units (Neurons, Actions, Rewards)')

		ax[0].set_title('Event Data')
		ax[1].set_title('Association')
		ax[2].set_title('L1 Loss')

		plt.suptitle('Full Encoding and Reconstruction Comparison')
		plt.show()
		plt.clf()
		plt.close()


def main(event_data):

	stimuli = event_data['encoded_stimuli']
	actions = event_data['actions']
	rewards = event_data['rewards']

	reward_index = np.array([par['reward_map'][(r+1e-6).round(2)] for r in np.squeeze(rewards)])
	rewards = np.zeros([rewards.shape[0],par['num_reward_types']])
	rewards[np.arange(rewards.shape[0]),reward_index] = 1.

	posterior_dist_data = 1-0.5*np.random.rand(stimuli.shape[0]) # 0.5 to 1
	posterior = np.stack([posterior_dist_data, 1-posterior_dist_data], axis=1)

	training_data_size = np.int32(stimuli.shape[0]*(1-par['test_sample_prop']))

	aggregate_events = np.concatenate([posterior, stimuli, actions, rewards], axis=1)
	np.random.shuffle(aggregate_events)
	vector_size = aggregate_events.shape[1]

	training_events = aggregate_events[:training_data_size,:]
	testing_events  = aggregate_events[training_data_size:,:]

	hippocampus = Hippocampus(vector_size, actions.shape[1], rewards.shape[1])
	hippocampus.train_events(training_events)
	hippocampus.test_events(testing_events)
	hippocampus.eval_events(testing_events)

	pickle.dump(hippocampus.M, open('./datadir/gotask_associative_memory.pkl','wb'))


if __name__ == '__main__':

	event_data_file = sys.argv[1]
	event_data = pickle.load(open(event_data_file, 'rb'))

	main(event_data)