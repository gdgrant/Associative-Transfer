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


def sigmoid(x):
	return 1/(1+np.exp(-x))


class Hippocampus:

	def __init__(self, vector_size, action_reward_size):
		self.M = np.zeros([vector_size, vector_size], dtype=np.float32)
		self.vector_size = vector_size
		self.action_reward_size = action_reward_size
		self.batch_size = 200


	def train_events(self, event_data):

		# Event data has shape [trials x neurons]
		trials = event_data.shape[0]
		num_batches = trials//self.batch_size

		for b in range(num_batches):

			trial_inds = np.arange(b*self.batch_size, (b+1)*self.batch_size).astype(np.int32)
			self.M = self.M + par['train_alpha']*np.mean(event_data[trial_inds,:,np.newaxis]*event_data[trial_inds,np.newaxis,:], axis=0)
			self.M = np.tanh(self.M)

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
				L_batch = np.tanh(L_batch)

			L.append(L_batch)

			print('{} of {} batches complete.'.format(b+1, num_batches), end='\r')

		self.L = np.concatenate(L, axis=0)
		print('\nTesting complete.\n')

	def eval_events(self, event_data):

		event_data = np.tanh(event_data)

		loss = np.abs(self.L - event_data)

		fig, ax = plt.subplots(1,3,figsize=(12,8), sharex=True, sharey=True)
		ax[0].imshow(event_data, aspect='auto', clim=(-1,1), cmap='magma')
		ax[1].imshow(self.L, aspect='auto', clim=(-2,2), cmap='magma')
		ax[2].imshow(loss, aspect='auto', clim=(-1,1), cmap='magma')

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

	action_reward_size = actions.shape[1] + rewards.shape[1]
	training_data_size = np.int32(stimuli.shape[0]*(1-par['test_sample_prop']))

	aggregate_events = np.concatenate([stimuli, actions, rewards], axis=1)
	np.random.shuffle(aggregate_events)
	vector_size = aggregate_events.shape[1]

	training_events = aggregate_events[:training_data_size,:]
	testing_events  = aggregate_events[training_data_size:,:]

	hippocampus = Hippocampus(vector_size, action_reward_size)
	hippocampus.train_events(training_events)
	hippocampus.test_events(testing_events)
	hippocampus.eval_events(testing_events)


if __name__ == '__main__':

	event_data_file = sys.argv[1]
	event_data = pickle.load(open(event_data_file, 'rb'))

	main(event_data)