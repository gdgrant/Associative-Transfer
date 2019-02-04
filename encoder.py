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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Encoder:

	def __init__(self, input_data, W=None, U=None):

		if type(W) is type(None):
			self.W = tf.get_variable('W', initializer=tf.random_uniform_initializer(-0.5,0.5), shape=[par['n_input'], par['n_latent']])
		else:
			self.W = tf.get_variable('W', initializer=W, trainable=False)

		if type(U) is type(None):
			self.U = tf.get_variable('U', initializer=tf.random_uniform_initializer(-0.5,0.5), shape=[par['n_latent'], par['n_input']])
		else:
			self.U = tf.get_variable('U', initializer=U, trainable=False)

		self.I = input_data

		self.E = []
		self.R = []
		for t in range(input_data.shape.as_list()[0]):
			E = tf.nn.relu(self.I[t] @ self.W)
			R = E @ self.U

			self.E.append(E)
			self.R.append(R)

		self.E = tf.stack(self.E, axis=0)
		self.R = tf.stack(self.R, axis=0)

		self.loss_plot = 0.5 * tf.square(self.I - self.R)

		self.rec_loss = tf.reduce_mean(self.loss_plot)
		self.act_loss = par['enc_activity_cost'] * tf.reduce_mean(tf.log(1+tf.abs(self.E)))
		self.wei_loss = par['enc_weight_cost'] * tf.reduce_mean(tf.abs(self.U))
		total_loss = self.rec_loss + self.act_loss + self.wei_loss

		if type(W) is type(None) or type(U) is type(None):
			opt = AdamOpt.AdamOpt(tf.trainable_variables(), learning_rate=0.01)
			self.train = opt.compute_gradients(total_loss)


def train_encoder(gpu_id=None):

	if gpu_id is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, [par['num_time_steps'], None, par['n_input']], 'input')

	update_parameters({'noise_in':0.05})
	stim = stimulus.Stimulus()

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8) if gpu_id == '0' else tf.GPUOptions()
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			encoder = Encoder(x)

		sess.run(tf.global_variables_initializer())

		print('\nTraining encoder for {} task:'.format(par['task']))
		for i in range(10001):

			name, trial_info = stim.generate_trial()
			feed_dict = {x:trial_info['neural_input']}
			_, rec_loss, act_loss, wei_loss, enc, rec, loss_plot = \
				sess.run([encoder.train, encoder.rec_loss, \
					encoder.act_loss, encoder.wei_loss, encoder.E, \
					encoder.R, encoder.loss_plot], feed_dict=feed_dict)

			if i%500 == 0:

				fig, ax = plt.subplots(3,5,figsize=(14,8))
				for t in range(5):
					ax[0,t].imshow(trial_info['neural_input'][:,t,:].T, aspect='auto')
					ax[1,t].imshow(rec[:,t,:].T, aspect='auto')
					ax[2,t].imshow(enc[:,t,:].T, aspect='auto')
					ax[0,t].set_title('Trial {}'.format(t))
					for p in range(3):
						ax[p,t].set_xticks([])
						ax[p,t].set_yticks([])

				ax[0,0].set_ylabel('Stimulus Input')
				ax[1,0].set_ylabel('Reconstruction')
				ax[2,0].set_ylabel('Encoding')
				ax[2,0].set_xlabel('Time')

				fig.suptitle('Input, Reconstruction, and Encoding')
				plt.savefig('./savedir/input_rec_and_enc.png', bbox_inches='tight')
				plt.clf()
				plt.close()

				print('Iter: {:>6} | Rec. Loss: {:7.5f} | Act. Loss: {:7.5f} | Wei. Loss: {:7.5f}'.format(i, rec_loss, act_loss, wei_loss))

		print('\nEncoder training complete.')

		# Record weights to check reconstruction
		W, U = sess.run([encoder.W, encoder.U])
		weights = {'W':W, 'U':U}

		if par['internal_sampling']:
			print('Sampling stimulus:')

			# Sample stimuli (enough for 200 batches, or ~100k trials, which should sample most of the distribution)
			for j in range(4):
				trial_info_list = []
				for i in range(50):

					name, trial_info = stim.generate_trial()
					feed_dict = {x:trial_info['neural_input']}
					enc, = sess.run([encoder.E], feed_dict=feed_dict)
					trial_info['encoded_input'] = enc

					trial_info_list.append(trial_info)

				aggregated_trial_info = {}
				for key in (list(trial_info.keys()) + ['encoded_input']):
					aggregated_trial_info[key] = np.concatenate([item[key] for item in trial_info_list], axis=1)


				# Put together save data and save
				save_data = {'task':name, 'trial_info':aggregated_trial_info, 'weights':weights, 'rec_loss':rec_loss, 'act_loss':act_loss}
				pickle.dump(save_data, open('./datadir/{}task_{}unit_input_encoding_part{}.pkl'.format(name, par['n_latent'], j), 'wb'))

			print('Encoded stimulus samples saved.  Model complete. \n')

		else:
			print('Saving weights...')
			save_data = {'task':name, 'weights':weights, 'rec_loss':rec_loss, 'act_loss':act_loss}
			pickle.dump(save_data, open('./datadir/{}task_{}unit_input_encoder_weights.pkl'.format(name, par['n_latent']), 'wb'))
			print('Encoder weights saved.  Model complete. \n')


def generate_encoding(gpu_id, weight_file, event_file):

	weight_data = pickle.load(open(weight_file, 'rb'))['weights']
	event_data  = pickle.load(open(event_file, 'rb'))

	W, U = weight_data['W'], weight_data['U']
	stimuli_to_encode = event_data['stimuli']

	if gpu_id is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, [1, None, par['n_input']], 'input')

	with tf.Session() as sess:

		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			encoder = Encoder(x, W=W, U=U)

		sess.run(tf.global_variables_initializer())

		events_per_batch = 500
		num_batches = stimuli_to_encode.shape[0]//events_per_batch
		encoded_stimuli = []

		print('\nEncoding event samples:')
		for i in range(num_batches):

			feed_dict={x:stimuli_to_encode[np.newaxis, i*events_per_batch:(i+1)*events_per_batch,:]}
			E, = sess.run([encoder.E], feed_dict=feed_dict)
			encoded_stimuli.append(E[0,...])

			if i%100 == 0 or i==num_batches-1:
				print('{} of {} batches complete.'.format(i+1, num_batches), end='\r')

	event_data['encoded_stimuli'] = np.concatenate(encoded_stimuli, axis=0)
	print('\n\nSaving event data with encoded stimuli...')
	pickle.dump(event_data, open(event_file, 'wb'))
	print('Updated event data saved.  Model complete. \n')


if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser(description='Train or run encoder network.')
	parser.add_argument('-g', '--gpu_id', default=None, type=str, help='GPU to be used.')
	parser.add_argument('-d', '--wedata', nargs=2, default=None, type=str, \
		help='Filenames for weight and event data to be loaded for generating encoding.')
	args = vars(parser.parse_args())

	# EX: python3 encoder.py -g 0 -d ./datadir/gotask_512unit_input_encoder_weights.pkl ./datadir/gotask_cortex_event_data.pkl

	try:
		if args['wedata'] is None:
			train_encoder(args['gpu_id'])
		else:
			print('Weight data file: {}'.format(args['wedata'][0]))
			print('Event data file:  {}'.format(args['wedata'][1]))
			generate_encoding(args['gpu_id'], args['wedata'][0], args['wedata'][1])

	except KeyboardInterrupt:
		quit('Quit by KeyboardInterrupt.')
