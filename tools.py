import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

import os
import pickle
import pandas as pd

def profile_model(create_model_func):
	# taken from https://stackoverflow.com/questions/45085938/tensorflow-is-there-a-way-to-measure-flops-for-a-model

	# create_model_func: fn that creates a model (python func)

	run_meta = tf.RunMetadata()
	with tf.Session(graph=tf.Graph()) as sess:
		K.set_session(sess)
		net = create_model_func()

		opts = tf.profiler.ProfileOptionBuilder.float_operation()
		flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

		opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
		params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

	print('total float ops: {}'.format(flops.total_float_ops))
	print('total num params: {}'.format(params.total_parameters))

	return flops, params

def read_robustness_packets(packets_loc):
	# reads robustness data dumps

	pieces = []
	for packet in os.listdir(packets_loc):
		if '.packet' not in packet:
			continue

		packet_data = pickle.load(open(os.path.join(packets_loc, packet), 'rb'))
		#hack to retrieve model names that are otherwise lost ..
		model_unofficial_name = [i for i in packet.split('_') if 'CNN' in i or 'ECNN' in i]
		model_random_gaze_tag = [not ('nonrandomgaze' in i) for i in packet.split('-') if ('randomgaze' in i) or ('nonrandomgaze' in i)]
		assert(len(model_unofficial_name) == 1), model_unofficial_name
		assert(len(model_random_gaze_tag) == 1), model_random_gaze_tag
		model_unofficial_name = model_unofficial_name[0]
		model_random_gaze_tag = model_random_gaze_tag[0]
		packet_data['model_tag'] = [model_unofficial_name] * len(packet_data['epsilon'])
		packet_data['random_gaze'] = [model_random_gaze_tag] * len(packet_data['epsilon']) 

		pieces.append(pd.DataFrame.from_records(packet_data))

	return pd.concat(pieces)

def _compat_read_robustness_packets(packets_loc):
	# reads robustness data dumps

	pieces = []
	for packet in os.listdir(packets_loc):
		if packet == 'touch':
			continue
		packet_code = packet.split('.')[0].split('_')[1]
		packet_metric, packet_name = packet_code.split('-')

		packet_data = pickle.load(open(os.path.join(packets_loc, packet), 'rb'))
		packet_data['model_name'] = [packet_name] * len(packet_data['epsilon'])
		packet_data['PGD_metric'] = [packet_metric] * len(packet_data['epsilon'])
		pieces.append(pd.DataFrame.from_records(packet_data))

	return pd.concat(pieces)
