#model trainer

import os
import warnings
import argparse
import datasets
import model_backbone
import attack_backbone

import numpy as np
import tensorflow as tf

from functools import partial

#input args
parser = argparse.ArgumentParser()

parser.add_argument('--name')
parser.add_argument('--model')
parser.add_argument('--dataset', default='imagenet10')

parser.add_argument('--augment')
parser.add_argument('--sampling')
parser.add_argument('--coarse_fixations')
parser.add_argument('--auxiliary')

parser.add_argument('--single_scale', default=0)
parser.add_argument('--scale4_freeze', default=0)
parser.add_argument('--branched_network', default=0)
parser.add_argument('--upsample_fixations', default=0)
parser.add_argument('--blur_fixations', default=0)
parser.add_argument('--pooling', default='None')
parser.add_argument('--dropout', default=0)
parser.add_argument('--cifar_ecnn', default=0)

parser.add_argument('--only_evaluate', default=0)
args = vars(parser.parse_args())

name = str(args['name'])
model = str(args['model'])
dataset = str(args['dataset'])
sampling  = bool(int(args['sampling']))
coarse_fixations = bool(int(args['coarse_fixations']))
auxiliary = bool(int(args['auxiliary']))
augment = bool(int(args['augment']))
single_scale = bool(int(args['single_scale']))
scale4_freeze = bool(int(args['scale4_freeze']))
branched_network = bool(int(args['branched_network']))
upsample_fixations = bool(int(args['upsample_fixations']))
blur_fixations = bool(int(args['blur_fixations']))
pooling = str(args['pooling'])
dropout = bool(int(args['dropout']))
only_evaluate = bool(int(args['only_evaluate']))
cifar_ecnn = bool(int(args['cifar_ecnn']))

pooling = None if pooling == 'None' else pooling 
scales = 'scale4' if single_scale else 'all'

if dataset == 'test10':
	warnings.warn('running in test mode!')

save_file = 'model_checkpoints/{}.h5'.format(name)

if only_evaluate:
	if not os.path.exists(save_file):
		raise ValueError
else:
	if os.path.exists(save_file):
		raise ValueError

distribution = tf.distribute.MirroredStrategy()

if dataset == 'cifar10' or dataset == 'imagenet10' or dataset == 'bbox_imagenet10':
	num_classes = 10
elif dataset == 'imagenet100':
	num_classes = 100
elif dataset == 'imagenet':
	num_classes = 1000
else:
	raise ValueError

with distribution.scope():	

	model_tag = model
	#build network
	if model == 'resnet':
		#check params
		assert(auxiliary is False)
		assert(single_scale is False)
		assert(scale4_freeze is False)
		assert(not (upsample_fixations and blur_fixations))

		if coarse_fixations:
			if upsample_fixations:
				base_model_input_shape = (320,320,3)
			elif blur_fixations:
				base_model_input_shape = (240,240,3)
			else:
				base_model_input_shape = (224,224,3)
		else:
			base_model_input_shape = (320,320,3)

		model = model_backbone.resnet(base_model_input_shape=base_model_input_shape, num_classes=num_classes, augment=augment, sampling=sampling, coarse_fixations=coarse_fixations, coarse_fixations_upsample=upsample_fixations, coarse_fixations_gaussianblur=blur_fixations, branched_network=branched_network)
		if only_evaluate:
			model.load_weights(save_file, by_name=True)

	elif model == 'resnet_cifar':
		#check params
		assert(auxiliary is False)
		assert(single_scale is False)
		assert(scale4_freeze is False)
		assert(blur_fixations is False)

		if coarse_fixations:
			if upsample_fixations:
				base_model_input_shape = (32, 32, 3)
			else:
				base_model_input_shape = (24, 24, 3)
		else:
			if cifar_ecnn:
				base_model_input_shape = (15, 15, 3)
			else:
				base_model_input_shape = (32, 32, 3)

		model = model_backbone.resnet_cifar(base_model_input_shape=base_model_input_shape, augment=augment, sampling=sampling, coarse_fixations=coarse_fixations, coarse_fixations_upsample=upsample_fixations, approx_ecnn=cifar_ecnn)
		if only_evaluate:
			#convention for resnet_cifar is to not use the name
			model.load_weights(save_file, by_name=False)

	elif model == 'ecnn':
		#check params
		assert(coarse_fixations is False)
		assert(upsample_fixations is False)
		assert(blur_fixations is False)
		if single_scale:
			assert(auxiliary is False)

		model = model_backbone.ecnn(num_classes=num_classes, augment=augment, auxiliary=auxiliary, sampling=sampling, scales=scales, pooling=pooling, dropout=dropout, scale4_freeze=scale4_freeze)
		if only_evaluate:
			model.load_weights(save_file, by_name=True)
			
	else:
		raise ValueError

	model.summary()

	#load dataset, set defaults
	if dataset == 'imagenet10' or dataset == 'bbox_imagenet10':
		epochs=400
		base_lr=1e-3
		batch_size=64
		checkpoint_interval=999
		optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)

		if dataset == 'imagenet10':
			x_train, y_train, x_test, y_test = datasets.load_imagenet10(only_test=only_evaluate, only_bbox=False)
		elif dataset == 'bbox_imagenet10':
			x_train, y_train, x_test, y_test = datasets.load_imagenet10(only_test=only_evaluate, only_bbox=True)

		def lr_schedule(epoch, lr, base_lr):
			#keeps learning rate to a schedule

			if epoch > 360:
				lr = base_lr * 0.5e-3
			elif epoch > 320:
				lr = base_lr * 1e-3
			elif epoch > 240:
				lr = base_lr * 1e-2
			elif epoch > 160:
				lr = base_lr * 1e-1

			return lr
	elif dataset == 'imagenet100' or dataset == 'imagenet':
		epochs = 130
		base_lr=1e-1
		batch_size=256 #128 #1024 #256
		checkpoint_interval=999
		optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr, decay=1e-4, momentum=0.9)

		if dataset == 'imagenet100':
			steps_per_epoch = 502 #1004 #126 #502
			validation_steps = 20 #40 #5 #20
		else:
			raise NotImplementedError

		train_dataset, test_dataset = datasets.load_imagenet(data_dir=dataset, only_test=only_evaluate, aux_labels=auxiliary, batch_size=batch_size)

		def lr_schedule(epoch, lr, base_lr):
			#keeps learning rate to a schedule

			if epoch > 120:
				lr = base_lr * 0.5e-3
			elif epoch > 90:
				lr = base_lr * 1e-3
			elif epoch > 60:
				lr = base_lr * 1e-2
			elif epoch > 30:
				lr = base_lr * 1e-1

			return lr
	elif dataset == 'cifar10' or dataset == 'integer_cifar10':
		epochs=200
		base_lr=1e-3
		batch_size=128
		checkpoint_interval=999
		optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)

		if dataset == 'cifar10':
			x_train, y_train, x_test, y_test = datasets.load_cifar10(only_test=only_evaluate)
		elif dataset == 'integer_cifar10':
			x_train, y_train, x_test, y_test = datasets.load_integer_cifar10(only_test=only_evaluate)

		def lr_schedule(epoch, lr, base_lr):
			#keeps learning rate to a schedule

		    if epoch > 180:
		        lr = base_lr * 0.5e-3
		    elif epoch > 160:
		        lr = base_lr * 1e-3
		    elif epoch > 120:
		        lr = base_lr * 1e-2
		    elif epoch > 80:
		        lr = base_lr * 1e-1

		    return lr
	elif dataset == 'test10':
		epochs=3
		base_lr=1e-6
		batch_size=2
		checkpoint_interval=2
		optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)

		input_size = 32 if model_tag == 'resnet_cifar' else 320
		x_train, y_train, x_test, y_test = datasets.load_test10(batch_size, input_size=input_size)

		def lr_schedule(epoch, lr, base_lr):
			return lr
	else:
		raise ValueError

	if only_evaluate:		
		model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.), metrics=['accuracy'])
	else:
		lr_schedule_filled = partial(lr_schedule, base_lr=base_lr)
		model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

		#create training callbacks
		lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule_filled, verbose=1)
		#lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
		oldest_model_saver = tf.keras.callbacks.ModelCheckpoint(filepath=save_file, save_best_only=False, save_weights_only=True, verbose=1)
		#interval_model_saver = tf.keras.callbacks.ModelCheckpoint(filepath='model_checkpoints/{}-'.format(name)+'{epoch:03d}.h5', period=checkpoint_interval, save_best_only=False, save_weights_only=True, verbose=1)

		callbacks = [lr_scheduler, oldest_model_saver]

		if dataset == 'imagenet100' or dataset =='imagenet':

			#stream from tfrecords
			#note: does not exactly partition train/test epochs
			model.fit(train_dataset, steps_per_epoch=steps_per_epoch, validation_data=test_dataset, validation_steps=validation_steps, epochs=epochs, callbacks=callbacks, verbose=1)
		else:
			#fit directly from memory
			if not auxiliary:
				model.fit(x_train, y_train, batch_size=batch_size, shuffle=True, validation_data=(x_test, y_test), epochs=epochs, callbacks=callbacks, verbose=1)
			else:
				model.fit(x_train, [y_train, y_train, y_train, y_train, y_train], batch_size=batch_size, shuffle=True, validation_data=(x_test, [y_test, y_test, y_test, y_test, y_test]), epochs=epochs, callbacks=callbacks, verbose=1)

	#evaluate model
	#repeats by default (sanity check for model stochasticit)
	repeats = 3

	for _ in range(repeats):

		if dataset == 'imagenet100' or dataset == 'imagenet':
			scores = model.evaluate(test_dataset, steps=validation_steps, verbose=0)
		else:
			if not auxiliary:	
				scores = model.evaluate(x_test, y_test, verbose=0)
			else:
				scores = model.evaluate(x_test, [y_test, y_test, y_test, y_test, y_test], verbose=0)

		print('({})Test loss: {}.'.format(_, scores[0]))
		print('({})Test accuracy: {}.'.format(_, scores[1]))
