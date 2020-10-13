#adversarial robustness eval

import os
import pickle
import warnings
import argparse
import datasets
import model_backbone
import attack_backbone

import numpy as np
import tensorflow as tf
from functools import partial

#patch for deterministic tensorflow ops on tf20
from tfdeterminism import patch
patch()


#input args
parser = argparse.ArgumentParser()

parser.add_argument('--name')
parser.add_argument('--model')
parser.add_argument('--dataset')

#model options
parser.add_argument('--sampling')
parser.add_argument('--coarse_fixations')
parser.add_argument('--single_scale', default=0)
parser.add_argument('--scale4_freeze', default=0)
parser.add_argument('--branched_network', default=0)
parser.add_argument('--upsample_fixations', default=0)
parser.add_argument('--blur_fixations', default=0)
parser.add_argument('--pooling', default='None')
parser.add_argument('--cifar_ecnn', default=0)
#model ensemble options
parser.add_argument('--random_gaze', default=0)

#eval options
parser.add_argument('--evaluate_mode')
parser.add_argument('--attack_algo', default='PGD')
parser.add_argument('--attack_iterations', default=5)
parser.add_argument('--attack_step_size', default=0.1) #normalized against epsilon=0.3
parser.add_argument('--attack_distance_metric', default='LINF')
parser.add_argument('--attack_criteria_targeted', default=0)
parser.add_argument('--attack_criteria_det', default=0)
parser.add_argument('--attack_random_init', default=0)

#quick workaround to parallelize building nonrobust features
parser.add_argument('--staggered_build', default=0)
parser.add_argument('--staggered_build_code', default=0)

args = vars(parser.parse_args())

name = str(args['name'])
model = str(args['model'])
dataset = str(args['dataset'])
sampling  = bool(int(args['sampling']))
coarse_fixations = bool(int(args['coarse_fixations']))
single_scale = bool(int(args['single_scale']))
scale4_freeze = bool(int(args['scale4_freeze']))
branched_network = bool(int(args['branched_network']))
upsample_fixations = bool(int(args['upsample_fixations']))
blur_fixations = bool(int(args['blur_fixations']))
pooling = str(args['pooling'])
cifar_ecnn = bool(int(args['cifar_ecnn']))
random_gaze = bool(int(args['random_gaze']))

augment = False
auxiliary = False
dropout = False
pooling = None if pooling == 'None' else pooling 
scales = 'scale4' if single_scale else 'all'

evaluate_mode = str(args['evaluate_mode'])
attack_algo = str(args['attack_algo'])
attack_iterations = int(args['attack_iterations'])
attack_step_size = float(args['attack_step_size'])
attack_distance_metric = str(args['attack_distance_metric'])
attack_criteria_targeted = bool(int(args['attack_criteria_targeted']))
attack_criteria_det = bool(int(args['attack_criteria_det']))
attack_random_init = bool(int(args['attack_random_init']))
staggered_build = bool(int(args['staggered_build']))
staggered_build_code = int(args['staggered_build_code'])

save_file = 'model_checkpoints/{}.h5'.format(name)

num_classes = 100 if dataset == 'imagenet100' else 10

if not os.path.exists(save_file):
	raise ValueError

#set defaults
if evaluate_mode not in ['robustness', 'nonrobust_features']:
	raise ValueError

if evaluate_mode == 'robustness':
	#evaluate adversarial robustness
	assert(not staggered_build)
	assert(staggered_build_code == 0)

	#defaults
	ensemble_size = 5

	#supported settings
	if attack_distance_metric not in ['LINF', 'L2', 'L1']:
		raise ValueError

	if attack_algo not in ['PGD', 'PGD_ADAM', 'FGSM']:
		raise ValueError

elif evaluate_mode == 'nonrobust_features':
	#evaluate generalizability of nonrobust features

	#add checks for expected settings, supported settings
	#implement imagenet10 nonrobust features eval, etc.etc. see todo
	raise NotImplementedError

	assert(attack_criteria_targeted)


	if not random_gaze:
		raise NotImplementedError

	#defaults
	ensemble_size = 3

	if not attack_criteria_det:
		random_relabel = True
		adv_save_file = 'model_checkpoints/nonrobust_rand_{}.h5'.format(name)
	else:
		random_relabel = False
		adv_save_file = 'model_checkpoints/nonrobust_norand_{}.h5'.format(name)

	if os.path.exists(adv_save_file):
		raise ValueError

if model == 'resnet' or model == 'resnet_cifar':
	if sampling or coarse_fixations or cifar_ecnn:
		stochastic_model = True
	else:
		stochastic_model = False
elif model == 'ecnn':
	if single_scale and scale4_freeze:
		stochastic_model = False
	else:
		stochastic_model = True

#build models
model_tag = model
if model == 'resnet_cifar':
	#check params
	assert(auxiliary is False)
	assert(single_scale is False)
	assert(scale4_freeze is False)
	assert(branched_network is False)
	assert(blur_fixations is False)


	if coarse_fixations:
		if upsample_fixations:
			base_model_input_shape = (32, 32, 3)
			gaze_val = 8
		else:
			base_model_input_shape = (24, 24, 3)
			gaze_val = 4
	else:
		if cifar_ecnn:
			base_model_input_shape = (15, 15, 3)
		else:
			base_model_input_shape = (32, 32, 3)

		if sampling:
			gaze_val = 8
		elif cifar_ecnn:
			gaze_val = 1
		else:
			gaze_val = None
	
	def build_model(augment=False, gaze=None):
		#augment as optional arg for nonrobust feature construction
		return model_backbone.resnet_cifar(base_model_input_shape=base_model_input_shape, augment=augment, sampling=sampling, coarse_fixations=coarse_fixations, coarse_fixations_upsample=upsample_fixations, gaze=gaze, return_logits=True, num_classes=num_classes, approx_ecnn=cifar_ecnn)

	if not stochastic_model:
		model = build_model()
		model.load_weights(save_file, by_name=False)
	else:
		model = attack_backbone.build_ensemble(build_model=build_model, save_file=save_file, ensemble_size=ensemble_size, input_size=(32,32,3), random_gaze=random_gaze, gaze_val=gaze_val, load_by_name=False)

elif model == 'resnet':
	#check params
	assert(auxiliary is False)
	assert(single_scale is False)
	assert(scale4_freeze is False)
	assert(not (upsample_fixations and blur_fixations))

	if coarse_fixations:
		if (not upsample_fixations) and (not blur_fixations):
			base_model_input_shape = (224, 224, 3)
			gaze_val = 48
		elif upsample_fixations:
			base_model_input_shape = (320, 320, 3)
			gaze_val = 80
		elif blur_fixations:
			base_model_input_shape = (240, 240, 3)
			gaze_val = 40
		else:
			raise ValueError
	else:
		base_model_input_shape = (320, 320, 3)
		if sampling:
			gaze_val = 80
		else:
			gaze_val = None

	def build_model(gaze=None):
		return model_backbone.resnet(base_model_input_shape=base_model_input_shape, augment=augment, sampling=sampling, coarse_fixations=coarse_fixations, coarse_fixations_upsample=upsample_fixations, coarse_fixations_gaussianblur=blur_fixations, branched_network=branched_network, gaze=gaze, return_logits=True, num_classes=num_classes)

	if not stochastic_model:
		model = build_model()
		model.load_weights(save_file, by_name=True)
	else:
		model = attack_backbone.build_ensemble(build_model=build_model, save_file=save_file, ensemble_size=ensemble_size, input_size=(320, 320, 3), random_gaze=random_gaze, gaze_val=gaze_val, load_by_name=True)

elif model == 'ecnn':
	#check params
	assert(coarse_fixations is False)
	assert(upsample_fixations is False)
	assert(blur_fixations is False)
	if single_scale:
		assert(auxiliary is False)

	if scale4_freeze and single_scale:
		gaze_val = None
	elif not scale4_freeze:
		gaze_val = 40
	elif scale4_freeze:
		gaze_val = 80
	else:
		raise ValueError

	def build_model(gaze=None):
		return model_backbone.ecnn(augment=augment, auxiliary=auxiliary, sampling=sampling, scales=scales, pooling=pooling, dropout=dropout, scale4_freeze=scale4_freeze, gaze=gaze, return_logits=True, num_classes=num_classes)

	if not stochastic_model:
		model = build_model()
		model.load_weights(save_file, by_name=True)
	else:
		model = attack_backbone.build_ensemble(build_model=build_model, save_file=save_file, ensemble_size=ensemble_size, input_size=(320, 320, 3), random_gaze=random_gaze, gaze_val=gaze_val, load_by_name=True)

else:
	raise ValueError

model.summary()

#load datasets
if dataset == 'imagenet10':
	x_train, y_train, x_test, y_test = datasets.load_imagenet10(only_test=True, only_bbox=False)
elif dataset == 'bbox_imagenet10':
	x_train, y_train, x_test, y_test = datasets.load_imagenet10(only_test=True, only_bbox=True)
elif dataset == 'cifar10':
	x_train, y_train, x_test, y_test = datasets.load_cifar10(only_test=True)
elif dataset == 'imagenet100':
	x_train, y_train, x_test, y_test = datasets.load_imagenet(data_dir='imagenet100', only_test=True)
elif dataset == 'integer_cifar10':
	raise NotImplementedError
	#x_train, y_train, x_test, y_test = datasets.load_integer_cifar10(only_test=True)
elif dataset == 'test10':
	raise NotImplementedError
	#x_train, y_train, x_test, y_test = datasets.load_test10(batch_size)
else:
	raise ValueError

#run adversary evaluations
if evaluate_mode == 'robustness':
	#evaluate adversarial robustness

	#container for robustness results
	robustness_packet = {}

	robustness_packet['epsilon'] = []
	robustness_packet['vanilla_accuracy'] = []
	robustness_packet['vanilla_loss'] = []
	robustness_packet['adversarial_accuracy'] = []
	robustness_packet['model_name'] = []
	robustness_packet['attack_algo'] = []
	robustness_packet['attack_distance_metric'] = []
	robustness_packet['attack_iterations'] = []
	robustness_packet['attack_step_size'] = []
	robustness_packet['attack_criteria_targeted'] = []
	robustness_packet['attack_criteria_det'] = []
	robustness_packet['attack_random_init'] = []

	attack_criteria_targeted_tag = 'targeted' if attack_criteria_targeted else 'untargeted'
	attack_criteria_det_tag = 'dettarget' if attack_criteria_det else 'nondettarget'
	attack_random_init_tag = 'randinit' if attack_random_init else 'nonrandinit'
	random_gaze_tag = 'randomgaze' if random_gaze else 'nonrandomgaze'
	robustness_packet_loc = './cluster_runs/adversary/{}_{}_{}-{}-{}-{}-{}-{}-{}-{}.packet'.format(evaluate_mode, name, attack_algo, attack_distance_metric, attack_iterations, attack_step_size, attack_criteria_targeted_tag, attack_criteria_det_tag, attack_random_init_tag, random_gaze_tag)

	if os.path.exists(robustness_packet_loc):
		raise ValueError

	epsilons = [0.5, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
	print('scanning epsilons: {}'.format(epsilons))

	x_test_backup = x_test.copy()
	y_test_backup = y_test.copy()

	for e in epsilons:

		#refresh copy of test images
		x_test = x_test_backup.copy()
		y_test = y_test_backup.copy()		

		#sanity check manual accuracy calculation
		y_test_pred = model.predict(x_test, verbose=0)
		vanilla_loss = np.mean(tf.keras.losses.categorical_crossentropy(y_true=y_test, y_pred=y_test_pred, from_logits=True))
		print('loss: {}'.format(vanilla_loss))

		y_test_pred = np.argmax(y_test_pred, axis=-1)
		y_test = np.argmax(y_test, axis=-1)
		vanilla_accuracy = np.sum(y_test_pred == y_test)/len(y_test)
		print('accuracy: {}'.format(vanilla_accuracy))

		num_vanilla_mispredicted = np.sum(y_test_pred != y_test)

		x_test_correctly_predicted = x_test[y_test_pred == y_test]
		y_test_correctly_predicted = y_test[y_test_pred == y_test]

		assert(len(x_test_correctly_predicted) + num_vanilla_mispredicted == len(x_test))

		x_test_adv = attack_backbone.run_attack(x_test_correctly_predicted, y_test_correctly_predicted, model, epsilon=e, algo=attack_algo, metric=attack_distance_metric, iterations=attack_iterations, step_size=attack_step_size, targeted=attack_criteria_targeted, det_targeted=attack_criteria_det, random_start=attack_random_init)

		num_adv_examples = len(x_test_adv)
		assert(num_adv_examples <= len(x_test) - num_vanilla_mispredicted)

		assert(num_adv_examples == len(x_test_adv))
		assert(num_vanilla_mispredicted == np.sum(y_test_pred != y_test))
		adversarial_accuracy = 1. - ((num_vanilla_mispredicted + num_adv_examples)/len(x_test))
		print('(epsilon={}) accuracy: {}'.format(e, adversarial_accuracy))

		#save results to container
		robustness_packet['epsilon'].append(e)
		robustness_packet['vanilla_accuracy'].append(vanilla_accuracy)
		robustness_packet['vanilla_loss'].append(vanilla_loss)
		robustness_packet['adversarial_accuracy'].append(adversarial_accuracy)
		robustness_packet['model_name'].append(model_tag)
		robustness_packet['attack_algo'].append(attack_algo)
		robustness_packet['attack_distance_metric'].append(attack_distance_metric)
		robustness_packet['attack_iterations'].append(attack_iterations)
		robustness_packet['attack_step_size'].append(attack_step_size)
		robustness_packet['attack_criteria_targeted'].append(attack_criteria_targeted)
		robustness_packet['attack_criteria_det'].append(attack_criteria_det)
		robustness_packet['attack_random_init'].append(attack_random_init)

	pickle.dump(robustness_packet, open(robustness_packet_loc, 'wb'))

elif evaluate_mode == 'nonrobust_features':
	#evaluate generalizability of nonrobust features

	if model_tag != 'resnet_cifar':
		raise NotImplementedError
	if dataset != 'cifar10':
		raise NotImplementedError

	#generate adversarial dataset
	x_train_adv, y_train_adv, x_test, y_test = datasets.load_nonrobust_cifar10(model=model, name=name, random_relabel=random_relabel, cache=True, staggered_build=staggered_build, staggered_build_code=staggered_build_code)

	if staggered_build:
		raise NotImplementedError('proceed with manually merging staggered build files and re-running without staggered build.')

	#train, evaluate
	model = build_model(augment=True)

	epochs=200
	base_lr=1e-3
	batch_size=128
	checkpoint_interval=999

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

	#create model
	model.summary()

	lr_schedule_filled = partial(lr_schedule, base_lr=base_lr)
	model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=base_lr), metrics=['accuracy'])

	#create training callbacks
	lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule_filled, verbose=1)
	#lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

	random_tag = 'random_relabel' if random_relabel else 'nonrandom_relabel'
	oldest_model_saver = tf.keras.callbacks.ModelCheckpoint(filepath='{}'.format(adv_save_file), save_best_only=False, save_weights_only=True, verbose=1)
	interval_model_saver = tf.keras.callbacks.ModelCheckpoint(filepath='{}-'.format(adv_save_file)+'{epoch:03d}', period=checkpoint_interval, save_best_only=False, save_weights_only=True, verbose=1)

	callbacks = [lr_scheduler, oldest_model_saver, interval_model_saver]

	model.fit(x_train_adv, y_train_adv, batch_size=batch_size, shuffle=True, validation_data=(x_test, y_test), epochs=epochs, callbacks=callbacks, verbose=1)

	scores = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss (w. augmentation !!!):', scores[0])
	print('Test accuracy (w. augmentation !!!):', scores[1])
