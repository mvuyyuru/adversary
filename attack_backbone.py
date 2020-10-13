#adversarial attack tools

import sys
sys.path.append('../foolbox')


import tensorflow as tf
import numpy as np
import foolbox as fb

from tqdm import tqdm

def build_ensemble(build_model, save_file, ensemble_size, input_size, random_gaze, gaze_val=None, load_by_name=True):
	#build an avg ensemble of models

	if not random_gaze:
		if gaze_val is None:
			raise ValueError

	gazes = [[0,0], [-gaze_val, gaze_val], [gaze_val, -gaze_val], [-gaze_val, -gaze_val], [gaze_val, gaze_val]]

	model_input = tf.keras.layers.Input(shape=input_size)

	models = []
	for i in range(ensemble_size):
		if not random_gaze:
			_model = build_model(gaze=gazes[i])
		else:
			_model = build_model()

		_model.load_weights(save_file, by_name=load_by_name)
		models.append(_model)

	model_preds = [m(model_input) for m in models]

	model_output = tf.keras.layers.Average()(model_preds)

	model = tf.keras.models.Model(inputs=model_input, outputs=model_output)

	print(_model.summary())
	return model

def run_attack(x, y, model, epsilon, algo, metric, iterations, step_size, targeted, det_targeted, random_start, batches=3, return_early=True):
	#carries out typical attack routine

	#init foolbox model
	fb_model = fb.models.TensorFlowEagerModel(model, bounds=(0.0, 1.0))

	#attack inits
	attack = None
	attack_distance = None

	if algo == 'PGD' and metric == 'LINF':
		attack = fb.attacks.LinfinityBasicIterativeAttack
		attack_distance = fb.distances.Linfinity
	elif algo == 'PGD' and metric == 'L2':
		attack = fb.attacks.L2BasicIterativeAttack
		attack_distance = fb.distances.MSE
	elif algo == 'PGD' and metric == 'L1':
		attack = fb.attacks.L1BasicIterativeAttack
		attack_distance = fb.distances.MAE
	elif algo == 'FGSM' and metric == 'LINF':
		attack = fb.attacks.GradientSignAttack
		attack_distance = fb.distances.Linfinity
		assert(iterations == 1)
		assert(step_size == -1)
		assert(not random_start)
	elif algo == 'PGD_ADAM' and metric == 'LINF':
		attack = fb.attacks.AdamProjectedGradientDescentAttack
		attack_distance = fb.distances.Linfinity


	if not targeted:
		y_adv = y.copy()
	else:
		y_adv = y.copy()
		if det_targeted:
			y_adv = np.mod(y_adv + 1, 10)
		else:
			raise NotImplementedError
			#but this has to be done for the ENTIRE dataset (probably in adversary.py)
			#since the new indices have to be consistent across models which will
			#have a different splits for y_correctly_predicted
			#if cache exists:
			#	load from cache
			#else:
			#	build new indices (make sure consistent across models)
			#	y_adv = np.random.shuffle(y_adv)

	x_adv = []

	#per adv class
	for y_adv_class in tqdm(np.unique(y_adv)):
		x_class = x[y_adv == y_adv_class]
		y_class = y[y_adv == y_adv_class]
		assert(len(x_class) == len(y_class))
		assert(len(x_class) == np.sum(y_adv == y_adv_class))

		#loop over the samples in a batched manner
		x_batches = np.array_split(x_class, batches)
		y_batches = np.array_split(y_class, batches)

		#init the attack
		if not targeted:
			#attack_criteria = fb.criteria.TopKMisclassification(3)
			attack_criteria = fb.criteria.TopKMisclassification(1)
		else:
			attack_criteria = fb.criteria.TargetClassProbability(y_adv_class, p=0.8)
			#attack_criteria = fb.criteria.TargetClass(y_adv_class)
		fb_attack = attack(model=fb_model, criterion=attack_criteria, distance=attack_distance)

		for x_batch, y_batch in zip(x_batches, y_batches):	

			if algo == 'FGSM':
				x_class_adv = fb_attack(x_batch, y_batch, epsilons=[epsilon], max_epsilon=epsilon)
			elif 'PGD' in algo:
				x_class_adv = fb_attack(x_batch, y_batch, binary_search=False, epsilon=epsilon, stepsize=(epsilon/0.3)*step_size, iterations=iterations, return_early=return_early, random_start=random_start)
			else:
				raise NotImplementedError

			#remove nans
			x_class_adv = x_class_adv[~np.isnan(x_class_adv).any(axis=(1,2,3))]

			x_adv.extend(x_class_adv)

	x_adv = np.array(x_adv)
	assert(~np.any(np.isnan(x_adv)))

	return x_adv

def build_nonrobust_features(model, x, y, y_adv_ref, batch_size=500):
	raise NotImplementedError
	#metric=LINF by default with PGD

	#init foolbox model
	fb_model = fb.models.TensorFlowEagerModel(model, bounds=(0.0, 1.0))

	x_adv = []
	y_adv = []

	#per adv class
	for y_adv_class in np.unique(y_adv_ref):

		x_class = x[y_adv_ref == y_adv_class]
		y_class = y[y_adv_ref == y_adv_class]
		y_adv_ref_class = y_adv_ref[y_adv_ref == y_adv_class]

		#sanity check that the number of samples is consistent
		assert(len(x_class) == len(y_class))
		assert(len(y_class) == len(y_adv_ref_class))

		#check that batch size is appropriate for batching logic
		assert(len(x_class) % batch_size == 0), 'len(x_class)={}, batch_size={} NOT PERFECTLY DIVISIBLE'.format(len(x_class), batch_size)
		num_batches = len(x_class) // batch_size

		#loop over the samples in a batched manner		
		x_batches = np.split(x_class, num_batches)
		y_batches = np.split(y_class, num_batches)
		y_batches_adv = np.split(y_adv_ref_class, num_batches)

		for x_batch, y_batch, y_batch_adv in tqdm(zip(x_batches, y_batches, y_batches_adv)):
			fb_attack = fb.attacks.L2BasicIterativeAttack(model=fb_model, criterion=fb.criteria.TargetClass(y_adv_class), distance=fb.distances.MSE)
			x_batch_adv = fb_attack(x_batch, y_batch, binary_search=False, epsilon=0.5, stepsize=0.1, iterations=50, return_early=False, random_start=False)

			#remove nans
			y_batch_adv = y_batch_adv[~np.isnan(x_batch_adv).any(axis=(1,2,3))]
			x_batch_adv = x_batch_adv[~np.isnan(x_batch_adv).any(axis=(1,2,3))]

			print('adding {} samples.'.format(len(x_batch_adv)))

			x_adv.extend(x_batch_adv)
			y_adv.extend(y_batch_adv)

	x_adv = np.array(x_adv)
	y_adv = np.array(y_adv)

	assert(~np.any(np.isnan(x_adv)))
	assert(~np.any(np.isnan(y_adv)))

	return x_adv, y_adv
