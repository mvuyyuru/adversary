#build models

import glimpse

import tensorflow as tf
import tensorflow.keras.layers as layers

from functools import partial
from resnet_backbone import ResNet18
from resnet_backbone import ResNet_CIFAR

def resnet(input_shape=(320,320,3), base_model_input_shape=(224,224,3), name='CNN', num_classes=10, augment=False, sampling=False, coarse_fixations=True, coarse_fixations_upsample=False, coarse_fixations_gaussianblur=False, branched_network=False, gaze=None, return_logits=False):
	#standard ImageNet models and derivatives 

	#check args
	if branched_network:
		if sampling:
			raise ValueError
		elif coarse_fixations:
			if coarse_fixations_upsample or coarse_fixations_gaussianblur:
				raise ValueError
		else:
			raise ValueError

	if sampling and coarse_fixations:
		raise NotImplementedError

	if coarse_fixations_upsample and (not coarse_fixations):
		raise ValueError

	if coarse_fixations_gaussianblur and (not coarse_fixations):
		raise ValueError

	if coarse_fixations_upsample and coarse_fixations_gaussianblur:
		raise ValueError

	if input_shape != (320, 320, 3):
		raise ValueError

	if coarse_fixations:
		if (not coarse_fixations_upsample) and (not coarse_fixations_gaussianblur):
			if base_model_input_shape != (224, 224, 3):
				raise ValueError
		elif coarse_fixations_upsample:
			if input_shape != base_model_input_shape:
				raise ValueError
		elif coarse_fixations_gaussianblur:
			if base_model_input_shape != (240, 240, 3):
				raise ValueError
		else:
			raise ValueError

	else:
		if input_shape != base_model_input_shape:
			raise ValueError

	#base model
	if not branched_network:
		network = ResNet18(include_top=False, input_shape=base_model_input_shape, subnetwork_name=name, pooling='avg')
	else:
		network_branch1 = ResNet18(include_top=False, input_shape=base_model_input_shape, subnetwork_name='branch1_{}'.format(name), pooling='avg', filters=27)
		network_branch2 = ResNet18(include_top=False, input_shape=base_model_input_shape, subnetwork_name='branch2_{}'.format(name), pooling='avg', filters=27)
		network_branch3 = ResNet18(include_top=False, input_shape=base_model_input_shape, subnetwork_name='branch3_{}'.format(name), pooling='avg', filters=27)
		network_branch4 = ResNet18(include_top=False, input_shape=base_model_input_shape, subnetwork_name='branch4_{}'.format(name), pooling='avg', filters=27)

	model_input = layers.Input(shape=input_shape)	

	#data augmentation
	if augment:
		x = layers.Lambda(lambda tensor: glimpse.image_augmentation(tensor, dataset='imagenet10'), name='image_augmentation')(model_input)
	else:
		x = model_input 

	#preprocess
	if coarse_fixations:

		if coarse_fixations_upsample:
			fixation_size = 160
		elif coarse_fixations_gaussianblur:
			fixation_size = 240
		else:
			fixation_size = 224

		if gaze is not None:
			coarse_foveation_x = tf.constant(gaze[0], tf.int32)
			coarse_foveation_y = tf.constant(gaze[1], tf.int32)
			coarse_fixation_center = [input_shape[0] // 2 + coarse_foveation_x, input_shape[0] // 2 + coarse_foveation_y]

			x = layers.Lambda(lambda tensor: glimpse.crop_square_patch(tensor, coarse_fixation_center, fixation_size), name='coarse_fixations')(x)
		else:
			x = layers.Lambda(lambda tensor: tf.image.random_crop(tensor, size=[tf.shape(tensor)[0], fixation_size, fixation_size, 3]), name='coarse_fixations')(x)


		if coarse_fixations_upsample:
			x = layers.Lambda(lambda tensor: glimpse.uniform_upsample(tensor, factor=2), name='uniform_upsampling')(x)
		if coarse_fixations_gaussianblur:
			x = layers.Lambda(lambda tensor: glimpse.gaussian_blur(tensor, radius=6), name='gaussian_blur')(x)

	if sampling:

		if gaze is not None:
			gaze_x = tf.constant(gaze[0], tf.int32)
			gaze_y = tf.constant(gaze[1], tf.int32)
			gaze = [gaze_x, gaze_y]
		else:
			#img shape (320, 320, 3)
			gaze = 80

		warp_image_filled = partial(glimpse.warp_image, output_size=base_model_input_shape[0], input_size=base_model_input_shape[0], gaze=gaze)
		x = layers.Lambda(lambda tensor: tf.map_fn(warp_image_filled, tensor, back_prop=True), name='nonuniform_sampling')(x)

	if not branched_network:
		x = network(x)
	else:
		x1 = network_branch1(x)
		x2 = network_branch2(x)
		x3 = network_branch3(x)
		x4 = network_branch4(x)
		x = layers.concatenate([x1, x2, x3, x4])

	if not return_logits:
		model_output = layers.Dense(num_classes, activation='softmax', name='probs')(x)
	else:
		model_output = layers.Dense(num_classes, activation=None, name='probs')(x)
	model = tf.keras.models.Model(inputs=model_input, outputs=model_output)

	return model

def ecnn(input_shape=(320,320,3), base_model_input_shape=(40,40,3), name='ECNN', num_classes=10, augment=False, auxiliary=False, sampling=True, scales='all', pooling=None, dropout=False, gaze=None, scale4_freeze=False, return_logits=False):
	#ImageNet cortical sampling model 

	#check args
	if input_shape != (320, 320, 3):
		raise ValueError

	if base_model_input_shape != (40, 40, 3):
		raise ValueError

	if scales not in ['all', 'scale4']:
		raise ValueError

	if scales != 'all' and auxiliary:
		raise ValueError

	if pooling is not None:
		if pooling not in ['max', 'avg']:
			raise ValueError

	if pooling is not None and dropout:
		raise NotImplementedError

	if scales == 'scale4':
		if dropout:
			raise NotImplementedError
		if pooling is not None:
			raise ValueError

	#base models
	if scales == 'all':
		scale1_network = ResNet18(include_top=False, input_shape=base_model_input_shape, conv1_stride=1, max_pool_stride=1, filters=45, subnetwork_name='scale1-{}'.format(name), pooling='avg')	
		scale2_network = ResNet18(include_top=False, input_shape=base_model_input_shape, conv1_stride=1, max_pool_stride=1, filters=45, subnetwork_name='scale2-{}'.format(name), pooling='avg')
		scale3_network = ResNet18(include_top=False, input_shape=base_model_input_shape, conv1_stride=1, max_pool_stride=1, filters=45, subnetwork_name='scale3-{}'.format(name), pooling='avg')
	scale4_network = ResNet18(include_top=False, input_shape=base_model_input_shape, conv1_stride=1, max_pool_stride=1, filters=45, subnetwork_name='scale4-{}'.format(name), pooling='avg')		
	
	model_input = layers.Input(shape=input_shape)

	#data augmentation
	if augment:
		x = layers.Lambda(lambda tensor: glimpse.image_augmentation(tensor, dataset='imagenet10'), name='image_augmentation')(model_input)
	else:
		x = model_input 

	#preprocess
	if gaze is not None:
		gaze_x = tf.constant(gaze[0], tf.int32)
		gaze_y = tf.constant(gaze[1], tf.int32)
		gaze = [gaze_x, gaze_y]
	else:
		#img shape (320, 320, 3)
		if not scale4_freeze:
			gaze = 40
		else:
			gaze = 80

	if not scale4_freeze:
		scale_sizes = [40, 80, 160, 240]
		scale_radii = [1, 2, 4, 6]
	else:
		scale_sizes = [40, 80, 160, 320]
		scale_radii = [1, 2, 4, 8]
		
	scale_center = [input_shape[0] // 2, input_shape[0] // 2]

	if not sampling:
		scales_x = layers.Lambda(lambda tensor: glimpse.image_scales(tensor, scale_center, scale_radii, scale_sizes, gaze, scale4_freeze), name='scale_sampling')(x)
	else:
		scales_x = layers.Lambda(lambda tensor: glimpse.warp_image_and_image_scales(tensor, input_shape[0], input_shape[0], scale_center, scale_radii, scale_sizes, gaze, scale4_freeze), name='nonuniform_and_scale_sampling')(x)
		
	#unpack scales
	scale1_x = scales_x[0]
	scale2_x = scales_x[1]
	scale3_x = scales_x[2]
	scale4_x = scales_x[3]

	if scales == 'all':
		scale1_x = scale1_network(scale1_x)
		scale2_x = scale2_network(scale2_x)
		scale3_x = scale3_network(scale3_x)
	scale4_x = scale4_network(scale4_x)

	if scales == 'all':
		if pooling is None:
			x = layers.concatenate([scale1_x, scale2_x, scale3_x, scale4_x])
		elif pooling == 'avg':
			x = layers.Average()([scale1_x, scale2_x, scale3_x, scale4_x])
		elif pooling == 'max':
			x = layers.Maximum()([scale1_x, scale2_x, scale3_x, scale4_x])
		else:
			raise ValueError

		if dropout:
			x = layers.Dropout(0.75)(x)
	elif scales == 'scale4':
		x = scale4_x
	else:
		raise ValueError

	if not return_logits:
		model_output = layers.Dense(num_classes, activation='softmax', name='probs')(x)
	else:
		model_output = layers.Dense(num_classes, activation=None, name='probs')(x)

	if auxiliary:
		#aux output

		if return_logits:
			raise NotImplementedError

		scale1_aux_out = layers.Dense(num_classes, activation='softmax', name='scale1_aux_probs')(scale1_x)
		scale2_aux_out = layers.Dense(num_classes, activation='softmax', name='scale2_aux_probs')(scale2_x)
		scale3_aux_out = layers.Dense(num_classes, activation='softmax', name='scale3_aux_probs')(scale3_x)
		scale4_aux_out = layers.Dense(num_classes, activation='softmax', name='scale4_aux_probs')(scale4_x)

		model = tf.keras.models.Model(inputs=model_input, outputs=[model_output, scale1_aux_out, scale2_aux_out, scale3_aux_out, scale4_aux_out])
	else:
		model = tf.keras.models.Model(inputs=model_input, outputs=model_output)

	return model

def resnet_cifar(input_shape=(32,32,3), base_model_input_shape=(24,24,3), name=None, num_classes=10, augment=False, sampling=False, coarse_fixations=True, coarse_fixations_upsample=False, gaze=None, return_logits=False, approx_ecnn=False):
	#all resnet architectures for cifar10
	
	#check args
	if name is not None:
		raise NotImplementedError

	if coarse_fixations and sampling:
		raise NotImplementedError

	if approx_ecnn and sampling:
		raise ValueError
	if approx_ecnn and coarse_fixations:
		raise ValueError

	if not coarse_fixations and coarse_fixations_upsample:
		raise ValueError

	if input_shape != (32, 32, 3):
		raise ValueError

	if coarse_fixations:
		if not coarse_fixations_upsample:
			if base_model_input_shape != (24, 24, 3):
				raise ValueError
		else:
			if base_model_input_shape != (32, 32, 3):
				raise ValueError
	else:
		if approx_ecnn:
			if base_model_input_shape != (15, 15, 3):
				raise ValueError
		else:
			if base_model_input_shape != (32, 32, 3):
				raise ValueError

	if input_shape != base_model_input_shape and (not coarse_fixations) and (not approx_ecnn):
		raise ValueError

	#base model
	if not approx_ecnn:
		network = ResNet_CIFAR(n=3, version=1, input_shape=base_model_input_shape, num_classes=num_classes, verbose=0, return_logits=return_logits)
	else:
		scale1_network = ResNet_CIFAR(n=3, version=1, input_shape=base_model_input_shape, num_classes=num_classes, verbose=0, return_logits=return_logits, num_filters=22, return_latent=True)
		scale2_network = ResNet_CIFAR(n=3, version=1, input_shape=base_model_input_shape, num_classes=num_classes, verbose=0, return_logits=return_logits, num_filters=22, return_latent=True)

	model_input = layers.Input(shape=input_shape)	

	#data augmentation
	if augment:
		x = layers.Lambda(lambda tensor: glimpse.image_augmentation(tensor, dataset='cifar10'), name='image_augmentation')(model_input)
	else:
		x = model_input 

	#preprocess
	if coarse_fixations:

		if not coarse_fixations_upsample:
			fixation_size = 24
		else:
			fixation_size = 16

		if gaze is not None:
			assert(isinstance(gaze, list))
			coarse_foveation_x = tf.constant(gaze[0], tf.int32)
			coarse_foveation_y = tf.constant(gaze[1], tf.int32)
			coarse_fixation_center = [input_shape[0] // 2 + coarse_foveation_x, input_shape[0] // 2 + coarse_foveation_y]

			x = layers.Lambda(lambda tensor: glimpse.crop_square_patch(tensor, coarse_fixation_center, fixation_size), name='coarse_fixations')(x)
		else:
			x = layers.Lambda(lambda tensor: tf.image.random_crop(tensor, size=[tf.shape(tensor)[0], fixation_size, fixation_size, 3]), name='coarse_fixations')(x)
			
		if coarse_fixations_upsample:
			x = layers.Lambda(lambda tensor: glimpse.uniform_upsample(tensor, factor=2), name='uniform_upsampling')(x)

	if sampling:
		if gaze is not None:
			assert(isinstance(gaze, list))
			gaze_x = tf.constant(gaze[0], tf.int32)
			gaze_y = tf.constant(gaze[1], tf.int32)
			gaze = [gaze_x, gaze_y]
		else:
			#img shape (32, 32, 3)
			gaze = 8

		warp_image_filled = partial(glimpse.warp_image, output_size=base_model_input_shape[0], input_size=base_model_input_shape[0], gaze=gaze)
		x = layers.Lambda(lambda tensor: tf.map_fn(warp_image_filled, tensor, back_prop=True), name='nonuniform_sampling')(x)

	if approx_ecnn:
		if gaze is not None:
			assert(isinstance(gaze, list))
			gaze_x = tf.constant(gaze[0], tf.int32)
			gaze_y = tf.constant(gaze[1], tf.int32)
			gaze = [gaze_x, gaze_y]
		else:
			#larger crop is (30, 30, 3)
			gaze = 1

		scale_sizes = [15, 30]
		scale_radii = [1, 2]

		scale_center = [input_shape[0] // 2, input_shape[0] // 2]

		scales_x = layers.Lambda(lambda tensor: glimpse.image_scales_CIFAR(tensor, scale_center, scale_radii, scale_sizes, gaze), name='scale_sampling')(x)

		scale1_x = scales_x[0]
		scale2_x = scales_x[1]

	if not approx_ecnn:
		model_output = network(x)		
	else:
		scale1_x = scale1_network(scale1_x)
		scale2_x = scale2_network(scale2_x)
		x = layers.concatenate([scale1_x, scale2_x])
		if not return_logits:
			model_output = layers.Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)
		else:
			model_output = layers.Dense(num_classes, kernel_initializer='he_normal')(x)

	model = tf.keras.models.Model(inputs=model_input, outputs=model_output)

	return model
