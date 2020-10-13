#image tools

import os
import warnings
import pickle

import tensorflow as tf
import numpy as np

from scipy.optimize import curve_fit, brenth
from functools import partial

def image_augmentation(image, dataset):
	#image augmentations

	if dataset == 'imagenet10':
		#random crops and resize params
		crop_min = 300
		crop_max = 320
		crop_resize = 320

	elif dataset == 'cifar10':
		#random crops and resize params
		crop_min = 30
		crop_max = 32
		crop_resize = 32

	else:
		raise ValueError

	#random crops and resize
	crop_size = tf.random.uniform(shape=[], minval=crop_min, maxval=crop_max, dtype=tf.int32)
	image = tf.image.random_crop(image, size=[tf.shape(image)[0], crop_size, crop_size, 3])
	image = tf.image.resize(image, size=[crop_resize,crop_resize])

	#random left/right flips
	image = tf.image.random_flip_left_right(image)

	#color augmentations
		# image = tf.image.adjust_brightness(image, tf.random.uniform(shape=[], minval=0, maxval=(32./255.), dtype=tf.float32)) # 0, 1
		# image = tf.image.adjust_saturation(image, tf.random.uniform(shape=[], minval=0.5, maxval=1.5, dtype=tf.float32)) # Factor to multiply the saturation by.
		# image = tf.image.adjust_hue(image, tf.random.uniform(shape=[], minval=-0.2, maxval=0.2, dtype=tf.float32)) # -1, 1    
		# image = tf.image.adjust_contrast(image, tf.random.uniform(shape=[], minval=0.5, maxval=1.5, dtype=tf.float32)) # Factor multiplier for adjusting contrast.

	return image

def uniform_upsample(image, factor=2):
	#uniformly resamples an image
	#assumes B H W D format for image

	assert(len(image.shape) == 4)
	out_size = image.shape[1] * factor

	return tf.image.resize(image, size=[out_size, out_size], method='nearest')

def warp_image_and_image_scales(images, output_size, input_size, scale_center, scale_radii, scale_sizes, gaze, scale4_freeze=False, debug_gaze=False):
	#nonuniform sampling followed by cortical magnification sampling

	#sanity checks and assignments
	assert(isinstance(gaze, int) or isinstance(gaze, list))
	assert(len(scale_radii) == 4)
	assert(len(scale_sizes) == 4)

	if isinstance(gaze, int):
		gaze_x = tf.random.uniform(shape=[], minval=-gaze, maxval=gaze, dtype=tf.int32)
		gaze_y = tf.random.uniform(shape=[], minval=-gaze, maxval=gaze, dtype=tf.int32)
		gaze = [gaze_x, gaze_y]

	#nonuniform sampling
	warp_image_filled = partial(warp_image, output_size=output_size, input_size=input_size, gaze=gaze)
	images = tf.map_fn(warp_image_filled, images, back_prop=True)

	#cortical sampling (in position and scale)
	images = image_scales(image=images, scale_center=scale_center, scale_radii=scale_radii, scale_sizes=scale_sizes, gaze=gaze, scale4_freeze=scale4_freeze)

	if not debug_gaze:
		return images
	else:
		return images, gaze

def single_image_scale(image, scale_center, scale_radius, scale_size):
	# sample image at a certain scale in the truncated pyramid of position and scale

	image = crop_square_patch(image, center_on=scale_center, patch_size=scale_size)
	image = gaussian_lowpass(image, scale_radius)

	return image

def image_scales_CIFAR(image, scale_center, scale_radii, scale_sizes, gaze):
	# chevron sampling for image (sampling in position and scale)

	assert(isinstance(gaze, int) or isinstance(gaze, list))
	assert(len(scale_radii) == 2)
	assert(len(scale_sizes) == 2)

	if isinstance(gaze, int):
		gaze_x = tf.random.uniform(shape=[], minval=-gaze, maxval=gaze, dtype=tf.int32)
		gaze_y = tf.random.uniform(shape=[], minval=-gaze, maxval=gaze, dtype=tf.int32)
		gaze = [gaze_x, gaze_y]

	gaze_center = [scale_center[0]+gaze[0], scale_center[1]+gaze[1]]

	image_scale1 = single_image_scale(image, scale_center=gaze_center, scale_radius=scale_radii[0], scale_size=scale_sizes[0])
	image_scale2 = single_image_scale(image, scale_center=gaze_center, scale_radius=scale_radii[1], scale_size=scale_sizes[1])
	
	return [image_scale1, image_scale2]

def image_scales(image, scale_center, scale_radii, scale_sizes, gaze, scale4_freeze):
	# chevron sampling for image (sampling in position and scale)

	assert(isinstance(gaze, int) or isinstance(gaze, list))
	assert(len(scale_radii) == 4)
	assert(len(scale_sizes) == 4)

	if isinstance(gaze, int):
		gaze_x = tf.random.uniform(shape=[], minval=-gaze, maxval=gaze, dtype=tf.int32)
		gaze_y = tf.random.uniform(shape=[], minval=-gaze, maxval=gaze, dtype=tf.int32)
		gaze = [gaze_x, gaze_y]

	gaze_center = [scale_center[0]+gaze[0], scale_center[1]+gaze[1]]

	image_scale1 = single_image_scale(image, scale_center=gaze_center, scale_radius=scale_radii[0], scale_size=scale_sizes[0])
	image_scale2 = single_image_scale(image, scale_center=gaze_center, scale_radius=scale_radii[1], scale_size=scale_sizes[1])
	image_scale3 = single_image_scale(image, scale_center=gaze_center, scale_radius=scale_radii[2], scale_size=scale_sizes[2])
	if not scale4_freeze:
		image_scale4 = single_image_scale(image, scale_center=gaze_center, scale_radius=scale_radii[3], scale_size=scale_sizes[3])
	else:
		image_scale4 = single_image_scale(image, scale_center=scale_center, scale_radius=scale_radii[3], scale_size=scale_sizes[3])

	return [image_scale1, image_scale2, image_scale3, image_scale4]

def make_gaussian_2d_kernel(sigma, truncate=4.0, dtype=tf.float32):
	# https://stackoverflow.com/questions/56258751/how-to-realise-the-2-d-gaussian-filter-like-the-scipy-ndimage-gaussian-filter
	# Make Gaussian kernel following SciPy logic

		radius = sigma * truncate
		x = tf.cast(tf.range(-radius, radius + 1), dtype=dtype)
		k = tf.exp(-0.5 * tf.square(x / sigma))
		k = k / tf.reduce_sum(k)

		return tf.expand_dims(k, 1) * k

def subsample(image, stride):
	# subsamples an image 4D: (batch, h,w,c)
	return image[::, stride//2::stride, stride//2::stride, ::]

def gaussian_blur(image, radius):
	# gaussian blurs the image

	gaussian_sigma = radius/2.

	# gaussian convolution kernel
	gaussian_kernel = make_gaussian_2d_kernel(gaussian_sigma)

	gaussian_kernel = tf.tile(gaussian_kernel[:, :, tf.newaxis, tf.newaxis], [1, 1, 3, 1])
	image = tf.nn.separable_conv2d(image, gaussian_kernel, tf.eye(3, batch_shape=[1, 1]), strides=[1, 1, 1, 1], padding='SAME') 

	return image

def gaussian_lowpass(image, radius, compat_mode=False):
	# gaussian subsamples the image

	gaussian_sigma = radius/2.
	subsample_stride = radius

	# gaussian convolution kernel
	gaussian_kernel = make_gaussian_2d_kernel(gaussian_sigma)


	# conv2d approach is significant slower than seperable_conv2d on tf20
	# # build filters compatible with conv2d
	#	kernel_shape = tf.shape(gaussian_kernel)
	# filter_channel0 = tf.stack([gaussian_kernel, tf.zeros(shape=kernel_shape), tf.zeros(shape=kernel_shape)], axis=-1)
	# filter_channel1 = tf.stack([tf.zeros(shape=kernel_shape), gaussian_kernel, tf.zeros(shape=kernel_shape)], axis=-1)
	# filter_channel2 = tf.stack([tf.zeros(shape=kernel_shape), tf.zeros(shape=kernel_shape), gaussian_kernel], axis=-1)
	# filters = tf.stack([filter_channel0, filter_channel1, filter_channel2], axis=-1)

	# # convolve image with filters
	# image = tf.nn.conv2d(image, filters, strides=1, padding='SAME', name='gaussian_lowpass')

	gaussian_kernel = tf.tile(gaussian_kernel[:, :, tf.newaxis, tf.newaxis], [1, 1, 3, 1])
	image = tf.nn.separable_conv2d(image, gaussian_kernel, tf.eye(3, batch_shape=[1, 1]), strides=[1, 1, 1, 1], padding='SAME') 

	# subsample
	if not compat_mode:
		image = subsample(image, subsample_stride)
	else:
		warnings.warn('subsampling in compatibility mode.')
		image = _compat_subsample(image, subsample_stride)

	return image

def crop_square_patch(image, center_on, patch_size):
	#crops out square patches centered on a point

	image = tf.image.crop_to_bounding_box(image, offset_height=center_on[0] - patch_size//2, offset_width=center_on[1] - patch_size//2, target_height=patch_size, target_width=patch_size)

	return image

######## IMAGE SAMPLING BASED ON https://github.com/dicarlolab/retinawarp and https://github.com/npant20/fish-eye-foveation-resnet #########
############################################################################################################################################
############################################################################################################################################

def sampling_mismatch(rf, in_size=None, out_size=None, max_ratio=10.):
	"""
	This function returns the mismatch between the radius of last sampled point and the image size.
	"""
	if out_size is None:
		out_size = in_size
	r_max = in_size // 2

	# Exponential relationship
	a = np.log(max_ratio) / r_max
	r, d = [0.], []
	for i in range(1, out_size // 2):
		d.append(1. / np.sqrt(np.pi * rf) * np.exp(a * r[-1] / 2.))
		r.append(r[-1] + d[-1])
	r = np.array(r)

	return in_size / 2 - r[-1]

def get_rf_value(input_size, output_size, rf_range=(0.01, 5.)):
	"""
	The RF parameter should be tuned in a way that the last sample would be taken from the outmost pixel of the image.
	This function returns the mismatch between the radius of last sampled point and the image size. We use this function
	together with classic root finding methods to find the optimal RF value given the input and output sizes.
	"""
	func = partial(sampling_mismatch, in_size=input_size, out_size=output_size)
	return brenth(func, rf_range[0], rf_range[1])


def get_foveal_density(output_image_size, input_image_size):
		return get_rf_value(input_image_size, output_image_size)

def delta_lookup(in_size, out_size=None, max_ratio=10.):
	"""
	Divides the range of radius values based on the image size and finds the distances between samples
	with respect to each radius value. Different function types can be used to form the mapping. All function
	map to delta values of min_delta in the center and max_delta at the outmost periphery.
	:param in_size: Size of the input image
	:param out_size: Size of the output (retina) image
	:param max_ratio: ratio between density at the fovea and periphery
	:return: Grid of points on the retinal image (r_prime) and original image (r)
	"""
	rf = get_foveal_density(out_size, in_size)
	if out_size is None:
		out_size = in_size
	r_max = in_size // 2

	# Exponential relationship
	a = np.log(max_ratio) / r_max
	r, d = [0.], []
	for i in range(out_size // 2):
		d.append(1. / np.sqrt(np.pi * rf) * np.exp(a * r[-1] / 2.))
		r.append(r[-1] + d[-1])
	r = np.array(r)
	r_prime = np.arange(out_size // 2)

	return r_prime, r[:-1]

def fit_func(func, r, r_raw):
	"""
	Fits a function to map the radius values in the
	:param func: function template
	:param r: Inputs to the function (grid points on the retinal image)
	:param r_raw: Outputs for the function (grid points on the original image)
	:return: Estimated parameters, estimaged covariance of parameters
	"""
	popt, pcov = curve_fit(func, r, r_raw, p0=[0, 0.4], bounds=(0, np.inf))
	return popt, pcov

def tf_exp_func(x, func_pars):
	return tf.exp(func_pars[0] * x) + func_pars[1]

def tf_quad_func(x, func_pars):
	return func_pars[0] * x ** 2 + func_pars[1] * x

def cached_find_retina_mapping(input_size, output_size, fit_mode='quad'):
	popt_cache_file = './cache_store/{}-{}-{}_retina_mapping_popt.pickle'.format(input_size, output_size, fit_mode)
	tf_func_cache_file = './cache_store/{}-{}-{}_retina_mapping_tf_func.pickle'.format(input_size, output_size, fit_mode)
	popt = None
	tf_func = None

	#if cache exists, load from cache
	if os.path.exists(popt_cache_file) and os.path.exists(tf_func_cache_file):
		popt = pickle.load(open(popt_cache_file, 'rb'))
		tf_func = pickle.load(open(tf_func_cache_file, 'rb'))
	#else resolve and save to cache
	else:
		popt, tf_func = find_retina_mapping(input_size, output_size, fit_mode)

		pickle.dump(popt, open(popt_cache_file, 'wb'))
		pickle.dump(tf_func, open(tf_func_cache_file, 'wb'))

	return popt, tf_func


def find_retina_mapping(input_size, output_size, fit_mode='quad'):
	"""
	Fits a function to the distance data so it will map the outmost pixel to the border of the image
	:param fit_mode:
	:return:
	"""
	warnings.warn('refitting retina mapping.')
	r, r_raw = delta_lookup(in_size=input_size, out_size=output_size)
	if fit_mode == 'quad':
		func = lambda x, a, b: a * x ** 2 + b * x
		tf_func = tf_quad_func
	elif fit_mode == 'exp':
		func = lambda x, a, b: np.exp(a * x) + b
		tf_func = tf_exp_func
	else:
		raise ValueError('Fit mode not defined. Choices are ''linear'', ''exp''.')
	popt, pcov = fit_func(func, r, r_raw)

	return popt, tf_func

def warp_func(xy, orig_img_size, func, func_pars, shift, gaze):
	# Centeralize the indices [-n, n]
	xy = tf.cast(xy, tf.float32)
	center = tf.reduce_mean(xy, axis=0)
	xy_cent = xy - center - gaze

	# Polar coordinates
	r = tf.sqrt(xy_cent[:, 0] ** 2 + xy_cent[:, 1] ** 2)
	theta = tf.atan2(xy_cent[:, 1], xy_cent[:, 0])
	r = func(r, func_pars)

	xs = r * tf.cos(theta)
	xs += gaze[0][0]
	xs += orig_img_size[0] / 2. - shift[0]

	# Added + 2.0 is for the additional zero padding
	xs = tf.minimum(orig_img_size[0] + 2.0, xs)
	xs = tf.maximum(0., xs)
	xs = tf.round(xs)

	ys = r * tf.sin(theta)
	ys += gaze[0][1]
	ys += orig_img_size[1] / 2 - shift[1]
	ys = tf.minimum(orig_img_size[1] + 2.0, ys)
	ys = tf.maximum(0., ys)
	ys = tf.round(ys)

	xy_out = tf.stack([xs, ys], 1)

	xy_out = tf.cast(xy_out, tf.int32)
	return xy_out


def warp_image(img, output_size, input_size, gaze, shift=None):
	"""
	:param img: (tensor) input image
	:param retina_func:
	:param retina_pars:
	:param shift:
	:param gaze:
	:return:
	"""
	original_shape = img.shape

	# if input_size is None:
	# 	input_size = np.min([original_shape[0], original_shape[1]])

	retina_pars, retina_func = cached_find_retina_mapping(input_size, output_size)

	assert(isinstance(gaze, int) or isinstance(gaze, list))
	if isinstance(gaze, int):
		gaze_x = tf.random.uniform(shape=[], minval=-gaze, maxval=gaze, dtype=tf.int32)
		gaze_y = tf.random.uniform(shape=[], minval=-gaze, maxval=gaze, dtype=tf.int32)
		gaze = tf.cast([[gaze_x, gaze_y]], tf.float32)
	elif isinstance(gaze, list):
		assert(len(gaze) == 2)
		gaze = tf.cast([gaze], tf.float32)
	else:
		raise ValueError

	if shift is None:
		shift = [tf.constant([0], tf.float32), tf.constant([0], tf.float32)]
	else:
		assert len(shift) == 2
		shift = [tf.constant([shift[0]], tf.float32), tf.constant([shift[1]], tf.float32)]

	paddings = tf.constant([[2, 2], [2, 2], [0, 0]])
	img = tf.pad(img, paddings, "CONSTANT")
	row_ind = tf.tile(tf.expand_dims(tf.range(output_size), axis=-1), [1, output_size])
	row_ind = tf.reshape(row_ind, [-1, 1])
	col_ind = tf.tile(tf.expand_dims(tf.range(output_size), axis=0), [1, output_size])
	col_ind = tf.reshape(col_ind, [-1, 1])
	indices = tf.concat([row_ind, col_ind], 1)
	xy_out = warp_func(indices, tf.cast(original_shape, tf.float32), retina_func, retina_pars, shift, gaze)

	out = tf.reshape(tf.gather_nd(img, xy_out), [output_size, output_size, 3]) 
	return out


########################################################### DEPRECATED FUNCTIONS ###########################################################
############################################################################################################################################
############################################################################################################################################

def _compat_gaussian_lowpass(image, radius):
	# deprecated implementation of gaussian subsample of the image with seperable convolutions 

		blur_radius = radius/2.
		subsample_stride = radius

		# https://stackoverflow.com/questions/56258751/how-to-realise-the-2-d-gaussian-filter-like-the-scipy-ndimage-gaussian-filter
		# Make Gaussian kernel following SciPy logic
		def make_gaussian_2d_kernel(sigma, truncate=4.0, dtype=tf.float32):

				#radius = tf.to_int32(sigma * truncate)
				radius = sigma * truncate
				x = tf.cast(tf.range(-radius, radius + 1), dtype=dtype)
				k = tf.exp(-0.5 * tf.square(x / sigma))
				k = k / tf.reduce_sum(k)

				return tf.expand_dims(k, 1) * k

		# Convolution kernel
		kernel = make_gaussian_2d_kernel(blur_radius)
		# Apply kernel to each channel (see https://stackoverflow.com/q/55687616/1782792)
		kernel = tf.tile(kernel[:, :, tf.newaxis, tf.newaxis], [1, 1, 3, 1])

		image_filtered = tf.nn.separable_conv2d(image, kernel, tf.eye(3, batch_shape=[1, 1]), strides=[1, 1, 1, 1], padding='SAME') 

		# Subsample
		image_filtered = image_filtered[::, ::subsample_stride, ::subsample_stride, ::]

		return image_filtered

def _compat_subsample(image, subsample_stride):
	return image[::, ::subsample_stride, ::subsample_stride, ::]

def _compat_warp_func(xy, orig_img_size, func, func_pars, shift, dxc = 0, dyc = 0):
	# Centeralize the indices [-n, n]
	xy = tf.cast(xy, tf.float32)
	center = tf.reduce_mean(xy, axis=0)
	center_shift = tf.cast(tf.constant([[dxc, dyc]]), tf.float32)
	xy_cent = xy - center - center_shift

	# Polar coordinates
	r = tf.sqrt(xy_cent[:, 0] ** 2 + xy_cent[:, 1] ** 2)
	theta = tf.atan2(xy_cent[:, 1], xy_cent[:, 0])
	r_old = r
	r = func(r, func_pars)
	ratio = r/(r_old+1e-10)

	xs = r * tf.cos(theta)
	xs = xs + tf.math.multiply(ratio, dxc)
	xs += orig_img_size[0] / 2. - shift[0]

	# Added + 2.0 is for the additional zero padding
	xs = tf.minimum(orig_img_size[0] + 2.0, xs)
	xs = tf.maximum(0., xs)
	xs = tf.round(xs)

	ys = r * tf.sin(theta)
	ys = ys + tf.math.multiply(ratio, dyc)
	ys += orig_img_size[1] / 2 - shift[1]
	ys = tf.minimum(orig_img_size[1] + 2.0, ys)
	ys = tf.maximum(0., ys)
	ys = tf.round(ys)

	xy_out = tf.stack([xs, ys], 1)

	xy_out = tf.cast(xy_out, tf.int32)
	return xy_out


def _compat_warp_image(img, output_size, input_size=None, shift=None, dxc = 0, dyc = 0):
	"""
	:param img: (tensor) input image
	:param retina_func:
	:param retina_pars:
	:param shift:
	:return:
	"""
	original_shape = img.shape

	if input_size is None:
		input_size = np.min([original_shape[0], original_shape[1]])

	retina_pars, retina_func = cached_find_retina_mapping(input_size, output_size)

	if shift is None:
		shift = [tf.constant([0], tf.float32), tf.constant([0], tf.float32)]
	else:
		assert len(shift) == 2
		shift = [tf.constant([shift[0]], tf.float32), tf.constant([shift[1]], tf.float32)]
	paddings = tf.constant([[2, 2], [2, 2], [0, 0]])
	img = tf.pad(img, paddings, "CONSTANT")
	row_ind = tf.tile(tf.expand_dims(tf.range(output_size), axis=-1), [1, output_size])
	row_ind = tf.reshape(row_ind, [-1, 1])
	col_ind = tf.tile(tf.expand_dims(tf.range(output_size), axis=0), [1, output_size])
	col_ind = tf.reshape(col_ind, [-1, 1])
	indices = tf.concat([row_ind, col_ind], 1)
	xy_out = warp_func(indices, tf.cast(original_shape, tf.float32), retina_func, retina_pars, shift, dxc, dyc)

	out = tf.reshape(tf.gather_nd(img, xy_out), [output_size, output_size, 3]) 
	return out