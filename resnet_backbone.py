#######################################################################################################################################
#######################################################################################################################################


# adapted from https://github.com/keras-team/keras-applications/edit/master/keras_applications/resnet_common.py


#######################################################################################################################################
#######################################################################################################################################


"""ResNet, ResNetV2, and ResNeXt models for Keras.

# Reference papers

- [Deep Residual Learning for Image Recognition]
  (https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
- [Identity Mappings in Deep Residual Networks]
  (https://arxiv.org/abs/1603.05027) (ECCV 2016)
- [Aggregated Residual Transformations for Deep Neural Networks]
  (https://arxiv.org/abs/1611.05431) (CVPR 2017)

# Reference implementations

- [TensorNets]
  (https://github.com/taehoonlee/tensornets/blob/master/tensornets/resnets.py)
- [Caffe ResNet]
  (https://github.com/KaimingHe/deep-residual-networks/tree/master/prototxt)
- [Torch ResNetV2]
  (https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua)
- [Torch ResNeXt]
  (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)

"""
#18: [2, 2, 2, 2] + no 1x1 kernels

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

bn_axis = 3 #3 if channels_last, else 1
name_tag = None

def block1_nobottleneck(x, filters, kernel_size=3, stride=1,
           conv_shortcut=True, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """

    if conv_shortcut is True:
        shortcut = layers.Conv2D(filters, 1, strides=stride,
                                 name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                             name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='SAME', name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='SAME',
                      name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x

def stack1_nobottleneck(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = block1_nobottleneck(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1_nobottleneck(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x

def ResNet(stack_fn, preact, use_bias, model_name='resnet', include_top=True, input_shape=None, pooling=None, classes=1000, input_tensor=None, conv1_stride=2, max_pool_stride=2, **kwargs):    

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
    	img_input = layers.Input(tensor=input_tensor)

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name=name_tag+'conv1_pad')(img_input)
    x = layers.Conv2D(64, 7, strides=conv1_stride, use_bias=use_bias, name=name_tag+'conv1_conv')(x)

    if preact is False:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name=name_tag+'conv1_bn')(x)
        x = layers.Activation('relu', name=name_tag+'conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name_tag+'pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=max_pool_stride, name=name_tag+'pool1_pool')(x)

    x = stack_fn(x)

    if preact is True:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name=name_tag+'post_bn')(x)
        x = layers.Activation('relu', name=name_tag+'post_relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name=name_tag+'avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name=name_tag+'probs')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name=name_tag+'avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name=name_tag+'max_pool')(x)

    # Create model.
    model = models.Model(img_input, x, name=model_name)
    return model

def ResNet18(include_top=True, input_shape=None, input_tensor=None, conv1_stride=2, max_pool_stride=2, filters=64, pooling=None, classes=1000, subnetwork_name='', **kwargs):
    global name_tag
    name_tag = subnetwork_name
    def stack_fn(x):
        x = stack1_nobottleneck(x, filters, 2, stride1=1, name=name_tag+'conv2')
        x = stack1_nobottleneck(x, filters*2, 2, name=name_tag+'conv3')
        x = stack1_nobottleneck(x, filters*4, 2, name=name_tag+'conv4')
        x = stack1_nobottleneck(x, filters*8, 2, name=name_tag+'conv5')
        return x
    return ResNet(stack_fn, False, True, name_tag+'resnet18', include_top, input_shape, pooling, classes, input_tensor=input_tensor, conv1_stride=conv1_stride, max_pool_stride=max_pool_stride)


#######################################################################################################################################
#######################################################################################################################################
# resnet cifar10

# adapted from https://keras.io/examples/cifar10_resnet/
#######################################################################################################################################
#######################################################################################################################################

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = layers.Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10, return_logits=False, input_tensor=None, num_filters=16, return_latent=False):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    #num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    if input_tensor is None:
        inputs = layers.Input(shape=input_shape)
    else:
        inputs = layers.Input(tensor=input_tensor)

    x = resnet_layer(inputs=inputs, num_filters=num_filters)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)

            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tf.keras.layers.add([x, y])
            x = layers.Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    #x = layers.AveragePooling2D(pool_size=8)(x)
    #y = layers.Flatten()(x)
    y = layers.GlobalAveragePooling2D()(x)

    if return_latent:

        model = models.Model(inputs=inputs, outputs=y)
        return model
    else:

        if not return_logits:
            outputs = layers.Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)
        else:
            outputs = layers.Dense(num_classes,
                        kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = models.Model(inputs=inputs, outputs=outputs)
        return model


# def resnet_v2(input_shape, depth, num_classes=10, return_logits=False, input_tensor=None):
#     """ResNet Version 2 Model builder [b]

#     Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
#     bottleneck layer
#     First shortcut connection per layer is 1 x 1 Conv2D.
#     Second and onwards shortcut connection is identity.
#     At the beginning of each stage, the feature map size is halved (downsampled)
#     by a convolutional layer with strides=2, while the number of filter maps is
#     doubled. Within each stage, the layers have the same number filters and the
#     same filter map sizes.
#     Features maps sizes:
#     conv1  : 32x32,  16
#     stage 0: 32x32,  64
#     stage 1: 16x16, 128
#     stage 2:  8x8,  256

#     # Arguments
#         input_shape (tensor): shape of input image tensor
#         depth (int): number of core convolutional layers
#         num_classes (int): number of classes (CIFAR10 has 10)

#     # Returns
#         model (Model): Keras model instance
#     """
#     if (depth - 2) % 9 != 0:
#         raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
#     # Start model definition.
#     num_filters_in = 16
#     num_res_blocks = int((depth - 2) / 9)

#     if input_tensor is None:
#         inputs = layers.Input(shape=input_shape)
#     else:
#         inputs = layers.Input(tensor=input_tensor)
#     # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
#     x = resnet_layer(inputs=inputs,
#                      num_filters=num_filters_in,
#                      conv_first=True)

#     # Instantiate the stack of residual units
#     for stage in range(3):
#         for res_block in range(num_res_blocks):
#             activation = 'relu'
#             batch_normalization = True
#             strides = 1
#             if stage == 0:
#                 num_filters_out = num_filters_in * 4
#                 if res_block == 0:  # first layer and first stage
#                     activation = None
#                     batch_normalization = False
#             else:
#                 num_filters_out = num_filters_in * 2
#                 if res_block == 0:  # first layer but not first stage
#                     strides = 2    # downsample

#             # bottleneck residual unit
#             y = resnet_layer(inputs=x,
#                              num_filters=num_filters_in,
#                              kernel_size=1,
#                              strides=strides,
#                              activation=activation,
#                              batch_normalization=batch_normalization,
#                              conv_first=False)
#             y = resnet_layer(inputs=y,
#                              num_filters=num_filters_in,
#                              conv_first=False)
#             y = resnet_layer(inputs=y,
#                              num_filters=num_filters_out,
#                              kernel_size=1,
#                              conv_first=False)
#             if res_block == 0:
#                 # linear projection residual shortcut connection to match
#                 # changed dims
#                 x = resnet_layer(inputs=x,
#                                  num_filters=num_filters_out,
#                                  kernel_size=1,
#                                  strides=strides,
#                                  activation=None,
#                                  batch_normalization=False)
#             x = tf.keras.layers.add([x, y])

#         num_filters_in = num_filters_out

#     # Add classifier on top.
#     # v2 has BN-ReLU before Pooling
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     #x = layers.AveragePooling2D(pool_size=8)(x)
#     #y = layers.Flatten()(x)
#     y = layers.GlobalAveragePooling2D()(x)
    
#     if not return_logits:
#         outputs = layers.Dense(num_classes,
#                     activation='softmax',
#                     kernel_initializer='he_normal')(y)
#     else:
#         outputs = layers.Dense(num_classes,
#                     kernel_initializer='he_normal')(y)

#     # Instantiate model.
#     model = models.Model(inputs=inputs, outputs=outputs)
#     return model

def ResNet_CIFAR(n=3, version=1, input_shape=(32,32,3), num_classes=10, verbose=True, return_logits=False, num_filters=16, return_latent=False):
    # taken from https://keras.io/examples/cifar10_resnet/

    # Model parameter
    # ----------------------------------------------------------------------------
    #           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
    # Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
    #           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
    # ----------------------------------------------------------------------------
    # ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
    # ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
    # ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
    # ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
    # ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
    # ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
    # ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
    # ---------------------------------------------------------------------------
    ## n = 3

    # Model version
    # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    ## version = 1

    # Computed depth from supplied model parameter n
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2
    
    if verbose:
        # Model name, depth and version
        model_type = 'ResNet%dv%d' % (depth, version)
        print('building model {}'.format(model_type))


    if version == 2:
        raise NotImplementedError
        model = resnet_v2(input_shape=input_shape, depth=depth, num_classes=num_classes, return_logits=return_logits)
    else:
        model = resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes, return_logits=return_logits, num_filters=num_filters, return_latent=return_latent)

    if verbose:
        model.summary()

    return model