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
import matplotlib.pyplot as plt
import foolbox as fb

#ORIGINAL CLASSES USED IN EXPERIMENTS: 21, 34, 55, 89, 97, 110, 144, 233, 377, 487
ORIG_CLASS = [21]

x_train, y_train, x_test, y_test = datasets.load_imagenet10(only_test=True, only_bbox=False)
start_cursor = 430

def build_the_model(name):
    if name == 'coarsefixations':
        def build_model(gaze=None):
            return model_backbone.resnet(base_model_input_shape=(224, 224, 3), augment=False, sampling=False, coarse_fixations=True, coarse_fixations_upsample=False, coarse_fixations_gaussianblur=False, branched_network=False, gaze=gaze, return_logits=True, num_classes=10)
        model = attack_backbone.build_ensemble(build_model=build_model, save_file='./model_checkpoints/CNNwoSAMPLINGwFIXATIONS.h5', ensemble_size=5, input_size=(320, 320, 3), random_gaze=False, gaze_val=48, load_by_name=True)
    elif name == 'vanilla':
        def build_model(gaze=None):
            return model_backbone.resnet(base_model_input_shape=(320, 320, 3), augment=False, sampling=False, coarse_fixations=False, coarse_fixations_upsample=False, coarse_fixations_gaussianblur=False, branched_network=False, gaze=gaze, return_logits=True, num_classes=10)
        model = build_model()
        model.load_weights('./model_checkpoints/CNNwoSAMPLINGwoFIXATIONS.h5', by_name=True)
    elif name == 'retinalfixations':
        def build_model(gaze=None):
            return model_backbone.resnet(base_model_input_shape=(320, 320, 3), augment=False, sampling=True, coarse_fixations=False, coarse_fixations_upsample=False, coarse_fixations_gaussianblur=False, branched_network=False, gaze=gaze, return_logits=True, num_classes=10)
        model = attack_backbone.build_ensemble(build_model=build_model, save_file='./model_checkpoints/CNNwSAMPLINGwoFIXATIONS.h5', ensemble_size=5, input_size=(320, 320, 3), random_gaze=False, gaze_val=80, load_by_name=True)
    elif name == 'corticalfixations':
        def build_model(gaze=None):
            return model_backbone.ecnn(augment=False, auxiliary=False, sampling=False, scales='all', pooling=None, dropout=False, scale4_freeze=False, gaze=gaze, return_logits=True, num_classes=10)
        model = attack_backbone.build_ensemble(build_model=build_model, save_file='./model_checkpoints/ECNNwoSAMPLINGwAUXILIARY.h5', ensemble_size=5, input_size=(320, 320, 3), random_gaze=False, gaze_val=40, load_by_name=True)
    else:
        raise ValueError
        
    print(np.argmax(model.predict(x_test[ORIG_CLASS])), np.argmax(y_test[ORIG_CLASS]))
        
    logits = model.predict(x_test[ORIG_CLASS])[0]
    print(fb.utils.softmax(logits)[np.argmax(logits)], np.argmax(logits))
        
    return model

def evaluate_model_boundary_attack(model, curse, iterations=1000):
    fb_model = fb.models.TensorFlowEagerModel(model, bounds=(0.0, 1.0))
    attack = fb.attacks.BoundaryAttack
    attack_distance = fb.distances.MSE
    attack_criteria = fb.criteria.TargetClass(8)
    
    fb_attack = attack(model=fb_model, criterion=attack_criteria, distance=attack_distance)
    x_adv = fb_attack(inputs=x_test[curse], labels=np.argmax(y_test[curse], axis=-1), starting_point=x_test[start_cursor], log_every_n_steps=1, loggingLevel=0, iterations=iterations)
    
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 4, 1)
    plt.title('original')
    plt.imshow(x_test[curse][0])
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title('target')
    plt.imshow(x_test[curse][0])
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title('adversary')
    plt.imshow(x_adv[0])
    plt.axis('off')

    plt.subplot(1, 4, 4)
    absdiff = np.abs(x_adv[0] - x_test[curse][0])
    scale_factor = 1./np.max(absdiff)
    MSE_val = np.square(np.subtract(x_adv[0], x_test[curse][0])).mean()
    plt.title('diff\ndist={}'.format(np.sum(MSE_val)))
    plt.imshow(absdiff*scale_factor)
    plt.axis('off')
    plt.show()
    
    logits = model.predict(x_adv)[0]
    
    for index, prob in zip(np.argsort(logits), fb.utils.softmax(logits)[np.argsort(logits)]):
        print('{}: {}'.format(index, prob))

def evaluate_model_PGD(model, curse, iterations=20):
    epsilon = 0.005
    step_size = 0.1
    iterations = iterations
    return_early = False
    random_start = False
    
    fb_model = fb.models.TensorFlowEagerModel(model, bounds=(0.0, 1.0))
    attack = fb.attacks.L2BasicIterativeAttack
    attack_distance = fb.distances.MSE
    attack_criteria = fb.criteria.TargetClass(8)

    fb_attack = attack(model=fb_model, criterion=attack_criteria, distance=attack_distance)
    x_adv = fb_attack(x_test[curse], y_test[curse], binary_search=False, epsilon=epsilon, stepsize=(epsilon/0.3)*step_size, iterations=iterations, return_early=return_early, random_start=random_start)
    
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 3, 1)
    plt.title('original')
    plt.imshow(x_test[curse][0])
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('adversary')
    plt.imshow(x_adv[0])
    plt.axis('off')

    plt.subplot(1, 3, 3)
    absdiff = np.abs(x_adv[0] - x_test[curse][0])
    scale_factor = 1./np.max(absdiff)
    MSE_val = np.square(np.subtract(x_adv[0], x_test[curse][0])).mean()
    plt.title('diff\ndist={}'.format(np.sum(MSE_val)))
    plt.imshow(absdiff*scale_factor)
    plt.axis('off')
    plt.show()
    
    logits = model.predict(x_adv)[0]
    
    for index, prob in zip(np.argsort(logits), fb.utils.softmax(logits)[np.argsort(logits)]):
        print('{}: {}'.format(index, prob))

model = build_the_model('retinalfixations')
print("\n \n RETINAL FIXATIONS BOUNDARY ATTACK")
evaluate_model_boundary_attack(model, ORIG_CLASS, 10000)
print("\n \n RETINAL FIXATIONS PGD")
evaluate_model_PGD(model, ORIG_CLASS, 20)
model = build_the_model('corticalfixations')
print("\n \n CORTICAL FIXATIONS BOUNDARY ATTACK")
evaluate_model_boundary_attack(model, ORIG_CLASS, 10000)
print("\n \n CORTICAL FIXATIONS PGD")
evaluate_model_PGD(model, ORIG_CLASS, 20)
