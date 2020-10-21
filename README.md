# BIOLOGICALLY INSPIRED MECHANISMS FOR ADVERSARIAL ROBUSTNESS

### Required Libraries

`Python 3.6.8`

`tensorflow-gpu             2.0.0`, `tensorflow-determinism     0.3.0`, `numpy                      1.17.2`, `scipy                      1.3.3`, `jupyter                    1.0.0`, `notebook                   6.0.2`, `ipython                    7.11.1`, `tqdm                       4.40.2`, `Pillow                     6.2.1`, `matplotlib                 3.1.2`, `seaborn       0.9.0`, `randomgen                  1.14.4`, `foolbox                    2.3.0 (see note below)`

notes:

do not install the foolbox package, instead move the foolbox source code to the parent directory containing this project folder. such that:
- `parent folder`
  - `foolbox`
    -  `...`
  - `adversary`
    - `...`



### Preparing Datasets

- CIFAR10: Nothing needs to be done.
- IMAGENET10: Move 10 classes of the ImageNet training split to folder `adversary/imagenet10/train`, validation split to folder `adversary/imagenet10/val`, bounding boxes to folder `adversary/imagenet10/bbox`
- IMAGENET100: Move shortlist.pickle (list of images to include in the dataset), move all classes of ImageNet training split to `../data/ImageNet/raw-data/train` and validation split to `../data/ImageNet/raw-data/validation`.

### Training a Model

- training a standard resnet model on cifar10:

`python trainer.py --name=CIFAR_RESNET --model=resnet_cifar --dataset=cifar10 --augment=1 --sampling=0 --coarse_fixations=0 --auxiliary=0 --only_evaluate=0`

- training a coarse fixations model on cifar10:

`python trainer.py --name=CIFAR_FIXATIONS --model=resnet_cifar --dataset=cifar10 --augment=1 --sampling
=0 --coarse_fixations=1 --auxiliary=0 --only_evaluate=0`

- training a retinal fixations model on cifar10:

`python trainer.py --name=CIFAR_SAMPLING --model=resnet_cifar --dataset=cifar10 --augment=1 --sampling=
1 --coarse_fixations=0 --auxiliary=0 --only_evaluate=0`

- training a cortical fixations model on cifar10:

`python trainer.py --name=CIFAR_ECNN --model=resnet_cifar --dataset=cifar10 --augment=1 --sampling=0 --coarse_fixations=0 --auxiliary=1 --cifar_ecnn=1 --only_evaluate=0`

- training a standard resnet model on imagenet10:

`python trainer.py --name=CNNwoSAMPLINGwoFIXATIONS --model=resnet --dataset=imagenet10 --augment=1 --sampling=0 --coarse_fixations=0 --auxiliary=0 --only_evaluate=0`

- training a coarse fixations model on imagenet10:

`python trainer.py --name=CNNwoSAMPLINGwFIXATIONS --model=resnet --dataset=imagenet10 --augment=1 --sampling=0 --coarse_fixations=1 --auxiliary=0 --only_evaluate=0`

- training a retinal fixations model on imagenet10:

`python trainer.py --name=CNNwSAMPLINGwoFIXATIONS --model=resnet --dataset=imagenet10 --augment=1 --sampling=1 --coarse_fixations=0 --auxiliary=0 --only_evaluate=0`

- training a cortical fixations model on imagenet10:

`python trainer.py --name=ECNNwoSAMPLINGwAUXILIARY --model=ecnn --dataset=imagenet10 --augment=1 --sampling=0 --coarse_fixations=0 --auxiliary=1 --only_evaluate=0`


For training models on ImageNet100, use the commands for ImageNet10 but replace the argument for dataset with `imagenet100` instead of `imagenet10`.
### Evaluating a Model (Standard Performance)

- evaluating the standard performance of a standard resnet model on cifar10:

`python trainer.py --name=CIFAR_RESNET --model=resnet_cifar --dataset=cifar10 --augment=0 --sampling=0 --coarse_fixations=0 --auxiliary=0 --only_evaluate=1`

- evaluating the standard performance of a coarse fixations model on cifar10:

`python trainer.py --name=CIFAR_FIXATIONS --model=resnet_cifar --dataset=cifar10 --augment=0 --sampling
=0 --coarse_fixations=1 --auxiliary=0 --only_evaluate=1`

- evaluating the standard performance of a retinal fixations model on cifar10:

`python trainer.py --name=CIFAR_SAMPLING --model=resnet_cifar --dataset=cifar10 --augment=0 --sampling=1 --coarse_fixations=0 --auxiliary=0 --only_evaluate=1`

- evaluating the standard performance of a cortical fixations model on cifar10:

`python trainer.py --name=CIFAR_ECNN --model=resnet_cifar --dataset=cifar10 --augment=0 --sampling=0 --coarse_fixations=0 --auxiliary=0 --cifar_ecnn=1 --only_evaluate=1`

- evaluating the standard performance of a standard resnet model on imagenet10:

`python trainer.py --name=CNNwoSAMPLINGwoFIXATIONS --model=resnet --dataset=imagenet10 --augment=0 --sampling=0 --coarse_fixations=0 --auxiliary=0 --only_evaluate=1`

- evaluating the standard performance of a coarse fixations model on imagenet10:

`python trainer.py --name=CNNwoSAMPLINGwFIXATIONS --model=resnet --dataset=imagenet10 --augment=0 --sampling=0 --coarse_fixations=1 --auxiliary=0 --only_evaluate=1`

- evaluating the standard performance of a retinal fixations model on imagenet10:

`python trainer.py --name=CNNwSAMPLINGwoFIXATIONS --model=resnet --dataset=imagenet10 --augment=0 --sampling=1 --coarse_fixations=0 --auxiliary=0 --only_evaluate=1`

- evaluating the standard performance of a cortical fixations model on imagenet10:

`python trainer.py --name=ECNNwoSAMPLINGwAUXILIARY --model=ecnn --dataset=imagenet10 --augment=0 --sampling=0 --coarse_fixations=0 --auxiliary=1 --only_evaluate=1`

For evaluating the standard performance of models on ImageNet100, use the commands for ImageNet10 but replace the argument for dataset with `imagenet100` instead of `imagenet10`.
### Evaluating a Model (Adversarial Robustness)

see below on notes for customizing adversarial robustness evaluation

- evaluating the adversarial robustness of standard resnet model with default parameters on cifar10

`python adversary.py --name=CIFAR_RESNET --model=resnet_cifar --dataset=cifar10 --sampling=0 --coarse_fixations=0 --evaluate_mode=robustness`

- evaluating the adversarial robustness of coarse fixations model on cifar10

`python adversary.py --name=CIFAR_FIXATIONS --model=resnet_cifar --dataset=cifar10 --sampling=0 --coarse_fixations=1 --evaluate_mode=robustness`

- evaluating the adversarial robustness of retinal fixations model on cifar10

`python adversary.py --name=CIFAR_SAMPLING --model=resnet_cifar --dataset=cifar10 --sampling=1 --coarse_fixations=0 --evaluate_mode=robustness`

- evaluating the adversarial robustness of cortical fixations model on cifar10

`python adversary.py --name=CIFAR_ECNN --model=resnet_cifar --dataset=cifar10 --sampling=0 --coarse_fixations=0 --cifar_ecnn=1 --evaluate_mode=robustness`

- evaluating the adversarial robustness of standard resnet model on imagenet10

`python adversary.py --name=CNNwoSAMPLINGwoFIXATIONS --model=resnet --dataset=imagenet10 --sampling=0 --coarse_fixations=0 --evaluate_mode=robustness`

- evaluating the adversarial robustness of coarse fixations model on imagenet10

`python adversary.py --name=CNNwoSAMPLINGwFIXATIONS --model=resnet --dataset=imagenet10 --sampling=0 --coarse_fixations=1 --evaluate_mode=robustness`

- evaluating the adversarial robustness of retinal fixations model on imagenet10

`python adversary.py --name=CNNwSAMPLINGwoFIXATIONS --model=resnet --dataset=imagenet10 --sampling=1 --coarse_fixations=0 --evaluate_mode=robustness`

- evaluating the adversarial robustness of cortical fixations model on imagenet10

`python adversary.py --name=ECNNwoSAMPLINGwAUXILIARY --model=ecnn --dataset=imagenet10 --sampling=0 --coarse_fixations=0 --evaluate_mode=robustness`

Optional Adversarial Robustness Arguments:
- attack_algo: attack algorithm to use ('PGD' or 'FGSM' or 'PGD_ADAM')
- attack_iterations: number of iterations for iterative algorithm (set to 1 for FGSM)
- attack_step_size: step size for algorithm 
- attack_distance_metric: Lp norm for distance metrics and algorithm variants ('LINF' or 'L2' or 'L1')
- attack_criteria_targeted: use a targeted or an untargeted loss for the algorithm
- attack_criteria_det: determine adversarial class for a targeted loss in a deterministic fashion or randomly (randomly is not implemented yet)
- attack_random_init: add a random noise to the image before running algorithm

For evaluating the adversarial robustness of models on ImageNet100, use the commands for ImageNet10 but replace the argument for dataset with `imagenet100` instead of `imagenet10`.

### Explore Pre-Calculated Adversarial Robustness 

We also provide directly the results of training and evaluating the adversarial robustness of the models. See all files with the extension `.packet` in the folders at `adversary/cluster_runs/adversary/final_store/`.

The packets are dictionaries stores as pickles. See `adversary.py` for how these dictionaries were generated and `playground/PAPER_PLOTS_V2.ipynb` notebook for how these dictionaries were read and parsed to generate the plots and tables in the paper.

### Evaluating a Model (Feature Robustness, Etc.)

work in progress. :)

### Models Checkpoints
https://www.dropbox.com/s/fph11yxcgntq1rv/model_checkpoints.7z?dl=0

### Paper
https://arxiv.org/abs/2006.16427
