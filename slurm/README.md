# Slurm job scripts

This directory contains both a Python module to deploy TensorFlow on Slurm (.py
file) and some scripts to launch jobs on a Slurm cluster (.sh files).

**Note**: must correctly configure paths on machines (`job.config` file).

The status of Slurm batch files:

* Scripts for preprocessing
  - tf_process_imagenet.sh

* Scripts that will only run on single Slurm node (for now...)
  - cnn_mnist.sh
  - inception.sh
  - inception_eval.sh
  - resnet_cifar10.sh

* Scripts that run on multiple nodes on Slurm cluster
  - run_test_slurm.sh, test_slurm.sh
  - run_dist_resnet_cifar10.sh, dist_resnet_cifar10.sh

