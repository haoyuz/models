#!/bin/bash

TF_MODELS=/home/haoyuz/tensorflow-models
DATASET_DIR=/tigress/haoyuz/cifar10-dataset
MODEL_DIR=/tigress/haoyuz/resnet_cifar10_model

module load cudnn/cuda-9.1/7.1.2
export PYTHONPATH="$PYTHONPATH:$TF_MODELS"
cd $TF_MODELS/official/resnet
python cifar10_main.py --data_dir=$DATASET_DIR \
                       --model_dir=$MODEL_DIR \
                       --num_gpus=4 \
                       --train_epochs=1000

