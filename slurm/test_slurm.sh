#!/bin/bash

module load cudnn/cuda-9.1/7.1.2
export PYTHONPATH="$PYTHONPATH:/home/haoyuz/tensorflow-models"
echo $SLURM_JOB_NODELIST
echo $SLURMD_NODENAME
date
sleep 60
date
echo "Hello World!"

