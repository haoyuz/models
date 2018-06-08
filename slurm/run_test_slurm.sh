#!/bin/bash
#
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#
#SBATCH --job-name=test_clurm
#SBATCH --output=slurm-%x-%j.out
#
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=haoyuz@princeton.edu

SLURM_LOG_DIR=/home/haoyuz
SCRIPT_PATH=/home/haoyuz/jobs/test_slurm.sh

srun --output=$SLURM_LOG_DIR/slurm-%x-%j-%n.out $SCRIPT_PATH

