#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=AAFFT_R2R_conv
#SBATCH --partition=htc
#SBATCH --gres=gpu:1 --constraint='gpu_sku:P100'

module load gpu/cuda/10.0.130

./benchmark_all.sh

