#!/bin/bash

#SBATCH -J craftax-ppo                 # job name
#SBATCH -p mit_normal_gpu              # partition
#SBATCH --gres=gpu:l40:1               # 1x L40 GPU
#SBATCH -N 1                           # 1 node
#SBATCH --cpus-per-task=16             # CPUs for parallel envs
#SBATCH --mem=32G                      # memory
#SBATCH -t 06:00:00                    # walltime limit
#SBATCH -o /home/melwani/fun/craftax/logs/%x-%j.out  # log file

cd /home/melwani/fun/craftax
source .venv/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python train.py \
    --require-num-devices 0 \
    --num-envs-per-device 1024 \
    --num-steps 32 \
    --num-minibatches 16 \
    --update-epochs 4 \
    --use-wandb \
    --wandb-project craftax-ppo
