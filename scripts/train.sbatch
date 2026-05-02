#!/bin/bash

#SBATCH -J craftax-ppo                 # job name (override with `sbatch -J ...`)
#SBATCH -p mit_normal_gpu              # partition
#SBATCH --gres=gpu:l40:1               # 1x L40 GPU
#SBATCH -N 1                           # 1 node
#SBATCH --cpus-per-task=16             # CPUs for parallel envs
#SBATCH --mem=32G                      # memory
#SBATCH -t 06:00:00                    # walltime limit
#SBATCH -o logs/%x-%j.out              # log path (relative to submit dir)

# Use the dir the job was submitted from. Lets the same script work from any
# worktree: e.g. `cd /home/melwani/fun/craftax-impl && sbatch scripts/train.sbatch`.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

source .venv/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Tunables (override at submit time, e.g.
#   SIL_COEF=0.1 RUN_NAME=sil_0p1_100m sbatch scripts/train.sbatch).
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000000}"
SIL_COEF="${SIL_COEF:-0.0}"
RUN_NAME="${RUN_NAME:-craftax_classic_ppo_pmap}"
WANDB_PROJECT="${WANDB_PROJECT:-craftax-ppo}"

python train.py \
    --require-num-devices 0 \
    --num-envs-per-device 1024 \
    --num-steps 32 \
    --num-minibatches 16 \
    --update-epochs 4 \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --sil-coef "$SIL_COEF" \
    --run-name "$RUN_NAME" \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    ${EXTRA_ARGS:-}
