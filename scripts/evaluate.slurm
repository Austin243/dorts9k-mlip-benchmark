#!/bin/bash
#SBATCH --job-name=inpaint_test
#SBATCH --array=0-15%8
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=eval-%A_%a.log

# Add your site's partition/account settings. Submit from the repository root
# after activating the model environment. Export these paths before sbatch.
set -euo pipefail
: "${DORTS_SOURCE:?Set the original DORTS HDF5 path}"
: "${MODEL_CHECKPOINT:?Set the checkpoint path}"
: "${BENCHMARK_MODEL:?Set a model id shown by dorts9k models}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
dorts9k evaluate --model "$BENCHMARK_MODEL" --source "$DORTS_SOURCE" \
  --checkpoint "$MODEL_CHECKPOINT" --cohort "${BENCHMARK_COHORT:-full}" \
  --outdir runs/production --num-shards 16 --shard-index "$SLURM_ARRAY_TASK_ID"
