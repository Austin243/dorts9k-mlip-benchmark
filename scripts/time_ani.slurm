#!/bin/bash
#SBATCH --job-name=inpaint_test
#SBATCH --array=0-7
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=ani-timing-%A_%a.log

# Add your site's partition/account settings; activate the ANI environment.
set -euo pipefail
: "${DORTS_SOURCE:?Set the original DORTS HDF5 path}"
: "${ANI_MODEL_INFO:?Set the ANI-1xnr .info path}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python scripts/time_ani.py --source-h5 "$DORTS_SOURCE" --manifest prepared/manifest.h5 \
  --model-info "$ANI_MODEL_INFO" --outdir results/ani_timing \
  --worker-count 8 --worker-id "$SLURM_ARRAY_TASK_ID"
