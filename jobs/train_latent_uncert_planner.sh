#!/bin/bash
#SBATCH --job-name=leflow_uq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=leflow_uq_reacher_%j.out
#SBATCH --error=leflow_uq_reacher_%j.err

set -euo pipefail

# SLURM executes a temporary copy of this file from /var/spool/slurm, so the
# repository cannot be located relative to BASH_SOURCE while running as a job.
REPO_DIR="${REPO_DIR:-/home/opechler/LeFlow-with-Uncertainty}"

# Keep all large datasets, checkpoints, and caches on Snellius scratch storage.
export STABLEWM_HOME=/gpfs/scratch1/nodespecific/int4/78836/stable-wm
export HF_HOME="${HF_HOME:-${STABLEWM_HOME}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${STABLEWM_HOME}/cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"
export WANDB_DIR="${WANDB_DIR:-${STABLEWM_HOME}/wandb}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "${REPO_DIR}"
source .venv/bin/activate
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

mkdir -p \
  "${HF_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${MPLCONFIGDIR}" \
  "${WANDB_DIR}" \
  "${STABLEWM_HOME}/checkpoints"

CHECKPOINT_DIR="${STABLEWM_HOME}/checkpoints/reacher"
HF_MODEL_REPO="quentinll/lewm-reacher"
LEWM_CHECKPOINT="${LEWM_CHECKPOINT:-${CHECKPOINT_DIR}/weights.pt}"
DATASET_NAME="reacher"
KEYS_TO_LOAD="[pixels,action,observation]"
KEYS_TO_CACHE="[action,observation]"
HORIZON="${HORIZON:-5}"
MAX_HORIZON="${MAX_HORIZON:-20}"
ACTION_BLOCK="${ACTION_BLOCK:-5}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
RUN_NAME="${RUN_NAME:-latent_uncert_planner_${DATASET_NAME}_h${HORIZON}_ep${EPOCHS}}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-lewm-latent-uncert-planner}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

DATASET_PATH="${STABLEWM_HOME}/datasets/${DATASET_NAME}.h5"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Missing dataset: ${DATASET_PATH}" >&2
  exit 1
fi

# The uncertainty planner freezes the pretrained LeWM encoder/dynamics model.
# Download its small public state dict once if it is not in the cache yet.
if [[ "${LEWM_CHECKPOINT}" == "${CHECKPOINT_DIR}/weights.pt" && ! -f "${LEWM_CHECKPOINT}" ]]; then
  mkdir -p "${CHECKPOINT_DIR}"
  hf download "${HF_MODEL_REPO}" weights.pt \
    --local-dir "${CHECKPOINT_DIR}"
fi

if [[ "${LEWM_CHECKPOINT}" == /* && ! -f "${LEWM_CHECKPOINT}" ]]; then
  echo "Missing LeWM checkpoint: ${LEWM_CHECKPOINT}" >&2
  exit 1
fi

echo "=== Snellius uncertainty latent planner training started ==="
echo "node=$(hostname)"
echo "repo=${REPO_DIR}"
echo "stablewm_home=${STABLEWM_HOME}"
echo "dataset=${DATASET_PATH}"
echo "lewm_checkpoint=${LEWM_CHECKPOINT}"
echo "python=$(which python)"
python -c 'import torch; print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun --unbuffered python train_latent_uncert_planner.py \
  lewm_checkpoint="${LEWM_CHECKPOINT}" \
  data.dataset.name="${DATASET_NAME}" \
  data.dataset.keys_to_load="${KEYS_TO_LOAD}" \
  data.dataset.keys_to_cache="${KEYS_TO_CACHE}" \
  planner.horizon="${HORIZON}" \
  planner.max_horizon="${MAX_HORIZON}" \
  planner.action_block="${ACTION_BLOCK}" \
  epochs="${EPOCHS}" \
  log_interval="${LOG_INTERVAL}" \
  loader.batch_size="${BATCH_SIZE}" \
  subdir="latent_uncert_planner/${RUN_NAME}" \
  wandb.enabled="${WANDB_ENABLED}" \
  wandb.project="${WANDB_PROJECT}" \
  wandb.entity="${WANDB_ENTITY}" \
  wandb.name="${RUN_NAME}" \
  wandb.mode="${WANDB_MODE}"

echo "=== Uncertainty latent planner training finished ==="
