#!/bin/bash
#SBATCH --job-name=leflow_uq_head
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=leflow_uq_head_%j.out
#SBATCH --error=leflow_uq_head_%j.err

set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "1" && "$1" != "2" ) ]]; then
  echo "Usage: sbatch jobs/train_uncertainty_head.sh <1|2>" >&2
  echo "The argument selects latent_planner_epoch_1.pt or latent_planner_epoch_2.pt." >&2
  exit 2
fi

SOURCE_EPOCH="$1"
REPO_DIR="${REPO_DIR:-/home/opechler/LeFlow-with-Uncertainty}"

export STABLEWM_HOME="${STABLEWM_HOME:-/gpfs/scratch1/nodespecific/int4/78836/stable-wm}"
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
  "${STABLEWM_HOME}/checkpoints/latent_planner_uncertainty_head"

DATASET_NAME="${DATASET_NAME:-reacher}"
PRETRAINED_RUN_NAME="${PRETRAINED_RUN_NAME:-latent_planner_reacher_full}"
PRETRAINED_DIR="${STABLEWM_HOME}/checkpoints/latent_planner/${PRETRAINED_RUN_NAME}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${PRETRAINED_DIR}/latent_planner_epoch_${SOURCE_EPOCH}.pt}"
DATASET_PATH="${STABLEWM_HOME}/datasets/${DATASET_NAME}.h5"

KEYS_TO_LOAD="[pixels,action,observation]"
KEYS_TO_CACHE="[action,observation]"
HORIZON="${HORIZON:-5}"
MAX_HORIZON="${MAX_HORIZON:-20}"
ACTION_BLOCK="${ACTION_BLOCK:-5}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
HEAD_LR="${HEAD_LR:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
RUN_NAME="${RUN_NAME:-reacher_from_epoch_${SOURCE_EPOCH}_head_ep${EPOCHS}}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-lewm-latent-planner-uncertainty-head}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-false}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Missing Reacher training dataset: ${DATASET_PATH}" >&2
  exit 1
fi

if [[ ! -f "${PRETRAINED_CHECKPOINT}" ]]; then
  echo "Missing epoch-${SOURCE_EPOCH} LeFlow checkpoint: ${PRETRAINED_CHECKPOINT}" >&2
  exit 1
fi

echo "=== Snellius uncertainty-head training started ==="
echo "node=$(hostname)"
echo "repo=${REPO_DIR}"
echo "source_checkpoint=${PRETRAINED_CHECKPOINT}"
echo "source_epoch=${SOURCE_EPOCH}"
echo "dataset=${DATASET_PATH}"
echo "head_epochs=${EPOCHS}"
echo "run_name=${RUN_NAME}"
echo "python=$(which python)"
python -c 'import torch; print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun --unbuffered python train_uncertainty_head.py \
  pretrained_checkpoint="${PRETRAINED_CHECKPOINT}" \
  data.dataset.name="${DATASET_NAME}" \
  data.dataset.keys_to_load="${KEYS_TO_LOAD}" \
  data.dataset.keys_to_cache="${KEYS_TO_CACHE}" \
  planner.horizon="${HORIZON}" \
  planner.max_horizon="${MAX_HORIZON}" \
  planner.action_block="${ACTION_BLOCK}" \
  epochs="${EPOCHS}" \
  log_interval="${LOG_INTERVAL}" \
  loader.batch_size="${BATCH_SIZE}" \
  optimizer.lr="${HEAD_LR}" \
  optimizer.warmup_steps="${WARMUP_STEPS}" \
  subdir="latent_planner_uncertainty_head/${RUN_NAME}" \
  wandb.enabled="${WANDB_ENABLED}" \
  wandb.project="${WANDB_PROJECT}" \
  wandb.entity="${WANDB_ENTITY}" \
  wandb.name="${RUN_NAME}" \
  wandb.mode="${WANDB_MODE}" \
  wandb.log_model="${WANDB_LOG_MODEL}"

echo "=== Snellius uncertainty-head training finished ==="
