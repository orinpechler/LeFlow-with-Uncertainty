#!/bin/bash
#SBATCH --job-name=leflow_uq_ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=leflow_uq_finetune_%j.out
#SBATCH --error=leflow_uq_finetune_%j.err

set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "1" && "$1" != "2" ) ]]; then
  echo "Usage: sbatch jobs/finetune_for_uncertainty.sh <1|2>" >&2
  echo "The argument selects latent_planner_epoch_1.pt or latent_planner_epoch_2.pt." >&2
  exit 2
fi

SOURCE_EPOCH="$1"

# SLURM executes a temporary copy of this file from /var/spool/slurm, so the
# repository cannot be located relative to BASH_SOURCE while running as a job.
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
  "${STABLEWM_HOME}/checkpoints/latent_planner_uncertainty"

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
BATCH_SIZE="${BATCH_SIZE:-128}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
HEAD_LR="${HEAD_LR:-1e-5}"
BACKBONE_LR_SCALE="${BACKBONE_LR_SCALE:-0.1}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
RUN_NAME="${RUN_NAME:-reacher_from_epoch_${SOURCE_EPOCH}_ft${EPOCHS}}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-lewm-latent-planner-uncertainty}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Missing Reacher training dataset: ${DATASET_PATH}" >&2
  exit 1
fi

if [[ ! -f "${PRETRAINED_CHECKPOINT}" ]]; then
  echo "Missing epoch-${SOURCE_EPOCH} LeFlow checkpoint: ${PRETRAINED_CHECKPOINT}" >&2
  exit 1
fi

echo "=== Snellius UA-Flow fine-tuning started ==="
echo "node=$(hostname)"
echo "repo=${REPO_DIR}"
echo "source_checkpoint=${PRETRAINED_CHECKPOINT}"
echo "source_epoch=${SOURCE_EPOCH}"
echo "dataset=${DATASET_PATH}"
echo "finetune_epochs=${EPOCHS}"
echo "run_name=${RUN_NAME}"
echo "python=$(which python)"
python -c 'import torch; print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun --unbuffered python finetune_for_uncertainty.py \
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
  optimizer.backbone_lr_scale="${BACKBONE_LR_SCALE}" \
  optimizer.warmup_steps="${WARMUP_STEPS}" \
  subdir="latent_planner_uncertainty/${RUN_NAME}" \
  wandb.enabled="${WANDB_ENABLED}" \
  wandb.project="${WANDB_PROJECT}" \
  wandb.entity="${WANDB_ENTITY}" \
  wandb.name="${RUN_NAME}" \
  wandb.mode="${WANDB_MODE}"

echo "=== Snellius UA-Flow fine-tuning finished ==="
