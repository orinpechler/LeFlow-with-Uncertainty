#!/bin/bash
#SBATCH --job-name=leflow_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=leflow_eval_reacher_%j.out
#SBATCH --error=leflow_eval_reacher_%j.err

set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "1" && "$1" != "2" ) ]]; then
  echo "Usage: sbatch jobs/eval_latent_planner.sh <1|2>" >&2
  echo "The argument selects latent_planner_epoch_1.pt or latent_planner_epoch_2.pt." >&2
  exit 2
fi

EPOCH="$1"

# SLURM executes a temporary copy of this file from /var/spool/slurm, so the
# repository cannot be located relative to BASH_SOURCE while running as a job.
REPO_DIR="${REPO_DIR:-/home/opechler/LeFlow-with-Uncertainty}"

# Match the cache layout used by the Snellius training jobs.
export STABLEWM_HOME="${STABLEWM_HOME:-/gpfs/scratch1/nodespecific/int4/78836/stable-wm}"
export HF_HOME="${HF_HOME:-${STABLEWM_HOME}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${STABLEWM_HOME}/cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "${REPO_DIR}"
source .venv/bin/activate
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

DATASET_NAME="${DATASET_NAME:-reacher}"
RUN_NAME="${RUN_NAME:-latent_planner_reacher_full}"
CHECKPOINT_DIR="${STABLEWM_HOME}/checkpoints/latent_planner/${RUN_NAME}"
PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-${CHECKPOINT_DIR}/latent_planner_epoch_${EPOCH}.pt}"
DATASET_PATH="${STABLEWM_HOME}/datasets/${DATASET_NAME}.h5"

HORIZON="${HORIZON:-5}"
ACTION_BLOCK="${ACTION_BLOCK:-5}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
FLOW_STEPS="${FLOW_STEPS:-16}"
SCORE_MODE="${SCORE_MODE:-rollout_goal}"
SOLVER_BATCH_SIZE="${SOLVER_BATCH_SIZE:-1}"

# LeFlow reports five-run results using seeds 42 through 46.
SEEDS=(42 43 44 45 46)
NUM_EVAL=50

RESULTS_ROOT="${RESULTS_ROOT:-${REPO_DIR}/results}"
RUN_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-${RESULTS_ROOT}/reacher/epoch_${EPOCH}/run_${RUN_ID}}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Missing Reacher evaluation dataset: ${DATASET_PATH}" >&2
  exit 1
fi

if [[ ! -f "${PLANNER_CHECKPOINT}" ]]; then
  echo "Missing epoch-${EPOCH} planner checkpoint: ${PLANNER_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p \
  "${HF_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${MPLCONFIGDIR}" \
  "${RESULTS_DIR}"

echo "=== Snellius LeFlow Reacher evaluation started ==="
echo "node=$(hostname)"
echo "repo=${REPO_DIR}"
echo "checkpoint=${PLANNER_CHECKPOINT}"
echo "checkpoint_epoch=${EPOCH}"
echo "dataset=${DATASET_PATH}"
echo "episodes_per_seed=${NUM_EVAL}"
echo "seeds=${SEEDS[*]}"
echo "results_dir=${RESULTS_DIR}"
echo "python=$(which python)"
python -c 'import torch; print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for seed in "${SEEDS[@]}"; do
  seed_prefix="${RESULTS_DIR}/seed_${seed}"
  echo "=== Evaluating seed ${seed} ==="

  # Videos are disabled for this quantitative sweep. Each run writes a readable
  # text result, a machine-readable JSON result, and its complete console log.
  srun --unbuffered python eval.py --config-name=reacher.yaml \
    solver=latent_flow \
    policy="${PLANNER_CHECKPOINT}" \
    seed="${seed}" \
    eval.num_eval="${NUM_EVAL}" \
    eval.dataset_name="${DATASET_NAME}" \
    plan_config.horizon="${HORIZON}" \
    plan_config.receding_horizon="${HORIZON}" \
    plan_config.action_block="${ACTION_BLOCK}" \
    solver.batch_size="${SOLVER_BATCH_SIZE}" \
    solver.num_samples="${NUM_SAMPLES}" \
    solver.flow_steps="${FLOW_STEPS}" \
    solver.score_mode="${SCORE_MODE}" \
    output.filename="${seed_prefix}.txt" \
    +output.json_filename="${seed_prefix}.json" \
    +output.video_dir=null \
    2>&1 | tee "${seed_prefix}.log"
done

python scripts/summarize_eval_results.py \
  --results-dir "${RESULTS_DIR}" \
  --checkpoint-epoch "${EPOCH}" \
  --seeds "${SEEDS[@]}"

echo "=== Aggregate results ==="
cat "${RESULTS_DIR}/summary.txt"
echo "=== Snellius LeFlow Reacher evaluation finished ==="
