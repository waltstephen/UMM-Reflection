#!/usr/bin/env bash
# Multi-round Flow-GRPO RL from the SFT checkpoint: 2 nodes x 8 GPUs, 1000 steps.
#
# Run once per node, in the trainer environment (requirements.txt). The reward
# service (scripts/rl/serve_reward.sh) must already be up on node 0.
#
# Required:  NODE_RANK (0|1), MASTER_ADDR (node-0 address), MASTER_PORT
# Optional:  RL_INIT_DIR           default outputs/rl_init (make_rl_init.sh)
#            UNIFY_RL_OUTPUT_ROOT  default outputs/f01_formal1000
#            REWARD_URL            default http://127.0.0.1:18092
#            RESUME_CHECKPOINT + START_STEP  resume a complete checkpoint;
#                                  set both or neither
#            INIT_SHA256           initialization lineage id; defaults to
#                                  RL_INIT_DIR/init_sha256.txt, else the
#                                  published SFT checkpoint's sha256
#            PYTHON_BIN            python of the trainer environment
#
# Resuming checks that the checkpoint was trained from the same RL_INIT_DIR
# path and weights size, so keep that directory in place for the whole run.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FLOW_ROOT="${ROOT_DIR}/third_party/flow_grpo"
RL_INIT_DIR="$(realpath -m "${RL_INIT_DIR:-${ROOT_DIR}/outputs/rl_init}")"
OUTPUT_ROOT="$(realpath -m "${UNIFY_RL_OUTPUT_ROOT:-${ROOT_DIR}/outputs/f01_formal1000}")"
PYTHON_BIN="${PYTHON_BIN:-python}"
NODE_RANK="${NODE_RANK:?set NODE_RANK=0 or 1}"
MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR to the node-0 address}"
MASTER_PORT="${MASTER_PORT:?set MASTER_PORT}"
REWARD_URL="${REWARD_URL:-http://127.0.0.1:18092}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
START_STEP="${START_STEP:-0}"
PUBLISHED_SFT_SHA256="52e1e11193f26471dc253d0b65089a2aa024a68ce218cd916ea911b38eddc370"

TRAIN_DATA="${ROOT_DIR}/assets/data/f01_geneval_train.jsonl"
POOL_MANIFEST="${ROOT_DIR}/assets/data/F01_PROMPT_POOL_MANIFEST.json"

if [[ "${NODE_RANK}" != "0" && "${NODE_RANK}" != "1" ]]; then
  echo "NODE_RANK must be 0 or 1" >&2
  exit 1
fi
# A start step without a checkpoint restarts mid-count from fresh optimizers;
# a checkpoint with start step 0 replays steps that are already durable.
if { [[ -n "${RESUME_CHECKPOINT}" ]] && [[ "${START_STEP}" == "0" ]]; } ||
   { [[ -z "${RESUME_CHECKPOINT}" ]] && [[ "${START_STEP}" != "0" ]]; }; then
  echo "set RESUME_CHECKPOINT and START_STEP together (or neither)" >&2
  exit 1
fi
for required in "${RL_INIT_DIR}/ema.safetensors" "${RL_INIT_DIR}/ae.safetensors" \
    "${TRAIN_DATA}" "${POOL_MANIFEST}"; do
  [[ -e "${required}" ]] || {
    echo "missing: ${required}" >&2
    exit 1
  }
done

if [[ -z "${INIT_SHA256:-}" ]]; then
  if [[ -f "${RL_INIT_DIR}/init_sha256.txt" ]]; then
    INIT_SHA256="$(tr -d '[:space:]' < "${RL_INIT_DIR}/init_sha256.txt")"
  else
    INIT_SHA256="${PUBLISHED_SFT_SHA256}"
  fi
fi

if [[ "${NODE_RANK}" == "0" ]]; then
  curl -fsS --max-time 30 "${REWARD_URL}/health" >/dev/null || {
    echo "reward service is not answering at ${REWARD_URL}/health;" \
      "start scripts/rl/serve_reward.sh first" >&2
    exit 1
  }
fi

mkdir -p "${OUTPUT_ROOT}/logs"

# --- run identity -----------------------------------------------------------
export G016_OUTPUT_DIR="${OUTPUT_ROOT}"
export G016_FLOWGRPO_MODEL_DIR="${RL_INIT_DIR}"
export G016_INITIALIZATION_SHA256="${INIT_SHA256}"
export G016_RUN_NAME="f01-geneval-full-formal1000"
export G016_LADDER_STAGE="f01_geneval_full_formal1000"
export G016_REWARD_STAGE="f01_geneval_full_v1"
export G016_SEED=20260830

# --- data ---------------------------------------------------------------------
export G016_TRAIN_DATA="${TRAIN_DATA}"
export G016_PROMPT_NARROWING_MANIFEST="${POOL_MANIFEST}"
export G016_PROMPT_NARROWING_MANIFEST_SHA256="$(sha256sum "${POOL_MANIFEST}" | awk '{print $1}')"

# --- schedule -----------------------------------------------------------------
export G016_TOTAL_STEPS=1000
export G016_CHECKPOINT_STEPS="50,100,200,300,400,500,600,700,800,900,1000"
export G016_START_STEP="${START_STEP}"
export G016_RESUME_CHECKPOINT="${RESUME_CHECKPOINT}"

# --- group construction: 16 siblings per root, 2 roots per rank --------------
export G016_GROUP_SIZE=16
export G016_NUM_STEPS=20
export G016_LOCAL_BATCH_SIZE=2
export G016_TRAIN_MICRO_BATCH_SIZE=1
export G016_GRADIENT_ACCUMULATION_STEPS=1
export G016_FLOW_LEARNING_RATE=5e-06
export G016_TEXT_LEARNING_RATE=5e-06

# --- memory / parallelism -----------------------------------------------------
export G016_LORA_RANK=0
export G016_OPTIMIZER_STATE_CPU_OFFLOAD=1
export G016_REFERENCE_CPU_STREAMING=1
export G016_BF16_POLICY_CPU_FP32_MASTER=1
export G016_NESTED_FSDP_WRAP=1
export G016_FSDP_STRATEGY=HYBRID_SHARD

# --- reward service -----------------------------------------------------------
export G016_REWARD_URL="${REWARD_URL}"
export G016_DETECTOR_SERVICE_VERSION="geneval_sixfamily_graded_detector_service_v1"

# --- runtime ------------------------------------------------------------------
export G016_WANDB_ENABLED=0
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FAILAWARE_USE_SEGMENTED_FLASH=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=4096
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export PYTHONPATH="${FLOW_ROOT}:${ROOT_DIR}/third_party/Bagel:${ROOT_DIR}/src:${ROOT_DIR}/scripts"

cd "${FLOW_ROOT}"
"${PYTHON_BIN}" -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/fsdp_g016_hybrid_16gpu.yaml \
  --num_machines 2 \
  --num_processes 16 \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  scripts/train_bagel_g016_counting.py \
  --config config/g016.py:multiround_counting \
  2>&1 | tee -a "${OUTPUT_ROOT}/logs/node${NODE_RANK}.train.log"
