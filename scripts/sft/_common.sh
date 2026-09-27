# Shared environment for the two SFT stages. Sourced, not executed.
#
# Required:  NODE_RANK (0|1), MASTER_ADDR, MASTER_PORT, RUN_DIR
# Optional:  BAGEL_BASE_DIR, UNIFY_RL_DATA_ROOT, RUN_MODE (smoke|full),
#            PYTHON_BIN, WANDB_API_KEY (enables online W&B logging)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BAGEL_BASE_DIR="${BAGEL_BASE_DIR:-${ROOT_DIR}/pretrained/BAGEL-7B-MoT}"
UNIFY_RL_DATA_ROOT="$(realpath -m "${UNIFY_RL_DATA_ROOT:-${ROOT_DIR}/data}")"
export UNIFY_RL_DATA_ROOT
DATASET_CONFIG="configs/sft/clean29529_multiround_transition_16gpu.yaml"
EXPECTED_EMA_SHA256="0b41c43835fd737b8c948e604870da522c091dcf151f3e8d55f84781765ee1a3"

PYTHON_BIN="${PYTHON_BIN:-python}"
NODE_RANK="${NODE_RANK:?set NODE_RANK=0 or 1}"
MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR to the node-0 address}"
MASTER_PORT="${MASTER_PORT:?set MASTER_PORT}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
RUN_MODE="${RUN_MODE:-full}"
NNODES=2
NPROC_PER_NODE=8

# Token caps selected from an all-row token audit of the 167,363 virtual rows.
EXPECTED_NUM_TOKENS=32768
MAX_NUM_TOKENS=40960
MAX_NUM_TOKENS_PER_SAMPLE=28672

if [[ "${NODE_RANK}" != "0" && "${NODE_RANK}" != "1" ]]; then
  echo "NODE_RANK must be 0 or 1" >&2
  exit 1
fi
if [[ "${RUN_MODE}" != "smoke" && "${RUN_MODE}" != "full" ]]; then
  echo "RUN_MODE must be smoke or full" >&2
  exit 1
fi

for required in \
  "${BAGEL_BASE_DIR}/ema.safetensors" \
  "${BAGEL_BASE_DIR}/ae.safetensors" \
  "${ROOT_DIR}/${DATASET_CONFIG}" \
  "${UNIFY_RL_DATA_ROOT}/sft/rows/controller_rows_100/parquet_info.json" \
  "${UNIFY_RL_DATA_ROOT}/sft/rows/transition_rows_96/parquet_info.json" \
  "${UNIFY_RL_DATA_ROOT}/sft/rows/verifier_rows_96/parquet_info.json" \
  "${UNIFY_RL_DATA_ROOT}/sft/rows/base_anchor_allowlist.json" \
  "${UNIFY_RL_DATA_ROOT}/sft/anchor/parquet/parquet_info.json"; do
  [[ -e "${required}" ]] || {
    echo "missing: ${required}" >&2
    exit 1
  }
done

"${PYTHON_BIN}" - "${ROOT_DIR}/${DATASET_CONFIG}" <<'PY'
import sys

import yaml

config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
assert list(config) == [
    "clean29529_multiround_controller",
    "clean29529_transition_mse",
    "clean29529_verifier_state",
    "clean29529_penultimate_verifier_state",
    "base_prompt_only_anchor_mse",
], list(config)
assert [config[name]["weight"] for name in config] == [4, 4, 4, 1, 1]
assert config["clean29529_multiround_controller"]["is_mandatory"] is True
PY

if [[ -n "${WANDB_API_KEY:-}" && "${RUN_MODE}" == "full" ]]; then
  export WANDB_MODE=online
  WANDB_OFFLINE=False
else
  export WANDB_MODE=offline
  WANDB_OFFLINE=True
fi

CHECKPOINT_DIR="${RUN_DIR}/ckpts"
RESULTS_DIR="${RUN_DIR}/logs"
export WANDB_DIR="${RUN_DIR}/wandb"
mkdir -p "${CHECKPOINT_DIR}" "${RESULTS_DIR}" "${WANDB_DIR}"
if [[ "${FRESH_RUN:-1}" == "1" ]] && \
   find "${CHECKPOINT_DIR}" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing non-empty checkpoint directory: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${ROOT_DIR}/third_party/Bagel:${ROOT_DIR}/src:${ROOT_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export FAILAWARE_USE_SEGMENTED_FLASH=1
export FAILAWARE_LOG_SKIPS=1
export FAILAWARE_LOG_FINITE_GATE=1
export FAILAWARE_LOG_BATCH_UIDS=0
export FAILAWARE_TRAIN_DATASET_ROWS=167363
export FAILAWARE_SKIP_MEM_METRICS=1
export UNIFY_RL_DISABLE_FLEX_ATTENTION_COMPILE=0

# Arguments shared by both stages. Stage scripts append the model source,
# learning-rate schedule, step budget, and W&B names.
COMMON_ARGS=(
  --model_path "${BAGEL_BASE_DIR}"
  --llm_path "${BAGEL_BASE_DIR}"
  --vit_path "${BAGEL_BASE_DIR}"
  --vae_path "${BAGEL_BASE_DIR}/ae.safetensors"
  --auto_resume False
  --finetune_from_hf True
  --layer_module Qwen2MoTDecoderLayer
  --dataset_config_file "${DATASET_CONFIG}"
  --visual_gen True
  --visual_und True
  --freeze_vae True
  --freeze_vit True
  --freeze_llm False
  --freeze_und False
  --use_flex False
  --sharding_strategy HYBRID_SHARD
  --backward_prefetch BACKWARD_PRE
  --num_replicate 2
  --num_shard 8
  --ce_weight 1.0
  --mse_weight 1.0
  --ce_loss_reweighting True
  --sample_milestones ""
  --write_checkpoint_completion_marker True
  --recovery_save_every_samples 0
  --gradient_accumulation_steps 1
  --expected_num_tokens "${EXPECTED_NUM_TOKENS}"
  --max_num_tokens "${MAX_NUM_TOKENS}"
  --max_num_tokens_per_sample "${MAX_NUM_TOKENS_PER_SAMPLE}"
  --max_latent_size 64
  --peak_device_tflops 989
  --checkpoint_dir "${CHECKPOINT_DIR}"
  --results_dir "${RESULTS_DIR}"
  --log_every 1
  --wandb_resume never
  --wandb_offline "${WANDB_OFFLINE}"
  --num_workers 2
  --global_seed 20260719
  --data_seed 20260719
)

TORCHRUN=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nnodes="${NNODES}"
  --node_rank="${NODE_RANK}"
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_addr="${MASTER_ADDR}"
  --master_port="${MASTER_PORT}"
  third_party/Bagel/train/pretrain_unified_navit.py
)
