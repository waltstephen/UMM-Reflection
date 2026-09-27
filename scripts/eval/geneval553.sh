#!/usr/bin/env bash
# GenEval-553 for one arm: sharded generation on N GPUs, then scoring.
#
# Generation runs in the trainer environment (requirements.txt) and scoring in
# the GenEval environment (requirements-geneval.txt); set PYTHON_BIN and
# GENEVAL_PYTHON_BIN to the two interpreters. Re-running the script resumes:
# finished samples are skipped and already-scored samples are not rescored.
#
# Required:  ARM                 base | sft | rl
#            CHECKPOINT          rl only: an RL checkpoint-N directory
# Optional:  MODEL_DIR           default: BAGEL_BASE_DIR for base,
#                                RL_INIT_DIR (the SFT overlay) for sft/rl
#            NUM_GPUS            default 8, one generation shard per GPU
#            OUTPUT_ROOT         default outputs/geneval553
#            LABEL               run name under OUTPUT_ROOT (default: arm and
#                                checkpoint name)
#            REPAIR_RNG_STEP     default 0 for base/sft, 500 for rl (the
#                                convention of the reported numbers)
#            GENERATE_EXTRA_ARGS e.g. "--raw-prompt" or "--r0-only"
#            GENEVAL_ASSETS_DIR  default pretrained/geneval
#            HF_HOME             must already contain bert-base-uncased
#            PYTHON_BIN / GENEVAL_PYTHON_BIN
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARM="${ARM:?set ARM=base, sft or rl}"
CHECKPOINT="${CHECKPOINT:-}"
NUM_GPUS="${NUM_GPUS:-8}"
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT:-${ROOT_DIR}/outputs/geneval553}")"
PYTHON_BIN="${PYTHON_BIN:-python}"
GENEVAL_PYTHON_BIN="${GENEVAL_PYTHON_BIN:-python}"
GENERATE_EXTRA_ARGS="${GENERATE_EXTRA_ARGS:-}"
export GENEVAL_ASSETS_DIR="$(realpath -m "${GENEVAL_ASSETS_DIR:-${ROOT_DIR}/pretrained/geneval}")"

case "${ARM}" in
  base)
    MODEL_DIR="${MODEL_DIR:-${BAGEL_BASE_DIR:-${ROOT_DIR}/pretrained/BAGEL-7B-MoT}}" ;;
  sft|rl)
    MODEL_DIR="${MODEL_DIR:-${RL_INIT_DIR:-${ROOT_DIR}/outputs/rl_init}}" ;;
  *)
    echo "ARM must be base, sft or rl" >&2
    exit 1 ;;
esac
MODEL_DIR="$(realpath -m "${MODEL_DIR}")"
if [[ "${ARM}" == "rl" ]]; then
  [[ -n "${CHECKPOINT}" ]] || { echo "ARM=rl needs CHECKPOINT" >&2; exit 1; }
  CHECKPOINT="$(realpath -m "${CHECKPOINT}")"
elif [[ -n "${CHECKPOINT}" ]]; then
  echo "CHECKPOINT is only valid with ARM=rl" >&2
  exit 1
fi

if [[ -z "${LABEL:-}" ]]; then
  LABEL="${ARM}"
  [[ "${ARM}" == "rl" ]] && LABEL+="-$(basename "${CHECKPOINT}")"
  [[ " ${GENERATE_EXTRA_ARGS} " == *" --raw-prompt "* ]] && LABEL+="-rawprompt"
  [[ " ${GENERATE_EXTRA_ARGS} " == *" --r0-only "* ]] && LABEL+="-r0only"
fi
RUN_DIR="${OUTPUT_ROOT}/${LABEL}"
mkdir -p "${RUN_DIR}/logs"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${ROOT_DIR}"

generate_args=(--arm "${ARM}" --model-dir "${MODEL_DIR}"
  --output-root "${OUTPUT_ROOT}" --label "${LABEL}" --shards "${NUM_GPUS}")
if [[ "${ARM}" == "rl" ]]; then
  generate_args+=(--checkpoint "${CHECKPOINT}")
  if [[ ! -f "${CHECKPOINT}/auxiliary.safetensors" ]]; then
    "${PYTHON_BIN}" scripts/rl/export_checkpoint_auxiliary.py --checkpoint "${CHECKPOINT}"
  fi
fi
[[ -n "${REPAIR_RNG_STEP:-}" ]] && generate_args+=(--repair-rng-step "${REPAIR_RNG_STEP}")
# shellcheck disable=SC2206  # intentional word splitting of extra flags
generate_args+=(${GENERATE_EXTRA_ARGS})

echo "generating ${LABEL} on ${NUM_GPUS} GPUs -> ${RUN_DIR}"
pids=()
for ((shard = 0; shard < NUM_GPUS; shard++)); do
  CUDA_VISIBLE_DEVICES="${shard}" "${PYTHON_BIN}" scripts/eval/generate_geneval553.py \
    "${generate_args[@]}" --shard "${shard}" \
    > "${RUN_DIR}/logs/generate_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for shard in "${!pids[@]}"; do
  if ! wait "${pids[${shard}]}"; then
    echo "shard ${shard} failed; see ${RUN_DIR}/logs/generate_shard${shard}.log" >&2
    failed=1
  fi
done
[[ "${failed}" == "0" ]] || exit 1

# Generation workers have exited, so GPU 0 is free for the verifiers.
CUDA_VISIBLE_DEVICES=0 "${GENEVAL_PYTHON_BIN}" scripts/eval/score_geneval553.py \
  --run-dir "${RUN_DIR}" --device cuda:0 2>&1 | tee "${RUN_DIR}/logs/score.log"
