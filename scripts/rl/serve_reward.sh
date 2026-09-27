#!/usr/bin/env bash
# Start the six-family GenEval reward service on node 0.
#
# Run it in the GenEval environment (requirements-geneval.txt), on a GPU the
# policy also uses: with --idle-offload the verifiers sit on CPU and move to
# the GPU only while a scoring request is being served.
#
# Optional:  GENEVAL_ASSETS_DIR  default pretrained/geneval
#            HF_HOME             must already contain bert-base-uncased
#            REWARD_PORT         default 18092 (train_rl.sh expects this)
#            REWARD_DEVICE       default cuda:0
#            UNIFY_RL_OUTPUT_ROOT  default outputs/f01_formal1000
#            PYTHON_BIN          python of the GenEval environment
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export GENEVAL_ASSETS_DIR="$(realpath -m "${GENEVAL_ASSETS_DIR:-${ROOT_DIR}/pretrained/geneval}")"
OUTPUT_ROOT="$(realpath -m "${UNIFY_RL_OUTPUT_ROOT:-${ROOT_DIR}/outputs/f01_formal1000}")"
PYTHON_BIN="${PYTHON_BIN:-python}"
REWARD_PORT="${REWARD_PORT:-18092}"
REWARD_DEVICE="${REWARD_DEVICE:-cuda:0}"

[[ -d "${GENEVAL_ASSETS_DIR}" ]] || {
  echo "missing GENEVAL_ASSETS_DIR: ${GENEVAL_ASSETS_DIR}" >&2
  exit 1
}
mkdir -p "${OUTPUT_ROOT}/reward_service"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" scripts/serve_clean29529_f01_geneval.py \
  --host 0.0.0.0 \
  --port "${REWARD_PORT}" \
  --device "${REWARD_DEVICE}" \
  --idle-offload \
  --manifest-output "${OUTPUT_ROOT}/reward_service/manifest.json" \
  --audit-log "${OUTPUT_ROOT}/reward_service/audit.jsonl"
