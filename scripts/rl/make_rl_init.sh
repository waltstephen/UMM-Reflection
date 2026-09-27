#!/usr/bin/env bash
# Build the RL initialization directory from a finished SFT checkpoint.
#
# The Flow-GRPO trainer loads BAGEL from a single directory and reads the
# policy weights from its ``ema.safetensors``. This script creates that
# directory without copying anything: every base BAGEL asset is a symlink, and
# ``ema.safetensors`` points at the SFT checkpoint's ``model.safetensors``.
#
# Required:  SFT_CHECKPOINT   e.g. outputs/sft_stage2/ckpts/0002969
# Optional:  BAGEL_BASE_DIR   default pretrained/BAGEL-7B-MoT
#            RL_INIT_DIR      default outputs/rl_init
#            COMPUTE_SHA256=1 hash the SFT weights (~29 GB read) and record it
#                             as the run's initialization lineage id
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BAGEL_BASE_DIR="$(realpath -m "${BAGEL_BASE_DIR:-${ROOT_DIR}/pretrained/BAGEL-7B-MoT}")"
SFT_CHECKPOINT="$(realpath -m "${SFT_CHECKPOINT:?set SFT_CHECKPOINT to an SFT ckpts/<step> directory}")"
RL_INIT_DIR="$(realpath -m "${RL_INIT_DIR:-${ROOT_DIR}/outputs/rl_init}")"

[[ -f "${SFT_CHECKPOINT}/model.safetensors" ]] || {
  echo "missing: ${SFT_CHECKPOINT}/model.safetensors" >&2
  exit 1
}
if [[ -e "${RL_INIT_DIR}" ]] && [[ -n "$(ls -A "${RL_INIT_DIR}")" ]]; then
  echo "refusing to write into non-empty ${RL_INIT_DIR}" >&2
  exit 1
fi
mkdir -p "${RL_INIT_DIR}"

for name in ae.safetensors config.json generation_config.json llm_config.json \
    merges.txt tokenizer.json tokenizer_config.json vit_config.json vocab.json; do
  [[ -e "${BAGEL_BASE_DIR}/${name}" ]] || {
    echo "missing base asset: ${BAGEL_BASE_DIR}/${name}" >&2
    exit 1
  }
  ln -s "${BAGEL_BASE_DIR}/${name}" "${RL_INIT_DIR}/${name}"
done
ln -s "${SFT_CHECKPOINT}/model.safetensors" "${RL_INIT_DIR}/ema.safetensors"

if [[ "${COMPUTE_SHA256:-0}" == "1" ]]; then
  sha256sum "${SFT_CHECKPOINT}/model.safetensors" | awk '{print $1}' \
    > "${RL_INIT_DIR}/init_sha256.txt"
  echo "initialization sha256: $(cat "${RL_INIT_DIR}/init_sha256.txt")"
fi
echo "RL init directory ready: ${RL_INIT_DIR}"
