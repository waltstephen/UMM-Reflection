#!/usr/bin/env bash
# SFT stage 1: fine-tune base BAGEL-7B-MoT on the reflection SFT mixture for
# 1,410 steps (cosine 2e-6 -> 2e-7, 50 warmup) on 2 nodes x 8 GPUs.
#
# Run once on each node:
#   NODE_RANK=0 MASTER_ADDR=<node0> MASTER_PORT=29500 RUN_DIR=outputs/sft_stage1 \
#       bash scripts/sft/train_sft_stage1.sh
#   NODE_RANK=1 MASTER_ADDR=<node0> MASTER_PORT=29500 RUN_DIR=outputs/sft_stage1 \
#       bash scripts/sft/train_sft_stage1.sh
#
# RUN_MODE=smoke runs two steps without saving. Checkpoints land in
# $RUN_DIR/ckpts/<step>; stage 2 continues from the last one through a seed
# built by scripts/sft/derive_stage2_seed.py.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [[ "${RUN_MODE}" == "smoke" ]]; then
  TOTAL_STEPS=2
  SAVE_EVERY=1000000
  export FAILAWARE_SKIP_FINAL_CKPT=1
else
  TOTAL_STEPS=1410
  SAVE_EVERY=350
  export FAILAWARE_SKIP_FINAL_CKPT=0
fi

if [[ "${VERIFY_BASE_SHA256:-1}" == "1" ]]; then
  echo "checking base ema.safetensors sha256 (VERIFY_BASE_SHA256=0 skips)"
  observed="$(sha256sum "${BAGEL_BASE_DIR}/ema.safetensors" | cut -d' ' -f1)"
  if [[ "${observed}" != "${EXPECTED_EMA_SHA256}" ]]; then
    echo "base ema.safetensors sha256 ${observed} != ${EXPECTED_EMA_SHA256}" >&2
    exit 1
  fi
fi

cd "${ROOT_DIR}"
exec "${TORCHRUN[@]}" \
  "${COMMON_ARGS[@]}" \
  --resume_from "${BAGEL_BASE_DIR}" \
  --resume_model_only True \
  --finetune_from_ema True \
  --lr 2e-6 \
  --lr_scheduler cosine \
  --warmup_steps 50 \
  --min_lr 2e-7 \
  --total_steps "${TOTAL_STEPS}" \
  --lr_schedule_steps 1410 \
  --target_global_samples 0 \
  --save_every "${SAVE_EVERY}" \
  --wandb_project unify-rl-sft \
  --wandb_name "sft-stage1-${RUN_MODE}" \
  --wandb_runid "sft-stage1-${RUN_MODE}-$(date +%Y%m%d%H%M)"
