#!/usr/bin/env bash
# SFT stage 2: continue stage 1 at constant LR 2e-7 until one full epoch of the
# 167,363-row virtual mixture has been consumed (step 2969 with 16 GPUs). The
# final checkpoint, ckpts/0002969/model.safetensors, is the RL initialization.
#
# RESUME_FROM is a seed derived from the last stage-1 checkpoint. The trainer
# requires the seed directory to be named after the last completed step, which
# is 1409 for the 1410-step stage 1:
#   python scripts/sft/derive_stage2_seed.py --source outputs/sft_stage1 \
#       --output-root outputs/sft_stage2_seed
# This writes smoke_seed/<step> (for RUN_MODE=smoke: 3 steps, no save) and
# one_epoch_seed/<step> (for RUN_MODE=full). Run once on each node:
#   NODE_RANK=0 MASTER_ADDR=<node0> MASTER_PORT=29500 RUN_DIR=outputs/sft_stage2 \
#       RESUME_FROM=outputs/sft_stage2_seed/one_epoch_seed/0001409 \
#       bash scripts/sft/train_sft_stage2.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
RESUME_FROM="${RESUME_FROM:?set RESUME_FROM to a seed from derive_stage2_seed.py}"
RESUME_FROM="$(realpath "${RESUME_FROM}")"

if [[ "${RUN_MODE}" == "smoke" ]]; then
  SOURCE_STEP=$((10#$(basename "${RESUME_FROM}")))
  TOTAL_STEPS=$((SOURCE_STEP + 3))
  TARGET_GLOBAL_SAMPLES=0
  SAVE_EVERY=1000000
  export FAILAWARE_SKIP_FINAL_CKPT=1
else
  TOTAL_STEPS=3400  # upper bound; the run stops once the sample target is reached
  TARGET_GLOBAL_SAMPLES=167363
  SAVE_EVERY=700
  export FAILAWARE_SKIP_FINAL_CKPT=0
fi

for required in DERIVED_SEED_COMPLETE.json model.safetensors scheduler.pt sample_state.json; do
  [[ -e "${RESUME_FROM}/${required}" ]] || {
    echo "missing seed file: ${RESUME_FROM}/${required}" >&2
    exit 1
  }
done

"${PYTHON_BIN}" - "${RESUME_FROM}/DERIVED_SEED_COMPLETE.json" "${RUN_MODE}" \
  "${TARGET_GLOBAL_SAMPLES}" <<'PY'
import json
import sys

seed = json.load(open(sys.argv[1], encoding="utf-8"))
mode, target = sys.argv[2], int(sys.argv[3])
state = seed["sample_state"]
assert seed["status"] == "complete"
assert seed["train_steps"] == int(state["last_completed_step"])
assert seed["constant_lr"] == 2e-7
assert state["target_global_samples"] == target, (
    f"seed target {state['target_global_samples']} != {target}; "
    "full needs one_epoch_seed/<step>, smoke needs smoke_seed/<step>"
)
assert state["sample_milestones"] == ([167363] if mode == "full" else [])
PY

cd "${ROOT_DIR}"
exec "${TORCHRUN[@]}" \
  "${COMMON_ARGS[@]}" \
  --resume_from "${RESUME_FROM}" \
  --resume_model_only False \
  --finetune_from_ema False \
  --lr 2e-7 \
  --lr_scheduler constant \
  --warmup_steps 0 \
  --min_lr 2e-7 \
  --total_steps "${TOTAL_STEPS}" \
  --lr_schedule_steps 0 \
  --target_global_samples "${TARGET_GLOBAL_SAMPLES}" \
  --save_every "${SAVE_EVERY}" \
  --wandb_project unify-rl-sft \
  --wandb_name "sft-stage2-${RUN_MODE}" \
  --wandb_runid "sft-stage2-${RUN_MODE}-$(date +%Y%m%d%H%M)"
