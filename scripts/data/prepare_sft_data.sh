#!/usr/bin/env bash
# Expand Clean29K trajectories into the three SFT row sets.
#
# Requires build_pixel_cache.sh to have finished. Writes, under
# $UNIFY_RL_DATA_ROOT/sft/rows:
#   controller_rows_100/   29,529 controller rows
#   transition_rows_96/    58,020 edit/image transition rows
#   verifier_rows_96/      58,020 verifier rows
#   base_anchor_allowlist.json
# Each row set carries its own parquet_info.json. The output root must not
# exist yet.
#
# Optional: UNIFY_RL_DATA_ROOT, BAGEL_BASE_DIR, PYTHON_BIN
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIFY_RL_DATA_ROOT="$(realpath -m "${UNIFY_RL_DATA_ROOT:-${ROOT_DIR}/data}")"
export UNIFY_RL_DATA_ROOT
BAGEL_BASE_DIR="${BAGEL_BASE_DIR:-${ROOT_DIR}/pretrained/BAGEL-7B-MoT}"
SFT_ROOT="${UNIFY_RL_DATA_ROOT}/sft"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" scripts/prepare_clean29529_multiround_transition_data.py \
  --source-root "${SFT_ROOT}/trajectory_parquet" \
  --cache-root "${SFT_ROOT}/pixel_cache" \
  --anchor-allowlist "${SFT_ROOT}/anchor/base_anchor_allowlist.json" \
  --base-model-dir "${BAGEL_BASE_DIR}" \
  --output-root "${SFT_ROOT}/rows"
