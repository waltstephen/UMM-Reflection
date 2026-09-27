#!/usr/bin/env bash
# Build the uint8 pixel caches that the SFT readers require.
#
# Input (under $UNIFY_RL_DATA_ROOT/sft):
#   trajectory_parquet/   Clean29K reflection trajectories
#   anchor/parquet/       base-BAGEL prompt-only anchor trajectories
# Output:
#   pixel_cache/          cache for trajectory_parquet
#   anchor/pixel_cache/   cache for anchor/parquet
#
# parquet_info.json is regenerated first because BAGEL keys it by absolute
# shard path. Only the pixel stage is built; it needs no model weights.
#
# Optional: UNIFY_RL_DATA_ROOT, NPROC (default 8), PYTHON_BIN, OVERWRITE=1
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIFY_RL_DATA_ROOT="$(realpath -m "${UNIFY_RL_DATA_ROOT:-${ROOT_DIR}/data}")"
export UNIFY_RL_DATA_ROOT
SFT_ROOT="${UNIFY_RL_DATA_ROOT}/sft"
PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC="${NPROC:-8}"
EXTRA_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

cd "${ROOT_DIR}"
"${PYTHON_BIN}" scripts/data/write_parquet_info.py \
  "${SFT_ROOT}/trajectory_parquet" \
  "${SFT_ROOT}/anchor/parquet"

build() {
  local parquet_dir="$1" cache_dir="$2"
  echo "building pixel cache: ${parquet_dir} -> ${cache_dir}"
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NPROC}" \
    scripts/precompute_failaware_cache.py \
    --parquet-dir "${parquet_dir}" \
    --cache-dir "${cache_dir}" \
    --stage pixels \
    "${EXTRA_ARGS[@]}"
}

build "${SFT_ROOT}/trajectory_parquet" "${SFT_ROOT}/pixel_cache"
build "${SFT_ROOT}/anchor/parquet" "${SFT_ROOT}/anchor/pixel_cache"
