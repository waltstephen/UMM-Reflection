"""Six-family GenEval reflection RL: stage identity, data pool and topology.

The learning rule is the whole-trajectory GRPO in ``g022_campaign``; this stage
fixes the task distribution to all six core GenEval families.

* **Graded family scores.** Five of six families ship as ``1.0 on pass else
  0.0``. ``R(tau)`` is built on per-round improvements, so a binary q would make
  the reflection bonus, the sustained-progress bonus and the regression penalty
  identically zero. The graded layer restores the gradient without moving the
  pass predicate: ``q == 1.0`` iff the GenEval verdict passed.
* **Credit builder.** ``f01_geneval_trajectory_credit_v1`` computes per-round
  credit from q directly; counting keeps its own equation.
* **Reward service.** ``scripts/serve_clean29529_f01_geneval.py`` scores all six
  families over HTTP.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unify_rl.reward_models.f01_geneval_trajectory_credit_v1 import (
    F01_FAMILIES,
    VERSION as F01_CREDIT_VERSION,
)
from unify_rl.reward_models.g022_whole_trajectory_reward import (
    VERSION as G022_ADVANTAGE_VERSION,
)
from unify_rl.reward_models.geneval_graded_evidence_v1 import (
    EVIDENCE_VERSION as F01_GRADED_EVIDENCE_VERSION,
)
from unify_rl.train.g022_campaign import (
    G022_CLASSIC_PARENT_SHA256,
    G022_ETA_CLAMP_MODE_NAME,
    G022_ROOT_GROUP_COUNT,
    G022_SIBLINGS_PER_ROOT,
    G022_SUSTAINED_PROGRESS_BETA,
    G022_TEXT_BOUND_MODE,
    G022_TRAIN_NUM_TIMESTEPS,
    G022_TRAJECTORY_COUNT,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

F01_REWARD_STAGE = "f01_geneval_full_v1"
F01_TEXT_HEAD_MODE = "g024_uniform"
F01_FORMAL1000_LADDER_STAGE = "f01_geneval_full_formal1000"
F01_NO_COMMIT_STAGE = "f01_geneval_full_no_commit"
F01_SEED = 20260830

# The advantage arithmetic is the whole-trajectory GRPO, so it carries that
# module's version string.
F01_ADVANTAGE_VERSION = G022_ADVANTAGE_VERSION
F01_CLASSIC_PARENT_SHA256 = G022_CLASSIC_PARENT_SHA256
F01_DURABLE_ORIGIN_KIND = "immutable_classic_sft_initialization"
F01_DETECTOR_PORT = int(os.environ.get("G016_DETECTOR_PORT", "18092"))

F01_CORE_FAMILIES = (
    "single_object", "two_object", "counting", "colors", "position", "color_attr",
)
F01_GRADED_FAMILIES = F01_FAMILIES

# ---------------------------------------------------------------------------
# Data. 1000 steps x 2 roots = 2000 distinct roots; the pool holds more, so no
# prompt is drawn twice.
# ---------------------------------------------------------------------------
POOL_ROOT = Path(os.environ.get("F01_POOL_ROOT", str(REPO_ROOT / "assets/data")))
F01_TRAIN_POOL = POOL_ROOT / "f01_geneval_train.jsonl"
F01_DEV_POOL = POOL_ROOT / "f01_geneval_dev.jsonl"
F01_TOTAL_STEPS = 1000
F01_ROOTS_PER_STEP = G022_ROOT_GROUP_COUNT
F01_FORMAL1000_STEPS = (50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)

# ---------------------------------------------------------------------------
# Topology: 32 trajectories per step on 2x8 (local batch 2) or 4x8 (local 1).
# ---------------------------------------------------------------------------
F01_RANKS_PER_NODE = 8
F01_PREFERRED_NODE_COUNT = 2
F01_PREFERRED_WORLD_SIZE = 16
F01_ADMITTED_NODE_COUNTS = (2, 4)
F01_LOCAL_BATCH_SIZE_BY_WORLD = {16: 2, 32: 1}
F01_ACCELERATE_CONFIGS = {
    16: "scripts/accelerate_configs/fsdp_g016_hybrid_16gpu.yaml",
    32: "scripts/accelerate_configs/fsdp_g016_hybrid_32gpu.yaml",
}


def f01_topology(node_count: int) -> dict[str, Any]:
    nodes = int(node_count)
    if nodes not in F01_ADMITTED_NODE_COUNTS:
        raise RuntimeError(
            f"training admits {F01_ADMITTED_NODE_COUNTS} nodes x "
            f"{F01_RANKS_PER_NODE} GPUs, not {nodes}"
        )
    world = nodes * F01_RANKS_PER_NODE
    local = F01_LOCAL_BATCH_SIZE_BY_WORLD[world]
    trajectories = world * local
    if trajectories != G022_TRAJECTORY_COUNT:
        raise RuntimeError(
            f"topology yields {trajectories} trajectories, not "
            f"{G022_TRAJECTORY_COUNT} (= {G022_ROOT_GROUP_COUNT} roots x "
            f"K{G022_SIBLINGS_PER_ROOT})"
        )
    return {
        "node_count": nodes,
        "ranks_per_node": F01_RANKS_PER_NODE,
        "world_size": world,
        "local_batch_size": local,
        "trajectories_per_step": trajectories,
        "root_groups_per_step": G022_ROOT_GROUP_COUNT,
        "siblings_per_root": G022_SIBLINGS_PER_ROOT,
        "accelerate_config": F01_ACCELERATE_CONFIGS[world],
    }


@dataclass(frozen=True)
class Campaign:
    name: str
    ladder_stage: str
    output_root: Path
    target_steps: int
    checkpoint_steps: tuple[int, ...]


FORMAL1000 = Campaign(
    "f01_formal1000",
    F01_FORMAL1000_LADDER_STAGE,
    Path(
        os.environ.get(
            "UNIFY_RL_OUTPUT_ROOT", str(REPO_ROOT / "outputs/f01_formal1000")
        )
    ),
    F01_TOTAL_STEPS,
    F01_FORMAL1000_STEPS,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_resumable_checkpoint(*, output_dir: Path, resume_checkpoint: str) -> None:
    """A resume checkpoint must be a complete commit inside this run root."""
    if not str(resume_checkpoint or "").strip():
        return
    resolved = Path(output_dir).resolve()
    resume = Path(str(resume_checkpoint)).resolve()
    if resume.parent != (resolved / "checkpoints"):
        raise RuntimeError(
            "resume checkpoint must be one this run committed: "
            f"{resume} is not under {resolved / 'checkpoints'}"
        )
    manifest = resume / "manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"resume checkpoint has no manifest: {resume}")
    written = json.loads(manifest.read_text(encoding="utf-8"))
    if written.get("status") != "complete":
        raise RuntimeError(
            f"resume checkpoint is not a completed commit: "
            f"{resume} status={written.get('status')!r}"
        )
    # Resume needs the optimizer moments and per-rank RNG, not just the model
    # weights; an inference-only checkpoint evaluates but cannot resume.
    ranks = sorted((resume / "rank_state").glob("rank-*.pt"))
    if len(ranks) != int(written.get("world_size", -1)):
        raise RuntimeError(
            f"resume needs {written.get('world_size')} rank_state files, "
            f"found {len(ranks)} in {resume / 'rank_state'}"
        )


def assert_pool_supports_independent_steps() -> dict[str, Any]:
    """Every one of the 1000 steps must draw prompts it has never seen."""
    rows = [
        json.loads(line)
        for line in F01_TRAIN_POOL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    uids = {str(row["uid"]) for row in rows}
    if len(uids) != len(rows):
        raise RuntimeError("training pool repeats a uid")
    required = F01_TOTAL_STEPS * F01_ROOTS_PER_STEP
    if len(rows) < required:
        raise RuntimeError(
            f"pool holds {len(rows)} prompts but {F01_TOTAL_STEPS} steps x "
            f"{F01_ROOTS_PER_STEP} roots needs {required}"
        )
    dev_uids = {
        str(json.loads(line)["uid"])
        for line in F01_DEV_POOL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if uids & dev_uids:
        raise RuntimeError("train and dev pools share a uid")
    families = sorted({str(row["family"]) for row in rows})
    if set(families) - set(F01_CORE_FAMILIES):
        raise RuntimeError(f"pool contains an unknown family: {families}")
    return {
        "train_rows": len(rows),
        "dev_rows": len(dev_uids),
        "required_roots": required,
        "headroom": len(rows) / required,
        "families": families,
        "train_sha256": sha256_file(F01_TRAIN_POOL),
        "dev_sha256": sha256_file(F01_DEV_POOL),
    }


def validate_launch_bindings(config: Any) -> dict[str, Any]:
    """Fail closed if a run is not configured as the six-family stage."""
    g016 = config.g016
    problems = []
    if str(g016.reward_stage) != F01_REWARD_STAGE:
        return {"f01": False, "checked": False}
    if str(getattr(g016, "text_head_mode", "")) != F01_TEXT_HEAD_MODE:
        problems.append("text_head_mode")
    if float(g016.sustained_progress_beta) != G022_SUSTAINED_PROGRESS_BETA:
        problems.append("sustained_progress_beta")
    if int(config.sample.num_image_per_prompt) != G022_SIBLINGS_PER_ROOT:
        problems.append("group_size")
    if str(getattr(g016, "text_bound_mode", "")) != G022_TEXT_BOUND_MODE:
        problems.append("text_bound_mode")
    if int(config.sample.num_steps) != G022_TRAIN_NUM_TIMESTEPS:
        problems.append("num_steps")
    if str(getattr(g016, "eta_clamp_mode", "")) != G022_ETA_CLAMP_MODE_NAME:
        problems.append("eta_clamp_mode")
    if problems:
        raise RuntimeError(f"launch bindings differ: {sorted(problems)}")
    return {
        "f01": True,
        "checked": True,
        "version": F01_ADVANTAGE_VERSION,
        "credit_builder_version": F01_CREDIT_VERSION,
        "graded_evidence_version": F01_GRADED_EVIDENCE_VERSION,
        "text_head_mode": F01_TEXT_HEAD_MODE,
        "families": list(F01_CORE_FAMILIES),
    }


__all__ = [
    "Campaign",
    "FORMAL1000",
    "F01_ACCELERATE_CONFIGS",
    "F01_ADMITTED_NODE_COUNTS",
    "F01_ADVANTAGE_VERSION",
    "F01_CLASSIC_PARENT_SHA256",
    "F01_CORE_FAMILIES",
    "F01_CREDIT_VERSION",
    "F01_DETECTOR_PORT",
    "F01_DEV_POOL",
    "F01_DURABLE_ORIGIN_KIND",
    "F01_FORMAL1000_LADDER_STAGE",
    "F01_FORMAL1000_STEPS",
    "F01_GRADED_FAMILIES",
    "F01_LOCAL_BATCH_SIZE_BY_WORLD",
    "F01_NO_COMMIT_STAGE",
    "F01_PREFERRED_NODE_COUNT",
    "F01_PREFERRED_WORLD_SIZE",
    "F01_RANKS_PER_NODE",
    "F01_REWARD_STAGE",
    "F01_ROOTS_PER_STEP",
    "F01_SEED",
    "F01_TEXT_HEAD_MODE",
    "F01_TOTAL_STEPS",
    "F01_TRAIN_POOL",
    "assert_pool_supports_independent_steps",
    "assert_resumable_checkpoint",
    "f01_topology",
    "sha256_file",
    "validate_launch_bindings",
]
