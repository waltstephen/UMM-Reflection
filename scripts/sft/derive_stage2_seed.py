#!/usr/bin/env python3
"""Derive the SFT stage-2 resume seeds from the final stage-1 checkpoint.

Stage 2 resumes the full stage-1 state (model, EMA, optimizer, data sampler)
but switches the LR schedule to a constant 2e-7 and adds a one-epoch sample
target of 167,363 virtual rows. This script never modifies the stage-1
checkpoint: it builds a new directory whose large files are symlinks into the
source and whose ``scheduler.pt`` / ``sample_state.json`` are patched copies.

Two seeds are written:
  <output-root>/smoke_seed/<step>      no sample target (for RUN_MODE=smoke)
  <output-root>/one_epoch_seed/<step>  target 167,363   (for RUN_MODE=full)

Usage:
  python scripts/sft/derive_stage2_seed.py \\
      --source outputs/sft_stage1 --output-root outputs/sft_stage2_seed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

TARGET_SAMPLES = 167_363
CONSTANT_LR = 2e-7
STAGE1_FINAL_STEP = 1_410
WORLD_SIZE = 16


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def discover_source_checkpoint(source: Path) -> Path:
    """Return ``source`` itself if it is a checkpoint, else its latest complete one."""
    if source.name.isdigit() and (source / "CHECKPOINT_COMPLETE.json").is_file():
        return source
    checkpoint_root = source / "ckpts" if (source / "ckpts").is_dir() else source
    candidates = []
    for path in checkpoint_root.glob("[0-9]*"):
        marker = path / "CHECKPOINT_COMPLETE.json"
        if marker.is_file() and read_json(marker).get("status") == "complete":
            candidates.append((int(path.name), path))
    if not candidates:
        raise FileNotFoundError(f"no complete checkpoint under {source}")
    step, path = max(candidates)
    if step < STAGE1_FINAL_STEP - 1:
        raise RuntimeError(
            f"latest checkpoint is step {step}; stage 1 ends at {STAGE1_FINAL_STEP}"
        )
    return path


def validate_source(source: Path) -> tuple[dict, dict, dict]:
    completion = read_json(source / "CHECKPOINT_COMPLETE.json")
    hashes = read_json(source / "checkpoint_hashes.json")
    sample_state = read_json(source / "sample_state.json")
    directory_step = int(source.name)
    if (
        completion.get("status") != "complete"
        or int(completion.get("train_steps", -1)) != directory_step
        or int(hashes.get("train_steps", -1)) != directory_step
        or completion.get("checkpoint_hashes_sha256")
        != sha256_file(source / "checkpoint_hashes.json")
    ):
        raise RuntimeError("checkpoint completion marker does not match its contents")
    last_completed_step = int(sample_state.get("last_completed_step", -1))
    if (
        last_completed_step not in {directory_step, directory_step - 1}
        or int(sample_state.get("world_size", -1)) != WORLD_SIZE
        or int(sample_state.get("target_global_samples", -1)) != 0
        or bool(sample_state.get("target_reached"))
    ):
        raise RuntimeError("sample state is not that of a 16-GPU stage-1 run")
    required = set(completion["required_files"])
    observed = {item["name"]: item for item in hashes["files"]}
    if not required.issubset(observed):
        raise RuntimeError("checkpoint file inventory is incomplete")
    for name in required:
        path = source / name
        if not path.is_file() or path.stat().st_size != int(observed[name]["size_bytes"]):
            raise RuntimeError(f"checkpoint file is missing or truncated: {name}")
    for name in ("sample_state.json", "scheduler.pt"):
        if sha256_file(source / name) != observed[name]["sha256"]:
            raise RuntimeError(f"checkpoint file hash differs: {name}")
    return completion, hashes, sample_state


def patched_scheduler(source: Path, target: Path) -> dict:
    value = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise RuntimeError("scheduler state is not a mapping")
    original = dict(value)
    value["base_lrs"] = [CONSTANT_LR for _ in value.get("base_lrs", [0])]
    value["_last_lr"] = [CONSTANT_LR for _ in value.get("_last_lr", [0])]
    torch.save(value, target)
    return {
        "source_last_epoch": int(original["last_epoch"]),
        "source_step_count": int(original["_step_count"]),
        "source_last_lr": list(original["_last_lr"]),
        "derived_base_lrs": list(value["base_lrs"]),
        "derived_last_lr": list(value["_last_lr"]),
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def derived_sample_state(source: dict, target_samples: int) -> dict:
    value = dict(source)
    milestones = [target_samples] if target_samples > 0 else []
    cumulative = int(value["cumulative_global_samples"])
    value.update(
        {
            "target_global_samples": int(target_samples),
            "sample_milestones": milestones,
            "crossed_milestones": [m for m in milestones if m <= cumulative],
            "newly_crossed_milestones": [],
            "target_reached": bool(target_samples > 0 and cumulative >= target_samples),
            "target_overshoot": (
                max(0, cumulative - target_samples) if target_samples > 0 else 0
            ),
            "recovery_save_every_samples": 0,
            "crossed_recovery_samples": [],
            "newly_crossed_recovery_samples": [],
        }
    )
    return value


def build_seed(
    *,
    source: Path,
    output: Path,
    source_hashes: dict,
    sample_state: dict,
    target_samples: int,
) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing seed: {output}")
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    linked = []
    patched_names = {"scheduler.pt", "sample_state.json"}
    for item in source_hashes["files"]:
        name = str(item["name"])
        if name in patched_names:
            continue
        (temporary / name).symlink_to((source / name).resolve())
        linked.append(
            {
                "name": name,
                "source_sha256": item["sha256"],
                "size_bytes": int(item["size_bytes"]),
            }
        )

    scheduler = patched_scheduler(source / "scheduler.pt", temporary / "scheduler.pt")
    derived_state = derived_sample_state(sample_state, target_samples)
    atomic_json(temporary / "sample_state.json", derived_state)
    record = {
        "status": "complete",
        "derived_at_utc": utc_now(),
        "derived_from": str(source.resolve()),
        "train_steps": int(sample_state["last_completed_step"]),
        "source_directory_step": int(source.name),
        "target_global_samples": target_samples,
        "constant_lr": CONSTANT_LR,
        "linked_files": linked,
        "derived_files": [
            {
                "name": name,
                "sha256": sha256_file(temporary / name),
                "size_bytes": (temporary / name).stat().st_size,
            }
            for name in ("scheduler.pt", "sample_state.json")
        ],
        "scheduler": scheduler,
        "sample_state": derived_state,
    }
    atomic_json(temporary / "DERIVED_SEED_COMPLETE.json", record)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output)
    return {**record, "path": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source", type=Path, required=True,
        help="stage-1 RUN_DIR (latest complete checkpoint is used) or one ckpts/<step> dir",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source = discover_source_checkpoint(args.source)
    _, hashes, sample_state = validate_source(source)
    step_name = f"{int(sample_state['last_completed_step']):07d}"
    seeds = {
        label: build_seed(
            source=source,
            output=args.output_root / label / step_name,
            source_hashes=hashes,
            sample_state=sample_state,
            target_samples=target,
        )
        for label, target in (("smoke_seed", 0), ("one_epoch_seed", TARGET_SAMPLES))
    }
    record = {
        "generated_at_utc": utc_now(),
        "source": str(source.resolve()),
        "last_completed_step": int(sample_state["last_completed_step"]),
        "target_samples": TARGET_SAMPLES,
        "constant_lr": CONSTANT_LR,
        "seeds": {label: seed["path"] for label, seed in seeds.items()},
    }
    atomic_json(args.output_root / "seed_record.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
