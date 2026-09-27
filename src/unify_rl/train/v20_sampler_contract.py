"""Step-indexed prompt and rollout-seed contract for V20."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from typing import Sequence

import torch


VERSION = "clean29529_v20_step_indexed_sampler_v1"
CONTRACT_HASH_KEY = "sampler_contract_sha256"
CONTRACT_FIELDS = (
    "version",
    "train_data_path",
    "train_data_sha256",
    "ordered_uid_sha256",
    "dataset_size",
    "local_batch_size",
    "group_size",
    "world_size",
    "release_prompt_indices",
    "release_resample_stride",
    "system_prompt_path",
    "system_prompt_sha256",
    "image_steps",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_uid_sha256(uids: Sequence[str]) -> str:
    payload = json.dumps(
        [str(value) for value in uids],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def sampler_contract_sha256(contract: dict[str, Any]) -> str:
    payload = {
        key: contract[key]
        for key in CONTRACT_FIELDS
        if key in contract
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def validate_sampler_contract(contract: dict[str, Any]) -> None:
    if not isinstance(contract, dict):
        raise RuntimeError("V20 sampler contract is not a mapping")
    missing = [key for key in CONTRACT_FIELDS if key not in contract]
    if missing:
        raise RuntimeError(
            f"V20 sampler contract is missing fields: {missing}"
        )
    if (
        contract["version"] != VERSION
        or not Path(str(contract["train_data_path"])).is_absolute()
        or not Path(str(contract["system_prompt_path"])).is_absolute()
        or int(contract["dataset_size"]) <= 0
        or int(contract["local_batch_size"]) <= 0
        or int(contract["group_size"]) <= 0
        or int(contract["world_size"]) <= 0
        or int(contract["release_resample_stride"]) <= 0
        or int(contract["image_steps"]) <= 0
        or not isinstance(contract["release_prompt_indices"], list)
    ):
        raise RuntimeError("V20 sampler contract values are invalid")
    expected_hash = sampler_contract_sha256(contract)
    if contract.get(CONTRACT_HASH_KEY) != expected_hash:
        raise RuntimeError("V20 sampler contract hash differs")


def sampler_config_identity(
    contract: dict[str, Any],
) -> dict[str, Any]:
    validate_sampler_contract(contract)
    return {
        "step_indexed_sampler_version": str(contract["version"]),
        CONTRACT_HASH_KEY: str(contract[CONTRACT_HASH_KEY]),
        "train_data_path": str(contract["train_data_path"]),
        "train_data_sha256": str(contract["train_data_sha256"]),
        "ordered_uid_sha256": str(contract["ordered_uid_sha256"]),
        "dataset_size": int(contract["dataset_size"]),
        "local_batch_size": int(contract["local_batch_size"]),
        "group_size": int(contract["group_size"]),
        "world_size": int(contract["world_size"]),
        "release_prompt_indices": [
            int(value) for value in contract["release_prompt_indices"]
        ],
        "release_resample_stride": int(
            contract["release_resample_stride"]
        ),
        "system_prompt_path": str(contract["system_prompt_path"]),
        "system_prompt_sha256": str(contract["system_prompt_sha256"]),
        "image_steps": int(contract["image_steps"]),
    }


def build_sampler_contract(
    *,
    train_data_path: str | Path,
    ordered_uids: Sequence[str],
    local_batch_size: int,
    group_size: int,
    world_size: int,
    release_prompt_indices: Sequence[int],
    release_resample_stride: int,
    system_prompt_path: str | Path,
    image_steps: int,
) -> dict[str, Any]:
    train_path = Path(train_data_path).resolve()
    prompt_path = Path(system_prompt_path).resolve()
    uids = [str(value) for value in ordered_uids]
    if (
        not train_path.is_file()
        or not prompt_path.is_file()
        or not uids
        or local_batch_size <= 0
        or group_size <= 0
        or world_size <= 0
        or release_resample_stride <= 0
        or image_steps <= 0
    ):
        raise ValueError("V20 sampler contract inputs are invalid")
    contract = {
        "version": VERSION,
        "train_data_path": str(train_path),
        "train_data_sha256": sha256_file(train_path),
        "ordered_uid_sha256": ordered_uid_sha256(uids),
        "dataset_size": len(uids),
        "local_batch_size": int(local_batch_size),
        "group_size": int(group_size),
        "world_size": int(world_size),
        "release_prompt_indices": [
            int(value) for value in release_prompt_indices
        ],
        "release_resample_stride": int(release_resample_stride),
        "system_prompt_path": str(prompt_path),
        "system_prompt_sha256": sha256_file(prompt_path),
        "image_steps": int(image_steps),
    }
    contract[CONTRACT_HASH_KEY] = sampler_contract_sha256(contract)
    validate_sampler_contract(contract)
    return contract


def step_batch_indices(
    *,
    dataset_size: int,
    batch_size: int,
    group_size: int,
    world_size: int,
    rank: int,
    seed: int,
    logical_step: int,
) -> list[int]:
    if (
        dataset_size <= 0
        or batch_size <= 0
        or group_size <= 0
        or world_size <= 0
        or not 0 <= rank < world_size
        or logical_step <= 0
    ):
        raise ValueError("V20 sampler coordinates are invalid")
    total_samples = int(world_size) * int(batch_size)
    if total_samples % int(group_size):
        raise ValueError("V20 group size does not divide the global batch")
    unique_count = total_samples // int(group_size)
    if unique_count > dataset_size:
        raise ValueError("V20 sampler requests too many unique prompts")
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(logical_step) - 1)
    indices = torch.randperm(
        int(dataset_size),
        generator=generator,
    )[:unique_count].tolist()
    repeated = [
        index for index in indices for _ in range(int(group_size))
    ]
    order = torch.randperm(
        len(repeated),
        generator=generator,
    ).tolist()
    shuffled = [repeated[index] for index in order]
    start = int(rank) * int(batch_size)
    return shuffled[start : start + int(batch_size)]


def rollout_seed(*, base_seed: int, logical_step: int) -> int:
    if logical_step <= 0:
        raise ValueError("V20 rollout logical step must be positive")
    return int(base_seed) + int(logical_step)


def checkpoint_sampler_state(
    *,
    rank: int,
    world_size: int,
    seed: int,
    batches_consumed: int,
    contract: dict[str, Any],
) -> dict[str, Any]:
    validate_sampler_contract(contract)
    if (
        batches_consumed < 0
        or int(world_size) != int(contract["world_size"])
        or not 0 <= int(rank) < int(world_size)
    ):
        raise ValueError("V20 checkpoint sampler coordinates are invalid")
    identity = sampler_config_identity(contract)
    return {
        "version": VERSION,
        "rank": int(rank),
        "world_size": int(world_size),
        "seed": int(seed),
        **identity,
        "contract": deepcopy(contract),
        "batches_consumed": int(batches_consumed),
        "completed_step": int(batches_consumed),
        "next_step": int(batches_consumed) + 1,
        "next_rollout_seed": (
            rollout_seed(
                base_seed=seed,
                logical_step=int(batches_consumed) + 1,
            )
        ),
    }


def validate_sampler_resume(
    state: dict[str, Any],
    *,
    rank: int,
    world_size: int,
    seed: int,
    completed_step: int,
    contract: dict[str, Any],
) -> None:
    validate_sampler_contract(contract)
    if state.get("contract") != contract:
        raise RuntimeError("V20 sampler resume contract differs")
    try:
        expected = checkpoint_sampler_state(
            rank=rank,
            world_size=world_size,
            seed=seed,
            batches_consumed=completed_step,
            contract=contract,
        )
    except ValueError as exc:
        raise RuntimeError(
            "V20 sampler resume coordinates differ"
        ) from exc
    if state != expected:
        raise RuntimeError(
            f"V20 sampler resume state differs: expected={expected} got={state}"
        )


__all__ = [
    "CONTRACT_FIELDS",
    "CONTRACT_HASH_KEY",
    "VERSION",
    "build_sampler_contract",
    "checkpoint_sampler_state",
    "ordered_uid_sha256",
    "rollout_seed",
    "sampler_config_identity",
    "sampler_contract_sha256",
    "sha256_file",
    "step_batch_indices",
    "validate_sampler_contract",
    "validate_sampler_resume",
]
