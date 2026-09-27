"""Optimizer isolation and checkpoint helpers for V20 full-rank training."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import socket
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.distributed as dist


VERSION = "clean29529_v20_update_contract_v1"
PROBE_CHECKPOINT_VERSION = "clean29529_v20_probe_checkpoint_v1"
SHARDED_CHECKPOINT_VERSION = "clean29529_v20_sharded_checkpoint_v1"
G016_WORLD32_MIGRATION_CHECKPOINT_VERSION = (
    "clean29529_g016_world32_migrated_checkpoint_v1"
)
REFERENCE_STATE_KEY_MARKERS = (
    "frozen_decoder_layers",
    "language_model_ref",
    "reference_decoder_layers",
    "shared_prefix_reference",
)


def _unwrap_optimizer(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.Optimizer:
    # accelerate's AcceleratedOptimizer is ITSELF a subclass of
    # torch.optim.Optimizer, so the old `while not isinstance(current,
    # Optimizer)` loop body never executed and this returned the WRAPPER.
    # The AdamW isinstance check below then failed and killed every rank at
    # canary startup. Unwrap to the innermost `.optimizer` regardless of
    # Optimizer-ness, keeping the cycle guard.
    current = optimizer
    seen: set[int] = {id(current)}
    while True:
        inner = getattr(current, "optimizer", None)
        if inner is None:
            break
        if id(inner) in seen:
            raise RuntimeError("V20 optimizer wrapper cycle detected")
        seen.add(id(inner))
        current = inner
    if not isinstance(current, torch.optim.Optimizer):
        raise TypeError("V20 optimizer is not an optimizer wrapper")
    return current


def materialize_adamw_state(
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    from torch.optim.optimizer import _get_scalar_dtype

    owner = _unwrap_optimizer(optimizer)
    if not isinstance(owner, torch.optim.AdamW):
        raise TypeError("V20 state materialization requires AdamW")
    parameter_count = 0
    preexisting_state_count = 0
    initialized_state_count = 0
    for group in owner.param_groups:
        fused = bool(group.get("fused", False))
        capturable = bool(group.get("capturable", False))
        amsgrad = bool(group.get("amsgrad", False))
        for parameter in group["params"]:
            parameter_count += 1
            if parameter.dtype is not torch.float32:
                raise RuntimeError(
                    "V20 AdamW state requires fp32 local parameters"
                )
            state = owner.state[parameter]
            if state:
                preexisting_state_count += 1
            else:
                state["step"] = (
                    torch.zeros(
                        (),
                        dtype=_get_scalar_dtype(is_fused=fused),
                        device=parameter.device,
                    )
                    if capturable or fused
                    else torch.tensor(
                        0.0,
                        dtype=_get_scalar_dtype(),
                    )
                )
                state["exp_avg"] = torch.zeros_like(
                    parameter,
                    memory_format=torch.preserve_format,
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    parameter,
                    memory_format=torch.preserve_format,
                )
                if amsgrad:
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format,
                    )
                initialized_state_count += 1
            required = {"step", "exp_avg", "exp_avg_sq"}
            if not required.issubset(state):
                raise RuntimeError(
                    "V20 AdamW state coverage is incomplete"
                )
            if (
                float(state["step"].item()) != 0.0
                or torch.count_nonzero(state["exp_avg"]).item() != 0
                or torch.count_nonzero(state["exp_avg_sq"]).item() != 0
                or state["exp_avg"].shape != parameter.shape
                or state["exp_avg_sq"].shape != parameter.shape
                or state["exp_avg"].dtype != parameter.dtype
                or state["exp_avg_sq"].dtype != parameter.dtype
                or state["step"].dtype is not torch.float32
            ):
                raise RuntimeError(
                    "V20 materialized AdamW state is not exact zero"
                )
    if len(owner.state) != parameter_count:
        raise RuntimeError("V20 AdamW state does not cover every parameter")
    return {
        "version": "clean29529_v20_adamw_state_materialization_v1",
        "parameter_count": parameter_count,
        "preexisting_state_count": preexisting_state_count,
        "initialized_state_count": initialized_state_count,
        "state_parameter_count": len(owner.state),
        "all_steps_zero": True,
        "all_moments_zero": True,
        "moment_dtype": "torch.float32",
        "step_dtype": "torch.float32",
        "optimizer_step_called": False,
    }


class OptimizerStepGuard:
    def __init__(
        self,
        named_optimizers: Iterable[
            tuple[str, torch.optim.Optimizer]
        ],
    ) -> None:
        self.attempt_counts: dict[str, int] = {}
        self._original_steps: list[tuple[Any, Any]] = []
        for name, optimizer in named_optimizers:
            label = str(name)
            self.attempt_counts[label] = 0
            original = optimizer.step

            def forbidden_step(
                *args,
                _label=label,
                **kwargs,
            ):
                del args, kwargs
                self.attempt_counts[_label] += 1
                raise RuntimeError(
                    "V20 memory probe forbids optimizer.step: "
                    f"{_label}"
                )

            self._original_steps.append((optimizer, original))
            optimizer.step = forbidden_step

    def restore(self) -> None:
        for optimizer, original in self._original_steps:
            optimizer.step = original
        self._original_steps.clear()

    def report(self) -> dict[str, Any]:
        return {
            "version": "clean29529_v20_optimizer_step_guard_v1",
            "attempt_counts": dict(self.attempt_counts),
            "total_attempt_count": sum(self.attempt_counts.values()),
            "passed": not any(self.attempt_counts.values()),
        }


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_snapshot(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> dict[str, dict[str, Any]]:
    return {
        str(name): {
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "sha256": tensor_sha256(parameter),
        }
        for name, parameter in named_parameters
    }


def gradient_inventory(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> dict[str, Any]:
    rows = []
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            state = "none"
            nonzero_numel = 0
            max_abs = 0.0
        else:
            detached = gradient.detach()
            nonzero_numel = int(torch.count_nonzero(detached).item())
            state = "nonzero" if nonzero_numel else "zero"
            max_abs = (
                float(detached.float().abs().max().item())
                if detached.numel()
                else 0.0
            )
        rows.append(
            {
                "name": str(name),
                "parameter_numel": int(parameter.numel()),
                "gradient_state": state,
                "gradient_nonzero_numel": nonzero_numel,
                "gradient_max_abs": max_abs,
            }
        )
    return {
        "parameter_count": len(rows),
        "parameter_numel": sum(row["parameter_numel"] for row in rows),
        "none_parameter_count": sum(
            row["gradient_state"] == "none" for row in rows
        ),
        "none_parameter_numel": sum(
            row["parameter_numel"]
            for row in rows
            if row["gradient_state"] == "none"
        ),
        "zero_parameter_count": sum(
            row["gradient_state"] == "zero" for row in rows
        ),
        "zero_parameter_numel": sum(
            row["parameter_numel"]
            for row in rows
            if row["gradient_state"] == "zero"
        ),
        "nonzero_parameter_count": sum(
            row["gradient_state"] == "nonzero" for row in rows
        ),
        "nonzero_parameter_numel": sum(
            row["parameter_numel"]
            for row in rows
            if row["gradient_state"] == "nonzero"
        ),
        "rows": rows,
    }


def optimizer_state_snapshot(
    optimizer: torch.optim.Optimizer,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> dict[str, Any]:
    rows = {}
    for name, parameter in named_parameters:
        state = optimizer.state.get(parameter, {})
        row = {}
        for key in ("step", "exp_avg", "exp_avg_sq"):
            value = state.get(key)
            if torch.is_tensor(value):
                row[key] = {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "sha256": tensor_sha256(value),
                    "scalar": (
                        float(value.item()) if value.numel() == 1 else None
                    ),
                }
            elif value is not None:
                row[key] = {"value": value}
            else:
                row[key] = None
        rows[str(name)] = row
    return {"parameters": rows}


def channel_state_snapshot(
    optimizer: torch.optim.Optimizer,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> dict[str, Any]:
    named = tuple(named_parameters)
    return {
        "parameters": parameter_snapshot(named),
        "optimizer": optimizer_state_snapshot(optimizer, named),
    }


def update_state_snapshot(
    *,
    text_optimizer: torch.optim.Optimizer,
    flow_optimizer: torch.optim.Optimizer,
    text_named_parameters: Iterable[
        tuple[str, torch.nn.Parameter]
    ],
    flow_named_parameters: Iterable[
        tuple[str, torch.nn.Parameter]
    ],
) -> dict[str, Any]:
    return {
        "text": channel_state_snapshot(
            text_optimizer,
            text_named_parameters,
        ),
        "flow": channel_state_snapshot(
            flow_optimizer,
            flow_named_parameters,
        ),
    }


def update_state_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    output = {}
    for channel in ("text", "flow"):
        before_channel = before[channel]
        after_channel = after[channel]
        output[channel] = {
            "changed_parameter_names": changed_keys(
                before_channel["parameters"],
                after_channel["parameters"],
            ),
            "changed_optimizer_parameter_names": changed_keys(
                before_channel["optimizer"]["parameters"],
                after_channel["optimizer"]["parameters"],
            ),
            "parameter_count": len(before_channel["parameters"]),
            "optimizer_parameter_count": len(
                before_channel["optimizer"]["parameters"]
            ),
        }
    return output


def changed_keys(
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[str]:
    names = sorted(set(before) | set(after))
    return [name for name in names if before.get(name) != after.get(name)]


def assert_gradients_none_or_zero(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> dict[str, Any]:
    inventory = gradient_inventory(named_parameters)
    if inventory["nonzero_parameter_count"]:
        names = [
            row["name"]
            for row in inventory["rows"]
            if row["gradient_state"] == "nonzero"
        ]
        raise RuntimeError(f"inactive V20 gradients are nonzero: {names}")
    return inventory


def clear_gradients(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> None:
    for _name, parameter in named_parameters:
        parameter.grad = None


def synchronize_global_activity(
    local_active: bool,
    *,
    device: torch.device | str | None = None,
    process_group: Any = None,
) -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return bool(local_active)
    if device is None:
        backend = dist.get_backend(process_group)
        if backend == "nccl":
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")
    flag = torch.tensor(
        int(bool(local_active)),
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=process_group)
    return bool(flag.item())


def step_channel_optimizer(
    optimizer: torch.optim.Optimizer,
    *,
    local_active: bool,
    activity_device: torch.device | str | None = None,
    process_group: Any = None,
    inactive_named_parameters: Iterable[
        tuple[str, torch.nn.Parameter]
    ] = (),
) -> dict[str, Any]:
    inactive = tuple(inactive_named_parameters)
    inventory = assert_gradients_none_or_zero(inactive)
    clear_gradients(inactive)
    globally_active = synchronize_global_activity(
        local_active,
        device=activity_device,
        process_group=process_group,
    )
    if not globally_active:
        optimizer.zero_grad(set_to_none=True)
        return {
            "stepped": False,
            "optimizer_step_call_count": 0,
            "reason": "globally_inactive_channel",
            "local_active": bool(local_active),
            "globally_active": False,
            "inactive_gradient_inventory": inventory,
        }
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "stepped": True,
        "optimizer_step_call_count": 1,
        "reason": None,
        "local_active": bool(local_active),
        "globally_active": True,
        "inactive_gradient_inventory": inventory,
    }


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rank(process_group: Any = None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("V20 formal checkpoint requires distributed init")
    return int(dist.get_rank(group=process_group))


def _world_size(process_group: Any = None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("V20 formal checkpoint requires distributed init")
    return int(dist.get_world_size(group=process_group))


def _group_global_ranks(process_group: Any = None) -> list[int]:
    if process_group is None:
        return list(range(dist.get_world_size()))
    return [
        int(value) for value in dist.get_process_group_ranks(process_group)
    ]


def _checkpoint_group_contract(
    *,
    process_group: Any,
    control_group: Any,
) -> tuple[list[int], int]:
    policy_ranks = _group_global_ranks(process_group)
    control_ranks = _group_global_ranks(control_group)
    if policy_ranks != control_ranks:
        raise RuntimeError(
            "V20 checkpoint policy/control group global ranks differ: "
            f"policy={policy_ranks} control={control_ranks}"
        )
    writer_global_rank = int(policy_ranks[0])
    if writer_global_rank not in control_ranks:
        raise RuntimeError(
            "V20 checkpoint control group omits the writer global rank"
        )
    return policy_ranks, writer_global_rank


def _cleanup_paths(paths: Iterable[Path]) -> str | None:
    try:
        for path in paths:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _write_checkpoint_rank_evidence(
    evidence_dir: Path | None,
    *,
    stage: str,
    status: str,
    error: BaseException | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Durable rank-local checkpoint evidence with no collective dependency.

    This is load-bearing: when one rank fails inside an FSDP collective, peers
    cannot join a post-failure Gloo error gather. The file is written before
    fail-fast termination and outside `.checkpoint-N.tmp`, so cleanup cannot
    erase the only useful exception.
    """
    if evidence_dir is None:
        return
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    global_rank = int(dist.get_rank()) if dist.is_initialized() else -1
    payload: dict[str, Any] = {
        "version": "clean29529_v20_checkpoint_rank_evidence_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "stage": str(stage),
        "status": str(status),
        "global_rank": global_rank,
        "local_rank": int(os.environ.get("LOCAL_RANK", "-1")),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        payload["cuda"] = {
            "device": int(device),
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
        }
    if error is not None:
        payload["error"] = {
            "exception_type": type(error).__name__,
            "message": str(error),
            "repr": repr(error),
            "traceback": "".join(
                traceback.format_exception(
                    type(error), error, error.__traceback__
                )
            ),
        }
    if extra:
        payload["extra"] = extra
    path = evidence_dir / (
        f"rank-{global_rank:05d}-{str(stage).replace('/', '_')}.json"
    )
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _fsync_file(temporary)
    os.replace(temporary, path)
    _fsync_directory(evidence_dir)


def _distributed_stage(
    stage: str,
    operation: Callable[[], Any],
    *,
    control_group: Any,
    writer_global_rank: int,
    cleanup_on_failure: Iterable[Path] = (),
    failure_evidence_dir: Path | None = None,
    fail_fast_local_error: bool = False,
) -> Any:
    result = None
    local_error = None
    try:
        result = operation()
    except Exception as exc:
        local_error = {
            "global_rank": int(dist.get_rank()),
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        _write_checkpoint_rank_evidence(
            failure_evidence_dir,
            stage=stage,
            status="failed",
            error=exc,
        )
        if fail_fast_local_error:
            # Do not enter the Gloo gather: peers may still be blocked inside
            # the NCCL/FSDP collective that this rank just left. Exiting this
            # rank makes torch elastic terminate the other ranks in seconds
            # instead of waiting for the 600-second NCCL watchdog.
            os._exit(86)
    gathered = [None] * _world_size(control_group)
    dist.all_gather_object(gathered, local_error, group=control_group)
    errors = sorted(
        (value for value in gathered if value is not None),
        key=lambda value: int(value["global_rank"]),
    )
    if not errors:
        return result
    cleanup_error = None
    if int(dist.get_rank()) == int(writer_global_rank):
        cleanup_error = _cleanup_paths(cleanup_on_failure)
    payload = [None]
    if int(dist.get_rank()) == int(writer_global_rank):
        payload[0] = json.dumps(
            {
                "version": "clean29529_v20_checkpoint_stage_failure_v1",
                "stage": str(stage),
                "errors": errors,
                "cleanup_error": cleanup_error,
            },
            sort_keys=True,
        )
    dist.broadcast_object_list(
        payload,
        src=int(writer_global_rank),
        group=control_group,
    )
    raise RuntimeError(str(payload[0]))


def _inject_fault(
    fault_injection: dict[str, Any] | None,
    *,
    stage: str,
) -> None:
    if not fault_injection:
        return
    if (
        str(fault_injection.get("stage") or "") == str(stage)
        and int(fault_injection.get("global_rank", -1))
        == int(dist.get_rank())
    ):
        raise RuntimeError(
            "injected V20 checkpoint failure "
            f"stage={stage} global_rank={dist.get_rank()}"
        )


def _rank_rng_state() -> dict[str, Any]:
    value = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
    try:
        import numpy

        value["numpy"] = numpy.random.get_state()
    except ImportError:
        value["numpy"] = None
    return value


def _restore_rank_rng_state(value: dict[str, Any]) -> None:
    random.setstate(value["python"])
    torch.set_rng_state(value["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(value["torch_cuda"])
    if value.get("numpy") is not None:
        import numpy

        numpy.random.set_state(value["numpy"])


def capture_rank_rng_state() -> dict[str, Any]:
    return _rank_rng_state()


def restore_rank_rng_state(value: dict[str, Any]) -> None:
    _restore_rank_rng_state(value)


def _checkpoint_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"manifest.json", "shard_inventory.json"}
    )


def _file_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "size_bytes": int(path.stat().st_size),
        }
        for path in _checkpoint_files(root)
    ]


def _validate_sharded_checkpoint(
    checkpoint_dir: Path,
    *,
    expected_initialization_reference: dict[str, Any],
    expected_world_size: int,
) -> dict[str, Any]:
    manifest_path = checkpoint_dir / "manifest.json"
    inventory_path = checkpoint_dir / "shard_inventory.json"
    if not manifest_path.is_file() or not inventory_path.is_file():
        raise RuntimeError("V20 sharded checkpoint metadata is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (checkpoint_dir / "INFERENCE_ONLY.json").exists():
        raise RuntimeError("V20 inference-only checkpoint cannot resume training")
    if manifest.get("version") == G016_WORLD32_MIGRATION_CHECKPOINT_VERSION:
        expected_rank_states = [
            checkpoint_dir
            / "rank_state"
            / f"rank-{rank:05d}-of-{expected_world_size:05d}.pt"
            for rank in range(expected_world_size)
        ]
        source_manifest_path = Path(
            str(manifest.get("source_checkpoint_manifest", ""))
        )
        source_manifest = (
            json.loads(source_manifest_path.read_text(encoding="utf-8"))
            if source_manifest_path.is_file()
            else {}
        )
        if (
            manifest.get("status") != "complete"
            or int(manifest.get("world_size", -1)) != int(expected_world_size)
            or int(expected_world_size) != 32
            or manifest.get("initialization_reference")
            != expected_initialization_reference
            or manifest.get("frozen_reference_duplicated") is not False
            or manifest.get("optimizer_state_migration")
            != "split_each_world16_local_shard_into_two_contiguous_world32_shards"
            or source_manifest.get("version") != SHARDED_CHECKPOINT_VERSION
            or source_manifest.get("status") != "complete"
            or int(source_manifest.get("world_size", -1)) != 16
            or int(source_manifest.get("logical_step", -1))
            != int(manifest.get("logical_step", -2))
            or not (checkpoint_dir / "dcp" / ".metadata").is_file()
            or any(
                not path.is_file() or path.stat().st_size <= 0
                for path in expected_rank_states
            )
        ):
            raise RuntimeError("G016 world32 migration checkpoint differs")
        return manifest
    if (
        manifest.get("version") != SHARDED_CHECKPOINT_VERSION
        or manifest.get("status") != "complete"
        or int(manifest.get("world_size", -1)) != int(expected_world_size)
        or manifest.get("initialization_reference")
        != expected_initialization_reference
        or manifest.get("frozen_reference_duplicated") is not False
        or manifest.get("state_dict_options", {}).get(
            "ignore_frozen_params"
        )
        is not True
        or manifest.get("state_dict_options", {}).get("strict") is not False
    ):
        raise RuntimeError("V20 sharded checkpoint manifest differs")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    records = inventory.get("files")
    model_state_keys = inventory.get("model_state_keys")
    reference_tensor_keys = inventory.get("reference_tensor_keys")
    if (
        inventory.get("version")
        != "clean29529_v20_checkpoint_inventory_v1"
        or not isinstance(records, list)
        or not records
        or not isinstance(model_state_keys, list)
        or not model_state_keys
        or int(inventory.get("model_state_key_count", -1))
        != len(model_state_keys)
        or reference_tensor_keys != []
        or any(
            any(
                marker in str(key).lower()
                for marker in REFERENCE_STATE_KEY_MARKERS
            )
            for key in model_state_keys
        )
    ):
        raise RuntimeError("V20 sharded checkpoint inventory is invalid")
    actual_paths = {
        str(path.relative_to(checkpoint_dir))
        for path in _checkpoint_files(checkpoint_dir)
    }
    recorded_paths = {str(record["path"]) for record in records}
    if actual_paths != recorded_paths:
        raise RuntimeError("V20 sharded checkpoint file coverage differs")
    for record in records:
        path = checkpoint_dir / str(record["path"])
        if path.stat().st_size != int(record["size_bytes"]):
            raise RuntimeError(
                f"V20 sharded checkpoint shard differs: {path}"
            )
    expected_rank_states = {
        f"rank_state/rank-{rank:05d}-of-{expected_world_size:05d}.pt"
        for rank in range(expected_world_size)
    }
    if not expected_rank_states.issubset(recorded_paths):
        raise RuntimeError("V20 sharded checkpoint rank state is incomplete")
    if "dcp/.metadata" not in recorded_paths:
        raise RuntimeError("V20 sharded checkpoint has no DCP shards")
    return manifest


def atomic_save_probe_checkpoint(
    checkpoint_dir: Path,
    *,
    payload: dict[str, Any],
    logical_step: int,
    initialization_reference: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.exists():
        raise FileExistsError(checkpoint_dir)
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_dir.with_name(
        f".{checkpoint_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    temporary.mkdir()
    try:
        state_path = temporary / "training_state.pt"
        torch.save(payload, state_path)
        _fsync_file(state_path)
        state_sha256 = file_sha256(state_path)
        manifest = {
            "version": PROBE_CHECKPOINT_VERSION,
            "scope": "cpu_probe_only_not_formal",
            "logical_step": int(logical_step),
            "state_file": state_path.name,
            "state_size_bytes": state_path.stat().st_size,
            "state_sha256": state_sha256,
            "initialization_reference": dict(initialization_reference),
            "frozen_reference_duplicated": False,
            "atomic_directory_commit": True,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _fsync_file(manifest_path)
        _fsync_directory(temporary)
        os.replace(temporary, checkpoint_dir)
        _fsync_directory(checkpoint_dir.parent)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_probe_checkpoint(
    checkpoint_dir: Path,
    *,
    expected_initialization_reference: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("version") != PROBE_CHECKPOINT_VERSION
        or manifest.get("scope") != "cpu_probe_only_not_formal"
        or manifest.get("initialization_reference")
        != expected_initialization_reference
        or manifest.get("frozen_reference_duplicated") is not False
        or manifest.get("atomic_directory_commit") is not True
    ):
        raise RuntimeError("V20 checkpoint manifest differs")
    state_path = checkpoint_dir / str(manifest["state_file"])
    if (
        not state_path.is_file()
        or state_path.stat().st_size != int(manifest["state_size_bytes"])
        or file_sha256(state_path) != manifest["state_sha256"]
    ):
        raise RuntimeError("V20 checkpoint state hash differs")
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("V20 checkpoint payload is invalid")
    return payload, manifest


def _get_model_state_dict_multishard_compat(
    model: torch.nn.Module,
    *,
    options: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pinned DCP model-state routine with a multi-local-shard-safe meta check.

    PyTorch DCP's private `_get_model_state_dict` calls
    `torch.is_tensor(p) and p.is_meta`. Legacy FSDP ShardedTensor dispatches
    `p.is_meta` through `local_tensor()`, which raises when a rank owns more
    than one local shard. Rank 13 owns such a value. This is DCP's routine
    copied from the installed pinned torch, except the final meta filter checks
    every local shard directly. All FQN, frozen-parameter, CPU-offload, and
    state-dict verification semantics are preserved.
    """
    import torch.distributed.checkpoint.state_dict as api

    with api._gc_context():
        info = api._verify_options(
            model,
            (),
            optim_only=False,
            submodules=None,
            options=options,
        )
        with info.fsdp_context():
            state_dict = api._state_dict_fn(model, "state_dict")()

        for key in list(state_dict.keys()):
            fqns = api._get_fqns(model, key)
            if len(fqns) != 1:
                raise RuntimeError(
                    f"V20 checkpoint key has non-unique FQN: {key} {fqns}"
                )
            fqn = next(iter(fqns))
            if fqn != key:
                def verify(candidate: str, expected: str) -> bool:
                    if len(expected) >= len(candidate):
                        return False
                    expected_split = expected.split(".")
                    candidate_split = candidate.split(".")
                    expected_index = 0
                    for candidate_index, candidate_name in enumerate(
                        candidate_split
                    ):
                        if candidate_name == expected_split[expected_index]:
                            expected_index += 1
                            if expected_index == len(expected_split):
                                return candidate_index == len(candidate_split) - 1
                        elif candidate_name in ("module", "_orig_mod"):
                            continue
                        else:
                            return False
                    return True

                if not verify(key, fqn):
                    raise RuntimeError(
                        f"An unexpected key, {key}, exists. FQN is {fqn}"
                    )
                state_dict[fqn] = state_dict.pop(key)

        if info.submodule_prefixes:
            new_state_dict = {}
            for fqn in state_dict.keys():
                for prefix in info.submodule_prefixes:
                    if not fqn.startswith(prefix):
                        continue
                    new_fqn = (
                        fqn if info.keep_submodule_prefixes
                        else fqn[len(prefix):]
                    )
                    new_state_dict[new_fqn] = state_dict[fqn]
            state_dict = new_state_dict

        if info.ignore_frozen_params:
            for key, parameter in model.named_parameters():
                if parameter.requires_grad:
                    continue
                for fqn in api._get_fqns(model, key):
                    state_dict.pop(fqn)

        multi_keys: dict[str, int] = {}
        meta_keys = []
        for key, value in list(state_dict.items()):
            local_shards_fn = getattr(value, "local_shards", None)
            if callable(local_shards_fn):
                local_shards = local_shards_fn()
                if len(local_shards) > 1:
                    multi_keys[str(key)] = len(local_shards)
                if local_shards and all(
                    shard.tensor.is_meta for shard in local_shards
                ):
                    meta_keys.append(str(key))
                    state_dict.pop(key)
                continue
            if torch.is_tensor(value) and value.is_meta:
                meta_keys.append(str(key))
                state_dict.pop(key)

        model_state = api._maybe_full_or_cpu_state_dict(state_dict, info)
        api._verify_state_dict(model_state, {}, info)
    return model_state, {
        "version": "clean29529_v20_dcp_multishard_compat_v2",
        "multi_local_shard_value_count": len(multi_keys),
        "multi_local_shard_keys": multi_keys,
        "removed_meta_keys": meta_keys,
    }

MERGED_OPTIMIZER_CHANNEL_KEY = "v20_optimizer_channel"


def _merged_adamw_for_checkpoint(
    text_optimizer: torch.optim.Optimizer,
    flow_optimizer: torch.optim.Optimizer,
    *,
    separate_channel_groups: bool = False,
) -> torch.optim.AdamW:
    """One-group optimizer union used only for FSDP state conversion.

    Each FSDP layer flat unit contains both regular (text) and `_moe_gen`
    (flow) originals. Even two groups in one optimizer are converted group by
    group by the pinned FSDP path, so the first group is rejected as missing
    the other half. A single union group presents the complete flat unit.
    Runtime text/flow optimizers are never replaced or stepped through this
    object.
    """
    owners = {
        "text": _unwrap_optimizer(text_optimizer),
        "flow": _unwrap_optimizer(flow_optimizer),
    }
    if any(not isinstance(owner, torch.optim.AdamW) for owner in owners.values()):
        raise TypeError("V20 checkpoint merge requires two AdamW optimizers")
    if any(len(owner.param_groups) != 1 for owner in owners.values()):
        raise RuntimeError("V20 checkpoint merge requires one group per optimizer")
    text_parameters = list(owners["text"].param_groups[0]["params"])
    flow_parameters = list(owners["flow"].param_groups[0]["params"])
    text_ids = {id(parameter) for parameter in text_parameters}
    flow_ids = {id(parameter) for parameter in flow_parameters}
    if text_ids & flow_ids:
        raise RuntimeError("V20 text/flow optimizers overlap")
    # Hyperparameters on temporary groups do not enter either runtime owner
    # on restore. `separate_channel_groups=True` is load compatibility for the
    # already-written checkpoint-0 schema; new saves use one complete union
    # group so FSDP conversion sees every flat-unit member together.
    if separate_channel_groups:
        groups = []
        for channel, parameters in (
            ("text", text_parameters),
            ("flow", flow_parameters),
        ):
            source = owners[channel].param_groups[0]
            group = {
                key: value for key, value in source.items() if key != "params"
            }
            group["params"] = parameters
            group[MERGED_OPTIMIZER_CHANNEL_KEY] = channel
            groups.append(group)
    else:
        source = owners["text"].param_groups[0]
        group = {
            key: value for key, value in source.items() if key != "params"
        }
        group["params"] = [*text_parameters, *flow_parameters]
        group[MERGED_OPTIMIZER_CHANNEL_KEY] = "union"
        groups = [group]
    merged = torch.optim.AdamW(groups)
    merged.state.clear()
    for owner in owners.values():
        for parameter, state in owner.state.items():
            merged.state[parameter] = state
    return merged


def _restore_split_adamw_from_merged(
    merged: torch.optim.AdamW,
    *,
    text_optimizer: torch.optim.Optimizer,
    flow_optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Split loaded union state back by exact runtime parameter identity."""
    owners = {
        "text": _unwrap_optimizer(text_optimizer),
        "flow": _unwrap_optimizer(flow_optimizer),
    }
    merged_parameters = {
        id(parameter)
        for group in merged.param_groups
        for parameter in group["params"]
    }
    expected_parameters = {
        id(parameter)
        for owner in owners.values()
        for group in owner.param_groups
        for parameter in group["params"]
    }
    if merged_parameters != expected_parameters:
        raise RuntimeError("V20 merged checkpoint parameter union changed")
    report = {}
    for channel, owner in owners.items():
        if len(owner.param_groups) != 1:
            raise RuntimeError("V20 restore requires one group per optimizer")
        target_parameters = list(owner.param_groups[0]["params"])
        owner.state.clear()
        for parameter in target_parameters:
            if parameter not in merged.state:
                raise RuntimeError(
                    f"V20 merged checkpoint omitted {channel} optimizer state"
                )
            owner.state[parameter] = merged.state[parameter]
        report[channel] = {
            "parameter_count": len(target_parameters),
            "state_count": len(owner.state),
        }
    return report

def _optimizer_checkpoint_inventory(
    model: torch.nn.Module,
    optimizers: tuple[torch.optim.Optimizer, ...],
) -> dict[str, Any]:
    """Rank-local inventory for diagnosing FSDP optimizer-state collection."""
    model_names = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    owners = [_unwrap_optimizer(optimizer) for optimizer in optimizers]
    owner_param_ids: list[set[int]] = []
    rows = []
    for index, (wrapper, owner) in enumerate(zip(optimizers, owners)):
        parameters = [
            parameter
            for group in owner.param_groups
            for parameter in group["params"]
        ]
        parameter_ids = {id(parameter) for parameter in parameters}
        owner_param_ids.append(parameter_ids)
        rows.append(
            {
                "optimizer_index": index,
                "wrapper_type": type(wrapper).__module__
                + "."
                + type(wrapper).__qualname__,
                "owner_type": type(owner).__module__
                + "."
                + type(owner).__qualname__,
                "param_group_count": len(owner.param_groups),
                "parameter_count": len(parameters),
                "state_count": len(owner.state),
                "zero_numel_parameter_count": sum(
                    int(parameter.numel() == 0) for parameter in parameters
                ),
                "requires_grad_parameter_count": sum(
                    int(parameter.requires_grad) for parameter in parameters
                ),
                "parameters_missing_from_model": [
                    {
                        "numel": int(parameter.numel()),
                        "shape": list(parameter.shape),
                    }
                    for parameter in parameters
                    if id(parameter) not in model_names
                ][:20],
                "model_parameter_names": sorted(
                    model_names[id(parameter)]
                    for parameter in parameters
                    if id(parameter) in model_names
                ),
            }
        )
    overlap = (
        owner_param_ids[0] & owner_param_ids[1]
        if len(owner_param_ids) == 2
        else set()
    )
    return {
        "optimizer_count": len(optimizers),
        "model_parameter_count": len(model_names),
        "model_trainable_parameter_count": sum(
            int(parameter.requires_grad) for parameter in model.parameters()
        ),
        "optimizers": rows,
        "cross_optimizer_parameter_overlap_count": len(overlap),
    }


def save_sharded_training_checkpoint(
    checkpoint_dir: Path,
    *,
    model: torch.nn.Module,
    text_optimizer: torch.optim.Optimizer,
    flow_optimizer: torch.optim.Optimizer,
    logical_step: int,
    initialization_reference: dict[str, Any],
    sampler_state: dict[str, Any],
    extra_state: dict[str, Any] | None = None,
    process_group: Any = None,
    control_group: Any = None,
    fault_injection: dict[str, Any] | None = None,
    auxiliary_modules: dict[str, torch.nn.Module] | None = None,
    skip_dcp_optimizer: bool = False,
) -> dict[str, Any] | None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemWriter
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
    )

    checkpoint_dir = Path(checkpoint_dir)
    rank = _rank(process_group)
    world_size = _world_size(process_group)
    control_group = control_group if control_group is not None else process_group
    _policy_ranks, writer_global_rank = _checkpoint_group_contract(
        process_group=process_group,
        control_group=control_group,
    )
    temporary = checkpoint_dir.with_name(f".{checkpoint_dir.name}.tmp")
    failure_evidence_dir = checkpoint_dir.parent / (
        f"{checkpoint_dir.name}.failure_evidence"
    )

    def setup() -> None:
        if int(dist.get_rank()) == writer_global_rank:
            if checkpoint_dir.exists():
                raise FileExistsError(checkpoint_dir)
            if temporary.exists():
                raise FileExistsError(temporary)
            checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
            if failure_evidence_dir.exists():
                shutil.rmtree(failure_evidence_dir)
            failure_evidence_dir.mkdir()
            temporary.mkdir()
            (temporary / "dcp").mkdir()
            (temporary / "rank_state").mkdir()
        _inject_fault(fault_injection, stage="setup")

    _distributed_stage(
        "setup",
        setup,
        control_group=control_group,
        writer_global_rank=writer_global_rank,
        cleanup_on_failure=(temporary, checkpoint_dir),
    )

    # No collective: every rank writes its own optimizer/model inventory before
    # entering FSDP optimizer-state collection. This survives a rank-local
    # exception and exposes asymmetry even if peers remain inside NCCL.
    _write_checkpoint_rank_evidence(
        failure_evidence_dir,
        stage="get_state_dict_preflight",
        status="ready",
        extra=_optimizer_checkpoint_inventory(
            model, (text_optimizer, flow_optimizer)
        ),
    )

    options = StateDictOptions(
        full_state_dict=False,
        cpu_offload=True,
        ignore_frozen_params=True,
        strict=False,
    )

    def collect_state_dicts():
        model_state, model_compat = (
            _get_model_state_dict_multishard_compat(
                model,
                options=options,
            )
        )
        _write_checkpoint_rank_evidence(
            failure_evidence_dir,
            stage="model_state_multishard_compat",
            status="completed",
            extra=model_compat,
        )
        # FSDP flat units contain both text and `_moe_gen` flow parameters.
        # Convert one temporary union optimizer with two tagged groups; each
        # real optimizer by itself owns only half a flat unit and FSDP rejects
        # it as incomplete.
        merged_optimizer = _merged_adamw_for_checkpoint(
            text_optimizer,
            flow_optimizer,
        )
        optimizer_state = (
            {}
            if skip_dcp_optimizer
            else get_optimizer_state_dict(
                model,
                merged_optimizer,
                options=options,
            )
        )
        _inject_fault(fault_injection, stage="get_state_dict")
        return model_state, optimizer_state

    model_state, optimizer_state = _distributed_stage(
        "get_state_dict",
        collect_state_dicts,
        control_group=control_group,
        writer_global_rank=writer_global_rank,
        cleanup_on_failure=(temporary, checkpoint_dir),
        failure_evidence_dir=failure_evidence_dir,
        fail_fast_local_error=(
            torch.cuda.is_available()
            and world_size > 1
            and fault_injection is None
        ),
    )

    def collect_state_key_inventory():
        local = {
            "model_state_keys": sorted(str(key) for key in model_state),
        }
        gathered = [None] * world_size
        dist.all_gather_object(
            gathered,
            local,
            group=control_group,
        )
        if any(value != gathered[0] for value in gathered[1:]):
            raise RuntimeError(
                "V20 sharded checkpoint logical state keys differ by rank"
            )
        reference_keys = [
            key
            for key in local["model_state_keys"]
            if any(
                marker in key.lower()
                for marker in REFERENCE_STATE_KEY_MARKERS
            )
        ]
        if reference_keys:
            raise RuntimeError(
                "V20 frozen reference tensors entered checkpoint state: "
                f"{reference_keys[:5]}"
            )
        return {
            **local,
            "model_state_key_count": len(local["model_state_keys"]),
            "reference_state_key_markers": list(
                REFERENCE_STATE_KEY_MARKERS
            ),
            "reference_tensor_keys": reference_keys,
        }

    state_key_inventory = _distributed_stage(
        "state_key_inventory",
        collect_state_key_inventory,
        control_group=control_group,
        writer_global_rank=writer_global_rank,
        cleanup_on_failure=(temporary, checkpoint_dir),
    )

    def save_dcp() -> None:
        dcp_payload = {"model": model_state}
        if not skip_dcp_optimizer:
            dcp_payload["optimizers"] = optimizer_state
        dcp.save(
            dcp_payload,
            storage_writer=FileSystemWriter(
                temporary / "dcp",
                single_file_per_rank=True,
                sync_files=True,
                overwrite=False,
            ),
            process_group=process_group,
        )
        _inject_fault(fault_injection, stage="dcp_save")

    _distributed_stage(
        "dcp_save",
        save_dcp,
        control_group=control_group,
        writer_global_rank=writer_global_rank,
        cleanup_on_failure=(temporary, checkpoint_dir),
        failure_evidence_dir=failure_evidence_dir,
        fail_fast_local_error=(
            torch.cuda.is_available()
            and world_size > 1
            and fault_injection is None
        ),
    )
    rank_state_path = (
        temporary
        / "rank_state"
        / f"rank-{rank:05d}-of-{world_size:05d}.pt"
    )

    def save_rank_state() -> None:
        _inject_fault(fault_injection, stage="rank_state_save")
        torch.save(
            {
                "version": SHARDED_CHECKPOINT_VERSION,
                "rank": rank,
                "world_size": world_size,
                "logical_step": int(logical_step),
                "initialization_reference": dict(initialization_reference),
                "rng_state": _rank_rng_state(),
                "sampler_state": dict(sampler_state),
                # Exact two-optimizer continuation state, local to this FSDP
                # rank. FSDP optimizer conversion cannot represent two
                # optimizers that split one flat unit, so load restores these
                # real AdamW states directly on the same 14-rank topology.
                "text_optimizer_state": _unwrap_optimizer(
                    text_optimizer
                ).state_dict(),
                "flow_optimizer_state": _unwrap_optimizer(
                    flow_optimizer
                ).state_dict(),
                "auxiliary_module_state": {
                    name: {
                        key: tensor.detach().cpu()
                        for key, tensor in module.state_dict().items()
                    }
                    for name, module in (auxiliary_modules or {}).items()
                },
                "skip_dcp_optimizer": bool(skip_dcp_optimizer),
                "extra_state": dict(extra_state or {}),
            },
            rank_state_path,
        )
        _fsync_file(rank_state_path)

    _distributed_stage(
        "rank_state_save",
        save_rank_state,
        control_group=control_group,
        writer_global_rank=writer_global_rank,
        cleanup_on_failure=(temporary, checkpoint_dir),
    )

    def finalize():
        if int(dist.get_rank()) == writer_global_rank:
            _inject_fault(fault_injection, stage="finalize")
            records = _file_inventory(temporary)
            inventory = {
                "version": "clean29529_v20_checkpoint_inventory_v1",
                "file_integrity": "size_and_presence_no_content_hash",
                "logical_step": int(logical_step),
                "world_size": world_size,
                "files": records,
                "aggregate_size_bytes": sum(
                    int(record["size_bytes"]) for record in records
                ),
                **state_key_inventory,
            }
            inventory_path = temporary / "shard_inventory.json"
            inventory_path.write_text(
                json.dumps(inventory, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _fsync_file(inventory_path)
            manifest = {
                "version": SHARDED_CHECKPOINT_VERSION,
                "status": "complete",
                "logical_step": int(logical_step),
                "world_size": world_size,
                "dcp_directory": "dcp",
                "rank_state_directory": "rank_state",
                "inventory": inventory_path.name,
                "inventory_sha256": file_sha256(inventory_path),
                "aggregate_size_bytes": inventory[
                    "aggregate_size_bytes"
                ],
                "initialization_reference": dict(
                    initialization_reference
                ),
                "frozen_reference_duplicated": False,
                "reference_tensor_keys_in_inventory": inventory[
                    "reference_tensor_keys"
                ],
                "state_dict_options": {
                    "full_state_dict": False,
                    "cpu_offload": True,
                    "ignore_frozen_params": True,
                    "strict": False,
                },
                "optimizers": ["text", "flow"],
                "optimizer_storage": {
                    "version": "clean29529_v20_rank_local_two_adamw_v1",
                    "exact_rank_local_state": True,
                    "world_size_bound": world_size,
                },
                "rng_and_sampler_per_rank": True,
                "atomic_directory_commit": True,
                "file_integrity": "size_and_presence_no_content_hash",
            }
            manifest_path = temporary / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _fsync_file(manifest_path)
            for directory in (
                temporary / "dcp",
                temporary / "rank_state",
                temporary,
            ):
                _fsync_directory(directory)
            os.replace(temporary, checkpoint_dir)
            _fsync_directory(checkpoint_dir.parent)
            return manifest
        return None

    manifest = _distributed_stage(
        "finalize",
        finalize,
        control_group=control_group,
        writer_global_rank=writer_global_rank,
        cleanup_on_failure=(temporary, checkpoint_dir),
    )
    return (
        manifest
        if int(dist.get_rank()) == writer_global_rank
        else None
    )


def load_sharded_training_checkpoint(
    checkpoint_dir: Path,
    *,
    model: torch.nn.Module,
    text_optimizer: torch.optim.Optimizer,
    flow_optimizer: torch.optim.Optimizer,
    expected_initialization_reference: dict[str, Any],
    process_group: Any = None,
    control_group: Any = None,
    fault_injection: dict[str, Any] | None = None,
    auxiliary_modules: dict[str, torch.nn.Module] | None = None,
) -> dict[str, Any]:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import DefaultLoadPlanner, FileSystemReader
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
    )

    checkpoint_dir = Path(checkpoint_dir)
    rank = _rank(process_group)
    world_size = _world_size(process_group)
    control_group = control_group if control_group is not None else process_group
    _policy_ranks, writer_global_rank = _checkpoint_group_contract(
        process_group=process_group,
        control_group=control_group,
    )

    def validate():
        if int(dist.get_rank()) == writer_global_rank:
            value = _validate_sharded_checkpoint(
                checkpoint_dir,
                expected_initialization_reference=(
                    expected_initialization_reference
                ),
                expected_world_size=world_size,
            )
            _inject_fault(fault_injection, stage="validate")
            return value
        return None

    manifest = _distributed_stage(
        "validate",
        validate,
        control_group=control_group,
        writer_global_rank=writer_global_rank,
    )
    manifest_payload = [
        manifest if int(dist.get_rank()) == writer_global_rank else None
    ]
    dist.broadcast_object_list(
        manifest_payload,
        src=writer_global_rank,
        group=control_group,
    )
    manifest = manifest_payload[0]

    options = StateDictOptions(
        full_state_dict=False,
        cpu_offload=True,
        ignore_frozen_params=True,
        strict=False,
    )
    def collect_model_state_dict():
        model_state, _model_compat = (
            _get_model_state_dict_multishard_compat(
                model,
                options=options,
            )
        )
        _inject_fault(fault_injection, stage="load_get_state_dict")
        return model_state

    model_state = _distributed_stage(
        "load_get_state_dict",
        collect_model_state_dict,
        control_group=control_group,
        writer_global_rank=writer_global_rank,
    )
    state = {"model": model_state}

    def load_dcp() -> None:
        # Optimizer DCP metadata from early checkpoints is intentionally
        # ignored. Exact two-AdamW state is rank-local below; DCP owns model
        # shards only on restore.
        dcp.load(
            state,
            storage_reader=FileSystemReader(checkpoint_dir / "dcp"),
            process_group=process_group,
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )
        _inject_fault(fault_injection, stage="dcp_load")

    _distributed_stage(
        "dcp_load",
        load_dcp,
        control_group=control_group,
        writer_global_rank=writer_global_rank,
    )

    def apply_model_state_dict() -> None:
        incompatible = set_model_state_dict(
            model,
            model_state_dict=state["model"],
            options=options,
        )
        import torch.distributed.checkpoint.state_dict as state_dict_api

        expected_missing = {
            fqn
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad
            for fqn in state_dict_api._get_fqns(model, name)
        }
        actual_missing = set(incompatible.missing_keys)
        if (
            actual_missing != expected_missing
            or incompatible.unexpected_keys
        ):
            raise RuntimeError(
                "V20 sharded checkpoint model coverage differs: "
                f"missing={sorted(actual_missing)[:5]} "
                f"expected_frozen={sorted(expected_missing)[:5]} "
                f"unexpected={incompatible.unexpected_keys[:5]}"
            )
        _inject_fault(fault_injection, stage="set_state_dict")

    _distributed_stage(
        "set_state_dict",
        apply_model_state_dict,
        control_group=control_group,
        writer_global_rank=writer_global_rank,
    )
    rank_state_path = (
        checkpoint_dir
        / "rank_state"
        / f"rank-{rank:05d}-of-{world_size:05d}.pt"
    )

    def load_rank_state():
        _inject_fault(fault_injection, stage="rank_state_load")
        value = torch.load(
            rank_state_path,
            map_location="cpu",
            weights_only=False,
        )
        if (
            value.get("version") != SHARDED_CHECKPOINT_VERSION
            or int(value.get("rank", -1)) != rank
            or int(value.get("world_size", -1)) != world_size
            or value.get("initialization_reference")
            != expected_initialization_reference
            or int(value.get("logical_step", -1))
            != int(manifest["logical_step"])
        ):
            raise RuntimeError("V20 sharded checkpoint rank state differs")
        _restore_rank_rng_state(value["rng_state"])
        return value

    rank_state = _distributed_stage(
        "rank_state_load",
        load_rank_state,
        control_group=control_group,
        writer_global_rank=writer_global_rank,
    )

    def restore_auxiliary_modules() -> dict[str, Any]:
        expected = auxiliary_modules or {}
        saved = rank_state.get("auxiliary_module_state") or {}
        if set(saved) != set(expected):
            raise RuntimeError(
                "V20 auxiliary checkpoint module coverage differs: "
                f"saved={sorted(saved)} expected={sorted(expected)}"
            )
        for name, module in expected.items():
            incompatible = module.load_state_dict(saved[name], strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"V20 auxiliary module {name} state differs"
                )
        return {
            "module_count": len(expected),
            "modules": sorted(expected),
        }

    auxiliary_restore = _distributed_stage(
        "auxiliary_module_restore",
        restore_auxiliary_modules,
        control_group=control_group,
        writer_global_rank=writer_global_rank,
    )

    def restore_two_optimizer_state() -> dict[str, Any]:
        text_owner = _unwrap_optimizer(text_optimizer)
        flow_owner = _unwrap_optimizer(flow_optimizer)
        if (
            "text_optimizer_state" in rank_state
            and "flow_optimizer_state" in rank_state
        ):
            text_owner.load_state_dict(rank_state["text_optimizer_state"])
            flow_owner.load_state_dict(rank_state["flow_optimizer_state"])
            source = "rank_local_checkpoint"
        elif int(rank_state["logical_step"]) == 0:
            # Existing checkpoint-0 predates rank-local optimizer storage.
            # At logical step zero the exact AdamW state is deterministic:
            # step=0, exp_avg=0, exp_avg_sq=0 for every parameter.
            materialize_adamw_state(text_owner)
            materialize_adamw_state(flow_owner)
            source = "exact_step0_zero_reconstruction"
        else:
            raise RuntimeError(
                "V20 nonzero checkpoint lacks exact rank-local optimizer state"
            )
        streaming_cpu_master = all(
            owner.__class__.__name__ == "G016StreamingCPUAdamW"
            for owner in (text_owner, flow_owner)
        )
        if streaming_cpu_master:
            # torch Optimizer.load_state_dict follows ordinary AdamW casting
            # rules and casts floating state to the BF16 parameter dtype/device.
            # G016's optimizer contract instead keeps master weights and all
            # moments as FP32 CPU tensors across resume.
            from unify_rl.train.g016_streaming_adamw import (
                normalize_g016_streaming_state,
            )

            streaming_normalization = {
                "text": normalize_g016_streaming_state(text_owner),
                "flow": normalize_g016_streaming_state(flow_owner),
            }
        else:
            streaming_normalization = None
        parameter_counts = {
            "text": sum(
                len(group["params"]) for group in text_owner.param_groups
            ),
            "flow": sum(
                len(group["params"]) for group in flow_owner.param_groups
            ),
        }
        observed = {
            "text": len(text_owner.state),
            "flow": len(flow_owner.state),
        }
        saved_counts = {
            "text": len(rank_state["text_optimizer_state"].get("state", {})),
            "flow": len(rank_state["flow_optimizer_state"].get("state", {})),
        }
        # AdamW state is intentionally lazy. Under FSDP use_orig_params, a
        # rank owns 56 original parameter handles per channel but only local
        # nonempty/gradient-bearing handles have moments. Requiring 56 state
        # entries on every rank rejected an exact rank-local restore (observed
        # counts legitimately ranged from 0 to 36). The load-bearing check is
        # exact saved-to-restored state coverage plus valid finite moments;
        # absent entries remain exact implicit-zero AdamW state and initialize
        # deterministically if that local handle later receives a gradient.
        if observed != saved_counts:
            raise RuntimeError(
                "V20 two-optimizer saved/loaded state coverage differs: "
                f"observed={observed} saved={saved_counts}"
            )
        for channel, owner in (("text", text_owner), ("flow", flow_owner)):
            parameter_ids = {
                id(parameter)
                for group in owner.param_groups
                for parameter in group["params"]
            }
            for parameter, state_value in owner.state.items():
                if id(parameter) not in parameter_ids:
                    raise RuntimeError(
                        f"V20 restored {channel} state has a foreign parameter"
                    )
                required = {"step", "exp_avg", "exp_avg_sq"}
                if streaming_cpu_master:
                    required.add("master_param")
                if not required.issubset(state_value):
                    raise RuntimeError(
                        f"V20 restored {channel} AdamW state is incomplete"
                    )
                if (
                    state_value["exp_avg"].shape != parameter.shape
                    or state_value["exp_avg_sq"].shape != parameter.shape
                    or not torch.isfinite(state_value["step"]).all()
                    or not torch.isfinite(state_value["exp_avg"]).all()
                    or not torch.isfinite(state_value["exp_avg_sq"]).all()
                    or (
                        streaming_cpu_master
                        and (
                            state_value["master_param"].shape != parameter.shape
                            or any(
                                tensor.device.type != "cpu"
                                or tensor.dtype != torch.float32
                                for tensor in (
                                    state_value["step"],
                                    state_value["master_param"],
                                    state_value["exp_avg"],
                                    state_value["exp_avg_sq"],
                                )
                            )
                        )
                    )
                ):
                    raise RuntimeError(
                        f"V20 restored {channel} AdamW state is malformed"
                    )
        return {
            "source": source,
            "state_counts": observed,
            "saved_state_counts": saved_counts,
            "parameter_counts": parameter_counts,
            "lazy_state_exact": True,
            "streaming_cpu_fp32_master_restored": streaming_cpu_master,
            "streaming_state_normalization": streaming_normalization,
        }

    optimizer_restore = _distributed_stage(
        "two_optimizer_state_load",
        restore_two_optimizer_state,
        control_group=control_group,
        writer_global_rank=writer_global_rank,
    )
    return {
        "manifest": manifest,
        "logical_step": int(rank_state["logical_step"]),
        "sampler_state": dict(rank_state["sampler_state"]),
        "extra_state": dict(rank_state["extra_state"]),
        "optimizer_restore": optimizer_restore,
        "auxiliary_restore": auxiliary_restore,
    }


__all__ = [
    "PROBE_CHECKPOINT_VERSION",
    "REFERENCE_STATE_KEY_MARKERS",
    "SHARDED_CHECKPOINT_VERSION",
    "VERSION",
    "OptimizerStepGuard",
    "assert_gradients_none_or_zero",
    "atomic_save_probe_checkpoint",
    "changed_keys",
    "channel_state_snapshot",
    "capture_rank_rng_state",
    "clear_gradients",
    "file_sha256",
    "gradient_inventory",
    "load_sharded_training_checkpoint",
    "materialize_adamw_state",
    "load_probe_checkpoint",
    "optimizer_state_snapshot",
    "parameter_snapshot",
    "restore_rank_rng_state",
    "save_sharded_training_checkpoint",
    "step_channel_optimizer",
    "synchronize_global_activity",
    "tensor_sha256",
    "update_state_delta",
    "update_state_snapshot",
]
