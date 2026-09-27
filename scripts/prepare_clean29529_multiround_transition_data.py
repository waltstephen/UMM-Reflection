#!/usr/bin/env python3
"""Build metadata-only Clean29K controller, transition, and verifier parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
BAGEL = ROOT / "third_party" / "Bagel"
if str(BAGEL) not in sys.path:
    sys.path.insert(0, str(BAGEL))

from data.existing_exact_edit_relocation_contract import (  # noqa: E402
    parse_relocated_response,
)
from data.existing_split_allmse_contract import (  # noqa: E402
    validate_enriched_metadata,
)
from data.interleave_datasets.clean29529_external_edit_only_dataset import (  # noqa: E402
    Clean29529ExternalEditOnlyIterableDataset,
)
from data.interleave_datasets.clean29529_multiround_transition_dataset import (  # noqa: E402
    SOURCE_SYSTEM_PROMPT_VERSION,
    SYSTEM_PROMPT_VERSION,
)


VERSION = "clean29529_multiround_transition_data_v1"
DATA_ROOT = Path(os.environ.get("UNIFY_RL_DATA_ROOT", ROOT / "data"))
SFT_ROOT = DATA_ROOT / "sft"
# Upstream BAGEL-7B-MoT revision 265d1d48ec8e850a29d3a1f208c2a2ec3cd7577b.
BASE_EMA_SHA256 = (
    "0b41c43835fd737b8c948e604870da522"
    "c091dcf151f3e8d55f84781765ee1a3"
)
PLAN_SYSTEM_SUFFIX = (
    "For a planned progression, begin first-round [THINKING] with the complete "
    "compact milestone plan, then execute only the currently due milestone. "
    "Keep that plan in the persistent history for later verification turns."
)
PLANNED_TRANSITION_KINDS = {"planned_initial", "planned_transition"}

SOURCE_COLUMNS = [
    "uid",
    "task",
    "trajectory_subtype",
    "user_prompt",
    "system_prompt",
    "system_prompt_version",
    "meta_path",
    "think_list",
    "step_action_list",
    "step_role_list",
    "step_round_list",
    "step_source_image_list",
    "step_image_name_list",
    "step_image_sha256",
    "step_image_size_bytes",
    "step_need_loss",
    "n_steps",
    "n_images",
    "source_image_step_index",
    "final_loss_step_index",
    "final_image_name",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parquet_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet shards under {root}")
    return paths


def iter_source_rows(source_root: Path):
    for path in parquet_paths(source_root):
        parquet_file = pq.ParquetFile(path)
        for row_group_index in range(parquet_file.num_row_groups):
            for batch in parquet_file.iter_batches(
                row_groups=[row_group_index],
                columns=SOURCE_COLUMNS,
                batch_size=128,
                use_threads=True,
            ):
                for row in batch.to_pylist():
                    yield path, row


def load_cache_steps(cache_root: Path) -> tuple[dict[str, set[int]], dict]:
    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("pixel_complete") is not True:
        raise RuntimeError(f"incomplete pixel cache: {manifest_path}")
    steps = {
        str(uid): {int(step) for step in entry.get("steps") or []}
        for uid, entry in (manifest.get("entries") or {}).items()
    }
    return steps, {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "size_bytes": manifest_path.stat().st_size,
        "entries": len(steps),
        "images": sum(len(value) for value in steps.values()),
    }


def source_step_from_reference(reference: str, *, task: str) -> int:
    value = str(reference)
    if value == "None":
        if task != "t2i":
            raise ValueError("None source is only valid for T2I")
        return -1
    if value == "given":
        if task != "edit":
            raise ValueError("given source is only valid for Edit")
        return 0
    match = re.fullmatch(r"Image #(?P<step>[0-9]+)", value)
    if match is None:
        raise ValueError(f"invalid source image reference: {value!r}")
    return int(match.group("step"))


def inject_plan(controller: str, plan_text: str) -> str:
    if not plan_text:
        return controller
    marker = "[THINKING] "
    if controller.count(marker) != 1:
        raise ValueError("controller thinking marker is not unique")
    return controller.replace(marker, f"{marker}{plan_text}\n", 1)


def canonical_response(
    parsed,
    *,
    task: str,
    round_index: int,
    source_image: str,
    plan_text: str = "",
    payload_override: str | None = None,
) -> str:
    controller = inject_plan(parsed.controller, plan_text)
    payload = (
        str(payload_override or "").strip()
        if parsed.action == "edit"
        else "None"
    )
    if parsed.action == "edit" and not payload:
        raise ValueError("canonical edit response has empty full payload")
    response = f"{controller}\n[EDIT] {payload}"
    validated = parse_relocated_response(
        response,
        task=task,
        expected_round_index=round_index,
        expected_source_image=source_image,
    )
    if validated.action != parsed.action:
        raise ValueError("canonical response action changed")
    return response


def load_meta_steps(row: dict, metadata: dict) -> tuple[list[dict], dict]:
    meta_path = Path(str(row["meta_path"]))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    steps = list(meta.get("steps") or [])
    if len(steps) != len(metadata["think_list"]):
        raise ValueError(f"{row['uid']}: meta/parquet step count mismatch")
    return steps, {
        "meta_path": str(meta_path),
        "meta_size_bytes": meta_path.stat().st_size,
    }


def planned_instruction_text(
    row: dict,
    metadata: dict,
    steps: list[dict],
) -> tuple[str, dict]:
    if str(row["trajectory_subtype"]) != "planned_progression":
        return "", {
            "planned_steps": 0,
            "corrective_repairs": 0,
            "meta_path": str(row["meta_path"]),
        }

    planned = []
    corrective_repairs = 0
    for step_index in metadata["generated_steps"]:
        step = steps[step_index]
        transition_kind = str(step.get("transition_kind") or "")
        instruction = str(step.get("edit_instruction") or "").strip()
        if not instruction:
            raise ValueError(f"{row['uid']}:{step_index}: empty meta payload")
        if transition_kind in PLANNED_TRANSITION_KINDS:
            milestone_index = int(step.get("milestone_index") or 0)
            if milestone_index <= 0:
                raise ValueError(
                    f"{row['uid']}:{step_index}: planned milestone missing index"
                )
            planned.append((milestone_index, instruction))
        elif transition_kind == "corrective_repair":
            corrective_repairs += 1
        else:
            raise ValueError(
                f"{row['uid']}:{step_index}: unknown transition kind "
                f"{transition_kind!r}"
            )
    indexes = [index for index, _ in planned]
    if len(planned) < 2 or indexes != list(range(1, len(planned) + 1)):
        raise ValueError(f"{row['uid']}: non-contiguous accepted plan {indexes}")
    text = "Milestone plan: " + " | ".join(
        f"m{index}: {instruction}" for index, instruction in planned
    )
    return text, {
        "planned_steps": len(planned),
        "corrective_repairs": corrective_repairs,
        "meta_path": str(row["meta_path"]),
    }


def transformed_system_prompt(row: dict) -> str:
    if str(row.get("system_prompt_version") or "") != SOURCE_SYSTEM_PROMPT_VERSION:
        raise ValueError(f"{row['uid']}: source system prompt version differs")
    value = Clean29529ExternalEditOnlyIterableDataset.rewrite_system_prompt(
        str(row["system_prompt"])
    )
    if PLAN_SYSTEM_SUFFIX not in value:
        value = f"{value}\n{PLAN_SYSTEM_SUFFIX}"
    return value


def build_rows(source_root: Path, cache_steps: dict[str, set[int]]):
    controllers = []
    transitions = []
    verifiers = []
    counts = Counter()
    subtypes = Counter()
    plan_counts = Counter()
    payload_defects = Counter()
    seen_uids = set()
    plan_records = []

    for _, row in iter_source_rows(source_root):
        uid = str(row["uid"])
        task = str(row["task"]).lower()
        if uid in seen_uids:
            raise ValueError(f"duplicate source UID: {uid}")
        seen_uids.add(uid)
        if task not in {"t2i", "edit"}:
            raise ValueError(f"{uid}: unsupported task {task!r}")

        image_presence = [
            bool(int(value or 0)) for value in row["step_image_size_bytes"]
        ]
        observed_cache_steps = cache_steps.get(uid)
        expected_cache_steps = {
            index for index, present in enumerate(image_presence) if present
        }
        if observed_cache_steps != expected_cache_steps:
            raise ValueError(f"{uid}: cache/source image steps differ")
        metadata = validate_enriched_metadata(row, image_presence)
        system_prompt = transformed_system_prompt(row)
        meta_steps, _ = load_meta_steps(row, metadata)
        plan_text, plan_record = planned_instruction_text(
            row,
            metadata,
            meta_steps,
        )
        if plan_text:
            plan_counts["rows_with_plan"] += 1
            plan_counts[
                f"planned_steps_{plan_record['planned_steps']}"
            ] += 1
            plan_counts["corrective_repairs_excluded"] += int(
                plan_record["corrective_repairs"]
            )
            plan_records.append(f"{uid}\t{plan_text}")

        start_index = int(metadata["start_index"])
        response_steps = list(range(start_index, len(metadata["think_list"])))
        responses_by_step = {}
        full_payloads_by_step = {}
        for response_position, step_index in enumerate(response_steps):
            parsed = metadata["parsed_controllers"][step_index]
            payload_override = None
            if parsed.action == "edit":
                meta_step = meta_steps[step_index]
                if str(meta_step.get("action") or "").lower() != "edit":
                    raise ValueError(
                        f"{uid}:{step_index}: meta action is not edit"
                    )
                payload_override = str(
                    meta_step.get("edit_instruction") or ""
                ).strip()
                if not payload_override:
                    raise ValueError(
                        f"{uid}:{step_index}: meta payload is empty"
                    )
                rendered_payload = parsed.payload
                if rendered_payload == payload_override:
                    payload_defects["exact"] += 1
                elif (
                    payload_override.startswith(rendered_payload)
                    and len(rendered_payload) <= 600
                ):
                    payload_defects["render_truncated"] += 1
                elif " ".join(rendered_payload.split()) == " ".join(
                    payload_override.split()
                ):
                    payload_defects["whitespace_normalized"] += 1
                else:
                    payload_defects["other_mismatch"] += 1
                full_payloads_by_step[step_index] = payload_override
            response = canonical_response(
                parsed,
                task=task,
                round_index=metadata["rounds"][step_index],
                source_image=metadata["source_refs"][step_index],
                plan_text=plan_text if response_position == 0 else "",
                payload_override=payload_override,
            )
            responses_by_step[step_index] = response

        controllers.append(
            {
                "uid": uid,
                "task": task,
                "trajectory_subtype": str(row["trajectory_subtype"]),
                "user_prompt": str(row["user_prompt"]),
                "system_prompt": system_prompt,
                "response_list": [
                    responses_by_step[index] for index in response_steps
                ],
                "response_action_list": [
                    metadata["recorded_actions"][index]
                    for index in response_steps
                ],
                "response_round_list": [
                    metadata["rounds"][index] for index in response_steps
                ],
                "response_source_image_list": [
                    metadata["source_refs"][index] for index in response_steps
                ],
                "response_image_step_list": [
                    index
                    if metadata["recorded_actions"][index] == "edit"
                    else -1
                    for index in response_steps
                ],
                "source_image_step_index": int(
                    row["source_image_step_index"]
                ),
                "milestone_plan_text": plan_text,
            }
        )

        generated_steps = list(metadata["generated_steps"])
        for generated_position, target_step in enumerate(generated_steps):
            parsed = metadata["parsed_controllers"][target_step]
            full_payload = full_payloads_by_step[target_step]
            source_step = source_step_from_reference(
                metadata["source_refs"][target_step],
                task=task,
            )
            if generated_position == 0:
                expected_source = -1 if task == "t2i" else 0
            else:
                expected_source = generated_steps[generated_position - 1]
            if source_step != expected_source:
                raise ValueError(
                    f"{uid}:{target_step}: transition source "
                    f"{source_step}!={expected_source}"
                )
            if task == "t2i" and source_step == -1:
                pair_type = "t2i_initial"
            elif task == "t2i":
                pair_type = "t2i_transition"
            else:
                pair_type = "edit_transition"
            pair_uid = f"{pair_type}::{uid}::{target_step:04d}"
            transitions.append(
                {
                    "pair_uid": pair_uid,
                    "uid": uid,
                    "task": task,
                    "pair_type": pair_type,
                    "payload": full_payload,
                    "source_step_index": source_step,
                    "target_step_index": target_step,
                    "target_image_sha256": metadata["image_sha256"][
                        target_step
                    ],
                }
            )
            counts[f"mse_{pair_type}"] += 1

            target_response_step = target_step + 1
            if target_response_step not in responses_by_step:
                raise ValueError(
                    f"{uid}:{target_step}: missing next verifier response"
                )
            target_action = metadata["recorded_actions"][target_response_step]
            expected_target_action = (
                "done"
                if target_step == generated_steps[-1]
                else "edit"
            )
            if target_action != expected_target_action:
                raise ValueError(
                    f"{uid}:{target_step}: verifier action "
                    f"{target_action}!={expected_target_action}"
                )
            prefix_steps = [
                index
                for index in response_steps
                if index <= target_step
            ]
            if any(
                metadata["recorded_actions"][index] != "edit"
                for index in prefix_steps
            ):
                raise ValueError(f"{uid}:{target_step}: done inside prefix")
            is_penultimate = (
                target_action == "edit"
                and target_response_step == generated_steps[-1]
            )
            verifiers.append(
                {
                    "state_uid": (
                        f"verifier::{uid}::{target_step:04d}::"
                        f"{target_action}"
                    ),
                    "uid": uid,
                    "task": task,
                    "trajectory_subtype": str(row["trajectory_subtype"]),
                    "user_prompt": str(row["user_prompt"]),
                    "system_prompt": system_prompt,
                    "prefix_response_list": [
                        responses_by_step[index] for index in prefix_steps
                    ],
                    "prefix_round_list": [
                        metadata["rounds"][index] for index in prefix_steps
                    ],
                    "prefix_source_image_list": [
                        metadata["source_refs"][index]
                        for index in prefix_steps
                    ],
                    "prefix_image_step_list": prefix_steps,
                    "target_response": responses_by_step[target_response_step],
                    "target_action": target_action,
                    "target_round_index": metadata["rounds"][
                        target_response_step
                    ],
                    "target_source_image": metadata["source_refs"][
                        target_response_step
                    ],
                    "current_step_index": target_step,
                    "is_penultimate": is_penultimate,
                }
            )
            counts[f"verifier_{target_action}"] += 1
            counts["verifier_penultimate"] += int(is_penultimate)

        counts["rows"] += 1
        counts[f"rows_{task}"] += 1
        counts["generated_images"] += len(generated_steps)
        counts["intermediate_images"] += len(generated_steps) - 1
        subtypes[(task, str(row["trajectory_subtype"]))] += 1

    controllers.sort(key=lambda item: item["uid"])
    transitions.sort(key=lambda item: item["pair_uid"])
    verifiers.sort(key=lambda item: item["state_uid"])
    expected = {
        "rows": 29_529,
        "rows_t2i": 15_000,
        "rows_edit": 14_529,
        "generated_images": 58_020,
        "intermediate_images": 28_491,
        "mse_t2i_initial": 15_000,
        "mse_t2i_transition": 16_242,
        "mse_edit_transition": 26_778,
        "verifier_edit": 28_491,
        "verifier_done": 29_529,
        "verifier_penultimate": 20_529,
    }
    for key, value in expected.items():
        if counts[key] != value:
            raise ValueError(f"count {key}={counts[key]} expected {value}")
    if not (
        len(controllers) == 29_529
        and len(transitions) == 58_020
        and len(verifiers) == 58_020
    ):
        raise ValueError("materialized row counts differ")
    expected_payload_defects = {
        "exact": 56_859,
        "render_truncated": 1_155,
        "whitespace_normalized": 6,
        "other_mismatch": 0,
    }
    if {
        key: payload_defects[key] for key in expected_payload_defects
    } != expected_payload_defects:
        raise ValueError(
            "source/meta payload defect counts differ: "
            f"{dict(payload_defects)}"
        )

    audit = {
        "counts": dict(sorted(counts.items())),
        "task_subtypes": {
            f"{task}:{subtype}": count
            for (task, subtype), count in sorted(subtypes.items())
        },
        "plans": {
            **dict(sorted(plan_counts.items())),
            "plan_text_sha256": sha256_lines(plan_records),
        },
        "payload_repair": {
            "exact": payload_defects["exact"],
            "render_truncated": payload_defects["render_truncated"],
            "whitespace_normalized": payload_defects[
                "whitespace_normalized"
            ],
            "other_mismatch": payload_defects["other_mismatch"],
            "authoritative_source": "meta.steps[].edit_instruction",
            "used_for_external_edit_ce": True,
            "used_for_transition_mse": True,
        },
        "controller_uid_sha256": sha256_lines(
            item["uid"] for item in controllers
        ),
        "transition_uid_sha256": sha256_lines(
            item["pair_uid"] for item in transitions
        ),
        "verifier_uid_sha256": sha256_lines(
            item["state_uid"] for item in verifiers
        ),
    }
    return controllers, transitions, verifiers, audit


def write_parquet_rows(
    rows: list[dict],
    output_root: Path,
    *,
    num_shards: int,
) -> dict:
    if output_root.exists():
        raise FileExistsError(f"refusing existing parquet root: {output_root}")
    temporary = output_root.with_name(
        f".{output_root.name}.tmp.{os.getpid()}"
    )
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    shards = [[] for _ in range(num_shards)]
    for index, row in enumerate(rows):
        shards[index % num_shards].append(row)
    if any(not shard for shard in shards):
        raise ValueError(f"{output_root}: an output shard would be empty")

    parquet_info = {}
    inventory = []
    for shard_index, shard_rows in enumerate(shards):
        temporary_path = temporary / f"shard-{shard_index:05d}.parquet"
        pq.write_table(
            pa.Table.from_pylist(shard_rows),
            temporary_path,
            compression="zstd",
            row_group_size=256,
        )
        final_path = output_root / temporary_path.name
        parquet_file = pq.ParquetFile(temporary_path)
        parquet_info[str(final_path.absolute())] = {
            "num_row_groups": parquet_file.num_row_groups,
            "num_rows": parquet_file.metadata.num_rows,
        }
        inventory.append(
            {
                "path": str(final_path.absolute()),
                "rows": parquet_file.metadata.num_rows,
                "row_groups": parquet_file.num_row_groups,
                "size_bytes": temporary_path.stat().st_size,
                "sha256": sha256_file(temporary_path),
            }
        )
    atomic_json(temporary / "parquet_info.json", parquet_info)
    os.replace(temporary, output_root)
    return {
        "root": str(output_root),
        "rows": len(rows),
        "num_shards": num_shards,
        "parquet_info": str(output_root / "parquet_info.json"),
        "parquet_info_sha256": sha256_file(
            output_root / "parquet_info.json"
        ),
        "shards": inventory,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=SFT_ROOT / "trajectory_parquet",
        help="trajectory parquet shards (with parquet_info.json)",
    )
    parser.add_argument(
        "--cache-root", type=Path, default=SFT_ROOT / "pixel_cache",
        help="pixel cache written by precompute_failaware_cache.py --stage pixels",
    )
    parser.add_argument(
        "--anchor-allowlist", type=Path,
        default=SFT_ROOT / "anchor" / "base_anchor_allowlist.json",
        help="the 1,265 base-BAGEL anchor rows (no benchmark prompt overlap)",
    )
    parser.add_argument(
        "--base-model-dir", type=Path,
        default=Path(os.environ.get("BAGEL_BASE_DIR", ROOT / "pretrained" / "BAGEL-7B-MoT")),
        help="official BAGEL-7B-MoT snapshot; its ema.safetensors is hash-checked",
    )
    parser.add_argument("--output-root", type=Path, default=SFT_ROOT / "rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(
            f"refusing existing output root: {args.output_root}"
        )
    args.output_root.mkdir(parents=True)

    cache_steps, cache_record = load_cache_steps(args.cache_root)
    controllers, transitions, verifiers, audit = build_rows(
        args.source_root, cache_steps
    )
    controller_record = write_parquet_rows(
        controllers,
        args.output_root / "controller_rows_100",
        num_shards=100,
    )
    transition_record = write_parquet_rows(
        transitions,
        args.output_root / "transition_rows_96",
        num_shards=96,
    )
    verifier_record = write_parquet_rows(
        verifiers,
        args.output_root / "verifier_rows_96",
        num_shards=96,
    )

    anchor_target = args.output_root / "base_anchor_allowlist.json"
    shutil.copyfile(args.anchor_allowlist, anchor_target)
    anchor = json.loads(anchor_target.read_text(encoding="utf-8"))
    if (
        anchor.get("pass") is not True
        or len(anchor.get("rows") or []) != 1_265
        or anchor.get("strict_benchmark_prompt_overlap") != 0
    ):
        raise ValueError("preserved Base anchor allowlist differs")

    base_ema = args.base_model_dir / "ema.safetensors"
    observed_base_sha256 = sha256_file(base_ema)
    if observed_base_sha256 != BASE_EMA_SHA256:
        raise RuntimeError(f"Base EMA SHA differs: {observed_base_sha256}")

    source_info = args.source_root / "parquet_info.json"
    record = {
        "version": VERSION,
        "pass": True,
        "generated_at_utc": utc_now(),
        "source": {
            "root": str(args.source_root),
            "parquet_info": str(source_info),
            "parquet_info_sha256": sha256_file(source_info),
            "cache": cache_record,
            "source_system_prompt_version": SOURCE_SYSTEM_PROMPT_VERSION,
        },
        "base_ema": {
            "snapshot": str(args.base_model_dir),
            "path": str(base_ema),
            "size_bytes": base_ema.stat().st_size,
            "sha256": observed_base_sha256,
        },
        "audit": audit,
        "controller_parquet": controller_record,
        "transition_parquet": transition_record,
        "verifier_parquet": verifier_record,
        "base_anchor_allowlist": {
            "path": str(anchor_target),
            "sha256": sha256_file(anchor_target),
            "rows": len(anchor["rows"]),
            "strict_benchmark_prompt_overlap": 0,
        },
        "contracts": {
            "unique_controller_rows": 29_529,
            "unique_mse_transitions": 58_020,
            "unique_verifier_states": 58_020,
            "verifier_edit_targets": 28_491,
            "verifier_done_targets": 29_529,
            "penultimate_oversampling_view_rows": 20_529,
            "mse_controller_or_thinking_tokens": 0,
            "new_image_generation": 0,
            "legacy_reader_modifications": 0,
            "system_prompt_version": SYSTEM_PROMPT_VERSION,
        },
    }
    atomic_json(args.output_root / "data_preparation.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
