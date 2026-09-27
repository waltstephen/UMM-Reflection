"""Fail-closed stage and capability gates for V20 launch supervision."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Sequence

import torch


VERSION = "clean29529_v20_release_gate_v3"
ROLLING_DONE_VERSION = "clean29529_v20_rolling_done_guard_v1"
GROUP_SIZE = 28
# Step-level bound, unchanged. This is the loose V19 bound and it provably
# cannot detect stale context: V19 step 199 recorded 1.089 and passed.
TEXT_RATIO_LIMIT = 1.5
# Defect 1: text runs before flow with exactly one update per logical group,
# so on the FIRST inner epoch the replay ratio must be exactly 1 up to
# floating point. Anything above 1e-3 means the replay context differs from
# the sampling context, which is the stale-context defect.
FIRST_INNER_EPOCH_TEXT_RATIO_LIMIT = 1e-3
FLOW_KL_LIMIT = 0.5
# G004: this band is report-only per step and hard-stop over trailing ten.
DONE_RATE_BAND = (0.05, 0.95)
ROLLING_DONE_WINDOW = 10
ZERO_DONE_CONSECUTIVE_LIMIT = 3
CONTROLLER_INVALID_RATE_LIMIT = 0.10
MAX_ABS_ADVANTAGE_LIMIT = 5.0
REQUIRED_REPAIR_ROUND_COUNT = 2


def build_optimizer_step_accounting(
    all_rank_update_metrics: Sequence[dict[str, Any]],
    *,
    expected_world_size: int,
    commit_optimizer_step: bool,
) -> dict[str, Any]:
    updates = list(all_rank_update_metrics)
    errors = []
    if len(updates) != int(expected_world_size):
        errors.append("rank_update_coverage")
    if any(
        update.get("text_before_flow") is not True for update in updates
    ):
        errors.append("text_before_flow")
    if any(
        "commit_optimizer_step" not in update
        or bool(update["commit_optimizer_step"])
        is not bool(commit_optimizer_step)
        for update in updates
    ):
        errors.append("commit_optimizer_step_mismatch")
    channels = {}
    for channel in ("text", "flow"):
        key = f"{channel}_step"
        rows = [
            update.get(key)
            for update in updates
            if isinstance(update.get(key), dict)
        ]
        if len(rows) != len(updates):
            errors.append(f"{channel}_step_result_coverage")
        globally_active_values = {
            bool(row.get("globally_active")) for row in rows
        }
        globally_active = (
            next(iter(globally_active_values))
            if len(globally_active_values) == 1
            else None
        )
        if len(globally_active_values) != 1:
            errors.append(f"{channel}_global_activity_mismatch")
        expected_step = bool(commit_optimizer_step and globally_active)
        rank_step_calls = sum(
            int(row.get("optimizer_step_call_count", -1))
            for row in rows
        )
        stepped_values = [bool(row.get("stepped")) for row in rows]
        if any(
            int(row.get("optimizer_step_call_count", -1))
            != int(expected_step)
            for row in rows
        ):
            errors.append(f"{channel}_rank_step_call_count")
        if any(value is not expected_step for value in stepped_values):
            errors.append(f"{channel}_stepped_state")
        channels[channel] = {
            "globally_active": globally_active,
            "commit_optimizer_step": bool(commit_optimizer_step),
            "logical_optimizer_step_count": int(expected_step),
            "expected_logical_optimizer_step_count": int(expected_step),
            "rank_optimizer_step_call_count": rank_step_calls,
            "expected_rank_optimizer_step_call_count": (
                int(expected_world_size) * int(expected_step)
            ),
            "all_rank_results_present": len(rows) == len(updates),
        }
    return {
        "version": "clean29529_v20_optimizer_step_accounting_v1",
        "expected_world_size": int(expected_world_size),
        "rank_update_count": len(updates),
        "commit_optimizer_step": bool(commit_optimizer_step),
        "text_before_flow": not any(
            update.get("text_before_flow") is not True
            for update in updates
        ),
        "channels": channels,
        "passed": not errors,
        "errors": errors,
    }


def nonfinite_numeric_paths(
    value: Any,
    *,
    prefix: str = "root",
) -> list[str]:
    output = []
    if torch.is_tensor(value):
        detached = value.detach().float().reshape(-1).cpu()
        output.extend(
            f"{prefix}[{index}]"
            for index, item in enumerate(detached.tolist())
            if not math.isfinite(float(item))
        )
    elif isinstance(value, dict):
        for key, item in value.items():
            output.extend(
                nonfinite_numeric_paths(
                    item,
                    prefix=f"{prefix}.{key}",
                )
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            output.extend(
                nonfinite_numeric_paths(
                    item,
                    prefix=f"{prefix}[{index}]",
                )
            )
    elif isinstance(value, Real) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            output.append(prefix)
    return output


def _done_evidence(
    row: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Return exact semantic/literal/alias DONE evidence or fail closed."""
    source = row.get("done_calibration")
    errors: list[str] = []
    if not isinstance(source, dict):
        return {}, ["done_evidence_missing"]

    count_names = (
        "done_count",
        "literal_sampled_done_count",
        "edit_none_alias_done_count",
        "strict_pass_done_count",
        "false_done_count",
    )
    rate_names = (
        "done_rate",
        "literal_sampled_done_rate",
        "edit_none_alias_done_rate",
        "false_done_rate",
    )
    counts: dict[str, int] = {}
    rates: dict[str, float] = {}
    for name in count_names:
        value = source.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or int(value) != float(value)
            or int(value) < 0
            or int(value) > GROUP_SIZE
        ):
            errors.append(f"{name}_malformed")
        else:
            counts[name] = int(value)
    for name in rate_names:
        value = source.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or float(value) > 1.0
        ):
            errors.append(f"{name}_malformed")
        else:
            rates[name] = float(value)
    if errors:
        return {}, errors

    expected_rates = {
        "done_rate": counts["done_count"] / GROUP_SIZE,
        "literal_sampled_done_rate": (
            counts["literal_sampled_done_count"] / GROUP_SIZE
        ),
        "edit_none_alias_done_rate": (
            counts["edit_none_alias_done_count"] / GROUP_SIZE
        ),
        "false_done_rate": (
            counts["false_done_count"] / counts["done_count"]
            if counts["done_count"]
            else 0.0
        ),
    }
    for name, expected in expected_rates.items():
        if not math.isclose(rates[name], expected, abs_tol=1e-12):
            errors.append(f"{name}_inconsistent")
    if counts["done_count"] != (
        counts["literal_sampled_done_count"]
        + counts["edit_none_alias_done_count"]
    ):
        errors.append("semantic_done_partition_inconsistent")
    if counts["done_count"] != (
        counts["strict_pass_done_count"] + counts["false_done_count"]
    ):
        errors.append("done_outcome_partition_inconsistent")
    if errors:
        return {}, errors
    return {
        **counts,
        **rates,
        "per_step_in_band": (
            DONE_RATE_BAND[0]
            <= rates["done_rate"]
            <= DONE_RATE_BAND[1]
        ),
    }, []


def validate_rolling_done_metrics(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Apply G004 rolling protection to the latest logical step."""
    ordered = list(rows)
    errors: list[str] = []
    steps = [int(row.get("step", -1)) for row in ordered]
    if (
        not ordered
        or len(steps) != len(set(steps))
        or steps != sorted(steps)
        or any(b != a + 1 for a, b in zip(steps, steps[1:]))
    ):
        errors.append("done_history_step_order")
    trailing = ordered[-ROLLING_DONE_WINDOW:]
    evidence_rows = []
    for row in trailing:
        evidence, row_errors = _done_evidence(row)
        if row_errors:
            errors.extend(
                f"step_{int(row.get('step', -1))}:{error}"
                for error in row_errors
            )
        else:
            evidence_rows.append(evidence)
    semantic_count = sum(
        value["done_count"] for value in evidence_rows
    )
    literal_count = sum(
        value["literal_sampled_done_count"] for value in evidence_rows
    )
    window_ready = len(trailing) == ROLLING_DONE_WINDOW
    semantic_rate = (
        semantic_count / (ROLLING_DONE_WINDOW * GROUP_SIZE)
        if window_ready and len(evidence_rows) == ROLLING_DONE_WINDOW
        else None
    )
    if (
        semantic_rate is not None
        and not DONE_RATE_BAND[0] <= semantic_rate <= DONE_RATE_BAND[1]
    ):
        errors.append("rolling_semantic_done_rate_outside_band")
    if (
        window_ready
        and len(evidence_rows) == ROLLING_DONE_WINDOW
        and literal_count == 0
    ):
        errors.append("rolling_literal_done_extinction")
    consecutive_zero = 0
    if len(ordered) >= ZERO_DONE_CONSECUTIVE_LIMIT:
        last_evidence = []
        for row in ordered[-ZERO_DONE_CONSECUTIVE_LIMIT:]:
            evidence, row_errors = _done_evidence(row)
            if row_errors:
                break
            last_evidence.append(evidence)
        if len(last_evidence) == ZERO_DONE_CONSECUTIVE_LIMIT:
            consecutive_zero = sum(
                value["done_count"] == 0 for value in last_evidence
            )
            if consecutive_zero == ZERO_DONE_CONSECUTIVE_LIMIT:
                errors.append("three_consecutive_zero_done_steps")
    return {
        "version": ROLLING_DONE_VERSION,
        "passed": not errors,
        "errors": sorted(set(errors)),
        "current_step": steps[-1] if steps else None,
        "window_steps": [int(row.get("step", -1)) for row in trailing],
        "window_ready": window_ready,
        "semantic_done_count": semantic_count,
        "semantic_done_denominator": len(evidence_rows) * GROUP_SIZE,
        "semantic_done_rate": semantic_rate,
        "literal_sampled_done_count": literal_count,
        "three_step_zero_done_count": consecutive_zero,
        "limits": {
            "trailing_window": ROLLING_DONE_WINDOW,
            "semantic_done_rate_band": list(DONE_RATE_BAND),
            "zero_done_consecutive_limit": ZERO_DONE_CONSECUTIVE_LIMIT,
            "literal_done_minimum_per_full_window": 1,
        },
    }


def validate_step_metrics(
    row: dict[str, Any],
    *,
    expected_step: int,
) -> dict[str, Any]:
    errors = []
    safety = row.get("numerical_safety") or {}
    dashboard = row.get("self_improvement") or {}
    reward_pipeline = row.get("reward_pipeline") or {}
    judge_response = row.get("judge_response") or {}
    detector_transport = row.get("detector_transport") or {}
    step_accounting = row.get("optimizer_step_accounting") or {}
    if int(row.get("step", -1)) != int(expected_step):
        errors.append("logical_step")
    if int(safety.get("nonfinite_metric_count", -1)) != 0:
        errors.append("nonfinite_metric_count")
    ratio = float(
        safety.get("text_ratio_max_abs_deviation", float("inf"))
    )
    flow_kl = float(safety.get("flow_kl_max", float("inf")))
    if not math.isfinite(ratio):
        errors.append("text_ratio_nonfinite")
    elif ratio > TEXT_RATIO_LIMIT:
        errors.append("text_ratio_max_abs_deviation")
    # Defect 1: the first inner epoch must replay the sampling context
    # exactly. Missing evidence fails closed.
    first_epoch = safety.get("text_ratio_first_inner_epoch_max_abs_deviation")
    if first_epoch is None:
        errors.append("text_ratio_first_inner_epoch_missing")
    else:
        first_epoch_value = float(first_epoch)
        if not math.isfinite(first_epoch_value):
            errors.append("text_ratio_first_inner_epoch_nonfinite")
        elif first_epoch_value > FIRST_INNER_EPOCH_TEXT_RATIO_LIMIT:
            errors.append("text_ratio_first_inner_epoch")
    if not math.isfinite(flow_kl):
        errors.append("flow_kl_nonfinite")
    elif flow_kl > FLOW_KL_LIMIT:
        errors.append("flow_kl_max")
    if nonfinite_numeric_paths(row, prefix="metrics"):
        errors.append("row_contains_nonfinite_numeric_value")
    if int(row.get("updated_sample_count", -1)) != GROUP_SIZE:
        errors.append("updated_sample_count")
    if int(dashboard.get("trajectory_count", -1)) != GROUP_SIZE:
        errors.append("trajectory_count")
    if int(dashboard.get("trainable_cap_action_count", -1)) != 0:
        errors.append("trainable_cap_action_count")
    if int(
        dashboard.get("detector_failed_positive_done_count", -1)
    ) != 0:
        errors.append("detector_failed_positive_done_count")
    # G004 changes only the noisy per-step DONE band to report-only. Exact
    # semantic/literal/alias evidence remains mandatory and rolling failures
    # are added by validate_online_release_metrics below.
    done_evidence, done_errors = _done_evidence(row)
    errors.extend(done_errors)
    protocol = row.get("controller_protocol") or {}
    invalid_rate = row.get(
        "controller_invalid_rate",
        protocol.get("controller_invalid_rate"),
    )
    if invalid_rate is None or not math.isfinite(float(invalid_rate)):
        errors.append("controller_invalid_rate_missing")
    elif float(invalid_rate) > CONTROLLER_INVALID_RATE_LIMIT:
        errors.append("controller_invalid_rate")
    repair_active = row.get("repair_flow_channel_active")
    if (
        not isinstance(repair_active, list)
        or len(repair_active) != REQUIRED_REPAIR_ROUND_COUNT
    ):
        errors.append("repair_flow_channel_active_missing")
    elif not all(bool(value) for value in repair_active):
        errors.append("repair_flow_channel_inactive")
    max_abs = row.get("channel_advantage_max_abs")
    if not isinstance(max_abs, dict) or not max_abs:
        errors.append("channel_advantage_max_abs_missing")
    else:
        observed = []
        for value in max_abs.values():
            if isinstance(value, list):
                observed.extend(
                    float(entry) for entry in value if entry is not None
                )
            elif value is not None:
                observed.append(float(value))
        if not observed or any(
            not math.isfinite(value) for value in observed
        ):
            errors.append("channel_advantage_max_abs_nonfinite")
        elif max(observed) > MAX_ABS_ADVANTAGE_LIMIT + 1e-9:
            errors.append("max_abs_advantage")
    if int(row.get("judge_available_trajectory_count", -1)) != GROUP_SIZE:
        errors.append("judge_coverage")
    if judge_response.get("ok") is not True:
        errors.append("judge_response")
    if row.get("judge_channel_mode") != "reward":
        errors.append("judge_channel_mode")
    detector_image_count = int(detector_transport.get("image_count", -1))
    detector_chunk_sizes = [
        int(value)
        for value in detector_transport.get("chunk_sizes") or []
    ]
    if (
        detector_transport.get("version")
        != "clean29529_v20_geneval_group_coverage_v1"
        or detector_transport.get("service_version")
        != "clean29529_v20_flowgrpo_geneval_service_v1"
        or detector_transport.get("protocol_version")
        != "flow_grpo_geneval_pickle_18085_v1"
        or int(detector_transport.get("trajectory_count", -1))
        != GROUP_SIZE
        or detector_image_count < GROUP_SIZE
        or detector_image_count > 3 * GROUP_SIZE
        or int(detector_transport.get("score_count", -1))
        != detector_image_count
        or int(detector_transport.get("strict_count", -1))
        != detector_image_count
        or sum(detector_chunk_sizes) != detector_image_count
        or int(detector_transport.get("chunk_count", -1))
        != len(detector_chunk_sizes)
        or int(detector_transport.get("total_attempts", -1))
        < len(detector_chunk_sizes)
        or int(detector_transport.get("total_attempts", -1))
        > 3 * len(detector_chunk_sizes)
    ):
        errors.append("detector_coverage")
    if reward_pipeline.get("fail_closed") is not True:
        errors.append("reward_fail_closed")
    if int(reward_pipeline.get("queue_depth", -1)) != 0:
        errors.append("reward_queue_depth")
    if reward_pipeline.get("next_rollout_overlap") is not False:
        errors.append("next_rollout_overlap")
    if int(row.get("rollout_policy_lag_updates", -1)) != 0:
        errors.append("rollout_policy_lag_updates")
    if int(row.get("num_timesteps", -1)) != 50:
        errors.append("num_timesteps")
    optimizer_precision = row.get("optimizer_state_precision") or {}
    if optimizer_precision.get("all_ranks_passed") is not True:
        errors.append("optimizer_state_precision")
    if (
        step_accounting.get("version")
        != "clean29529_v20_optimizer_step_accounting_v1"
        or step_accounting.get("passed") is not True
        or int(step_accounting.get("expected_world_size", -1)) != 14
        or int(step_accounting.get("rank_update_count", -1)) != 14
        or step_accounting.get("commit_optimizer_step") is not True
    ):
        errors.append("optimizer_step_accounting")
    else:
        for channel in ("text", "flow"):
            accounting = (
                step_accounting.get("channels", {}).get(channel) or {}
            )
            globally_active = accounting.get("globally_active")
            expected_steps = int(bool(globally_active))
            if (
                accounting.get("commit_optimizer_step") is not True
                or int(
                    accounting.get(
                        "logical_optimizer_step_count",
                        -1,
                    )
                )
                != expected_steps
                or int(
                    accounting.get(
                        "expected_logical_optimizer_step_count",
                        -1,
                    )
                )
                != expected_steps
                or int(
                    accounting.get(
                        "rank_optimizer_step_call_count",
                        -1,
                    )
                )
                != 14 * expected_steps
                or int(
                    accounting.get(
                        "expected_rank_optimizer_step_call_count",
                        -1,
                    )
                )
                != 14 * expected_steps
                or accounting.get("all_rank_results_present") is not True
            ):
                errors.append(f"{channel}_optimizer_step_accounting")
    return {
        "version": VERSION,
        "step": int(expected_step),
        "passed": not errors,
        "errors": errors,
        "per_step_done_rate_report_only": {
            "visible": bool(done_evidence),
            "hard_stop": False,
            "done_count": done_evidence.get("done_count"),
            "done_rate": done_evidence.get("done_rate"),
            "in_band": done_evidence.get("per_step_in_band"),
            "band": list(DONE_RATE_BAND),
        },
        "limits": {
            "text_ratio_max_abs_deviation": TEXT_RATIO_LIMIT,
            "flow_kl_max": FLOW_KL_LIMIT,
            "group_size": GROUP_SIZE,
        },
    }


def validate_online_release_metrics(
    row: dict[str, Any],
    *,
    expected_step: int,
    done_history_rows: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    step_report = validate_step_metrics(row, expected_step=expected_step)
    rolling = validate_rolling_done_metrics([*done_history_rows, row])
    errors = list(step_report["errors"]) + list(rolling["errors"])
    return {
        "version": VERSION,
        "step": int(expected_step),
        "passed": not errors,
        "errors": sorted(set(errors)),
        "limits": step_report["limits"],
        "per_step_done_rate_report_only": step_report[
            "per_step_done_rate_report_only"
        ],
        "rolling_done_guard": rolling,
        "non_done_step_gate": step_report,
    }


def validate_stage_metrics(
    rows: Sequence[dict[str, Any]],
    *,
    start_step: int,
    end_step: int,
    done_history_rows: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    expected = list(range(int(start_step) + 1, int(end_step) + 1))
    by_step = {}
    duplicate_steps = []
    for row in rows:
        step = int(row.get("step", -1))
        if step in by_step:
            duplicate_steps.append(step)
        by_step[step] = row
    missing = [step for step in expected if step not in by_step]
    unexpected = sorted(set(by_step) - set(expected))
    reports = []
    history = list(done_history_rows)
    for step in expected:
        if step not in by_step:
            continue
        reports.append(
            validate_online_release_metrics(
                by_step[step],
                expected_step=step,
                done_history_rows=history,
            )
        )
        history.append(by_step[step])
    errors = []
    if duplicate_steps:
        errors.append("duplicate_steps")
    if missing:
        errors.append("missing_steps")
    if unexpected:
        errors.append("unexpected_steps")
    if any(not report["passed"] for report in reports):
        errors.append("step_gate_failure")
    return {
        "version": VERSION,
        "start_step": int(start_step),
        "end_step": int(end_step),
        "passed": not errors,
        "errors": errors,
        "missing_steps": missing,
        "unexpected_steps": unexpected,
        "duplicate_steps": sorted(set(duplicate_steps)),
        "step_reports": reports,
    }


def validate_step25_capability_gate(
    gate: dict[str, Any],
) -> dict[str, Any]:
    errors = []
    if gate.get("passed") is not True:
        errors.append("gate_not_passed")
    if int(gate.get("checkpoint_step", -1)) != 25:
        errors.append("checkpoint_step")
    if not str(gate.get("checkpoint_manifest_sha256") or ""):
        errors.append("checkpoint_manifest_sha256")
    if gate.get("official_geneval_consumed") is not False:
        errors.append("official_geneval_consumed")
    if gate.get("full_evaluation_coverage") is not True:
        errors.append("full_evaluation_coverage")
    if gate.get("padding_state_invariance") is not True:
        errors.append("padding_state_invariance")
    if (
        gate.get("repair2_minus_repair1_not_structurally_negative")
        is not True
    ):
        errors.append("repair_advantage_structure")
    defect = gate.get("frozen_defect_probe") or {}
    if defect.get("nonregression") is not True:
        errors.append("frozen_defect_probe_nonregression")
    paired = gate.get("paired_final_minus_r0") or {}
    if float(paired.get("mean", float("-inf"))) < 0.0:
        errors.append("paired_final_minus_r0_direction")
    if int(gate.get("detector_failed_positive_done_count", -1)) != 0:
        errors.append("detector_failed_positive_done_count")
    for name in (
        "step25_training_metrics",
        "padding_evidence",
        "objective_evidence",
    ):
        binding = gate.get(name) or {}
        if (
            not str(binding.get("path") or "")
            or len(str(binding.get("sha256") or "")) != 64
        ):
            errors.append(f"{name}_binding")
    if (
        (gate.get("step25_training_metrics") or {}).get(
            "online_release_gate_passed"
        )
        is not True
    ):
        errors.append("step25_online_release_gate")
    return {
        "version": VERSION,
        "passed": not errors,
        "errors": errors,
        "checkpoint_step": 25,
    }


__all__ = [
    "FLOW_KL_LIMIT",
    "GROUP_SIZE",
    "ROLLING_DONE_VERSION",
    "ROLLING_DONE_WINDOW",
    "TEXT_RATIO_LIMIT",
    "VERSION",
    "build_optimizer_step_accounting",
    "nonfinite_numeric_paths",
    "validate_online_release_metrics",
    "validate_rolling_done_metrics",
    "validate_stage_metrics",
    "validate_step25_capability_gate",
    "validate_step_metrics",
]
