"""G011 stop-now-baselined counting process credit.

The counting score and both reward equations are imported unchanged from G009.
The hidden ``(round_index, exact_before_action)`` bucket supplies only the
population-STD scale.  A pluggable scorer supplies the center: the value of
stopping at the state before the action.  Singleton buckets remain inactive
and are never merged across round or exactness.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

import torch

from unify_rl.reward_models.g009_counting_process_reward import (
    EDIT_COST,
    EDIT_PROGRESS_COEFFICIENT,
    EPSILON,
    GROUP_SIZE,
    MAX_ABS_ADVANTAGE,
    MAX_EDIT_ROUNDS,
    MINIMUM_STD,
    TERMINAL_FAILURE,
    TERMINAL_SUCCESS,
    VERSION as G009_REWARD_VERSION,
    build_trajectory_process_credit,
    build_verification_contract,
    counting_score,
    edit_reward,
    image_state_sha256,
    terminal_reward,
    validate_cap_terminal_does_not_underprice_done,
    validate_no_positive_detector_failed_done,
    validate_reward_ordering,
)

# The reward identity intentionally remains G009's frozen two-equation reward.
VERSION = G009_REWARD_VERSION
ADVANTAGE_VERSION = "clean29529_g011_stop_now_baseline_grpo_v1"
SECRECY_VERSION = "clean29529_g011_runtime_policy_observation_secrecy_guard_v1"
STOP_NOW_SCORER_IDENTITY = "counting_strict_exact_terminal_value"
STOP_NOW_SCORER_VERSION = "clean29529_g011_counting_stop_now_value_v1"

_FORBIDDEN_POLICY_KEYS = {
    "exact_before_action",
    "detected_count",
    "detected_count_before_action",
    "target_distance",
    "counting_q",
    "q_before",
    "q_after",
    "strict_pass",
    "strict_correct",
    "boxes",
    "confidence",
    "confidence_scores",
    "threshold_margin",
    "threshold_margins",
    "reward",
    "return_to_go",
    "bucket",
    "bucket_identity",
    "group_mean",
    "group_std",
    "population_std",
    "advantage",
    "verifier_result",
}
_FORBIDDEN_POLICY_MARKERS = tuple(
    f"[{value.upper()}]"
    for value in (
        "exact_before_action",
        "detected_count",
        "target_distance",
        "counting_q",
        "strict_pass",
        "threshold_margin",
        "detector_boxes",
        "detector_confidence",
        "reward",
        "return_to_go",
        "bucket",
        "group_mean",
        "group_std",
        "advantage",
        "verifier_result",
    )
)


def _action_state_index(actions: Sequence[Any], round_index: int) -> int:
    return sum(
        str(value).strip().casefold() == "edit"
        for value in actions[: int(round_index)]
    )


def exact_before_action(
    credit: Mapping[str, Any], round_index: int
) -> bool:
    """Hidden reward-worker label for one already-complete trajectory."""

    actions = list(credit["actions"])
    index = int(round_index)
    if index < 0 or index >= len(actions):
        raise ValueError("G011 policy round is outside the trajectory")
    state_index = _action_state_index(actions, index)
    counts = list(credit["detected_counts"])
    if state_index >= len(counts):
        raise ValueError("G011 hidden detector state coverage differs")
    return int(counts[state_index]) == int(credit["target_count"])


def counting_stop_now_value(state: Mapping[str, Any]) -> float:
    """Terminal value obtained by stopping now under the counting scorer."""

    return TERMINAL_SUCCESS if bool(state["exact_before_action"]) else TERMINAL_FAILURE


# Callable metadata is persisted in config and every metric row.  A new task
# replaces this callable (and metadata), not normalization code.
counting_stop_now_value.identity = STOP_NOW_SCORER_IDENTITY  # type: ignore[attr-defined]
counting_stop_now_value.version = STOP_NOW_SCORER_VERSION  # type: ignore[attr-defined]


def _scorer_metadata(stop_now_value: Callable[[Mapping[str, Any]], float]) -> tuple[str, str]:
    identity = str(getattr(stop_now_value, "identity", "")).strip()
    version = str(getattr(stop_now_value, "version", "")).strip()
    if not identity or not version:
        raise ValueError("G011 stop_now_value callable requires identity and version")
    return identity, version


def stratified_bucket_advantages(
    returns: Sequence[Any] | torch.Tensor,
    stop_now_values: Sequence[Any] | torch.Tensor,
) -> dict[str, Any]:
    """Scale by one pure bucket's STD while centering on stop-now values."""

    source = returns.detach() if torch.is_tensor(returns) else torch.as_tensor(returns, dtype=torch.float64)
    baseline_source = (
        stop_now_values.detach()
        if torch.is_tensor(stop_now_values)
        else torch.as_tensor(stop_now_values, dtype=torch.float64)
    )
    values = source.detach().to(device="cpu", dtype=torch.float64)
    baselines = baseline_source.detach().to(device="cpu", dtype=torch.float64)
    if values.ndim != 1 or int(values.numel()) < 1:
        raise ValueError("G011 observed bucket must contain a record")
    if baselines.shape != values.shape:
        raise ValueError("G011 stop-now values do not cover the bucket")
    if not bool(torch.isfinite(values).all().item()) or not bool(
        torch.isfinite(baselines).all().item()
    ):
        raise ValueError("G011 bucket received nonfinite return or stop-now value")
    mean = values.mean(dtype=torch.float64)
    return_centered = values - mean
    residuals = values - baselines
    std = torch.sqrt(torch.mean(return_centered * return_centered, dtype=torch.float64))
    active = int(values.numel()) >= 2
    if active:
        scale = torch.maximum(std, torch.tensor(MINIMUM_STD, dtype=torch.float64))
        unclipped = residuals / scale
        advantages = torch.clamp(
            unclipped, -MAX_ABS_ADVANTAGE, MAX_ABS_ADVANTAGE
        ).detach()
    else:
        scale = None
        unclipped = torch.zeros_like(values)
        advantages = torch.zeros_like(values)
    return {
        "returns_to_go": values.tolist(),
        "stop_now_values": baselines.tolist(),
        "stop_now_residuals": residuals.tolist(),
        "record_count": int(values.numel()),
        "active_count": int(values.numel()) if active else 0,
        "return_mean_report_only": float(mean.item()),
        "baseline_source": "pluggable_stop_now_value",
        "population_std": float(std.item()),
        "scale": None if scale is None else float(scale.item()),
        "std_floor_binds": bool(active and float(std.item()) < MINIMUM_STD),
        "unclipped_advantages": unclipped.tolist(),
        "advantages": advantages.tolist(),
        "policy_active_for_update": active,
        "disabled_record_count": 0 if active else int(values.numel()),
        "zero_std": abs(float(std.item())) <= EPSILON,
        "minimum_std": MINIMUM_STD,
        "clamp": [-MAX_ABS_ADVANTAGE, MAX_ABS_ADVANTAGE],
        "reduction_dtype": str(values.dtype),
        "detached": True,
        "hard_projection": False,
        "empirical_mean_used_as_center": False,
        "mixed_exactness_fallback": False,
        "singleton_fallback": None,
    }


def assign_stop_now_advantages(
    credits: Sequence[Mapping[str, Any]],
    *,
    stop_now_value: Callable[[Mapping[str, Any]], float] = counting_stop_now_value,
    require_exact_group: bool = True,
) -> dict[str, Any]:
    """Assign G011 credit after all complete K=28 trajectories terminate."""

    scorer_identity, scorer_version = _scorer_metadata(stop_now_value)
    rows = [dict(value) for value in credits]
    if require_exact_group and len(rows) != GROUP_SIZE:
        raise ValueError("G011 advantage assignment requires exact K=28")
    if len(rows) < 2:
        raise ValueError("G011 advantage assignment requires a group")
    ordered = sorted(rows, key=lambda value: int(value["trajectory_index"]))
    indexes = [int(value["trajectory_index"]) for value in ordered]
    if len(indexes) != len(set(indexes)):
        raise ValueError("G011 trajectory indexes are duplicated")
    if require_exact_group and indexes != list(range(GROUP_SIZE)):
        raise ValueError("G011 K=28 trajectory coverage is not 0..27")
    identities = {
        (
            str(value.get("prompt") or ""),
            str(value.get("uid") or ""),
            str(value.get("r0_sha256") or ""),
            int(value.get("target_count", -1)),
        )
        for value in ordered
    }
    if len(identities) != 1 or "" in next(iter(identities))[:3]:
        raise ValueError("G011 group mixed prompt/UID/exact R0 state")
    if any(
        value.get("version") != G009_REWARD_VERSION
        or bool(value.get("r0_policy_active"))
        or int(value.get("r0_text_record_count", -1)) != 0
        or int(value.get("r0_flow_record_count", -1)) != 0
        or value.get("r0_advantage") is not None
        or value.get("r0_reward") is not None
        for value in ordered
    ):
        raise ValueError("G011 detached R0 or frozen reward contract differs")

    buckets: dict[tuple[int, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        for round_index in range(len(row["actions"])):
            buckets[(round_index, exact_before_action(row, round_index))].append(row)

    bucket_reports: list[dict[str, Any]] = []
    by_trajectory: dict[int, dict[int, tuple[float, bool, float]]] = defaultdict(dict)
    for (round_index, exact), members in sorted(buckets.items()):
        stop_states = [
            {
                "credit": value,
                "round_index": round_index,
                "action": value["actions"][round_index],
                "state_index_before_action": _action_state_index(
                    value["actions"], round_index
                ),
                "exact_before_action": exact,
                "detected_count_before_action": value["detected_counts"][
                    _action_state_index(value["actions"], round_index)
                ],
                "target_count": value["target_count"],
            }
            for value in members
        ]
        report = stratified_bucket_advantages(
            [value["return_to_go"][round_index] for value in members],
            [stop_now_value(state) for state in stop_states],
        )
        active = bool(report["policy_active_for_update"])
        for member, advantage, baseline in zip(
            members, report["advantages"], report["stop_now_values"]
        ):
            by_trajectory[int(member["trajectory_index"])][round_index] = (
                float(advantage),
                active,
                float(baseline),
            )
        bucket_reports.append(
            {
                "round_index": round_index,
                "exact_before_action": exact,
                **report,
            }
        )

    records = []
    for row in ordered:
        trajectory_index = int(row["trajectory_index"])
        controller_advantages: list[float] = []
        controller_active: list[bool] = []
        repair_flow_advantages: list[float] = []
        repair_flow_active: list[bool] = []
        rounds = []
        for round_index, action in enumerate(row["actions"]):
            advantage, active, stop_value = by_trajectory[trajectory_index][round_index]
            exact = exact_before_action(row, round_index)
            report = next(
                value
                for value in bucket_reports
                if int(value["round_index"]) == round_index
                and bool(value["exact_before_action"]) is exact
            )
            controller_advantages.append(advantage)
            controller_active.append(active)
            if action == "edit":
                repair_flow_advantages.append(advantage)
                repair_flow_active.append(active)
            rounds.append(
                {
                    **dict(row["rounds"][round_index]),
                    "exact_before_action": exact,
                    "bucket_round_index": round_index,
                    "bucket_exact_before_action": exact,
                    "bucket_record_count": int(report["record_count"]),
                    "bucket_return_mean_report_only": float(
                        report["return_mean_report_only"]
                    ),
                    "bucket_population_std": float(report["population_std"]),
                    "bucket_std_floor_binds": bool(report["std_floor_binds"]),
                    "stop_now_value": stop_value,
                    "stop_now_scorer_identity": scorer_identity,
                    "stop_now_scorer_version": scorer_version,
                    "advantage": advantage,
                    "controller_policy_active": active,
                    "repair_flow_policy_active": bool(
                        active and action == "edit"
                    ),
                }
            )
        records.append(
            {
                **row,
                "advantage_version": ADVANTAGE_VERSION,
                "stop_now_scorer_identity": scorer_identity,
                "stop_now_scorer_version": scorer_version,
                "controller_advantages": controller_advantages,
                "controller_active": controller_active,
                "repair_flow_advantages": repair_flow_advantages,
                "repair_flow_active": repair_flow_active,
                "rounds": rounds,
                "trajectory_scalar_advantage_broadcast": False,
                "r0_policy_active": False,
                "r0_text_record_count": 0,
                "r0_flow_record_count": 0,
                "r0_reward": None,
                "r0_advantage": None,
            }
        )

    active_advantages = [
        advantage
        for record in records
        for advantage, active in zip(
            record["controller_advantages"], record["controller_active"]
        )
        if active
    ]
    if any(
        not math.isfinite(value) or abs(value) > MAX_ABS_ADVANTAGE + EPSILON
        for value in active_advantages
    ):
        raise RuntimeError("G011 active advantage is nonfinite or unclamped")
    for record in records:
        flow_position = 0
        for action, controller_value, controller_is_active in zip(
            record["actions"],
            record["controller_advantages"],
            record["controller_active"],
        ):
            if action != "edit":
                continue
            if (
                record["repair_flow_advantages"][flow_position]
                != controller_value
                or record["repair_flow_active"][flow_position]
                != controller_is_active
            ):
                raise AssertionError("G011 EDIT controller/flow credit differs")
            flow_position += 1

    # Explicitly prove every observed bucket is pure and every singleton was
    # disabled rather than falling back to a mixed baseline.
    if any(
        len(
            {
                exact_before_action(member, int(report["round_index"]))
                for member in ordered
                if len(member["actions"]) > int(report["round_index"])
                and exact_before_action(member, int(report["round_index"]))
                is bool(report["exact_before_action"])
            }
        )
        != 1
        for report in bucket_reports
    ):
        raise AssertionError("G011 exact/non-exact baseline purity differs")
    if any(
        int(report["record_count"]) == 1
        and (
            report["policy_active_for_update"] is not False
            or report["advantages"] != [0.0]
            or report["mixed_exactness_fallback"] is not False
        )
        for report in bucket_reports
    ):
        raise AssertionError("G011 singleton bucket used a fallback")

    identity = next(iter(identities))
    result = {
        "version": ADVANTAGE_VERSION,
        "stop_now_scorer_identity": scorer_identity,
        "stop_now_scorer_version": scorer_version,
        "group_identity": {
            "prompt": identity[0],
            "uid": identity[1],
            "r0_sha256": identity[2],
            "target_count": identity[3],
        },
        "trajectory_count": len(records),
        "records": records,
        "round_diagnostics": bucket_reports,
        "bucket_diagnostics": bucket_reports,
        "observed_bucket_count": len(bucket_reports),
        "singleton_bucket_count": sum(
            int(value["record_count"]) == 1 for value in bucket_reports
        ),
        "disabled_record_count": sum(
            int(value["disabled_record_count"]) for value in bucket_reports
        ),
        "exact_nonexact_mixed_bucket_count": 0,
        "r0_policy_active_count": 0,
        "r0_text_record_count": 0,
        "r0_flow_record_count": 0,
        "r0_gradient_contribution": 0.0,
        "trajectory_level_advantage_broadcast": False,
        "hard_projection": False,
        "empirical_mean_used_as_center": False,
        "std_floor_binding_bucket_count": sum(
            bool(value["std_floor_binds"]) for value in bucket_reports
        ),
        "max_abs_active_advantage": max(
            (abs(value) for value in active_advantages), default=0.0
        ),
    }
    result["stop_now_table_invariants"] = (
        validate_stop_now_table_invariants(result)
        if scorer_identity == STOP_NOW_SCORER_IDENTITY
        and scorer_version == STOP_NOW_SCORER_VERSION
        else {
            "passed": None,
            "not_applicable": True,
            "reason": "task-specific invariants belong to the supplied scorer",
        }
    )
    return result


def validate_stop_now_table_invariants(
    projection: Mapping[str, Any], *, require_category_coverage: bool = False
) -> dict[str, Any]:
    """Execute the four construction-level G011 sign invariants."""

    categories: dict[str, list[float]] = {
        "already_exact_done": [],
        "already_exact_edit": [],
        "broken_false_done": [],
        "broken_edit_then_exact": [],
    }
    inactive_counts = {key: 0 for key in categories}
    violations: list[str] = []
    for record in projection["records"]:
        final_exact = bool(record["terminal_strict_exact_success"])
        for round_index, (action, active, advantage) in enumerate(
            zip(
                record["actions"],
                record["controller_active"],
                record["controller_advantages"],
            )
        ):
            exact = exact_before_action(record, round_index)
            key = None
            if exact and action == "done":
                key = "already_exact_done"
            elif exact and action == "edit":
                key = "already_exact_edit"
            elif not exact and action == "done":
                key = "broken_false_done"
            elif not exact and action == "edit" and final_exact:
                key = "broken_edit_then_exact"
            if key is None:
                continue
            if not active:
                inactive_counts[key] += 1
                continue
            value = float(advantage)
            categories[key].append(value)
            valid = {
                "already_exact_done": abs(value) <= EPSILON,
                "already_exact_edit": value < -EPSILON,
                "broken_false_done": abs(value) <= EPSILON,
                "broken_edit_then_exact": value > EPSILON,
            }[key]
            if not valid:
                violations.append(
                    f"trajectory{record['trajectory_index']}/round{round_index}/{key}={value}"
                )
    missing = [key for key, values in categories.items() if not values]
    if violations or (require_category_coverage and missing):
        raise AssertionError(
            "G011 stop-now table invariant differs: "
            f"violations={violations[:20]}, missing={missing}"
        )
    return {
        "passed": not violations and (not require_category_coverage or not missing),
        "active_counts": {key: len(values) for key, values in categories.items()},
        "inactive_counts": inactive_counts,
        "missing_active_categories": missing,
        "violation_count": len(violations),
    }


def _forbidden_keys(value: Any, prefix: str = "root") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().casefold()
            if normalized in _FORBIDDEN_POLICY_KEYS:
                found.append(f"{prefix}.{key}")
            found.extend(_forbidden_keys(item, f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_forbidden_keys(item, f"{prefix}[{index}]"))
    return found


def assert_policy_observation_secrecy(
    trajectories: Sequence[Any],
) -> dict[str, Any]:
    """Fail before reward/update if a hidden detector field enters policy IO.

    Model-authored prose may reason about the visible object count.  The guard
    therefore checks host field names/markers and policy surfaces, not ordinary
    words sampled by the model after seeing an image.
    """

    rows = list(trajectories)
    if not rows:
        raise ValueError("G011 secrecy guard requires trajectories")
    forbidden_paths: list[str] = []
    forbidden_markers: list[dict[str, Any]] = []
    turn_count = 0
    for trajectory_index, trajectory in enumerate(rows):
        forbidden_paths.extend(
            _forbidden_keys(
                getattr(trajectory, "metadata", {}),
                f"trajectory[{trajectory_index}].metadata",
            )
        )
        for turn_index, turn in enumerate(getattr(trajectory, "text_turns", [])):
            turn_count += 1
            forbidden_paths.extend(
                _forbidden_keys(
                    {
                        "context_terms": list(turn.context_terms),
                        "forced_prefix_token_ids": list(turn.forced_prefix_token_ids),
                        "sampled_token_ids": list(turn.sampled_token_ids),
                        "sampled_token_allowed_ids": turn.sampled_token_allowed_ids,
                        "sampled_token_policy_active": turn.sampled_token_policy_active,
                    },
                    f"trajectory[{trajectory_index}].turn[{turn_index}]",
                )
            )
            host_strings = [
                value
                for value in [*turn.context_terms, turn.canonical_response]
                if isinstance(value, str)
            ]
            for marker in _FORBIDDEN_POLICY_MARKERS:
                if any(marker.casefold() in value.casefold() for value in host_strings):
                    forbidden_markers.append(
                        {
                            "trajectory_index": trajectory_index,
                            "turn_index": turn_index,
                            "marker": marker,
                        }
                    )
    if forbidden_paths or forbidden_markers:
        raise RuntimeError(
            "G011 hidden detector/exact field entered a policy surface: "
            f"paths={forbidden_paths[:20]}, markers={forbidden_markers[:20]}"
        )
    return {
        "version": SECRECY_VERSION,
        "passed": True,
        "trajectory_count": len(rows),
        "text_turn_count": turn_count,
        "forbidden_key_path_count": 0,
        "forbidden_marker_count": 0,
        "exact_label_available_to_guard": False,
        "sampled_token_constraint_role": "SFT-native SCORE trie only",
    }


# Compatibility spellings for the inherited trainer plumbing.  Both execute
# the stop-now center; neither retains G010's empirical mean center.
assign_exactness_stratified_advantages = assign_stop_now_advantages
assign_round_causal_advantages = assign_stop_now_advantages


__all__ = [
    "ADVANTAGE_VERSION",
    "EDIT_COST",
    "EDIT_PROGRESS_COEFFICIENT",
    "GROUP_SIZE",
    "MAX_ABS_ADVANTAGE",
    "MAX_EDIT_ROUNDS",
    "MINIMUM_STD",
    "SECRECY_VERSION",
    "STOP_NOW_SCORER_IDENTITY",
    "STOP_NOW_SCORER_VERSION",
    "TERMINAL_FAILURE",
    "TERMINAL_SUCCESS",
    "VERSION",
    "assign_exactness_stratified_advantages",
    "assign_round_causal_advantages",
    "assign_stop_now_advantages",
    "assert_policy_observation_secrecy",
    "build_trajectory_process_credit",
    "build_verification_contract",
    "counting_score",
    "counting_stop_now_value",
    "edit_reward",
    "exact_before_action",
    "image_state_sha256",
    "stratified_bucket_advantages",
    "terminal_reward",
    "validate_cap_terminal_does_not_underprice_done",
    "validate_stop_now_table_invariants",
    "validate_no_positive_detector_failed_done",
    "validate_reward_ordering",
]
