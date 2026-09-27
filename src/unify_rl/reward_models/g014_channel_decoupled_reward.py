"""G014 channel-decoupled counting credit.

The frozen G011 stop-now projection remains the controller source of truth.
Only EDIT renderer credit changes: each flow call receives a normalized local
``q_after - q_before`` transition delta from a same-step fallback bucket.
No bucket crosses exactness or a logical step.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

from unify_rl.reward_models.g009_counting_process_reward import (
    EPSILON,
    MAX_ABS_ADVANTAGE,
    MINIMUM_STD,
)
from unify_rl.reward_models.g011_stopnow_reward import (
    ADVANTAGE_VERSION as CONTROLLER_CREDIT_VERSION,
    STOP_NOW_SCORER_IDENTITY,
    STOP_NOW_SCORER_VERSION,
    VERSION,
    assign_stop_now_advantages,
    counting_stop_now_value,
)

CHANNEL_CREDIT_VERSION = "clean29529_g014_channel_decoupled_credit_v1"
RENDERER_CREDIT_VERSION = "clean29529_g014_transition_local_renderer_qdelta_v1"
RENDERER_BUCKET_VERSION = "clean29529_g014_same_step_renderer_fallback_v1"


def _q_key(value: Any) -> float:
    result = round(float(value), 12)
    if not math.isfinite(result):
        raise ValueError("G014 renderer bucket received nonfinite q_before")
    return result


def _local_credit(round_row: Mapping[str, Any]) -> float:
    value = float(round_row["q_after"]) - float(round_row["q_before"])
    if not math.isfinite(value):
        raise ValueError("G014 renderer received nonfinite local transition credit")
    return value


def assign_channel_decoupled_advantages(
    credits: Sequence[Mapping[str, Any]],
    *,
    stop_now_value: Callable[[Mapping[str, Any]], float] = counting_stop_now_value,
    require_exact_group: bool = True,
) -> dict[str, Any]:
    """Keep G011 controller credit and replace only renderer EDIT credit."""

    projection = assign_stop_now_advantages(
        credits,
        stop_now_value=stop_now_value,
        require_exact_group=require_exact_group,
    )
    records = list(projection["records"])
    descriptors: list[dict[str, Any]] = []
    exact_preserved_count = 0
    exact_preserved_active_count = 0
    level0: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        flow_position = 0
        target = int(record["target_count"])
        for round_index, (action, round_row) in enumerate(
            zip(record["actions"], record["rounds"])
        ):
            round_row["controller_credit_version"] = CONTROLLER_CREDIT_VERSION
            round_row["renderer_credit_version"] = (
                RENDERER_CREDIT_VERSION if action == "edit" else None
            )
            if action != "edit":
                continue
            exact = bool(round_row["exact_before_action"])
            q_before = _q_key(round_row["q_before"])
            descriptor = {
                "record": record,
                "round": round_row,
                "trajectory_index": int(record["trajectory_index"]),
                "round_index": round_index,
                "flow_position": flow_position,
                "exact": exact,
                "q_before": q_before,
                "target_count": target,
                "local_credit": _local_credit(round_row),
            }
            flow_position += 1
            if exact:
                # The only proven G011 guarantee is retained on both channels:
                # an exact-state EDIT receives the same stop-now credit as its
                # controller action and can never become positive through local
                # centering against damaged siblings.
                exact_preserved_count += 1
                exact_active = bool(record["repair_flow_active"][flow_position - 1])
                exact_preserved_active_count += exact_active
                round_row.update(
                    {
                        "renderer_local_credit": None,
                        "renderer_advantage": float(
                            record["repair_flow_advantages"][flow_position - 1]
                        ),
                        "renderer_policy_active": exact_active,
                        "renderer_bucket_level": "exact_stopnow_preserved",
                        "renderer_bucket_key": [True, round_index],
                        "renderer_bucket_record_count": None,
                        "renderer_bucket_mean": None,
                        "renderer_bucket_population_std": None,
                        "renderer_bucket_scale": None,
                        "renderer_bucket_std_floor_binds": None,
                    }
                )
                continue
            descriptors.append(descriptor)
            level0[(False, q_before, round_index, target)].append(descriptor)
        if flow_position != len(record["repair_flow_advantages"]):
            raise AssertionError("G014 EDIT/renderer record coverage differs")

    selected: dict[tuple[str, tuple[Any, ...]], list[dict[str, Any]]] = defaultdict(list)
    coverage = {
        "exact_stopnow_preserved": exact_preserved_count,
        "exact_stopnow_preserved_active": exact_preserved_active_count,
        "level0_exact_q_round_target_direct": 0,
        "level1_exact_q_round_rescued": 0,
        "level2_exact_round_rescued": 0,
        "disabled": 0,
    }
    unresolved = []
    for descriptor in descriptors:
        key = (
            False,
            descriptor["q_before"],
            descriptor["round_index"],
            descriptor["target_count"],
        )
        if len(level0[key]) >= 2:
            descriptor["bucket_level"] = "level0_exact_q_round_target_direct"
            descriptor["bucket_key"] = key
            selected[(descriptor["bucket_level"], key)].append(descriptor)
            coverage[descriptor["bucket_level"]] += 1
        else:
            unresolved.append(descriptor)

    # Rebuild each fallback only from records that actually reached that level.
    # This prevents a large pre-selection group from activating a singleton
    # after direct-bucket members choose an earlier level.
    for level, key_fn, coverage_name in (
        (
            "level1",
            lambda row: (False, row["q_before"], row["round_index"]),
            "level1_exact_q_round_rescued",
        ),
        (
            "level2",
            lambda row: (False, row["round_index"]),
            "level2_exact_round_rescued",
        ),
    ):
        fallback_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for descriptor in unresolved:
            fallback_groups[key_fn(descriptor)].append(descriptor)
        next_unresolved = []
        for key, members in fallback_groups.items():
            if len(members) < 2:
                next_unresolved.extend(members)
                continue
            for descriptor in members:
                descriptor["bucket_level"] = coverage_name
                descriptor["bucket_key"] = key
            selected[(coverage_name, key)].extend(members)
            coverage[coverage_name] += len(members)
        unresolved = next_unresolved

    for descriptor in unresolved:
        descriptor["bucket_level"] = "disabled"
        descriptor["bucket_key"] = None
        coverage["disabled"] += 1

    bucket_reports = []
    for (level, key), members in sorted(
        selected.items(), key=lambda item: (item[0][0], repr(item[0][1]))
    ):
        values = [float(row["local_credit"]) for row in members]
        center = statistics.mean(values)
        population_std = statistics.pstdev(values)
        scale = max(population_std, MINIMUM_STD)
        advantages = [
            max(
                -MAX_ABS_ADVANTAGE,
                min(MAX_ABS_ADVANTAGE, (value - center) / scale),
            )
            for value in values
        ]
        if any(not math.isfinite(value) for value in advantages):
            raise RuntimeError("G014 renderer advantage is nonfinite")
        for descriptor, advantage in zip(members, advantages):
            record = descriptor["record"]
            flow_position = int(descriptor["flow_position"])
            record["repair_flow_advantages"][flow_position] = float(advantage)
            record["repair_flow_active"][flow_position] = True
            descriptor["round"].update(
                {
                    "renderer_local_credit": float(descriptor["local_credit"]),
                    "renderer_advantage": float(advantage),
                    "renderer_policy_active": True,
                    "renderer_bucket_level": level,
                    "renderer_bucket_key": list(key),
                    "renderer_bucket_record_count": len(members),
                    "renderer_bucket_mean": float(center),
                    "renderer_bucket_population_std": float(population_std),
                    "renderer_bucket_scale": float(scale),
                    "renderer_bucket_std_floor_binds": population_std < MINIMUM_STD,
                }
            )
        bucket_reports.append(
            {
                "version": RENDERER_BUCKET_VERSION,
                "level": level,
                "key": list(key),
                "exact_before_action": bool(key[0]),
                "record_count": len(members),
                "local_credit_mean": float(center),
                "local_credit_population_std": float(population_std),
                "scale": float(scale),
                "std_floor_binds": population_std < MINIMUM_STD,
                "advantages": [float(value) for value in advantages],
                "policy_active_for_update": True,
            }
        )

    for descriptor in descriptors:
        if descriptor["bucket_level"] != "disabled":
            continue
        record = descriptor["record"]
        flow_position = int(descriptor["flow_position"])
        record["repair_flow_advantages"][flow_position] = 0.0
        record["repair_flow_active"][flow_position] = False
        descriptor["round"].update(
            {
                "renderer_local_credit": float(descriptor["local_credit"]),
                "renderer_advantage": 0.0,
                "renderer_policy_active": False,
                "renderer_bucket_level": "disabled",
                "renderer_bucket_key": None,
                "renderer_bucket_record_count": 1,
                "renderer_bucket_mean": None,
                "renderer_bucket_population_std": None,
                "renderer_bucket_scale": None,
                "renderer_bucket_std_floor_binds": None,
            }
        )

    for record in records:
        record["advantage_version"] = CHANNEL_CREDIT_VERSION
        record["channel_credit_version"] = CHANNEL_CREDIT_VERSION
        record["controller_credit_version"] = CONTROLLER_CREDIT_VERSION
        record["renderer_credit_version"] = RENDERER_CREDIT_VERSION

    active_renderer = [
        float(value)
        for record in records
        for value, active in zip(
            record["repair_flow_advantages"], record["repair_flow_active"]
        )
        if active
    ]
    difference_count = 0
    edit_count = 0
    for record in records:
        flow_position = 0
        for action, controller in zip(
            record["actions"], record["controller_advantages"]
        ):
            if action != "edit":
                continue
            renderer = float(record["repair_flow_advantages"][flow_position])
            difference_count += abs(renderer - float(controller)) >= 1e-12
            edit_count += 1
            flow_position += 1

    projection.update(
        {
            "version": CHANNEL_CREDIT_VERSION,
            "channel_credit_version": CHANNEL_CREDIT_VERSION,
            "controller_credit_version": CONTROLLER_CREDIT_VERSION,
            "renderer_credit_version": RENDERER_CREDIT_VERSION,
            "renderer_bucket_version": RENDERER_BUCKET_VERSION,
            "records": records,
            "controller_credit_unchanged_from_stopnow_projection": True,
            "renderer_credit_source": "nonexact_transition_local_qdelta_exact_stopnow_preserved",
            "renderer_trajectory_return_used": False,
            "renderer_nonexact_trajectory_return_used": False,
            "renderer_exact_credit_source": "controller_stopnow_advantage",
            "renderer_bucket_fallback_ladder": [
                "(exactness,q_before,round_index,target_count)",
                "(exactness,q_before,round_index)",
                "(exactness,round_index)",
                "disable",
            ],
            "renderer_bucket_diagnostics": bucket_reports,
            "renderer_coverage_counts": coverage,
            "renderer_edit_record_count": edit_count,
            "renderer_active_record_count": len(active_renderer),
            "renderer_disabled_record_count": coverage["disabled"],
            "renderer_exact_stopnow_preserved_count": exact_preserved_count,
            "renderer_exact_stopnow_preserved_active_count": exact_preserved_active_count,
            "renderer_active_fraction": (
                len(active_renderer) / edit_count if edit_count else None
            ),
            "renderer_controller_difference_count": difference_count,
            "renderer_controller_difference_fraction": (
                difference_count / edit_count if edit_count else None
            ),
            "renderer_exact_nonexact_mixed_bucket_count": 0,
            "renderer_cross_step_pooling_used": False,
            "renderer_max_abs_active_advantage": max(
                (abs(value) for value in active_renderer), default=0.0
            ),
            "renderer_std_floor_binding_bucket_count": sum(
                bool(row["std_floor_binds"]) for row in bucket_reports
            ),
        }
    )
    if projection["renderer_max_abs_active_advantage"] > MAX_ABS_ADVANTAGE + EPSILON:
        raise AssertionError("G014 renderer advantage clamp differs")
    if any(len(members) < 2 for members in selected.values()):
        raise AssertionError("G014 renderer activated a singleton fallback bucket")
    if any(
        len({bool(row["exact"]) for row in members}) != 1
        for members in selected.values()
    ):
        raise AssertionError("G014 renderer bucket mixed exactness")
    return projection


__all__ = [
    "CHANNEL_CREDIT_VERSION",
    "CONTROLLER_CREDIT_VERSION",
    "RENDERER_BUCKET_VERSION",
    "RENDERER_CREDIT_VERSION",
    "STOP_NOW_SCORER_IDENTITY",
    "STOP_NOW_SCORER_VERSION",
    "VERSION",
    "assign_channel_decoupled_advantages",
    "counting_stop_now_value",
]
