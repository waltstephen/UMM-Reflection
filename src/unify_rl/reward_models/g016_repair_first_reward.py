"""G016 independent repair and action-calibration credit heads."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Iterable

VERSION = "clean29529_g017_a1_progress_reward_v1"
CONTROLLER_CREDIT_VERSION = VERSION
REPAIR_FLOW_CREDIT_VERSION = (
    "clean29529_g017_a1_progress_reward_repair_flow_v1"
)
REPAIR_BUCKET_VERSION = (
    "clean29529_g017_a1_progress_reward_round_exactness_bucket_v1"
)
REPAIR_CLAMP = 1.0
ACTION_CLAMP = 1.0


def _clamp(value: float, bound: float) -> float:
    return max(-bound, min(bound, float(value)))


def _deadzone(value: float, floor: float) -> float:
    """Discard detector movement at or below the measured noise floor."""

    value = float(value)
    floor = float(floor)
    if not math.isfinite(value):
        raise ValueError("G017 repair progress value must be finite")
    if not math.isfinite(floor) or not 0.0 <= floor < 0.5:
        raise ValueError("G017 repair progress deadzone must be in [0, 0.5)")
    return 0.0 if abs(value) <= floor else value


def _leave_one_out(values: list[float], index: int) -> float:
    if len(values) < 2:
        return 0.0
    return (sum(values) - values[index]) / (len(values) - 1)


def assign_g016_two_head_advantages(
    records: Iterable[dict[str, Any]],
    *,
    repair_progress_deadzone: float,
) -> list[dict[str, Any]]:
    rows = [dict(value) for value in records]
    buckets: dict[tuple[int, bool], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        buckets[
            (
                int(row["round_index"]),
                bool(row["exact_before_action"]),
            )
        ].append(index)

    output: list[dict[str, Any] | None] = [None] * len(rows)
    for bucket, indexes in buckets.items():
        bucket_active = len(indexes) >= 2
        action_correct = []
        for index in indexes:
            row = rows[index]
            exact = bool(row["exact_before_action"])
            semantic_action = str(row.get("semantic_action", "")).lower()
            actionable_edit = bool(row.get("actionable_edit"))
            action_correct.append(
                float(
                    (not exact and actionable_edit)
                    or (exact and semantic_action == "stop")
                )
            )
        for local_index, index in enumerate(indexes):
            row = rows[index]
            repair_active = (
                bucket_active
                and not bool(row["exact_before_action"])
                and bool(row.get("actionable_edit"))
            )
            q_delta = (
                float(row.get("q_after", row.get("q_before", 0.0)))
                - float(row.get("q_before", 0.0))
                if repair_active
                else 0.0
            )
            reached_exact = bool(row.get("reached_exact_after_action"))
            repair_raw_reward = (
                _clamp(
                    _deadzone(q_delta, repair_progress_deadzone),
                    REPAIR_CLAMP,
                )
                if repair_active
                else 0.0
            )
            repair_advantage = repair_raw_reward
            action_advantage = (
                _clamp(
                    action_correct[local_index]
                    - _leave_one_out(action_correct, local_index),
                    ACTION_CLAMP,
                )
                if bucket_active
                else 0.0
            )
            output[index] = {
                **row,
                "g016_bucket": [bucket[0], bucket[1]],
                "g016_bucket_record_count": len(indexes),
                "repair_raw_reward": repair_raw_reward,
                "repair_head_advantage": repair_advantage,
                "repair_head_active": repair_active,
                "repair_success": reached_exact,
                "q_delta": q_delta,
                "repair_progress_deadzone": float(repair_progress_deadzone),
                "repair_payload_active": repair_active,
                "repair_flow_active": repair_active,
                "action_head_advantage": action_advantage,
                "action_head_active": bucket_active,
                "positive_done_bonus": 0.0,
                "raw_positive_one_bypass": False,
                "mixed_exactness_baseline": False,
                "version": VERSION,
            }
    if any(value is None for value in output):
        raise AssertionError("G016 two-head output coverage differs")
    return [value for value in output if value is not None]


def _classify(semantic_action: str, raw_response: str) -> dict[str, Any]:
    # The sampled action is the protocol field on its own line before
    # ``[THINKING]``.  Reasoning may quote e.g. ``[ACTION] done``; treating
    # the last textual occurrence as the action breaks generation-span
    # lineage and incorrectly routes the bounded invalid-action credit.
    # Keep strict parser-v2 semantics unchanged, but identify the first exact
    # sampled action-like value for token routing.  The observed ``[^ACTION]``
    # near miss remains semantic_action=invalid; recognizing its edit/done
    # value only allows bounded invalid-action credit to land on that sampled
    # value instead of aborting the whole pre-update group.
    action_match = re.search(
        r"(?m)^\[(?P<caret>\^?)ACTION\][ \t]*(?P<action>[^\r\n]+?)[ \t]*$",
        raw_response,
        re.IGNORECASE,
    )
    raw_action = (
        action_match.group("action").strip().lower()
        if action_match
        and (
            action_match.group("caret") == ""
            or action_match.group("action").strip().lower() in {"edit", "done"}
        )
        else "invalid"
    )
    action_marker_strict = bool(
        action_match and action_match.group("caret") == ""
    )
    payload_match = re.search(
        r"</think>\s*\[EDIT\]\s*(.*)\Z",
        raw_response,
        re.IGNORECASE | re.DOTALL,
    )
    payload = payload_match.group(1).strip() if payload_match else ""
    alias_stop = (
        semantic_action == "done"
        and raw_action == "edit"
        and payload.casefold() == "none"
    )
    literal_stop = semantic_action == "done" and raw_action == "done"
    invalid_sampled_action = semantic_action == "invalid" and raw_action != "invalid"
    actionable_edit = (
        semantic_action == "edit"
        and raw_action == "edit"
        and payload.casefold() not in {"", "none"}
    )
    return {
        "semantic_action": semantic_action,
        "raw_action": raw_action,
        "raw_action_marker_strict": action_marker_strict,
        "payload": payload,
        "actionable_edit": actionable_edit,
        "alias_stop": alias_stop,
        "literal_stop": literal_stop,
        "invalid_sampled_action": invalid_sampled_action,
        "token_route": (
            "repair_payload_and_edit_action"
            if actionable_edit
            else "none_payload_action"
            if alias_stop
            else "done_action"
            if literal_stop
            else "invalid_sampled_action"
            if invalid_sampled_action
            else "invalid"
        ),
    }


def assign_g016_advantages(
    credits: Any,
    *,
    repair_progress_deadzone: float,
    require_exact_group: bool = True,
) -> dict[str, Any]:
    from unify_rl.reward_models.g014_channel_decoupled_reward import (
        assign_channel_decoupled_advantages,
    )

    projection = assign_channel_decoupled_advantages(
        credits,
        require_exact_group=require_exact_group,
    )
    records = list(projection["records"])
    descriptors = []
    for record in records:
        flow_position = 0
        for round_index, (round_row, semantic_action, raw_response) in enumerate(
            zip(
                record["rounds"],
                record["actions"],
                record["raw_controller_responses"],
            )
        ):
            decision = _classify(
                str(semantic_action),
                str(raw_response),
            )
            round_row["g016_decision"] = decision
            has_flow_call = str(semantic_action).strip().casefold() == "edit"
            descriptor = {
                "record": record,
                "row": round_row,
                "round_index": round_index,
                "flow_position": flow_position if has_flow_call else None,
                "exact_before_action": bool(
                    round_row["exact_before_action"]
                ),
                "semantic_action": (
                    "stop"
                    if decision["alias_stop"] or decision["literal_stop"]
                    else str(semantic_action)
                ),
                "actionable_edit": decision["actionable_edit"],
                "reached_exact_after_action": (
                    abs(float(round_row.get("q_after", 0.0)) - 1.0) <= 1e-12
                ),
                "q_before": float(round_row.get("q_before", 0.0)),
                "q_after": float(round_row.get("q_after", 0.0)),
            }
            descriptors.append(descriptor)
            if has_flow_call:
                flow_position += 1
        if flow_position != len(record["repair_flow_advantages"]):
            raise RuntimeError("G016 round-to-flow lineage coverage differs")

    assigned = assign_g016_two_head_advantages(
        [
            {
                key: descriptor[key]
                for key in (
                    "round_index",
                    "exact_before_action",
                    "semantic_action",
                    "actionable_edit",
                    "reached_exact_after_action",
                    "q_before",
                    "q_after",
                )
            }
            for descriptor in descriptors
        ],
        repair_progress_deadzone=repair_progress_deadzone,
    )
    for descriptor, value in zip(descriptors, assigned):
        record = descriptor["record"]
        round_index = int(descriptor["round_index"])
        record["controller_advantages"][round_index] = float(
            value["action_head_advantage"]
        )
        record["controller_active"][round_index] = bool(
            value["action_head_active"]
        )
        flow_position = descriptor["flow_position"]
        if flow_position is not None:
            record["repair_flow_advantages"][flow_position] = float(
                value["repair_head_advantage"]
            )
            record["repair_flow_active"][flow_position] = bool(
                value["repair_head_active"]
            )
        descriptor["row"].update(value)
        descriptor["row"]["g016_two_head_credit_applied"] = True

    bucket_members: dict[tuple[int, bool], list[dict[str, Any]]] = defaultdict(
        list
    )
    for value in assigned:
        key = (
            int(value["g016_bucket"][0]),
            bool(value["g016_bucket"][1]),
        )
        bucket_members[key].append(value)
    bucket_reports = []
    for key, members in sorted(bucket_members.items()):
        active = len(members) >= 2
        bucket_reports.append(
            {
                "version": REPAIR_BUCKET_VERSION,
                "key": [key[0], key[1]],
                "round_index": key[0],
                "exact_before_action": key[1],
                "record_count": len(members),
                "advantages": [
                    float(value["action_head_advantage"])
                    for value in members
                ],
                "repair_advantages": [
                    float(value["repair_head_advantage"])
                    for value in members
                ],
                "policy_active_for_update": active,
                "mixed_exactness_fallback": False,
            }
        )

    for record in records:
        record["advantage_version"] = VERSION
        record["channel_credit_version"] = VERSION
        record["controller_credit_version"] = VERSION
        record["renderer_credit_version"] = REPAIR_FLOW_CREDIT_VERSION

    active_repair = [
        float(value["repair_head_advantage"])
        for value in assigned
        if value["repair_head_active"]
    ]
    active_action = [
        float(value["action_head_advantage"])
        for value in assigned
        if value["action_head_active"]
    ]
    projection.update(
        {
            "version": VERSION,
            "records": records,
            "channel_credit_version": VERSION,
            "controller_credit_version": VERSION,
            "renderer_credit_version": REPAIR_FLOW_CREDIT_VERSION,
            "renderer_bucket_version": REPAIR_BUCKET_VERSION,
            "round_diagnostics": bucket_reports,
            "g016_bucket_diagnostics": bucket_reports,
            "controller_credit_unchanged_from_stopnow_projection": False,
            "action_credit_source": (
                "exactness_stratified_symmetric_leave_one_out"
            ),
            "repair_credit_source": (
                "broken_edit_detector_deadzone_absolute_qdelta_zero_baseline"
            ),
            "repair_progress_deadzone": float(repair_progress_deadzone),
            "repair_raw_reward_equals_advantage": True,
            "repair_peer_baseline_used": False,
            "repair_exact_bonus": 0.0,
            "strict_exact_flip_bypasses_deadzone": False,
            "repair_payload_and_flow_share_head": True,
            "repair_trajectory_return_used": False,
            "renderer_trajectory_return_used": False,
            "decision_cross_step_pooling_used": False,
            "renderer_cross_step_pooling_used": False,
            "decision_exact_nonexact_mixed_bucket_count": 0,
            "renderer_exact_nonexact_mixed_bucket_count": 0,
            "exact_nonexact_mixed_bucket_count": 0,
            "positive_done_bonus": 0.0,
            "raw_positive_one_bypass": False,
            "max_abs_active_action_advantage": max(
                (abs(value) for value in active_action),
                default=0.0,
            ),
            "max_abs_active_repair_advantage": max(
                (abs(value) for value in active_repair),
                default=0.0,
            ),
            "token_routing_version": (
                "clean29529_g016_two_head_generation_span_routing_v2"
            ),
        }
    )
    return projection


__all__ = [
    "ACTION_CLAMP",
    "CONTROLLER_CREDIT_VERSION",
    "_deadzone",
    "REPAIR_BUCKET_VERSION",
    "REPAIR_CLAMP",
    "REPAIR_FLOW_CREDIT_VERSION",
    "VERSION",
    "assign_g016_advantages",
    "assign_g016_two_head_advantages",
]
