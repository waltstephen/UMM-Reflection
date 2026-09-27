"""Frozen G009 counting process reward and round-causal GRPO credit.

G009 has exactly two reward equations:

* executed EDIT: ``0.4 * (q_next - q_current) - 0.01``;
* terminal DONE/cap: ``1`` only for a parse-valid detector-exact final state,
  otherwise ``0``.

R0 is environment state only.  It has no action record, reward channel,
advantage, log probability, or gradient.  All arithmetic in this module is
Python binary64 (or detached CPU torch.float64 for tensor inputs).
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch


VERSION = "clean29529_g009_counting_process_reward_v1"
ADVANTAGE_VERSION = "clean29529_g009_round_causal_grpo_v1"
GROUP_SIZE = 28
MAX_EDIT_ROUNDS = 3
EDIT_PROGRESS_COEFFICIENT = 0.4
EDIT_COST = -0.01
TERMINAL_SUCCESS = 1.0
TERMINAL_FAILURE = 0.0
MINIMUM_STD = 0.1
MAX_ABS_ADVANTAGE = 5.0
EPSILON = 1e-12


def _finite(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def counting_score(target_count: Any, detected_count: Any) -> float:
    """Frozen hidden detector score for counting targets 2..6."""

    if isinstance(target_count, bool) or isinstance(detected_count, bool):
        raise ValueError("counting target/detection must be integers")
    target = int(target_count)
    detected = int(detected_count)
    if target != float(target_count) or detected != float(detected_count):
        raise ValueError("counting target/detection must be integers")
    if target not in range(2, 7) or detected < 0:
        raise ValueError("G009 requires target in [2,6] and detection >= 0")
    if detected == target:
        return 1.0
    return float(0.5 * max(0.0, 1.0 - abs(detected - target) / target))


def image_state_sha256(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(bytes(value)).hexdigest()


def build_verification_contract(
    constraints: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [dict(value) for value in constraints]
    if len(rows) != 1:
        raise ValueError("G009 requires exactly one counting clause")
    row = rows[0]
    target = int(row.get("exact_count", row.get("count", 0)) or 0)
    if (
        str(row.get("kind") or "") != "count_exact"
        or not str(row.get("id") or "")
        or not str(row.get("class") or "")
        or target not in range(2, 7)
    ):
        raise ValueError("G009 counting verification clause is invalid")
    row["exact_count"] = target
    return [row]


def edit_reward(q_current: Any, q_next: Any) -> float:
    current = _finite(q_current, name="q_current")
    following = _finite(q_next, name="q_next")
    if not (0.0 <= current <= 1.0 and 0.0 <= following <= 1.0):
        raise ValueError("counting scores must lie in [0,1]")
    result = EDIT_PROGRESS_COEFFICIENT * (following - current) + EDIT_COST
    if not math.isfinite(result):
        raise ValueError("G009 EDIT reward is nonfinite")
    return float(result)


def terminal_reward(*, parse_valid: bool, strict_exact_success: bool) -> float:
    if not isinstance(parse_valid, bool) or not isinstance(
        strict_exact_success, bool
    ):
        raise ValueError("terminal predicates must be boolean")
    return (
        TERMINAL_SUCCESS
        if parse_valid and strict_exact_success
        else TERMINAL_FAILURE
    )


def _normalize_actions(actions: Sequence[Any]) -> list[str]:
    normalized = [str(value).strip().casefold() for value in actions]
    if not normalized or any(
        value not in {"edit", "done", "invalid"} for value in normalized
    ):
        raise ValueError("G009 controller action topology is invalid")
    if len(normalized) > MAX_EDIT_ROUNDS + 1:
        raise ValueError("G009 trajectory exceeds its action horizon")
    terminals = [
        index for index, action in enumerate(normalized) if action != "edit"
    ]
    if len(terminals) > 1 or (terminals and terminals[0] != len(normalized) - 1):
        raise ValueError("the first DONE/invalid action must terminate generation")
    if sum(action == "edit" for action in normalized) > MAX_EDIT_ROUNDS:
        raise ValueError("G009 trajectory exceeds three executed EDITs")
    return normalized


def build_trajectory_process_credit(
    *,
    prompt: str,
    uid: str,
    trajectory_index: int,
    r0_sha256: str,
    target_count: int,
    detected_counts: Sequence[Any],
    actions: Sequence[Any],
    raw_controller_responses: Sequence[str] | None = None,
    canonical_controller_responses: Sequence[str] | None = None,
    parse_valid: bool = True,
    stop_reason: str | None = None,
    image_state_sha256s: Sequence[str] | None = None,
    image_states: Sequence[Mapping[str, Any]] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Recompute one complete G009 trajectory from hidden detector counts.

    ``detected_counts[0]`` is detached R0.  Every later count corresponds to
    exactly one executed EDIT.  DONE/invalid has no image call.  A trajectory
    with three EDIT actions and no explicit DONE is environment-cap terminal.
    """

    if not prompt or not uid or len(str(r0_sha256)) != 64:
        raise ValueError("G009 trajectory identity is incomplete")
    index = int(trajectory_index)
    if index < 0 or index >= GROUP_SIZE:
        raise ValueError("G009 trajectory index must lie in [0,27]")
    normalized_actions = _normalize_actions(actions)
    counts = [int(value) for value in detected_counts]
    if not counts or any(value < 0 for value in counts):
        raise ValueError("G009 detector state coverage is empty or invalid")
    edit_count = sum(action == "edit" for action in normalized_actions)
    if len(counts) != edit_count + 1:
        raise ValueError("G009 EDIT/image detector coverage differs")
    if normalized_actions[-1] == "edit" and edit_count != MAX_EDIT_ROUNDS:
        raise ValueError("missing DONE is terminal only after the third EDIT")

    inferred_stop = (
        "done"
        if normalized_actions[-1] == "done"
        else "parse_error"
        if normalized_actions[-1] == "invalid"
        else "environment_cap"
    )
    if stop_reason is not None:
        supplied = str(stop_reason)
        aliases = {"round_cap": "environment_cap", "cap": "environment_cap"}
        supplied = aliases.get(supplied, supplied)
        if supplied != inferred_stop:
            raise ValueError(
                f"G009 stop reason differs: {supplied!r} != {inferred_stop!r}"
            )
    stop = inferred_stop

    raw = None if raw_controller_responses is None else [str(v) for v in raw_controller_responses]
    canonical = (
        None
        if canonical_controller_responses is None
        else [str(v) for v in canonical_controller_responses]
    )
    if raw is not None and len(raw) != len(normalized_actions):
        raise ValueError("G009 raw controller response coverage differs")
    if canonical is not None and len(canonical) != len(normalized_actions):
        raise ValueError("G009 canonical controller response coverage differs")

    hashes: list[str] = []
    if image_state_sha256s is not None:
        hashes = [str(value) for value in image_state_sha256s]
        if len(hashes) != len(counts) or any(len(value) != 64 for value in hashes):
            raise ValueError("G009 image hash coverage differs")
        if hashes[0] != str(r0_sha256):
            raise ValueError("G009 trajectory changed its detached R0 identity")
    states = [] if image_states is None else [dict(value) for value in image_states]
    if states and len(states) != len(counts):
        raise ValueError("G009 persisted image-state coverage differs")

    scores = [counting_score(target_count, value) for value in counts]
    process_rewards = [
        edit_reward(scores[position], scores[position + 1])
        for position in range(edit_count)
    ]
    parse_ok = bool(parse_valid) and "invalid" not in normalized_actions
    strict_exact = counts[-1] == int(target_count)
    terminal = terminal_reward(
        parse_valid=parse_ok, strict_exact_success=strict_exact
    )

    # One immediate policy-round reward per action.  Cap terminal is an extra
    # environment event after the third EDIT, so the final EDIT's RTG includes
    # both its process reward and the terminal reward.
    policy_round_rewards: list[float] = []
    edit_position = 0
    for action in normalized_actions:
        if action == "edit":
            policy_round_rewards.append(process_rewards[edit_position])
            edit_position += 1
        else:
            policy_round_rewards.append(terminal)
    terminal_event_is_separate = normalized_actions[-1] == "edit"
    reward_events = [
        {
            "event_index": round_index,
            "event_kind": action,
            "policy_round_index": round_index,
            "reward": policy_round_rewards[round_index],
        }
        for round_index, action in enumerate(normalized_actions)
    ]
    if terminal_event_is_separate:
        reward_events.append(
            {
                "event_index": len(reward_events),
                "event_kind": "terminal_cap",
                "policy_round_index": None,
                "reward": terminal,
            }
        )

    returns_to_go = [0.0] * len(normalized_actions)
    running = terminal if terminal_event_is_separate else 0.0
    for round_index in range(len(normalized_actions) - 1, -1, -1):
        running += policy_round_rewards[round_index]
        returns_to_go[round_index] = float(running)
    trajectory_return = float(sum(process_rewards) + terminal)
    if not math.isfinite(trajectory_return) or any(
        not math.isfinite(value) for value in returns_to_go
    ):
        raise ValueError("G009 reward or return-to-go is nonfinite")
    if abs(returns_to_go[0] - trajectory_return) > EPSILON:
        raise AssertionError("G009 round-0 RTG differs from trajectory return")

    rounds = []
    edit_position = 0
    for round_index, action in enumerate(normalized_actions):
        flow_active = action == "edit"
        q_before = scores[edit_position] if flow_active else scores[-1]
        q_after = scores[edit_position + 1] if flow_active else scores[-1]
        rounds.append(
            {
                "round_index": round_index,
                "action": action,
                "controller_policy_active": True,
                "repair_flow_policy_active": flow_active,
                "q_before": q_before,
                "q_after": q_after,
                "process_reward": (
                    process_rewards[edit_position] if flow_active else None
                ),
                "terminal_reward": terminal if action != "edit" else None,
                "return_to_go": returns_to_go[round_index],
                "raw_controller_response": None if raw is None else raw[round_index],
                "canonical_controller_response": (
                    None if canonical is None else canonical[round_index]
                ),
            }
        )
        if flow_active:
            edit_position += 1

    return {
        "version": VERSION,
        "advantage_version": ADVANTAGE_VERSION,
        "prompt": str(prompt),
        "uid": str(uid),
        "seed": None if seed is None else int(seed),
        "trajectory_index": index,
        "r0_sha256": str(r0_sha256),
        "r0_policy_active": False,
        "r0_text_record_count": 0,
        "r0_flow_record_count": 0,
        "r0_reward": None,
        "r0_advantage": None,
        "target_count": int(target_count),
        "detected_counts": counts,
        "counting_scores": scores,
        "image_state_sha256s": hashes,
        "image_states": states,
        "actions": normalized_actions,
        "raw_controller_responses": raw,
        "canonical_controller_responses": canonical,
        "parse_valid": parse_ok,
        "stop_reason": stop,
        "first_done_round": (
            normalized_actions.index("done")
            if "done" in normalized_actions
            else None
        ),
        "edit_count": edit_count,
        "process_rewards": process_rewards,
        "terminal_reward": terminal,
        "terminal_reward_event_count": 1,
        "terminal_strict_exact_success": bool(parse_ok and strict_exact),
        "final_strict_exact": strict_exact,
        "terminal_soft_q_reward_effect": 0.0,
        "score_field_reward_effect": 0.0,
        "gpt_reward_effect": 0.0,
        "forbidden_reward_term_count": 0,
        "policy_round_rewards": policy_round_rewards,
        "reward_events": reward_events,
        "return_to_go": returns_to_go,
        "trajectory_return_report_only": trajectory_return,
        "rounds": rounds,
    }


def round_standard_advantages(
    returns: Sequence[Any] | torch.Tensor,
) -> dict[str, Any]:
    """Population-STD centering for one active round index."""

    source = (
        returns.detach()
        if torch.is_tensor(returns)
        else torch.as_tensor(returns, dtype=torch.float64)
    )
    values = source.detach().to(device="cpu", dtype=torch.float64)
    if values.ndim != 1 or int(values.numel()) < 2:
        raise ValueError("G009 active round centering requires at least two returns")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("G009 round centering received nonfinite returns")
    mean = values.mean(dtype=torch.float64)
    centered = values - mean
    std = torch.sqrt(torch.mean(centered * centered, dtype=torch.float64))
    scale = torch.maximum(std, torch.tensor(MINIMUM_STD, dtype=torch.float64))
    unclipped = centered / scale
    advantages = torch.clamp(
        unclipped, -MAX_ABS_ADVANTAGE, MAX_ABS_ADVANTAGE
    ).detach()
    return {
        "returns_to_go": values.tolist(),
        "active_count": int(values.numel()),
        "mean": float(mean.item()),
        "population_std": float(std.item()),
        "scale": float(scale.item()),
        "unclipped_advantages": unclipped.tolist(),
        "advantages": advantages.tolist(),
        "zero_std": abs(float(std.item())) <= EPSILON,
        "minimum_std": MINIMUM_STD,
        "clamp": [-MAX_ABS_ADVANTAGE, MAX_ABS_ADVANTAGE],
        "reduction_dtype": str(values.dtype),
        "detached": True,
        "hard_projection": False,
    }


def assign_round_causal_advantages(
    credits: Sequence[Mapping[str, Any]],
    *,
    require_exact_group: bool = True,
) -> dict[str, Any]:
    """Center RTG independently at every policy-active round index.

    The complete K=28 group must share prompt, UID, and exact R0 bytes.  A
    round with fewer than two active trajectories contributes to no optimizer.
    """

    rows = [dict(value) for value in credits]
    if require_exact_group and len(rows) != GROUP_SIZE:
        raise ValueError("G009 advantage assignment requires exact K=28")
    if len(rows) < 2:
        raise ValueError("G009 advantage assignment requires a group")
    ordered = sorted(rows, key=lambda value: int(value["trajectory_index"]))
    indexes = [int(value["trajectory_index"]) for value in ordered]
    if len(indexes) != len(set(indexes)):
        raise ValueError("G009 trajectory indexes are duplicated")
    if require_exact_group and indexes != list(range(GROUP_SIZE)):
        raise ValueError("G009 K=28 trajectory coverage is not 0..27")
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
        raise ValueError("G009 group mixed prompt/UID/exact R0 state")
    if any(
        value.get("version") != VERSION
        or bool(value.get("r0_policy_active"))
        or int(value.get("r0_text_record_count", -1)) != 0
        or int(value.get("r0_flow_record_count", -1)) != 0
        or value.get("r0_advantage") is not None
        or value.get("r0_reward") is not None
        for value in ordered
    ):
        raise ValueError("G009 detached R0 contract differs")

    max_rounds = max(len(value["actions"]) for value in ordered)
    round_reports: list[dict[str, Any]] = []
    by_trajectory: dict[int, dict[int, tuple[float, bool]]] = defaultdict(dict)
    for round_index in range(max_rounds):
        active = [
            value for value in ordered if len(value["actions"]) > round_index
        ]
        if len(active) >= 2:
            report = round_standard_advantages(
                [value["return_to_go"][round_index] for value in active]
            )
            for value, advantage in zip(active, report["advantages"]):
                by_trajectory[int(value["trajectory_index"])][round_index] = (
                    float(advantage),
                    True,
                )
        else:
            report = {
                "returns_to_go": [
                    float(value["return_to_go"][round_index]) for value in active
                ],
                "active_count": len(active),
                "mean": (
                    float(active[0]["return_to_go"][round_index])
                    if active
                    else None
                ),
                "population_std": None,
                "scale": None,
                "unclipped_advantages": [],
                "advantages": [],
                "zero_std": None,
                "minimum_std": MINIMUM_STD,
                "clamp": [-MAX_ABS_ADVANTAGE, MAX_ABS_ADVANTAGE],
                "reduction_dtype": "torch.float64",
                "detached": True,
                "hard_projection": False,
            }
            for value in active:
                by_trajectory[int(value["trajectory_index"])][round_index] = (
                    0.0,
                    False,
                )
        round_reports.append({"round_index": round_index, **report})

    records = []
    for row in ordered:
        trajectory_index = int(row["trajectory_index"])
        controller_advantages = []
        controller_active = []
        repair_flow_advantages = []
        repair_flow_active = []
        rounds = []
        for round_index, action in enumerate(row["actions"]):
            advantage, active = by_trajectory[trajectory_index][round_index]
            controller_advantages.append(advantage)
            controller_active.append(active)
            if action == "edit":
                repair_flow_advantages.append(advantage)
                repair_flow_active.append(active)
            rounds.append(
                {
                    **dict(row["rounds"][round_index]),
                    "group_mean": round_reports[round_index]["mean"],
                    "group_population_std": round_reports[round_index][
                        "population_std"
                    ],
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
        raise RuntimeError("G009 active advantage is nonfinite or unclamped")
    for record in records:
        flow_position = 0
        for action, controller_value, controller_active in zip(
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
                != controller_active
            ):
                raise AssertionError("G009 EDIT controller/flow credit differs")
            flow_position += 1

    return {
        "version": ADVANTAGE_VERSION,
        "group_identity": {
            "prompt": next(iter(identities))[0],
            "uid": next(iter(identities))[1],
            "r0_sha256": next(iter(identities))[2],
            "target_count": next(iter(identities))[3],
        },
        "trajectory_count": len(records),
        "records": records,
        "round_diagnostics": round_reports,
        "r0_policy_active_count": 0,
        "r0_text_record_count": 0,
        "r0_flow_record_count": 0,
        "r0_gradient_contribution": 0.0,
        "trajectory_level_advantage_broadcast": False,
        "hard_projection": False,
        "max_abs_active_advantage": max(
            (abs(value) for value in active_advantages), default=0.0
        ),
    }


def validate_reward_ordering() -> dict[str, Any]:
    """Executable checks for the frozen reward's required orderings."""

    no_op = edit_reward(1.0, 1.0)
    # Smallest positive one-count q delta occurs between adjacent non-exact
    # counts at target six: q(4)=1/3 and q(5)=5/12, a delta of 1/12.
    minimum_improvement = edit_reward(
        counting_score(6, 4), counting_score(6, 5)
    )
    immediate_exact_done = terminal_reward(
        parse_valid=True, strict_exact_success=True
    )
    exact_noop_then_done = no_op + immediate_exact_done
    # The largest nonexact terminal score is q=0.5.  From a common R0, the
    # edit shaping telescopes, so even one edit (least cost) maximizes it.
    maximum_nonexact_from_q0_zero = edit_reward(0.0, 0.5)
    minimum_exact_from_q0_zero = edit_reward(0.0, 1.0) + immediate_exact_done
    checks = {
        "no_op_edit_reward_exact": abs(no_op - (-0.01)) <= EPSILON,
        "minimum_one_count_improvement_target6_positive": minimum_improvement > 0.0,
        "immediate_exact_done_exceeds_noop_then_done_by_0_01": abs(
            immediate_exact_done - exact_noop_then_done - 0.01
        )
        <= EPSILON,
        "three_edit_nonexact_cannot_outscore_exact_same_r0": (
            maximum_nonexact_from_q0_zero < minimum_exact_from_q0_zero
        ),
        "failed_terminal_exact_zero": terminal_reward(
            parse_valid=True, strict_exact_success=False
        )
        == 0.0,
        "parse_invalid_exact_terminal_zero": terminal_reward(
            parse_valid=False, strict_exact_success=True
        )
        == 0.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "no_op_edit_reward": no_op,
        "minimum_one_count_improvement_target6": minimum_improvement,
        "immediate_exact_done": immediate_exact_done,
        "exact_noop_then_done": exact_noop_then_done,
        "maximum_nonexact_from_q0_zero": maximum_nonexact_from_q0_zero,
        "minimum_exact_from_q0_zero": minimum_exact_from_q0_zero,
    }


def validate_no_positive_detector_failed_done(
    credits: Sequence[Mapping[str, Any]],
) -> int:
    """A failed terminal must never receive the unit terminal anchor."""

    bad = [
        row
        for row in credits
        if not bool(row.get("terminal_strict_exact_success"))
        and abs(float(row.get("terminal_reward", float("nan")))) > EPSILON
    ]
    if bad:
        raise RuntimeError("G009 detector-failed terminal received reward")
    return 0


def validate_cap_terminal_does_not_underprice_done(
    credits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """A failed-R0 to exact-final repair must beat immediate failed DONE."""

    repaired = [
        row
        for row in credits
        if float(row["counting_scores"][0]) < 1.0
        and bool(row.get("terminal_strict_exact_success"))
    ]
    underpriced = [
        row
        for row in repaired
        if float(row["trajectory_return_report_only"]) <= 0.0
    ]
    if underpriced:
        raise RuntimeError("G009 exact repair is underpriced versus failed DONE")
    return {
        "repaired_success_count": len(repaired),
        "minimum_repaired_success_return": (
            min(float(row["trajectory_return_report_only"]) for row in repaired)
            if repaired
            else None
        ),
        "immediate_failed_done_return": 0.0,
        "passed": True,
    }


__all__ = [
    "ADVANTAGE_VERSION",
    "EDIT_COST",
    "EDIT_PROGRESS_COEFFICIENT",
    "GROUP_SIZE",
    "MAX_ABS_ADVANTAGE",
    "MAX_EDIT_ROUNDS",
    "MINIMUM_STD",
    "TERMINAL_FAILURE",
    "TERMINAL_SUCCESS",
    "VERSION",
    "assign_round_causal_advantages",
    "build_trajectory_process_credit",
    "build_verification_contract",
    "counting_score",
    "edit_reward",
    "image_state_sha256",
    "round_standard_advantages",
    "terminal_reward",
    "validate_cap_terminal_does_not_underprice_done",
    "validate_no_positive_detector_failed_done",
    "validate_reward_ordering",
]
