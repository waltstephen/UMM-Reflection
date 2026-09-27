"""Trajectory credit for the F01 full-GenEval campaign.

``g009_counting_process_reward.build_trajectory_process_credit`` is frozen and
hash-pinned by the trainer (``FROZEN_G009_REWARD_SHA256``), so it cannot be
edited.  Exactly two of its lines are counting-specific::

    scores = [counting_score(target_count, value) for value in counts]
    strict_exact = counts[-1] == int(target_count)

Everything downstream -- process rewards, policy round rewards, reward events,
returns-to-go, the round table -- depends only on ``scores``.  This module is a
faithful port that takes the quality sequence directly, so the other five
GenEval families can supply their graded q instead of a detected count.

``counting`` never comes through here: it keeps routing to the frozen builder so
its numbers stay byte-identical with G022-G026.  ``test_port_matches_the_frozen_
builder_on_counting`` drives this port with counting's own q values and asserts
field-for-field equality, which is what makes the port trustworthy.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from unify_rl.reward_models.g009_counting_process_reward import (
    EPSILON,
    MAX_EDIT_ROUNDS,
    _normalize_actions,
    edit_reward,
    terminal_reward,
)

VERSION = "clean29529_f01_geneval_trajectory_credit_v1"
ADVANTAGE_VERSION = "clean29529_g022_whole_trajectory_grpo_v1"

#: Families this builder serves.  counting is deliberately absent.
F01_FAMILIES = ("single_object", "two_object", "colors", "position", "color_attr")


def build_f01_trajectory_credit(
    *,
    family: str,
    prompt: str,
    uid: str,
    trajectory_index: int,
    r0_sha256: str,
    quality_scores: Sequence[Any],
    strict_exact: bool,
    actions: Sequence[Any],
    siblings_per_root: int,
    max_edit_rounds: int = MAX_EDIT_ROUNDS,
    allow_counting: bool = False,
    raw_controller_responses: Sequence[str] | None = None,
    canonical_controller_responses: Sequence[str] | None = None,
    parse_valid: bool = True,
    stop_reason: str | None = None,
    image_state_sha256s: Sequence[str] | None = None,
    image_states: Sequence[Mapping[str, Any]] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """One F01 trajectory's credit record, driven by graded quality scores."""

    if str(family) not in F01_FAMILIES and not (
        bool(allow_counting) and str(family) == "counting"
    ):
        raise ValueError(
            f"F01 credit builder refuses family {family!r}; counting must use "
            "the frozen G009 builder"
        )
    if not prompt or not uid or len(str(r0_sha256)) != 64:
        raise ValueError("F01 trajectory identity is incomplete")
    index = int(trajectory_index)
    if index < 0:
        raise ValueError("F01 trajectory index must be non-negative")
    siblings = int(siblings_per_root)
    if siblings <= 0:
        raise ValueError("F01 sibling group must be positive")
    if not isinstance(strict_exact, bool):
        raise ValueError("F01 strict_exact must be a bool")

    cap = int(max_edit_rounds)
    if cap <= 0:
        raise ValueError("F01 maximum EDIT rounds must be positive")
    if cap == MAX_EDIT_ROUNDS:
        normalized_actions = _normalize_actions(actions)
    else:
        normalized_actions = [str(value).strip().casefold() for value in actions]
        if not normalized_actions or any(
            value not in {"edit", "done", "invalid"}
            for value in normalized_actions
        ):
            raise ValueError("F01 controller action topology is invalid")
        if len(normalized_actions) > cap + 1:
            raise ValueError("F01 trajectory exceeds its action horizon")
        terminals = [
            index
            for index, action in enumerate(normalized_actions)
            if action != "edit"
        ]
        if len(terminals) > 1 or (
            terminals and terminals[0] != len(normalized_actions) - 1
        ):
            raise ValueError(
                "the first DONE/invalid action must terminate generation"
            )
        if sum(action == "edit" for action in normalized_actions) > cap:
            raise ValueError(
                f"F01 trajectory exceeds {cap} executed EDITs"
            )
    scores = [float(value) for value in quality_scores]
    if not scores or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in scores
    ):
        raise ValueError("F01 quality scores must be finite and in [0,1]")
    edit_count = sum(action == "edit" for action in normalized_actions)
    if len(scores) != edit_count + 1:
        raise ValueError("F01 EDIT/quality-state coverage differs")
    if normalized_actions[-1] == "edit" and edit_count != cap:
        raise ValueError(
            f"missing DONE is terminal only after the {cap}th EDIT"
        )
    if strict_exact != (scores[-1] == 1.0):
        raise ValueError(
            "F01 strict_exact must agree with a terminal quality of exactly 1.0"
        )

    hashes = [] if image_state_sha256s is None else [str(v) for v in image_state_sha256s]
    if hashes:
        if len(hashes) != len(scores):
            raise ValueError("F01 image-state hash coverage differs")
        if hashes[0] != str(r0_sha256):
            raise ValueError("F01 trajectory changed its detached R0 identity")
    states = [] if image_states is None else [dict(value) for value in image_states]
    if states and len(states) != len(scores):
        raise ValueError("F01 persisted image-state coverage differs")
    raw = None if raw_controller_responses is None else [str(v) for v in raw_controller_responses]
    canonical = (
        None if canonical_controller_responses is None
        else [str(v) for v in canonical_controller_responses]
    )

    process_rewards = [
        edit_reward(scores[position], scores[position + 1])
        for position in range(edit_count)
    ]
    parse_ok = bool(parse_valid) and "invalid" not in normalized_actions
    terminal = terminal_reward(parse_valid=parse_ok, strict_exact_success=strict_exact)

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
        reward_events.append({
            "event_index": len(reward_events),
            "event_kind": "terminal_cap",
            "policy_round_index": None,
            "reward": terminal,
        })

    returns_to_go = [0.0] * len(normalized_actions)
    running = terminal if terminal_event_is_separate else 0.0
    for round_index in range(len(normalized_actions) - 1, -1, -1):
        running += policy_round_rewards[round_index]
        returns_to_go[round_index] = float(running)
    trajectory_return = float(sum(process_rewards) + terminal)
    if not math.isfinite(trajectory_return) or any(
        not math.isfinite(value) for value in returns_to_go
    ):
        raise ValueError("F01 reward or return-to-go is nonfinite")
    if abs(returns_to_go[0] - trajectory_return) > EPSILON:
        raise AssertionError("F01 round-0 RTG differs from trajectory return")

    rounds = []
    edit_position = 0
    for round_index, action in enumerate(normalized_actions):
        flow_active = action == "edit"
        q_before = scores[edit_position] if flow_active else scores[-1]
        q_after = scores[edit_position + 1] if flow_active else scores[-1]
        rounds.append({
            "round_index": round_index,
            "action": action,
            "controller_policy_active": True,
            "repair_flow_policy_active": flow_active,
            "q_before": q_before,
            "q_after": q_after,
            "process_reward": process_rewards[edit_position] if flow_active else None,
            "terminal_reward": terminal if action != "edit" else None,
            "return_to_go": returns_to_go[round_index],
            "raw_controller_response": None if raw is None else raw[round_index],
            "canonical_controller_response": (
                None if canonical is None else canonical[round_index]
            ),
        })
        if flow_active:
            edit_position += 1

    # Stop-reason vocabulary and validation copied verbatim from the frozen
    # builder -- an equivalence test caught this drifting on the first attempt.
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
                f"F01 stop reason differs: {supplied!r} != {inferred_stop!r}"
            )
    stop = inferred_stop

    return {
        "version": VERSION,
        "advantage_version": ADVANTAGE_VERSION,
        "family": str(family),
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
        # `assign_g022_advantages` -- shared with the LIVE G025 run and therefore
        # not to be edited -- computes group identity and per-state exactness as
        # `int(detected_counts[i]) == int(target_count)`. Encoding exactness as
        # 1-of-1 makes that arithmetic correct for F01 by construction, with no
        # change to the shared module: the predicate becomes
        # `quality_scores[i] == 1.0`, which is exactly what exact means here.
        # The flag below exists so nothing can silently read these as object
        # counts; `quality_scores` carries the real signal.
        "target_count": 1,
        "detected_counts": [1 if value == 1.0 else 0 for value in scores],
        "detected_counts_are_exactness_flags": True,
        "counting_scores": scores,
        "quality_scores": scores,
        "image_state_sha256s": hashes,
        "image_states": states,
        "actions": normalized_actions,
        "raw_controller_responses": raw,
        "canonical_controller_responses": canonical,
        "parse_valid": parse_ok,
        "stop_reason": stop,
        "first_done_round": (
            normalized_actions.index("done") if "done" in normalized_actions else None
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
        "f01_frozen_g009_port": {
            "frozen_reward_file_edited": False,
            "counting_routed_to_frozen_builder": True,
            "siblings_per_root": siblings,
        },
    }


__all__ = ["VERSION", "F01_FAMILIES", "build_f01_trajectory_credit"]
