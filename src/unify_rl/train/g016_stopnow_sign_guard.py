"""Contrast-aware absolute-sign monitoring for G016 action-head credit."""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

VERSION = "clean29529_g016_contrast_aware_stopnow_sign_guard_v2"
_EPS = 1e-12
_CATEGORIES = (
    "already_exact_qneutral_edit",
    "already_exact_done",
    "broken_false_done",
    "broken_improving_edit",
)


def _bucket_key(round_row: Mapping[str, Any], round_index: int) -> tuple[int, bool]:
    raw = round_row.get("g016_bucket", [round_index, bool(round_row["exact_before_action"])])
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise RuntimeError("G016 sign guard bucket key is malformed")
    return int(raw[0]), bool(raw[1])


def classify_g016_stopnow_signs(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Classify sign categories without calling a zero-contrast tie an inversion.

    Broken-state EDIT is the correct action.  If a bucket contains no incorrect
    action comparator, leave its zero-centered sign check inactive.  Once a
    comparator exists, its mean must be strictly positive.  A negative active
    broken-state actionable-EDIT advantage is always a hard failure, regardless
    of the bucket mean or comparator count.
    """

    entries: dict[str, list[dict[str, Any]]] = {name: [] for name in _CATEGORIES}
    bucket_correct: dict[tuple[int, bool], list[bool]] = defaultdict(list)
    for record in records:
        actions = list(record["actions"])
        advantages = list(record["controller_advantages"])
        active = list(record["controller_active"])
        rounds = list(record["rounds"])
        if not (len(actions) == len(advantages) == len(active) == len(rounds)):
            raise RuntimeError("G016 sign guard round coverage differs")
        for round_index, (action, advantage, is_active, round_row) in enumerate(
            zip(actions, advantages, active, rounds)
        ):
            exact = bool(round_row["exact_before_action"])
            action_text = str(action).casefold()
            action_correct = action_text == ("done" if exact else "edit")
            key = _bucket_key(round_row, round_index)
            if bool(round_row.get("action_head_active", is_active)):
                bucket_correct[key].append(action_correct)
            if not is_active:
                continue
            q_delta = float(round_row["q_after"]) - float(round_row["q_before"])
            name = None
            if exact and action_text == "edit" and abs(q_delta) <= _EPS:
                name = "already_exact_qneutral_edit"
            elif exact and action_text == "done":
                name = "already_exact_done"
            elif not exact and action_text == "done":
                name = "broken_false_done"
            elif not exact and action_text == "edit" and q_delta > _EPS:
                name = "broken_improving_edit"
            if name is not None:
                entries[name].append(
                    {
                        "advantage": float(advantage),
                        "action_correct": action_correct,
                        "actionable_edit": bool(round_row.get("actionable_edit")),
                        "bucket": key,
                    }
                )

    output: dict[str, dict[str, Any]] = {}
    for name in _CATEGORIES:
        rows = entries[name]
        values = [row["advantage"] for row in rows]
        mean = statistics.mean(values) if values else None
        represented_buckets = {row["bucket"] for row in rows}
        contrast_count = sum(
            sum(not correct for correct in bucket_correct[key])
            for key in represented_buckets
        )
        negative_actionable_edit_count = sum(
            row["actionable_edit"]
            and row["action_correct"]
            and row["advantage"] < -_EPS
            for row in rows
        )
        if mean is None:
            sign_aligned = None
        elif name == "broken_improving_edit":
            if negative_actionable_edit_count:
                sign_aligned = False
            elif contrast_count == 0:
                sign_aligned = None
            else:
                sign_aligned = mean > 0.0
        elif name == "already_exact_qneutral_edit":
            sign_aligned = mean < 0.0
        elif name == "already_exact_done":
            sign_aligned = mean >= 0.0
        else:
            sign_aligned = mean <= 0.0
        output[name] = {
            "active_count": len(values),
            "mean_advantage": mean,
            "min_advantage": min(values) if values else None,
            "positive_advantage_count": sum(value > _EPS for value in values),
            "positive_advantage_rate": (
                sum(value > _EPS for value in values) / len(values)
                if values
                else None
            ),
            "action_correct_vector": [row["action_correct"] for row in rows],
            "contrast_count": contrast_count,
            "negative_actionable_edit_count": negative_actionable_edit_count,
            "sign_gate_active": sign_aligned is not None,
            "sign_aligned": sign_aligned,
            "version": VERSION,
        }
    return output


__all__ = ["VERSION", "classify_g016_stopnow_signs"]
