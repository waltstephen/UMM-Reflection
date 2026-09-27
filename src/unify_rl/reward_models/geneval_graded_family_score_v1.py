"""Graded GenEval family scores.

The frozen registry ships five of the six core families as
``1.0 on pass, else 0.0`` (:mod:`geneval_family_registry_v1`).  Counting is the
lone exception and the reason counting is the only family we have ever trained
multi-round reflection on: a binary ``q`` annihilates every process term in the
whole-trajectory reward.  A broken state has ``q = 0``; an edit that improves it
without fully fixing it still yields ``dq = 0``; so the reflection bonus, the
sustained-progress bonus and the regression penalty are all zero and the reward
collapses to terminal-only -- the regime measured not to teach repair.

This module supplies the missing gradient by copying counting's proven shape::

    exact   ->  q = 1.0
    partial ->  0 <= q < 0.5

The gap between ``1.0`` and the partial ceiling is deliberate and is counting's
own design: exactness stays qualitatively distinct from being close.

**The official verdict is never recomputed here.**  Every entry point takes the
frozen registry's ``passed`` flag as the authority and asserts
``q == 1.0`` exactly when it is set, so the GenEval pass rate -- the number the
benchmark reports -- is bit-identical with grading switched on.  Only the
failure region gains structure.
"""

from __future__ import annotations

import math
from typing import Sequence


GRADED_VERSION = "clean29529_geneval_graded_family_score_v1"

EXACT_SCORE = 1.0
PARTIAL_CEILING = 0.5

#: Official detector cut; a confidence at or above it is a detection.
DETECTOR_THRESHOLD = 0.3
#: Official ``relative_position`` cut on the normalized offset component.
POSITION_CUT = 0.5

#: Largest double strictly below the partial ceiling.  Partial credit is capped
#: here so that ``q == 1.0`` can only ever mean "the official predicate passed".
#:
#: ``math.nextafter`` is Python 3.9+.  The training nodes run 3.10, but the
#: control host -- where every checkpoint evaluation runs -- is 3.8, and this
#: module is imported transitively by ``config.g016``, so the 3.9-only spelling
#: made the whole evaluator unimportable there.  For a positive, normal double,
#: decrementing the IEEE-754 bit pattern is exactly one ULP toward zero, which
#: is what ``nextafter(x, 0.0)`` returns.
if hasattr(math, "nextafter"):  # Python 3.9+
    _PARTIAL_MAX = math.nextafter(PARTIAL_CEILING, 0.0)
else:  # Python 3.8
    import struct as _struct

    _PARTIAL_MAX = _struct.unpack(
        "<d", _struct.pack("<Q", _struct.unpack("<Q", _struct.pack("<d", PARTIAL_CEILING))[0] - 1)
    )[0]

# colors / color_attr: how the partial half is split between "the object is
# there at all" and "its colour is the requested one".
_PRESENCE_WEIGHT = 0.5
_COLOR_WEIGHT = 0.5

# position: presence dominates the margin, because two objects in frame is the
# precondition for any relation edit to be meaningful at all.
_POSITION_PRESENCE_WEIGHT = 0.6
_POSITION_MARGIN_WEIGHT = 0.4

RELATIONS = ("left of", "right of", "above", "below")


def _finite(value: object, *, name: str) -> float:
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _unit(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if not (0.0 <= result <= 1.0):
        raise ValueError(f"{name} must lie in [0,1]")
    return result


def _confidence(value: object, *, name: str) -> float:
    """Detector confidences are probabilities; reject anything else."""

    return _unit(value, name=name)


def _presence(confidence: float) -> float:
    """Ramp from "invisible" to "detected" over the official threshold.

    Reaching the threshold is presence 1.0; below it the score decays linearly
    so that an object the detector is beginning to see outranks an empty frame.
    """

    if DETECTOR_THRESHOLD <= 0.0:
        raise ValueError("detector threshold must be positive")
    return min(1.0, max(0.0, confidence / DETECTOR_THRESHOLD))


def _partial(fraction: float) -> float:
    """Map a satisfaction fraction in [0,1] into the partial band [0, 0.5)."""

    bounded = min(1.0, max(0.0, _finite(fraction, name="fraction")))
    return min(PARTIAL_CEILING * bounded, _PARTIAL_MAX)


def _graded(passed: bool, fraction: float) -> float:
    return EXACT_SCORE if passed else _partial(fraction)


def single_object_score(*, passed: object, max_confidence: object) -> float:
    """One conjunct: is the requested class visible at all?"""

    is_passed = _boolean(passed, name="passed")
    confidence = _confidence(max_confidence, name="max_confidence")
    return _graded(is_passed, _presence(confidence))


def two_object_score(
    *,
    passed: object,
    confidences: Sequence[object],
    present: Sequence[object],
) -> float:
    """Two independent presence conjuncts; partial credit is the mean."""

    is_passed = _boolean(passed, name="passed")
    if len(confidences) != 2 or len(present) != 2:
        raise ValueError("two_object requires exactly two objects")
    terms = []
    for index, (confidence, flag) in enumerate(zip(confidences, present)):
        value = _confidence(confidence, name=f"confidences[{index}]")
        seen = _boolean(flag, name=f"present[{index}]")
        terms.append(1.0 if seen else _presence(value))
    return _graded(is_passed, sum(terms) / 2.0)


def colors_score(
    *,
    passed: object,
    max_confidence: object,
    expected_color_probability: object,
) -> float:
    """Presence of the object, then the requested colour on it.

    The colour term is gated by presence: the colour of an object that is not in
    the frame must not earn credit, or the model can farm partial reward by
    painting the background.
    """

    is_passed = _boolean(passed, name="passed")
    confidence = _confidence(max_confidence, name="max_confidence")
    probability = _unit(
        expected_color_probability, name="expected_color_probability"
    )
    presence = _presence(confidence)
    fraction = _PRESENCE_WEIGHT * presence + _COLOR_WEIGHT * presence * probability
    return _graded(is_passed, fraction)


def position_score(
    *,
    passed: object,
    confidences: Sequence[object],
    margin: object,
) -> float:
    """Both objects in frame, then how far the arrangement has moved toward the
    requested relation.

    The margin is gated by the weaker of the two presences for the same reason
    the colour term is gated: a relation between one real and one imagined box
    is not progress.
    """

    is_passed = _boolean(passed, name="passed")
    if len(confidences) != 2:
        raise ValueError("position requires exactly two objects")
    presences = [
        _presence(_confidence(value, name=f"confidences[{index}]"))
        for index, value in enumerate(confidences)
    ]
    progress = _unit(margin, name="margin")
    fraction = (
        _POSITION_PRESENCE_WEIGHT * (sum(presences) / 2.0)
        + _POSITION_MARGIN_WEIGHT * min(presences) * progress
    )
    return _graded(is_passed, fraction)


def color_attr_score(
    *,
    passed: object,
    rows: Sequence[Sequence[object]],
) -> float:
    """Two ``(object, colour)`` assignments; partial credit is their mean.

    Each row that fully passes contributes 1.0; a row that does not falls back
    to the same presence/colour split :func:`colors_score` uses.
    """

    is_passed = _boolean(passed, name="passed")
    if len(rows) != 2:
        raise ValueError("color_attr requires exactly two attribute rows")
    terms = []
    for index, row in enumerate(rows):
        if len(row) != 3:
            raise ValueError(
                f"rows[{index}] must be (passed, max_confidence, probability)"
            )
        row_passed = _boolean(row[0], name=f"rows[{index}].passed")
        confidence = _confidence(row[1], name=f"rows[{index}].max_confidence")
        probability = _unit(row[2], name=f"rows[{index}].probability")
        if row_passed:
            terms.append(1.0)
            continue
        presence = _presence(confidence)
        terms.append(
            _PRESENCE_WEIGHT * presence
            + _COLOR_WEIGHT * presence * probability
        )
    return _graded(is_passed, sum(terms) / 2.0)


def position_margin(
    subject_box: Sequence[float],
    object_box: Sequence[float],
    relation: str,
) -> float:
    """Progress toward the official relation, in [0,1].

    Replicates :func:`geneval_family_registry_v1.relative_position` exactly --
    same threshold-shrunk offsets, same normalization -- and reports how far the
    relevant component has travelled toward the official ``0.5`` cut.  A value of
    ``1.0`` means the relation holds; ``0.0`` means the arrangement is neutral or
    reversed.
    """

    if relation not in RELATIONS:
        raise ValueError(f"unsupported position relation: {relation!r}")
    if len(subject_box) != 4 or len(object_box) != 4:
        raise ValueError("boxes must be (x1, y1, x2, y2)")
    ax1, ay1, ax2, ay2 = (_finite(v, name="subject_box") for v in subject_box)
    bx1, by1, bx2, by2 = (_finite(v, name="object_box") for v in object_box)

    center_a = ((ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0)
    center_b = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
    dim_a = (abs(ax2 - ax1), abs(ay2 - ay1))
    dim_b = (abs(bx2 - bx1), abs(by2 - by1))
    offset = (center_a[0] - center_b[0], center_a[1] - center_b[1])

    # POSITION_THRESHOLD is frozen at 0.1 in the official calculation.
    revised = []
    for axis in range(2):
        magnitude = max(
            abs(offset[axis]) - 0.1 * (dim_a[axis] + dim_b[axis]),
            0.0,
        )
        revised.append(math.copysign(magnitude, offset[axis]))
    if all(abs(value) < 1e-3 for value in revised):
        return 0.0
    norm = math.hypot(*offset)
    if norm <= 0.0:
        return 0.0
    dx, dy = (revised[0] / norm, revised[1] / norm)

    signed = {
        "left of": -dx,
        "right of": dx,
        "above": -dy,
        "below": dy,
    }[relation]
    return min(1.0, max(0.0, signed / POSITION_CUT))


__all__ = [
    "GRADED_VERSION",
    "EXACT_SCORE",
    "PARTIAL_CEILING",
    "DETECTOR_THRESHOLD",
    "POSITION_CUT",
    "RELATIONS",
    "color_attr_score",
    "colors_score",
    "position_margin",
    "position_score",
    "single_object_score",
    "two_object_score",
]
