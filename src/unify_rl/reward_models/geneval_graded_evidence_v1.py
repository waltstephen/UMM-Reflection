"""Extract graded-score evidence from the frozen GenEval verifier results.

**Why this module does not lower the detector threshold.**  The obvious way to
grade a failing family is to look at sub-threshold detector confidences -- "the
dog is at 0.22, it is nearly there".  ``Mask2FormerGenevalDetector._detect_all``
refuses that outright::

    if threshold not in (DETECTOR_THRESHOLD, COUNTING_THRESHOLD):
        raise ValueError("unsupported frozen GenEval detector threshold")

That guard is what keeps every GenEval number in this repo reproducible, so it is
respected rather than widened.  Everything here is derived from what the frozen
verifier already returns at its own two legal thresholds:

* how many of the required objects were detected,
* the detected boxes, hence the continuous ``position`` margin,
* the colour backend's distribution, when it exposes one.

The cost of respecting the guard is that ``single_object`` -- a single presence
conjunct with nothing else to measure -- stays effectively binary.  It is 3% of
the training mix, has a hard ceiling of 80 prompts in the world, and is the
family flow_grpo drops from training altogether, so this is the cheapest of the
five to lose.  The other four all keep a usable gradient.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from unify_rl.reward_models.geneval_family_registry_v1 import COLOR_VQA_MARGIN
from unify_rl.reward_models.geneval_graded_family_score_v1 import (
    DETECTOR_THRESHOLD,
    color_attr_score,
    colors_score,
    position_margin,
    position_score,
    single_object_score,
    two_object_score,
)

EVIDENCE_VERSION = "clean29529_geneval_graded_evidence_v1"

#: Families this module grades.  ``counting`` is absent on purpose: it is
#: already graded by ``deterministic_count_v1`` and is frozen.
GRADED_FAMILIES = (
    "single_object",
    "two_object",
    "colors",
    "position",
    "color_attr",
)


def _detected_confidence(detections: Sequence[Mapping[str, Any]] | None) -> float:
    """Top confidence among detections the frozen verifier kept, else 0.0.

    A missing object contributes 0.0 rather than a sub-threshold confidence,
    because the frozen detector will not report one.
    """

    if not detections:
        return 0.0
    return max(float(row.get("score", 0.0)) for row in detections)


def _present(detections: Sequence[Mapping[str, Any]] | None) -> bool:
    return bool(detections)


def _expected_color_probability(diagnostics: Mapping[str, Any], expected: str) -> float:
    """Probability mass the colour backend put on the requested colour.

    ``ColorPrediction`` carries only the arg-max label and its confidence, so a
    full distribution is available only when the backend was extended to emit
    one.  Without it, a correct arg-max contributes its confidence and a wrong
    arg-max contributes nothing -- still monotone, just coarser.
    """

    distribution = diagnostics.get("color_probabilities")
    if isinstance(distribution, Mapping):
        value = distribution.get(expected)
        if value is not None:
            return min(1.0, max(0.0, float(value)))
    prediction = diagnostics.get("color_prediction") or {}
    if str(prediction.get("label") or "") == expected:
        return min(1.0, max(0.0, float(prediction.get("confidence", 0.0))))
    return 0.0


def _blip_margin_progress(row: Mapping[str, Any]) -> float | None:
    """How far the BLIP cross-check has travelled toward its pass margin."""

    margin = row.get("blip_margin")
    if margin is None:
        return None
    # The margin is a difference of two probabilities, so it lives in [-1,1],
    # and the row passes at COLOR_VQA_MARGIN (frozen at 0.0 -- "the expected
    # phrase merely has to beat the swapped one").  Progress is measured from
    # the worst possible margin up to that cut, which stays well defined when
    # the cut is zero.
    span = COLOR_VQA_MARGIN + 1.0
    if span <= 0.0:
        raise ValueError("BLIP colour margin threshold is out of range")
    return min(1.0, max(0.0, (float(margin) + 1.0) / span))


def _colour_evidence(row: Mapping[str, Any], expected: str) -> float:
    """Combine the colour signals a color_attr row actually reports.

    CLIP contributes its mass on the requested colour; BLIP contributes how far
    its margin has moved toward the cross-check threshold.  Both are monotone in
    "the object is closer to the requested colour", so their mean is too.
    """

    clip = _expected_color_probability(row, expected)
    blip = _blip_margin_progress(row)
    if blip is None:
        return clip
    return min(1.0, max(0.0, 0.5 * (clip + blip)))


def graded_score(
    *,
    family: str,
    passed: bool,
    diagnostics: Mapping[str, Any],
    context: Mapping[str, Any],
) -> float:
    """Graded q in [0,1] for one frozen ``VerifierResult``.

    ``passed`` is the frozen verifier's own verdict and stays the authority:
    the returned value is 1.0 exactly when it is set.
    """

    if not isinstance(passed, bool):
        raise ValueError("passed must be a bool")
    if family not in GRADED_FAMILIES:
        raise ValueError(f"{family!r} is not graded by this module")

    if family == "single_object":
        detections = diagnostics.get("detections")
        return single_object_score(
            passed=passed, max_confidence=_detected_confidence(detections)
        )

    if family == "two_object":
        names = list(context["objects"])
        per_class = diagnostics.get("detections") or {}
        confidences, present = [], []
        for name in names:
            rows = per_class.get(name)
            confidences.append(_detected_confidence(rows))
            present.append(_present(rows))
        return two_object_score(
            passed=passed, confidences=tuple(confidences), present=tuple(present)
        )

    if family == "colors":
        detection = diagnostics.get("detection")
        return colors_score(
            passed=passed,
            max_confidence=float(detection["score"]) if detection else 0.0,
            expected_color_probability=_expected_color_probability(
                diagnostics, str(context["color"])
            ),
        )

    if family == "position":
        subject = diagnostics.get("subject_detection")
        target = diagnostics.get("object_detection")
        margin = 0.0
        if subject and target:
            margin = position_margin(
                subject["box"], target["box"], str(context["relation"])
            )
        return position_score(
            passed=passed,
            confidences=(
                float(subject["score"]) if subject else 0.0,
                float(target["score"]) if target else 0.0,
            ),
            margin=margin,
        )

    # color_attr -- the frozen verifier reports its per-row detail under
    # "attributes", and each row carries a continuous "blip_margin" (expected
    # minus counterfactual yes-probability) that crosses COLOR_VQA_MARGIN on a
    # pass.  That margin is a second colour signal and is used alongside CLIP.
    reported = diagnostics.get("attributes") or []
    attributes = list(context["attributes"])
    if len(attributes) != 2:
        raise ValueError("color_attr requires exactly two attribute rows")
    rows = []
    for index, attribute in enumerate(attributes):
        row = reported[index] if index < len(reported) else {}
        detection = row.get("detection")
        rows.append((
            bool(row.get("passed", False)),
            float(detection["score"]) if detection else 0.0,
            _colour_evidence(row, str(attribute["color"])),
        ))
    return color_attr_score(passed=passed, rows=tuple(rows))


def assert_verdict_preserved(*, passed: bool, score: float) -> None:
    """Fail closed if grading ever disagrees with the frozen verdict.

    A graded q of 1.0 on a failing image would silently inflate the reported
    GenEval pass rate; a q below 1.0 on a passing image would deflate it.  Either
    makes every downstream number incomparable with the benchmark, so this is
    checked on every scored image rather than sampled.
    """

    if passed and score != 1.0:
        raise RuntimeError(
            f"graded score {score!r} contradicts a passing frozen verdict"
        )
    if not passed and score >= 0.5:
        raise RuntimeError(
            f"graded score {score!r} reaches the exact band on a failing verdict"
        )


__all__ = [
    "EVIDENCE_VERSION",
    "GRADED_FAMILIES",
    "DETECTOR_THRESHOLD",
    "assert_verdict_preserved",
    "graded_score",
]
