"""G024 protocol-precursor instrumentation.

Why this file exists, stated plainly: G023's protocol started degrading at step
11 and nobody could see it until the run was dead at step 50. The invalid rate
was visible the whole time -- 0.085 -> 0.183 -> 0.274 -> 0.562 -- but "invalid
rate rising" is a symptom that arrives late and says nothing about the cause.

What was actually happening was measurable from step 11 and was not measured:
the model kept emitting `[ACTION] edit` correctly (99% of invalid responses did)
and then failed to STOP -- extra `---` separators, hallucinated `<step>N</step>`
environment turns, self-generated Round#1/Round#2 blocks, all after `</think>`.
THINKING length and payload length were flat throughout. The corruption was
entirely in the stop protocol.

These are the counters that would have shown it, per step, from the first step:

  * `[ACTION]` per response and the multi-ACTION rate  (1.03 -> 1.72, 1.8% -> 22%)
  * `---` per response                                 (0.09 -> 12.4)
  * `<step>` tags and Round headers the model wrote itself
  * suffix length after `</think>` -- where 100% of the growth lived
  * THINKING and payload length, to show what did NOT grow

and, per Goal SS4, everything above ALSO split by advantage sign, because the
named risk of uniform credit is that corruption inside a positively-advantaged
trajectory gets reinforced until it becomes parse-invalid. If the positive
group's corruption rises, that risk is materialising and it is visible here
before the invalid rate moves.

Pure text in, numbers out. No model, no tokenizer, no torch -- so it runs in a
unit test on a laptop, which is why it can be trusted at 3am.
"""
from __future__ import annotations

import re
from statistics import mean
from typing import Any, Mapping, Sequence

VERSION = "clean29529_g024_protocol_precursors_v1"

_ACTION_RE = re.compile(r"\[ACTION\]", re.IGNORECASE)
_SEPARATOR_RE = re.compile(r"^[ \t]*-{3,}[ \t]*$", re.MULTILINE)
_STEP_TAG_RE = re.compile(r"<\s*step\s*>\s*\d+\s*<\s*/\s*step\s*>", re.IGNORECASE)
_ROUND_HEADER_RE = re.compile(r"Round\s*#?\s*\d+", re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<\s*think\s*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)
_SCORE_LINE_RE = re.compile(
    r"^[ \t]*\[SCORE\][ \t]*(?P<value>[^\r\n]*)$", re.IGNORECASE | re.MULTILINE
)
_SCORE_VALUE_RE = re.compile(r"^[ \t]*(?P<numerator>\d{1,2})[ \t]*/[ \t]*10[ \t]*$")
_SCORE_MAX = 10
_EDIT_PAYLOAD_RE = re.compile(
    r"\[EDIT\](?P<body>.*?)(?=\[[A-Z]+\]|$)", re.IGNORECASE | re.DOTALL
)


def response_precursors(text: str) -> dict[str, Any]:
    """Every per-response counter, from the raw generated text."""

    raw = str(text or "")
    think_close = list(_THINK_CLOSE_RE.finditer(raw))
    # The suffix is what follows the FIRST `</think>` -- everything the model
    # wrote after closing its own reasoning, which is where the stop protocol
    # lives and where 100% of G023's growth happened.
    #
    # Measuring from the LAST `</think>` instead looks equivalent and is not:
    # a self-written second thinking block puts the corruption BEFORE the last
    # close tag, so the very failure this counter exists to catch would be
    # hidden by it. A unit test caught that; the live run would not have.
    suffix = raw[think_close[0].end():] if think_close else raw
    thinking = raw[: think_close[0].start()] if think_close else ""
    payloads = [m.group("body") for m in _EDIT_PAYLOAD_RE.finditer(raw)]
    action_count = len(_ACTION_RE.findall(raw))
    return {
        "action_count": action_count,
        "multi_action": action_count > 1,
        "separator_count": len(_SEPARATOR_RE.findall(raw)),
        "step_tag_count": len(_STEP_TAG_RE.findall(raw)),
        "round_header_count": len(_ROUND_HEADER_RE.findall(raw)),
        "think_open_count": len(_THINK_OPEN_RE.findall(raw)),
        "think_close_count": len(think_close),
        # More than one open/close pair means the model wrote its own extra
        # thinking block -- one of the concrete G023 corruptions.
        "nested_think": len(_THINK_OPEN_RE.findall(raw)) > 1 or len(think_close) > 1,
        "suffix_chars_after_think": len(suffix),
        "thinking_chars": len(thinking),
        "payload_chars": sum(len(body) for body in payloads),
        "response_chars": len(raw),
        "score_line_count": len(_SCORE_LINE_RE.findall(raw)),
    }


def response_score(text: str) -> int | None:
    """The model's own `[SCORE] n/10`, or None when absent/malformed."""

    lines = list(_SCORE_LINE_RE.finditer(str(text or "")))
    if not lines:
        return None
    match = _SCORE_VALUE_RE.fullmatch(lines[-1].group("value"))
    if not match:
        return None
    value = int(match.group("numerator"))
    # `\d{1,2}` admits 11..99. Out of range is malformed, not a score, and must
    # be counted as such rather than skewing the mean and the histogram.
    return value if 0 <= value <= _SCORE_MAX else None


_MEAN_KEYS = (
    "action_count",
    "separator_count",
    "step_tag_count",
    "round_header_count",
    "suffix_chars_after_think",
    "thinking_chars",
    "payload_chars",
    "response_chars",
)
_RATE_KEYS = ("multi_action", "nested_think")


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        # Explicitly zero-count rather than absent: a missing key downstream
        # reads as "not measured", and an empty group is a fact worth seeing.
        return {"count": 0}
    out: dict[str, Any] = {"count": len(rows)}
    for key in _MEAN_KEYS:
        out[f"{key}_mean"] = float(mean(float(row[key]) for row in rows))
        out[f"{key}_max"] = float(max(float(row[key]) for row in rows))
    for key in _RATE_KEYS:
        out[f"{key}_rate"] = sum(bool(row[key]) for row in rows) / len(rows)
    return out


def g024_protocol_precursors(
    *,
    responses: Sequence[str],
    advantages: Sequence[float] | None = None,
    truncated: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Per-step protocol health, overall and split by advantage sign.

    `advantages` is the per-response trajectory advantage. The split is the
    whole point: uniform credit reinforces whatever sits inside a
    positively-advantaged trajectory, corruption included, right up until it
    becomes parse-invalid. Watch `positive_advantage.separator_count_mean` and
    `positive_advantage.multi_action_rate`. If those climb while the negative
    group's stay flat, G024's named risk is happening and the run should be
    stopped and looked at, not left to reach step 400.
    """

    texts = [str(value or "") for value in responses]
    rows = [response_precursors(text) for text in texts]
    scores = [response_score(text) for text in texts]

    report: dict[str, Any] = {
        "version": VERSION,
        "response_count": len(rows),
        "overall": _aggregate(rows),
    }

    if truncated is not None:
        flags = [bool(value) for value in truncated]
        report["max_token_cap_rate"] = (
            sum(flags) / len(flags) if flags else 0.0
        )
        report["max_token_cap_count"] = sum(flags)

    if advantages is not None and len(advantages) == len(rows):
        values = [float(value) for value in advantages]
        positive = [row for row, adv in zip(rows, values) if adv > 0.0]
        negative = [row for row, adv in zip(rows, values) if adv < 0.0]
        zero = [row for row, adv in zip(rows, values) if adv == 0.0]
        report["positive_advantage"] = _aggregate(positive)
        report["negative_advantage"] = _aggregate(negative)
        report["zero_advantage"] = _aggregate(zero)
        report["advantage_split_available"] = True
    else:
        # Say so rather than silently omitting the split -- a missing section
        # would read as "no corruption in the positive group".
        report["advantage_split_available"] = False

    present = [value for value in scores if value is not None]
    histogram = {str(n): present.count(n) for n in range(0, 11) if present.count(n)}
    report["score_distribution"] = {
        "reported_count": len(present),
        "missing_or_malformed_count": len(scores) - len(present),
        "histogram": histogram,
        "mean": float(mean(present)) if present else None,
        "distinct_values": len(set(present)),
        # The reward-hacking signature for crediting SCORE, named in Goal SS4:
        # the model discovers one value that correlates with reward and emits
        # only that. One distinct value across a whole step is the alarm.
        "collapsed_to_single_value": bool(present) and len(set(present)) == 1,
    }
    return report


__all__ = [
    "VERSION",
    "g024_protocol_precursors",
    "response_precursors",
    "response_score",
]
