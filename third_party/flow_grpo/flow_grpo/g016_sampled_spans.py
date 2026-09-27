"""Exact generation-time field spans for G016 sampled controller tokens."""
from __future__ import annotations

import re
from typing import Any, Sequence

VERSION = "clean29529_g016_generation_time_sampled_field_spans_v2"
# The strict V20/V22 parser still accepts only ``[ACTION]``/``[EDIT]``.  Span
# capture also recognizes the observed caret-prefixed near miss so a
# strict-invalid response can route bounded invalid-action credit to the exact
# sampled edit/done value without post-hoc tokenization or thinking broadcast.
_FIELD = re.compile(
    r"(?im)^\s*\[(?P<caret>\^?)(?P<field>ACTION|EDIT)\]\s*"
)


def _token_span_for_chars(
    tokenizer: Any,
    ids: list[int],
    start: int,
    stop: int,
) -> tuple[int, int]:
    if not 0 <= start < stop:
        raise RuntimeError("G016 sampled character span is empty")
    full = tokenizer.decode(ids)
    boundaries: dict[int, int] = {0: 0}
    for index in range(1, len(ids) + 1):
        prefix = tokenizer.decode(ids[:index])
        if full.startswith(prefix):
            boundaries[len(prefix)] = index
    starts = [
        (characters, index)
        for characters, index in boundaries.items()
        if characters <= start
    ]
    stops = [
        (characters, index)
        for characters, index in boundaries.items()
        if characters >= stop
    ]
    if not starts or not stops:
        raise RuntimeError(
            "G016 sampled span has no exact token boundary enclosure"
        )
    begin = max(starts)[1]
    end = min(stops)[1]
    if (
        begin >= end
        or tokenizer.decode(ids[begin:end]).strip()
        != full[start:stop].strip()
    ):
        raise RuntimeError("G016 sampled span/token lineage is ambiguous")
    return begin, end


def capture_sampled_field_spans(
    *,
    tokenizer: Any,
    sampled_token_ids: Sequence[int],
) -> dict[str, Any]:
    ids = [int(value) for value in sampled_token_ids]
    text = tokenizer.decode(ids)
    matches = list(_FIELD.finditer(text))
    action_markers = [
        match
        for match in matches
        if match.group("field").upper() == "ACTION"
    ]
    edit_markers = [
        match
        for match in matches
        if match.group("field").upper() == "EDIT"
    ]
    if not action_markers:
        return {
            "version": VERSION,
            "status": "unparseable",
            "decoded_sampled": text,
            "sampled_token_ids": ids,
        }
    # The sampled action is the first protocol action-like field. Malformed
    # model text may quote or append another marker in THINKING; choosing the
    # final marker breaks exact sampled action-token lineage.
    action_marker = action_markers[0]
    edit_marker = next(
        (
            match
            for match in reversed(edit_markers)
            if match.start() > action_marker.end()
        ),
        None,
    )
    # The action value is exactly the remainder of the first ACTION line.
    # Valid values stay literal edit/done. A non-empty natural-language value
    # remains protocol-invalid but still has exact sampled-token lineage for
    # bounded invalid-action credit.
    action_region_stop = text.find("\n", action_marker.end())
    if action_region_stop < 0:
        action_region_stop = len(text)
    action_region = text[action_marker.end() : action_region_stop]
    leading = len(action_region) - len(action_region.lstrip())
    stripped = action_region.strip()
    if not stripped or (
        action_marker.group("caret") != ""
        and stripped.casefold() not in {"edit", "done"}
    ):
        return {
            "version": VERSION,
            "status": "unparseable",
            "decoded_sampled": text,
            "sampled_token_ids": ids,
        }
    action_start = action_marker.end() + leading
    action_stop = action_start + len(stripped)
    action_span = _token_span_for_chars(
        tokenizer,
        ids,
        action_start,
        action_stop,
    )
    common = {
        "version": VERSION,
        "decoded_sampled": text,
        "sampled_token_ids": ids,
        "action_value": text[action_start:action_stop].strip().lower(),
        "action_token_span": list(action_span),
        "action_token_ids": ids[action_span[0] : action_span[1]],
        "action_marker_strict": action_marker.group("caret") == "",
        "action_marker_kind": (
            "strict" if action_marker.group("caret") == "" else "caret_near_miss"
        ),
        "capture_stage": (
            "immediately_after_generation_before_parse_or_canonical_replay"
        ),
        "posthoc_retokenization_used": False,
    }
    if edit_marker is None:
        return {
            **common,
            "status": "action_only",
            "payload_value": "",
            "payload_token_span": None,
            "payload_token_ids": [],
        }
    payload_start = edit_marker.end()
    while payload_start < len(text) and text[payload_start].isspace():
        payload_start += 1
    payload_stop = len(text)
    while payload_stop > payload_start and text[payload_stop - 1].isspace():
        payload_stop -= 1
    payload_span = (
        None
        if payload_start == payload_stop
        else _token_span_for_chars(
            tokenizer,
            ids,
            payload_start,
            payload_stop,
        )
    )
    return {
        **common,
        "status": "captured",
        "payload_value": text[payload_start:payload_stop],
        "payload_token_span": (
            None if payload_span is None else list(payload_span)
        ),
        "payload_token_ids": (
            []
            if payload_span is None
            else ids[payload_span[0] : payload_span[1]]
        ),
        "edit_marker_strict": edit_marker.group("caret") == "",
    }
