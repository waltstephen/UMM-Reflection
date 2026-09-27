"""Strict one-response contract for the 74,116-row all-MSE baseline."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .existing_split_allmse_contract import ParsedController


SYSTEM_PROMPT_VERSION = "existing_single_response_allmse_full_schema_v1"
SYSTEM_PROMPT = """Inspect the current image against the request.
Return exactly one assistant response using this complete schema:
<think>
[CURRENT_ROUND] Round#N
[SCORE] N/10
[ACTION] edit or done
[THINKING] Ground the decision in the visible image and list what is complete,
pending, or incorrect.
[SOURCE_IMAGE] given, None, or Image #N
</think>
Never emit [EDIT]. If [ACTION] is edit, immediately after </think> on the next
line emit only the executable natural edit instruction in this same assistant
response. If [ACTION] is done, end the response at </think> with no suffix."""

FULL_SCHEMA_FIELDS = (
    "[CURRENT_ROUND]",
    "[SCORE]",
    "[ACTION]",
    "[THINKING]",
    "[SOURCE_IMAGE]",
)
_FIELD_START_RE = re.compile(
    r"^[ \t]*(?P<field>\[(?:CURRENT_ROUND|SCORE|ACTION|THINKING|SOURCE_IMAGE)\])"
    r"[ \t]*(?P<inline>[^\r\n]*)$",
    re.MULTILINE,
)
_ANY_FIELD_RE = re.compile(
    r"^[ \t]*(?P<field>\[[A-Z][A-Z _-]*\])",
    re.MULTILINE,
)
_PROTOCOL_SUFFIX_RE = re.compile(
    r"(?i)(?:<think|</think|\[(?:CURRENT_ROUND|SCORE|ACTION|THINKING|"
    r"SOURCE_IMAGE|EDIT)\]|<\|[^>\r\n]+\|>)"
)
_CURRENT_ROUND_RE = re.compile(r"Round#(?P<index>\d+)")
_SCORE_RE = re.compile(r"(?P<score>\d+(?:\.\d+)?)/10")
_SOURCE_IMAGE_RE = re.compile(r"(?:given|None|Image #\d+)")


@dataclass(frozen=True)
class ParsedCombinedResponse:
    action: str
    controller: str
    payload: str
    raw_response: str
    fields: dict[str, str]
    round_index: int
    score: float
    source_image: str


def _field_values(value: str) -> dict[str, str]:
    matches = list(_FIELD_START_RE.finditer(value))
    fields = {}
    for index, match in enumerate(matches):
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else value.rfind("</think>")
        )
        continuation = value[match.end() : end]
        field_value = (
            match.group("inline") + continuation
        ).strip()
        fields[match.group("field").upper()] = field_value
    return fields


def validate_full_schema_controller(
    controller: str,
    *,
    expected_round_index: int | None = None,
    expected_source_image: str | None = None,
) -> tuple[str, dict[str, str]]:
    value = str(controller or "")
    if (
        not value.startswith("<think>")
        or not value.endswith("</think>")
        or value.count("<think>") != 1
        or value.count("</think>") != 1
        or re.search(r"\[EDIT\]", value, flags=re.IGNORECASE)
    ):
        raise ValueError("malformed_full_schema_controller")

    all_fields = [
        match.group("field").upper()
        for match in _ANY_FIELD_RE.finditer(value)
    ]
    if all_fields != list(FULL_SCHEMA_FIELDS):
        raise ValueError(f"full_schema_field_order:{all_fields}")

    fields = _field_values(value)
    if list(fields) != list(FULL_SCHEMA_FIELDS):
        raise ValueError(f"full_schema_field_values:{list(fields)}")
    for field in (
        "[CURRENT_ROUND]",
        "[SCORE]",
        "[ACTION]",
        "[SOURCE_IMAGE]",
    ):
        if "\n" in fields[field] or "\r" in fields[field]:
            raise ValueError(f"full_schema_multiline_context_field:{field}")
    if not fields["[THINKING]"]:
        raise ValueError("empty_thinking")

    round_match = _CURRENT_ROUND_RE.fullmatch(fields["[CURRENT_ROUND]"])
    if round_match is None:
        raise ValueError("current_round_syntax")
    round_index = int(round_match.group("index"))
    if (
        expected_round_index is not None
        and round_index != int(expected_round_index)
    ):
        raise ValueError(
            f"current_round_index:{round_index}!={expected_round_index}"
        )

    score_match = _SCORE_RE.fullmatch(fields["[SCORE]"])
    if score_match is None:
        raise ValueError("score_syntax")
    score = float(score_match.group("score"))
    if not 0.0 <= score <= 10.0:
        raise ValueError(f"score_range:{score}")

    source_image = fields["[SOURCE_IMAGE]"]
    if _SOURCE_IMAGE_RE.fullmatch(source_image) is None:
        raise ValueError("source_image_syntax")
    if (
        expected_source_image is not None
        and source_image != expected_source_image
    ):
        raise ValueError(
            f"source_image_state:{source_image}!={expected_source_image}"
        )

    action = fields["[ACTION]"].strip().lower()
    if action not in {"edit", "done"}:
        raise ValueError(f"unknown_action:{action}")
    return action, fields


def build_combined_response(
    parsed: ParsedController,
    *,
    expected_round_index: int | None = None,
    expected_source_image: str | None = None,
) -> str:
    action, _ = validate_full_schema_controller(
        parsed.controller,
        expected_round_index=expected_round_index,
        expected_source_image=expected_source_image,
    )
    if action != parsed.action:
        raise ValueError(
            f"parsed_action_mismatch:{parsed.action}!={action}"
        )
    if action == "edit":
        payload = str(parsed.payload)
        if not payload or payload != payload.strip():
            raise ValueError("edit_payload_not_canonical")
        if _PROTOCOL_SUFFIX_RE.search(payload):
            raise ValueError("edit_payload_contains_protocol_text")
        return f"{parsed.controller}\n{payload}"
    if parsed.payload:
        raise ValueError("done_action_has_payload")
    return parsed.controller


def parse_combined_response(
    text: str,
    *,
    expected_round_index: int | None = None,
    expected_source_image: str | None = None,
    task: str | None = None,
) -> ParsedCombinedResponse:
    raw = str(text or "")
    if not raw.startswith("<think>") or raw.count("</think>") != 1:
        raise ValueError("malformed_combined_response")
    close_end = raw.find("</think>") + len("</think>")
    controller = raw[:close_end]
    action, fields = validate_full_schema_controller(
        controller,
        expected_round_index=expected_round_index,
        expected_source_image=expected_source_image,
    )
    suffix = raw[close_end:]

    if action == "edit":
        if not suffix.startswith("\n") or suffix.startswith("\n\n"):
            raise ValueError("edit_response_missing_immediate_suffix")
        payload = suffix[1:]
        if not payload or payload != payload.strip():
            raise ValueError("edit_response_junk_suffix")
        if _PROTOCOL_SUFFIX_RE.search(payload):
            raise ValueError("edit_response_protocol_suffix")
    elif suffix:
        raise ValueError("done_response_has_suffix")
    else:
        payload = ""

    round_index = int(
        _CURRENT_ROUND_RE.fullmatch(
            fields["[CURRENT_ROUND]"]
        ).group("index")
    )
    score = float(
        _SCORE_RE.fullmatch(fields["[SCORE]"]).group("score")
    )
    source_image = fields["[SOURCE_IMAGE]"]
    if task == "t2i" and source_image == "given":
        raise ValueError("task_source_image_class:t2i:given")
    if task == "edit" and source_image == "None":
        raise ValueError("task_source_image_class:edit:None")
    if task not in {None, "t2i", "edit"}:
        raise ValueError(f"unknown_task:{task}")

    return ParsedCombinedResponse(
        action=action,
        controller=controller,
        payload=payload,
        raw_response=raw,
        fields=fields,
        round_index=round_index,
        score=score,
        source_image=source_image,
    )


def expected_legacy_training_context(
    *,
    source_family: str,
    task: str,
    logical_round: int,
) -> tuple[int, str]:
    if logical_round < 0:
        raise ValueError("logical_round must be nonnegative")
    if source_family == "reflection_t2i":
        return logical_round, "None"
    if source_family == "reflection_edit":
        return logical_round + 1, "given"
    if source_family != "ocr10k":
        raise ValueError(f"unknown legacy source family:{source_family}")
    if task == "t2i":
        source = (
            "None"
            if logical_round == 0
            else f"Image #{logical_round - 1}"
        )
    elif task == "edit":
        source = (
            "given"
            if logical_round == 0
            else f"Image #{logical_round}"
        )
    else:
        raise ValueError(f"unknown task:{task}")
    return logical_round, source
