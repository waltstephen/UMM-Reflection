"""Strict one-response contract with the complete EDIT field relocated."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .existing_single_response_allmse_contract import (
    FULL_SCHEMA_FIELDS,
    validate_full_schema_controller,
)
from .existing_split_allmse_contract import ParsedController


SYSTEM_PROMPT_VERSION = "existing_exact_edit_relocation_v1"
SYSTEM_PROMPT = """Given the request and any available image context, assess the result.
Return exactly one assistant response using this schema:
<think>
[CURRENT_ROUND] Round#N
[SCORE] N/10
[ACTION] edit or done
[THINKING] Ground the decision in the visible image.
[SOURCE_IMAGE] given, None, or Image #N
</think>
[EDIT] executable payload or None
Place exactly one [EDIT] field immediately after </think>, never inside it.
For edit, use a nonempty executable payload. For done, use None."""

EDIT_FIELD_PREFIX = "[EDIT] "
_PROTOCOL_TEXT_RE = re.compile(
    r"(?i)(?:<think|</think|\[(?:CURRENT_ROUND|SCORE|ACTION|THINKING|"
    r"SOURCE_IMAGE|EDIT)\]|<\|[^>\r\n]+\|>)"
)
_NULL_PAYLOADS = {"", "none", "null", "n/a", "na"}
_CURRENT_ROUND_RE = re.compile(r"Round#(?P<index>\d+)")
_SCORE_RE = re.compile(r"(?P<score>\d+(?:\.\d+)?)/10")
GENERATED_PLACEHOLDER_PAYLOADS = frozenset(
    {
        "executable payload",
        "executable payload or none",
        "concrete generation instruction",
        "concrete edit instruction",
    }
)
GENERATED_PLACEHOLDER_THINKING = frozenset(
    {
        "ground the decision in the visible image.",
        (
            "ground the decision only in the request and supplied image "
            "context."
        ),
    }
)


@dataclass(frozen=True)
class ParsedRelocatedResponse:
    action: str
    controller: str
    edit_field: str
    payload: str
    raw_payload: str
    raw_response: str
    fields: dict[str, str]
    round_index: int
    score: float
    source_image: str


def _normalized_placeholder(value: str | None) -> str:
    return str(value or "").strip().casefold()


def is_generated_placeholder_payload(value: str | None) -> bool:
    return _normalized_placeholder(value) in GENERATED_PLACEHOLDER_PAYLOADS


def is_generated_placeholder_thinking(value: str | None) -> bool:
    return _normalized_placeholder(value) in GENERATED_PLACEHOLDER_THINKING


def _without_field_separator(edit_field: str) -> str:
    value = str(edit_field)
    if value.endswith("\r\n"):
        return value[:-2]
    if value.endswith("\n"):
        return value[:-1]
    return value


def relocated_edit_field(parsed: ParsedController) -> str:
    field = _without_field_separator(parsed.edit_field)
    if not field.startswith(EDIT_FIELD_PREFIX):
        raise ValueError("edit_field_prefix_not_exact")
    raw_payload = field[len(EDIT_FIELD_PREFIX) :]
    if parsed.action == "edit":
        if (
            not raw_payload
            or raw_payload.strip() != parsed.payload
        ):
            raise ValueError("edit_payload_bytes_mismatch")
        if _PROTOCOL_TEXT_RE.search(raw_payload):
            raise ValueError("edit_payload_contains_protocol_text")
    elif field != "[EDIT] None":
        raise ValueError("done_edit_field_not_exact_none")
    return field


def build_relocated_response(
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
    return f"{parsed.controller}\n{relocated_edit_field(parsed)}"


def parse_relocated_response(
    text: str,
    *,
    expected_round_index: int | None = None,
    expected_source_image: str | None = None,
    task: str | None = None,
) -> ParsedRelocatedResponse:
    raw = str(text or "")
    if (
        not raw.startswith("<think>")
        or raw.count("<think>") != 1
        or raw.count("</think>") != 1
    ):
        raise ValueError("malformed_relocated_response")
    close_end = raw.find("</think>") + len("</think>")
    controller = raw[:close_end]
    action, fields = validate_full_schema_controller(
        controller,
        expected_round_index=expected_round_index,
        expected_source_image=expected_source_image,
    )
    if is_generated_placeholder_thinking(fields["[THINKING]"]):
        raise ValueError("generated_thinking_placeholder")
    suffix = raw[close_end:]
    if not suffix.startswith("\n") or suffix.startswith("\n\n"):
        raise ValueError("external_edit_missing_immediately_after_think")
    edit_field = suffix[1:]
    if edit_field.count("[EDIT]") != 1:
        raise ValueError("external_edit_field_count")
    if not edit_field.startswith(EDIT_FIELD_PREFIX):
        raise ValueError("external_edit_prefix_not_exact")
    raw_payload = edit_field[len(EDIT_FIELD_PREFIX) :]
    if action == "edit":
        if (
            not raw_payload
            or raw_payload != raw_payload.strip()
            or raw_payload.lower() in _NULL_PAYLOADS
        ):
            raise ValueError("external_edit_payload_invalid")
        if _PROTOCOL_TEXT_RE.search(raw_payload):
            raise ValueError("external_edit_payload_protocol_suffix")
        if is_generated_placeholder_payload(raw_payload):
            raise ValueError("generated_edit_payload_placeholder")
        payload = raw_payload
    elif edit_field != "[EDIT] None":
        raise ValueError("done_external_edit_not_exact_none")
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

    return ParsedRelocatedResponse(
        action=action,
        controller=controller,
        edit_field=edit_field,
        payload=payload,
        raw_payload=raw_payload,
        raw_response=raw,
        fields=fields,
        round_index=round_index,
        score=score,
        source_image=source_image,
    )


__all__ = [
    "EDIT_FIELD_PREFIX",
    "FULL_SCHEMA_FIELDS",
    "GENERATED_PLACEHOLDER_PAYLOADS",
    "GENERATED_PLACEHOLDER_THINKING",
    "ParsedRelocatedResponse",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_VERSION",
    "build_relocated_response",
    "is_generated_placeholder_payload",
    "is_generated_placeholder_thinking",
    "parse_relocated_response",
    "relocated_edit_field",
]
