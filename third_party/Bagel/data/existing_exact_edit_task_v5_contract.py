"""No-source-valid task prompt for exact external EDIT relocation."""

from __future__ import annotations

from .existing_exact_edit_relocation_contract import SYSTEM_PROMPT


SYSTEM_PROMPT_VERSION = "existing_exact_edit_relocation_task_v5_v1"
RESPONSE_INVARIANTS = (
    "In every assistant response, emit all five fields in the schema order. "
    "The [SOURCE_IMAGE] field is mandatory and must be the final field "
    "inside <think>, immediately before </think>. Keep [THINKING] finite "
    "and non-repetitive; never repeat a sentence to fill space. Later "
    "responses must assess the available generated-image context normally "
    "and may choose edit or done. A done response must end exactly with "
    "</think> followed by a single newline and [EDIT] None, with no "
    "explanatory suffix."
)
T2I_SUFFIX = (
    "This is a valid text-to-image request. No source image is expected "
    "before the first assistant response; that absence is normal and is "
    "not an error. Never call the request invalid, never wait for an image, "
    "and never repeat that an image has not been generated. The following "
    "state rule applies only to the first assistant response: use "
    "[CURRENT_ROUND] Round#0, [ACTION] edit, [SOURCE_IMAGE] None, concise "
    "generation planning in [THINKING], and a nonempty concrete generation "
    "instruction in the external [EDIT] field. "
    f"{RESPONSE_INVARIANTS}"
)
EDIT_SUFFIX = (
    "A source image appears before the request. The following state rule "
    "applies only to the first assistant response: use [ACTION] edit, "
    "[SOURCE_IMAGE] given, a nonempty concrete edit instruction in the "
    "external [EDIT] field, and preserve the learned family-specific "
    "round label. "
    f"{RESPONSE_INVARIANTS}"
)
TASK_SUFFIXES = {
    "t2i": T2I_SUFFIX,
    "edit": EDIT_SUFFIX,
}


def system_prompt_for_task(task: str) -> str:
    normalized = str(task).lower()
    if normalized not in TASK_SUFFIXES:
        raise ValueError(f"unknown_task:{task}")
    return f"{SYSTEM_PROMPT}\n\n{TASK_SUFFIXES[normalized]}"


__all__ = [
    "EDIT_SUFFIX",
    "RESPONSE_INVARIANTS",
    "SYSTEM_PROMPT_VERSION",
    "T2I_SUFFIX",
    "TASK_SUFFIXES",
    "system_prompt_for_task",
]
