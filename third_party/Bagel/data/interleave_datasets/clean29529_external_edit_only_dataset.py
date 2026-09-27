"""Clean29,529 reader that relocates the stored EDIT field at parse time."""

from __future__ import annotations

import re

from .unit_edit_dataset import FailawareTrajIterableDataset


OLD_SYSTEM_FORMAT_INSTRUCTION = (
    "At every reasoning turn, output exactly one <think> block using these exact tags: "
    "[CURRENT_ROUND], [SCORE], [ACTION], [THINKING], [SOURCE_IMAGE], and [EDIT]."
)
NEW_SYSTEM_FORMAT_INSTRUCTION = (
    "At every reasoning turn, output exactly one assistant response: use one <think> "
    "block with these exact tags: [CURRENT_ROUND], [SCORE], [ACTION], [THINKING], "
    "and [SOURCE_IMAGE], then place exactly one [EDIT] field immediately after "
    "</think> in the same response."
)
SYSTEM_PROMPT_VERSION = "clean29529_external_edit_only_aligned_v1"

_EMBEDDED_EDIT_RE = re.compile(
    r"\n(?P<field>\[EDIT\][^\r\n]*)(?P<separator>\n)(?P<close></think>)\Z"
)


class Clean29529ExternalEditOnlyIterableDataset(
    FailawareTrajIterableDataset
):
    """Reuse the legacy reader after an in-memory response-only transform."""

    @staticmethod
    def embedded_edit_field(text: str) -> str:
        value = str(text)
        match = _EMBEDDED_EDIT_RE.search(value)
        if match is None:
            raise ValueError("response_missing_terminal_embedded_edit_field")
        return match.group("field")

    @classmethod
    def relocate_response(cls, text: str) -> str:
        value = str(text)
        if not value:
            return value
        if (
            value.count("<think>") != 1
            or value.count("</think>") != 1
            or value.count("[EDIT]") != 1
        ):
            raise ValueError("response_schema_count_mismatch")
        match = _EMBEDDED_EDIT_RE.search(value)
        if match is None:
            raise ValueError("response_edit_field_not_immediately_before_close")
        field = match.group("field")
        relocated = (
            value[: match.start()]
            + "\n"
            + match.group("close")
            + "\n"
            + field
        )
        if (
            relocated.count("[EDIT]") != 1
            or "[EDIT]" in relocated.split("</think>", 1)[0]
            or not relocated.endswith(field)
            or field.encode("utf-8")
            != relocated.rsplit("\n", 1)[-1].encode("utf-8")
        ):
            raise ValueError("response_relocation_invariant_failed")
        return relocated

    @staticmethod
    def rewrite_system_prompt(system_prompt: str) -> str:
        value = str(system_prompt)
        if value.count(OLD_SYSTEM_FORMAT_INSTRUCTION) != 1:
            raise ValueError("system_prompt_format_instruction_mismatch")
        return value.replace(
            OLD_SYSTEM_FORMAT_INSTRUCTION,
            NEW_SYSTEM_FORMAT_INSTRUCTION,
            1,
        )

    @classmethod
    def transform_row(cls, row):
        transformed = dict(row)
        transformed["think_list"] = [
            cls.relocate_response(item) if str(item) else str(item)
            for item in list(row["think_list"])
        ]
        transformed["system_prompt"] = cls.rewrite_system_prompt(
            row.get("system_prompt", "")
        )
        return transformed

    def parse_row(self, row):
        return super().parse_row(self.transform_row(row))


__all__ = [
    "Clean29529ExternalEditOnlyIterableDataset",
    "NEW_SYSTEM_FORMAT_INSTRUCTION",
    "OLD_SYSTEM_FORMAT_INSTRUCTION",
    "SYSTEM_PROMPT_VERSION",
]
