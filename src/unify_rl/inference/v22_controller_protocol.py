"""V22 adapter for the frozen SFT/V20 controller protocol.

This module deliberately adds no policy-visible syntax.  Every forced-prefix,
score-trie, strict-parse, and canonical-replay operation delegates to the
existing V20 implementation.  The optional compatibility arguments exist only
so the state-grouped V21 host can call this adapter without exposing its former
budget/diagnosis fields.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from unify_rl.inference.v20_controller_protocol import (
    PROTOCOL_SCORE_SCALE,
    ScoreFieldSchedule,
    V20_FORCED_PREFIX_VERSION,
    V20_PROTOCOL_VERSION,
    V20_SCORE_FIELD_VERSION,
    V20_STRICT_PARSER_VERSION,
    V20_SYSTEM_PROMPT_PATH,
    v20_canonicalize_model_score_response,
    v20_forced_controller_prefix,
    v20_parse_controller_response,
    v20_protocol_contract,
    v20_score_field_continuations,
)

WRAPPER_VERSION = "clean29529_v22_sft_protocol_adapter_v1"
# These identities intentionally remain the frozen V20 identities.
VERSION = V20_PROTOCOL_VERSION
FORCED_PREFIX_VERSION = V20_FORCED_PREFIX_VERSION
SCORE_FIELD_VERSION = V20_SCORE_FIELD_VERSION
STRICT_PARSER_VERSION = V20_STRICT_PARSER_VERSION
SYSTEM_PROMPT_PATH = V20_SYSTEM_PROMPT_PATH
MAX_HOST_REPAIR_BUDGET = 3


def configure_host_repair_budget(value: int) -> None:
    """Change only the hidden host-side cap; policy-visible bytes stay V20."""
    global MAX_HOST_REPAIR_BUDGET
    value = int(value)
    if value < 1:
        raise ValueError("V22 host repair budget must be positive")
    MAX_HOST_REPAIR_BUDGET = value


def score_field_continuations() -> list[str]:
    return v20_score_field_continuations()


def forced_controller_prefix(
    round_index: int,
    remaining_edits: int | None = None,
) -> str:
    """Return the exact V20 prefix; budget is host-only and never rendered."""
    if (
        remaining_edits is not None
        and not 1 <= int(remaining_edits) <= MAX_HOST_REPAIR_BUDGET
    ):
        raise ValueError("V22 host repair budget is invalid")
    return v20_forced_controller_prefix(round_index)


def parse_controller_response(
    raw_response: str,
    *,
    stable_clause_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Strict V20 parse; hidden verifier clause IDs are intentionally ignored."""
    del stable_clause_ids
    return v20_parse_controller_response(raw_response)


def canonicalize_response(
    proposal: Mapping[str, Any],
    *,
    round_index: int,
    remaining_edits: int | None = None,
    expected_source_image: str,
) -> str:
    """Render the exact V20/SFT canonical replay bytes for a T2I repair turn."""
    if (
        remaining_edits is not None
        and not 1 <= int(remaining_edits) <= MAX_HOST_REPAIR_BUDGET
    ):
        raise ValueError("V22 host repair budget is invalid")
    return v20_canonicalize_model_score_response(
        dict(proposal),
        task="t2i",
        expected_round_index=int(round_index),
        expected_source_image=str(expected_source_image),
    )


def protocol_contract() -> dict[str, Any]:
    frozen = v20_protocol_contract()
    return {
        "wrapper_version": WRAPPER_VERSION,
        "delegates_to_frozen_v20": True,
        "policy_visible_contract": frozen,
        "new_policy_visible_fields": [],
        "diagnosis_visible": False,
        "remaining_edits_visible": False,
        "verifier_contract_visible": False,
    }


__all__ = [
    "FORCED_PREFIX_VERSION",
    "PROTOCOL_SCORE_SCALE",
    "SCORE_FIELD_VERSION",
    "STRICT_PARSER_VERSION",
    "SYSTEM_PROMPT_PATH",
    "ScoreFieldSchedule",
    "VERSION",
    "WRAPPER_VERSION",
    "canonicalize_response",
    "configure_host_repair_budget",
    "forced_controller_prefix",
    "parse_controller_response",
    "protocol_contract",
    "score_field_continuations",
]
