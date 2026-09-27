"""V20 controller protocol: trie-constrained SCORE field, strict online parse.

Root cause (archived V19, 549 requests / 14,748 turn-0 events, 8,999 failures):
`raw_forced_prefix` is exactly

    "<think>\\n[CURRENT_ROUND] Round#N\\n[SCORE] "

40 characters, 15 tokens. Both the numerator *and* the denominator of the
self-assessment were free-sampled at temperature 0.5, and the denominator is a
protocol constant carrying zero information. Free sampling of it produced
`0/1` (4,135), `0/100` (645), bare `0` (531), `0/5` (338) and so on: 6,689 of
8,999 turn-0 failures, all well-formed self-assessments rejected by
`_SCORE_VALUE_RE` in `bagel_external_edit_payload_executor_v5.py`.

**P1**: constrained
decoding over the eleven complete legal continuations `0/10` .. `10/10` with
exact renormalization of the branch probabilities at every constrained
position. The model's 0..10 judgment is preserved and correctly renormalized;
only illegal branches are removed. The numerator is *not* free-sampled and the
denominator is *not* hard-inserted afterwards, because inserting would turn a
model that meant `3/6` into `3/10` and condition the downstream `[THINKING]`
and action on a token sequence the model considers unlikely.

Accounting:

* constrained positions stay in `sampled_token_ids` / `old_log_probs`;
* a position where the trie leaves exactly one legal token records log-prob
  `0.0` and `sampled_token_policy_active = False`;
* a branching position stays policy-active with the renormalized log-prob and
  carries `sampled_token_allowed_ids`, so old/current/reference log-probs use
  the identical constraint.

**P2**: `[ACTION] edit` with a null or placeholder `[EDIT]` payload is a
semantic DONE and is routed to DONE instead of killing the trajectory. The
rewrite is recorded on the event so the trainer can log `edit_none_alias_rate`.

**P3**: the online parser below is strict and fail-loud. Tolerant parsing lives
in `bagel_external_edit_payload_executor_v6.py` and is offline-diagnostic only.

`bagel_multiround_transition_v6.py` is the legacy V17/V19 path and is not
modified.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from unify_rl.inference.bagel_existing_split_allmse import validate_payload


def _relocation_contract():
    """Import the frozen relocation contract lazily.

    `v20_multiround` imports this module, and several CPU probes run without
    `third_party/Bagel` on `sys.path`. Keeping the frozen-contract import lazy
    keeps those probes importable while the online rollout, which always has
    the Bagel path, still validates against the same frozen parser.
    """
    from data import existing_exact_edit_relocation_contract

    return existing_exact_edit_relocation_contract


V20_PROTOCOL_VERSION = "clean29529_v20_trie_constrained_score_protocol_v1"
V20_FORCED_PREFIX_VERSION = "clean29529_v20_forced_round_score_prefix_v1"
V20_SCORE_FIELD_VERSION = "clean29529_v20_trie_constrained_score_field_v1"
V20_STRICT_PARSER_VERSION = "clean29529_v20_strict_online_proposal_v1"
PROTOCOL_SCORE_SCALE = 10
V20_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "assets/prompts/controller_system_prompt.txt"
)

_ACTION_MARKER_RE = re.compile(r"\[ACTION\]", re.IGNORECASE)
_ACTION_FIELD_RE = re.compile(
    r"\[ACTION\][ \t]*(?P<action>edit|done)"
    r"(?=[ \t]*(?:,|\r\n|\r|\n|\[[A-Z]|</think>))",
    re.IGNORECASE,
)
_THINKING_FIELD_RE = re.compile(
    r"\[THINKING\][ \t]*(?P<thinking>.*?)"
    r"(?=[ \t]*(?:\[SOURCE_IMAGE\]|</think>))",
    re.IGNORECASE | re.DOTALL,
)
_THINKING_MARKER_RE = re.compile(r"\[THINKING\][ \t]*", re.IGNORECASE)
_SCORE_LINE_RE = re.compile(
    r"^[ \t]*\[SCORE\][ \t]*(?P<value>[^\r\n]*)$",
    re.IGNORECASE | re.MULTILINE,
)
_STRICT_SCORE_VALUE_RE = re.compile(
    rf"^[ \t]*(?P<numerator>\d{{1,2}})[ \t]*/[ \t]*{PROTOCOL_SCORE_SCALE}"
    r"[ \t]*$"
)
_NULL_PAYLOADS = {"", "none", "null", "n/a", "na"}


def v20_forced_controller_prefix(round_index: int) -> str:
    """Host-owned round state; byte-identical to the archived V19 prefix."""
    if int(round_index) < 0:
        raise ValueError("forced prefix round must be non-negative")
    return (
        "<think>\n"
        f"[CURRENT_ROUND] Round#{int(round_index)}\n"
        "[SCORE] "
    )


def v20_score_field_continuations() -> list[str]:
    """The eleven complete legal score fields `0/10` .. `10/10`."""
    return [
        f"{value}/{PROTOCOL_SCORE_SCALE}"
        for value in range(PROTOCOL_SCORE_SCALE + 1)
    ]


class ScoreFieldSchedule:
    """Token trie over the eleven legal `<score>/10` continuations."""

    version = V20_SCORE_FIELD_VERSION

    def __init__(self, sequences: Sequence[Sequence[int]]):
        rows = [tuple(int(token) for token in row) for row in sequences]
        if len(rows) != PROTOCOL_SCORE_SCALE + 1:
            raise ValueError("V20 score schedule needs 0..10 continuations")
        if any(not row for row in rows):
            raise ValueError("V20 score schedule continuation is empty")
        if len(set(rows)) != len(rows):
            raise ValueError("V20 score schedule continuations collide")
        for index, row in enumerate(rows):
            for other in rows[index + 1 :]:
                shorter, longer = sorted((row, other), key=len)
                if longer[: len(shorter)] == shorter:
                    raise ValueError(
                        "V20 score schedule continuation is a prefix of "
                        "another continuation"
                    )
        self.sequences = rows
        self.score_by_sequence = {row: index for index, row in enumerate(rows)}
        self.max_length = max(len(row) for row in rows)

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> "ScoreFieldSchedule":
        sequences = []
        for text in v20_score_field_continuations():
            try:
                encoded = tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                encoded = tokenizer.encode(text)
            sequences.append([int(token) for token in encoded])
        return cls(sequences)

    def allowed_next(self, prefix: Iterable[int]) -> list[int]:
        """Sorted legal next tokens after `prefix`; empty when complete."""
        head = tuple(int(token) for token in prefix)
        allowed = {
            row[len(head)]
            for row in self.sequences
            if len(row) > len(head) and row[: len(head)] == head
        }
        return sorted(allowed)

    def is_complete(self, prefix: Iterable[int]) -> bool:
        return tuple(int(token) for token in prefix) in self.score_by_sequence

    def score_for(self, tokens: Iterable[int]) -> int:
        head = tuple(int(token) for token in tokens)
        if head not in self.score_by_sequence:
            raise ValueError("V20 score field tokens are not a legal score")
        return int(self.score_by_sequence[head])

    def nodes(self) -> dict[tuple[int, ...], int]:
        """Every reachable trie node mapped to its legal-continuation count."""
        found: dict[tuple[int, ...], int] = {}
        frontier: list[tuple[int, ...]] = [()]
        while frontier:
            head = frontier.pop()
            allowed = self.allowed_next(head)
            if not allowed:
                continue
            found[head] = len(allowed)
            for token in allowed:
                frontier.append((*head, token))
        return found

    def describe(self) -> dict[str, Any]:
        nodes = self.nodes()
        return {
            "version": self.version,
            "continuations": v20_score_field_continuations(),
            "sequences": [list(row) for row in self.sequences],
            "max_length": self.max_length,
            "policy_active_node_count": sum(
                1 for count in nodes.values() if count > 1
            ),
            "deterministic_node_count": sum(
                1 for count in nodes.values() if count == 1
            ),
            "root_allowed_token_ids": self.allowed_next(()),
        }


def v20_canonicalize_model_score_response(
    proposal: dict[str, Any],
    *,
    task: str,
    expected_round_index: int,
    expected_source_image: str,
) -> str:
    """Render the host-owned canonical response for KV replay."""
    from data.existing_exact_edit_relocation_contract import (
        parse_relocated_response,
    )

    action = str(proposal.get("action") or "").lower()
    thinking = str(proposal.get("thinking") or "").strip()
    payload = str(proposal.get("payload") or "")
    score = int(proposal.get("score"))
    if action not in {"edit", "done"}:
        raise ValueError("model-score canonical action is invalid")
    if not 0 <= score <= PROTOCOL_SCORE_SCALE:
        raise ValueError("model-score canonical score is invalid")
    if not thinking:
        raise ValueError("model-score canonical thinking is empty")
    if action == "edit" and not payload:
        raise ValueError("model-score canonical edit payload is empty")
    if action == "done" and payload:
        raise ValueError("model-score canonical done payload is not empty")
    edit_value = payload if action == "edit" else "None"
    response = (
        v20_forced_controller_prefix(expected_round_index)
        + f"{score}/{PROTOCOL_SCORE_SCALE}\n"
        f"[ACTION] {action}\n"
        f"[THINKING] {thinking}\n"
        f"[SOURCE_IMAGE] {expected_source_image}\n"
        "</think>\n"
        f"[EDIT] {edit_value}"
    )
    validated = parse_relocated_response(
        response,
        task=task,
        expected_round_index=expected_round_index,
        expected_source_image=expected_source_image,
    )
    if (
        validated.action != action
        or validated.payload != payload
        or int(validated.score) != score
    ):
        raise ValueError("model-score canonical response changed semantics")
    return response


def v20_normalize_strict_model_score(value: Any) -> int:
    """Strict protocol score. Only `n/10` is legal (P1 guarantees it)."""
    text = str(value if value is not None else "")
    match = _STRICT_SCORE_VALUE_RE.fullmatch(text)
    if match is None:
        raise ValueError("proposal_score_format")
    score = int(match.group("numerator"))
    if not 0 <= score <= PROTOCOL_SCORE_SCALE:
        raise ValueError("proposal_score_range")
    return score


def v20_parse_controller_response(raw_response: str) -> dict[str, Any]:
    """Strict online V20 proposal parse with the P2 EDIT-None DONE alias.

    Everything except the P2 alias is byte-for-byte the frozen V5/V4 contract:
    exact `<think>` topology, exactly one external `[EDIT]` field immediately
    after `</think>`, exactly one `[ACTION]` marker and field, exactly one
    `[THINKING]` field, exactly one `[SCORE]` line, and a strict `n/10` score.
    """
    raw = str(raw_response or "")
    contract = _relocation_contract()
    if (
        not raw.startswith("<think>")
        or raw.count("<think>") != 1
        or raw.count("</think>") != 1
    ):
        raise ValueError("proposal_malformed_think")

    close_end = raw.find("</think>") + len("</think>")
    controller = raw[:close_end]
    suffix = raw[close_end:]
    if not suffix.startswith("\n[EDIT] ") or suffix.startswith("\n\n"):
        raise ValueError("proposal_external_edit_marker_not_exact")
    edit_field = suffix[1:]
    if len(re.findall(r"\[EDIT\]", edit_field, flags=re.IGNORECASE)) != 1:
        raise ValueError("proposal_external_edit_field_count")
    if not edit_field.startswith("[EDIT] "):
        raise ValueError("proposal_external_edit_prefix_not_exact")

    action_markers = list(_ACTION_MARKER_RE.finditer(controller))
    actions = list(_ACTION_FIELD_RE.finditer(controller))
    if len(action_markers) != 1 or len(actions) != 1:
        raise ValueError(
            f"proposal_action_count:{len(action_markers)}:{len(actions)}"
        )
    raw_action = actions[0].group("action")
    action = raw_action.lower()

    thinking_markers = list(_THINKING_MARKER_RE.finditer(controller))
    thinking_matches = list(_THINKING_FIELD_RE.finditer(controller))
    if len(thinking_markers) != 1 or len(thinking_matches) != 1:
        raise ValueError(
            "proposal_thinking_count:"
            f"{len(thinking_markers)}:{len(thinking_matches)}"
        )
    raw_thinking = thinking_matches[0].group("thinking")
    thinking = raw_thinking.strip()
    if not thinking:
        raise ValueError("proposal_thinking_empty")
    if contract.is_generated_placeholder_thinking(thinking):
        raise ValueError("proposal_thinking_placeholder")

    score_lines = list(_SCORE_LINE_RE.finditer(raw))
    if len(score_lines) != 1:
        raise ValueError(f"proposal_score_count:{len(score_lines)}")
    raw_score = score_lines[0].group("value")
    score = v20_normalize_strict_model_score(raw_score)

    raw_payload = edit_field[len("[EDIT] ") :]
    edit_none_alias = False
    if action == "edit":
        if (
            not raw_payload
            or raw_payload.strip().lower() in _NULL_PAYLOADS
            or contract.is_generated_placeholder_payload(
                raw_payload.strip()
            )
        ):
            # P2: a null or placeholder payload under `[ACTION] edit` is a
            # semantic DONE. Route it instead of killing the trajectory and
            # record the alias so `edit_none_alias_rate` can be logged.
            edit_none_alias = True
            action = "done"
            payload = ""
        else:
            if raw_payload != raw_payload.strip():
                raise ValueError("proposal_edit_payload_invalid")
            payload = validate_payload(raw_payload)
            if payload != raw_payload:
                raise ValueError("proposal_edit_payload_changed")
    elif edit_field != "[EDIT] None":
        raise ValueError("proposal_done_payload_not_exact_none")
    else:
        payload = ""

    return {
        "proposal_parser_version": V20_STRICT_PARSER_VERSION,
        "action": action,
        "raw_action": raw_action,
        "thinking": thinking,
        "raw_thinking": raw_thinking,
        "payload": payload,
        "raw_payload": raw_payload,
        "controller": controller,
        "edit_field": edit_field,
        "score": int(score),
        "raw_score": str(raw_score),
        "edit_none_alias": bool(edit_none_alias),
    }


def v20_protocol_contract() -> dict[str, Any]:
    return {
        "version": V20_PROTOCOL_VERSION,
        "forced_prefix_version": V20_FORCED_PREFIX_VERSION,
        "score_field_version": V20_SCORE_FIELD_VERSION,
        "online_parser_version": V20_STRICT_PARSER_VERSION,
        "online_parser_is_strict": True,
        "forced_prefix_example": v20_forced_controller_prefix(0),
        "forced_prefix_example_length": len(v20_forced_controller_prefix(0)),
        "score_field_continuations": v20_score_field_continuations(),
        "score_scale": PROTOCOL_SCORE_SCALE,
        "edit_none_alias_routes_to_done": True,
        "system_prompt_path": str(V20_SYSTEM_PROMPT_PATH),
    }


__all__ = [
    "PROTOCOL_SCORE_SCALE",
    "ScoreFieldSchedule",
    "V20_FORCED_PREFIX_VERSION",
    "V20_PROTOCOL_VERSION",
    "V20_SCORE_FIELD_VERSION",
    "V20_STRICT_PARSER_VERSION",
    "V20_SYSTEM_PROMPT_PATH",
    "v20_canonicalize_model_score_response",
    "v20_forced_controller_prefix",
    "v20_normalize_strict_model_score",
    "v20_parse_controller_response",
    "v20_protocol_contract",
    "v20_score_field_continuations",
]
