"""Bounded future-forensic evidence for G016 text updates.

The ring is deliberately diagnostic-only: it cannot alter masks, losses,
gradients, optimizer state, checkpoint state, or sampling.  It retains exact
generation token IDs/log-probs, train/reference replay values, analytical
pre-clip dloss/dlogp attribution, and context image hash/seed lineage around an
excursion while bounding both per-turn and cross-step storage.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

VERSION = "clean29529_g016_forensic_ring_buffer_v1"
TOKEN_TOPK_VERSION = "clean29529_g016_per_token_topk_v1"
ID_LABEL_VERSION = "clean29529_g022_forensic_id_labels_v1"

LEGACY_BOUND_MODE = "clamp20"
G022_BOUND_MODE = "g022_linear"
# Mirrors unify_rl.train.g022_bounded_kl.LINEARIZATION_THRESHOLD. Imported
# lazily below so this diagnostic module keeps no import-time dependency on
# the loss modules it describes.
G022_LINEARIZATION_THRESHOLD = 10.0
LEGACY_CLAMP = 20.0

# Host-owned marker literals. These are OUR strings, not model output, so
# encoding them is not a decode of sampled content.
MARKER_LITERALS = (
    ("</think>", "think_close"),
    ("[EDIT]", "external_EDIT_marker"),
)


def _bound_terms(delta: float, bound_mode: str) -> tuple[float, float, float, bool]:
    """`(exp_value, d/d(delta) of exp_value, effective_delta, saturated)`.

    The attribution has to use the bound the *loss* used.  Under G022 the loss
    is a C1 linearization above 10 nats whose slope is the constant `exp(10)`,
    so the old `clamp(d, -20, 20)` report described a zero gradient where the
    live objective applies a large finite one -- the report would have been
    wrong in exactly the regime it exists to explain.
    """
    value = float(delta)
    if str(bound_mode) == G022_BOUND_MODE:
        limit = float(G022_LINEARIZATION_THRESHOLD)
        if value > limit:
            return (
                math.exp(limit) * (1.0 + value - limit),
                math.exp(limit),
                value,
                False,
            )
        return math.exp(value), math.exp(value), value, False
    if str(bound_mode) != LEGACY_BOUND_MODE:
        raise ValueError(f"unknown text loss bound mode: {bound_mode!r}")
    clamped = max(-LEGACY_CLAMP, min(LEGACY_CLAMP, value))
    saturated = value < -LEGACY_CLAMP or value > LEGACY_CLAMP
    return math.exp(clamped), (0.0 if saturated else math.exp(clamped)), clamped, saturated


def _marker_id_sequences(tokenizer: Any) -> list[tuple[tuple[int, ...], str]]:
    cached = getattr(tokenizer, "_g022_marker_id_sequences", None)
    if cached is not None:
        return cached
    sequences: list[tuple[tuple[int, ...], str]] = []
    for literal, label in MARKER_LITERALS:
        try:
            encoded = tokenizer.encode(literal, add_special_tokens=False)
        except TypeError:
            encoded = tokenizer.encode(literal)
        ids = tuple(int(value) for value in encoded)
        if ids:
            sequences.append((ids, label))
    try:
        setattr(tokenizer, "_g022_marker_id_sequences", sequences)
    except Exception:
        pass
    return sequences


def _subsequence_positions(ids: list[int], pattern: tuple[int, ...]) -> set[int]:
    found: set[int] = set()
    width = len(pattern)
    if not width or width > len(ids):
        return found
    for start in range(len(ids) - width + 1):
        if tuple(ids[start : start + width]) == pattern:
            found.update(range(start, start + width))
    return found


def label_sampled_tokens_from_ids(
    *,
    tokenizer: Any,
    sampled_token_ids: Sequence[int],
    action_mask: Sequence[bool],
    repair_mask: Sequence[bool],
    eos_token_id: int,
) -> list[str]:
    """Span labels derived from token IDs only -- model content is never decoded.

    `label_sampled_tokens` re-decodes the sampled sequence to locate the two
    marker substrings; that decode is what killed Formal400 attempt B when the
    model sampled an LM-head row with no tokenizer symbol.  This variant
    matches the same markers as token-id subsequences of the host-owned
    literals, so an unsupported id can be labelled `other` instead of raising.
    """
    ids = [int(value) for value in sampled_token_ids]
    if len(ids) != len(action_mask) or len(ids) != len(repair_mask):
        raise ValueError("G016 forensic token-label masks differ")
    spans: dict[str, set[int]] = {}
    for pattern, label in _marker_id_sequences(tokenizer):
        spans.setdefault(label, set()).update(_subsequence_positions(ids, pattern))
    think_close = spans.get("think_close", set())
    edit_marker = spans.get("external_EDIT_marker", set())
    labels = []
    for index, token_id in enumerate(ids):
        if int(token_id) == int(eos_token_id):
            label = "EOS"
        elif bool(action_mask[index]):
            label = "ACTION_value"
        elif bool(repair_mask[index]):
            label = "repair_payload"
        elif index in think_close:
            label = "think_close"
        elif index in edit_marker:
            label = "external_EDIT_marker"
        else:
            label = "other"
        labels.append(label)
    return labels


def _token_span_for_substring(tokenizer: Any, ids: list[int], substring: str) -> set[int]:
    text = tokenizer.decode(ids)
    start = text.find(substring)
    if start < 0:
        return set()
    stop = start + len(substring)
    boundaries = [(len(tokenizer.decode(ids[:index])), index) for index in range(len(ids) + 1)]
    begin = max((index for chars, index in boundaries if chars <= start), default=0)
    end = min((index for chars, index in boundaries if chars >= stop), default=len(ids))
    return set(range(begin, end))


def label_sampled_tokens(
    *,
    tokenizer: Any,
    sampled_token_ids: Sequence[int],
    action_mask: Sequence[bool],
    repair_mask: Sequence[bool],
    eos_token_id: int,
) -> list[str]:
    ids = [int(value) for value in sampled_token_ids]
    if len(ids) != len(action_mask) or len(ids) != len(repair_mask):
        raise ValueError("G016 forensic token-label masks differ")
    think_close = _token_span_for_substring(tokenizer, ids, "</think>")
    edit_marker = _token_span_for_substring(tokenizer, ids, "[EDIT]")
    labels = []
    for index, token_id in enumerate(ids):
        if int(token_id) == int(eos_token_id):
            label = "EOS"
        elif bool(action_mask[index]):
            label = "ACTION_value"
        elif bool(repair_mask[index]):
            label = "repair_payload"
        elif index in think_close:
            label = "think_close"
        elif index in edit_marker:
            label = "external_EDIT_marker"
        else:
            label = "other"
        labels.append(label)
    return labels


def build_token_topk(
    *,
    sampled_token_ids: Sequence[int],
    sampling_logprobs: torch.Tensor,
    train_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    loss_mask: Sequence[bool],
    action_mask: Sequence[bool],
    repair_mask: Sequence[bool],
    token_labels: Sequence[str],
    action_advantage: float,
    repair_advantage: float,
    kl_beta: float,
    backward_loss_scale: float,
    top_k: int = 8,
    bound_mode: str = LEGACY_BOUND_MODE,
) -> dict[str, Any]:
    ids = [int(value) for value in sampled_token_ids]
    tensors = (sampling_logprobs, train_logprobs, reference_logprobs)
    if top_k <= 0 or any(value.ndim != 1 or value.numel() != len(ids) for value in tensors):
        raise ValueError("G016 forensic token arrays differ")
    if any(len(value) != len(ids) for value in (loss_mask, action_mask, repair_mask, token_labels)):
        raise ValueError("G016 forensic token metadata differs")
    sampling = sampling_logprobs.detach().float().cpu()
    train = train_logprobs.detach().float().cpu()
    reference = reference_logprobs.detach().float().cpu()
    action_count = max(1, sum(bool(value) for value in action_mask))
    repair_count = max(1, sum(bool(value) for value in repair_mask))
    kl_count = max(1, sum(bool(value) for value in loss_mask))
    rows = []
    for index, token_id in enumerate(ids):
        behavior_delta = float(train[index] - sampling[index])
        reference_delta = float(train[index] - reference[index])
        kl_exp, kl_slope, clamped, saturated = _bound_terms(
            reference_delta, bound_mode
        )
        ratio_exp, _, _, _ = _bound_terms(behavior_delta, bound_mode)
        action_gradient = (
            -float(action_advantage) / action_count * float(backward_loss_scale)
            if bool(action_mask[index]) else 0.0
        )
        repair_gradient = (
            -float(repair_advantage) / repair_count * float(backward_loss_scale)
            if bool(repair_mask[index]) else 0.0
        )
        k3_gradient = 0.0
        if bool(loss_mask[index]) and not saturated:
            k3_gradient = (
                float(kl_beta) * (kl_slope - 1.0)
                / kl_count * float(backward_loss_scale)
            )
        k3_value = (
            float(kl_beta) * (kl_exp - 1.0 - clamped)
            if bool(loss_mask[index]) else 0.0
        )
        score = max(
            abs(behavior_delta), abs(reference_delta), abs(action_gradient),
            abs(repair_gradient), abs(k3_gradient),
        )
        rows.append({
            "token_idx": index,
            "token_id": token_id,
            "sampling_logprob": float(sampling[index]),
            "train_logprob": float(train[index]),
            "sampling_train_delta_logprob": behavior_delta,
            "sampling_train_importance_ratio": ratio_exp,
            "kl_bound_mode": str(bound_mode),
            "reference_logprob": float(reference[index]),
            "train_reference_delta_logprob": reference_delta,
            "k3_weighted_token_value": k3_value,
            "loss_mask": bool(loss_mask[index]),
            "span_label": str(token_labels[index]),
            "action_advantage": float(action_advantage),
            "repair_advantage": float(repair_advantage),
            "action_route": "action" if bool(action_mask[index]) else "repair" if bool(repair_mask[index]) else "kl_only",
            "preclip_dlogp": {
                "ordinary_action": action_gradient,
                "ordinary_repair": repair_gradient,
                "k3": k3_gradient,
            },
            "k3_clamp_saturated": saturated,
            "ranking_score": score,
        })
    selected = sorted(rows, key=lambda row: (-float(row["ranking_score"]), int(row["token_idx"])))[: min(top_k, len(rows))]
    return {
        "version": TOKEN_TOPK_VERSION,
        "top_k": int(top_k),
        "source_token_count": len(rows),
        "selected_token_count": len(selected),
        "selection": "descending max(abs behavior delta, abs reference delta, abs channel preclip dlogp)",
        "kl_bound_mode": str(bound_mode),
        "rows": selected,
    }


@dataclass
class G016ForensicRingBuffer:
    capacity_steps: int = 6
    top_k_per_step: int = 64
    steps: deque[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.capacity_steps <= 0 or self.top_k_per_step <= 0:
            raise ValueError("G016 forensic ring bounds must be positive")
        self.steps = deque(list(self.steps or []), maxlen=int(self.capacity_steps))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None, *, capacity_steps: int = 6, top_k_per_step: int = 64) -> "G016ForensicRingBuffer":
        if not payload:
            return cls(capacity_steps=capacity_steps, top_k_per_step=top_k_per_step)
        if payload.get("version") != VERSION:
            raise ValueError("G016 forensic ring version differs")
        return cls(capacity_steps=capacity_steps, top_k_per_step=top_k_per_step, steps=deque(list(payload.get("steps") or [])))

    def begin_step(self, *, step: int, marker: Mapping[str, Any], context_lineage: Sequence[Mapping[str, Any]]) -> None:
        if any(
            not str(row.get("r0_image_sha256") or "")
            or not isinstance(row.get("r0_seed"), int)
            or not isinstance(row.get("round_image_sha256s"), list)
            for row in context_lineage
        ):
            raise ValueError("G016 forensic context hash/seed lineage differs")
        if any(int(row.get("step", -1)) == int(step) for row in self.steps):
            raise ValueError("G016 forensic ring duplicate step")
        self.steps.append({
            "step": int(step),
            "pre_update_marker": dict(marker),
            "context_lineage": [dict(row) for row in context_lineage],
            "token_topk": [],
            "post_update_marker": None,
        })

    def finish_step(self, *, step: int, marker: Mapping[str, Any], token_rows: Sequence[Mapping[str, Any]]) -> None:
        matches = [row for row in self.steps if int(row["step"]) == int(step)]
        if len(matches) != 1 or matches[0]["post_update_marker"] is not None:
            raise ValueError("G016 forensic post marker differs")
        selected = sorted(
            (dict(row) for row in token_rows),
            key=lambda row: (-float(row.get("ranking_score", 0.0)), str(row.get("sample_id", "")), int(row.get("token_idx", -1))),
        )[: int(self.top_k_per_step)]
        matches[0]["token_topk"] = selected
        matches[0]["post_update_marker"] = dict(marker)

    def truncate_after(self, step: int) -> list[dict[str, Any]]:
        """Discard diagnostic entries newer than a durable recovery step."""
        assert self.steps is not None
        retained = [row for row in self.steps if int(row.get("step", -1)) <= int(step)]
        discarded = [dict(row) for row in self.steps if int(row.get("step", -1)) > int(step)]
        self.steps = deque(retained, maxlen=int(self.capacity_steps))
        return discarded

    def payload(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "capacity_steps": int(self.capacity_steps),
            "top_k_per_step": int(self.top_k_per_step),
            "step_count": len(self.steps),
            "steps": list(self.steps),
            "diagnostic_only": True,
            "optimizer_or_loss_mutation": False,
        }
