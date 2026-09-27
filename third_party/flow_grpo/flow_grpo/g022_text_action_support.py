"""G022 text-action support contract.

The controller LM head has more output rows than the active tokenizer defines
symbols for.  On the Bagel classic parent the head is 152064 rows wide while
the tokenizer supports ids ``0..151664`` only, leaving 399 unsupported rows
``151665..152063`` whose ``convert_ids_to_tokens`` result is ``None``.

Unconstrained persistent generation normalises over every head row, so an
unsupported row can be selected as a genuine, policy-active, model-sampled
action.  Two G022 Formal400 attempts died that way -- once in rollout span
capture, once in unconditional forensic labelling -- because the *consumers*
were patched rather than the sampler's probability space.

This module is the single source of truth for that probability space:

* :func:`build_text_action_support` derives the supported id set from the
  live tokenizer and the live head width -- never from a hard-coded number.
* :meth:`TextActionSupport.additive_mask` produces the ``0.0``/``-inf`` row
  mask that must be added to logits **before** sampling, greedy selection and
  behaviour log-prob calculation, and again before the policy and frozen
  reference teacher-forced ``log_softmax``, so PPO ratios and KL live in
  exactly the sampler's probability space.
* :func:`assert_support_contract_synchronized` proves every rank derived the
  same contract, and raises on *all* ranks when they disagree.
* :func:`first_unsupported_token` is the defence-in-depth detector: if an
  unsupported id ever reaches a consumer anyway it is retained exactly --
  never dropped, coerced to UNK, or rendered as a placeholder -- so the
  refusal can persist its id, position and log-prob.

Parameter and checkpoint shape are untouched: the head keeps all 152064 rows,
the mask is applied to logits only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch

VERSION = "clean29529_g022_text_action_support_v1"


class UnsupportedTextActionTokenError(RuntimeError):
    """An LM-head row outside the tokenizer's support reached a consumer.

    This is an infra invariant, not a model-content classification: under the
    support mask it is unreachable, so seeing one means the mask was not
    applied on some path.  It fails closed.
    """


class TextActionSupportSyncError(RuntimeError):
    """Ranks disagree on the text-action support contract."""


@dataclass(frozen=True)
class TextActionSupport:
    """Immutable supported-id contract for one tokenizer/head pair."""

    version: str
    head_rows: int
    supported_count: int
    unsupported_ids: tuple[int, ...]
    support_hash: str

    def __post_init__(self) -> None:
        if self.head_rows <= 0:
            raise ValueError("text-action support needs a positive head width")
        if self.supported_count <= 0:
            raise ValueError("text-action support needs a supported id")
        if self.supported_count + len(self.unsupported_ids) != self.head_rows:
            raise ValueError("text-action support partition is not exhaustive")
        object.__setattr__(self, "_unsupported_set", frozenset(self.unsupported_ids))
        object.__setattr__(self, "_mask_cache", {})

    # -- queries ---------------------------------------------------------
    @property
    def unsupported_count(self) -> int:
        return len(self.unsupported_ids)

    @property
    def fully_supported(self) -> bool:
        return not self.unsupported_ids

    def is_supported(self, token_id: int) -> bool:
        value = int(token_id)
        if value < 0 or value >= self.head_rows:
            return False
        return value not in getattr(self, "_unsupported_set")

    def describe(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "head_rows": int(self.head_rows),
            "supported_count": int(self.supported_count),
            "unsupported_count": int(self.unsupported_count),
            "unsupported_min": (
                int(min(self.unsupported_ids)) if self.unsupported_ids else None
            ),
            "unsupported_max": (
                int(max(self.unsupported_ids)) if self.unsupported_ids else None
            ),
            "support_hash": str(self.support_hash),
        }

    # -- masking ---------------------------------------------------------
    def additive_mask(
        self,
        *,
        device: Any = None,
        dtype: Any = None,
    ) -> torch.Tensor:
        """A ``[head_rows]`` additive mask: ``0.0`` supported, ``-inf`` not.

        Cached per ``(device, dtype)``.  The returned tensor is shared, so
        callers must treat it as read-only.
        """
        key = (str(device), str(dtype))
        cache = getattr(self, "_mask_cache")
        cached = cache.get(key)
        if cached is not None:
            return cached
        mask = torch.zeros(
            self.head_rows,
            device=device,
            dtype=torch.float32 if dtype is None else dtype,
        )
        if self.unsupported_ids:
            index = torch.tensor(
                [int(value) for value in self.unsupported_ids],
                device=mask.device,
                dtype=torch.long,
            )
            mask.index_fill_(0, index, float("-inf"))
        cache[key] = mask
        return mask

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Add the support mask to the last dimension of ``logits``.

        A no-op when the head and the tokenizer agree, so a model whose head
        is exactly the tokenizer width pays nothing and stays bit-identical.
        """
        # The width check runs even when nothing is masked: it is the live
        # proof that the contract describes the head this model actually has,
        # and a contract that is never checked is indistinguishable from none.
        if logits.shape[-1] != self.head_rows:
            raise UnsupportedTextActionTokenError(
                "text-action support width "
                f"{self.head_rows} != logits width {logits.shape[-1]}"
            )
        if self.fully_supported:
            return logits
        mask = self.additive_mask(device=logits.device, dtype=logits.dtype)
        return logits + mask

    # -- defence in depth ------------------------------------------------
    def unsupported_positions(
        self, token_ids: Sequence[int]
    ) -> list[dict[str, int]]:
        """Every ``(position, token_id)`` outside the support, in order."""
        found: list[dict[str, int]] = []
        for position, value in enumerate(token_ids):
            token = int(value)
            if not self.is_supported(token):
                found.append({"position": int(position), "token_id": token})
        return found

    def validate_allowed_ids(
        self, allowed: Iterable[int], *, where: str
    ) -> None:
        bad = sorted({int(v) for v in allowed if not self.is_supported(int(v))})
        if bad:
            raise UnsupportedTextActionTokenError(
                f"{where}: constrained allowed set leaves text-action support: {bad}"
            )

    def validate_schedule(self, schedule: Any, *, where: str) -> dict[str, Any]:
        """Prove a constrained token schedule is a subset of the support."""
        sequences = getattr(schedule, "sequences", None)
        if sequences is None:
            raise UnsupportedTextActionTokenError(
                f"{where}: constrained schedule exposes no token sequences"
            )
        ids = sorted({int(token) for row in sequences for token in row})
        self.validate_allowed_ids(ids, where=where)
        return {
            "version": VERSION,
            "where": str(where),
            "schedule_version": str(getattr(schedule, "version", "")),
            "schedule_token_count": len(ids),
            "schedule_subset_of_support": True,
            "support_hash": str(self.support_hash),
        }


def _support_hash(head_rows: int, unsupported_ids: Sequence[int]) -> str:
    payload = json.dumps(
        {
            "version": VERSION,
            "head_rows": int(head_rows),
            "unsupported_ids": [int(value) for value in unsupported_ids],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_text_action_support(
    *,
    tokenizer: Any,
    head_rows: int,
) -> TextActionSupport:
    """Derive the contract from the live tokenizer and live head width.

    ``head_rows`` must come from the model's actual output projection, not
    from a constant, so a checkpoint whose head width changes is caught here
    rather than by a decode failure hours into a run.
    """
    rows = int(head_rows)
    if rows <= 0:
        raise ValueError("text-action support needs a positive head width")
    tokens = tokenizer.convert_ids_to_tokens(list(range(rows)))
    if len(tokens) != rows:
        raise UnsupportedTextActionTokenError(
            "tokenizer returned "
            f"{len(tokens)} symbols for {rows} head rows"
        )
    unsupported = tuple(
        index for index, token in enumerate(tokens) if token is None
    )
    support = TextActionSupport(
        version=VERSION,
        head_rows=rows,
        supported_count=rows - len(unsupported),
        unsupported_ids=unsupported,
        support_hash=_support_hash(rows, unsupported),
    )
    return support


def head_rows_from_model(model: Any) -> int:
    """The live LM-head output width, read from the parameter itself."""
    for attribute in ("lm_head", "language_model"):
        node = getattr(model, attribute, None)
        if node is None:
            continue
        head = getattr(node, "lm_head", node)
        weight = getattr(head, "weight", None)
        if weight is not None and weight.dim() == 2:
            return int(weight.shape[0])
    raise UnsupportedTextActionTokenError(
        "could not read the LM-head output width from the model"
    )


def assert_support_contract_synchronized(
    support: TextActionSupport,
    *,
    process_group: Any = None,
    require_distributed: bool = True,
) -> dict[str, Any]:
    """All-gather the contract descriptor and raise on *every* rank if it differs.

    Rank-local validation is exactly the failure mode that produced a
    silent hang before (A4): one rank raises, the others enter the next
    collective and wait for the allocation to time out.  Every rank here sees
    every descriptor, so every rank reaches the same verdict.
    """
    import torch.distributed as dist

    descriptor = support.describe()
    if not (dist.is_available() and dist.is_initialized()):
        if require_distributed:
            raise TextActionSupportSyncError(
                "text-action support synchronisation requires distributed ranks"
            )
        return {
            "version": VERSION,
            "world_size": 1,
            "synchronized": True,
            "descriptor": descriptor,
        }
    world_size = dist.get_world_size(group=process_group)
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, descriptor, group=process_group)
    mismatched = [
        rank
        for rank, value in enumerate(gathered)
        if value != descriptor
    ]
    if mismatched:
        raise TextActionSupportSyncError(
            "text-action support contract differs across ranks: "
            f"local={descriptor} mismatched_ranks={mismatched} "
            f"gathered={gathered}"
        )
    return {
        "version": VERSION,
        "world_size": int(world_size),
        "synchronized": True,
        "descriptor": descriptor,
    }


def first_unsupported_token(
    support: TextActionSupport | None,
    token_ids: Sequence[int],
) -> dict[str, int] | None:
    if support is None:
        return None
    for position, value in enumerate(token_ids):
        token = int(value)
        if not support.is_supported(token):
            return {"position": int(position), "token_id": token}
    return None


def unsupported_token_evidence(
    *,
    support: TextActionSupport,
    token_ids: Sequence[int],
    log_probs: Sequence[float] | None = None,
    attempt_id: str = "",
    where: str = "",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Exact, non-lossy evidence for a refusal.

    Retains the numeric id and position verbatim.  Nothing is dropped,
    coerced to UNK, or replaced with a placeholder: the id *is* the evidence.
    """
    positions = support.unsupported_positions(token_ids)
    records = []
    for entry in positions:
        position = int(entry["position"])
        log_prob = None
        if log_probs is not None and position < len(log_probs):
            log_prob = float(log_probs[position])
        records.append(
            {
                "position": position,
                "token_id": int(entry["token_id"]),
                "sampling_log_prob": log_prob,
            }
        )
    payload = {
        "version": VERSION,
        "attempt_id": str(attempt_id),
        "where": str(where),
        "sampled_token_count": int(len(token_ids)),
        "unsupported_token_count": len(records),
        "unsupported_tokens": records,
        "support": support.describe(),
    }
    if extra:
        payload["context"] = dict(extra)
    return payload


__all__ = [
    "VERSION",
    "TextActionSupport",
    "TextActionSupportSyncError",
    "UnsupportedTextActionTokenError",
    "assert_support_contract_synchronized",
    "build_text_action_support",
    "first_unsupported_token",
    "head_rows_from_model",
    "unsupported_token_evidence",
]
