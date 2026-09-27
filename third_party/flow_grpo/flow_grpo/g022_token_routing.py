"""G022 item 3: model-output failures are classified and penalised, never raised.

Design notes: Appendix A-1 and decision B-8 (Appendix F: fail-closed **off** for
model output).

`g016_token_routing.build_g016_token_masks` is 42 lines with eight hard
`raise RuntimeError` paths. One of them stopped the entire 16-GPU G021-A run at
step 89 because a single trajectory wrote

    [ACTION] Keep the current image with the plain background and five horses.

instead of `edit` / `done`. No span could be captured, the branch chain fell
through to `else: raise RuntimeError('G016 decision is not routable')`, and the
run required an emergency checkpoint.

Under RL the policy drifts into malformed output -- measured, not hypothetical:
G021-B's invalid rate went 0.6% -> 16.4%. G022 runs 2 roots x 16 siblings x up
to 3 rounds = up to 96 controller decisions per step, so at a 16% invalid rate
the probability that at least one is malformed is effectively 1. Every step
would crash.

**The rule this module implements.** A failure whose cause is *what the model
wrote* is a policy behaviour: it is classified, the round is routed no policy
credit, and the reward charges the malformed-action penalty. A failure whose
cause is *our own state* -- token-id lineage that does not match the recorded
turn, a mask longer than the turn, credit claimed for a token the turn never
sampled -- is an infra invariant and still raises, exactly as before.

Concretely, of `g016_token_routing`'s raises:

| old raise | G022 |
|---|---|
| span status is `unparseable` | classify `unparseable_response` |
| `sampled_token_ids` differ from the turn | **still raises** (infra lineage) |
| `{action,payload}_token_span` missing/ambiguous | classify `span_unavailable` |
| actionable / alias-stop / literal-stop / invalid-action lineage differs | classify `decision_span_disagreement` |
| terminal EOS outside the repair span | classify `eos_outside_payload` |
| decision is not routable (the G021-A killer) | classify `unroutable_decision` |
| action mask empty | classify `empty_action_mask` |
| repair payload mask empty | classify `empty_payload_mask` |
| action/repair heads overlap | classify `head_overlap` |

`head_overlap` is classified rather than raised deliberately: overlapping spans
can only come from what the model wrote (a response whose `[ACTION]` value and
`[EDIT]` payload occupy the same tokens), and routing a token to two heads is
prevented by returning no credit at all for that turn.
"""
from __future__ import annotations

from typing import Any

VERSION = "clean29529_g022_classified_token_routing_v1"

# Every non-"routed" value here means: this turn receives no policy credit and
# the trajectory is charged the malformed-action penalty.
ROUTED = "routed"
CLASSIFICATIONS = (
    "unparseable_response",
    "span_unavailable",
    "decision_span_disagreement",
    "eos_outside_payload",
    "unroutable_decision",
    "empty_action_mask",
    "empty_payload_mask",
    "head_overlap",
)


class TokenRoutingLineageError(RuntimeError):
    """Infra invariant. Our own recorded state is inconsistent -- fail closed."""


def _malformed(
    count: int,
    status: str,
    detail: str,
    *,
    credited: list[bool] | None = None,
) -> tuple[dict[str, list[bool]], dict[str, Any]]:
    """Classify a malformed response AND give its penalty somewhere to land.

    T1.4 / finding O-5. The first version of this returned both masks all-False
    with `policy_active: False`. A trajectory malformed in its first round then
    carried the -0.5 `R(tau)` penalty and contributed **exactly zero gradient**:
    the penalty was recorded and never reached a parameter. Invalid syntax could
    drift without correction -- precisely the failure mode that took G021-B from
    0.6% to 16.4% invalid.

    The trajectory advantage is now routed to every **model-sampled,
    non-host-forced** token of the malformed response, which is exactly
    `turn.credited_token_mask()`. Host-forced tokens stay excluded: the model did
    not choose them, so they must not be credited or penalised.

    The tokens go on the `action` mask because under G022 there is one head and
    one scalar, so the two masks differ only in which PPO term they feed; the
    single-head loss takes their union either way.
    """

    routed = [False] * count if credited is None else [bool(v) for v in credited]
    return (
        {"action": routed, "repair": [False] * count},
        {
            "version": VERSION,
            "routing_status": status,
            "routing_detail": detail,
            "policy_active": any(routed),
            "malformed_action": True,
            "penalty_routed_token_count": sum(1 for v in routed if v),
            # A1 / Q-3: these two were a single hard-coded `True` claim. The
            # claim was false whenever the caller handed us an all-True mask,
            # which `malformed_text_turn_from_event` used to do. Report what was
            # actually consulted and how many tokens it actually excluded.
            "host_forced_tokens_excluded": credited is not None,
            "host_forced_token_excluded_count": sum(1 for v in routed if not v),
            "generation_time_lineage": True,
            "posthoc_retokenization_used": False,
            "action_indexes": [i for i, x in enumerate(routed) if x],
            "repair_indexes": [],
            "terminal_eos_index": None,
            "terminal_eos_excluded_from_repair": False,
            "score_or_thinking_broadcast": False,
            "alias_edit_token_penalized": False,
        },
    )


def build_g022_token_masks(
    *,
    turn: Any,
    decision: dict[str, Any],
    sampled_spans: dict[str, Any],
) -> tuple[dict[str, list[bool]], dict[str, Any]]:
    """Route generation-time sampled spans, classifying every model-output failure.

    Returns ``(masks, audit)``. ``audit["routing_status"]`` is ``"routed"`` on
    success, otherwise one of ``CLASSIFICATIONS``; on any classification both
    masks are all-``False`` and ``audit["malformed_action"]`` is ``True``.

    Raises ``TokenRoutingLineageError`` only for infra invariants.
    """

    turn.validate()
    count = len(turn.sampled_token_ids)
    allowed = turn.credited_token_mask()

    # ---- infra invariants: our own recorded state, not the model's text -----
    if len(allowed) != count:
        raise TokenRoutingLineageError(
            "G022 credited-token mask does not cover the sampled turn"
        )
    recorded_ids = list(sampled_spans.get("sampled_token_ids", []))
    if recorded_ids and recorded_ids != list(turn.sampled_token_ids):
        raise TokenRoutingLineageError(
            "G022 generation-time sampled span lineage differs from the turn"
        )

    # ---- everything below is driven by what the model wrote ----------------
    status = sampled_spans.get("status")
    if status not in {"captured", "action_only"}:
        return _malformed(
            count,
            "unparseable_response",
            f"span capture status is {status!r}",
            credited=allowed,
        )
    if not recorded_ids:
        return _malformed(
            count,
            "span_unavailable",
            "span record carries no sampled token ids", credited=allowed)

    action = [False] * count
    repair = [False] * count

    def select(mask: list[bool], key: str) -> bool:
        span = sampled_spans.get(key)
        if (
            not isinstance(span, list)
            or len(span) != 2
            or not 0 <= span[0] < span[1] <= count
        ):
            return False
        for index in range(span[0], span[1]):
            mask[index] = bool(allowed[index])
        return True

    raw = str(decision.get("raw_action", "")).lower()
    payload = str(decision.get("payload") or "")
    terminal_eos_index = None

    if decision.get("actionable_edit"):
        if (
            status != "captured"
            or raw != "edit"
            or sampled_spans.get("action_value") != "edit"
        ):
            return _malformed(
                count,
                "decision_span_disagreement",
                "actionable-edit decision disagrees with the captured span", credited=allowed)
        if not select(action, "action_token_span") or not select(
            repair, "payload_token_span"
        ):
            return _malformed(count, "span_unavailable", "edit action/payload span missing", credited=allowed)
        # R1, preserved verbatim: the generation-time
        # payload span extends through the model-selected/host-appended
        # <|im_end|>. Keep every semantic payload token byte-identical while
        # removing only that terminal EOS from ordinary repair credit.
        if str(sampled_spans.get("decoded_sampled", "")).rstrip().endswith("<|im_end|>"):
            terminal_eos_index = count - 1
            if not repair[terminal_eos_index]:
                return _malformed(
                    count,
                    "eos_outside_payload",
                    "terminal EOS is outside the repair span", credited=allowed)
            repair[terminal_eos_index] = False
    elif decision.get("alias_stop"):
        if status != "captured" or raw != "edit" or payload.casefold() != "none":
            return _malformed(
                count,
                "decision_span_disagreement",
                "alias-stop decision disagrees with the captured span", credited=allowed)
        if not select(action, "payload_token_span"):
            return _malformed(count, "span_unavailable", "alias-stop payload span missing", credited=allowed)
    elif decision.get("literal_stop"):
        if sampled_spans.get("action_value") != "done":
            return _malformed(
                count,
                "decision_span_disagreement",
                "literal-stop decision disagrees with the captured span", credited=allowed)
        if not select(action, "action_token_span"):
            return _malformed(count, "span_unavailable", "literal-stop action span missing", credited=allowed)
    elif decision.get("invalid_sampled_action"):
        if not raw or raw == "invalid" or sampled_spans.get("action_value") != raw:
            return _malformed(
                count,
                "decision_span_disagreement",
                "invalid sampled-action decision disagrees with the captured span", credited=allowed)
        # The reward already classifies this protocol-invalid decision as
        # incorrect. Route that bounded credit only to the sampled action value;
        # malformed payload/thinking/SCORE tokens stay outside both heads.
        if not select(action, "action_token_span"):
            return _malformed(
                count, "span_unavailable", "invalid-action action span missing", credited=allowed)
    else:
        # This is the exact branch that stopped G021-A at step 89.
        return _malformed(
            count,
            "unroutable_decision",
            f"decision matches no route: raw_action={raw!r} "
            f"token_route={decision.get('token_route')!r}", credited=allowed)

    if not any(action):
        return _malformed(count, "empty_action_mask", "no credited action token", credited=allowed)
    if decision.get("actionable_edit") and not any(repair):
        return _malformed(count, "empty_payload_mask", "no credited repair payload token", credited=allowed)
    if any(a and r for a, r in zip(action, repair)):
        return _malformed(count, "head_overlap", "action and payload spans overlap", credited=allowed)

    return (
        {"action": action, "repair": repair},
        {
            "version": VERSION,
            "routing_status": ROUTED,
            "routing_detail": "",
            "policy_active": True,
            "malformed_action": False,
            "generation_time_lineage": True,
            "posthoc_retokenization_used": False,
            "action_indexes": [i for i, x in enumerate(action) if x],
            "repair_indexes": [i for i, x in enumerate(repair) if x],
            "terminal_eos_index": terminal_eos_index,
            "terminal_eos_excluded_from_repair": terminal_eos_index is not None,
            "score_or_thinking_broadcast": False,
            "alias_edit_token_penalized": False,
        },
    )


def build_g024_uniform_masks(*, turn: Any) -> tuple[dict[str, list[bool]], dict[str, Any]]:
    """G024: one scalar, every token the model generated. No routing.

    G022's routing was broken -- `g016_decision` was always empty, so every turn
    fell to `_malformed`, which credits `turn.credited_token_mask()`: the WHOLE
    response. Its protocol converged to perfect and held for 60 steps.

    G023 repaired the routing and, per its Goal SS4, put THINKING, SCORE and every
    structural token in NEITHER mask. Those positions were then held only by
    `text_kl_beta = 1e-4`. Its protocol diverged from step 11 and the run died at
    step 50: `[ACTION]` per response 1.03 -> 1.72, `---` 0.09 -> 12.4, and 100% of
    the growth was after `</think>` -- extra separators, hallucinated `<step>`
    environment turns, self-generated Round#1/Round#2 blocks. THINKING length and
    payload length were FLAT throughout.

    The tokens that degraded are exactly the tokens G023 removed from the
    gradient. G024 credits all of them, uniformly, including SCORE (an explicit
    decision; see the SCORE-distribution monitor).

    There is no failure mode here: this cannot be "unroutable", because it does
    not route. The only thing that can go wrong is an infra invariant, which
    raises rather than classifying.
    """

    turn.validate()
    count = len(turn.sampled_token_ids)
    allowed = turn.credited_token_mask()
    if len(allowed) != count:
        raise TokenRoutingLineageError(
            "G024 credited-token mask does not cover the sampled turn"
        )
    if not any(allowed):
        raise TokenRoutingLineageError(
            "G024 turn has no policy-active token to credit"
        )
    # Q: should this require `all(allowed)`, on
    # the reading that G024 credits "every sampled position". It must not.
    # `credited_token_mask()` returns the generation-time
    # `raw_content_policy_active` record, and the sampler can legitimately
    # force a position INSIDE the content span. Those are host-forced tokens,
    # which the Goal's own table excludes from credit because they are not the
    # model's choices -- forcing `all()` would raise on correct data.
    #
    # The precise claim is therefore "every POLICY-ACTIVE sampled position, and
    # nothing else", not "every sampled position". The gap between the two is
    # made explicit here rather than left as a silent difference, because a
    # credited fraction that starts DRIFTING is the observable that matters.
    credited = sum(1 for value in allowed if value)
    partial = credited != count
    return (
        {"action": list(allowed), "repair": [False] * count},
        {
            "version": VERSION,
            "routing_status": ROUTED,
            "routing_detail": "g024_uniform_credit",
            "policy_active": True,
            "malformed_action": False,
            "generation_time_lineage": True,
            "posthoc_retokenization_used": False,
            "action_indexes": [i for i, x in enumerate(allowed) if x],
            "repair_indexes": [],
            "terminal_eos_index": None,
            "terminal_eos_excluded_from_repair": False,
            # True and deliberate: this is the whole point of G024, where in
            # G022 the same coverage was an accident of the malformed fallback.
            "score_or_thinking_broadcast": True,
            "alias_edit_token_penalized": False,
            "g024_uniform_credit": True,
            "g024_credited_token_count": credited,
            "g024_sampled_token_count": count,
            "g024_partial_credit_mask": partial,
            "g024_host_forced_inside_span": count - credited,
        },
    )


def _safe_decode(tokenizer: Any, ids: list[int]) -> str:
    """Decode that cannot raise, for the classification path.

    The classifier must be able to describe a response whose token ids the
    tokenizer cannot decode -- that is the whole situation it exists to report.
    Falls back to per-token decoding, then to the raw ids, so a run is never
    lost to a failure inside the handler for that failure.
    """

    try:
        return tokenizer.decode(ids)
    except Exception:
        pass
    pieces: list[str] = []
    for value in ids:
        try:
            piece = tokenizer.decode([value])
        except Exception:
            piece = None
        pieces.append(piece if isinstance(piece, str) else f"<undecodable:{value}>")
    return "".join(pieces)


def capture_sampled_field_spans_classified(
    *,
    tokenizer: Any,
    sampled_token_ids: Any,
) -> dict[str, Any]:
    """`capture_sampled_field_spans` that returns `unparseable` instead of raising.

    `g016_sampled_spans._token_span_for_chars` raises on three conditions --
    empty character span, no exact token-boundary enclosure, ambiguous
    span/token lineage. All three are reachable from model text whose field
    values do not land on token boundaries, so under G022 they become the same
    `unparseable` status the capturer already returns for a missing `[ACTION]`
    marker, and the round is classified and penalised rather than killing the run.
    """

    from flow_grpo.g016_sampled_spans import capture_sampled_field_spans

    ids = [int(value) for value in sampled_token_ids]
    try:
        return capture_sampled_field_spans(
            tokenizer=tokenizer,
            sampled_token_ids=ids,
        )
    except TokenRoutingLineageError:
        # Infra invariant: our own recorded state is inconsistent. A-1 keeps
        # these raising. Never classified away.
        raise
    except Exception as exc:
        # A-1 / item 3: a failure whose cause is WHAT THE MODEL WROTE is a
        # policy behaviour -- classified and penalised, never fatal.
        #
        # This caught only `RuntimeError`, and it killed the formal run at step
        # 25 after 24 committed steps. rank 6 sampled a token id with no
        # vocabulary entry, so `convert_ids_to_tokens` returned None and
        # transformers raised
        #     TypeError: sequence item 481: expected str instance, NoneType found
        # inside `tokenizer.decode`. A TypeError is not a RuntimeError, so it
        # escaped, rank 6 died, and the other fifteen ranks sat on a collective
        # until the NCCL watchdog aborted them. An out-of-vocabulary sampled id
        # is model output by definition -- exactly the class A-1 says must be
        # classified.
        return {
            "version": VERSION,
            "status": "unparseable",
            # The old handler called `tokenizer.decode(ids)` here -- the very
            # operation that had just failed -- so widening the catch alone
            # would have raised again from inside the handler.
            "decoded_sampled": _safe_decode(tokenizer, ids),
            "sampled_token_ids": ids,
            "g022_capture_classified": True,
            "g022_capture_error": f"{type(exc).__name__}: {exc}",
            "g022_capture_error_is_decode_failure": True,
        }


__all__ = [
    "CLASSIFICATIONS",
    "ROUTED",
    "VERSION",
    "TokenRoutingLineageError",
    "build_g022_token_masks",
    "capture_sampled_field_spans_classified",
]
