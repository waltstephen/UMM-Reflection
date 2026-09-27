"""G022 item 2: bounds that do not kill the gradient, and a hard divergence stop.

Design notes: section 5, Appendix J and Appendix M-2.

`torch.clamp` has **zero gradient outside its range**. Once a token's
log-probability delta passes the clamp bound the penalty stops exerting any
force on it: the token is free to diverge without limit while the logged loss
sits at a constant. This is the mechanism that produced the G021-B blowup --
23 of 54 steps over loss 100, 60k-88k of it entirely KL, driven by 5-16 tokens
out of ~13,000 whose delta climbed from 5 to 45 nats.

Two live-path sites have the defect (M-2, verified):

* `flow_grpo/g008_counting.py:753`  -- the PPO **ratio**  `exp(clamp(d, -20, 20))`
* `flow_grpo/g008_counting.py:786`  -- the **KL delta**   `clamp(d, -20, 20)`

and the same code exists in the legacy
`unify_rl/train/bagel_online_rollout_flow_text_grpo.py:17566` and
`unify_rl/train/flow_grpo_gen_helper.py::flow_k3_reference_kl`.

The replacement is a C1 linearization of `exp` above a threshold `T`:

```
bounded_exp(d) = exp(d)                     d <= T
               = exp(T) * (1 + d - T)       d >  T
```

Value and first derivative agree at `d = T`, and the derivative above `T` is the
constant `exp(T)` -- large, finite, and **never zero**. Below `T` nothing
changes, so ordinary training is bit-for-bit the same as an unclamped `exp`.

Consequences:

* KL `k3 = bounded_exp(d) - 1 - d` keeps `dk3/dd = exp(T) - 1 > 0` for every
  `d > T`, so a token 45 nats out is still being pulled back;
* `k3` stays non-negative everywhere (it is non-negative at `T` and increasing
  above it), so the KL cannot go negative;
* the PPO ratio can no longer overflow fp32 (plain `exp` overflows past ~88).

The bound is deliberately *not* a safety net on its own. Doc section 5 also
requires a hard runtime stop on `max |delta|` so that divergence halts the run
instead of being silently absorbed; that is `assert_delta_within_stop` below,
which the trainer applies at the post-update world-consensus point.
"""
from __future__ import annotations

import math
from typing import Any

import torch

VERSION = "clean29529_g022_non_vanishing_kl_bound_v1"

# Linearize `exp` above this many nats. exp(10) ~= 2.2e4, so a token past the
# threshold receives a constant restoring gradient of that size instead of zero,
# while values stay ~4 orders of magnitude below the exp(20) ~= 4.9e8 that the
# old clamp permitted to accumulate with no gradient at all.
LINEARIZATION_THRESHOLD = 10.0

# Hard runtime stop, doc section 5 part 3. Past this the run halts rather than
# absorbing the divergence. G021-B reached 45 nats.
MAX_ABS_DELTA_STOP = 20.0


def bounded_exp(
    delta: torch.Tensor,
    *,
    threshold: float = LINEARIZATION_THRESHOLD,
) -> torch.Tensor:
    """`exp(delta)`, linearized above `threshold`, with a non-vanishing slope."""

    limit = float(threshold)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("G022 exp linearization threshold must be positive")
    value = delta.float()
    # excess is 0 below the threshold and (delta - T) above it, and carries the
    # whole gradient above it. `lower` is therefore delta below the threshold
    # and the constant T above it, including exactly at the boundary.
    excess = torch.clamp(value - limit, min=0.0)
    lower = value - excess
    return torch.exp(lower) + math.exp(limit) * excess


def bounded_k3(
    delta: torch.Tensor,
    *,
    threshold: float = LINEARIZATION_THRESHOLD,
) -> torch.Tensor:
    """Schulman k3 = exp(d) - 1 - d with the exp bounded as above."""

    value = delta.float()
    return bounded_exp(value, threshold=threshold) - 1.0 - value


def bounded_ratio(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    *,
    threshold: float = LINEARIZATION_THRESHOLD,
) -> torch.Tensor:
    """PPO importance ratio with the same bound. Replaces exp(clamp(d, -20, 20))."""

    return bounded_exp(
        new_log_probs.float() - old_log_probs.to(
            device=new_log_probs.device, dtype=torch.float32
        ),
        threshold=threshold,
    )


def delta_diagnostics(delta: torch.Tensor, *, threshold: float = LINEARIZATION_THRESHOLD) -> dict[str, float]:
    detached = delta.detach().float()
    if detached.numel() == 0:
        return {
            "max_abs_delta": 0.0,
            "max_delta": 0.0,
            "min_delta": 0.0,
            "linearized_token_count": 0,
            "linearization_threshold": float(threshold),
        }
    return {
        "max_abs_delta": float(detached.abs().max().item()),
        "max_delta": float(detached.max().item()),
        "min_delta": float(detached.min().item()),
        "linearized_token_count": int((detached > float(threshold)).sum().item()),
        "linearization_threshold": float(threshold),
    }


def assert_delta_within_stop(
    max_abs_delta: Any,
    *,
    limit: float = MAX_ABS_DELTA_STOP,
    channel: str = "text",
) -> float:
    """Hard runtime stop, doc section 5. Divergence halts instead of being absorbed.

    This is a numerical-safety invariant, not a model-output classification, so
    it stays fail-closed (item 3 converts only model-output-triggered raises).
    """

    value = float(max_abs_delta)
    bound = float(limit)
    if not math.isfinite(value):
        raise RuntimeError(
            f"G022 {channel} KL delta is nonfinite: {max_abs_delta!r}"
        )
    if value > bound:
        raise RuntimeError(
            f"G022 {channel} KL divergence stop: max |logp delta| = {value:.4f} nats "
            f"exceeds the {bound:.4f} nat limit. The bounded k3 keeps a "
            f"non-vanishing gradient past {LINEARIZATION_THRESHOLD} nats, but a "
            "delta this large means the policy has left the reference "
            "distribution and the run stops rather than absorbing it."
        )
    return value


__all__ = [
    "LINEARIZATION_THRESHOLD",
    "MAX_ABS_DELTA_STOP",
    "VERSION",
    "assert_delta_within_stop",
    "bounded_exp",
    "bounded_k3",
    "bounded_ratio",
    "delta_diagnostics",
]
