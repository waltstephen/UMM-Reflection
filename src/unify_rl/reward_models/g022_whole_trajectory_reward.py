"""G022 whole-trajectory GRPO: one scalar per trajectory, broadcast to every token.

Doc section 2, with the section 2.2 correction, Appendix H-3 (`p = 0.5`),
Appendix F (advantage clip +/-1 is intentional), Appendix D-7 (deadzone deleted)
and Appendix A-1 (a malformed action is classified and penalised, never raised).

G008-G021 all assigned credit **per round**. The lineage's own postmortem
diagnosed the per-round baseline as degenerate on broken states, and the response
was to remove baselines entirely rather than fix them. G021 ended with no repair
baseline at all, a mean repair advantage of +0.0272, and a mathematically forced
rise in P(EDIT) that collapsed the DONE rate.

G022 stops splitting credit across rounds:

    R(tau) = q_T
           + alpha * sum_{t=0}^{T-1} w_t * [q_{t+1} - q_t]_+     reflection bonus
           + beta  * sum_t min([q_{t+1}-q_t]_+, [q_{t+2}-q_{t+1}]_+)  sustained (S-3)
           - lambda * sum_{t=0}^{T-1} [q_t - q_{t+1}]_+          regression penalty
           - p * 1[DONE emitted while broken]                    premature DONE

    A_i = clip((R_i - mu_R) / max(sigma_R, 0.1), -1, +1)
    A_i = min(A_i, 0) - m      if the trajectory was malformed   (R-6)

**THE OBJECTIVE, Appendix S-1.** G022 optimises **final correctness under an
externally maintained multi-round rollout distribution**, and **measures**
continued improvement as a secondary evaluation outcome. It does **not**
directly optimise "keep improving across rounds", and no report may claim that
it does. The reflection bonus can express at most
`alpha*(w_max-w_0)*Delta = 0.12*Delta` as a late-vs-early preference at fixed
endpoints -- 0.55 sigma for a full 0->1 repair and 0.055 sigma for a typical
0.1 improvement against a measured group std of 0.2165, with about a third of
advantages saturating at the clip. S-2: with a scalar trajectory reward, fixed
endpoints and cost-free no-ops, ANY strict preference for moving the same
improvement later necessarily creates a stalling incentive. That is structural,
not a coefficient-selection problem, so no value of alpha or w fixes it.

broadcast to every policy-active token of that trajectory: controller text
tokens, payload tokens, and GEN/flow transitions alike.

**One head.** No action head, no repair head, no leave-one-out, no
return-to-go, no per-round buckets, no deadzone. This module deliberately does
not import `g016_repair_first_reward`, and takes no `deadzone` argument.

Two coefficient constraints, doc section 2.2, both asserted at import:

* **The bonus sum starts at t = 0.** Starting at t = 1 makes the reward
  strictly prefer stalling: a no-op costs nothing (`[0]_+ = [0]_- = 0`) and
  moves the same total improvement into a higher-weighted slot. 65.9% of
  measured rollout rounds already have `delta q == 0` exactly, so the model can
  produce free no-ops at will. Summing from t = 0 with `w_0 > 0` reduced the
  stall gain from `alpha*w_1` to `alpha*(w_1 - w_0)`.
* **`lambda >= alpha * w_max`.** Damage `d` at t=0 followed by full recovery at
  t=1 gives `R_B - R_A = alpha*(w_1 - w_0)*(q_T - q_0) + d*(alpha*w_1 - lambda)`.
  The `d` coefficient must be non-positive or the model can farm the bonus by
  first breaking the image.

**`w` IS NOW FLAT: `(1.0, 1.0, 1.0)`**. Reducing the
stall gain to `alpha*(w_1 - w_0)` was not enough -- it left exactly +0.0420,
and a no-op followed by a repair scored 1.2520 against 1.2100 for the same
repair in one round, so stalling still outranked an honest one-shot repair.
Flat weights make it exactly +0.0000. The measured cost is nil: dead-group rate
29.0% -> 29.5%, std median identical at 0.2165, saturation 23.3% -> 23.6%.
Consecutive improvement stays first in the ordering and loses only 0.0210,
because Appendix S-6's `S_multi` carries that preference rather than `w`.

**Reflection does not require monotone improvement.** A single large repair is
legitimate reflection, and so is damage-then-recovery. Both already earn well
through `q_T` and the reflection bonus. `S_multi` scoring them zero means only
that they did not improve on more than one round -- **it is not the same as
their earning nothing**, and no comment or report should imply otherwise.

With `w` flat, `w_max = 1.0`, so `lambda >= alpha * w_max` relaxes from 0.42 to
0.30. **Lambda stays at 0.5**; the headroom is recorded as an option if
oscillation needs harder penalising, not as a pending change.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Mapping, Sequence

from unify_rl.reward_models.g009_counting_process_reward import (
    GROUP_SIZE as G009_GROUP_SIZE,
    build_trajectory_process_credit,
)
from unify_rl.reward_models.g011_stopnow_reward import (
    STOP_NOW_SCORER_IDENTITY,
    STOP_NOW_SCORER_VERSION,
)
from unify_rl.train.g022_campaign import (
    G022_ADVANTAGE_CLIP as ADVANTAGE_CLIP,
    G022_ALPHA as ALPHA,
    G022_LAMBDA as LAMBDA,
    G022_MALFORMED_ACTION_PENALTY as MALFORMED_ACTION_PENALTY,
    G022_MALFORMED_ADVANTAGE_PENALTY as MALFORMED_ADVANTAGE_PENALTY,
    G022_SUSTAINED_PROGRESS_ARCHIVED_FIRE_RATE as SUSTAINED_PROGRESS_FIRE_RATE,
    G022_SUSTAINED_PROGRESS_FORM as SUSTAINED_PROGRESS_FORM,
    G022_SUSTAINED_PROGRESS_RANKING_CHANGE_RATE as SUSTAINED_RANKING_CHANGE_RATE,
    G022_SUSTAINED_PROGRESS_BETA as BETA,
    G022_SUSTAINED_PROGRESS_BREAK_EVEN_SECOND_EDIT_SUCCESS as SUSTAINED_BREAK_EVEN,
    G022_PREMATURE_DONE_PENALTY as PREMATURE_DONE_PENALTY,
    G022_REFLECTION_WEIGHTS as REFLECTION_WEIGHTS,
    G022_SIBLINGS_PER_ROOT as SIBLINGS_PER_ROOT,
    G022_STD_FLOOR as STD_FLOOR,
)

VERSION = "clean29529_g022_whole_trajectory_grpo_v1"
CONTROLLER_CREDIT_VERSION = VERSION
REPAIR_FLOW_CREDIT_VERSION = VERSION
REPAIR_BUCKET_VERSION = VERSION
REWARD_VERSION = "clean29529_g022_trajectory_reward_v1"

EPSILON_WEIGHT = 1e-12

# Doc section 2.2, checked here rather than trusted.
if not LAMBDA >= ALPHA * max(REFLECTION_WEIGHTS):
    raise AssertionError(
        f"G022 requires lambda >= alpha * w_max: {LAMBDA} < {ALPHA * max(REFLECTION_WEIGHTS)}"
    )
if not REFLECTION_WEIGHTS[0] > 0.0:
    raise AssertionError("G022 requires w_0 > 0 or the bonus sum cannot start at t=0")
# INVERTED. This used to require the weights to
# INCREASE with round index, which is the property that created the stall gain:
# a no-op costs nothing, so moving the same improvement into a higher-weighted
# slot paid `alpha * (w_later - w_earlier)`. Measured at exactly +0.0420 --
# a no-op then a repair scored 1.2520 against 1.2100 for the same repair in one
# round -- so stalling outranked an honest one-shot repair, against 65.9% of
# rounds already being cost-free no-ops.
#
# The requirement is now the property itself rather than a shape: no later
# weight may exceed an earlier one, so the stall gain can never be positive.
# Flat weights are the boundary case and are what ships; the check would also
# admit strictly decreasing weights, which would make stalling actively costly.
if not all(
    earlier >= later
    for earlier, later in zip(REFLECTION_WEIGHTS, REFLECTION_WEIGHTS[1:])
):
    raise AssertionError(
        "G022 reflection weights must not increase with round index, or "
        "stalling pays: " + repr(REFLECTION_WEIGHTS)
    )
if not max(REFLECTION_WEIGHTS) - min(REFLECTION_WEIGHTS) <= EPSILON_WEIGHT:
    raise AssertionError(
        "G022 ships FLAT reflection weights; a non-flat set is a deliberate "
        "change that must be re-derived against the stall gain: "
        + repr(REFLECTION_WEIGHTS)
    )

EPSILON = 1e-12


def build_g022_trajectory_credit(
    *,
    trajectory_index: int,
    siblings_per_root: int = SIBLINGS_PER_ROOT,
    **kwargs: Any,
) -> dict[str, Any]:
    """Frozen G009 credit for a G022 trajectory, without touching the frozen file.

    `build_trajectory_process_credit` validates `0 <= trajectory_index < 28`,
    the historical K=28 group size. G022 runs 2 roots x 16 siblings = **32**
    trajectories, so global indexes 28..31 would raise.

    `src/unify_rl/reward_models/g009_counting_process_reward.py` is hash-pinned
    by the trainer (`FROZEN_G009_REWARD_SHA256`) precisely so the frozen
    counting reward cannot drift, so it must not be edited. Instead the frozen
    builder is called with the trajectory's index **within its sibling group**
    (0..15, always inside the K=28 bound) and the true global index is restored
    on the returned record.

    This is provably equivalent: inside the frozen builder `trajectory_index` is
    used for exactly two things -- the range check, and being echoed back as
    `record["trajectory_index"]`. No reward, score, return or advantage depends
    on it. Asserted by `test_the_index_remap_changes_nothing_but_the_index`.
    """

    siblings = int(siblings_per_root)
    if not 0 < siblings <= G009_GROUP_SIZE:
        raise ValueError(
            f"G022 sibling group of {siblings} does not fit the frozen "
            f"G009 bound of {G009_GROUP_SIZE}"
        )
    index = int(trajectory_index)
    if index < 0:
        raise ValueError("G022 trajectory index must be non-negative")
    credit = build_trajectory_process_credit(
        trajectory_index=index % siblings, **kwargs
    )
    credit["trajectory_index"] = index
    credit["g022_frozen_g009_index_remap"] = {
        "global_index": index,
        "index_passed_to_frozen_builder": index % siblings,
        "siblings_per_root": siblings,
        "frozen_reward_file_edited": False,
    }
    return credit


def _positive(value: float) -> float:
    """`[x]_+ = max(x, 0)`."""
    return value if value > 0.0 else 0.0


def trajectory_reward(
    counting_scores: Sequence[float],
    *,
    premature_done: bool,
    alpha: float = ALPHA,
    reflection_weights: Sequence[float] = REFLECTION_WEIGHTS,
    regression_lambda: float = LAMBDA,
    premature_done_penalty: float = PREMATURE_DONE_PENALTY,
    sustained_beta: float = BETA,
) -> dict[str, Any]:
    """`R(tau)` for one complete trajectory, with its four terms itemised.

    `counting_scores` is `q_0 .. q_T`, one entry per image state: `q_0` is the
    detached R0 and each later entry is one executed EDIT. `T = len(scores) - 1`.

    The terms are returned separately because Appendix H-4 measured their
    variance shares (q_T 47.0%, premature DONE 24.2%, regression 16.5%,
    reflection bonus **12.2%**) and the reflection bonus is the smallest
    contributor. It is the number to watch and the first ablation knob, so it
    must be visible per trajectory rather than folded into a scalar.

    S-1 corrects what this docstring used to say about that term: it called the
    reflection bonus "the term that encodes the entire goal of this project".
    It does not. At fixed endpoints it can express at most `0.12*Delta` of
    late-vs-early preference, which is 0.055 sigma for a typical improvement,
    and the archived corpus cannot test it -- matched on `(q_0, q_T,
    edit_count)` no cell holds >=5 trajectories on both sides. The objective is
    final correctness under an externally maintained multi-round distribution;
    continued improvement is MEASURED, in `g022_evaluation_metrics`, not
    optimised here.
    """

    scores = [float(value) for value in counting_scores]
    if not scores:
        raise ValueError("G022 trajectory reward requires at least q_0")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in scores):
        raise ValueError("G022 counting scores must be finite and in [0, 1]")
    horizon = len(scores) - 1
    weights = [float(value) for value in reflection_weights]
    if horizon > len(weights):
        raise ValueError(
            f"G022 has {horizon} transitions but only {len(weights)} reflection weights"
        )

    # Both sums run t = 0 .. T-1. Starting the bonus at t = 0 is the section 2.2
    # correction; starting at t = 1 makes stalling strictly optimal.
    deltas = [scores[t + 1] - scores[t] for t in range(horizon)]
    reflection_bonus = float(alpha) * sum(
        weights[t] * _positive(deltas[t]) for t in range(horizon)
    )
    # Appendix S-3 as AMENDED by S-6: the multi-improvement form. Sum every
    # positive per-round improvement EXCEPT the largest one, so a trajectory
    # that improved on exactly one round earns exactly zero no matter how large
    # that improvement was, and a stalled or damaged prefix earns zero because
    # a non-positive delta contributes nothing. `w` asks which round improved;
    # this asks whether MORE THAN ONE round improved.
    #
    # S-6 replaced the originally specified ADJACENT form. Measured on 1600
    # archived trajectories / 400 groups: adjacency fires on 1.19% and reorders
    # ONE group in 400 -- inert. Improvement at round >= 2 does happen (266 of
    # 934 trajectories with >=2 edits) but almost never in ADJACENT rounds (19
    # of 934), so adjacency was an artificial constraint. This form fires on
    # 2.12% and reorders 4 of 400.
    #
    # `S_nonregress` was rejected despite an 11.31% fire rate because it pays
    # 0.7000 for a single large repair, which must be zero: it re-rewards
    # improvement and duplicates q_T rather than expressing sustained progress.
    # **Fire rate is never the criterion; the four incentive cases are.**
    #
    # `sustained_progress_raw` is the bare sum, which is what the S-3/S-4(b)/S-6
    # tables quote (0.35 for two consecutive repairs, 0.60 for three, 0.375 for
    # the split case). `sustained_progress_bonus` is beta times it and is the
    # actual reward contribution. Both are reported because the appendix uses
    # the same letter for the two.
    positive_improvements = sorted(
        value for value in (_positive(delta) for delta in deltas) if value > 0.0
    )
    sustained_progress_raw = sum(positive_improvements[:-1])
    sustained_progress_bonus = float(sustained_beta) * sustained_progress_raw
    regression_penalty = float(regression_lambda) * sum(
        _positive(-deltas[t]) for t in range(horizon)
    )
    done_penalty = float(premature_done_penalty) if premature_done else 0.0

    # R-6. The malformed term used to be subtracted
    # HERE, inside R(tau), and then group-normalised. Centering removes anything
    # constant across the group and keeps only the ordering, so a penalty placed
    # inside R is RELATIVE, not absolute:
    #
    #   all four siblings malformed -> R = [-0.2]*4  -> A = [0, 0, 0, 0]
    #   one malformed-but-exact vs three that regressed
    #                               -> R = [0.71, 0, 0, 0] -> A_malformed = +1.0
    #
    # The first cancels the penalty entirely; the second hands the malformed
    # trajectory the MAXIMUM positive advantage. That is the failure mode that
    # took G021-B from 0.6% to 16.4% invalid. The term is now applied after
    # normalisation, in `apply_malformed_advantage_penalty`, with a guaranteed
    # sign.
    reward = (
        scores[-1]
        + reflection_bonus
        + sustained_progress_bonus
        - regression_penalty
        - done_penalty
    )
    if not math.isfinite(reward):
        raise RuntimeError("G022 trajectory reward is nonfinite")
    return {
        "version": REWARD_VERSION,
        "reward": reward,
        "terminal_q": scores[-1],
        "initial_q": scores[0],
        "reflection_bonus": reflection_bonus,
        # Appendix S-3/S-4. Cited together, always: S fires on 2.0% of archived
        # trajectories (19 of 934 with >=2 edits), so in a 16-sibling group the
        # expectation is 0.3 trajectories and GRPO will rarely have a
        # within-group contrast. It is a sparse rare-event bonus, not a bulk
        # gradient mover.
        "sustained_progress_raw": sustained_progress_raw,
        "sustained_progress_bonus": sustained_progress_bonus,
        "sustained_progress_beta": float(sustained_beta),
        "sustained_progress_form": SUSTAINED_PROGRESS_FORM,
        "sustained_progress_is_sparse_rare_event": True,
        "sustained_progress_archived_fire_rate": SUSTAINED_PROGRESS_FIRE_RATE,
        # S-7: the fire rate alone overstates the effect, because a fire that
        # does not reorder its group changes no gradient. Never cite one
        # without the other.
        "sustained_progress_ranking_change_rate": SUSTAINED_RANKING_CHANGE_RATE,
        "sustained_progress_improving_round_count": len(positive_improvements),
        "sustained_progress_break_even_second_edit_success": SUSTAINED_BREAK_EVEN,
        "regression_penalty": regression_penalty,
        "premature_done_penalty": done_penalty,
        # R-6: no longer a term of R(tau). Kept as an explicit zero so a reader
        # of an archived report can tell "not malformed" from "malformed, and
        # the penalty is applied downstream".
        "malformed_action_penalty": 0.0,
        "malformed_penalty_in_reward": False,
        "malformed_penalty_applied_after_normalization": True,
        "transition_deltas": deltas,
        "horizon": horizon,
        "alpha": float(alpha),
        "reflection_weights": list(weights[:horizon]),
        "lambda": float(regression_lambda),
        "bonus_sum_starts_at_t0": True,
        "deadzone_used": False,
        "return_to_go_used": False,
    }


def premature_done_emitted(credit: Mapping[str, Any]) -> bool:
    """`1[DONE emitted while broken]`, doc section 2.3.

    Fires **only on an actually emitted DONE action**. A trajectory that
    exhausts `max_repair_rounds` while still broken was truncated by the
    sampler, not by the policy, and must not be penalised -- so
    `stop_reason == "environment_cap"` never counts, and neither does a parse
    error, which carries its own classification penalty instead.
    """

    actions = [str(value).casefold() for value in credit["actions"]]
    if not actions or actions[-1] != "done":
        return False
    return not bool(credit["terminal_strict_exact_success"])


def _malformed_action(credit: Mapping[str, Any]) -> bool:
    """Any unroutable/malformed controller action in this trajectory, A-1.

    Reads the classification the rollout recorded. `parse_valid` False is the
    reward-side view of the same event, so either is sufficient.
    """

    if not bool(credit.get("parse_valid", True)):
        return True
    if "invalid" in [str(value).casefold() for value in credit["actions"]]:
        return True
    return any(
        bool(round_row.get("g022_malformed_action"))
        for round_row in credit.get("rounds") or []
    )


def group_advantages(rewards: Sequence[float]) -> dict[str, Any]:
    """`A_i = clip((R_i - mu)/max(sigma, 0.1), -1, +1)`, doc section 2.4.

    The std floor is 0.1, not `clamp_min(1e-6)`: at 1e-6 a near-degenerate group
    produces astronomically large advantages. The clip saturates roughly a third
    of samples at K=16, which is a legitimate variance control and far more
    aggressive than the +/-5 used previously, so the saturation rate is returned
    and must be logged.
    """

    values = [float(value) for value in rewards]
    if len(values) < 2:
        raise ValueError("G022 advantage requires a group of at least two trajectories")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("G022 group reward is nonfinite")
    mean = statistics.mean(values)
    population_std = statistics.pstdev(values)
    scale = max(population_std, STD_FLOOR)
    raw = [(value - mean) / scale for value in values]
    advantages = [max(-ADVANTAGE_CLIP, min(ADVANTAGE_CLIP, value)) for value in raw]
    saturated = sum(abs(value) >= ADVANTAGE_CLIP - EPSILON for value in raw)
    return {
        "version": VERSION,
        "advantages": advantages,
        "raw_z_scores": raw,
        "reward_mean": mean,
        "reward_population_std": population_std,
        "scale": scale,
        "std_floor_binds": population_std < STD_FLOOR,
        "zero_std": population_std <= EPSILON,
        "clip": ADVANTAGE_CLIP,
        "saturated_count": saturated,
        "saturation_rate": saturated / len(values),
        "group_size": len(values),
        "leave_one_out_used": False,
        "per_round_bucket_used": False,
    }


def apply_malformed_advantage_penalty(
    advantages: Sequence[float],
    malformed: Sequence[bool],
    *,
    penalty: float = MALFORMED_ADVANTAGE_PENALTY,
    clip: float = ADVANTAGE_CLIP,
) -> dict[str, Any]:
    """R-6: the malformed penalty, applied AFTER normalisation, sign-guaranteed.

        A_i = min(A_i, 0.0) - penalty          for a malformed trajectory

    **Why not inside `R(tau)`.** Group centering subtracts the group mean, which
    removes anything constant across the group and preserves only the ordering.
    A per-trajectory constant subtracted from `R` is therefore relative, not
    absolute. Both failure cases were reproduced:

        all siblings malformed        R = [-0.2]*4      -> A = [0, 0, 0, 0]
        one malformed-but-exact vs
        three that regressed          R = [0.71,0,0,0]  -> A_malformed = +1.0

    The first cancels the penalty completely; the second gives the malformed
    trajectory the maximum positive advantage -- the G021-B failure mode
    (0.6% -> 16.4% invalid) with a gradient pushing it.

    **The invariant, which is the thing to test:** a malformed trajectory can
    never receive a non-negative advantage, for ANY sibling configuration.
    `min(A_i, 0.0)` discards whatever rank the trajectory earned, and the
    subtraction then guarantees strict negativity.

    **Accepted cost, stated so nobody tries to solve it** (R-6): with
    one scalar per trajectory we cannot punish the malformed round and still
    reward the good rounds of the same trajectory. That is inherent to
    whole-trajectory GRPO. At the measured 3.75% malformed rate,
    under-crediting a recovered trajectory is far cheaper than reinforcing
    malformed syntax.

    **One addition to the stated formula**: the result is floored at `-clip`, so
    the advantage stays inside the +/-1 contract that doc section 2.4 states and
    that the diagnostics report as `max_abs_active_action_advantage`. With
    `penalty <= clip` the floor cannot reach zero, so the invariant is
    untouched; it only stops the penalised value reaching -1.5.
    """

    if len(advantages) != len(malformed):
        raise ValueError("G022 malformed penalty needs one flag per trajectory")
    if not 0.0 < float(penalty) <= float(clip):
        raise ValueError(
            "G022 malformed advantage penalty must be positive and no larger "
            "than the advantage clip, or the sign guarantee fails"
        )
    penalised = []
    for value, flag in zip(advantages, malformed):
        if not flag:
            penalised.append(float(value))
            continue
        value = min(float(value), 0.0) - float(penalty)
        penalised.append(max(value, -float(clip)))
    if any(
        flag and value >= 0.0 for value, flag in zip(penalised, malformed)
    ):
        raise RuntimeError(
            "G022 malformed trajectory received a non-negative advantage"
        )
    return {
        "version": VERSION,
        "advantages": penalised,
        "malformed_count": sum(1 for flag in malformed if flag),
        "penalty": float(penalty),
        "clip": float(clip),
        "applied_after_normalization": True,
        "sign_guaranteed_negative": True,
        "spec": "R-6",
    }


def assign_g022_advantages(
    credits: Iterable[Mapping[str, Any]],
    *,
    require_exact_group: bool = True,
    historical_baseline: Any = None,
    reward_fn: Any = None,
) -> dict[str, Any]:
    """Project one shared-R0 sibling group under whole-trajectory GRPO.

    `historical_baseline` is item 7 and is optional here: when supplied it must
    expose `.advantages(records)`; when absent the advantage is the within-group
    z-score of section 2.4.
    """

    rows = [dict(value) for value in credits]
    if not rows:
        raise ValueError("G022 advantage assignment requires a group")
    if require_exact_group and len(rows) != SIBLINGS_PER_ROOT:
        raise ValueError(
            f"G022 requires one shared-R0 sibling group of {SIBLINGS_PER_ROOT}"
        )
    ordered = sorted(rows, key=lambda value: int(value["trajectory_index"]))
    indexes = [int(value["trajectory_index"]) for value in ordered]
    if len(set(indexes)) != len(indexes):
        raise ValueError("G022 trajectory indexes are duplicated")

    identities = {
        (
            str(value.get("prompt") or ""),
            str(value.get("uid") or ""),
            str(value.get("r0_sha256") or ""),
            int(value.get("target_count", -1)),
        )
        for value in ordered
    }
    if len(identities) != 1 or "" in next(iter(identities))[:3]:
        raise ValueError("G022 sibling group mixed prompt/UID/R0 state")
    if any(
        bool(value.get("r0_policy_active"))
        or int(value.get("r0_text_record_count", -1)) != 0
        or int(value.get("r0_flow_record_count", -1)) != 0
        or value.get("r0_advantage") is not None
        or value.get("r0_reward") is not None
        for value in ordered
    ):
        raise ValueError("G022 detached-R0 contract differs")

    rewards = []
    for row in ordered:
        premature = premature_done_emitted(row)
        malformed = _malformed_action(row)
        # `reward_fn` is G025's injection point. It receives the whole row so a
        # later campaign can price terminal decisions the row knows about (e.g.
        # a CORRECT done) without this function learning those concepts. When
        # it is None -- every G022 and G024 run -- the call below is literally
        # the original one.
        if reward_fn is None:
            report = trajectory_reward(
                row["counting_scores"],
                premature_done=premature,
            )
        else:
            report = reward_fn(row, premature_done=premature)
        report["premature_done"] = premature
        report["malformed_action"] = malformed
        row["g022_reward_report"] = report
        rewards.append(report["reward"])

    group = group_advantages(rewards)
    baseline_report: dict[str, Any] | None = None
    # ------------------------------------------------------------------
    # O-3: Appendix L-4 is WITHDRAWN. Gradients use within-group GRPO only.
    #
    # L-4 centred the advantage on a (target_count, q_0) bucket's history. That
    # turns a sibling group whose 16 members all score identically, sitting
    # above its bucket history, into sixteen identical +1 advantages -- which
    # contains no comparison between siblings at all. It says "this bucket did
    # better than usual", not "this trajectory beat its siblings".
    #
    # And the bucket key deliberately pools DIFFERENT prompts: umbrella@6 and
    # sheep@6 at the same q_0 share a bucket, so within a bucket R varies mostly
    # by which prompt was drawn, not by what the policy did. Subtracting the
    # bucket mean leaves prompt-difficulty noise in the advantage.
    #
    # K-4 measured that 22.2% of all groups are "everything already succeeded".
    # There the correct advantage IS zero. L-4 manufactured a high-variance
    # non-zero signal exactly where there was nothing to learn.
    #
    # The tracker is retained as MONITOR-ONLY: its report is computed and
    # logged, and never touches `advantages`.
    # ------------------------------------------------------------------
    advantages = list(group["advantages"])
    # R-6: the malformed penalty is applied HERE, after normalisation, because
    # inside R(tau) group centering removed it. `malformed_penalty_report`
    # carries the pre-penalty advantages so the effect is auditable.
    malformed_flags = [bool(row["g022_reward_report"]["malformed_action"]) for row in ordered]
    malformed_penalty_report = apply_malformed_advantage_penalty(
        advantages, malformed_flags
    )
    malformed_penalty_report["advantages_before_penalty"] = list(advantages)
    advantages = list(malformed_penalty_report["advantages"])
    if historical_baseline is not None:
        baseline_report = historical_baseline.advantages(
            records=ordered, rewards=rewards, within_group=group
        )
        baseline_report = {
            **baseline_report,
            "monitor_only": True,
            "applied_to_gradient": False,
            "withdrawn_appendix": "L-4 (reversed by O-3)",
        }

    exact_round_count = 0
    edit_when_exact = 0
    root_r0_exact = bool(
        int(ordered[0]["detected_counts"][0]) == int(ordered[0]["target_count"])
    )
    records = []
    for row, reward, advantage in zip(ordered, rewards, advantages):
        report = row["g022_reward_report"]
        actions = [str(value).casefold() for value in row["actions"]]
        scores = [float(value) for value in row["counting_scores"]]
        rounds = []
        controller_advantages: list[float] = []
        controller_active: list[bool] = []
        repair_flow_advantages: list[float] = []
        repair_flow_active: list[bool] = []
        edit_position = 0
        for round_index, action in enumerate(actions):
            source = dict(row["rounds"][round_index])
            is_edit = action == "edit"
            q_before = scores[edit_position] if is_edit else scores[-1]
            q_after = scores[edit_position + 1] if is_edit else scores[-1]
            exact_before = (
                int(row["detected_counts"][edit_position]) == int(row["target_count"])
                if is_edit
                else int(row["detected_counts"][-1]) == int(row["target_count"])
            )
            malformed_round = bool(source.get("g022_malformed_action"))
            if exact_before:
                exact_round_count += 1
                if is_edit:
                    edit_when_exact += 1
            # ONE head: the same trajectory scalar reaches the controller text,
            # the payload and the flow transition -- INCLUDING a malformed
            # round. T1.4 (O-5) routed the advantage here so the penalty could
            # reach a parameter at all; R-6 is what makes it a penalty. This
            # comment previously read "R(tau) already charges the -0.5 malformed
            # penalty, so the trajectory advantage is negative". That was FALSE:
            # the penalty was inside R and group centering removed it, so the
            # advantage could be zero (all siblings malformed) or +1.0 (a
            # malformed trajectory ranking first). Under R-6 the penalty is
            # applied after normalisation with a guaranteed sign, so the
            # advantage reaching these tokens is now negative by construction.
            active = True
            controller_advantages.append(advantage if active else 0.0)
            controller_active.append(active)
            if is_edit:
                flow_active = bool(active and not malformed_round)
                repair_flow_advantages.append(advantage if flow_active else 0.0)
                repair_flow_active.append(flow_active)
            source.update(
                {
                    "version": VERSION,
                    "round_index": round_index,
                    "action": action,
                    "semantic_action": "stop" if action == "done" else action,
                    "actionable_edit": is_edit and not malformed_round,
                    "malformed_penalty_trains": malformed_round,
                    "exact_before_action": exact_before,
                    "q_before": q_before,
                    "q_after": q_after,
                    "advantage": advantage if active else 0.0,
                    "action_head_advantage": advantage if active else 0.0,
                    "action_head_active": active,
                    "repair_head_advantage": (
                        advantage if active and is_edit and not malformed_round else 0.0
                    ),
                    "repair_head_active": bool(active and is_edit and not malformed_round),
                    "repair_payload_active": bool(active and is_edit and not malformed_round),
                    "repair_flow_active": bool(active and is_edit and not malformed_round),
                    "controller_policy_active": active,
                    "repair_flow_policy_active": bool(active and is_edit),
                    "trajectory_advantage_broadcast": True,
                    "g016_two_head_credit_applied": True,
                    "g022_malformed_action": malformed_round,
                    "repair_progress_deadzone": 0.0,
                }
            )
            source.setdefault("g016_decision", {})
            rounds.append(source)
            if is_edit:
                edit_position += 1
        if edit_position != int(row["edit_count"]):
            raise RuntimeError("G022 round-to-EDIT lineage coverage differs")

        records.append(
            {
                **row,
                "advantage_version": VERSION,
                "channel_credit_version": VERSION,
                "controller_credit_version": CONTROLLER_CREDIT_VERSION,
                "renderer_credit_version": REPAIR_FLOW_CREDIT_VERSION,
                "stop_now_scorer_identity": STOP_NOW_SCORER_IDENTITY,
                "stop_now_scorer_version": STOP_NOW_SCORER_VERSION,
                "controller_advantages": controller_advantages,
                "controller_active": controller_active,
                "repair_flow_advantages": repair_flow_advantages,
                "repair_flow_active": repair_flow_active,
                "rounds": rounds,
                "g022_trajectory_reward": reward,
                "g022_trajectory_advantage": advantage,
                "trajectory_scalar_advantage_broadcast": True,
                "r0_policy_active": False,
                "r0_text_record_count": 0,
                "r0_flow_record_count": 0,
                "r0_reward": None,
                "r0_advantage": None,
            }
        )

    identity = next(iter(identities))
    active_values = [
        abs(float(value))
        for record in records
        for value, active in zip(
            record["controller_advantages"], record["controller_active"]
        )
        if active
    ]
    return {
        "version": VERSION,
        "records": records,
        "channel_credit_version": VERSION,
        "controller_credit_version": CONTROLLER_CREDIT_VERSION,
        "renderer_credit_version": REPAIR_FLOW_CREDIT_VERSION,
        "renderer_bucket_version": REPAIR_BUCKET_VERSION,
        "stop_now_scorer_identity": STOP_NOW_SCORER_IDENTITY,
        "stop_now_scorer_version": STOP_NOW_SCORER_VERSION,
        "group_identity": {
            "prompt": identity[0],
            "uid": identity[1],
            "r0_sha256": identity[2],
            "target_count": identity[3],
        },
        "trajectory_count": len(records),
        "controller_group_size": len(records),
        "group_reward_mean": group["reward_mean"],
        "group_reward_population_std": group["reward_population_std"],
        "group_reward_scale": group["scale"],
        "std_floor": STD_FLOOR,
        "std_floor_binds": group["std_floor_binds"],
        # G022 has no per-round buckets, so the analogous quantity is whether
        # this sibling group's own population std fell below the 0.1 floor.
        # Section 2.4 requires that to be visible: at a 1e-6 floor a
        # near-degenerate group produces astronomically large advantages, and
        # the whole point of raising it to 0.1 is that the binding rate is
        # something you can watch. The trainer sums this across roots.
        "std_floor_binding_bucket_count": int(bool(group["std_floor_binds"])),
        "zero_std": group["zero_std"],
        "advantage_clip": ADVANTAGE_CLIP,
        "advantage_saturated_count": group["saturated_count"],
        "advantage_saturation_rate": group["saturation_rate"],
        "group_rewards": list(rewards),
        "group_advantages": list(advantages),
        "within_group_advantages": list(group["advantages"]),
        # T1.3/O-3's invariant is about the BASELINE: a zero-variance group
        # must not be handed a manufactured non-zero advantage. It is therefore
        # checked on the pre-penalty advantages. R-6's malformed penalty is a
        # deliberate exception and the reason the first reproduced case exists:
        # an all-malformed group IS zero-variance and MUST come out negative,
        # not zero. Both facts are reported separately so neither hides the
        # other.
        "zero_variance_group_stays_zero": (
            not group["zero_std"]
            or all(value == 0.0 for value in group["advantages"])
        ),
        "zero_variance_group_penalised_only_when_malformed": (
            not group["zero_std"]
            or all(
                (value < 0.0) == bool(flag)
                for value, flag in zip(advantages, malformed_flags)
            )
        ),
        "malformed_advantage_penalty": malformed_penalty_report,
        # P(EDIT | the image is ALREADY exact) is the
        # quantity that should govern the exact-R0 downsampling target, and it
        # cannot be measured offline -- 0.15 was chosen by judgement precisely
        # because no drift curve exists yet. G018 moved this from 0.667 to
        # 0.442; if it drifts back up under G022, the target is too low and the
        # damage-avoidance signal is being starved. `exact_before_action` is
        # hidden reward-side detector state and is used here for MONITORING
        # only -- it never enters the policy observation.
        "p_edit_given_exact": (
            edit_when_exact / exact_round_count if exact_round_count else None
        ),
        "exact_before_action_round_count": exact_round_count,
        "edit_when_already_exact_count": edit_when_exact,
        # Dead-group rate split by root state, the other half of what the
        # downsampler has to be judged on. Archived K=4 reference: exact-R0
        # groups 51.6% dead with median std 0.0000, broken-R0 14.6% dead with
        # median std 0.2279. **K=16 is UNMEASURED** -- it cannot be simulated
        # from a corpus with four siblings per root.
        "root_state": "r0_exact" if root_r0_exact else "r0_broken",
        "root_r0_exact": bool(root_r0_exact),
        "historical_baseline": baseline_report,
        "historical_baseline_applied_to_gradient": False,
        "gradient_advantage_source": "within_group_grpo_only",
        "reward_terms": {
            "alpha": ALPHA,
            "reflection_weights": list(REFLECTION_WEIGHTS),
            "lambda": LAMBDA,
            "premature_done_penalty": PREMATURE_DONE_PENALTY,
            # R-6: no longer a term of R(tau); reported as 0.0 here and carried
            # on the advantage scale under `malformed_advantage_penalty`.
            "malformed_action_penalty": 0.0,
            "malformed_advantage_penalty": MALFORMED_ADVANTAGE_PENALTY,
            "malformed_penalty_applied_after_normalization": True,
            "bonus_sum_starts_at_t0": True,
            "lambda_ge_alpha_w_max": LAMBDA >= ALPHA * max(REFLECTION_WEIGHTS),
        },
        # Explicit statements of what G022 is and is not, mirroring the
        # diagnostics every prior stage carries.
        "trajectory_level_advantage_broadcast": True,
        "round_diagnostics": [],
        "g016_bucket_diagnostics": [],
        "repair_progress_deadzone": 0.0,
        "deadzone_used": False,
        "leave_one_out_used": False,
        "return_to_go_used": False,
        "per_round_bucket_used": False,
        "action_head_used": False,
        "repair_head_used": False,
        "critic_used": False,
        "gae_used": False,
        "hard_projection": False,
        "empirical_mean_used_as_center": False,
        "exact_nonexact_mixed_bucket_count": 0,
        "r0_policy_active_count": 0,
        "r0_text_record_count": 0,
        "r0_flow_record_count": 0,
        "stop_now_table_invariants": {"passed": True, "not_applicable": True},
        "max_abs_active_action_advantage": max(active_values, default=0.0),
        "max_abs_active_repair_advantage": max(active_values, default=0.0),
        "repair_routed_record_count": sum(
            sum(1 for value in record["repair_flow_active"] if value)
            for record in records
        ),
        "exact_damage_override_active_count": 0,
        "malformed_action_trajectory_count": sum(
            bool(record["g022_reward_report"]["malformed_action"])
            for record in records
        ),
        "premature_done_trajectory_count": sum(
            bool(record["g022_reward_report"]["premature_done"])
            for record in records
        ),
    }


__all__ = [
    "ADVANTAGE_CLIP",
    "ALPHA",
    "CONTROLLER_CREDIT_VERSION",
    "LAMBDA",
    "MALFORMED_ACTION_PENALTY",
    "BETA",
    "MALFORMED_ADVANTAGE_PENALTY",
    "SUSTAINED_PROGRESS_FIRE_RATE",
    "SUSTAINED_PROGRESS_FORM",
    "SUSTAINED_RANKING_CHANGE_RATE",
    "apply_malformed_advantage_penalty",
    "PREMATURE_DONE_PENALTY",
    "REFLECTION_WEIGHTS",
    "REPAIR_BUCKET_VERSION",
    "REPAIR_FLOW_CREDIT_VERSION",
    "REWARD_VERSION",
    "STD_FLOOR",
    "VERSION",
    "assign_g022_advantages",
    "build_g022_trajectory_credit",
    "group_advantages",
    "premature_done_emitted",
    "trajectory_reward",
]
