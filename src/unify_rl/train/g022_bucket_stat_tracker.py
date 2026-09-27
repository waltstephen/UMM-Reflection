"""G022 item 7: a historical baseline keyed by bucket, not by prompt.

Design notes: section 3, as revised by Appendix L-4.

Section 3 identified official Flow-GRPO's `PerPromptStatTracker`
(`flow_grpo/stat_tracking.py:16-31`) as the third and least-examined cause of
dead groups, and proposed porting it: a prompt's rewards accumulate across steps
and the running mean becomes the baseline, so a group whose K members all agree
is **not** dead -- the advantage measures this step against that prompt's own
history.

**That port does not work at our scale.** Its value comes entirely from a prompt
recurring often enough to build a history. Official gets ~92 exposures per
prompt over its run. We get **2.3** (173 unique prompts, 2 roots x 200 steps =
400 draws). A per-prompt history two entries deep is not a baseline; it is noise
with extra bookkeeping.

L-4: keep the mechanism, change the key.

    bucket key = (target_count, q_0)

Both fields are known before any action is taken. `target_count` has 5 values
and `q_0` has 13 (Appendix K-1: `counting_score` takes only 13 distinct values
over targets 2..6), so the key space is at most **65 cells against 400 draws**
and histories are deep from the first few steps.

Neither field leaks detector state into the policy observation: the bucket is
used by the advantage estimator, never by the model. And the bucket mean is a
valid baseline regardless of how well it matches any individual state -- any
action-independent baseline leaves the expected gradient unmoved (doc section
1). Bucketing is a variance choice, not a correctness one.

Two things kept from section 3 unchanged: the **additive** `std + 1e-4` guard
(not the `max(sigma, 0.1)` floor, which stays on the within-group path), and
`zero_std_ratio` logged per step from step 1.

Section 3 also warns about the upstream implementation: it stacks
`self.stats[prompt]` into an ndarray inside the same call, which breaks a later
`.extend()`. **Port the idea, not the code** -- this uses a plain deque and is
tested on the repeat path, which is exactly where upstream would break.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

VERSION = "clean29529_g022_bucket_historical_baseline_v1"

# Official Flow-GRPO's additive guard, kept as-is (section 3, unchanged by L-4).
STD_EPSILON = 1e-4
# How many historical entries a bucket needs before it is trusted as a baseline.
# Below this the group falls back to the within-group z-score of section 2.4 --
# reported, never silent.
DEFAULT_MIN_COUNT = 16
# Per-bucket ring buffer. 65 cells x 256 entries is trivially small and bounds
# how far back a bucket's history reaches.
DEFAULT_BUFFER_SIZE = 256


def bucket_key(target_count: Any, q_0: Any) -> str:
    """`(target_count, q_0)` as a stable string key.

    `q_0` comes off a 13-value lattice, so rounding is exact in practice and
    guards against float formatting drift across processes.
    """

    target = int(target_count)
    initial = float(q_0)
    if not math.isfinite(initial) or not 0.0 <= initial <= 1.0:
        raise ValueError(f"G022 bucket q_0 must be a score in [0,1]: {q_0!r}")
    return f"{target}|{round(initial, 6):.6f}"


class BucketStatTracker:
    """Historical reward statistics keyed by `(target_count, q_0)`."""

    def __init__(
        self,
        *,
        min_count: int = DEFAULT_MIN_COUNT,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        clip: float = 1.0,
    ) -> None:
        if int(min_count) < 2:
            raise ValueError("G022 bucket baseline needs min_count >= 2")
        if int(buffer_size) < int(min_count):
            raise ValueError("G022 bucket buffer cannot be smaller than min_count")
        self.min_count = int(min_count)
        self.buffer_size = int(buffer_size)
        self.clip = float(clip)
        self._stats: dict[str, deque[float]] = {}
        self.observed_reward_count = 0

    # -- history ---------------------------------------------------------

    def extend(self, key: str, rewards: Iterable[float]) -> None:
        """Append this step's rewards to a bucket's history.

        Deliberately a plain deque of floats. Upstream stacks the buffer into an
        ndarray inside the same call and then cannot `.extend()` it again;
        `test_a_bucket_survives_being_read_then_extended_again` is the case that
        breaks there.
        """

        values = [float(value) for value in rewards]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("G022 bucket baseline received a nonfinite reward")
        buffer = self._stats.setdefault(key, deque(maxlen=self.buffer_size))
        buffer.extend(values)
        self.observed_reward_count += len(values)

    def history(self, key: str) -> list[float]:
        return list(self._stats.get(key, ()))

    def depth(self, key: str) -> int:
        return len(self._stats.get(key, ()))

    # -- baseline --------------------------------------------------------

    def baseline(self, key: str) -> dict[str, Any] | None:
        """`(mean, std + 1e-4)` for a bucket deep enough to be trusted."""

        values = self.history(key)
        if len(values) < self.min_count:
            return None
        return {
            "mean": statistics.mean(values),
            "population_std": statistics.pstdev(values),
            "scale": statistics.pstdev(values) + STD_EPSILON,
            "depth": len(values),
        }

    def advantages(
        self,
        *,
        records: Sequence[Mapping[str, Any]],
        rewards: Sequence[float],
        within_group: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Advantage for one shared-R0 sibling group.

        The whole group shares one `(target_count, q_0)` bucket, because every
        sibling starts from the same R0. Uses the bucket's history when it is
        deep enough, otherwise the within-group z-score of section 2.4 --
        and **says which**, per group, so the mix is visible rather than assumed.

        History is updated *after* the baseline is read, so a group is never
        centred on itself.
        """

        values = [float(value) for value in rewards]
        keys = {
            bucket_key(row["target_count"], float(row["counting_scores"][0]))
            for row in records
        }
        if len(keys) != 1:
            raise ValueError(
                "G022 sibling group spans more than one (target_count, q_0) bucket"
            )
        key = next(iter(keys))
        history = self.baseline(key)
        if history is None:
            self.extend(key, values)
            return {
                "version": VERSION,
                "bucket_key": key,
                "baseline_source": "within_group",
                "reason": (
                    f"bucket depth {self.depth(key)} < min_count {self.min_count}"
                ),
                "advantages": list(within_group["advantages"]),
                "bucket_depth_before_update": self.depth(key) - len(values),
                "bucket_depth_after_update": self.depth(key),
                "zero_std": bool(within_group["zero_std"]),
                "saturation_rate": float(within_group["saturation_rate"]),
            }
        raw = [(value - history["mean"]) / history["scale"] for value in values]
        advantages = [max(-self.clip, min(self.clip, value)) for value in raw]
        self.extend(key, values)
        return {
            "version": VERSION,
            "bucket_key": key,
            "baseline_source": "bucket_history",
            "reason": "",
            "advantages": advantages,
            "raw_z_scores": raw,
            "baseline_mean": history["mean"],
            "baseline_population_std": history["population_std"],
            "baseline_scale": history["scale"],
            "bucket_depth_before_update": history["depth"],
            "bucket_depth_after_update": self.depth(key),
            # A group whose K members all agree is NOT dead under a historical
            # baseline -- this is the whole point of section 3.
            "within_group_zero_std": bool(within_group["zero_std"]),
            "zero_std": bool(within_group["zero_std"])
            and all(abs(value) <= 1e-12 for value in advantages),
            "saturation_rate": (
                sum(abs(value) >= self.clip - 1e-12 for value in raw) / len(raw)
            ),
        }

    # -- persistence -----------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "min_count": self.min_count,
            "buffer_size": self.buffer_size,
            "clip": self.clip,
            "observed_reward_count": self.observed_reward_count,
            "stats": {key: list(value) for key, value in sorted(self._stats.items())},
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        """Restore across a resume. Without this the history resets to empty and
        every bucket silently falls back to the within-group z-score."""

        if str(payload.get("version")) != VERSION:
            raise ValueError("G022 bucket baseline state version differs")
        self.min_count = int(payload["min_count"])
        self.buffer_size = int(payload["buffer_size"])
        self.clip = float(payload["clip"])
        self.observed_reward_count = int(payload.get("observed_reward_count", 0))
        self._stats = {
            str(key): deque(
                (float(value) for value in values), maxlen=self.buffer_size
            )
            for key, values in dict(payload.get("stats") or {}).items()
        }

    def save(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(
            json.dumps(self.state_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)

    @classmethod
    def load(cls, path: Path | str, **kwargs: Any) -> "BucketStatTracker":
        tracker = cls(**kwargs)
        source = Path(path)
        if source.is_file():
            tracker.load_state_dict(json.loads(source.read_text(encoding="utf-8")))
        return tracker

    # -- reporting -------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        depths = [len(value) for value in self._stats.values()]
        ready = [depth for depth in depths if depth >= self.min_count]
        return {
            "version": VERSION,
            "bucket_count": len(self._stats),
            "ready_bucket_count": len(ready),
            "min_count": self.min_count,
            "observed_reward_count": self.observed_reward_count,
            "median_bucket_depth": statistics.median(depths) if depths else 0,
            "max_bucket_depth": max(depths, default=0),
        }


def step_zero_std_ratio(group_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """`zero_std_ratio` as a first-class per-step metric, doc section 3.

    Official logs this (`scripts/train_bagel.py:137-171`); we never have.

    C2 / finding Q-9. The primary number used to be computed from the bucket
    history's `zero_std`, and `rescued_by_history_count` counted groups the
    history had "rescued". Under O-3 the bucket baseline reaches **no
    gradient at all** -- L-4 is withdrawn -- so those groups' actual gradient
    is zero and reporting them as rescued describes a mechanism that no longer
    runs. T1.3 fixed the gradient and left this describing the old design.

    The primary `zero_std_ratio` is therefore the **within-group** ratio, which
    is the one that corresponds to real gradient. The history figures are kept
    under explicitly monitor-only names, because the withdrawn claim should
    stay falsifiable, and are prefixed so they cannot be mistaken for the
    effective number.

    Appendix N-1: `zero_std_ratio` stays **log-only with no stop condition**.
    This docstring previously carried Appendix H-9 #2's instruction to halt the
    run past a ~40% threshold. N-1 supersedes it, so the instruction is removed
    rather than left sitting in the code for someone to implement, and
    `stop_condition: None` is returned so the decision is visible in the
    artifact instead of only in a comment.
    """

    total = len(group_reports)
    if not total:
        return {
            "version": VERSION,
            "group_count": 0,
            "zero_std_ratio": 0.0,
            "within_group_zero_std_ratio": 0.0,
            "monitor_only_history_zero_std_ratio": 0.0,
            "monitor_only_history_would_have_rescued_count": 0,
            "bucket_history_group_count": 0,
            "gradient_source": "within_group_grpo_only",
            "stop_condition": None,
        }
    within = sum(
        bool(row.get("within_group_zero_std", row.get("zero_std")))
        for row in group_reports
    )
    history = sum(bool(row.get("zero_std")) for row in group_reports)
    return {
        "version": VERSION,
        "group_count": total,
        # The effective number: what the gradient actually sees.
        "zero_std_ratio": within / total,
        "within_group_zero_std_ratio": within / total,
        # Monitor-only. L-4 is withdrawn (O-3); these groups' gradient IS zero.
        "monitor_only_history_zero_std_ratio": history / total,
        "monitor_only_history_would_have_rescued_count": within - history,
        "bucket_history_group_count": sum(
            row.get("baseline_source") == "bucket_history" for row in group_reports
        ),
        "gradient_source": "within_group_grpo_only",
        "applied_to_gradient": False,
        "withdrawn_appendix": "L-4 (reversed by O-3)",
        "stop_condition": None,
    }


__all__ = [
    "DEFAULT_BUFFER_SIZE",
    "DEFAULT_MIN_COUNT",
    "STD_EPSILON",
    "VERSION",
    "BucketStatTracker",
    "bucket_key",
    "step_zero_std_ratio",
]
