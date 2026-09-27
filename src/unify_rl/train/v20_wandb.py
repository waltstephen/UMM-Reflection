"""Optional rank-0 Weights & Biases logging for the RL trainer.

Disabled unless ``G016_WANDB_ENABLED=1``.

* The API key is read only from ``WANDB_API_KEY`` or ``~/.netrc``. It is never
  returned, stored in the run config, or written to a log line.
* Only rank 0 logs; every other rank is a no-op.
* Training nodes may have no egress, so connectivity is probed first and the
  logger falls back to ``WANDB_MODE=offline`` for a later ``wandb sync``.
* Every wandb call is wrapped, so a wandb failure never blocks or crashes a
  training step.
"""

from __future__ import annotations

import math
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION = "clean29529_v20_wandb_logging_v1"
PROJECT = os.environ.get("WANDB_PROJECT", "unify-rl-reflection")
ENTITY = os.environ.get("WANDB_ENTITY") or None
WANDB_HOST = "api.wandb.ai"
WANDB_PORT = 443
EGRESS_PROBE_TIMEOUT_SEC = 5.0
NETRC_PATHS = (Path.home() / ".netrc",)
CHANNEL_NAMES = (
    "r0_flow",
    "repair_flow_round_0",
    "repair_flow_round_1",
    "controller_round_0",
    "controller_round_1",
)


def utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def run_name(ladder_stage: str, *, date: str | None = None) -> str:
    stage = "".join(
        character
        for character in str(ladder_stage or "").strip().lower()
        if character.isalnum() or character in {"-", "_"}
    )
    if not stage:
        raise ValueError("V20 wandb run name requires a ladder stage")
    return f"v20-{stage}-{date or utc_date()}"


def credentials_present() -> dict[str, Any]:
    """Report *whether* credentials exist. The key itself never leaves here."""
    if os.environ.get("WANDB_API_KEY"):
        return {"present": True, "source": "WANDB_API_KEY"}
    for path in NETRC_PATHS:
        try:
            if not path.is_file():
                continue
            import netrc

            entry = netrc.netrc(str(path)).authenticators(WANDB_HOST)
        except Exception:
            continue
        if entry and entry[2]:
            return {"present": True, "source": f"netrc:{path}"}
    return {"present": False, "source": None}


def probe_egress(
    *,
    host: str = WANDB_HOST,
    port: int = WANDB_PORT,
    timeout_sec: float = EGRESS_PROBE_TIMEOUT_SEC,
) -> dict[str, Any]:
    """TCP-probe the wandb endpoint. Never raises."""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_sec):
            return {"reachable": True, "host": host, "port": int(port)}
    except Exception as exc:
        return {
            "reachable": False,
            "host": host,
            "port": int(port),
            "error": f"{type(exc).__name__}",
        }


def resolve_mode(
    *,
    egress: dict[str, Any] | None = None,
    credentials: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Choose online/offline/disabled without ever consulting the key value."""
    forced = str(os.environ.get("WANDB_MODE") or "").strip().lower()
    if forced in {"offline", "disabled", "online"}:
        return {"mode": forced, "reason": "WANDB_MODE"}
    credentials = credentials or credentials_present()
    if not credentials["present"]:
        return {"mode": "offline", "reason": "no_credentials"}
    egress = egress if egress is not None else probe_egress()
    if not egress.get("reachable"):
        return {"mode": "offline", "reason": "no_egress"}
    return {"mode": "online", "reason": "egress_and_credentials"}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_run_config(
    *,
    ladder_stage: str,
    experiment_record_path: str,
    logdir: str,
    r0_channel_weight: float,
    group_size: int,
    world_size: int,
    candidates_per_rank: int,
    flow_learning_rate: float,
    text_learning_rate: float,
    initialization_checkpoint: str,
    judge_concurrency_version: str,
    code_hashes: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run config. Contains no credential material by construction."""
    config = {
        "version": VERSION,
        "ladder_stage": str(ladder_stage),
        "experiment_record_path": str(experiment_record_path),
        "logdir": str(logdir),
        "r0_channel_weight": float(r0_channel_weight),
        "K": int(group_size),
        "world_size": int(world_size),
        "candidates_per_rank": int(candidates_per_rank),
        "flow_learning_rate": float(flow_learning_rate),
        "text_learning_rate": float(text_learning_rate),
        "initialization_checkpoint": str(initialization_checkpoint),
        "judge_concurrency_version": str(judge_concurrency_version),
        "code_hashes": {
            str(key): str(value) for key, value in dict(code_hashes).items()
        },
    }
    if extra:
        config.update(
            {
                str(key): value
                for key, value in dict(extra).items()
                if "key" not in str(key).lower()
                and "token" not in str(key).lower()
                and "secret" not in str(key).lower()
            }
        )
    return config


def build_step_metrics(row: dict[str, Any]) -> dict[str, float]:
    """Map one V20 metrics row onto the fixed metric set."""
    metrics: dict[str, float] = {}

    def put(name: str, value: Any) -> None:
        number = _finite(value)
        if number is not None:
            metrics[name] = number

    put("train/reward_mean", row.get("reward_mean"))
    put("train/reward_std", row.get("reward_std"))
    done = row.get("done_calibration") or {}
    put("train/done_rate", done.get("done_rate"))
    protocol = row.get("controller_protocol") or {}
    put(
        "train/controller_invalid_rate",
        row.get("controller_invalid_rate", protocol.get(
            "controller_invalid_rate"
        )),
    )
    put(
        "train/edit_none_alias_rate",
        row.get("edit_none_alias_rate", protocol.get("edit_none_alias_rate")),
    )
    safety = row.get("numerical_safety") or {}
    put("train/nonfinite_metric_count", safety.get("nonfinite_metric_count"))
    put("train/flow_kl_max", safety.get("flow_kl_max"))
    put(
        "train/text_ratio_first_inner_epoch",
        safety.get(
            "text_ratio_first_inner_epoch_max_abs_deviation",
            safety.get("text_ratio_max_abs_deviation"),
        ),
    )
    max_abs = row.get("channel_advantage_max_abs") or {}
    put("channel/max_abs_advantage/r0_flow", max_abs.get("r0_flow"))
    for round_index, value in enumerate(max_abs.get("repair_flow") or []):
        put(
            f"channel/max_abs_advantage/repair_flow_round_{round_index}",
            value,
        )
    for round_index, value in enumerate(max_abs.get("controller") or []):
        put(
            f"channel/max_abs_advantage/controller_round_{round_index}",
            value,
        )
    for round_index, value in enumerate(
        row.get("repair_flow_channel_active") or []
    ):
        put(f"channel/repair_flow_active/{round_index}", float(bool(value)))
    put(
        "channel/r0_gradient_share",
        row.get("r0_flow_gradient_share", protocol.get(
            "r0_flow_gradient_share"
        )),
    )
    # Mandatory per-repair-round state
    # leak, plus the share of repairs launched on an already-perfect image.
    # V19 archive reference: correlation -0.990, already-perfect rate 57%.
    leak = row.get("repair_credit_state_leak") or {}
    for entry in leak.get("per_repair_round") or []:
        index = int(entry.get("repair_round_index", -1))
        if index < 0:
            continue
        put(f"repair/credit_state_leak/round_{index}", entry.get(
            "correlation"
        ))
        put(f"repair/credit_state_leak_slope/round_{index}", entry.get(
            "slope"
        ))
        put(f"repair/eligible_count/round_{index}", entry.get(
            "eligible_count"
        ))
    put(
        "repair/credit_state_leak_count_weighted_slope",
        leak.get("count_weighted_slope"),
    )
    put(
        "repair/on_already_perfect_rate",
        row.get(
            "repair_on_already_perfect_rate",
            leak.get("repair_on_already_perfect_rate"),
        ),
    )
    put(
        "repair/executed_repair_count",
        leak.get("executed_repair_count"),
    )
    residualization = row.get("repair_residualization") or {}
    if "enabled" in residualization:
        put(
            "repair/residualization_enabled",
            float(bool(residualization.get("enabled"))),
        )
    curve = row.get("per_round_score_curve") or []
    for entry in curve:
        if int(entry.get("round_index", -1)) == 0:
            put("capability/r0_score_mean", entry.get("score_mean"))
    improvement = row.get("self_improvement") or {}
    final_mean = improvement.get("final_score_mean")
    if final_mean is None and curve:
        final_mean = curve[-1].get("score_mean")
    put("capability/final_score_mean", final_mean)
    r0_mean = _finite(safety.get("r0_detector_score_mean"))
    if r0_mean is None:
        r0_mean = _finite(metrics.get("capability/r0_score_mean"))
    final_value = _finite(final_mean)
    if r0_mean is not None and final_value is not None:
        put("capability/final_minus_r0", final_value - r0_mean)
    put(
        "capability/repair_success_rate_on_failed_r0",
        row.get(
            "repair_success_rate",
            improvement.get("repair_success_rate_on_bad_r0"),
        ),
    )
    judge_latency = (row.get("judge_latency") or {})
    put("judge/latency_p50", judge_latency.get("p50_sec"))
    put("judge/latency_p90", judge_latency.get("p90_sec"))
    put(
        "judge/coverage_loss",
        row.get(
            "judge_coverage_loss",
            max(
                0,
                int(row.get("updated_sample_count") or 0)
                - int(row.get("judge_available_trajectory_count") or 0),
            ),
        ),
    )
    fallback = row.get("content_policy_fallback") or {}
    put(
        "judge/fallback_count",
        float(bool(fallback.get("active"))),
    )
    dashboard = row.get("self_improvement") or {}
    put(
        "safety/detector_failed_done_positive_credit",
        dashboard.get("detector_failed_positive_done_count", 0),
    )
    return metrics


class V20WandbLogger:
    """Rank-0-only, never-raising wandb logger."""

    def __init__(
        self,
        *,
        rank: int,
        ladder_stage: str,
        run_config: dict[str, Any],
        offline_dir: Path | str,
        enabled: bool = True,
        date: str | None = None,
    ) -> None:
        self.version = VERSION
        self.rank = int(rank)
        self.ladder_stage = str(ladder_stage)
        self.enabled = bool(enabled) and self.rank == 0
        self.run_config = dict(run_config)
        self.offline_dir = Path(offline_dir)
        self.run = None
        self.mode: str | None = None
        self.status: dict[str, Any] = {
            "version": VERSION,
            "rank": self.rank,
            "rank_zero_only": True,
            "active": False,
            "mode": None,
            "run_name": None,
            "project": PROJECT,
            "entity": ENTITY,
            "logged_step_count": 0,
            "failure_count": 0,
            "last_error": None,
            "credentials_source": None,
            "egress": None,
            "date": date or utc_date(),
        }

    def start(self) -> dict[str, Any]:
        if not self.enabled:
            self.status["skip_reason"] = (
                "non_zero_rank" if self.rank != 0 else "disabled"
            )
            return dict(self.status)
        try:
            credentials = credentials_present()
            egress = probe_egress()
            decision = resolve_mode(egress=egress, credentials=credentials)
            self.status["credentials_source"] = credentials["source"]
            self.status["egress"] = egress
            self.status["mode_reason"] = decision["reason"]
            self.mode = decision["mode"]
            self.status["mode"] = self.mode
            name = run_name(self.ladder_stage, date=self.status["date"])
            self.status["run_name"] = name
            if self.mode == "disabled":
                return dict(self.status)
            import wandb

            self.offline_dir.mkdir(parents=True, exist_ok=True)
            os.environ["WANDB_DIR"] = str(self.offline_dir)
            self.run = wandb.init(
                project=PROJECT,
                entity=ENTITY,
                name=name,
                mode=self.mode,
                dir=str(self.offline_dir),
                config=self.run_config,
                reinit=True,
            )
            self.status["active"] = True
            self.status["wandb_version"] = str(
                getattr(wandb, "__version__", "")
            )
            self.status["run_dir"] = str(
                getattr(getattr(self.run, "dir", None), "__str__", str)()
            )
        except Exception as exc:
            self.run = None
            self.status["active"] = False
            self.status["failure_count"] += 1
            self.status["last_error"] = f"{type(exc).__name__}: {exc}"[:400]
        return dict(self.status)

    def log_step(self, row: dict[str, Any]) -> dict[str, float]:
        metrics = build_step_metrics(row)
        if not self.enabled or self.run is None:
            return metrics
        try:
            self.run.log(metrics, step=int(row.get("step", 0)))
            self.status["logged_step_count"] += 1
        except Exception as exc:
            self.status["failure_count"] += 1
            self.status["last_error"] = f"{type(exc).__name__}: {exc}"[:400]
        return metrics

    def finish(self) -> dict[str, Any]:
        if self.enabled and self.run is not None:
            try:
                self.run.finish()
            except Exception as exc:
                self.status["failure_count"] += 1
                self.status["last_error"] = (
                    f"{type(exc).__name__}: {exc}"[:400]
                )
            self.run = None
        return dict(self.status)


def redacted_status(status: dict[str, Any]) -> dict[str, Any]:
    """Persist-safe status: strips anything that could carry a secret."""
    forbidden = ("key", "token", "secret", "password", "netrc_line")
    return {
        str(name): value
        for name, value in dict(status).items()
        if not any(word in str(name).lower() for word in forbidden)
    }


__all__ = [
    "CHANNEL_NAMES",
    "ENTITY",
    "PROJECT",
    "VERSION",
    "V20WandbLogger",
    "build_run_config",
    "build_step_metrics",
    "credentials_present",
    "probe_egress",
    "redacted_status",
    "resolve_mode",
    "run_name",
    "utc_date",
]
