"""Incremental monitoring history without retaining full rollout diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RUNTIME_PERFORMANCE_REVISION = "no_checkpoint_scan_incremental_metrics_20260906"


def monitoring_row(row: dict[str, Any]) -> dict[str, Any]:
    value = {
        key: row[key]
        for key in (
            "step", "completed_at_utc", "completion_interval_sec",
            "numerical_safety", "g017_protocol_degradation",
            "stopnow_live_metrics", "rollout_elapsed_sec", "update_elapsed_sec",
            "phase_timings",
        )
        if key in row
    }
    reward = row.get("reward_diagnostics", {})
    value["reward_diagnostics"] = {
        key: reward[key]
        for key in (
            "controller_protocol", "trajectory_count",
            "false_done_count", "parse_failure_count",
        )
        if key in reward
    }
    monitoring = row.get("g016_live_monitoring", {})
    value["g016_live_monitoring"] = {
        "step_sufficient_statistics": monitoring.get("step_sufficient_statistics")
    }
    return value


class MetricHistory:
    """Read old rows once, then only bytes appended since the previous call."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.offset = 0
        self.identity: tuple[Any, ...] | None = None

    def read(self, path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            self.rows = []
            self.offset = 0
            self.identity = None
            return self.rows
        stat = path.stat()
        identity = (str(path.resolve()), stat.st_dev, stat.st_ino)
        if identity != self.identity or stat.st_size < self.offset:
            self.rows = []
            self.offset = 0
            self.identity = identity
        with path.open("rb") as handle:
            handle.seek(self.offset)
            while line := handle.readline():
                if line.strip():
                    self.rows.append(monitoring_row(json.loads(line)))
                self.offset = handle.tell()
        return self.rows
