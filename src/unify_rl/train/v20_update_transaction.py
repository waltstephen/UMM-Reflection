"""Synchronized logical-update transaction state for V20."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch.distributed as dist


VERSION = "clean29529_v20_update_transaction_v1"
PHASES = {
    "idle",
    "update_started",
    "text_backward_complete",
    "text_phase_complete",
    "flow_backward_complete",
    "flow_phase_complete",
    "local_update_complete",
    "commit_synchronized",
    "metrics_persisted",
    "checkpoint_durable",
}


@dataclass
class LogicalUpdateTransaction:
    completed_step: int
    durable_step: int
    durable_checkpoint: str
    durable_origin_kind: str = "resumable_optimizer_checkpoint"
    optimizer_rng_resumable: bool = True
    current_step: int | None = None
    phase: str = "idle"
    text_step_state: str = "none"
    flow_step_state: str = "none"
    commit_synchronized: bool = False
    metrics_persisted: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)

    def _event(self, name: str, **values: Any) -> None:
        self.events.append({"event": str(name), **values})

    def begin(self, logical_step: int) -> None:
        if (
            self.current_step is not None
            or self.phase not in {"idle", "checkpoint_durable"}
            or int(logical_step) != int(self.completed_step) + 1
        ):
            raise RuntimeError("V20 logical update transaction cannot begin")
        self.current_step = int(logical_step)
        self.phase = "update_started"
        self.text_step_state = "none"
        self.flow_step_state = "none"
        self.commit_synchronized = False
        self.metrics_persisted = False
        self._event("begin", logical_step=int(logical_step))

    def observe_phase(
        self,
        phase: str,
    ) -> None:
        transitions = {
            "after_text_backward": (
                "update_started",
                "text_backward_complete",
            ),
            "after_text_phase": (
                "text_backward_complete",
                "text_phase_complete",
            ),
            "after_flow_backward": (
                "text_phase_complete",
                "flow_backward_complete",
            ),
            "after_flow_phase": (
                "flow_backward_complete",
                "flow_phase_complete",
            ),
        }
        if phase not in transitions or self.current_step is None:
            raise RuntimeError("V20 update transaction phase is invalid")
        expected, target = transitions[phase]
        if self.phase != expected:
            raise RuntimeError("V20 update transaction phase is out of order")
        if (
            phase == "after_text_phase"
            and self.text_step_state not in {"stepped", "skipped"}
        ) or (
            phase == "after_flow_phase"
            and self.flow_step_state not in {"stepped", "skipped"}
        ):
            raise RuntimeError(
                "V20 optimizer step result is missing before phase completion"
            )
        self.phase = target
        self._event(phase)

    def mark_optimizer_step_possible(self, channel: str) -> None:
        if channel == "text":
            if (
                self.phase != "text_backward_complete"
                or self.text_step_state != "none"
            ):
                raise RuntimeError("V20 text step marker is out of order")
            self.text_step_state = "possible"
        elif channel == "flow":
            if (
                self.phase != "flow_backward_complete"
                or self.flow_step_state != "none"
            ):
                raise RuntimeError("V20 flow step marker is out of order")
            self.flow_step_state = "possible"
        else:
            raise ValueError("V20 optimizer channel is invalid")
        self._event("optimizer_step_possible", channel=channel)

    def mark_optimizer_step_result(
        self,
        channel: str,
        *,
        stepped: bool,
    ) -> None:
        target = "stepped" if stepped else "skipped"
        if channel == "text":
            if self.phase != "text_backward_complete":
                raise RuntimeError("V20 text step result is out of order")
            if self.text_step_state not in {"none", "possible"}:
                raise RuntimeError("V20 text step result was already recorded")
            self.text_step_state = target
        elif channel == "flow":
            if self.phase != "flow_backward_complete":
                raise RuntimeError("V20 flow step result is out of order")
            if self.flow_step_state not in {"none", "possible"}:
                raise RuntimeError("V20 flow step result was already recorded")
            self.flow_step_state = target
        else:
            raise ValueError("V20 optimizer channel is invalid")
        self._event(
            "optimizer_step_result",
            channel=channel,
            stepped=bool(stepped),
        )

    def record_local_update_complete(
        self,
        *,
        text_stepped: bool,
        flow_stepped: bool,
    ) -> None:
        if self.current_step is None or self.phase != "flow_phase_complete":
            raise RuntimeError("V20 local update completion is out of order")
        self.text_step_state = "stepped" if text_stepped else "skipped"
        self.flow_step_state = "stepped" if flow_stepped else "skipped"
        self.phase = "local_update_complete"
        self._event(
            "local_update_complete",
            text_stepped=bool(text_stepped),
            flow_stepped=bool(flow_stepped),
        )

    def mark_commit_synchronized(self) -> None:
        if self.current_step is None or self.phase != "local_update_complete":
            raise RuntimeError("V20 synchronized commit is out of order")
        self.completed_step = int(self.current_step)
        self.commit_synchronized = True
        self.phase = "commit_synchronized"
        self._event(
            "commit_synchronized",
            logical_step=int(self.completed_step),
        )

    def mark_metrics_persisted(self) -> None:
        if not self.commit_synchronized:
            raise RuntimeError("V20 metrics preceded synchronized commit")
        self.metrics_persisted = True
        self.phase = "metrics_persisted"
        self._event("metrics_persisted")

    def mark_checkpoint_durable(
        self,
        *,
        logical_step: int,
        checkpoint: str,
    ) -> None:
        if (
            int(logical_step) != int(self.completed_step)
            or not self.commit_synchronized
            or not self.metrics_persisted
        ):
            raise RuntimeError("V20 durable checkpoint step differs")
        self.durable_step = int(logical_step)
        self.durable_checkpoint = str(checkpoint)
        self.durable_origin_kind = "resumable_optimizer_checkpoint"
        self.optimizer_rng_resumable = True
        self.current_step = None
        self.phase = "checkpoint_durable"
        self.text_step_state = "none"
        self.flow_step_state = "none"
        self.commit_synchronized = False
        self.metrics_persisted = False
        self._event(
            "checkpoint_durable",
            logical_step=int(logical_step),
            checkpoint=str(checkpoint),
        )

    def finish_without_checkpoint(self) -> None:
        if not self.commit_synchronized or not self.metrics_persisted:
            raise RuntimeError("V20 update cannot finish before persistence")
        self.current_step = None
        self.phase = "idle"
        self.text_step_state = "none"
        self.flow_step_state = "none"
        self.commit_synchronized = False
        self.metrics_persisted = False
        self._event("finish_without_checkpoint")

    def snapshot(self) -> dict[str, Any]:
        if self.phase not in PHASES:
            raise RuntimeError("V20 transaction phase is unknown")
        return {
            "version": VERSION,
            "completed_step": int(self.completed_step),
            "durable_step": int(self.durable_step),
            "durable_checkpoint": str(self.durable_checkpoint),
            "durable_origin_kind": str(self.durable_origin_kind),
            "optimizer_rng_resumable": bool(self.optimizer_rng_resumable),
            "current_step": self.current_step,
            "phase": self.phase,
            "text_step_state": self.text_step_state,
            "flow_step_state": self.flow_step_state,
            "commit_synchronized": bool(self.commit_synchronized),
            "metrics_persisted": bool(self.metrics_persisted),
            "events": list(self.events),
        }


def gather_transaction_states(
    transaction: LogicalUpdateTransaction,
    *,
    process_group: Any = None,
) -> list[dict[str, Any]]:
    local = transaction.snapshot()
    if not dist.is_available() or not dist.is_initialized():
        return [local]
    gathered = [None] * dist.get_world_size(group=process_group)
    dist.all_gather_object(gathered, local, group=process_group)
    return gathered


def synchronize_update_commit(
    transaction: LogicalUpdateTransaction,
    *,
    process_group: Any = None,
) -> list[dict[str, Any]]:
    states = gather_transaction_states(
        transaction,
        process_group=process_group,
    )
    validate_update_commit_states(states)
    transaction.mark_commit_synchronized()
    committed = gather_transaction_states(
        transaction,
        process_group=process_group,
    )
    if any(
        state["phase"] != "commit_synchronized"
        or state["commit_synchronized"] is not True
        for state in committed
    ):
        raise RuntimeError("V20 synchronized commit marker differs")
    return committed


def validate_update_commit_states(
    states: list[dict[str, Any]],
) -> None:
    identity = {
        (
            state["current_step"],
            state["completed_step"],
            state["phase"],
            state["text_step_state"],
            state["flow_step_state"],
        )
        for state in states
    }
    if (
        len(identity) != 1
        or any(
            state["phase"] != "local_update_complete"
            for state in states
        )
    ):
        raise RuntimeError(
            "V20 update commit state differs across ranks"
        )


def emergency_checkpoint_decision(
    rank_states: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rank_states:
        raise ValueError("V20 emergency decision has no rank states")
    durable = {
        (
            int(state["durable_step"]),
            str(state["durable_checkpoint"]),
        )
        for state in rank_states
    }
    if len(durable) != 1:
        return {
            "allow_checkpoint": False,
            "classification": "durable_lineage_mismatch",
            "resume_step": None,
            "resume_checkpoint": None,
        }
    durable_step, durable_checkpoint = next(iter(durable))
    origin_identity = {
        (
            str(state.get("durable_origin_kind", "resumable_optimizer_checkpoint")),
            bool(state.get("optimizer_rng_resumable", True)),
        )
        for state in rank_states
    }
    if len(origin_identity) != 1:
        return {
            "allow_checkpoint": False,
            "classification": "durable_lineage_mismatch",
            "resume_step": None,
            "resume_checkpoint": None,
        }
    durable_origin_kind, optimizer_rng_resumable = next(iter(origin_identity))
    if (
        durable_origin_kind
        == "classic_parent_fresh_optimizer_no_intermediate_resume"
        and optimizer_rng_resumable is False
    ):
        return {
            "allow_checkpoint": False,
            "classification": "fresh_origin_requires_full_restart",
            "checkpoint_step": None,
            "resume_step": 0,
            "resume_checkpoint": None,
            "durable_origin": durable_checkpoint,
            "durable_origin_kind": durable_origin_kind,
        }
    if all(
        state["commit_synchronized"] is True
        and state["phase"]
        in {
            "commit_synchronized",
            "metrics_persisted",
            "checkpoint_durable",
        }
        for state in rank_states
    ):
        completed = {int(state["completed_step"]) for state in rank_states}
        if len(completed) == 1:
            step = next(iter(completed))
            return {
                "allow_checkpoint": True,
                "classification": "complete_update_commit",
                "checkpoint_step": step,
                "resume_step": step,
                "resume_checkpoint": "new_emergency_checkpoint",
            }
    possible_steps = any(
        state["text_step_state"] in {"possible", "stepped"}
        or state["flow_step_state"] in {"possible", "stepped"}
        for state in rank_states
    )
    completed = {int(state["completed_step"]) for state in rank_states}
    if not possible_steps and len(completed) == 1:
        step = next(iter(completed))
        return {
            "allow_checkpoint": True,
            "classification": "clean_pre_step_boundary",
            "checkpoint_step": step,
            "resume_step": step,
            "resume_checkpoint": "new_emergency_checkpoint",
        }
    return {
        "allow_checkpoint": False,
        "classification": "partial_update_contamination",
        "checkpoint_step": None,
        "resume_step": durable_step,
        "resume_checkpoint": durable_checkpoint,
    }


__all__ = [
    "LogicalUpdateTransaction",
    "VERSION",
    "emergency_checkpoint_decision",
    "gather_transaction_states",
    "synchronize_update_commit",
    "validate_update_commit_states",
]
