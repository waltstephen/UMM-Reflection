"""Deterministic 4-of-49 transition layout for G021 attempt2."""
from __future__ import annotations

VERSION = "clean29529_g021_stratified_4_of_49_flow_v2"


def stratified_transition_indices(total_steps: int = 50, count: int = 4) -> tuple[int, ...]:
    transitions = int(total_steps) - 1
    if total_steps != 50 or count != 4:
        raise ValueError("G021 attempt2 freezes 4 transitions from 50 generation steps")
    values = tuple(min(transitions - 1, int((i + 0.5) * transitions / count)) for i in range(count))
    if len(set(values)) != 4 or values[0] >= 8 or not any(18 <= x <= 31 for x in values) or values[-1] < 42:
        raise RuntimeError("G021 early/middle/late transition coverage differs")
    return values


TRANSITION_INDICES = stratified_transition_indices()
TRANSITION_COUNT = len(TRANSITION_INDICES)
EXPECTED_LOSS_REDUCTION = "mean_over_selected_transitions_then_existing_call_scaling"
