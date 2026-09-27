"""Deterministic detector-count reward for V18 trajectory RL."""

from __future__ import annotations

import hashlib
import statistics
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol


CONTRACT_VERSION = "clean29529_deterministic_count_contract_v1"
SERVICE_VERSION = "clean29529_deterministic_count_service_v1"
REWARD_BACKEND = "deterministic_count_v1"
SCORE_KIND = "trajectory_terminal_count_grpo_v1"
ADVANTAGE_SCALE_FLOOR = 0.15
ADVANTAGE_CLIP = 1.0
COUNTING_THRESHOLD = 0.9
MAX_OBJECTS = 16
NMS_THRESHOLD = 1.0
MAX_ROUNDS = 4


@dataclass(frozen=True)
class CountDetection:
    image_path: str
    target_class: str
    detected_count: int
    boxes: list[list[float]]
    scores: list[float]


class CountDetector(Protocol):
    def count_many(
        self,
        image_paths: Iterable[str],
        target_class: str,
    ) -> dict[str, CountDetection]: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_reward(target_count: int, detected_count: int) -> float:
    target = int(target_count)
    detected = int(detected_count)
    if target <= 0 or detected < 0:
        raise ValueError("count reward requires positive target and count >= 0")
    error = abs(detected - target)
    if error == 0:
        return 1.0
    return max(0.0, 1.0 - error / target) * 0.5


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    def area(box: list[float]) -> float:
        return max(box[2] - box[0] + 1.0, 0.0) * max(
            box[3] - box[1] + 1.0,
            0.0,
        )

    intersection = [
        max(box_a[0], box_b[0]),
        max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]),
        min(box_a[3], box_b[3]),
    ]
    intersection_area = area(intersection)
    union = area(box_a) + area(box_b) - intersection_area
    return intersection_area / union if union else 0.0


def _context_from_constraints(
    constraints: Iterable[dict[str, Any]],
) -> tuple[str, int] | None:
    values = [
        value
        for value in constraints
        if isinstance(value, dict)
        and str(value.get("kind") or "") == "count_exact"
    ]
    if not values:
        return None
    if len(values) != 1:
        raise ValueError("deterministic count requires one count_exact constraint")
    value = values[0]
    return str(value.get("class") or "").strip(), int(value.get("count", 0))


def reward_context(payload: dict[str, Any]) -> dict[str, Any]:
    context = payload.get("reward_context")
    context = dict(context) if isinstance(context, dict) else {}
    object_name = str(context.get("count_object") or "").strip()
    target_count = int(context.get("count_target", 0) or 0)
    constrained = _context_from_constraints(payload.get("constraints") or [])
    if constrained is not None:
        constrained_object, constrained_count = constrained
        if object_name and object_name != constrained_object:
            raise ValueError("count object disagrees with constraint")
        if target_count and target_count != constrained_count:
            raise ValueError("count target disagrees with constraint")
        object_name = constrained_object
        target_count = constrained_count
    if not object_name or target_count not in range(2, 7):
        raise ValueError("deterministic count context is incomplete")
    return {
        "count_object": object_name,
        "count_target": target_count,
        "count_combo_id": str(
            context.get("count_combo_id") or f"{object_name}::{target_count}"
        ),
        "pool_uid": str(context.get("pool_uid") or ""),
        "prompt_sha256": str(context.get("prompt_sha256") or ""),
    }


def _valid_done_turn(candidate: dict[str, Any]) -> int:
    result = -1
    for position, step in enumerate(candidate.get("trajectory") or []):
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "").lower()
        is_done = action == "done" or bool(step.get("done", False))
        if not is_done or step.get("policy_optimized", True) is False:
            continue
        if not bool(step.get("route_valid", False)):
            continue
        if str(step.get("canonical_error") or "").strip():
            continue
        result = int(step.get("round", position))
    return result


def _candidate_states(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    initial_generated = bool(
        candidate.get("rollout_start_image_generated", False)
    )
    source_path = str(candidate.get("source_image_path") or "")
    if initial_generated and source_path:
        states.append(
            {
                "round_index": 0,
                "controller_round_index": None,
                "image_path": source_path,
                "is_initial": True,
                "policy_image": False,
            }
        )
    for position, step in enumerate(candidate.get("trajectory") or []):
        if not isinstance(step, dict):
            continue
        image_path = str(step.get("image_path") or "")
        if not image_path:
            continue
        controller_round = int(step.get("round", position))
        states.append(
            {
                "round_index": (
                    controller_round + 1
                    if initial_generated
                    else controller_round
                ),
                "controller_round_index": controller_round,
                "image_path": image_path,
                "is_initial": False,
                "policy_image": bool(
                    step.get("policy_optimized", True)
                    and (
                        step.get("image_generated", False)
                        or step.get("has_image", False)
                    )
                ),
            }
        )
    generated_path = str(candidate.get("generated_image_path") or "")
    if generated_path and (
        not states or states[-1]["image_path"] != generated_path
    ):
        states.append(
            {
                "round_index": (
                    int(states[-1]["round_index"]) + 1 if states else 0
                ),
                "controller_round_index": None,
                "image_path": generated_path,
                "is_initial": not states,
                "policy_image": False,
            }
        )
    return states


def _centered_advantages(
    rewards: dict[int, float],
    *,
    all_candidate_ids: Iterable[int],
) -> tuple[dict[int, float], float, float]:
    ids = sorted(rewards)
    result = {int(value): 0.0 for value in all_candidate_ids}
    if len(ids) < 2:
        return result, 0.0, 0.0
    values = [float(rewards[candidate_id]) for candidate_id in ids]
    mean = statistics.mean(values)
    std = statistics.pstdev(values)
    scale = max(std, ADVANTAGE_SCALE_FLOOR)
    for candidate_id in ids:
        result[candidate_id] = max(
            -ADVANTAGE_CLIP,
            min(
                ADVANTAGE_CLIP,
                (float(rewards[candidate_id]) - mean) / scale,
            ),
        )
    return result, float(mean), float(std)


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return statistics.mean(items) if items else 0.0


def _pstdev(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return statistics.pstdev(items) if len(items) > 1 else 0.0


def score_group(
    payload: dict[str, Any],
    detector: CountDetector,
) -> dict[str, Any]:
    if str(payload.get("contract_version") or "") != CONTRACT_VERSION:
        raise ValueError("deterministic count contract mismatch")
    context = reward_context(payload)
    candidates = [
        dict(value)
        for value in payload.get("candidates") or []
        if isinstance(value, dict)
    ]
    candidate_by_id = {
        int(candidate["candidate_id"]): candidate for candidate in candidates
    }
    if len(candidate_by_id) != len(candidates) or len(candidates) < 2:
        raise ValueError("deterministic count candidate ids are invalid")
    active_ids = {
        int(value)
        for value in payload.get(
            "policy_active_candidate_ids",
            sorted(candidate_by_id),
        )
    }
    if len(active_ids) < 2 or not active_ids.issubset(candidate_by_id):
        raise ValueError("deterministic count active candidate ids are invalid")

    states_by_candidate = {
        candidate_id: _candidate_states(candidate)
        for candidate_id, candidate in candidate_by_id.items()
    }
    image_paths = sorted(
        {
            str(state["image_path"])
            for states in states_by_candidate.values()
            for state in states
        }
    )
    for image_path in image_paths:
        if not Path(image_path).is_file():
            raise FileNotFoundError(image_path)
    detections = detector.count_many(
        image_paths,
        context["count_object"],
    )
    if set(detections) != set(image_paths):
        raise RuntimeError("detector response does not cover every image")

    raw_rewards: dict[int, float] = {}
    done_turns: dict[int, int] = {}
    final_state_by_id: dict[int, dict[str, Any] | None] = {}
    diagnostics_by_id: dict[int, list[dict[str, Any]]] = {}
    for candidate_id, candidate in candidate_by_id.items():
        diagnostics = []
        states = states_by_candidate[candidate_id]
        for state in states:
            detection = detections[str(state["image_path"])]
            error = int(detection.detected_count) - int(
                context["count_target"]
            )
            diagnostics.append(
                {
                    **state,
                    "detected_count": int(detection.detected_count),
                    "count_error": error,
                    "abs_count_error": abs(error),
                    "exact_match": error == 0,
                    "boxes": detection.boxes,
                    "scores": detection.scores,
                }
            )
        final_state = diagnostics[-1] if diagnostics else None
        diagnostics_by_id[candidate_id] = diagnostics
        final_state_by_id[candidate_id] = final_state
        done_turns[candidate_id] = _valid_done_turn(candidate)
        raw_rewards[candidate_id] = (
            count_reward(
                context["count_target"],
                int(final_state["detected_count"]),
            )
            if final_state is not None
            else 0.0
        )

    centered_rewards = {
        candidate_id: raw_rewards[candidate_id]
        for candidate_id in active_ids
        if final_state_by_id[candidate_id] is not None
    }
    advantages, reward_mean, reward_std = _centered_advantages(
        centered_rewards,
        all_candidate_ids=candidate_by_id,
    )
    records = []
    for candidate_id in sorted(candidate_by_id):
        diagnostics = diagnostics_by_id[candidate_id]
        final_state = final_state_by_id[candidate_id]
        done_turn = done_turns[candidate_id]
        eligible = (
            candidate_id in active_ids
            and done_turn >= 0
            and final_state is not None
        )
        advantage = advantages[candidate_id] if eligible else 0.0
        text_turn_advantages = (
            {str(done_turn): advantage} if eligible else {}
        )
        final_controller_round = (
            final_state.get("controller_round_index")
            if final_state is not None
            else None
        )
        image_turn_advantages = (
            {str(int(final_controller_round)): advantage}
            if (
                eligible
                and final_controller_round is not None
                and bool(final_state.get("policy_image", False))
            )
            else {}
        )
        round0_error = (
            int(diagnostics[0]["abs_count_error"]) if diagnostics else None
        )
        final_error = (
            int(final_state["abs_count_error"])
            if final_state is not None
            else None
        )
        records.append(
            {
                "candidate_id": candidate_id,
                "trajectory_reward": raw_rewards[candidate_id],
                "trajectory_advantage": advantage,
                "terminal_count_reward": raw_rewards[candidate_id],
                "count_object": context["count_object"],
                "count_target": context["count_target"],
                "count_combo_id": context["count_combo_id"],
                "final_detected_count": (
                    int(final_state["detected_count"])
                    if final_state is not None
                    else None
                ),
                "final_count_error": (
                    int(final_state["count_error"])
                    if final_state is not None
                    else None
                ),
                "final_abs_count_error": final_error,
                "round0_abs_count_error": round0_error,
                "count_error_improvement": (
                    int(round0_error - final_error)
                    if round0_error is not None and final_error is not None
                    else 0
                ),
                "exact_count_match": bool(
                    final_state is not None
                    and final_state["exact_match"]
                ),
                "done_turn_index": done_turn,
                "semantic_eligible": eligible,
                "count_round_diagnostics": diagnostics,
                "total_reward": advantage,
                "ungated_total_reward": raw_rewards[candidate_id],
                "reward_score": advantage,
                "absolute_score": raw_rewards[candidate_id],
                "relative_score": advantage,
                "task_score": advantage,
                "trajectory_score": raw_rewards[candidate_id],
                "trajectory_text_advantage": advantage,
                "trajectory_image_advantage": (
                    advantage if image_turn_advantages else 0.0
                ),
                "text_turn_advantages": text_turn_advantages,
                "image_turn_advantages": image_turn_advantages,
                "text_turn_modes": {
                    turn: "deterministic_count_done_final_v1"
                    for turn in text_turn_advantages
                },
                "image_turn_modes": {
                    turn: "deterministic_count_final_image_v1"
                    for turn in image_turn_advantages
                },
                "parser_failure": False,
                "judge_reasoning": "",
            }
        )

    active_records = [
        record
        for record in records
        if int(record["candidate_id"]) in active_ids
    ]
    informative = any(
        abs(float(record["trajectory_advantage"])) > 1e-12
        for record in active_records
    )
    component_values = {
        "terminal_count_reward": [
            float(record["terminal_count_reward"])
            for record in active_records
        ],
        "exact_count_match": [
            float(record["exact_count_match"]) for record in active_records
        ],
        "round0_abs_count_error": [
            float(record["round0_abs_count_error"])
            for record in active_records
            if record["round0_abs_count_error"] is not None
        ],
        "final_abs_count_error": [
            float(record["final_abs_count_error"])
            for record in active_records
            if record["final_abs_count_error"] is not None
        ],
        "count_error_improvement": [
            float(record["count_error_improvement"])
            for record in active_records
        ],
        "valid_done_rate": [
            float(int(record["done_turn_index"]) >= 0)
            for record in active_records
        ],
    }
    return {
        "ok": True,
        "contract_version": CONTRACT_VERSION,
        "service_version": SERVICE_VERSION,
        "reward_backend": REWARD_BACKEND,
        "score_kind": SCORE_KIND,
        "skip_update": not informative,
        "skip_policy": (
            "all_trajectory_turns_tied"
            if not informative
            else ""
        ),
        "fallback_reason": (
            "all active terminal count rewards are tied or ineligible"
            if not informative
            else ""
        ),
        "group_reward_mean": reward_mean,
        "group_reward_std": reward_std,
        "mean": reward_mean,
        "std": reward_std,
        "best": max(centered_rewards.values()) if centered_rewards else 0.0,
        "worst": min(centered_rewards.values()) if centered_rewards else 0.0,
        "component_means": {
            key: _mean(values) for key, values in component_values.items()
        },
        "component_stds": {
            key: _pstdev(values) for key, values in component_values.items()
        },
        "parser_failure_rate": 0.0,
        "candidates": records,
        "policy_active_candidate_ids": sorted(active_ids),
        "judge_consensus": {
            "passed": True,
            "version": REWARD_BACKEND,
        },
        "reward_context": context,
        "attempts": 1,
        "rm_latency_sec": 0.0,
    }


def validate_group_response(
    raw: dict[str, Any],
    request_candidates: list[dict[str, Any]],
    *,
    expected_context: dict[str, Any],
) -> dict[str, Any]:
    if str(raw.get("contract_version") or "") != CONTRACT_VERSION:
        raise RuntimeError("deterministic count response contract mismatch")
    if str(raw.get("service_version") or "") != SERVICE_VERSION:
        raise RuntimeError("deterministic count service version mismatch")
    if str(raw.get("reward_backend") or "") != REWARD_BACKEND:
        raise RuntimeError("deterministic count reward backend mismatch")
    observed_context = dict(raw.get("reward_context") or {})
    for key in ("count_object", "count_target", "count_combo_id"):
        if observed_context.get(key) != expected_context.get(key):
            raise RuntimeError(
                f"deterministic count context mismatch for {key}"
            )
    expected_ids = {
        int(candidate["candidate_id"]) for candidate in request_candidates
    }
    records = [
        dict(value)
        for value in raw.get("candidates") or []
        if isinstance(value, dict)
    ]
    record_by_id = {
        int(record["candidate_id"]): record for record in records
    }
    if set(record_by_id) != expected_ids or len(record_by_id) != len(records):
        raise RuntimeError(
            "deterministic count response candidate ids do not match request"
        )
    active_ids = {
        int(value)
        for value in raw.get(
            "policy_active_candidate_ids",
            sorted(expected_ids),
        )
    }
    centered_rewards = {
        candidate_id: float(
            record_by_id[candidate_id]["terminal_count_reward"]
        )
        for candidate_id in active_ids
        if record_by_id[candidate_id].get("final_detected_count") is not None
    }
    expected_advantages, expected_mean, expected_std = _centered_advantages(
        centered_rewards,
        all_candidate_ids=expected_ids,
    )
    if abs(float(raw.get("group_reward_mean", 0.0)) - expected_mean) > 1e-9:
        raise RuntimeError("deterministic count reward mean mismatch")
    if abs(float(raw.get("group_reward_std", 0.0)) - expected_std) > 1e-9:
        raise RuntimeError("deterministic count reward std mismatch")
    for candidate_id, record in record_by_id.items():
        reward = float(record.get("terminal_count_reward", -1.0))
        detected = record.get("final_detected_count")
        if detected is not None:
            expected_reward = count_reward(
                int(expected_context["count_target"]),
                int(detected),
            )
            if abs(reward - expected_reward) > 1e-9:
                raise RuntimeError(
                    "deterministic count reward formula mismatch"
                )
        expected_advantage = (
            expected_advantages[candidate_id]
            if bool(record.get("semantic_eligible", False))
            else 0.0
        )
        if (
            abs(
                float(record.get("trajectory_advantage", 0.0))
                - expected_advantage
            )
            > 1e-9
        ):
            raise RuntimeError(
                "deterministic count GRPO normalization mismatch"
            )
        done_turn = int(record.get("done_turn_index", -1))
        text_turns = {
            str(key): float(value)
            for key, value in (
                record.get("text_turn_advantages") or {}
            ).items()
        }
        expected_text_turns = (
            {str(done_turn): expected_advantage}
            if bool(record.get("semantic_eligible", False))
            else {}
        )
        if text_turns != expected_text_turns:
            raise RuntimeError(
                "deterministic count DONE-turn credit mismatch"
            )
    return dict(raw)


class Mask2FormerCountDetector:
    """Offline-bound GenEval detector with the official counting settings."""

    def __init__(
        self,
        *,
        model_config: Path,
        model_checkpoint: Path,
        object_names: Path,
        device: str = "cuda:0",
        threshold: float = COUNTING_THRESHOLD,
        max_objects: int = MAX_OBJECTS,
        nms_threshold: float = NMS_THRESHOLD,
    ) -> None:
        from mmdet.apis import init_detector

        self.model_config = Path(model_config).resolve()
        self.model_checkpoint = Path(model_checkpoint).resolve()
        self.object_names_path = Path(object_names).resolve()
        self.device = str(device)
        self.threshold = float(threshold)
        self.max_objects = int(max_objects)
        self.nms_threshold = float(nms_threshold)
        if (
            self.threshold != COUNTING_THRESHOLD
            or self.max_objects != MAX_OBJECTS
            or self.nms_threshold != NMS_THRESHOLD
        ):
            raise ValueError(
                "detector settings must match the frozen GenEval contract"
            )
        self.classnames = [
            line.strip()
            for line in self.object_names_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        if len(self.classnames) != 80:
            raise RuntimeError("detector vocabulary must contain 80 classes")
        self.class_index = {
            value: index for index, value in enumerate(self.classnames)
        }
        self.model = init_detector(
            str(self.model_config),
            str(self.model_checkpoint),
            device=self.device,
        )
        self._cache: dict[
            tuple[str, int, int],
            dict[str, CountDetection],
        ] = {}
        self._lock = threading.Lock()

    def manifest(self) -> dict[str, Any]:
        return {
            "service_version": SERVICE_VERSION,
            "contract_version": CONTRACT_VERSION,
            "reward_backend": REWARD_BACKEND,
            "device": self.device,
            "model_config": str(self.model_config),
            "model_config_sha256": sha256_file(self.model_config),
            "model_checkpoint": str(self.model_checkpoint),
            "model_checkpoint_sha256": sha256_file(
                self.model_checkpoint
            ),
            "object_names": str(self.object_names_path),
            "object_names_sha256": sha256_file(self.object_names_path),
            "counting_threshold": self.threshold,
            "max_objects": self.max_objects,
            "max_overlap": self.nms_threshold,
        }

    def _detect_all(self, image_path: str) -> dict[str, CountDetection]:
        from mmdet.apis import inference_detector
        import numpy as np

        path = Path(image_path).resolve()
        stat = path.stat()
        key = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = inference_detector(self.model, str(path))
        bbox = result[0] if isinstance(result, tuple) else result
        detections: dict[str, CountDetection] = {}
        for index, classname in enumerate(self.classnames):
            values = bbox[index]
            ordering = np.argsort(values[:, 4])[::-1]
            ordering = ordering[
                values[ordering, 4] > self.threshold
            ][: self.max_objects].tolist()
            selected: list[int] = []
            while ordering:
                best = ordering.pop(0)
                selected.append(best)
                ordering = [
                    value
                    for value in ordering
                    if (
                        self.nms_threshold == 1.0
                        or compute_iou(
                            values[best, :4].tolist(),
                            values[value, :4].tolist(),
                        )
                        < self.nms_threshold
                    )
                ]
            detections[classname] = CountDetection(
                image_path=str(path),
                target_class=classname,
                detected_count=len(selected),
                boxes=[
                    [float(value) for value in values[item, :4]]
                    for item in selected
                ],
                scores=[
                    float(values[item, 4]) for item in selected
                ],
            )
        with self._lock:
            self._cache[key] = detections
        return detections

    def count_many(
        self,
        image_paths: Iterable[str],
        target_class: str,
    ) -> dict[str, CountDetection]:
        target = str(target_class)
        if target not in self.class_index:
            raise ValueError(f"unknown detector class: {target!r}")
        result = {}
        for image_path in dict.fromkeys(str(value) for value in image_paths):
            path = str(Path(image_path).resolve())
            result[image_path] = self._detect_all(path)[target]
        return result


def detection_to_json(value: CountDetection) -> dict[str, Any]:
    return asdict(value)
