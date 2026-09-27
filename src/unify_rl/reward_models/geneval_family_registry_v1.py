"""Frozen verifier registry for the GenEval-family rewards."""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from unify_rl.reward_models.deterministic_count_v1 import (
    MAX_OBJECTS,
    NMS_THRESHOLD,
    compute_iou,
    count_reward,
)


REGISTRY_CONTRACT_VERSION = "clean29529_geneval_family_registry_v1"
CONTRACT_VERSION = REGISTRY_CONTRACT_VERSION
REWARD_BACKEND = "geneval_family_v1"
MAX_ROUNDS = 4

DETECTOR_THRESHOLD = 0.3
COUNTING_THRESHOLD = 0.9
POSITION_THRESHOLD = 0.1
COLOR_VQA_MARGIN = 0.0

COLORS = (
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
)
POSITION_RELATIONS = ("left of", "right of", "above", "below")
FAMILY_ORDER = (
    "single_object",
    "two_object",
    "counting",
    "colors",
    "position",
    "color_attr",
    "ocr_text",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
# Verifier assets live under one directory (see README, "GenEval assets"):
#   mmdetection/          mmdetection 2.x checkout (Mask2Former configs)
#   mask2former/          mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
#   openclip_hub/         HF hub cache holding timm/vit_large_patch14_clip_224.openai
#   T2I-CompBench/        T2I-CompBench checkout (BLIPvqa_eval)
#   blip/                 model_base_vqa_capfilt_large.pth
# Every file is pinned by EXPECTED_ASSET_SHA256 below.
ASSETS_ROOT = Path(
    os.environ.get("GENEVAL_ASSETS_DIR", REPO_ROOT / "pretrained" / "geneval")
)
GENEVAL_ENV = Path(sys.prefix)
DETECTOR_CONFIG = (
    ASSETS_ROOT
    / "mmdetection/configs/mask2former/"
    "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
)
DETECTOR_CHECKPOINT = (
    ASSETS_ROOT
    / "mask2former/"
    "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
)
OBJECT_NAMES = (
    REPO_ROOT
    / "third_party/Bagel/eval/gen/geneval/evaluation/object_names.txt"
)
OPENCLIP_REVISION = "18d0535469bb561bf468d76c1d73aa35156c922b"
OPENCLIP_CHECKPOINT = (
    ASSETS_ROOT
    / "openclip_hub/"
    "models--timm--vit_large_patch14_clip_224.openai/"
    f"snapshots/{OPENCLIP_REVISION}/open_clip_model.safetensors"
)
BLIP_REPO = ASSETS_ROOT / "T2I-CompBench/BLIPvqa_eval"
BLIP_CHECKPOINT = ASSETS_ROOT / "blip/model_base_vqa_capfilt_large.pth"
BLIP_MED_CONFIG = BLIP_REPO / "configs/med_config.json"

EXPECTED_ASSET_SHA256 = {
    "detector_config": (
        "9074c4681870644756c0c5b813995bc06749844be8f24fb2eca3e494d5b9eadb"
    ),
    "detector_checkpoint": (
        "743b7d99015f1224c6d57fd4b14d04b15cc8ec72ae7ee7831e7c71d8873b7a54"
    ),
    "object_names": (
        "608f6a0e5c8ca1c7a92b818430141fe268c85b7930b50f4160f1d65893b5bacd"
    ),
    "openclip_checkpoint": (
        "9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e"
    ),
    "blip_checkpoint": (
        "7a7d546209f1ccfa8b3cd3a0138c53e0d1e95e4a4bc280bef8f67e20fe4925ae"
    ),
}


@dataclass(frozen=True)
class FamilyContract:
    family: str
    verifier_version: str
    verifier_kind: str
    context_schema: Mapping[str, str]
    required_backends: tuple[str, ...]
    decision_rule: str
    raw_score_rule: str
    reward_scale: str = "reward=2*raw_score-1, bounded to [-1,1]"
    terminal_only: bool = True
    registry_contract_version: str = REGISTRY_CONTRACT_VERSION
    reward_backend: str = REWARD_BACKEND

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FAMILY_CONTRACTS: dict[str, FamilyContract] = {
    "single_object": FamilyContract(
        family="single_object",
        verifier_version="clean29529_geneval_single_object_presence_v1",
        verifier_kind="detector_presence",
        context_schema={"object": "COCO detector class"},
        required_backends=("detector",),
        decision_rule=(
            "pass iff at least one target-class detection has score > 0.3"
        ),
        raw_score_rule="1.0 on pass, else 0.0",
    ),
    "two_object": FamilyContract(
        family="two_object",
        verifier_version="clean29529_geneval_two_object_presence_v1",
        verifier_kind="detector_presence_conjunction",
        context_schema={
            "objects": "two distinct COCO detector classes in prompt order"
        },
        required_backends=("detector",),
        decision_rule=(
            "pass iff both distinct classes have at least one score > 0.3 "
            "detection"
        ),
        raw_score_rule="1.0 only when both objects pass, else 0.0",
    ),
    "counting": FamilyContract(
        family="counting",
        verifier_version="clean29529_geneval_count_exact_v1",
        verifier_kind="detector_exact_count",
        context_schema={
            "object": "COCO detector class",
            "count": "integer in [2,6]",
        },
        required_backends=("detector",),
        decision_rule=(
            "pass iff score > 0.9 detection count equals the target count"
        ),
        raw_score_rule=(
            "deterministic_count_v1 count_reward(target, detected)"
        ),
    ),
    "colors": FamilyContract(
        family="colors",
        verifier_version="clean29529_geneval_clip_color_v1",
        verifier_kind="detector_then_geneval_clip_color",
        context_schema={
            "object": "COCO detector class",
            "color": "one frozen GenEval color label",
        },
        required_backends=("detector", "color"),
        decision_rule=(
            "pass iff the top score > 0.3 target box exists and the official "
            "GenEval ViT-L-14 OpenAI CLIP crop pipeline predicts the color"
        ),
        raw_score_rule="1.0 on conjunction pass, else 0.0",
    ),
    "position": FamilyContract(
        family="position",
        verifier_version="clean29529_geneval_box_position_v1",
        verifier_kind="detector_box_geometry",
        context_schema={
            "subject": "COCO detector class",
            "relation": "left of|right of|above|below",
            "object": "distinct COCO detector class",
        },
        required_backends=("detector",),
        decision_rule=(
            "pass iff the official GenEval relative_position rule with "
            "position_threshold=0.1 finds the requested relation between "
            "the top subject and object boxes"
        ),
        raw_score_rule="1.0 on geometry pass, else 0.0",
    ),
    "color_attr": FamilyContract(
        family="color_attr",
        verifier_version="clean29529_geneval_per_box_color_blip_v1",
        verifier_kind="per_box_clip_color_with_blip_vqa_crosscheck",
        context_schema={
            "attributes": (
                "two distinct {object: COCO class, color: GenEval color} rows"
            )
        },
        required_backends=("detector", "color", "vqa"),
        decision_rule=(
            "pass iff both top score > 0.3 boxes receive their assigned "
            "colors from the official GenEval CLIP crop pipeline and BLIP "
            "gives each expected CompBench-style color noun phrase a greater "
            "yes probability than the phrase with the two colors swapped"
        ),
        raw_score_rule="1.0 only when every assignment passes, else 0.0",
    ),
    "ocr_text": FamilyContract(
        family="ocr_text",
        verifier_version="clean29529_geneval_ocr_exact_numeric_boundary_v1",
        verifier_kind="ocr_exact_match",
        context_schema={"text": "non-empty short quoted text"},
        required_backends=("ocr",),
        decision_rule=(
            "pass iff NFKC-normalized, whitespace-collapsed, outer-quote-"
            "stripped OCR text exactly equals the target; case, punctuation, "
            "and complete numeric runs must match"
        ),
        raw_score_rule="1.0 on exact match, else 0.0",
    ),
}


@dataclass(frozen=True)
class Detection:
    class_name: str
    box: tuple[float, float, float, float]
    score: float
    mask: Any = field(default=None, repr=False, compare=False)

    def public_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "box": [float(value) for value in self.box],
            "score": float(self.score),
            "has_mask": self.mask is not None,
        }


@dataclass(frozen=True)
class ColorPrediction:
    label: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerifierResult:
    family: str
    verifier_version: str
    passed: bool
    raw_score: float
    reward: float
    diagnostics: Mapping[str, Any]
    registry_contract_version: str = REGISTRY_CONTRACT_VERSION
    reward_backend: str = REWARD_BACKEND

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DetectorBackend(Protocol):
    def detect(
        self,
        image_path: str,
        class_names: Sequence[str],
        *,
        threshold: float,
    ) -> dict[str, list[Detection]]: ...


class ColorBackend(Protocol):
    def classify(
        self,
        image_path: str,
        detections: Sequence[Detection],
        class_name: str,
    ) -> list[ColorPrediction]: ...


class VQABackend(Protocol):
    def yes_probability(
        self,
        image_path: str,
        detection: Detection,
        question: str,
    ) -> float: ...


class OCRBackend(Protocol):
    def read_text(self, image_path: str) -> str: ...


@dataclass
class VerifierBackends:
    detector: DetectorBackend | None = None
    color: ColorBackend | None = None
    vqa: VQABackend | None = None
    ocr: OCRBackend | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_result(
    family: str,
    *,
    passed: bool,
    raw_score: float,
    diagnostics: Mapping[str, Any],
) -> VerifierResult:
    contract = FAMILY_CONTRACTS[family]
    raw = max(0.0, min(1.0, float(raw_score)))
    reward = max(-1.0, min(1.0, 2.0 * raw - 1.0))
    return VerifierResult(
        family=family,
        verifier_version=contract.verifier_version,
        passed=bool(passed),
        raw_score=raw,
        reward=reward,
        diagnostics=dict(diagnostics),
    )


def _required_string(context: Mapping[str, Any], key: str) -> str:
    value = str(context.get(key) or "").strip()
    if not value:
        raise ValueError(f"missing verifier context field: {key}")
    return value


def _require_backend(value: Any, name: str) -> Any:
    if value is None:
        raise RuntimeError(f"{name} backend is required")
    return value


def _detected(
    backend: DetectorBackend,
    image_path: str,
    class_names: Sequence[str],
    *,
    threshold: float,
) -> dict[str, list[Detection]]:
    names = tuple(dict.fromkeys(str(value).strip() for value in class_names))
    if not names or any(not value for value in names):
        raise ValueError("detector class names must be non-empty")
    observed = backend.detect(
        image_path,
        names,
        threshold=float(threshold),
    )
    if set(observed) != set(names):
        raise RuntimeError("detector response does not cover requested classes")
    return {
        name: sorted(
            list(observed[name]),
            key=lambda value: float(value.score),
            reverse=True,
        )
        for name in names
    }


def relative_position(
    object_a: Detection,
    object_b: Detection,
    *,
    position_threshold: float = POSITION_THRESHOLD,
) -> set[str]:
    """Replicate the official GenEval box relation calculation."""

    if float(position_threshold) != POSITION_THRESHOLD:
        raise ValueError("position threshold is frozen at 0.1")
    ax1, ay1, ax2, ay2 = object_a.box
    bx1, by1, bx2, by2 = object_b.box
    center_a = ((ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0)
    center_b = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
    dim_a = (abs(ax2 - ax1), abs(ay2 - ay1))
    dim_b = (abs(bx2 - bx1), abs(by2 - by1))
    offset = (
        center_a[0] - center_b[0],
        center_a[1] - center_b[1],
    )

    revised = []
    for axis in range(2):
        magnitude = max(
            abs(offset[axis])
            - position_threshold * (dim_a[axis] + dim_b[axis]),
            0.0,
        )
        revised.append(math.copysign(magnitude, offset[axis]))
    if all(abs(value) < 1e-3 for value in revised):
        return set()
    norm = math.hypot(*offset)
    if norm <= 0.0:
        return set()
    dx, dy = (revised[0] / norm, revised[1] / norm)
    relations = set()
    if dx < -0.5:
        relations.add("left of")
    if dx > 0.5:
        relations.add("right of")
    if dy < -0.5:
        relations.add("above")
    if dy > 0.5:
        relations.add("below")
    return relations


_OUTER_QUOTES = {
    ('"', '"'),
    ("'", "'"),
    ("\u201c", "\u201d"),
    ("\u2018", "\u2019"),
}
_NUMERIC_RUN = re.compile(r"\d+")


def normalize_ocr_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value))
    normalized = " ".join(normalized.split())
    if len(normalized) >= 2 and (
        normalized[0],
        normalized[-1],
    ) in _OUTER_QUOTES:
        normalized = normalized[1:-1].strip()
    return normalized


def ocr_exact_match(expected: str, observed: str) -> tuple[bool, dict[str, Any]]:
    expected_normalized = normalize_ocr_text(expected)
    observed_normalized = normalize_ocr_text(observed)
    if not expected_normalized:
        raise ValueError("OCR target text must be non-empty")
    expected_numeric_runs = _NUMERIC_RUN.findall(expected_normalized)
    observed_numeric_runs = _NUMERIC_RUN.findall(observed_normalized)
    exact = observed_normalized == expected_normalized
    numeric_boundaries_ok = expected_numeric_runs == observed_numeric_runs
    return exact and numeric_boundaries_ok, {
        "expected_normalized": expected_normalized,
        "observed_normalized": observed_normalized,
        "expected_numeric_runs": expected_numeric_runs,
        "observed_numeric_runs": observed_numeric_runs,
        "numeric_boundaries_ok": numeric_boundaries_ok,
        "case_sensitive": True,
        "punctuation_sensitive": True,
    }


def _valid_done_turn(candidate: Mapping[str, Any]) -> int:
    result = -1
    for position, step in enumerate(candidate.get("trajectory") or []):
        if not isinstance(step, Mapping):
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
    if result >= MAX_ROUNDS:
        raise RuntimeError("DONE turn exceeds the frozen maximum round count")
    return result


def _policy_image_turn(
    candidate: Mapping[str, Any],
    record: Mapping[str, Any],
) -> int | None:
    explicit = record.get(
        "image_turn_index",
        record.get("final_image_turn_index"),
    )
    if explicit is not None:
        value = int(explicit)
        if value < 0 or value >= MAX_ROUNDS:
            raise RuntimeError("image turn index is outside the frozen range")
        return value
    result = None
    for position, step in enumerate(candidate.get("trajectory") or []):
        if not isinstance(step, Mapping):
            continue
        if step.get("policy_optimized", True) is False:
            continue
        if not (
            bool(step.get("image_generated", False))
            or bool(step.get("has_image", False))
        ):
            continue
        result = int(step.get("round", position))
    if result is None and bool(
        candidate.get("rollout_start_image_generated", False)
    ):
        result = 0
    if result is not None and (result < 0 or result >= MAX_ROUNDS):
        raise RuntimeError("derived image turn is outside the frozen range")
    return result


def _candidate_raw_score(record: Mapping[str, Any]) -> float:
    for key in (
        "terminal_family_reward",
        "raw_score",
        "absolute_score",
        "trajectory_reward",
    ):
        if record.get(key) is not None:
            value = float(record[key])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise RuntimeError(
                    f"candidate {key} must be finite and in [0,1]"
                )
            return value
    raise RuntimeError("candidate terminal family score is missing")


def _center_group_scores(
    scores: Mapping[int, float],
    *,
    all_candidate_ids: Sequence[int],
) -> tuple[dict[int, float], float, float]:
    result = {int(value): 0.0 for value in all_candidate_ids}
    if len(scores) < 2:
        return result, 0.0, 0.0
    values = [float(scores[candidate_id]) for candidate_id in sorted(scores)]
    mean = statistics.mean(values)
    std = statistics.pstdev(values)
    scale = max(std, 0.15)
    for candidate_id, value in scores.items():
        result[int(candidate_id)] = max(
            -1.0,
            min(1.0, (float(value) - mean) / scale),
        )
    return result, float(mean), float(std)


def _assert_expected_context(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    for key, expected_value in expected.items():
        if key not in observed or observed[key] != expected_value:
            raise RuntimeError(
                f"GenEval family reward context mismatch for {key}"
            )


def _normalized_turn_map(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise RuntimeError("trajectory turn advantage map must be a mapping")
    return {str(key): float(item) for key, item in value.items()}


def _validate_optional_equal(
    record: Mapping[str, Any],
    key: str,
    expected: Any,
) -> None:
    if key not in record:
        return
    observed = record[key]
    if key in ("text_turn_advantages", "image_turn_advantages"):
        observed = _normalized_turn_map(observed)
    if observed != expected:
        raise RuntimeError(f"GenEval candidate field mismatch for {key}")


def validate_group_response(
    raw: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    expected_context: dict[str, Any],
) -> dict[str, Any]:
    """Validate and canonicalize a service response for the trainer adapter."""

    if str(raw.get("contract_version") or "") != CONTRACT_VERSION:
        raise RuntimeError("GenEval family response contract mismatch")
    registry_version = str(
        raw.get("registry_contract_version") or CONTRACT_VERSION
    )
    if registry_version != REGISTRY_CONTRACT_VERSION:
        raise RuntimeError("GenEval registry contract mismatch")
    if str(raw.get("reward_backend") or "") != REWARD_BACKEND:
        raise RuntimeError("GenEval family reward backend mismatch")

    expected_by_id = {
        int(candidate["candidate_id"]): dict(candidate)
        for candidate in candidates
    }
    if len(expected_by_id) != len(candidates) or len(candidates) < 2:
        raise RuntimeError("GenEval request candidate ids are invalid")
    raw_records = raw.get("candidates")
    if not isinstance(raw_records, list):
        raise RuntimeError("GenEval response candidates are missing")
    record_by_id = {
        int(record["candidate_id"]): dict(record)
        for record in raw_records
        if isinstance(record, Mapping)
    }
    if (
        set(record_by_id) != set(expected_by_id)
        or len(record_by_id) != len(raw_records)
    ):
        raise RuntimeError(
            "GenEval response candidate ids do not match request"
        )
    active_ids = {
        int(value)
        for value in raw.get(
            "policy_active_candidate_ids",
            sorted(expected_by_id),
        )
    }
    if len(active_ids) < 2 or not active_ids.issubset(expected_by_id):
        raise RuntimeError("GenEval active candidate ids are invalid")

    observed_context = raw.get("reward_context")
    if not isinstance(observed_context, Mapping):
        raise RuntimeError("GenEval reward context is missing")
    _assert_expected_context(observed_context, expected_context)
    family = str(
        expected_context.get("family")
        or observed_context.get("family")
        or raw.get("family")
        or ""
    )
    if family not in FAMILY_CONTRACTS:
        raise RuntimeError("GenEval reward context family is invalid")
    verifier_version = FAMILY_CONTRACTS[family].verifier_version
    if raw.get("verifier_version") not in (None, verifier_version):
        raise RuntimeError("GenEval verifier version mismatch")

    active_scores = {
        candidate_id: _candidate_raw_score(record_by_id[candidate_id])
        for candidate_id in active_ids
    }
    advantages, group_mean, group_std = _center_group_scores(
        active_scores,
        all_candidate_ids=sorted(expected_by_id),
    )
    for key, expected_value in (
        ("group_reward_mean", group_mean),
        ("group_reward_std", group_std),
    ):
        if key in raw and abs(float(raw[key]) - expected_value) > 1e-9:
            raise RuntimeError(f"GenEval response {key} mismatch")

    adapted_records = []
    for candidate_id in sorted(expected_by_id):
        candidate = expected_by_id[candidate_id]
        source = record_by_id[candidate_id]
        raw_score = _candidate_raw_score(source)
        done_turn = _valid_done_turn(candidate)
        eligible = candidate_id in active_ids and done_turn >= 0
        advantage = advantages[candidate_id] if eligible else 0.0
        image_turn = (
            _policy_image_turn(candidate, source) if eligible else None
        )
        text_turn_advantages = (
            {str(done_turn): advantage} if eligible else {}
        )
        image_turn_advantages = (
            {str(image_turn): advantage}
            if image_turn is not None and eligible
            else {}
        )
        text_turn_modes = {
            turn: "geneval_family_done_final_v1"
            for turn in text_turn_advantages
        }
        image_turn_modes = {
            turn: "geneval_family_final_image_v1"
            for turn in image_turn_advantages
        }
        expected_fields = {
            "trajectory_advantage": advantage,
            "trajectory_text_advantage": advantage,
            "trajectory_image_advantage": (
                advantage if image_turn_advantages else 0.0
            ),
            "text_turn_advantages": text_turn_advantages,
            "image_turn_advantages": image_turn_advantages,
            "text_turn_modes": text_turn_modes,
            "image_turn_modes": image_turn_modes,
        }
        for key, expected_value in expected_fields.items():
            _validate_optional_equal(source, key, expected_value)
        adapted_records.append(
            {
                **source,
                "family": family,
                "verifier_version": verifier_version,
                "terminal_family_reward": raw_score,
                "trajectory_reward": raw_score,
                "trajectory_advantage": advantage,
                "done_turn_index": done_turn,
                "image_turn_index": image_turn,
                "semantic_eligible": eligible,
                "total_reward": advantage,
                "ungated_total_reward": raw_score,
                "reward_score": advantage,
                "absolute_score": raw_score,
                "relative_score": advantage,
                "task_score": advantage,
                "trajectory_score": raw_score,
                **expected_fields,
                "parser_failure": False,
                "judge_reasoning": "",
            }
        )

    active_records = [
        record
        for record in adapted_records
        if int(record["candidate_id"]) in active_ids
    ]
    informative = any(
        abs(float(record["trajectory_advantage"])) > 1e-12
        for record in active_records
    )
    advantage_values = [
        float(record["trajectory_advantage"])
        for record in active_records
    ]
    result = {
        **raw,
        "ok": True,
        "contract_version": CONTRACT_VERSION,
        "registry_contract_version": REGISTRY_CONTRACT_VERSION,
        "reward_backend": REWARD_BACKEND,
        "score_kind": "trajectory_terminal_geneval_family_grpo_v1",
        "family": family,
        "verifier_version": verifier_version,
        "skip_update": not informative,
        "skip_policy": (
            "all_trajectory_turns_tied" if not informative else ""
        ),
        "fallback_reason": (
            "all active terminal family rewards are tied or ineligible"
            if not informative
            else ""
        ),
        "group_reward_mean": group_mean,
        "group_reward_std": group_std,
        "mean": group_mean,
        "std": group_std,
        "best": max(active_scores.values()) if active_scores else 0.0,
        "worst": min(active_scores.values()) if active_scores else 0.0,
        "component_means": {
            "terminal_family_reward": group_mean,
            "trajectory_turn_advantage": (
                statistics.mean(advantage_values)
                if advantage_values
                else 0.0
            ),
            "valid_done_rate": (
                statistics.mean(
                    float(int(record["done_turn_index"]) >= 0)
                    for record in active_records
                )
                if active_records
                else 0.0
            ),
            "flow_image_credit_rate": (
                statistics.mean(
                    float(bool(record["image_turn_advantages"]))
                    for record in active_records
                )
                if active_records
                else 0.0
            ),
        },
        "component_stds": {
            "terminal_family_reward": group_std,
            "trajectory_turn_advantage": (
                statistics.pstdev(advantage_values)
                if len(advantage_values) > 1
                else 0.0
            ),
        },
        "parser_failure_rate": 0.0,
        "candidates": adapted_records,
        "policy_active_candidate_ids": sorted(active_ids),
        "judge_consensus": {
            "passed": True,
            "version": REWARD_BACKEND,
        },
        "reward_context": dict(observed_context),
        "attempts": int(raw.get("attempts", 1)),
        "rm_latency_sec": float(
            raw.get("rm_latency_sec", raw.get("latency_sec", 0.0))
        ),
    }
    return result


class GenevalFamilyRegistry:
    def __init__(self, backends: VerifierBackends) -> None:
        self.backends = backends

    @staticmethod
    def manifest() -> dict[str, Any]:
        return {
            "registry_contract_version": REGISTRY_CONTRACT_VERSION,
            "reward_backend": REWARD_BACKEND,
            "family_order": list(FAMILY_ORDER),
            "contracts": {
                family: FAMILY_CONTRACTS[family].to_dict()
                for family in FAMILY_ORDER
            },
            "frozen_constants": {
                "detector_threshold": DETECTOR_THRESHOLD,
                "counting_threshold": COUNTING_THRESHOLD,
                "max_objects": MAX_OBJECTS,
                "nms_threshold": NMS_THRESHOLD,
                "position_threshold": POSITION_THRESHOLD,
                "color_vqa_margin": COLOR_VQA_MARGIN,
                "colors": list(COLORS),
                "position_relations": list(POSITION_RELATIONS),
            },
        }

    def verify(
        self,
        family: str,
        image_path: str,
        context: Mapping[str, Any],
    ) -> VerifierResult:
        name = str(family)
        if name not in FAMILY_CONTRACTS:
            raise ValueError(f"unknown GenEval verifier family: {name!r}")
        path = str(image_path)
        if not path:
            raise ValueError("image path must be non-empty")
        dispatch = {
            "single_object": self._single_object,
            "two_object": self._two_object,
            "counting": self._counting,
            "colors": self._colors,
            "position": self._position,
            "color_attr": self._color_attr,
            "ocr_text": self._ocr_text,
        }
        return dispatch[name](path, context)

    def _single_object(
        self,
        image_path: str,
        context: Mapping[str, Any],
    ) -> VerifierResult:
        object_name = _required_string(context, "object")
        detector = _require_backend(self.backends.detector, "detector")
        objects = _detected(
            detector,
            image_path,
            [object_name],
            threshold=DETECTOR_THRESHOLD,
        )[object_name]
        passed = len(objects) >= 1
        return _bounded_result(
            "single_object",
            passed=passed,
            raw_score=float(passed),
            diagnostics={
                "object": object_name,
                "detected_count": len(objects),
                "detections": [value.public_dict() for value in objects],
                "threshold": DETECTOR_THRESHOLD,
            },
        )

    def _two_object(
        self,
        image_path: str,
        context: Mapping[str, Any],
    ) -> VerifierResult:
        objects = context.get("objects")
        if not isinstance(objects, Sequence) or isinstance(objects, str):
            raise ValueError("two_object context requires an objects sequence")
        names = tuple(str(value).strip() for value in objects)
        if len(names) != 2 or any(not value for value in names):
            raise ValueError("two_object requires exactly two object names")
        if len(set(names)) != 2:
            raise ValueError("two_object requires distinct object names")
        detector = _require_backend(self.backends.detector, "detector")
        detected = _detected(
            detector,
            image_path,
            names,
            threshold=DETECTOR_THRESHOLD,
        )
        counts = {name: len(detected[name]) for name in names}
        passed = all(counts[name] >= 1 for name in names)
        return _bounded_result(
            "two_object",
            passed=passed,
            raw_score=float(passed),
            diagnostics={
                "objects": list(names),
                "detected_counts": counts,
                "detections": {
                    name: [
                        value.public_dict() for value in detected[name]
                    ]
                    for name in names
                },
                "threshold": DETECTOR_THRESHOLD,
            },
        )

    def _counting(
        self,
        image_path: str,
        context: Mapping[str, Any],
    ) -> VerifierResult:
        object_name = _required_string(context, "object")
        target = int(context.get("count", 0) or 0)
        if target not in range(2, 7):
            raise ValueError("counting target must be in [2,6]")
        detector = _require_backend(self.backends.detector, "detector")
        objects = _detected(
            detector,
            image_path,
            [object_name],
            threshold=COUNTING_THRESHOLD,
        )[object_name]
        observed = len(objects)
        raw_score = count_reward(target, observed)
        return _bounded_result(
            "counting",
            passed=observed == target,
            raw_score=raw_score,
            diagnostics={
                "object": object_name,
                "target_count": target,
                "detected_count": observed,
                "count_error": observed - target,
                "absolute_count_error": abs(observed - target),
                "detections": [value.public_dict() for value in objects],
                "threshold": COUNTING_THRESHOLD,
            },
        )

    def _colors(
        self,
        image_path: str,
        context: Mapping[str, Any],
    ) -> VerifierResult:
        object_name = _required_string(context, "object")
        expected_color = _required_string(context, "color")
        if expected_color not in COLORS:
            raise ValueError(f"unsupported GenEval color: {expected_color!r}")
        detector = _require_backend(self.backends.detector, "detector")
        color = _require_backend(self.backends.color, "color")
        objects = _detected(
            detector,
            image_path,
            [object_name],
            threshold=DETECTOR_THRESHOLD,
        )[object_name]
        predictions = (
            color.classify(image_path, objects[:1], object_name)
            if objects
            else []
        )
        if len(predictions) not in (0, 1):
            raise RuntimeError("single-color verifier returned invalid coverage")
        observed_color = predictions[0].label if predictions else None
        passed = bool(objects and observed_color == expected_color)
        return _bounded_result(
            "colors",
            passed=passed,
            raw_score=float(passed),
            diagnostics={
                "object": object_name,
                "expected_color": expected_color,
                "observed_color": observed_color,
                "detection": (
                    objects[0].public_dict() if objects else None
                ),
                "color_prediction": (
                    predictions[0].to_dict() if predictions else None
                ),
                "threshold": DETECTOR_THRESHOLD,
            },
        )

    def _position(
        self,
        image_path: str,
        context: Mapping[str, Any],
    ) -> VerifierResult:
        subject = _required_string(context, "subject")
        target = _required_string(context, "object")
        relation = _required_string(context, "relation")
        if subject == target:
            raise ValueError("position requires distinct object classes")
        if relation not in POSITION_RELATIONS:
            raise ValueError(f"unsupported position relation: {relation!r}")
        detector = _require_backend(self.backends.detector, "detector")
        detected = _detected(
            detector,
            image_path,
            [subject, target],
            threshold=DETECTOR_THRESHOLD,
        )
        subject_box = detected[subject][0] if detected[subject] else None
        target_box = detected[target][0] if detected[target] else None
        observed_relations = (
            relative_position(subject_box, target_box)
            if subject_box is not None and target_box is not None
            else set()
        )
        passed = relation in observed_relations
        return _bounded_result(
            "position",
            passed=passed,
            raw_score=float(passed),
            diagnostics={
                "subject": subject,
                "object": target,
                "expected_relation": relation,
                "observed_relations": sorted(observed_relations),
                "subject_detection": (
                    subject_box.public_dict()
                    if subject_box is not None
                    else None
                ),
                "object_detection": (
                    target_box.public_dict()
                    if target_box is not None
                    else None
                ),
                "detector_threshold": DETECTOR_THRESHOLD,
                "position_threshold": POSITION_THRESHOLD,
            },
        )

    def _color_attr(
        self,
        image_path: str,
        context: Mapping[str, Any],
    ) -> VerifierResult:
        attributes = context.get("attributes")
        if (
            not isinstance(attributes, Sequence)
            or isinstance(attributes, (str, bytes))
            or len(attributes) != 2
        ):
            raise ValueError("color_attr requires exactly two attributes")
        parsed = []
        for value in attributes:
            if not isinstance(value, Mapping):
                raise ValueError("color_attr rows must be mappings")
            object_name = _required_string(value, "object")
            expected_color = _required_string(value, "color")
            if expected_color not in COLORS:
                raise ValueError(
                    f"unsupported GenEval color: {expected_color!r}"
                )
            parsed.append((object_name, expected_color))
        if len({value[0] for value in parsed}) != 2:
            raise ValueError("color_attr requires distinct object classes")
        if len({value[1] for value in parsed}) != 2:
            raise ValueError("color_attr requires distinct colors")

        detector = _require_backend(self.backends.detector, "detector")
        color = _require_backend(self.backends.color, "color")
        vqa = _require_backend(self.backends.vqa, "vqa")
        names = [value[0] for value in parsed]
        detected = _detected(
            detector,
            image_path,
            names,
            threshold=DETECTOR_THRESHOLD,
        )
        rows = []
        for row_index, (object_name, expected_color) in enumerate(parsed):
            counterfactual_color = parsed[1 - row_index][1]
            detection = (
                detected[object_name][0]
                if detected[object_name]
                else None
            )
            predictions = (
                color.classify(
                    image_path,
                    [detection],
                    object_name,
                )
                if detection is not None
                else []
            )
            if len(predictions) not in (0, 1):
                raise RuntimeError(
                    "color_attr CLIP verifier returned invalid coverage"
            )
            observed_color = predictions[0].label if predictions else None
            question = f"{expected_color} {object_name}?"
            counterfactual_question = (
                f"{counterfactual_color} {object_name}?"
            )
            expected_yes_probability = (
                float(
                    vqa.yes_probability(
                        image_path,
                        detection,
                        question,
                    )
                )
                if detection is not None
                else 0.0
            )
            counterfactual_yes_probability = (
                float(
                    vqa.yes_probability(
                        image_path,
                        detection,
                        counterfactual_question,
                    )
                )
                if detection is not None
                else 0.0
            )
            if not (
                0.0 <= expected_yes_probability <= 1.0
                and 0.0 <= counterfactual_yes_probability <= 1.0
            ):
                raise RuntimeError("BLIP yes probability is outside [0,1]")
            blip_margin = (
                expected_yes_probability
                - counterfactual_yes_probability
            )
            row_passed = bool(
                detection is not None
                and observed_color == expected_color
                and blip_margin > COLOR_VQA_MARGIN
            )
            rows.append(
                {
                    "object": object_name,
                    "expected_color": expected_color,
                    "observed_color": observed_color,
                    "detection": (
                        detection.public_dict()
                        if detection is not None
                        else None
                    ),
                    "color_prediction": (
                        predictions[0].to_dict()
                        if predictions
                        else None
                    ),
                    "blip_question": question,
                    "blip_yes_probability": expected_yes_probability,
                    "blip_counterfactual_color": counterfactual_color,
                    "blip_counterfactual_question": (
                        counterfactual_question
                    ),
                    "blip_counterfactual_yes_probability": (
                        counterfactual_yes_probability
                    ),
                    "blip_margin": blip_margin,
                    "passed": row_passed,
                }
            )
        passed = all(value["passed"] for value in rows)
        return _bounded_result(
            "color_attr",
            passed=passed,
            raw_score=float(passed),
            diagnostics={
                "attributes": rows,
                "detector_threshold": DETECTOR_THRESHOLD,
                "blip_margin_threshold": COLOR_VQA_MARGIN,
            },
        )

    def _ocr_text(
        self,
        image_path: str,
        context: Mapping[str, Any],
    ) -> VerifierResult:
        expected = _required_string(context, "text")
        ocr = _require_backend(self.backends.ocr, "ocr")
        observed = str(ocr.read_text(image_path))
        passed, diagnostics = ocr_exact_match(expected, observed)
        return _bounded_result(
            "ocr_text",
            passed=passed,
            raw_score=float(passed),
            diagnostics={
                **diagnostics,
                "observed_raw": observed,
            },
        )


class Mask2FormerGenevalDetector:
    """Offline Mask2Former adapter with official GenEval filtering."""

    def __init__(
        self,
        *,
        model_config: Path = DETECTOR_CONFIG,
        model_checkpoint: Path = DETECTOR_CHECKPOINT,
        object_names: Path = OBJECT_NAMES,
        device: str = "cuda:0",
    ) -> None:
        from mmdet.apis import init_detector

        self.model_config = Path(model_config).resolve()
        self.model_checkpoint = Path(model_checkpoint).resolve()
        self.object_names_path = Path(object_names).resolve()
        self.device = str(device)
        self.class_names = tuple(
            value.strip()
            for value in self.object_names_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if value.strip()
        )
        if len(self.class_names) != 80:
            raise RuntimeError("GenEval detector vocabulary must have 80 rows")
        self.class_index = {
            value: index for index, value in enumerate(self.class_names)
        }
        self.model = init_detector(
            str(self.model_config),
            str(self.model_checkpoint),
            device=self.device,
        )
        self._cache: dict[
            tuple[str, int, int, float],
            dict[str, list[Detection]],
        ] = {}
        self._lock = threading.Lock()

    def manifest(self) -> dict[str, Any]:
        return {
            "adapter_version": "clean29529_mask2former_geneval_adapter_v1",
            "model_config": str(self.model_config),
            "model_config_sha256": sha256_file(self.model_config),
            "model_checkpoint": str(self.model_checkpoint),
            "model_checkpoint_sha256": sha256_file(
                self.model_checkpoint
            ),
            "object_names": str(self.object_names_path),
            "object_names_sha256": sha256_file(self.object_names_path),
            "device": self.device,
            "thresholds": {
                "default": DETECTOR_THRESHOLD,
                "counting": COUNTING_THRESHOLD,
            },
            "max_objects": MAX_OBJECTS,
            "nms_threshold": NMS_THRESHOLD,
        }

    def _detect_all(
        self,
        image_path: str,
        *,
        threshold: float,
    ) -> dict[str, list[Detection]]:
        from mmdet.apis import inference_detector
        import numpy as np

        if threshold not in (DETECTOR_THRESHOLD, COUNTING_THRESHOLD):
            raise ValueError("unsupported frozen GenEval detector threshold")
        path = Path(image_path).resolve()
        stat = path.stat()
        key = (
            str(path),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            float(threshold),
        )
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        result = inference_detector(self.model, str(path))
        bbox = result[0] if isinstance(result, tuple) else result
        segm = (
            result[1]
            if isinstance(result, tuple) and len(result) > 1
            else None
        )
        detected: dict[str, list[Detection]] = {}
        for index, class_name in enumerate(self.class_names):
            values = bbox[index]
            ordering = np.argsort(values[:, 4])[::-1]
            ordering = ordering[
                values[ordering, 4] > threshold
            ][:MAX_OBJECTS].tolist()
            selected: list[int] = []
            while ordering:
                best = int(ordering.pop(0))
                selected.append(best)
                ordering = [
                    int(value)
                    for value in ordering
                    if (
                        NMS_THRESHOLD == 1.0
                        or compute_iou(
                            values[best, :4].tolist(),
                            values[int(value), :4].tolist(),
                        )
                        < NMS_THRESHOLD
                    )
                ]
            detected[class_name] = [
                Detection(
                    class_name=class_name,
                    box=tuple(
                        float(value)
                        for value in values[item, :4].tolist()
                    ),
                    score=float(values[item, 4]),
                    mask=(
                        None
                        if segm is None
                        else segm[index][item]
                    ),
                )
                for item in selected
            ]
        with self._lock:
            self._cache[key] = detected
        return detected

    def detect(
        self,
        image_path: str,
        class_names: Sequence[str],
        *,
        threshold: float,
    ) -> dict[str, list[Detection]]:
        names = tuple(dict.fromkeys(str(value) for value in class_names))
        unknown = [value for value in names if value not in self.class_index]
        if unknown:
            raise ValueError(f"unknown detector classes: {unknown}")
        detected = self._detect_all(image_path, threshold=float(threshold))
        return {name: list(detected[name]) for name in names}


class OpenClipGenevalColorBackend:
    """Official GenEval ViT-L-14 color classifier over detector crops."""

    model_name = "ViT-L-14"
    pretrained_name = "openai"
    templates = (
        "a photo of a {c} {classname}",
        "a photo of a {c}-colored {classname}",
        "a photo of a {c} object",
    )

    def __init__(
        self,
        *,
        checkpoint: Path = OPENCLIP_CHECKPOINT,
        device: str = "cuda:0",
    ) -> None:
        import open_clip
        import torch

        self.checkpoint = Path(checkpoint).absolute()
        self.device = str(device)
        self.classification_device = torch.device(self.device).type
        observed_sha256 = sha256_file(self.checkpoint)
        if (
            observed_sha256
            != EXPECTED_ASSET_SHA256["openclip_checkpoint"]
        ):
            raise RuntimeError("frozen OpenCLIP checkpoint hash mismatch")
        cache_dir = self.checkpoint.parents[3]
        self.model, _, self.transform = (
            open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained_name,
                device=self.device,
                cache_dir=str(cache_dir),
            )
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self._classifiers: dict[str, Any] = {}

    def manifest(self) -> dict[str, Any]:
        return {
            "adapter_version": "clean29529_geneval_openclip_color_adapter_v1",
            "model_name": self.model_name,
            "pretrained_name": self.pretrained_name,
            "revision": OPENCLIP_REVISION,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": sha256_file(self.checkpoint),
            "device": self.device,
            "classification_device": self.classification_device,
            "colors": list(COLORS),
            "templates": [
                value.format(c="{c}", classname="{classname}")
                for value in self.templates
            ],
            "background": "#999",
            "crop": True,
        }

    def _classifier(self, class_name: str) -> Any:
        from clip_benchmark.metrics import zeroshot_classification as zsc

        if class_name not in self._classifiers:
            templates = [
                value.format(c="{c}", classname=class_name)
                for value in self.templates
            ]
            self._classifiers[class_name] = zsc.zero_shot_classifier(
                self.model,
                self.tokenizer,
                list(COLORS),
                templates,
                self.classification_device,
            )
        return self._classifiers[class_name]

    def classify(
        self,
        image_path: str,
        detections: Sequence[Detection],
        class_name: str,
    ) -> list[ColorPrediction]:
        if not detections:
            return []
        from PIL import Image, ImageOps
        from clip_benchmark.metrics import zeroshot_classification as zsc
        import numpy as np
        import torch

        image = ImageOps.exif_transpose(
            Image.open(image_path)
        ).convert("RGB")
        blank = Image.new("RGB", image.size, color="#999")
        tensors = []
        for detection in detections:
            crop_source = image
            if detection.mask is not None:
                mask = np.asarray(detection.mask)
                if tuple(mask.shape) != tuple(image.size[::-1]):
                    raise RuntimeError("detector mask/image shape mismatch")
                mask_image = Image.fromarray(
                    (mask.astype("uint8") * 255),
                    mode="L",
                )
                crop_source = Image.composite(image, blank, mask_image)
            crop = crop_source.crop(detection.box)
            tensors.append(self.transform(crop))
        batch = torch.stack(tensors).to(self.device)
        dataset = torch.utils.data.TensorDataset(
            batch.cpu(),
            torch.zeros(len(tensors), dtype=torch.long),
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            num_workers=0,
        )
        zsc.tqdm = lambda values, *args, **kwargs: values
        with torch.no_grad():
            logits, _ = zsc.run_classification(
                self.model,
                self._classifier(class_name),
                loader,
                self.classification_device,
            )
            probabilities = logits.softmax(dim=1)
            indexes = logits.argmax(dim=1)
        return [
            ColorPrediction(
                label=COLORS[int(indexes[position].item())],
                confidence=float(
                    probabilities[position, indexes[position]].item()
                ),
            )
            for position in range(len(detections))
        ]


class BLIPVQACrossCheckBackend:
    """Offline CompBench BLIP yes/no probability on one detector crop."""

    def __init__(
        self,
        *,
        repo: Path = BLIP_REPO,
        checkpoint: Path = BLIP_CHECKPOINT,
        med_config: Path = BLIP_MED_CONFIG,
        device: str = "cuda:0",
    ) -> None:
        self.repo = Path(repo).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.med_config = Path(med_config).resolve()
        self.device = str(device)
        repo_string = str(self.repo)
        if repo_string not in sys.path:
            sys.path.insert(0, repo_string)

        import torch
        from models.blip_vqa import blip_vqa
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode

        self.torch = torch
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (480, 480),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.model = blip_vqa(
            pretrained=str(self.checkpoint),
            image_size=480,
            vit="base",
            vit_grad_ckpt=False,
            vit_ckpt_layer=0,
            med_config=str(self.med_config),
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def manifest(self) -> dict[str, Any]:
        return {
            "adapter_version": "clean29529_blip_vqa_color_crosscheck_v1",
            "repo": str(self.repo),
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": sha256_file(self.checkpoint),
            "med_config": str(self.med_config),
            "med_config_sha256": sha256_file(self.med_config),
            "device": self.device,
            "image_size": 480,
            "decision_rule": (
                "P(yes|expected_color object?) > "
                "P(yes|swapped_color object?)"
            ),
            "margin_threshold": COLOR_VQA_MARGIN,
            "question_template": "{color} {object}?",
            "inference": "vqa_prob",
        }

    def yes_probability(
        self,
        image_path: str,
        detection: Detection,
        question: str,
    ) -> float:
        from PIL import Image, ImageOps

        image = ImageOps.exif_transpose(
            Image.open(image_path)
        ).convert("RGB")
        crop = image.crop(detection.box)
        tensor = self.transform(crop).unsqueeze(0).to(self.device)
        normalized_question = " ".join(str(question).lower().split())
        with self.torch.no_grad():
            values = self.model(
                tensor,
                [normalized_question],
                train=False,
                inference="vqa_prob",
            )
        if len(values) != 1:
            raise RuntimeError("BLIP VQA returned invalid coverage")
        return float(values[0])


class TesseractOCRBackend:
    """Subprocess OCR adapter with a frozen single-line configuration."""

    def __init__(
        self,
        *,
        executable: str | Path = "tesseract",
        language: str = "eng",
        page_segmentation_mode: int = 7,
        timeout_sec: int = 60,
    ) -> None:
        executable_string = str(executable)
        resolved = (
            shutil.which(executable_string)
            if not Path(executable_string).is_absolute()
            else executable_string
        )
        if not resolved or not Path(resolved).is_file():
            raise FileNotFoundError(
                f"tesseract executable is unavailable: {executable_string}"
            )
        self.executable = Path(resolved).resolve()
        self.language = str(language)
        self.page_segmentation_mode = int(page_segmentation_mode)
        self.timeout_sec = int(timeout_sec)

    def manifest(self) -> dict[str, Any]:
        version = subprocess.run(
            [str(self.executable), "--version"],
            check=True,
            text=True,
            capture_output=True,
            timeout=self.timeout_sec,
        ).stdout.splitlines()[0]
        return {
            "adapter_version": "clean29529_tesseract_ocr_exact_v1",
            "executable": str(self.executable),
            "executable_sha256": sha256_file(self.executable),
            "version": version,
            "language": self.language,
            "page_segmentation_mode": self.page_segmentation_mode,
        }

    def read_text(self, image_path: str) -> str:
        result = subprocess.run(
            [
                str(self.executable),
                str(Path(image_path).resolve()),
                "stdout",
                "--psm",
                str(self.page_segmentation_mode),
                "-l",
                self.language,
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=self.timeout_sec,
            env={
                **os.environ,
                "OMP_THREAD_LIMIT": os.environ.get("OMP_THREAD_LIMIT", "1"),
            },
        )
        return result.stdout


def frozen_asset_manifest(*, hash_existing: bool = True) -> dict[str, Any]:
    paths = {
        "detector_config": DETECTOR_CONFIG,
        "detector_checkpoint": DETECTOR_CHECKPOINT,
        "object_names": OBJECT_NAMES,
        "openclip_checkpoint": OPENCLIP_CHECKPOINT,
        "blip_checkpoint": BLIP_CHECKPOINT,
        "blip_med_config": BLIP_MED_CONFIG,
        "geneval_environment": GENEVAL_ENV,
    }
    result = {}
    for name, path in paths.items():
        canonical = Path(path).absolute()
        resolved = canonical.resolve()
        is_file = canonical.is_file()
        is_dir = canonical.is_dir()
        observed_hash = (
            sha256_file(canonical)
            if hash_existing and is_file
            else None
        )
        expected_hash = EXPECTED_ASSET_SHA256.get(name)
        result[name] = {
            "path": str(canonical),
            "resolved_path": str(resolved),
            "exists": is_file or is_dir,
            "kind": "file" if is_file else "directory" if is_dir else "missing",
            "size_bytes": canonical.stat().st_size if is_file else None,
            "sha256": observed_hash,
            "expected_sha256": expected_hash,
            "hash_matches": (
                observed_hash == expected_hash
                if expected_hash is not None and observed_hash is not None
                else None
            ),
        }
    return result


def validate_registry() -> None:
    if tuple(FAMILY_CONTRACTS) != FAMILY_ORDER:
        raise RuntimeError("GenEval family registry order mismatch")
    versions = [
        FAMILY_CONTRACTS[family].verifier_version
        for family in FAMILY_ORDER
    ]
    if len(set(versions)) != len(versions):
        raise RuntimeError("GenEval verifier versions must be unique")
    for family in FAMILY_ORDER:
        contract = FAMILY_CONTRACTS[family]
        if contract.family != family:
            raise RuntimeError(f"contract family mismatch: {family}")
        if contract.registry_contract_version != REGISTRY_CONTRACT_VERSION:
            raise RuntimeError(f"registry contract mismatch: {family}")
        if contract.reward_backend != REWARD_BACKEND:
            raise RuntimeError(f"reward backend mismatch: {family}")


validate_registry()
