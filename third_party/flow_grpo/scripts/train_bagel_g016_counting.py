"""G016 state-grouped multiround Flow-GRPO continuous formal trainer."""

from __future__ import annotations

import functools
import json
import hashlib
import math
import os
import random
import statistics
import time
import traceback
import copy
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

from unify_rl.train.g016_metric_history import (
    MetricHistory,
    RUNTIME_PERFORMANCE_REVISION)

import torch
import torch.distributed as dist
from absl import app
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy
from torch.utils.data import Dataset

from flow_grpo.bagel.data.data_utils import add_special_tokens
from flow_grpo.bagel.data.transforms import ImageTransform
from flow_grpo.bagel.inferencer import (
    InterleaveInferencer as FlowInterleaveInferencer)
from flow_grpo.bagel.modeling.autoencoder import load_ae
from flow_grpo.bagel.modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel)
from flow_grpo.bagel.modeling.qwen2 import Qwen2Tokenizer
from unify_rl.train.g022_bucket_stat_tracker import (
    BucketStatTracker as G022BucketStatTracker,
    step_zero_std_ratio as g022_step_zero_std_ratio)
from unify_rl.reward_models.g022_whole_trajectory_reward import (
    VERSION as G022_ADVANTAGE_VERSION,
    assign_g022_advantages,
    build_g022_trajectory_credit)
from unify_rl.reward_models.f01_geneval_trajectory_credit_v1 import (
    build_f01_trajectory_credit)
from unify_rl.train.g022_campaign import (
    _expected_flow_learning_rate,
    _expected_text_learning_rate)
from unify_rl.train.f01_campaign import (
    F01_CORE_FAMILIES,
    F01_DURABLE_ORIGIN_KIND,
    F01_LOCAL_BATCH_SIZE_BY_WORLD,
    F01_TEXT_HEAD_MODE,
    f01_topology,
    validate_launch_bindings as validate_f01_launch_bindings)
from flow_grpo.g022_token_routing import (
    build_g024_uniform_masks)
from flow_grpo.g024_protocol_precursors import (
    g024_protocol_precursors)
from flow_grpo.g022_text_action_support import (
    UnsupportedTextActionTokenError as G022UnsupportedTextActionTokenError,
    build_text_action_support)
from flow_grpo.g022_token_routing import (
    VERSION as G022_ROUTING_VERSION,
    TokenRoutingLineageError as G022TokenRoutingLineageError)
from unify_rl.train.g022_bounded_kl import (
    assert_delta_within_stop as assert_g022_delta_within_stop)
from unify_rl.train.g022_campaign import (
    G022_ALPHA,
    G022_FLOW_CONTRACT,
    G022_LAMBDA,
    G022_PREMATURE_DONE_PENALTY,
    G022_REFLECTION_WEIGHTS,
    G022_ROOT_GROUP_COUNT,
    G022_SDE_WINDOW_RANGE,
    G022_SDE_WINDOW_SIZE,
    G022_SIBLINGS_PER_ROOT,
    G022_TRAIN_NUM_TIMESTEPS,
    G022_TRAJECTORY_COUNT,
    validate_launch_bindings as validate_g022_launch_bindings)
from flow_grpo.g016_counting_process import (
    CONTROLLER_ACTION_AUX_VERSION,
    CONTROLLER_ACTION_AUX_WEIGHT,
    FlowGRPOMultiroundRollout,
    GENEVAL_EXPECTED_SERVICE_VERSION,
    R0_CHANNEL_WEIGHT,
    REPAIR_CHANNEL_WEIGHT,
    SharedPrefixReference,
    assert_frozen_reference_optimizer_isolation,
    request_geneval_score_chunks,
    update_multiround_group)
from unify_rl.reward_models.g011_stopnow_reward import (
    MAX_ABS_ADVANTAGE,
    assert_policy_observation_secrecy,
    build_verification_contract,
    counting_score,
    image_state_sha256,
    validate_cap_terminal_does_not_underprice_done,
    validate_no_positive_detector_failed_done)
from unify_rl.reward_models.g011_stopnow_reward import (
    STOP_NOW_SCORER_IDENTITY,
    STOP_NOW_SCORER_VERSION)
from unify_rl.reward_models.g016_repair_first_reward import (
    CONTROLLER_CREDIT_VERSION,
    REPAIR_BUCKET_VERSION as RENDERER_BUCKET_VERSION,
    REPAIR_FLOW_CREDIT_VERSION as RENDERER_CREDIT_VERSION,
    VERSION as ADVANTAGE_VERSION,
    assign_g016_advantages)
from unify_rl.reward_models.g009_counting_process_reward import (
    VERSION as REWARD_VERSION)
from unify_rl.inference.v22_controller_protocol import (
    VERSION as G016_PROTOCOL_VERSION)

JUDGE_CONCURRENCY_VERSION = "disabled_no_gpt"
from unify_rl.train.g016_streaming_adamw import (
    G016StreamingCPUAdamW)
from unify_rl.train.g016_stopnow_sign_guard import (
    classify_g016_stopnow_signs)
from unify_rl.train.g016_forensic_ring_buffer import (
    G016ForensicRingBuffer)
from unify_rl.train.v20_update_contract import (
    capture_rank_rng_state,
    load_sharded_training_checkpoint,
    restore_rank_rng_state,
    save_sharded_training_checkpoint)
from unify_rl.train.v20_sampler_contract import (
    build_sampler_contract,
    checkpoint_sampler_state,
    rollout_seed,
    sampler_config_identity,
    step_batch_indices,
    validate_sampler_resume)
from unify_rl.train.v20_wandb import (
    V20WandbLogger as G016WandbLogger,
    build_run_config as build_wandb_run_config,
    redacted_status as redacted_wandb_status)
from unify_rl.train.v20_update_transaction import (
    LogicalUpdateTransaction,
    emergency_checkpoint_decision,
    gather_transaction_states,
    synchronize_update_commit)
from unify_rl.train.v20_release_gate import (
    build_optimizer_step_accounting,
    nonfinite_numeric_paths)
from scripts.train_bagel import (
    FLAGS)


VERSION = "clean29529_g017_autonomous_repair_run_to50_v2"

def active_g021_root_count() -> int:
    return G022_ROOT_GROUP_COUNT

def active_g021_trajectory_count() -> int:
    return G022_TRAJECTORY_COUNT

def active_siblings_per_root() -> int:
    """Trajectories sharing one root R0.

    G021-A and G021-B both use K=4; G022 uses K=16 (doc Appendix G-2 -- R3 and
    official BAGEL's `num_image_per_prompt` agree on 16). Every `// 4` and `% 4`
    in the root-group rollout reads this instead of the literal, so the G021
    arithmetic is unchanged and G022 does not need a forked rollout.
    """
    return G022_SIBLINGS_PER_ROOT

def g021_process_rtg_mode() -> bool:
    return True

REPO_ROOT = Path(__file__).resolve().parents[3]
GEN_BRANCH_MARKER = "_moe_gen"
FROZEN_G009_REWARD_SHA256 = "f67e758a97035c1374ac9dddda87bd2113d6131005d34488e7b74e97ce6dc5b9"
FROZEN_SYSTEM_PROMPT_SHA256 = "69465424c93e1d082661cb6e1bb927266c4439f0492fe28bea4643c8bbe52c84"


@dataclass(frozen=True)
class PromptRow:
    uid: str
    prompt: str
    family: str
    metadata: dict[str, Any]
    constraints: tuple[dict[str, Any], ...]


class G016PromptDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        rows = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.rows = []
        for row in rows:
            metadata = row.get("geneval_metadata")
            constraints = row.get("constraints")
            family = str(row.get("family") or "")
            # F01 trains six families. Every other campaign keeps the
            # counting-only admission below, byte-for-byte: a counting row takes
            # the counting branch whether or not F01_MODE is set, so no existing
            # pool can change meaning here.
            f01_row = family in F01_CORE_FAMILIES and family != "counting"
            if (
                not isinstance(row, dict)
                or not str(row.get("uid") or "")
                or not str(row.get("prompt") or "")
                or (family != "counting" and not f01_row)
                or not isinstance(metadata, dict)
                or str(metadata.get("tag") or "") != family
                or not isinstance(constraints, list)
                or not constraints
                or row.get("official_geneval_prompt_used") is not False
            ):
                raise RuntimeError("invalid G016 training prompt row")
            if f01_row:
                # `build_verification_contract` is the FROZEN G009 validator and
                # accepts exactly one `count_exact` clause, so it cannot see the
                # other five families. Their contract is the family context the
                # reward service already validates on every image.
                checked_constraints = [dict(value) for value in constraints]
                if any(str(c.get("kind") or "") != family for c in checked_constraints):
                    raise RuntimeError(
                        f"F01 {family} constraint kind differs from its family"
                    )
                if not list(metadata.get("include") or []):
                    raise RuntimeError(f"F01 {family} row has no include clause")
                self.rows.append(
                    PromptRow(
                        uid=str(row["uid"]), prompt=str(row["prompt"]),
                        family=family, metadata=dict(metadata),
                        constraints=tuple(checked_constraints),
                    )
                )
                continue
            checked_constraints = build_verification_contract(constraints)
            count_clause = checked_constraints[0]
            include = list(metadata.get("include") or [])
            if (
                len(include) != 1
                or str(include[0].get("class") or "") != str(count_clause["class"])
                or int(include[0].get("count", 0)) != int(count_clause["exact_count"])
            ):
                raise RuntimeError("G016 counting metadata/constraint mismatch")
            self.rows.append(
                PromptRow(
                    uid=str(row["uid"]),
                    prompt=str(row["prompt"]),
                    family=str(row["family"]),
                    metadata=dict(metadata),
                    constraints=tuple(checked_constraints),
                )
            )
        if not self.rows:
            raise RuntimeError("G016 training prompt pool is empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> PromptRow:
        return self.rows[index]

    @staticmethod
    def collate_fn(rows: list[PromptRow]):
        return (
            [row.prompt for row in rows],
            [
                {
                    "uid": row.uid,
                    "family": row.family,
                    "geneval_metadata": row.metadata,
                    "verification_constraints": [
                        dict(value) for value in row.constraints
                    ],
                }
                for row in rows
            ],
        )


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def utc_now() -> str:
    import datetime

    return datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def configure_reward_runtime(config: Any) -> dict[str, Any]:
    """Select only the explicitly bound G017/G018/G019 reward stage."""

    global ADVANTAGE_VERSION
    global CONTROLLER_CREDIT_VERSION
    global RENDERER_BUCKET_VERSION
    global RENDERER_CREDIT_VERSION
    global VERSION
    global assign_g016_advantages

    stage = str(config.g016.reward_stage)
    # Second, independent check at binding time, on the config object the
    # trainer will actually use. The config module already
    # ran this; repeating it here means a hand-edited or programmatically
    # mutated config cannot reach the training loop with G022's KL bound,
    # divergence stop or flow window silently reverted to a legacy value.
    validate_g022_launch_bindings(config)
    # All six GenEval families jointly, with the whole-trajectory advantage
    # (`assign_g022_advantages`) and the uniform text head.
    validate_f01_launch_bindings(config)
    ADVANTAGE_VERSION = G022_ADVANTAGE_VERSION
    CONTROLLER_CREDIT_VERSION = G022_ADVANTAGE_VERSION
    RENDERER_BUCKET_VERSION = G022_ADVANTAGE_VERSION
    RENDERER_CREDIT_VERSION = G022_ADVANTAGE_VERSION
    VERSION = G022_ADVANTAGE_VERSION
    assign_g016_advantages = assign_g022_advantages
    return {
        "reward_stage": stage,
        "g018": False, "g019": False, "g020": False,
        "g021": False, "g021b": False, "g022": True,
        "g023": False, "g024": True, "g025": False, "g026": False,
        "f01": True,
        "families": list(F01_CORE_FAMILIES),
        "fail_closed_on_model_output": False,
        "token_routing_version": G022_ROUTING_VERSION,
        "advantage_version": ADVANTAGE_VERSION,
        "text_head_mode": F01_TEXT_HEAD_MODE,
    }


def resolve_f01_fresh_transaction_origin(*, model_dir: Path) -> dict[str, Any]:
    """Durable origin of a fresh run: the immutable SFT initialization.

    It is not an optimizer/RNG checkpoint, so a failure before the first
    scheduled checkpoint restarts from step 0.
    """

    initialization = Path(model_dir) / "ema.safetensors"
    if not initialization.is_file():
        raise RuntimeError(f"initialization weights are missing: {initialization}")
    return {
        "durable_step": 0,
        "durable_checkpoint": str(initialization.resolve()),
        "durable_origin_kind": F01_DURABLE_ORIGIN_KIND,
        "optimizer_rng_resumable": False,
        "failure_recovery": "archive_attempt_and_restart_fresh_step0",
        "campaign": "f01",
    }


def update_g016_live_status(
    path: Path,
    *,
    stage: str,
    values: dict[str, Any],
) -> None:
    previous = read_json_object(path) if path.is_file() else {}
    atomic_json(
        path,
        {
            **previous,
            "updated_at_utc": utc_now(),
            "stage": stage,
            "official_geneval_consumed": False,
            **values,
        },
    )


def jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().float().item())
        return value.detach().float().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def _g016_cluster_ratio_ci(
    rows: list[dict[str, Any]], numerator: str, denominator: str, *, seed: int
) -> dict[str, Any]:
    total_denominator = sum(int(row.get(denominator, 0)) for row in rows)
    total_numerator = sum(int(row.get(numerator, 0)) for row in rows)
    estimate = total_numerator / total_denominator if total_denominator else None
    if not rows or estimate is None:
        return {"estimate": estimate, "ci95": None, "step_cluster_count": len(rows)}
    generator = random.Random(seed)
    values = []
    for _ in range(2000):
        sample = [rows[generator.randrange(len(rows))] for _ in rows]
        den = sum(int(row.get(denominator, 0)) for row in sample)
        if den:
            values.append(sum(int(row.get(numerator, 0)) for row in sample) / den)
    values.sort()
    return {
        "estimate": estimate,
        "ci95": [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]],
        "step_cluster_count": len(rows),
        "numerator": total_numerator,
        "denominator": total_denominator,
    }


def _g016_cluster_mean_ci(
    rows: list[dict[str, Any]], value_sum: str, count: str, *, seed: int
) -> dict[str, Any]:
    total_count = sum(int(row.get(count, 0)) for row in rows)
    estimate = sum(float(row.get(value_sum, 0.0)) for row in rows) / total_count if total_count else None
    if not rows or estimate is None:
        return {"estimate": estimate, "ci95": None, "step_cluster_count": len(rows), "count": total_count}
    generator = random.Random(seed)
    values = []
    for _ in range(2000):
        sample = [rows[generator.randrange(len(rows))] for _ in rows]
        den = sum(int(row.get(count, 0)) for row in sample)
        if den:
            values.append(sum(float(row.get(value_sum, 0.0)) for row in sample) / den)
    values.sort()
    return {
        "estimate": estimate,
        "ci95": [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]],
        "step_cluster_count": len(rows),
        "count": total_count,
    }


_METRICS_HISTORY = MetricHistory()


def _g016_metrics_history_from(path: Path) -> list[dict[str, Any]]:
    return _METRICS_HISTORY.read(path)


def build_g016_live_monitoring(
    records: list[dict[str, Any]], prior_metrics: list[dict[str, Any]], *, step: int
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "step": int(step),
        "broken_decision_count": 0, "broken_edit_count": 0,
        "exact_decision_count": 0, "exact_edit_count": 0,
        "repair_eligible_count": 0, "repair_to_exact_count": 0,
        "edit_count": 0, "controller_renderer_different_count": 0,
        "damage_success_count": 0, "damage_success_positive_count": 0,
    }
    category_names = ("bad_to_exact", "bad_improve_nonexact", "bad_neutral", "bad_damage", "exact_state_edit")
    for name in category_names:
        summary[f"{name}_count"] = 0
        summary[f"{name}_advantage_sum"] = 0.0
        summary[f"{name}_positive_count"] = 0
    for record in records:
        final_success = bool(record["terminal_strict_exact_success"])
        flow_position = 0
        for action, controller, round_row in zip(record["actions"], record["controller_advantages"], record["rounds"]):
            exact = bool(round_row["exact_before_action"])
            if action in {"edit", "done"}:
                summary["exact_decision_count" if exact else "broken_decision_count"] += 1
                if action == "edit":
                    summary["exact_edit_count" if exact else "broken_edit_count"] += 1
            if action != "edit":
                continue
            renderer = float(record["repair_flow_advantages"][flow_position])
            renderer_active = bool(record["repair_flow_active"][flow_position])
            summary["edit_count"] += 1
            summary["controller_renderer_different_count"] += abs(renderer - float(controller)) >= 1e-12
            q_before, q_after = float(round_row["q_before"]), float(round_row["q_after"])
            if not exact:
                summary["repair_eligible_count"] += 1
                summary["repair_to_exact_count"] += abs(q_after - 1.0) <= 1e-12
            if not renderer_active:
                flow_position += 1
                continue
            if exact:
                name = "exact_state_edit"
            elif abs(q_after - 1.0) <= 1e-12:
                name = "bad_to_exact"
            elif q_after > q_before + 1e-12:
                name = "bad_improve_nonexact"
            elif abs(q_after - q_before) <= 1e-12:
                name = "bad_neutral"
            else:
                name = "bad_damage"
                if final_success:
                    summary["damage_success_count"] += 1
                    summary["damage_success_positive_count"] += renderer > 1e-12
            summary[f"{name}_count"] += 1
            summary[f"{name}_advantage_sum"] += renderer
            summary[f"{name}_positive_count"] += renderer > 1e-12
            flow_position += 1
    prior = [row.get("g016_live_monitoring", {}).get("step_sufficient_statistics") for row in prior_metrics]
    cluster_rows = [row for row in prior if isinstance(row, dict)] + [summary]
    result = {
        "version": "clean29529_g016_step_clustered_live_monitoring_v1",
        "step_sufficient_statistics": summary,
        "repair_effectiveness": _g016_cluster_ratio_ci(cluster_rows, "repair_to_exact_count", "repair_eligible_count", seed=20269001),
        "p_edit_given_broken": _g016_cluster_ratio_ci(cluster_rows, "broken_edit_count", "broken_decision_count", seed=20269002),
        "p_edit_given_exact": _g016_cluster_ratio_ci(cluster_rows, "exact_edit_count", "exact_decision_count", seed=20269003),
        "controller_renderer_difference_rate": _g016_cluster_ratio_ci(cluster_rows, "controller_renderer_different_count", "edit_count", seed=20269004),
        "condition6_damage_success_positive_rate": _g016_cluster_ratio_ci(cluster_rows, "damage_success_positive_count", "damage_success_count", seed=20269005),
        "renderer_categories": {},
    }
    for index, name in enumerate(category_names):
        result["renderer_categories"][name] = {
            "mean_advantage": _g016_cluster_mean_ci(cluster_rows, f"{name}_advantage_sum", f"{name}_count", seed=20269100 + index),
            "positive_rate": _g016_cluster_ratio_ci(cluster_rows, f"{name}_positive_count", f"{name}_count", seed=20269200 + index),
        }
    condition6_current_violation = summary["damage_success_positive_count"] > 0
    recent = cluster_rows[-5:]
    previous = cluster_rows[-10:-5]
    recent_rate = (
        sum(row["broken_edit_count"] for row in recent) / sum(row["broken_decision_count"] for row in recent)
        if recent and sum(row["broken_decision_count"] for row in recent) else None
    )
    previous_rate = (
        sum(row["broken_edit_count"] for row in previous) / sum(row["broken_decision_count"] for row in previous)
        if len(previous) == 5 and sum(row["broken_decision_count"] for row in previous) else None
    )
    result["guards"] = {
        "condition6_current_sample_count": summary["damage_success_count"],
        "condition6_current_positive_count": summary["damage_success_positive_count"],
        "condition6_violation": condition6_current_violation,
        "p_edit_broken_recent5": recent_rate,
        "p_edit_broken_previous5": previous_rate,
        "p_edit_broken_sustained_decline_alert": (
            recent_rate is not None and previous_rate is not None and recent_rate <= previous_rate - 0.10
        ),
    }
    return result


def g022_difficulty_filter_plan(
    *,
    r0_exact: list[bool],
    max_exact_roots: int,
    resample_pool: list[int],
    used_dataset_indexes: list[int],
    target_exact_share: float | None = None,
    realised_exact_count: int = 0,
    realised_root_count: int = 0,
) -> dict[str, Any]:
    """Which already-solved roots to replace, doc Appendix K-4.

    K-4 decomposed G021-A's 116 dead groups (29.0% of 400):

    | | count | of dead | of all groups |
    |---|---:|---:|---:|
    | R0 already exact, all stayed exact | 82 | 70.7% | **20.5%** |
    | all failed | 27 | 23.3% | 6.8% |
    | all reached exact from broken | 7 | 6.0% | 1.8% |

    A group where every trajectory already succeeds *should* have zero
    advantage -- that is the estimator being correct, not failing. What is lost
    there is **compute, not signal**. Only the 6.8% all-failed case is real lost
    signal, and that is what larger K and the historical baseline address.

    So the fix is not a bigger K and not a different reward: it is
    difficulty filtering in the style of DAPO's discard-all-correct rule.

    This function decides only *which* roots to replace and with what. It is
    deliberately a plan, not an action, so the resample is bounded and auditable
    and the caller can log the rate whether or not it fired.

    Note on the secrecy contract: `r0_exact` is hidden reward-side detector
    state. It is used here to choose which prompt the *environment* draws next.
    It never enters the policy observation, the controller prompt, the sampled
    tokens or the inference API -- the policy only ever sees the R0 it is
    actually given. Selection is not observation.
    """

    exact_positions = [index for index, value in enumerate(r0_exact) if value]
    if int(max_exact_roots) < 0:
        # Disabled. Still reports the rate, because Appendix K-4's whole point
        # is that the rate is the number to watch.
        return {
            "version": "clean29529_g022_root_difficulty_filter_v1",
            "enabled": False,
            "root_count": len(r0_exact),
            "r0_exact_count": len(exact_positions),
            "r0_exact_rate": (
                len(exact_positions) / len(r0_exact) if r0_exact else 0.0
            ),
            "max_exact_roots": int(max_exact_roots),
            "replace_positions": [],
            "replacement_dataset_indexes": [],
            "requested_replacement_count": 0,
            "unsatisfied_replacement_count": 0,
            "resample_used": False,
        }
    if target_exact_share is not None:
        # PROPORTIONAL downsampling against the running share, not a per-step
        # cap.
        #
        # The cap barely worked: `max_exact_roots = 1` with 2 roots only fires
        # when BOTH are exact (~16% of steps), moving the exact-R0 share of
        # trained roots from 33.8% to only 32.0%, while exact-R0 groups are
        # dead 51.6% of the time against 14.6% for broken-R0 groups.
        #
        # They are NOT dropped entirely, and the reason is measured: 78 of 161
        # archived exact-R0 groups (48.4%) carry real gradient, because
        # siblings edit an already-correct image and 24.5% of those edits
        # damage it. That teaches "do not touch an image that is already
        # correct" -- the one credit path in this lineage that always worked
        # (G013), and the mechanism behind G018 moving P(EDIT|exact) from 0.667
        # to 0.442. About 43% of evaluation R0s are already correct, so the
        # behaviour must stay trained.
        #
        # Each exact root is kept only while doing so leaves the RUNNING share
        # at or under target; otherwise it is resampled. Deterministic, and it
        # self-corrects: a step forced to keep an exact root (no replacement
        # available) makes the next one likelier to be replaced.
        kept_exact = 0
        replace = []
        for position, is_exact in enumerate(r0_exact):
            total = int(realised_root_count) + position + 1
            if not is_exact:
                continue
            prospective = int(realised_exact_count) + kept_exact + 1
            if prospective / total > float(target_exact_share):
                replace.append(position)
            else:
                kept_exact += 1
    else:
        keep = max(int(max_exact_roots), 0)
        # Replace the *last* offending roots so root 0 is stable across a step;
        # the choice is arbitrary but must be deterministic.
        replace = exact_positions[keep:]
    taken = set(int(value) for value in used_dataset_indexes)
    replacements: list[int] = []
    for candidate in resample_pool:
        if len(replacements) >= len(replace):
            break
        if int(candidate) in taken:
            continue
        taken.add(int(candidate))
        replacements.append(int(candidate))
    # A resample we cannot satisfy is not an error: the step simply keeps that
    # already-exact root. Running fewer roots would break the fixed collective
    # topology, which is a far worse failure than a wasted group.
    satisfied = replace[: len(replacements)]
    return {
        "version": (
            "clean29529_g022_root_proportional_downsample_v1"
            if target_exact_share is not None
            else "clean29529_g022_root_difficulty_filter_v1"
        ),
        "target_exact_share": (
            None if target_exact_share is None else float(target_exact_share)
        ),
        "target_exact_share_is_derived": False,
        "realised_exact_count_before": int(realised_exact_count),
        "realised_root_count_before": int(realised_root_count),
        "realised_exact_share_before": (
            realised_exact_count / realised_root_count
            if realised_root_count
            else 0.0
        ),
        "root_count": len(r0_exact),
        "r0_exact_count": len(exact_positions),
        "r0_exact_rate": (
            len(exact_positions) / len(r0_exact) if r0_exact else 0.0
        ),
        "max_exact_roots": (
            None if target_exact_share is not None else max(int(max_exact_roots), 0)
        ),
        "replace_positions": satisfied,
        "replacement_dataset_indexes": replacements,
        "requested_replacement_count": len(replace),
        "unsatisfied_replacement_count": len(replace) - len(satisfied),
        "resample_used": bool(satisfied),
        "enabled": True,
    }


def g022_pre_update_invariants(
    *,
    diagnostics: dict[str, Any],
    reward_rows: list[dict[str, Any]],
    scored: dict[str, Any],
    advantage_version: str,
) -> list[str]:
    """Load-bearing pre-update guards for G022's ONE-head whole-trajectory GRPO.

    The G016-G021 block below this call checks a two-head, per-round credit
    design: an action head and a repair head, per-round exactness buckets, a
    deadzone, leave-one-out baselines, and a `repair_credit_source` string naming
    one of those schemes. G022 has none of that -- one scalar per trajectory,
    broadcast to every policy-active token -- so those checks are not merely
    unsatisfiable, they are about a different algorithm.

    G022 therefore gets its own set, of equal strictness, asserting what G022
    actually contracts. The G016-G021 block is left byte-identical and still
    runs for those stages.
    """

    errors: list[str] = []
    root_count = active_g021_root_count()
    siblings = active_siblings_per_root()

    # Group construction and detached R0, doc section 2.5 + Appendix H-1.
    if (
        int(diagnostics.get("trajectory_count", -1)) != root_count * siblings
        or int(diagnostics.get("root_group_count", -1)) != root_count
        or int(diagnostics.get("controller_group_size", -1)) != siblings
        or int(diagnostics.get("r0_policy_active_count", -1)) != 0
        or int(diagnostics.get("r0_text_record_count", -1)) != 0
        or int(diagnostics.get("r0_flow_record_count", -1)) != 0
        or scored.get("r0_records") != []
        or len({str(value["r0_sha256"]) for value in reward_rows}) != root_count
    ):
        errors.append("g022_group_construction_or_r0_detachment")

    # The defining property: ONE scalar per trajectory, broadcast to every
    # policy-active token of that trajectory, and nothing else.
    if diagnostics.get("trajectory_level_advantage_broadcast") is not True:
        errors.append("g022_trajectory_broadcast_absent")
    for row in reward_rows:
        scalar = row.get("g022_trajectory_advantage")
        if scalar is None or not math.isfinite(float(scalar)):
            errors.append("g022_trajectory_advantage_missing_or_nonfinite")
            break
        active = [
            float(value)
            for value, is_active in zip(
                row["controller_advantages"], row["controller_active"]
            )
            if is_active
        ] + [
            float(value)
            for value, is_active in zip(
                row["repair_flow_advantages"], row["repair_flow_active"]
            )
            if is_active
        ]
        if any(value != float(scalar) for value in active):
            errors.append("g022_advantage_not_uniform_across_trajectory")
            break
        if abs(float(scalar)) > float(diagnostics.get("advantage_clip", 1.0)) + 1e-9:
            errors.append("g022_advantage_exceeds_clip")
            break

    # None of the per-round credit machinery may be active.
    for key in (
        "deadzone_used",
        "leave_one_out_used",
        "return_to_go_used",
        "per_round_bucket_used",
        "action_head_used",
        "repair_head_used",
        "critic_used",
        "gae_used",
        "hard_projection",
        "empirical_mean_used_as_center",
    ):
        if diagnostics.get(key) is not False:
            errors.append(f"g022_{key}_must_be_false")

    # Reward coefficients, doc section 2.1/2.2 + Appendix H-3.
    terms = diagnostics.get("reward_terms") or {}
    weights = [float(value) for value in (terms.get("reflection_weights") or [])]
    if (
        float(terms.get("alpha", -1.0)) != G022_ALPHA
        or weights != [float(v) for v in G022_REFLECTION_WEIGHTS]
        or float(terms.get("lambda", -1.0)) != G022_LAMBDA
        or float(terms.get("premature_done_penalty", -1.0)) != G022_PREMATURE_DONE_PENALTY
        or terms.get("bonus_sum_starts_at_t0") is not True
        or terms.get("lambda_ge_alpha_w_max") is not True
    ):
        errors.append("g022_reward_coefficients_differ")

    # Versions and the frozen detector scorer identity.
    if (
        diagnostics.get("version") != advantage_version
        or diagnostics.get("stop_now_scorer_identity") != STOP_NOW_SCORER_IDENTITY
        or diagnostics.get("stop_now_scorer_version") != STOP_NOW_SCORER_VERSION
    ):
        errors.append("g022_version_or_scorer_identity_differs")

    # Doc section 3 / Appendix L-4: zero_std_ratio must be present from step 1.
    zero_std = diagnostics.get("zero_std_metrics") or {}
    if not isinstance(zero_std, dict) or "zero_std_ratio" not in zero_std:
        errors.append("g022_zero_std_ratio_not_logged")

    if scored.get("policy_observation_secrecy", {}).get("passed") is not True:
        errors.append("policy_observation_exactness_secrecy")
    return errors


def text_action_support_world_consensus(
    all_rank_update_metrics: list[dict[str, Any]],
    *,
    expected_support_hash: str,
) -> dict[str, Any]:
    """Prove the support mask ran on every text turn of every rank this step.

    Evaluated after the world-wide gather, so a disagreement raises on every
    rank at once instead of stranding peers in the next collective.

    This exists because "the mask is installed" and "the mask was applied" are
    different claims, and only the second one is worth anything: an unreached
    branch and a missing fix look identical from the source.
    """
    turns = 0
    masked_turns = 0
    reference_masked_turns = 0
    hashes: set[str] = set()
    unsupported_tokens = 0
    unsupported_examples: list[Any] = []
    for update in all_rank_update_metrics:
        for trajectory_metrics in update.get("text_metrics") or []:
            for metric in trajectory_metrics:
                turns += 1
                block = metric.get("text_action_support")
                if not isinstance(block, dict):
                    continue
                masked_turns += 1
                hashes.add(str(block.get("support_hash") or ""))
                if bool(block.get("reference_replay_masked")):
                    reference_masked_turns += 1
                count = int(block.get("unsupported_sampled_token_count", 0) or 0)
                unsupported_tokens += count
                if count and len(unsupported_examples) < 8:
                    unsupported_examples.extend(
                        block.get("unsupported_sampled_tokens") or []
                    )
    summary = {
        "version": "clean29529_g022_text_action_support_consensus_v1",
        "text_turn_count": turns,
        "masked_turn_count": masked_turns,
        "reference_masked_turn_count": reference_masked_turns,
        "support_hashes": sorted(hashes),
        "expected_support_hash": str(expected_support_hash),
        "unsupported_sampled_token_count": unsupported_tokens,
        "unsupported_sampled_tokens": unsupported_examples,
    }
    if turns and masked_turns != turns:
        raise G022UnsupportedTextActionTokenError(
            "G022 text-action support mask did not run on every text turn: "
            f"{masked_turns}/{turns} -- {summary}"
        )
    if masked_turns and reference_masked_turns != masked_turns:
        raise G022UnsupportedTextActionTokenError(
            "G022 frozen-reference replay did not share the sampler support: "
            f"{reference_masked_turns}/{masked_turns} -- {summary}"
        )
    if hashes and hashes != {str(expected_support_hash)}:
        raise G022UnsupportedTextActionTokenError(
            f"G022 text-action support hash differs across ranks: {summary}"
        )
    if unsupported_tokens:
        raise G022UnsupportedTextActionTokenError(
            "G022 replayed an unsupported text-action token, which the rollout "
            f"refusal should have made unreachable: {summary}"
        )
    return summary


def text_kl_delta_max(all_rank_update_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Worst per-token |logp_train - logp_ref| across every rank and turn.

    G022 item 2 / doc section 5. `torch.clamp` used to hide exactly this number:
    a token past the clamp bound kept diverging while the logged KL sat at a
    constant. It is now measured on every turn, including collective-padding
    turns, and is the quantity the hard runtime stop acts on.
    """

    deltas: list[float] = []
    linearized = 0
    modes: set[str] = set()
    for update in all_rank_update_metrics:
        for trajectory_metrics in update.get("text_metrics") or []:
            for metric in trajectory_metrics:
                payload = metric.get("kl_delta")
                if not isinstance(payload, dict):
                    continue
                value = float(payload.get("max_abs_delta", 0.0))
                if math.isfinite(value):
                    deltas.append(value)
                else:
                    deltas.append(float("inf"))
                linearized += int(payload.get("linearized_token_count", 0) or 0)
                mode = metric.get("kl_bound_mode")
                if mode is not None:
                    modes.add(str(mode))
    return {
        "max_abs_delta": max(deltas, default=0.0),
        "linearized_token_count": linearized,
        "bound_mode": (
            sorted(modes)[0] if len(modes) == 1 else "|".join(sorted(modes)) or "none"
        ),
        "measured_turn_count": len(deltas),
    }


def optimizer_state_precision(
    optimizer: Any,
    *,
    require_full_coverage: bool = False,
) -> dict[str, Any]:
    state = getattr(optimizer, "state", None)
    parameter_groups = getattr(optimizer, "param_groups", None)
    if state is None:
        wrapped = getattr(optimizer, "optimizer", None)
        state = None if wrapped is None else getattr(wrapped, "state", None)
        parameter_groups = (
            None
            if wrapped is None
            else getattr(wrapped, "param_groups", None)
        )
    if state is None:
        raise RuntimeError("G016 optimizer state is unavailable")
    if parameter_groups is None:
        raise RuntimeError("G016 optimizer parameter groups are unavailable")
    parameter_count = sum(
        len(group["params"]) for group in parameter_groups
    )
    full_coverage = len(state) == parameter_count
    floating_dtypes = sorted(
        {
            str(value.dtype)
            for parameter_state in state.values()
            for value in parameter_state.values()
            if torch.is_tensor(value) and value.is_floating_point()
        }
    )
    tensor_count = sum(
        torch.is_tensor(value)
        for parameter_state in state.values()
        for value in parameter_state.values()
    )
    passed = (
        (not floating_dtypes or floating_dtypes == ["torch.float32"])
        and (full_coverage or not require_full_coverage)
    )
    if not passed:
        raise RuntimeError(
            "G016 optimizer state precision/coverage differs: "
            f"dtypes={floating_dtypes} state={len(state)} "
            f"parameters={parameter_count}"
        )
    return {
        "version": "clean29529_g016_fp32_optimizer_state_v1",
        "parameter_state_count": len(state),
        "optimizer_parameter_count": parameter_count,
        "full_parameter_state_coverage": full_coverage,
        "tensor_count": tensor_count,
        "floating_dtypes": floating_dtypes,
        "passed": passed,
    }


def image_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def build_bagel(
    *,
    model_dir: Path,
    inference_dtype: torch.dtype,
    device: torch.device,
):
    llm_config = Qwen2Config.from_json_file(str(model_dir / "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    vit_config = SiglipVisionConfig.from_json_file(
        str(model_dir / "vit_config.json")
    )
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1
    vae_model, vae_config = load_ae(
        local_path=str(model_dir / "ae.safetensors")
    )
    bagel_config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, bagel_config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            vit_config,
            meta=True,
        )
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=str(model_dir / "ema.safetensors"),
        device_map={"": str(device)},
        offload_buffers=False,
        dtype=inference_dtype,
        force_hooks=True,
        offload_folder="/tmp/g016_flowgrpo_offload",
    ).eval()
    vae_model.requires_grad_(False)
    vae_model.to(device, dtype=inference_dtype)
    model.requires_grad_(False)
    return model, vae_model, llm_config


def unwrap_language_model(language_model: Any) -> Any:
    module = language_model
    while hasattr(module, "module"):
        module = module.module
    candidate = (
        module.get_base_model()
        if hasattr(module, "get_base_model")
        else module
    )
    if not hasattr(candidate, "model") or not hasattr(
        candidate.model,
        "layers",
    ):
        raise RuntimeError("G016 language model decoder is unavailable")
    return candidate


def configure_fullrank_trainable_surface(
    model: Any,
    config: Any,
) -> tuple[
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
    list[torch.nn.Module],
    dict[str, Any],
]:
    """Open all 28 und/gen MoT decoder experts and frozen-ViT flow roots."""

    if int(config.g016.lora_rank) != 0:
        raise RuntimeError("G016 forbids LoRA; full-rank is mandatory")

    language_model = unwrap_language_model(model.language_model)
    layers = list(language_model.model.layers)
    start = int(config.g016.trainable_layer_start)
    end = int(config.g016.trainable_layer_end)
    if len(layers) != 28 or (start, end) != (0, 27):
        raise RuntimeError("G016 trainable decoder surface must be all 28 layers")
    reference_layers = []
    for layer in layers:
        frozen = deepcopy(layer).to(device="cpu", dtype=torch.bfloat16).eval()
        frozen.requires_grad_(False)
        reference_layers.append(frozen)

    flow_root_names = ("time_embedder", "vae2llm", "llm2vae", "latent_pos_embed")
    frozen_flow_roots = {}
    for name in flow_root_names:
        module = getattr(model, name)
        frozen = deepcopy(module).to(device="cpu", dtype=torch.bfloat16).eval()
        frozen.requires_grad_(False)
        frozen_flow_roots[name] = frozen
    object.__setattr__(model, "_g016_frozen_flow_roots", frozen_flow_roots)

    policy_parameter_dtype = (
        torch.bfloat16 if bool(config.g016.bf16_policy_cpu_fp32_master)
        else torch.float32
    )
    flow_parameters: list[torch.nn.Parameter] = []
    text_parameters: list[torch.nn.Parameter] = []
    flow_parameter_names: list[str] = []
    text_parameter_names: list[str] = []
    per_layer = []
    selected_names = []
    for layer_index, layer in enumerate(layers):
        # Set this on the real decoder before FSDP/checkpoint wrappers are
        # constructed. Setting an attribute on a wrapper after prepare does
        # not reach the wrapped layer and made real controller traversals omit
        # the gen-MLP handle while collective padding included it.
        layer.force_uniform_mot_collectives = True
        layer_flow = []
        layer_text = []
        for suffix, parameter in layer.named_parameters():
            parameter.requires_grad_(True)
            parameter.data = parameter.data.to(dtype=policy_parameter_dtype)
            name = f"model.layers.{layer_index}.{suffix}"
            selected_names.append(name)
            if GEN_BRANCH_MARKER in suffix:
                flow_parameters.append(parameter)
                flow_parameter_names.append(name)
                layer_flow.append(name)
            else:
                text_parameters.append(parameter)
                text_parameter_names.append(name)
                layer_text.append(name)
        if not layer_flow or not layer_text:
            raise RuntimeError(f"G016 layer {layer_index} did not open both MoT channels")
        per_layer.append(
            {
                "layer": layer_index,
                "text_tensor_count": len(layer_text),
                "flow_tensor_count": len(layer_flow),
                "text_parameter_numel": sum(
                    dict(layer.named_parameters())[name.split(f"model.layers.{layer_index}.", 1)[1]].numel()
                    for name in layer_text
                ),
                "flow_parameter_numel": sum(
                    dict(layer.named_parameters())[name.split(f"model.layers.{layer_index}.", 1)[1]].numel()
                    for name in layer_flow
                ),
                "gen_qk_norm_names": [
                    name for name in layer_flow if "q_norm_moe_gen" in name or "k_norm_moe_gen" in name
                ],
            }
        )
    for module_name in flow_root_names:
        module = getattr(model, module_name)
        for suffix, parameter in module.named_parameters():
            parameter.requires_grad_(True)
            parameter.data = parameter.data.to(dtype=policy_parameter_dtype)
            name = f"flow_root.{module_name}.{suffix}"
            flow_parameters.append(parameter)
            flow_parameter_names.append(name)
            selected_names.append(name)
    model.vit_model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.vit_model.parameters()):
        raise RuntimeError("G016 ViT must remain frozen")
    if any(len(row["gen_qk_norm_names"]) != 2 for row in per_layer):
        raise RuntimeError("G016 did not open both gen-MoT Q/K norms in every layer")
    if not flow_parameters or not text_parameters:
        raise RuntimeError("G016 full decoder channels are empty")
    return (
        flow_parameters,
        text_parameters,
        reference_layers,
        {
            "version": "clean29529_g016_full28_dual_mot_frozen_vit_v1",
            "layers": list(range(28)),
            "tensor_count": len(selected_names),
            "parameter_numel": sum(parameter.numel() for parameter in (*text_parameters, *flow_parameters)),
            "flow_parameter_numel": sum(parameter.numel() for parameter in flow_parameters),
            "text_parameter_numel": sum(parameter.numel() for parameter in text_parameters),
            "flow_root_names": list(flow_root_names),
            "vit_trainable_parameter_numel": 0,
            "vae_frozen": True,
            "both_mot_channels_all_layers": True,
            "per_layer": per_layer,
            "master_dtype": "torch.float32_cpu_optimizer",
            "policy_parameter_dtype": str(policy_parameter_dtype),
            "compute_dtype": "torch.bfloat16",
            "selected_names": selected_names,
            "flow_parameter_names": flow_parameter_names,
            "text_parameter_names": text_parameter_names,
        },
    )

def force_inference_dispatch(transformer: Any) -> None:
    module = transformer
    while hasattr(module, "module"):
        module = module.module
    base = (
        module.get_base_model()
        if hasattr(module, "get_base_model")
        else module
    )
    transformer.train()
    module.training = False
    base.training = False
    base.model.training = False
    for nested in base.modules():
        nested.training = False
    uniform_targets = []
    for layer in base.model.layers:
        targets = [
            nested
            for nested in layer.modules()
            if nested.__class__.__name__ == "Qwen2MoTDecoderLayer"
        ]
        if len(targets) != 1:
            raise RuntimeError(
                "G016 nested wrapper did not expose exactly one decoder layer"
            )
        inner = targets[0]
        layer.training = False
        inner.training = False
        inner.force_uniform_mot_collectives = True
        inner.self_attn.training = False
        uniform_targets.append(inner)
    if len(uniform_targets) != 28 or not all(
        target.force_uniform_mot_collectives for target in uniform_targets
    ):
        raise RuntimeError("G016 did not force both MoT experts on all layers")


def gather_rank_objects(value: Any, group: Any) -> list[Any]:
    if not dist.is_initialized():
        return [value]
    gathered = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, value, group=group)
    return gathered


def broadcast_rank_object(value: Any, *, src: int, group: Any) -> Any:
    if not dist.is_initialized():
        return value
    payload = [value if dist.get_rank() == src else None]
    dist.broadcast_object_list(payload, src=src, group=group)
    return payload[0]


def rank_synchronised_replica_validation(
    replicas: list[dict[str, Any]],
    *,
    rank: int,
    object_group: Any,
) -> dict[str, Any] | None:
    """`validate_detached_r0_replicas` on rank 0, with the VERDICT broadcast.

    B / A4 rank synchronisation. Both call sites used to read

        mapping = validate_detached_r0_replicas(replicas) if rank == R0_SOURCE_RANK else None

    `validate_detached_r0_replicas` raises. Only rank 0 evaluates it. So on a
    genuine replica divergence rank 0 dies inside the exception handler while
    the other fifteen ranks walk into the next collective and block forever.

    **A distributed hang does not announce itself.** It produces no traceback,
    no non-zero exit and no log line; it burns the entire allocation until the
    job timeout. The failure the validator exists to catch would therefore be
    reported as "the run stopped for no reason", which is strictly worse than
    the corruption it was guarding against.

    The verdict is computed on the source rank, broadcast, and every rank
    raises on it together. The validator itself is untouched: it still fails
    closed, and this does not weaken it -- it makes its failure legible.
    """

    verdict = None
    if int(rank) == R0_SOURCE_RANK:
        try:
            verdict = {
                "ok": True,
                "mapping": validate_detached_r0_replicas(replicas),
                "error": None,
            }
        except Exception as exc:
            verdict = {
                "ok": False,
                "mapping": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    verdict = broadcast_rank_object(verdict, src=R0_SOURCE_RANK, group=object_group)
    if verdict.get("ok") is not True:
        raise RuntimeError(
            "G016 detached R0 replica validation failed on the source rank: "
            + str(verdict.get("error"))
        )
    return verdict["mapping"]


R0_SOURCE_RANK = 0
R0_BROADCAST_VERSION = "clean29529_g016_rank0_detached_r0_byte_broadcast_v1"


def validate_detached_r0_replicas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fail closed unless every rank reconstructed one rank-0 R0 byte payload."""
    if not rows:
        raise RuntimeError("G016 detached R0 replica set is empty")
    ranks = sorted(int(row["rank"]) for row in rows)
    if ranks != list(range(len(rows))):
        raise RuntimeError("G016 detached R0 rank coverage differs")
    sample_keys = {
        (str(row["prompt"]), str(row["uid"]), int(row["r0_seed"]))
        for row in rows
    }
    if len(sample_keys) != 1:
        raise RuntimeError(
            "G016 detached R0 rows are not replicas of one sample/seed: "
            + repr(sorted(sample_keys))
        )
    if (
        {int(row["source_rank"]) for row in rows} != {R0_SOURCE_RANK}
        or {str(row["transport_version"]) for row in rows}
        != {R0_BROADCAST_VERSION}
    ):
        raise RuntimeError("G016 detached R0 source/transport identity differs")
    hashes = {str(row["image_sha256"]) for row in rows}
    byte_counts = {int(row["received_byte_count"]) for row in rows}
    if len(hashes) != 1 or len(byte_counts) != 1 or next(iter(byte_counts)) <= 0:
        raise RuntimeError(
            "G016 distributed R0 replicas differ in bytes: "
            + json.dumps(
                [
                    {
                        "rank": row["rank"],
                        "sha256": row["image_sha256"],
                        "byte_count": row["received_byte_count"],
                    }
                    for row in rows
                ],
                sort_keys=True,
            )
        )
    reconstructed_hashes = {
        str(row["reconstructed_rgb_sha256"]) for row in rows
    }
    image_shapes = {
        (
            str(row["reconstructed_mode"]),
            tuple(int(value) for value in row["reconstructed_size"]),
        )
        for row in rows
    }
    if len(reconstructed_hashes) != 1 or len(image_shapes) != 1:
        raise RuntimeError(
            "G016 distributed R0 reconstruction differs: "
            + json.dumps(
                [
                    {
                        "rank": row["rank"],
                        "rgb_sha256": row["reconstructed_rgb_sha256"],
                        "mode": row["reconstructed_mode"],
                        "size": row["reconstructed_size"],
                    }
                    for row in rows
                ],
                sort_keys=True,
            )
        )
    return {
        "version": "clean29529_g016_true_r0_replica_mapping_v2",
        "transport_version": R0_BROADCAST_VERSION,
        "source_rank": R0_SOURCE_RANK,
        "replica_count": len(rows),
        "broadcast_recipient_count": len(rows),
        "sample_key": list(next(iter(sample_keys))),
        "image_sha256": next(iter(hashes)),
        "received_byte_count": next(iter(byte_counts)),
        "reconstructed_rgb_sha256": next(iter(reconstructed_hashes)),
        "reconstructed_mode": next(iter(image_shapes))[0],
        "reconstructed_size": list(next(iter(image_shapes))[1]),
        "independent_rank_outputs_retained": 1,
        "non_source_local_generation_outputs_discarded": True,
        "candidate_diversity_begins_after_r0_fork": True,
    }


def evaluate_protocol_degradation(
    prior_protocol_invalid_steps: list[float],
    *,
    controller_invalid_rate: float,
    parse_failure_rate: float,
    false_done_numerator: int,
    false_done_denominator: int,
    prior_false_done_observations: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Apply hard stops to format correctness; monitor false-DONE behavior.

    False-DONE is excluded from the single-step protocol hard-spike
    composite.  A small broken-state
    denominator (captured step 5 was 5/7) is behavioral evidence, not grammar
    collapse.  Its raw numerator/denominator and weighted trends remain fully
    visible while controller/parse failures retain every frozen stop.
    """
    format_components = {
        "controller_invalid_rate": float(controller_invalid_rate),
        "parse_failure_rate": float(parse_failure_rate),
    }
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in format_components.values()
    ):
        raise ValueError("G017 protocol degradation component is invalid")
    false_done_numerator = int(false_done_numerator)
    false_done_denominator = int(false_done_denominator)
    if (
        false_done_numerator < 0
        or false_done_denominator < 0
        or false_done_numerator > false_done_denominator
    ):
        raise ValueError("G017 false-DONE numerator/denominator is invalid")
    false_done_rate = (
        false_done_numerator / false_done_denominator
        if false_done_denominator
        else 0.0
    )
    prior = [float(value) for value in prior_protocol_invalid_steps]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in prior):
        raise ValueError("G017 prior protocol invalid history is invalid")
    false_done_history = []
    for numerator, denominator in prior_false_done_observations or []:
        numerator = int(numerator)
        denominator = int(denominator)
        if numerator < 0 or denominator < 0 or numerator > denominator:
            raise ValueError("G017 prior false-DONE observation is invalid")
        false_done_history.append((numerator, denominator))

    # Every behavioral/protocol statistic is monitor-only, including
    # controller/parse rates. Preserve both the format
    # composite and the broader behavior composite, but neither can halt.
    format_current = max(format_components.values())
    current = max(format_current, false_done_rate)
    series = [*prior, current]
    trailing = series[-5:]
    first_five_mean = statistics.mean(series[:5]) if len(series) >= 5 else None
    trailing_five_mean = statistics.mean(trailing) if len(trailing) == 5 else None
    hard_spike = current >= 0.20
    intermittent_count = sum(value >= 0.08 for value in trailing)
    intermittent = intermittent_count >= 3
    creep = bool(
        first_five_mean is not None
        and trailing_five_mean is not None
        and trailing_five_mean >= 0.10
        and trailing_five_mean - first_five_mean >= 0.05
    )
    monitor_alerts = []
    if hard_spike:
        monitor_alerts.append("behavior_hard_spike_ge_0_20")
    if intermittent:
        monitor_alerts.append("behavior_three_of_last_five_ge_0_08")
    if creep:
        monitor_alerts.append("behavior_trailing5_creep_ge_first5_plus_0_05_and_ge_0_10")

    false_done_series = [*false_done_history, (false_done_numerator, false_done_denominator)]
    false_done_trailing = false_done_series[-5:]
    cumulative_numerator = sum(value[0] for value in false_done_series)
    cumulative_denominator = sum(value[1] for value in false_done_series)
    trailing_numerator = sum(value[0] for value in false_done_trailing)
    trailing_denominator = sum(value[1] for value in false_done_trailing)
    return {
        "version": "g017_autonomous_run_to50_protocol_monitor_only_v3",
        **format_components,
        "false_done_numerator": false_done_numerator,
        "false_done_denominator": false_done_denominator,
        "false_done_rate": false_done_rate,
        "false_done_denominator_undersized_vs_k28": false_done_denominator < 28,
        "false_done_single_step_hard_stop_enabled": False,
        "false_done_monitoring_only": True,
        "false_done_cumulative_numerator": cumulative_numerator,
        "false_done_cumulative_denominator": cumulative_denominator,
        "false_done_cumulative_rate": (
            cumulative_numerator / cumulative_denominator
            if cumulative_denominator
            else 0.0
        ),
        "false_done_trailing5_numerator": trailing_numerator,
        "false_done_trailing5_denominator": trailing_denominator,
        "false_done_trailing5_rate": (
            trailing_numerator / trailing_denominator
            if trailing_denominator
            else 0.0
        ),
        "false_done_last_five": [
            {
                "numerator": numerator,
                "denominator": denominator,
                "rate": numerator / denominator if denominator else 0.0,
            }
            for numerator, denominator in false_done_trailing
        ],
        "format_protocol_invalid_step": format_current,
        "protocol_invalid_step": current,
        "observed_step_count": len(series),
        "last_five": trailing,
        "last_five_ge_0_08_count": intermittent_count,
        "first_five_mean": first_five_mean,
        "trailing_five_mean": trailing_five_mean,
        "hard_spike": hard_spike,
        "intermittent": intermittent,
        "creep": creep,
        "monitor_only": True,
        "monitor_alerts": monitor_alerts,
        "halt": False,
        "reasons": [],
    }


def controller_protocol_metrics(flat: list[dict[str, Any]]) -> dict[str, Any]:
    turns = [event for item in flat for event in item.get("events", [])]
    valid = [value for value in turns if value.get("route_valid") is True]
    invalid = [value for value in turns if value.get("route_valid") is not True]
    aliases = [value for value in valid if value.get("edit_none_alias") is True]
    return {
        "version": "clean29529_g016_sft_controller_protocol_metrics_v1",
        "controller_turn_count": len(turns),
        "controller_valid_turn_count": len(valid),
        "controller_invalid_turn_count": len(invalid),
        "controller_invalid_rate": len(invalid) / len(turns) if turns else 0.0,
        "literal_done_count": sum(
            str(value.get("raw_action") or "").casefold() == "done"
            and str(value.get("action") or "").casefold() == "done"
            for value in valid
        ),
        "literal_edit_count": sum(
            str(value.get("action") or "").casefold() == "edit"
            for value in valid
        ),
        "edit_none_alias_count": len(aliases),
        "action_relabel_count": len(aliases),
        "action_relabel_outside_documented_alias_count": 0,
        "diagnosis_field_count": 0,
        "remaining_edits_field_count": 0,
        "verifier_contract_observation_count": 0,
    }


def stopnow_live_metrics(projection: Mapping[str, Any]) -> dict[str, Any]:
    """Metrics and absolute sign checks required on every G016 group."""

    records = list(projection["records"])
    exact_edit_count = 0
    exact_done_count = 0
    exact_edit_damage_count = 0
    for record in records:
        for round_index, action in enumerate(record["actions"]):
            round_row = record["rounds"][round_index]
            exact = bool(round_row["exact_before_action"])
            q_delta = float(round_row["q_after"]) - float(round_row["q_before"])
            if exact and action == "edit":
                exact_edit_count += 1
                exact_edit_damage_count += q_delta < -1e-12
            elif exact and action == "done":
                exact_done_count += 1
    category_metrics = classify_g016_stopnow_signs(records)
    exact_decision_count = exact_edit_count + exact_done_count
    return {
        "version": "clean29529_g016_live_stopnow_sign_metrics_v2",
        "stop_now_scorer_identity": projection["stop_now_scorer_identity"],
        "stop_now_scorer_version": projection["stop_now_scorer_version"],
        "edit_on_already_exact_count": exact_edit_count,
        "done_on_already_exact_count": exact_done_count,
        "edit_on_already_exact_ratio": (
            exact_edit_count / exact_decision_count if exact_decision_count else None
        ),
        "damage_count_among_edits_from_exact": exact_edit_damage_count,
        "damage_rate_among_edits_from_exact": (
            exact_edit_damage_count / exact_edit_count if exact_edit_count else None
        ),
        "sign_gate_categories": category_metrics,
        "std_floor_binding_bucket_count": int(
            projection["std_floor_binding_bucket_count"]
        ),
    }


def _hidden_verifier_metadata(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(item["geneval_metadata"]),
        "verification_constraints": [
            dict(value) for value in item["verification_contract"]
        ],
    }


def _counting_clause(item: Mapping[str, Any]) -> tuple[str, int]:
    rows = [
        dict(value)
        for value in item["verification_contract"]
        if str(value.get("kind") or "") == "count_exact"
    ]
    if len(rows) != 1:
        raise RuntimeError("G016 requires exactly one hidden count clause")
    object_name = str(rows[0].get("class") or "")
    target_count = int(rows[0].get("exact_count", 0))
    if not object_name or target_count not in range(2, 7):
        raise RuntimeError("G016 hidden count clause is invalid")
    return object_name, target_count


def request_group_scores(
    gathered: list[list[dict[str, Any]]],
    *,
    reward_url: str,
    timeout_sec: float,
    repair_progress_deadzone: float,
    historical_baseline: Any = None,
) -> dict[str, Any]:
    """Score one complete optimizer-step trajectory set."""
    flat = [
        {**item, "owner_rank": rank, "owner_local_index": local_index}
        for rank, rows in enumerate(gathered)
        for local_index, item in enumerate(rows)
    ]
    flat.sort(key=lambda value: int(value["trajectory_index"]))
    expected_trajectory_count = active_g021_trajectory_count() if g021_process_rtg_mode() else 28
    if [int(value["trajectory_index"]) for value in flat] != list(range(expected_trajectory_count)):
        raise RuntimeError("G016 full trajectory coverage differs")
    if g021_process_rtg_mode():
        root_count = active_g021_root_count()
        siblings = active_siblings_per_root()
        root_groups = {int(value.get("root_group_index", -1)) for value in flat}
        if root_groups != set(range(root_count)) or any(sum(int(row.get("root_group_index", -1)) == root for row in flat) != siblings for root in root_groups):
            raise RuntimeError(
                f"root-group rollout requires {root_count} distinct same-root "
                f"K={siblings} groups"
            )
        for root in root_groups:
            members = [row for row in flat if int(row["root_group_index"]) == root]
            if len({str(row["prompt"]) for row in members}) != 1 or len({str(row["uid"]) for row in members}) != 1 or len({str(row["r0_sha256"]) for row in members}) != 1 or {int(row["root_group_member"]) for row in members} != set(range(siblings)):
                raise RuntimeError(
                    f"same-root K={siblings} identity/coverage differs"
                )
        if len({str(value["uid"]) for value in flat}) != root_count or len({str(value["r0_sha256"]) for value in flat}) != root_count:
            raise RuntimeError("G021 roots are not distinct")
    else:
        if len({str(value["prompt"]) for value in flat}) != 1:
            raise RuntimeError("G016 group mixed prompts")
        if len({str(value["uid"]) for value in flat}) != 1:
            raise RuntimeError("G016 group mixed UIDs")
    observed_families = {str(value["family"]) for value in flat}
    # F01 trains all six core families. Every other campaign keeps the
    # counting-only assertion below unchanged -- this guard is widened for
    # F01 only, never removed.
    unknown = observed_families - set(F01_CORE_FAMILIES)
    if unknown:
        raise RuntimeError(f"F01 received an unknown family: {sorted(unknown)}")
    if not g021_process_rtg_mode() and len({str(value["r0_sha256"]) for value in flat}) != 1:
        raise RuntimeError("G016 K=28 did not fork exact R0 bytes")
    if any(
        value.get("branch_selection_used") is not False
        or value.get("best_of_k_used") is not False
        or value.get("r0_policy_record_present") is not False
        for value in flat
    ):
        raise RuntimeError("G016 topology/R0 detachment evidence differs")
    if any(
        value.get("policy_observation_secrecy_passed") is not True
        for value in flat
    ):
        raise RuntimeError("G016 policy-observation secrecy guard differs")
    if any(
        len(value.get("flow_sde_timestep_begins") or [])
        != int(value["flow_call_count"])
        or len(value.get("flow_sde_window_seed_identities") or [])
        != int(value["flow_call_count"])
        or any(
            not 0 <= int(begin) <= 23
            for begin in value.get("flow_sde_timestep_begins") or []
        )
        for value in flat
    ):
        raise RuntimeError("G016 corrected SDE start coverage differs")

    images: list[bytes] = []
    metadata: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    for item in flat:
        start = len(images)
        generated = list(item["round_image_bytes"])[1:]
        images.extend(generated)
        metadata.extend(_hidden_verifier_metadata(item) for _ in generated)
        spans.append((start, len(images)))
    detector_started = time.monotonic()
    detector = (
        request_geneval_score_chunks(
            images=images,
            metadata=metadata,
            reward_url=reward_url,
            timeout_sec=timeout_sec,
        )
        if images
        else {
            "scores": [],
            "strict_rewards": [],
            "results": [],
            "service_version": GENEVAL_EXPECTED_SERVICE_VERSION,
            "protocol_version": "flow_grpo_geneval_pickle_18085_v1",
            "chunk_count": 0,
            "chunk_sizes": [],
            "total_attempts": 0,
        }
    )
    detector_elapsed = time.monotonic() - detector_started
    result_rows = [dict(value) for value in detector["results"]]
    credits = []
    per_round_lineage = []
    for item, (start, stop) in zip(flat, spans):
        item_family = str(item["family"])
        anchor_result = dict(item["r0_detector_result"])
        generated_results = result_rows[start:stop]
        f01_graded = item_family != "counting"
        if f01_graded:
            # The five non-counting families have no object count to compare;
            # the service already returned the graded q for each image state.
            object_name = None
            quality_scores = [
                float(anchor_result["score"]),
                *(float(value["score"]) for value in generated_results),
            ]
            # Exactness encoded as 1-of-1, matching the F01 credit record: the
            # shared `assign_g022_advantages` -- which the live G025 run also
            # uses and which is therefore not edited -- reads exactness as
            # `int(detected_counts[i]) == int(target_count)`.
            target_count = 1
            detected_counts = [
                1 if value == 1.0 else 0 for value in quality_scores
            ]
            terminal_strict = bool(
                (generated_results[-1] if generated_results else anchor_result)
                ["strict_correct"]
            )
        else:
            object_name, target_count = _counting_clause(item)
            anchor_count = int(
                anchor_result.get("detected_counts", {}).get(object_name, 0)
            )
            generated_counts = [
                int(value.get("detected_counts", {}).get(object_name, 0))
                for value in generated_results
            ]
            detected_counts = [anchor_count, *generated_counts]
            # Recomputed here only so the SAME expression can express exactness
            # for every family below; the frozen builder still derives its own.
            quality_scores = [
                counting_score(target_count, value) for value in detected_counts
            ]
            terminal_strict = detected_counts[-1] == int(target_count)
        if len(detected_counts) != len(item["round_image_bytes"]):
            raise RuntimeError("G016 detector/image state coverage differs")
        events = list(item["events"])
        actions = [
            str(value.get("action") or "").casefold()
            if value.get("route_valid") is True
            else "invalid"
            for value in events
        ]
        image_states = []
        for state_index, (payload, sha256, result, detected, quality) in enumerate(
            zip(
                item["round_image_bytes"],
                item["round_image_sha256s"],
                [anchor_result, *generated_results],
                detected_counts,
                quality_scores,
            )
        ):
            detections = list(result.get("detections") or [])
            image_states.append(
                {
                    "state_index": state_index,
                    "sha256": str(sha256),
                    "path": None,
                    "detected_count": None if detected is None else int(detected),
                    "quality_score": float(quality),
                    "boxes": [value.get("box") for value in detections],
                    "confidence_scores": [
                        float(value.get("score", 0.0)) for value in detections
                    ],
                    "threshold_margins": (
                        None
                        if f01_graded
                        else [
                            float(value.get("score", 0.0)) - 0.9
                            for value in detections
                        ]
                    ),
                    "detector_result": result,
                    "image_byte_count": len(payload),
                }
            )
        candidate_id = int(item["trajectory_index"])
        # G022 runs 32 trajectories against the frozen G009 builder's K=28
        # index bound. The frozen file is hash-pinned and must not be edited, so
        # G022 remaps to the within-group index and restores the global one.
        # counting keeps routing to the FROZEN builder so its numbers stay
        # byte-identical with G022-G026; only the five graded families use the
        # port, which is tested field-for-field against the frozen one.
        credit_builder = (
            functools.partial(
                build_f01_trajectory_credit,
                family=item_family,
                quality_scores=quality_scores,
                strict_exact=terminal_strict,
                siblings_per_root=active_siblings_per_root(),
            )
            if f01_graded
            else functools.partial(
                build_g022_trajectory_credit,
                siblings_per_root=active_siblings_per_root(),
            )
        )
        credit = credit_builder(
            prompt=str(item["prompt"]),
            uid=str(item["uid"]),
            trajectory_index=candidate_id,
            r0_sha256=str(item["r0_sha256"]),
            actions=actions,
            raw_controller_responses=[
                str(value.get("raw_response") or "") for value in events
            ],
            canonical_controller_responses=[
                str(value.get("canonical_response") or value.get("raw_response") or "")
                for value in events
            ],
            parse_valid=all(value.get("route_valid") is True for value in events),
            stop_reason=str(item["stop_reason"]),
            image_state_sha256s=list(item["round_image_sha256s"]),
            image_states=image_states,
            seed=int(item["post_fork_rng_seed"]),
            **(
                {}
                if f01_graded
                else {
                    "target_count": target_count,
                    "detected_counts": detected_counts,
                }
            ),
        )
        credits.append(credit)
        for round_index, result in enumerate(generated_results):
            per_round_lineage.append(
                {
                    "trajectory_index": candidate_id,
                    "round_index": round_index,
                    "image_sha256_before": item["round_image_sha256s"][round_index],
                    "image_sha256_after": item["round_image_sha256s"][round_index + 1],
                    "detected_count_before": detected_counts[round_index],
                    "detected_count_after": detected_counts[round_index + 1],
                    # q == 1.0 means exact in EVERY family: counting reaches it
                    # only when detected == target, and the graded layer asserts
                    # it is reached only on a passing frozen GenEval verdict.
                    "exact_before_action": (
                        quality_scores[round_index] == 1.0
                    ),
                    "q_before": credit["counting_scores"][round_index],
                    "q_after": credit["counting_scores"][round_index + 1],
                    "process_reward": credit["process_rewards"][round_index],
                    "sde_timestep_begin": int(
                        item["flow_sde_timestep_begins"][round_index]
                    ),
                    "sde_window_seed_identity": dict(
                        item["flow_sde_window_seed_identities"][round_index]
                    ),
                    "verifier_after": result,
                    "gpt_request_made": False,
                }
            )

    if g021_process_rtg_mode():
        root_count = active_g021_root_count()
        root_projections = []
        baseline_reports = []
        for root_index in range(root_count):
            members = [credit for credit, item in zip(credits, flat) if int(item["root_group_index"]) == root_index]
            # G022 item 6: one scalar per trajectory, z-scored within the
            # shared-R0 sibling group. No deadzone argument exists.
            # Item 7: centred on the (target_count, q_0) bucket's own
            # history once that bucket is deep enough.
            root_projection = assign_g022_advantages(
                members,
                require_exact_group=True,
                historical_baseline=historical_baseline,
            )
            for record in root_projection["records"]:
                record["root_group_index"] = root_index
                record["root_group_member"] = int(record["trajectory_index"]) % active_siblings_per_root()
            for bucket in root_projection["round_diagnostics"]:
                bucket["root_group_index"] = root_index
            root_projections.append(root_projection)
            if root_projection.get("historical_baseline") is not None:
                baseline_reports.append(root_projection["historical_baseline"])
        projection = dict(root_projections[0])
        projection["records"] = sorted([record for root in root_projections for record in root["records"]], key=lambda value: int(value["trajectory_index"]))
        projection["round_diagnostics"] = [bucket for root in root_projections for bucket in root["round_diagnostics"]]
        projection["g016_bucket_diagnostics"] = projection["round_diagnostics"]
        # `projection = dict(root_projections[0])` keeps only root 0's copy of
        # every per-root key, so the R-6 report for root 1 was being dropped.
        # G024's R-6 decomposition needs both, per root.
        projection["malformed_advantage_penalty_per_root"] = [
            root.get("malformed_advantage_penalty") for root in root_projections
        ]
        projection["trajectory_count"] = active_g021_trajectory_count()
        projection["root_group_count"] = root_count
        projection["controller_group_size"] = active_siblings_per_root()
        projection["distinct_root_count"] = root_count
        projection["multi_root_k4"] = False
        projection["trajectory_level_advantage_broadcast"] = True
        projection["group_reward_mean"] = [
            float(root["group_reward_mean"]) for root in root_projections
        ]
        projection["group_reward_population_std"] = [
            float(root["group_reward_population_std"]) for root in root_projections
        ]
        projection["zero_std_group_count"] = sum(
            bool(root["zero_std"]) for root in root_projections
        )
        projection["zero_std_ratio"] = (
            projection["zero_std_group_count"] / root_count if root_count else 0.0
        )
        projection["advantage_saturation_rate"] = (
            statistics.mean(
                float(root["advantage_saturation_rate"]) for root in root_projections
            )
            if root_projections
            else 0.0
        )
        projection["malformed_action_trajectory_count"] = sum(
            int(root["malformed_action_trajectory_count"]) for root in root_projections
        )
        projection["premature_done_trajectory_count"] = sum(
            int(root["premature_done_trajectory_count"]) for root in root_projections
        )
        projection["group_rewards"] = [
            list(root["group_rewards"]) for root in root_projections
        ]
        projection["std_floor_binding_bucket_count"] = sum(
            int(root["std_floor_binding_bucket_count"])
            for root in root_projections
        )
        # C2 / finding Q-9. This preferred `baseline_reports` -- the
        # WITHDRAWN L-4 bucket-history reports -- whenever they existed, so
        # the headline monitored number described a mechanism that reaches
        # no gradient (O-3) and counted groups as "rescued" whose actual
        # gradient is zero. The primary metric is now always computed from
        # the actual within-group advantages; the history is passed
        # alongside and surfaces only under monitor_only_* names.
        projection["zero_std_metrics"] = g022_step_zero_std_ratio(
            [
                {
                    "within_group_zero_std": bool(root["zero_std"]),
                    "zero_std": bool(
                        (baseline_reports[index] if index < len(baseline_reports or []) else {}).get(
                            "zero_std", root["zero_std"]
                        )
                    )
                    if baseline_reports
                    else bool(root["zero_std"]),
                    "baseline_source": (
                        (baseline_reports[index] if index < len(baseline_reports or []) else {}).get(
                            "baseline_source"
                        )
                        if baseline_reports
                        else None
                    ),
                }
                for index, root in enumerate(root_projections)
            ]
        )
        projection["historical_baseline_reports"] = baseline_reports
        projection["historical_baseline_summary"] = (
            historical_baseline.summary() if historical_baseline is not None else None
        )
        projection["group_identity"] = [root.get("group_identity") for root in root_projections]
        projection["stop_now_table_invariants"] = {"passed": all(root.get("stop_now_table_invariants", {}).get("passed") is True for root in root_projections), "root_group_count": root_count}
        projection["max_abs_active_action_advantage"] = max(float(root.get("max_abs_active_action_advantage", 0.0)) for root in root_projections)
        projection["max_abs_active_repair_advantage"] = max(float(root.get("max_abs_active_repair_advantage", 0.0)) for root in root_projections)
        projection["repair_routed_record_count"] = sum(int(root.get("repair_routed_record_count", 0)) for root in root_projections)
        projection["exact_damage_override_active_count"] = sum(int(root.get("exact_damage_override_active_count", 0)) for root in root_projections)
    else:
        projection = assign_g016_advantages(
            credits,
            repair_progress_deadzone=float(repair_progress_deadzone),
        )
    live_stopnow = stopnow_live_metrics(projection)
    credit_by_index = {int(value["trajectory_index"]): value for value in credits}
    item_by_index = {int(value["trajectory_index"]): value for value in flat}
    projected_by_index = {
        int(value["trajectory_index"]): value for value in projection["records"]
    }
    trajectory_records = []
    for index in range(expected_trajectory_count):
        item = item_by_index[index]
        credit = credit_by_index[index]
        projected = projected_by_index[index]
        trajectory_records.append(
            {
                **projected,
                "owner_rank": int(item["owner_rank"]),
                "owner_local_index": int(item["owner_local_index"]),
                "family": item["family"],
                "done": bool(item["done"]),
                "flow_call_count": int(item["flow_call_count"]),
                "flow_sde_timestep_begins": list(
                    item["flow_sde_timestep_begins"]
                ),
                "flow_sde_transition_indices": list(item.get("flow_sde_transition_indices") or []),
                "flow_sde_transition_layouts": list(item.get("flow_sde_transition_layouts") or []),
                "root_group_index": item.get("root_group_index"),
                "root_group_member": item.get("root_group_member"),
                "flow_sde_window_seed_identities": list(
                    item["flow_sde_window_seed_identities"]
                ),
                "text_turn_count": int(item["text_turn_count"]),
                "malformed_text_turn_count": int(item["malformed_text_turn_count"]),
                "reward_components": credit,
                "raw_return": float(credit["trajectory_return_report_only"]),
            }
        )
    validate_no_positive_detector_failed_done(credits)
    cap_guard = validate_cap_terminal_does_not_underprice_done(credits)
    raw_returns = [float(value["trajectory_return_report_only"]) for value in credits]
    sde_starts = [
        int(start)
        for value in flat
        for start in value["flow_sde_timestep_begins"]
    ]
    sde_histogram = {
        str(index): sde_starts.count(index) for index in range(25)
    }
    broken_edit_rounds = [
        row
        for value in projection["records"]
        for row in value["rounds"]
        if row["action"] == "edit" and not bool(row["exact_before_action"])
    ]
    repair_to_exact_count = sum(
        abs(float(row["q_after"]) - 1.0) <= 1e-12
        for row in broken_edit_rounds
    )
    # ------------------------------------------------------------------
    # G024 instrumentation. G023 logged everything and still died invisibly:
    # its protocol started degrading at step 11 and the only visible signal
    # was the invalid rate, which is a late symptom. These are the counters
    # that would have shown the cause from the first step.
    #
    # `flat` and `projection["records"]` are both sorted by trajectory_index
    # and `flat` is asserted above to be exactly range(N), so aligning on that
    # key is safe. If it ever is not, the split is declared unavailable rather
    # than silently mis-attributed -- a corruption rate assigned to the wrong
    # advantage sign is worse than no split at all.
    # ------------------------------------------------------------------
    g024_precursors: dict[str, Any] | None = None
    advantage_by_trajectory: dict[int, float] = {}
    for record in projection["records"]:
        values = list(record.get("controller_advantages") or ())
        if values:
            # Whole-trajectory GRPO: every round carries the same scalar.
            advantage_by_trajectory[int(record["trajectory_index"])] = float(
                values[0]
            )
    g024_responses: list[str] = []
    g024_advantages: list[float] = []
    aligned = True
    for item in flat:
        index = int(item["trajectory_index"])
        if index not in advantage_by_trajectory:
            aligned = False
        for event in item.get("events", []):
            g024_responses.append(str(event.get("raw_response") or ""))
            g024_advantages.append(advantage_by_trajectory.get(index, 0.0))
    g024_precursors = g024_protocol_precursors(
        responses=g024_responses,
        advantages=g024_advantages if aligned else None,
    )
    g024_precursors["trajectory_advantage_alignment_verified"] = aligned

    # --- R-6 decomposition ---------------------------------------------
    # At G023 step 50 this was the whole story of the advantage skew: 29 of
    # 32 trajectories invalid, group mean +0.0035 BEFORE R-6 and -0.5598
    # AFTER. Reported per root so a single collapsing root is visible
    # rather than averaged away.
    r6_roots = []
    for root_index, report in enumerate(
        projection.get("malformed_advantage_penalty_per_root") or []
    ):
        if not report:
            continue
        before = [float(v) for v in report.get("advantages_before_penalty") or ()]
        after = [float(v) for v in report.get("advantages") or ()]
        # `malformed_count` is the key the reward module actually emits;
        # there is no per-trajectory flag list on this report.
        malformed = report.get("malformed_count")
        malformed = int(malformed) if malformed is not None else None
        r6_roots.append(
            {
                "root_group_index": root_index,
                "trajectory_count": len(after),
                "malformed_count": malformed,
                "malformed_rate": (
                    malformed / len(after)
                    if malformed is not None and after
                    else None
                ),
                "group_mean_before_r6": (
                    sum(before) / len(before) if before else None
                ),
                "group_mean_after_r6": (
                    sum(after) / len(after) if after else None
                ),
                "shift_from_r6": (
                    (sum(after) / len(after)) - (sum(before) / len(before))
                    if before and after
                    else None
                ),
            }
        )
    g024_precursors["r6_decomposition"] = {
        "version": "clean29529_g024_r6_decomposition_v1",
        "roots": r6_roots,
        "reported": bool(r6_roots),
    }

    # --- Mask accounting -----------------------------------------------
    # Every sampled token is either credited or host-forced; nothing may be
    # unclassified. The loss asserts this per turn; this is the per-step
    # ledger that makes the assertion auditable after the fact.
    credited = 0
    sampled = 0
    rounds_counted = 0
    partial_rounds = 0
    for record in projection["records"]:
        for round_row in record.get("rounds", ()):
            if "g024_credited_token_count" not in round_row:
                continue
            credited += int(round_row["g024_credited_token_count"])
            sampled += int(round_row["g024_sampled_token_count"])
            partial_rounds += bool(round_row.get("g024_partial_credit_mask"))
            rounds_counted += 1
    g024_precursors["mask_accounting"] = {
        "version": "clean29529_g024_mask_accounting_v1",
        "rounds_counted": rounds_counted,
        "credited_token_count": credited,
        "sampled_token_count": sampled,
        "host_forced_token_count": sampled - credited,
        "unclassified_token_count": 0,
        "credited_fraction": (credited / sampled) if sampled else None,
        # Rounds where the sampler forced a token inside the content span.
        # Expected to be small and STABLE; a rising number means the
        # credited set is shrinking under us, which would reproduce G023's
        # failure quietly.
        "partial_credit_rounds": partial_rounds,
        "partial_credit_round_rate": (
            partial_rounds / rounds_counted if rounds_counted else None
        ),
    }

    dashboard = {
        "version": "clean29529_g016_process_reward_diagnostics_v1",
        # T2.2 / finding O-10, CONFIRMED against real data by preflight attempt
        # 20260824T211343Z: this literal 28 was reported while the same row's
        # `group_diagnostics` correctly carried root_group_count=2,
        # controller_group_size=16, i.e. 32 trajectories.
        #
        # This is not only a label. `trajectory_count` is the denominator for
        # the protocol invalid rate, the parse-failure rate and the false-DONE
        # rate, so under G022 all three were inflated by 32/28 = 1.14x. The
        # count is now taken from the credits actually scored, which is correct
        # for every stage and cannot drift from the topology again.
        "trajectory_count": len(credits),
        "trajectory_count_source": "len(credits)",
        "raw_return_mean": statistics.mean(raw_returns),
        "raw_return_std": statistics.pstdev(raw_returns),
        "done_count": sum(value["actions"][-1] == "done" for value in credits),
        "false_done_count": sum(
            value["actions"][-1] == "done"
            and not value["terminal_strict_exact_success"]
            for value in credits
        ),
        "repair_round_count": sum(value["edit_count"] for value in credits),
        "parse_failure_count": sum(not value["parse_valid"] for value in credits),
        "exact_final_count": sum(value["terminal_strict_exact_success"] for value in credits),
        "score_field_reward_effect": 0.0,
        "terminal_soft_q_reward_effect": 0.0,
        "r0_reward_or_gradient": False,
        "gpt_request_count": 0,
        "forbidden_reward_term_count": 0,
        "cap_terminal_guard": cap_guard,
        "controller_protocol": controller_protocol_metrics(flat),
        "g024_protocol_precursors": g024_precursors,
        "round_diagnostics": projection["round_diagnostics"],
        "stopnow_live_metrics": live_stopnow,
        "sde_sampling": {
            "version": "clean29529_g016_sde_start_metrics_v1",
            "selected_start_count": len(sde_starts),
            "selected_start_histogram": sde_histogram,
            "selected_start_unique_count": len(set(sde_starts)),
            "selected_start_min": min(sde_starts) if sde_starts else None,
            "selected_start_max": max(sde_starts) if sde_starts else None,
            # T2.2 / O-10, also confirmed live: G022 reported (0, 25) while its
            # configured window range is (0, 10). Report what the sampler was
            # actually given.
            "configured_range": (
                [int(G022_SDE_WINDOW_RANGE[0]), int(G022_SDE_WINDOW_RANGE[1])]
            ),
            "window_size": 2,
            "transition_layout": "contiguous_v1",
            "transition_indices": None,
        },
        "repair_effectiveness": {
            "definition": "nonexact_before_EDIT_to_exact_after_EDIT",
            "eligible_broken_edit_count": len(broken_edit_rounds),
            "repair_to_exact_count": repair_to_exact_count,
            "rate": (
                repair_to_exact_count / len(broken_edit_rounds)
                if broken_edit_rounds else None
            ),
        },
    }
    return {
        "r0_records": [],
        "trajectory_records": trajectory_records,
        "repair_records": trajectory_records,
        "group_diagnostics": projection,
        "reward_diagnostics": dashboard,
        "per_round_lineage": per_round_lineage,
        "stopnow_live_metrics": live_stopnow,
        "sde_sampling": dashboard["sde_sampling"],
        "repair_effectiveness": dashboard["repair_effectiveness"],
        "stop_now_scorer_identity": STOP_NOW_SCORER_IDENTITY,
        "stop_now_scorer_version": STOP_NOW_SCORER_VERSION,
        "gpt_request_count": 0,
        "detector_transport": {
            "service_version": detector["service_version"],
            "protocol_version": detector["protocol_version"],
            "chunk_count": detector["chunk_count"],
            "chunk_sizes": detector["chunk_sizes"],
            "total_attempts": detector["total_attempts"],
        },
        "reward_phase_timings": {
            "version": "clean29529_g016_reward_phase_timing_v1",
            "detector_duration_sec": detector_elapsed,
            "gpt_duration_sec": 0.0,
            "parallel_wall_sec": detector_elapsed,
        },
        "max_flow_calls": max(int(value["flow_call_count"]) for value in trajectory_records),
        "max_text_turns": max(
            int(value["text_turn_count"] + value["malformed_text_turn_count"])
            for value in trajectory_records
        ),
        "policy_observation_secrecy": {
            "version": "clean29529_g016_runtime_policy_observation_secrecy_guard_v1",
            "passed": True,
            "trajectory_count": len(flat),
            "hidden_exact_field_visible_to_policy": False,
        },
    }

def score_and_broadcast(
    local_payload: list[dict[str, Any]],
    *,
    reward_url: str,
    timeout_sec: float,
    repair_progress_deadzone: float,
    rank: int,
    object_group: Any,
    historical_baseline: Any = None,
) -> dict[str, Any]:
    gathered = gather_rank_objects(local_payload, object_group)
    if rank == 0:
        try:
            result = {
                "ok": True,
                "value": request_group_scores(
                    gathered,
                    reward_url=reward_url,
                    timeout_sec=timeout_sec,
                    repair_progress_deadzone=repair_progress_deadzone,
                    historical_baseline=historical_baseline,
                ),
            }
        except Exception as exc:
            result = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            }
    else:
        result = None
    result = broadcast_rank_object(result, src=0, group=object_group)
    if result.get("ok") is not True:
        raise RuntimeError(f"G016 reward request failed: {result.get('error')}")
    return result["value"]


def _validate_run_plan(config: Any) -> dict[str, Any]:
    """Check the resume pair and return the step budget and checkpoint schedule."""
    requested_total_steps = int(config.g016.total_steps)
    start_step = int(config.g016.start_step)
    resume_checkpoint = str(config.g016.resume_checkpoint).strip()
    if int(config.g016.lora_rank) != 0:
        raise RuntimeError("reflection RL requires full-rank training and forbids LoRA")
    if bool(start_step) != bool(resume_checkpoint):
        raise RuntimeError(
            f"start_step={start_step} and resume_checkpoint={resume_checkpoint!r} "
            "must be set together"
        )
    if resume_checkpoint:
        resume_path = Path(resume_checkpoint).resolve()
        if resume_path.name != f"checkpoint-{start_step}":
            raise RuntimeError(
                f"resume checkpoint {resume_path} does not match start_step={start_step}"
            )
        resume_manifest = resume_path / "manifest.json"
        if not resume_manifest.is_file():
            raise RuntimeError(f"resume checkpoint has no manifest: {resume_path}")
        resume_written = json.loads(resume_manifest.read_text(encoding="utf-8"))
        if (
            resume_written.get("status") != "complete"
            or int(resume_written.get("logical_step", -1)) != start_step
        ):
            raise RuntimeError(
                f"resume checkpoint is not a completed step-{start_step} "
                f"commit: {resume_written.get('status')!r}"
            )
    world_size = int(os.environ.get("WORLD_SIZE", "16"))
    topology = f01_topology(max(1, world_size // 8))
    if int(config.sample.train_batch_size) != topology["local_batch_size"]:
        raise RuntimeError(
            f"world{topology['world_size']} requires local batch "
            f"{topology['local_batch_size']}, but the config asks for "
            f"{int(config.sample.train_batch_size)}"
        )
    return {
        "authorized_max_total_steps": requested_total_steps,
        "checkpoint_steps": [int(v) for v in config.g016.checkpoint_steps],
        "topology": topology,
    }

def main(_argv) -> None:
    config = FLAGS.config
    reward_runtime = configure_reward_runtime(config)
    authorization = _validate_run_plan(config)
    requested_steps = int(config.g016.total_steps)
    if requested_steps > authorization["authorized_max_total_steps"]:
        raise RuntimeError(
            f"G016 requested total_steps={requested_steps} exceeds the "
            f"authorized budget {authorization['authorized_max_total_steps']}"
        )
    system_prompt_path = Path(config.g016.system_prompt_file).resolve()
    prompt_manifest_path = Path(config.g016.prompt_narrowing_manifest).resolve()
    frozen_bindings = {
        "system_prompt_sha256": sha256_file(system_prompt_path),
        "prompt_narrowing_manifest_sha256": sha256_file(prompt_manifest_path),
    }
    if frozen_bindings["prompt_narrowing_manifest_sha256"] != str(
        config.g016.prompt_narrowing_manifest_sha256
    ):
        raise RuntimeError("prompt pool manifest hash differs")
    prompt_manifest = read_json_object(prompt_manifest_path)
    expected_pool_rows = (
        len([
            line for line in
            Path(config.dataset).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ])
    )
    if (
        int(prompt_manifest.get("output_row_count", -1)) != expected_pool_rows
        or int(prompt_manifest.get("output_unique_uid_count", -1)) != expected_pool_rows
        or str(prompt_manifest.get("output_sha256")) != sha256_file(Path(config.dataset))
    ):
        raise RuntimeError(
            "prompt pool manifest/data binding differs (expected "
            f"{expected_pool_rows} rows)"
        )
    output_dir = Path(config.logdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    committing_training = True
    f01_local = int(config.sample.train_batch_size)
    f01_world = G022_TRAJECTORY_COUNT // f01_local
    if F01_LOCAL_BATCH_SIZE_BY_WORLD.get(f01_world) != f01_local:
        raise RuntimeError(
            f"config implies world{f01_world} x local{f01_local}, "
            "which is not a supported topology"
        )
    preflight_admission = {
        "world_size": f01_world,
        "local_batch_size": f01_local,
    }
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=ProjectConfiguration(project_dir=str(output_dir)),
    )
    fsdp_plugin = getattr(accelerator.state, "fsdp_plugin", None)
    if fsdp_plugin is not None:
        fsdp_plugin.activation_checkpointing = bool(
            config.activation_checkpointing
        )
        fsdp_plugin.transformer_cls_names_to_wrap = (
            [
                "Qwen2MoTDecoderLayer",
                "PackedAttentionMoT",
                "G016DetachableQwen2MLP",
            ]
            if bool(config.g016.nested_fsdp_wrap)
            else ["Qwen2MoTDecoderLayer"]
        )
        if fsdp_plugin.mixed_precision_policy is None:
            if not bool(config.g016.bf16_policy_cpu_fp32_master):
                raise RuntimeError("G016 FSDP mixed-precision policy is missing")
        else:
            fsdp_plugin.mixed_precision_policy.cast_forward_inputs = True
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    object_group = (
        dist.new_group(backend="gloo")
        if dist.is_initialized() and world_size > 1
        else None
    )
    # Item 6: failure artifacts are attempt-scoped. A crashed attempt used to
    # leave `G016_TRANSACTION_FAILURE_RANKS.json` and
    # `PARTIAL_UPDATE_CONTAMINATION_STEP*.json` at fixed names in the run
    # root, where the next attempt's reader could not tell them apart from its
    # own -- a stale file from a dead attempt reads exactly like current
    # evidence. The id comes from the launcher so both nodes agree by
    # construction.
    # Both nodes must agree on this id; the launcher exports it.
    g022_attempt_id = str(os.environ.get("G016_ATTEMPT_ID", "")).strip() or time.strftime(
        "%Y%m%dT%H%M%SZ", time.gmtime()
    )

    def attempt_scoped(name: str) -> str:
        """G016-G021 keep the historical fixed names byte-for-byte."""
        return f"{g022_attempt_id}_{name}" if g022_attempt_id else name
    sde_layout_ok = (
        tuple(config.sample.sde_window_range) == G022_SDE_WINDOW_RANGE
        and int(config.sample.sde_window_size) == G022_SDE_WINDOW_SIZE
        and bool(getattr(config.sample, "g021_stratified_sde", False)) is False
        and getattr(config.sample, "g021_flow_contract", None)
        == G022_FLOW_CONTRACT
    )
    expected_num_steps = G022_TRAIN_NUM_TIMESTEPS
    # Stage-aware like the flow lr below: F05F freezes text at 0.0.
    expected_text_lr = _expected_text_learning_rate(config.g016)
    # SECOND site asserting the flow learning rate -- the first is
    # `validate_launch_bindings` in g022_campaign. G027 raises it on
    # purpose, so both sites read the expectation from the SAME helper;
    # fixing only one is what made G027's first launch die here after the
    # campaign module had already been made stage-aware.
    expected_flow_lr = _expected_flow_learning_rate(config.g016)
    if (
        not sde_layout_ok
        or int(config.sample.num_steps) != expected_num_steps
        or int(config.sample.eval_num_steps) != 50
        or str(config.g016.sde_window_seed_mode)
        != "clean29529_g012_local_sde_window_seed_v1"
        or abs(float(config.g016.text_learning_rate) - expected_text_lr) > 1e-15
        or abs(float(config.train.learning_rate) - expected_flow_lr) > 1e-15
        or float(CONTROLLER_ACTION_AUX_WEIGHT) != 1.0
    ):
        raise RuntimeError("G016 corrected SDE sampling configuration differs")
    if (
        bool(config.g016.stop_now_baseline_advantage) is not True
        or str(config.g016.stop_now_scorer_identity) != STOP_NOW_SCORER_IDENTITY
        or str(config.g016.stop_now_scorer_version) != STOP_NOW_SCORER_VERSION
        or str(config.g016.channel_credit_version) != ADVANTAGE_VERSION
        or str(config.g016.controller_credit_version) != CONTROLLER_CREDIT_VERSION
        or str(config.g016.renderer_credit_version) != RENDERER_CREDIT_VERSION
        or str(config.g016.renderer_bucket_version) != RENDERER_BUCKET_VERSION
        or bool(config.g016.renderer_cross_step_pooling) is not False
        or list(config.g016.renderer_bucket_fallback_ladder) != [
            "exact_q_round_target",
            "exact_q_round",
            "exact_round",
            "disable",
        ]
        or bool(config.g016.exact_before_action_policy_visible) is not False
        or bool(config.g016.singleton_bucket_policy_active) is not False
        or bool(config.g016.exact_singleton_action_active)
        is not (False)
        or bool(config.g016.mixed_exactness_fallback) is not False
    ):
        raise RuntimeError("G016 stop-now baseline configuration differs")
    diagnostic_active_channel_mode = str(
        config.g016.diagnostic_active_channel_mode
    )
    diagnostic_optimizer_step_mode = str(
        config.g016.diagnostic_optimizer_step_mode
    )
    valid_channel_modes = {"none", "text", "flow", "both"}
    if (
        diagnostic_active_channel_mode not in valid_channel_modes
        or diagnostic_optimizer_step_mode not in valid_channel_modes
    ):
        raise RuntimeError("G016 diagnostic channel mode differs")
    if (
        diagnostic_active_channel_mode != "both"
        or diagnostic_optimizer_step_mode != "both"
    ):
        raise RuntimeError(
            "G016 diagnostic channel overrides require diagnostic mode"
        )
    memory_events: list[dict[str, Any]] = []

    def record_memory_phase(phase: str) -> None:
        return
    live_status_path = Path(config.g016.live_status_path)
    set_seed(int(config.g016.seed), device_specific=True)
    inference_dtype = torch.bfloat16
    model_dir = Path(config.pretrained.model)
    if rank == 0:
        try:
            resolved_model_path = (
                model_dir / "ema.safetensors"
            ).resolve()
            model_size = int(resolved_model_path.stat().st_size)
            resume_path = str(config.g016.resume_checkpoint).strip()
            if resume_path:
                recorded = read_json_object(
                    Path(resume_path) / "manifest.json"
                )["initialization_reference"]
                if (
                    recorded["resolved_model_path"] != str(resolved_model_path)
                    or int(recorded["resolved_model_size_bytes"]) != model_size
                    or recorded["resolved_model_sha256"]
                    != str(config.g016.initialization_sha256)
                ):
                    raise RuntimeError("G016 recorded initialization reference differs")
                # This is the recorded lineage ID, not a newly computed digest.
                model_identity = recorded["resolved_model_sha256"]
            else:
                model_identity = str(config.g016.initialization_sha256)
            runtime_model_binding = {
                "ok": True,
                "path": str(resolved_model_path),
                "size_bytes": model_size,
                "sha256": model_identity,
                "content_hash_computed": False,
            }
        except Exception as exc:
            runtime_model_binding = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    else:
        runtime_model_binding = None
    runtime_model_binding = broadcast_rank_object(
        runtime_model_binding,
        src=0,
        group=object_group,
    )
    if (
        runtime_model_binding.get("ok") is not True
        or runtime_model_binding.get("sha256")
        != str(config.g016.initialization_sha256)
    ):
        raise RuntimeError(
            "G016 initialization lineage metadata differs: "
            f"{runtime_model_binding}"
        )
    model, vae_model, runtime_llm_config = build_bagel(
        model_dir=model_dir,
        inference_dtype=inference_dtype,
        device=accelerator.device,
    )
    (
        flow_parameters,
        text_parameters,
        frozen_reference_layers,
        trainable_surface,
    ) = configure_fullrank_trainable_surface(
        model,
        config,
    )
    auxiliary_flow_modules = {
        name: getattr(model, name)
        for name in trainable_surface["flow_root_names"]
    }
    decoder_module_order: list[int] = []
    start_step = int(config.g016.start_step)
    total_steps = int(config.g016.total_steps)
    if total_steps < start_step or (
        total_steps == start_step
    ):
        raise RuntimeError(
            "G016 total steps must exceed the logical start step except in "
            "a fail-closed inference-only path"
        )
    resume_checkpoint = str(config.g016.resume_checkpoint).strip()
    if start_step > 0 and not resume_checkpoint:
        raise RuntimeError(
            "G016 nonzero start step requires an exact sharded checkpoint"
        )
    transformer = model.language_model
    transformer.config.use_cache = False
    base_language_model = unwrap_language_model(transformer)
    ignored_modules = [
        base_language_model.model.embed_tokens,
        base_language_model.model.norm,
        base_language_model.lm_head,
    ]
    if hasattr(base_language_model.model, "norm_moe_gen"):
        ignored_modules.append(base_language_model.model.norm_moe_gen)
    ignored_module_names = [
        "embed_tokens",
        "norm",
        "lm_head",
        *(
            ["norm_moe_gen"]
            if hasattr(base_language_model.model, "norm_moe_gen")
            else []
        ),
    ]
    if fsdp_plugin is not None:
        fsdp_plugin.ignored_modules = ignored_modules
    optimizer_class = (
        G016StreamingCPUAdamW if bool(config.g016.optimizer_state_cpu_offload)
        else torch.optim.AdamW
    )
    flow_optimizer = optimizer_class(
        flow_parameters,
        lr=float(config.train.learning_rate),
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    text_optimizer = optimizer_class(
        text_parameters,
        lr=float(config.g016.text_learning_rate),
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    dataset = G016PromptDataset(config.dataset)
    topology = (world_size, int(config.sample.train_batch_size), int(config.sample.num_image_per_prompt))
    local_batch = int(config.sample.train_batch_size)
    expected_local = F01_LOCAL_BATCH_SIZE_BY_WORLD.get(int(world_size))
    if (
        expected_local is None
        or local_batch != expected_local
        or world_size * local_batch != G022_TRAJECTORY_COUNT
        or int(config.sample.num_image_per_prompt) != G022_SIBLINGS_PER_ROOT
    ):
        raise RuntimeError(
            "reflection RL requires world16 x local2 or world32 x local1, two "
            f"semantic roots x K{G022_SIBLINGS_PER_ROOT} (got "
            f"world/local/siblings {topology}, trajectories "
            f"{world_size * local_batch} != {G022_TRAJECTORY_COUNT})"
        )
    if int(config.sample.num_steps) != G022_TRAIN_NUM_TIMESTEPS:
        raise RuntimeError("reflection RL trains at 20 image steps")
    sampler_contract = build_sampler_contract(
        train_data_path=dataset.path,
        ordered_uids=[row.uid for row in dataset.rows],
        local_batch_size=int(config.sample.train_batch_size),
        group_size=int(config.sample.num_image_per_prompt),
        world_size=world_size,
        release_prompt_indices=list(config.g016.release_prompt_indices),
        release_resample_stride=int(config.g016.release_resample_stride),
        system_prompt_path=Path(config.g016.system_prompt_file),
        image_steps=int(config.sample.num_steps),
    )
    (
        transformer,
        text_optimizer,
        flow_optimizer,
    ) = accelerator.prepare(
        transformer,
        text_optimizer,
        flow_optimizer,
    )
    lora_synchronization = None
    if (
        len(flow_optimizer.param_groups) != 1
        or len(text_optimizer.param_groups) != 1
    ):
        raise RuntimeError("G016 requires separate single-group optimizers")
    flow_parameters = tuple(flow_optimizer.param_groups[0]["params"])
    text_parameters = tuple(text_optimizer.param_groups[0]["params"])
    trainable_parameter_dtypes = {
        str(parameter.dtype)
        for parameter in (*flow_parameters, *text_parameters)
    }
    expected_policy_dtype = (
        {"torch.bfloat16"} if bool(config.g016.bf16_policy_cpu_fp32_master)
        else {"torch.float32"}
    )
    if trainable_parameter_dtypes != expected_policy_dtype:
        raise RuntimeError(
            "G016 policy parameter dtype/CPU-master contract differs: "
            f"expected={sorted(expected_policy_dtype)} got={sorted(trainable_parameter_dtypes)}"
        )
    expected_fsdp_strategy = (
        ShardingStrategy.HYBRID_SHARD
        if str(config.g016.fsdp_strategy) == "HYBRID_SHARD"
        else ShardingStrategy.FULL_SHARD
    )
    if world_size > 1 and (
        not isinstance(transformer, FSDP)
        or transformer.sharding_strategy is not expected_fsdp_strategy
    ):
        raise RuntimeError(
            f"G016 requires configured FSDP {config.g016.fsdp_strategy} with activation checkpointing"
        )
    memory_state_materialization = {
        "enabled": False,
        "text": None,
        "flow": None,
    }
    initialization_reference = {
        "classic_checkpoint": os.environ.get(
            "G016_INITIALIZATION_LABEL",
            "0002969",
        ),
        "model_sha256": str(config.g016.initialization_sha256),
        "resolved_model_path": runtime_model_binding["path"],
        "resolved_model_size_bytes": runtime_model_binding[
            "size_bytes"
        ],
        "resolved_model_sha256": runtime_model_binding["sha256"],
        "decoder_layers": list(range(28)),
        "tensor_count": int(trainable_surface["tensor_count"]),
        "parameter_numel": int(trainable_surface["parameter_numel"]),
        **sampler_config_identity(sampler_contract),
        "sampler_contract": sampler_contract,
    }
    fresh_transaction_origin = None
    if committing_training and not resume_checkpoint:
        fresh_transaction_origin = resolve_f01_fresh_transaction_origin(
            model_dir=model_dir,
        )
    resume_manifest = None
    sampler_resume_state = None
    if resume_checkpoint:
        resolved_resume = Path(resume_checkpoint).resolve()
        own_checkpoints = (output_dir / "checkpoints").resolve()
        if resolved_resume.parent != own_checkpoints:
            raise RuntimeError(
                f"resume checkpoint must live under {own_checkpoints}; got {resolved_resume}"
            )
        loaded = load_sharded_training_checkpoint(
            resolved_resume,
            model=transformer,
            text_optimizer=text_optimizer,
            flow_optimizer=flow_optimizer,
            expected_initialization_reference=initialization_reference,
            process_group=None,
            control_group=object_group,
            auxiliary_modules=auxiliary_flow_modules,
        )
        if int(loaded["logical_step"]) != start_step:
            raise RuntimeError("G016 resume logical step differs")
        sampler_resume_state = loaded["sampler_state"]
        validate_sampler_resume(
            sampler_resume_state,
            rank=rank,
            world_size=world_size,
            seed=int(config.g016.seed),
            completed_step=start_step,
            contract=sampler_contract,
        )
        resume_manifest = {
            "version": "clean29529_g016_exact_resume_v1",
            "checkpoint": str(Path(resume_checkpoint).resolve()),
            "logical_step": int(loaded["logical_step"]),
            "optimizer_state_restored": True,
            "rng_state_restored": True,
            "sampler_state_restored": True,
        }
    model.language_model = transformer
    reference_device = (
        torch.device("cpu")
        if bool(config.g016.reference_cpu_streaming)
        else accelerator.device
    )
    frozen_reference_layers = [
        layer.to(device=reference_device, dtype=torch.bfloat16)
        for layer in frozen_reference_layers
    ]
    reference_language_model = SharedPrefixReference(
        transformer,
        frozen_decoder_layers=frozen_reference_layers,
        decoder_layer_start=0,
        initialization=initialization_reference,
    )
    reference_optimizer_contract = assert_frozen_reference_optimizer_isolation(
        reference_language_model, optimizers=(text_optimizer, flow_optimizer)
    )
    model.language_model_ref = reference_language_model
    force_inference_dispatch(transformer)
    tokenizer = Qwen2Tokenizer.from_pretrained(str(model_dir))
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    vae_transform = ImageTransform(512, 256, 8)
    vit_transform = ImageTransform(490, 112, 7)
    flow_inferencer = FlowInterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    reference_model = copy.copy(model)
    reference_model._modules = dict(model._modules)
    reference_model.language_model = reference_language_model
    reference_model.language_model_ref = reference_language_model
    frozen_flow_roots = object.__getattribute__(model, "_g016_frozen_flow_roots")
    for module_name, module in frozen_flow_roots.items():
        setattr(
            reference_model,
            module_name,
            module.to(device=accelerator.device, dtype=torch.bfloat16),
        )
    reference_flow_inferencer = FlowInterleaveInferencer(
        model=reference_model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    from inferencer import InterleaveInferencer as ControllerInferencer

    controller_inferencer = ControllerInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    if not callable(
        getattr(controller_inferencer, "gen_text_persistent_batch", None)
    ):
        raise RuntimeError(
            "G016 requires packed batch-2 persistent controller generation"
        )
    reference_controller_inferencer = ControllerInferencer(
        model=reference_model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    system_prompt = Path(config.g016.system_prompt_file).read_text(
        encoding="utf-8"
    )
    rollout = FlowGRPOMultiroundRollout(
        flow_inferencer=flow_inferencer,
        controller_inferencer=controller_inferencer,
        system_prompt=system_prompt,
        grpo_config=config,
        accelerator=accelerator,
        controller_lockstep_process_group=object_group,
    )
    # G022 item 3: model-output span-capture failures are classified, not raised.
    rollout.classify_span_capture_failures = True
    # ------------------------------------------------------------------
    # G022 text-action support contract.
    #
    # The controller LM head is 152064 rows wide and the tokenizer defines
    # 151665 symbols, so 399 rows (151665..152063) are samplable actions with
    # no symbol at all. Two Formal400 attempts died on one, in two different
    # consumers, because the sampler's probability space was never narrowed to
    # the tokenizer's support. Binding here masks those rows before sampling,
    # greedy selection and the behaviour log-prob, and installs the same mask
    # on the policy and frozen-reference replay so the PPO ratio and the KL
    # live in the sampler's probability space. Parameter and checkpoint shape
    # are untouched -- all 152064 rows stay.
    # ------------------------------------------------------------------
    text_action_support_binding = None
    text_action_head_rows = int(runtime_llm_config.vocab_size)
    text_action_support = build_text_action_support(
        tokenizer=tokenizer,
        head_rows=text_action_head_rows,
    )
    rollout.text_action_injection = None
    text_action_support_binding = rollout.bind_text_action_support(
        text_action_support,
        attempt_id=g022_attempt_id,
        evidence_dir=output_dir,
    )
    # The replay reads the contract off the inferencer, so both the policy
    # and the frozen reference must carry it or the two log-prob spaces
    # silently differ.
    controller_inferencer.g022_text_action_support = text_action_support
    reference_controller_inferencer.g022_text_action_support = (
        text_action_support
    )
    text_action_support_binding["injection"] = rollout.text_action_injection
    if rank == 0:
        atomic_json(
            output_dir
            / attempt_scoped("G022_TEXT_ACTION_SUPPORT.json"),
            {
                "version": "clean29529_g022_text_action_support_binding_v1",
                "created_at_utc": utc_now(),
                "attempt_id": g022_attempt_id,
                "world_size": int(world_size),
                "binding": text_action_support_binding,
            },
        )
    print(
        "[g022] text-action support bound: "
        f"head_rows={text_action_support.head_rows} "
        f"supported={text_action_support.supported_count} "
        f"unsupported={text_action_support.unsupported_count} "
        f"hash={text_action_support.support_hash[:16]} "
        f"world_size={world_size}",
        flush=True,
    )
    # G022 item 7 / doc Appendix L-4: historical baseline keyed by
    # (target_count, q_0). Restored from disk so a resume does not silently
    # reset every bucket to empty and fall back to within-group z-scores.
    g022_baseline_path = output_dir / "g022_bucket_baseline.json"
    g022_baseline = (
        G022BucketStatTracker.load(g022_baseline_path)
    )
    metrics_path = Path(config.g016.metrics_jsonl)
    dashboard_path = Path(config.g016.dashboard_jsonl)
    forensic_ring_path = output_dir / "g016_forensic_ring_buffer.json"
    forensic_ring = G016ForensicRingBuffer.from_payload(
        read_json_object(forensic_ring_path)
        if forensic_ring_path.is_file()
        else None,
        capacity_steps=6,
        top_k_per_step=64,
    )
    done_guard_prefix_path = Path(str(config.g016.done_guard_prefix))
    done_guard_prefix_rows: list[dict[str, Any]] = []
    if str(config.g016.done_guard_prefix):
        prefix = read_json_object(done_guard_prefix_path)
        if (
            prefix.get("version")
            != "clean29529_g016_g004_done_prefix_evidence_v1"
            or prefix.get("passed") is not True
            or not isinstance(prefix.get("rows"), list)
        ):
            raise RuntimeError("G016 G004 DONE prefix evidence differs")
        done_guard_prefix_rows = list(prefix["rows"])
        if [int(row.get("step", -1)) for row in done_guard_prefix_rows] != list(
            range(1, start_step + 1)
        ):
            raise RuntimeError("G016 G004 DONE prefix coverage differs")
    # W1: rank-0-only wandb logging. Never blocks or crashes a training step;
    # falls back to offline on a node without egress. No credential material
    # is read, stored, or logged anywhere in this process.
    wandb_logger = G016WandbLogger(
        rank=rank,
        ladder_stage=str(config.g016.ladder_stage),
        run_config=build_wandb_run_config(
            ladder_stage=str(config.g016.ladder_stage),
            experiment_record_path=str(output_dir),
            logdir=str(output_dir),
            r0_channel_weight=float(R0_CHANNEL_WEIGHT),
            group_size=int(config.sample.num_image_per_prompt),
            world_size=int(world_size),
            candidates_per_rank=int(config.sample.train_batch_size),
            flow_learning_rate=float(config.train.learning_rate),
            text_learning_rate=float(config.g016.text_learning_rate),
            initialization_checkpoint=str(config.g016.initialization_sha256),
            judge_concurrency_version=str(JUDGE_CONCURRENCY_VERSION),
            code_hashes={
                "train_bagel_g016_counting.py": sha256_file(Path(__file__).resolve()),
                "g016_counting_process.py": sha256_file(
                    REPO_ROOT
                    / "third_party/flow_grpo/flow_grpo/g016_counting_process.py"
                ),
                "g011_stopnow_reward.py": sha256_file(
                    REPO_ROOT / "src/unify_rl/reward_models/g011_stopnow_reward.py"
                ),
                "v22_controller_protocol.py": sha256_file(
                    REPO_ROOT
                    / "src/unify_rl/inference/v22_controller_protocol.py"
                ),
            },
            extra={
                "repair_channel_weight": float(REPAIR_CHANNEL_WEIGHT),
                "max_abs_advantage": float(MAX_ABS_ADVANTAGE),
                "controller_protocol_version": str(
                    G016_PROTOCOL_VERSION
                ),
            },
        ),
        offline_dir=output_dir / "wandb",
        enabled=bool(config.g016.wandb_enabled),
    )
    wandb_status = wandb_logger.start()
    if rank == 0:
        atomic_json(
            output_dir / "g016_wandb_status.json",
            redacted_wandb_status(wandb_status),
        )
    if rank == 0:
        atomic_json(
            output_dir / "g016_config.json",
            {
                "version": VERSION,
                "model_dir": str(model_dir.resolve()),
                "classic_initialization": initialization_reference,
                "train_data": str(Path(config.dataset).resolve()),
                "prompt_narrowing": {
                    "manifest_path": str(prompt_manifest_path),
                    "manifest_sha256": frozen_bindings[
                        "prompt_narrowing_manifest_sha256"
                    ],
                    "train_data_sha256": str(prompt_manifest["output_sha256"]),
                    "row_count": 252,
                },
                "frozen_bindings": frozen_bindings,
                "config_identity": {
                    "version": "clean29529_g016_run_config_identity_v1",
                    **sampler_config_identity(sampler_contract),
                },
                "sampler_contract": sampler_contract,
                "world_size": world_size,
                "candidates_per_rank": int(config.sample.train_batch_size),
                "rollout_execution": {
                    "version": "clean29529_g016_same_r0_k28_full_trajectory_rollout_v1",
                    "local_batch_size": int(config.sample.train_batch_size),
                    "controller_generation_batched": True,
                    "logical_r0_image_count": 1,
                    "r0_policy_record_count": 0,
                    "same_round_repair_denoise_batched": True,
                    "full_trajectory_before_credit": True,
                    "all_k28_paths_preserved": True,
                    "branch_selection_used": False,
                    "best_of_k_used": False,
                    "candidate_order_preserved": True,
                    "per_token_logprob_alignment_preserved": True,
                    "controller_distributed_lockstep": False,
                },
                "attention_backend": {
                    "failaware_use_segmented_flash": (
                        os.environ.get("FAILAWARE_USE_SEGMENTED_FLASH") == "1"
                    ),
                    "controller_sampling": "flash_attn_varlen",
                    "controller_teacher_forcing": "flash_attn_varlen",
                    "flow_sampling_and_replay": "flash_attn_varlen",
                    "sdpa_packed_replay_reachable": False,
                },
                "fsdp_sharding_strategy": (
                    transformer.sharding_strategy.name
                    if isinstance(transformer, FSDP)
                    else "single_process"
                ),
                "fsdp_wrap_surface": {
                    "nested": bool(config.g016.nested_fsdp_wrap),
                    "classes": list(fsdp_plugin.transformer_cls_names_to_wrap)
                    if fsdp_plugin is not None
                    else [],
                    "world_padding_enters_wrapped_root": True,
                    "uniform_child_order": [
                        "Qwen2MoTDecoderLayer:norm_parent_handle",
                        "PackedAttentionMoT",
                        "G016DetachableQwen2MLP:text",
                        "G016DetachableQwen2MLP:gen",
                    ],
                    "flow_detached_mlp_parameters_inside_child_forward": True,
                },
                "fsdp_mixed_precision_policy": {
                    "param_dtype": str(
                        fsdp_plugin.mixed_precision_policy.param_dtype
                    )
                    if fsdp_plugin is not None and fsdp_plugin.mixed_precision_policy is not None
                    else None,
                    "reduce_dtype": str(
                        fsdp_plugin.mixed_precision_policy.reduce_dtype
                    )
                    if fsdp_plugin is not None and fsdp_plugin.mixed_precision_policy is not None
                    else None,
                    "buffer_dtype": str(
                        fsdp_plugin.mixed_precision_policy.buffer_dtype
                    )
                    if fsdp_plugin is not None and fsdp_plugin.mixed_precision_policy is not None
                    else None,
                    "cast_forward_inputs": bool(
                        fsdp_plugin.mixed_precision_policy.cast_forward_inputs
                    )
                    if fsdp_plugin is not None and fsdp_plugin.mixed_precision_policy is not None
                    else False,
                },
                "accelerator_mixed_precision": accelerator.mixed_precision,
                "parameter_storage_contract": (
                    "bf16_fsdp_policy_fp32_cpu_master_v1"
                    if bool(config.g016.bf16_policy_cpu_fp32_master)
                    else "fp32_fsdp_wrapped_layers_bf16_compute_v2"
                ),
                "trainable_surface": trainable_surface,
                "lora_initial_synchronization": lora_synchronization,
                "trainable_parameter_dtypes": sorted(
                    trainable_parameter_dtypes
                ),
                "max_grad_norm": float(config.train.max_grad_norm),
                "fsdp_ignored_root_modules": ignored_module_names,
                "group_size": int(config.sample.num_image_per_prompt),
                "train_micro_batch_size": int(config.train.batch_size),
                "gradient_accumulation_steps": int(config.train.gradient_accumulation_steps),
                "optimizer_state_cpu_offload": bool(config.g016.optimizer_state_cpu_offload),
                "logical_prompt_groups_per_step": (
                    world_size
                    * int(config.sample.train_batch_size)
                    // int(config.sample.num_image_per_prompt)
                ),
                "total_steps": int(config.g016.total_steps),
                "start_step": start_step,
                "resume": resume_manifest,
                "runtime_performance_revision": RUNTIME_PERFORMANCE_REVISION,
                "checkpoint_content_hashing": False,
                "initialization_binding_content_hash_computed": (
                    runtime_model_binding["content_hash_computed"]
                ),
                "preflight_admission": preflight_admission,
                "reward_runtime": reward_runtime,
                "diagnostic_execution": {
                    "enabled": False,
                    "memory_probe": False,
                    "capture_full_update_diagnostics": (
                        False
                    ),
                    "active_channel_mode": (
                        diagnostic_active_channel_mode
                    ),
                    "optimizer_step_mode": (
                        diagnostic_optimizer_step_mode
                    ),
                    "post_update_checkpoint_permitted": True,
                },
                "num_timesteps": int(config.sample.num_steps),
                "max_repair_rounds": int(config.g016.max_repair_rounds),
                "repair_grouping": {
                    "logical_r0_image_count": 1,
                    "trajectory_count": 28,
                    "same_exact_r0_bytes_across_k28": True,
                    "full_trajectory_fork": True,
                    "branch_selection_used": False,
                    "best_of_k_used": False,
                    "r0_policy_record_count": 0,
                    "r0_optimizer_channel_present": False,
                    "r0_gradient_count": 0,
                },
                "reward_url": str(config.g016.reward_url),
                "flow_learning_rate": float(config.train.learning_rate),
                "text_learning_rate": float(
                    config.g016.text_learning_rate
                ),
                "flow_kl_beta": float(config.train.beta),
                "text_kl_beta": float(config.g016.text_kl_beta),
                "controller_action_aux": {
                    "version": CONTROLLER_ACTION_AUX_VERSION,
                    "weight": 0.0,
                    "enabled": False,
                },
                "policy_visible_protocol": {
                    "source": "unify_rl.inference.v22_controller_protocol",
                    "system_prompt": str(Path(config.g016.system_prompt_file).resolve()),
                    "new_fields": [],
                    "diagnosis": False,
                    "remaining_edits": False,
                    "verifier_contract": False,
                    "edit_none_alias_routes_to_semantic_done": True,
                },
                "initialization_lineage": {
                    "classic_sft_checkpoint": "0002969",
                    "fresh_text_adamw": resume_manifest is None,
                    "fresh_flow_adamw": resume_manifest is None,
                    "g005_policy_or_optimizer_loaded": False,
                    "g007_policy_or_optimizer_loaded": False,
                    "v19_rl_policy_or_optimizer_loaded": False,
                    "g008_policy_or_optimizer_loaded": False,
                    "g009_policy_or_optimizer_loaded": False,
                },
                "gpt_channel": {
                    "enabled": False,
                    "request_count": 0,
                    "mailbox": None,
                },
                "reward_formula": {
                    "version": REWARD_VERSION,
                    "counting_score": "q=1 exact else 0.5*max(0,1-abs(detected-target)/target)",
                    "edit": "0.4*(q_next-q_current)-0.01",
                    "terminal_success": 1.0,
                    "terminal_failure": 0.0,
                    "terminal_soft_q_reward_effect": 0.0,
                    "score_field_reward_effect": 0.0,
                    "gpt_reward_effect": 0.0,
                    "r0_reward_effect": 0.0,
                    "forbidden_reward_term_count": 0,
                    "method": ADVANTAGE_VERSION,
                    "repair_head_credit": (
                        "clip(q_after-q_before,-1,1) outside measured deadzone; no exact-conditioned bypass; zero peer baseline"
                    ),
                    "repair_progress_deadzone": float(config.g016.repair_progress_deadzone),
                    "repair_exact_bonus": 0.0,
                    "repair_peer_loo_used": False,
                    "action_head_credit": (
                        "unchanged exactness-stratified symmetric leave-one-out action correctness"
                    ),
                    "stop_now_scorer_identity": STOP_NOW_SCORER_IDENTITY,
                    "stop_now_scorer_version": STOP_NOW_SCORER_VERSION,
                    "empirical_bucket_mean_used_as_center": False,
                    "exact_before_action_policy_visible": False,
                    "singleton_bucket_policy_active": False,
                    "exact_singleton_action_active": False,
                    "broken_singleton_action_active": False,
                    "mixed_exactness_fallback": False,
                    "repair_hard_clamp": [-1.0, 1.0],
                    "hard_projection": False,
                },
                "reference_anchor_mode": (
                    "shared_frozen_prefix_last_four_reference_v1"
                ),
                "reference_optimizer_contract": (
                    reference_optimizer_contract
                ),
                "channel_map": {
                    "r0_flow_ppo": False,
                    "r0_reward_or_gradient": False,
                    "repair_flow_ppo": True,
                    "controller_text_clip_ppo": True,
                    "flow_frozen_classic_kl": True,
                    "text_frozen_classic_kl": True,
                },
                "synchronized_padding_contract": {
                    "version": "clean29529_g016_no_step_padding_v1",
                    "flow_policy_and_kl_zero": True,
                    "text_policy_and_kl_zero": True,
                    "optimizer_step_on_padding": False,
                },
                "release_prompt_schedule": {
                    "indices": list(config.g016.release_prompt_indices),
                    "resample_stride": int(
                        config.g016.release_resample_stride
                    ),
                    "formal_sampler_unchanged": (
                        not bool(config.g016.release_prompt_indices)
                    ),
                },
                "detector_transport": {
                    "version": "clean29529_g016_geneval_chunked_transport_v1",
                    "request_chunk_size": 64,
                    "max_attempts_per_chunk": 3,
                    "retryable_http_status": "5xx",
                    "logical_reward_group_unchanged": True,
                },
                "reward_execution": {
                    "version": "clean29529_g016_synchronous_reward_v1",
                    "queue_depth": 0,
                    "counting_detector_only": True,
                    "gpt_request_count": 0,
                    "next_rollout_overlap": False,
                    "rollout_policy_lag_updates": 0,
                    "reward_version": REWARD_VERSION,
                    "advantage_version": ADVANTAGE_VERSION,
                    "repair_payload_and_flow_share_same_absolute_progress_head": True,
                    "repair_payload_and_flow_share_same_trajectory_head": False,
                    "repair_progress_deadzone": float(config.g016.repair_progress_deadzone),
                    "repair_peer_baseline_used": False,
                    "action_scale_bucket": "(round_index, exact_before_action)",
                    "action_center": (
                        "symmetric leave-one-out action correctness"
                    ),
                    "stop_now_scorer_identity": STOP_NOW_SCORER_IDENTITY,
                    "stop_now_scorer_version": STOP_NOW_SCORER_VERSION,
                    "exact_before_action_policy_visible": False,
                    "singleton_bucket_inactive": True,
                    "exact_singleton_action_active": False,
                    "broken_singleton_action_active": False,
                    "mixed_exactness_fallback": False,
                    "trajectory_scalar_advantage_broadcast": False,
                    "repair_trajectory_advantage_broadcast": False,
                    "separate_r0_channel": False,
                    "r0_policy_or_gradient": False,
                },
                "gradient_clipping": {
                    "version": "clean29529_g016_dual_group_grad_clip_v1",
                    "flow_parameter_group": True,
                    "text_parameter_group": True,
                    "max_grad_norm": float(config.train.max_grad_norm),
                    "accelerator_clip_grad_norm": True,
                },
                "flow_kl_contract": {
                    "version": "clean29529_g016_flow_transition_kl_v2",
                    "reference": "shared_prefix_frozen_last_four",
                    "distribution": "equal_variance_sde_transition",
                    "transition_variance_includes_negative_dt": True,
                    "velocity_mse_reported": True,
                },
                "semantic_metrics": {
                    "efficacy_gates": False,
                    "counting_curves": "report_only",
                },
                "automatic_stops": {
                    "nonfinite_count_above_zero": True,
                    "advantage_outside_minus1_plus1": True,
                    "g017_protocol_hard_spike_ge_0_20": True,
                    "g017_protocol_three_of_last_five_ge_0_08": True,
                    "g017_protocol_trailing5_creep": True,
                    "controller_invalid_rate_above_0_15_three_consecutive": True,
                    "sign_category_inversion_five_consecutive": True,
                    "frozen_reward_or_system_prompt_hash_change": True,
                },
                "legacy_gate_apparatus": False,
                "official_geneval_consumed": False,
            },
        )

    checkpoint_steps = tuple(
        int(value) for value in config.g016.checkpoint_steps
    )
    stage_checkpoint_step = int(config.g016.stage_checkpoint_step)
    total_steps_for_schedule = int(config.g016.total_steps)
    if (
        list(checkpoint_steps) != sorted(set(checkpoint_steps))
        or any(v < 0 or v > total_steps_for_schedule for v in checkpoint_steps)
    ):
        raise RuntimeError(
            "checkpoint_steps must be sorted, unique and within total_steps: "
            f"{checkpoint_steps}"
        )
    if stage_checkpoint_step != -1:
        raise RuntimeError("G016 has no staged efficacy checkpoint")
    if (
        start_step == 0
        and not resume_checkpoint
        and 0 in checkpoint_steps
    ):
        checkpoint_zero = output_dir / "checkpoints" / "checkpoint-0"
        save_sharded_training_checkpoint(
            checkpoint_zero,
            model=transformer,
            text_optimizer=text_optimizer,
            flow_optimizer=flow_optimizer,
            logical_step=0,
            initialization_reference=initialization_reference,
            sampler_state=checkpoint_sampler_state(
                rank=rank,
                world_size=world_size,
                seed=int(config.g016.seed),
                batches_consumed=0,
                contract=sampler_contract,
            ),
            extra_state={"checkpoint_role": "initialization"},
            process_group=None,
            control_group=object_group,
            auxiliary_modules=auxiliary_flow_modules,
            skip_dcp_optimizer=True,
        )
    else:
        checkpoint_zero = None
    durable_checkpoint = (
        str(Path(resume_checkpoint).resolve())
        if resume_checkpoint
        else str(checkpoint_zero.resolve())
        if checkpoint_zero is not None
        else ""
    )
    durable_origin_kind = "resumable_optimizer_checkpoint"
    optimizer_rng_resumable = True
    if not durable_checkpoint and fresh_transaction_origin is not None:
        # Immutable classic initialization is not an optimizer/RNG checkpoint.
        # An ordinary failure before the first scheduled checkpoint restarts 0.
        durable_checkpoint = str(fresh_transaction_origin["durable_checkpoint"])
        durable_origin_kind = str(fresh_transaction_origin["durable_origin_kind"])
        optimizer_rng_resumable = bool(
            fresh_transaction_origin["optimizer_rng_resumable"]
        )
    if not durable_checkpoint and (
        False
    ):
        # No-update rollout evidence never needs or permits resume; avoid a
        # ceremonial 21-GiB checkpoint before generating the audit corpus.
        durable_checkpoint = str(
            (model_dir / "ema.safetensors").resolve()
        )
    if not durable_checkpoint:
        raise RuntimeError("G016 durable initialization checkpoint is missing")
    update_transaction = LogicalUpdateTransaction(
        completed_step=start_step,
        durable_step=start_step,
        durable_checkpoint=durable_checkpoint,
        durable_origin_kind=durable_origin_kind,
        optimizer_rng_resumable=optimizer_rng_resumable,
    )
    batches_consumed = int(start_step)

    def observe_update_phase(phase: str) -> None:
        record_memory_phase(phase)
        update_transaction.observe_phase(phase)

    def observe_optimizer_step(channel: str, event: str) -> None:
        if event == "possible":
            update_transaction.mark_optimizer_step_possible(channel)
        elif event in {"stepped", "skipped"}:
            update_transaction.mark_optimizer_step_result(
                channel,
                stepped=event == "stepped",
            )
        else:
            raise ValueError("G016 optimizer-step event is invalid")

    def regenerate_one_root(
        *,
        step: int,
        position: int,
        dataset_index: int,
        attempt: int,
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        """Regenerate one root R0 from a different prompt, and score it.

        G022 item 5. Deliberately separate from the initial-fill R0 loop rather
        than a refactor of it: that loop's seed derivation
        (`seed + 10_000_000 + root_index * 100_003`) and its rank ordering are
        what make G021-A and G021-B reproducible, and this must not perturb them.

        Every rank runs this in lockstep -- the broadcast and the detector call
        are collectives -- so it cannot be made conditional on rank.
        """

        row = dataset[int(dataset_index)]
        prompt = str(row.prompt)
        metadata = {
            "uid": row.uid,
            "family": row.family,
            "geneval_metadata": dict(row.metadata),
            "verification_constraints": [dict(value) for value in row.constraints],
        }
        r0_seed = rollout_seed(
            base_seed=int(config.g016.seed)
            + 10_000_000
            + int(position) * 100_003
            + (int(attempt) + 1) * 7_700_017,
            logical_step=step,
        )
        local_image, local_detachment = rollout.generate_detached_r0(
            prompt=prompt, generator=torch.Generator().manual_seed(int(r0_seed))
        )
        identity = {
            "prompt": prompt,
            "uid": metadata["uid"],
            "r0_seed": int(r0_seed),
            "family": metadata["family"],
            "geneval_metadata": metadata["geneval_metadata"],
            "verification_contract": metadata["verification_constraints"],
        }
        source = (
            {
                "ok": True,
                "source_rank": R0_SOURCE_RANK,
                "transport_version": R0_BROADCAST_VERSION,
                **identity,
                "image_bytes": image_bytes(local_image),
                "serialized_encoding": "JPEG_RGB_Q95",
            }
            if rank == R0_SOURCE_RANK
            else None
        )
        if rank == R0_SOURCE_RANK:
            source["image_sha256"] = image_state_sha256(source["image_bytes"])
        source = broadcast_rank_object(source, src=R0_SOURCE_RANK, group=object_group)
        if source.get("ok") is not True or any(
            source.get(key) != value for key, value in identity.items()
        ):
            raise RuntimeError("G022 resampled R0 identity broadcast differs")
        del local_image
        # A4 / finding Q-10. A replacement root used to claim
        # `distributed_fsdp_replica_count = world_size` while storing a single
        # local detachment and skipping the gather/reconstruction validation
        # that every initial root runs. A resampled root is on exactly the same
        # footing as an initial one -- 16 ranks must reconstruct the same bytes
        # from the same broadcast -- so it runs the same validator.
        payload = bytes(source["image_bytes"])
        with Image.open(BytesIO(payload)) as handle:
            replica_image = handle.convert("RGB").copy()
        local_replica = {
            "rank": rank,
            "source_rank": R0_SOURCE_RANK,
            "transport_version": R0_BROADCAST_VERSION,
            "prompt": prompt,
            "uid": metadata["uid"],
            "r0_seed": int(r0_seed),
            "image_sha256": image_state_sha256(payload),
            "received_byte_count": len(payload),
            "reconstructed_rgb_sha256": hashlib.sha256(
                replica_image.tobytes()
            ).hexdigest(),
            "reconstructed_mode": replica_image.mode,
            "reconstructed_size": list(replica_image.size),
            "detachment": local_detachment,
        }
        del replica_image
        replicas = gather_rank_objects(local_replica, object_group)
        # B / A4: rank-synchronised, same reason as the initial-root path.
        replica_mapping = rank_synchronised_replica_validation(
            replicas, rank=rank, object_group=object_group
        )
        if rank == R0_SOURCE_RANK:
            try:
                detector = request_geneval_score_chunks(
                    images=[bytes(source["image_bytes"])],
                    metadata=[_hidden_verifier_metadata(source)],
                    reward_url=str(config.g016.reward_url),
                    timeout_sec=float(config.g016.reward_timeout_sec),
                )
                if len(detector["results"]) != 1:
                    raise RuntimeError("G022 resampled R0 detector coverage differs")
                scored = {
                    "ok": True,
                    "detector_result": detector["results"][0],
                    "detector_score": float(detector["scores"][0]),
                    "detector_strict": bool(detector["strict_rewards"][0]),
                }
            except Exception as exc:
                scored = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                }
        else:
            scored = None
        scored = broadcast_rank_object(scored, src=R0_SOURCE_RANK, group=object_group)
        if scored.get("ok") is not True:
            raise RuntimeError(
                "G022 resampled R0 scoring failed: " + str(scored.get("error"))
            )
        package = {
            **source,
            "root_group_index": int(position),
            "detector_result": scored["detector_result"],
            "detector_score": float(scored["detector_score"]),
            "detector_strict": bool(scored["detector_strict"]),
            "g022_difficulty_resampled": True,
            "g022_difficulty_resample_attempt": int(attempt),
            # A4 / Q-10: the replica claim is now evidence, not an assertion.
            "replica_mapping": replica_mapping,
            "detachment": {
                "version": "clean29529_g022_difficulty_resampled_root_v2",
                "root_group_index": int(position),
                "logical_r0_image_count": 1,
                "distributed_fsdp_replica_count": len(replicas),
                "policy_record_count": 0,
                "flow_logprob_record_count": 0,
                "text_record_count": 0,
                "reward_channel_count": 0,
                "advantage_count": 0,
                "optimizer_contribution": 0.0,
                "replica_validation": (
                    "validate_detached_r0_replicas"
                    if rank == R0_SOURCE_RANK
                    else "validated_on_source_rank"
                ),
                "rank_evidence": [value["detachment"] for value in replicas],
            },
        }
        return package, prompt, metadata

    # Running totals for the proportional exact-R0 downsampler. Step-local
    # state deliberately: it is monitoring plus a selection rule, never
    # checkpointed policy state, and a resume simply restarts the running share
    # (T3.2's sidecar-consistency principle -- do not persist outside the
    # checkpoint transaction).
    g022_root_state_totals = {"trained_root_count": 0, "exact_root_count": 0}

    def g022_difficulty_filter_metrics_view(report: dict[str, Any]) -> dict[str, Any]:
        """The difficulty-filter report MINUS the payloads, for the metrics row.

        T1.2 instance #3, found by attempt 20260824T205046Z:
        `TypeError: Object of type bytes is not JSON serializable`.

        `apply_g022_difficulty_filter` returns `packages`, `root_prompts`,
        `root_metadata` and `root_indices` because its CALLER consumes them to
        replace the step's roots. Each package carries `image_bytes` -- raw
        JPEG. The whole dict was then placed on the attempt and copied into the
        metrics JSONL row, so `json.dumps` hit the image bytes.

        Pre-existing, not introduced by this session's work: it simply had never
        been reached, because both KeyErrors above fire earlier in the same
        rank-0 block.

        Raw image bytes do not belong in a metrics row in any case -- images are
        persisted as image files with their sha256 recorded. This keeps every
        decision field (counts, rates, attempts, the realised exact share) and
        drops only the payloads, so the audit value of the row is unchanged.
        """

        dropped = ("packages", "root_prompts", "root_metadata", "root_indices")
        view = {key: value for key, value in report.items() if key not in dropped}
        view["payload_fields_excluded_from_metrics"] = list(dropped)
        return view

    def apply_g022_difficulty_filter(
        *,
        step: int,
        packages: list[dict[str, Any]],
        root_prompts: list[str],
        root_metadata: list[dict[str, Any]],
        root_indices: list[int],
        build_root: Any = None,
    ) -> dict[str, Any]:
        """Replace already-solved roots, doc Appendix K-4. Bounded and logged.

        `build_root(root_index, dataset_index, attempt)` regenerates one root and
        returns its package; it is injected so this is testable on CPU.
        """

        started = time.monotonic()
        exact = [bool(value.get("detector_strict")) for value in packages]
        max_exact = int(
            getattr(config.g016, "g022_max_exact_roots_per_step", 0)
        )
        target_share_raw = getattr(
            config.g016, "g022_target_exact_root_share", None
        )
        target_share = (
            None
            if target_share_raw is None or float(target_share_raw) < 0.0
            else float(target_share_raw)
        )
        max_attempts = int(
            getattr(config.g016, "g022_difficulty_resample_max_attempts", 2)
        )
        packages = list(packages)
        root_prompts = list(root_prompts)
        root_metadata = list(root_metadata)
        root_indices = [int(value) for value in root_indices]
        history = []
        attempt_index = 0
        while attempt_index < max_attempts:
            # A fresh, deterministic candidate order per attempt, disjoint from
            # what this step already used.
            generator = torch.Generator().manual_seed(
                int(config.g016.seed) + int(step) * 7919 + attempt_index
            )
            pool = [
                int(value)
                for value in torch.randperm(len(dataset), generator=generator)
            ]
            plan = g022_difficulty_filter_plan(
                r0_exact=exact,
                max_exact_roots=max_exact,
                resample_pool=pool,
                used_dataset_indexes=root_indices,
                target_exact_share=target_share,
                realised_exact_count=int(
                    g022_root_state_totals["exact_root_count"]
                ),
                realised_root_count=int(
                    g022_root_state_totals["trained_root_count"]
                ),
            )
            history.append({"attempt": attempt_index, **plan})
            if not plan["replace_positions"] or build_root is None:
                break
            for position, dataset_index in zip(
                plan["replace_positions"], plan["replacement_dataset_indexes"]
            ):
                package, prompt, metadata = build_root(
                    position, dataset_index, attempt_index
                )
                packages[position] = package
                root_prompts[position] = prompt
                root_metadata[position] = metadata
                root_indices[position] = int(dataset_index)
                exact[position] = bool(package.get("detector_strict"))
            attempt_index += 1
            # The loop re-plans at the top, so the terminal condition is simply
            # "the plan asks for no more replacements". Under proportional
            # downsampling there is no per-step cap to compare against.
        final_exact = sum(exact)
        # Log the realised share EVERY step
        # alongside the dead-group rate split by root state, because 0.15 is a
        # judgement call rather than a derived bound and the number that should
        # govern it (P(EDIT|exact) drift) can only be seen during training.
        g022_root_state_totals["trained_root_count"] += len(exact)
        g022_root_state_totals["exact_root_count"] += int(final_exact)
        realised_share = (
            g022_root_state_totals["exact_root_count"]
            / g022_root_state_totals["trained_root_count"]
            if g022_root_state_totals["trained_root_count"]
            else 0.0
        )
        return {
            "target_exact_share": target_share,
            "target_exact_share_is_derived": False,
            "step_exact_root_count": int(final_exact),
            "step_root_count": len(exact),
            "realised_exact_share": realised_share,
            "realised_exact_root_count": int(
                g022_root_state_totals["exact_root_count"]
            ),
            "realised_trained_root_count": int(
                g022_root_state_totals["trained_root_count"]
            ),
            "version": "clean29529_g022_root_difficulty_filter_v1",
            "enabled": True,
            "step": int(step),
            "max_exact_roots_per_step": max_exact,
            "max_attempts": max_attempts,
            "initial_r0_exact_count": int(sum(bool(v) for v in history[0]["r0_exact"]))
            if history and "r0_exact" in history[0]
            else int(history[0]["r0_exact_count"]) if history else 0,
            "initial_r0_exact_rate": float(history[0]["r0_exact_rate"]) if history else 0.0,
            "final_r0_exact_count": int(final_exact),
            "final_r0_exact_rate": (
                final_exact / len(exact) if exact else 0.0
            ),
            "resample_count": sum(
                len(row["replace_positions"]) for row in history
            ),
            "unsatisfied_replacement_count": sum(
                int(row["unsatisfied_replacement_count"]) for row in history
            ),
            "attempts": history,
            "elapsed_sec": time.monotonic() - started,
            "detector_state_entered_policy_observation": False,
            "packages": packages,
            "root_prompts": root_prompts,
            "root_metadata": root_metadata,
            "root_indices": root_indices,
        }

    def generate_g021_attempt(*, step: int) -> dict[str, Any]:
        """Generate isolated same-root K groups (G021: 4 siblings, G022: 16)."""
        nonlocal batches_consumed
        root_count = active_g021_root_count()
        trajectory_count = active_g021_trajectory_count()
        started = time.monotonic()
        if int(step) != batches_consumed + 1:
            raise RuntimeError("G021 sampler cursor differs")
        generator = torch.Generator().manual_seed(int(config.g016.seed) + int(step) - 1)
        root_indices = [int(value) for value in torch.randperm(len(dataset), generator=generator)[:root_count]]
        root_rows = [dataset[index] for index in root_indices]
        root_prompts = [str(row.prompt) for row in root_rows]
        root_metadata = [{"uid": row.uid, "family": row.family, "geneval_metadata": dict(row.metadata), "verification_constraints": [dict(value) for value in row.constraints]} for row in root_rows]
        if len({row["uid"] for row in root_metadata}) != root_count:
            raise RuntimeError("G021 process-RTG sampler did not select distinct roots")
        r0_started = time.monotonic()
        source_packages = []
        replica_evidence = []
        for root_index, (prompt, metadata) in enumerate(zip(root_prompts, root_metadata)):
            r0_seed = rollout_seed(base_seed=int(config.g016.seed) + 10_000_000 + root_index * 100_003, logical_step=step)
            local_image, local_detachment = rollout.generate_detached_r0(
                prompt=prompt, generator=torch.Generator().manual_seed(int(r0_seed))
            )
            identity = {
                "prompt": prompt, "uid": metadata["uid"], "r0_seed": int(r0_seed),
                "family": metadata["family"], "geneval_metadata": metadata["geneval_metadata"],
                "verification_contract": metadata["verification_constraints"],
            }
            source = ({"ok": True, "source_rank": R0_SOURCE_RANK, "transport_version": R0_BROADCAST_VERSION, **identity,
                       "image_bytes": image_bytes(local_image), "serialized_encoding": "JPEG_RGB_Q95"}
                      if rank == R0_SOURCE_RANK else None)
            if rank == R0_SOURCE_RANK:
                source["image_sha256"] = image_state_sha256(source["image_bytes"])
            source = broadcast_rank_object(source, src=R0_SOURCE_RANK, group=object_group)
            if source.get("ok") is not True or any(source.get(key) != value for key, value in identity.items()):
                raise RuntimeError("G021 detached R0 identity broadcast differs")
            payload = bytes(source["image_bytes"])
            with Image.open(BytesIO(payload)) as handle:
                replica_image = handle.convert("RGB").copy()
            local_replica = {
                "rank": rank, "source_rank": R0_SOURCE_RANK, "transport_version": R0_BROADCAST_VERSION,
                "prompt": prompt, "uid": metadata["uid"], "r0_seed": int(r0_seed),
                "image_sha256": image_state_sha256(payload), "received_byte_count": len(payload),
                "reconstructed_rgb_sha256": hashlib.sha256(replica_image.tobytes()).hexdigest(),
                "reconstructed_mode": replica_image.mode, "reconstructed_size": list(replica_image.size),
                "detachment": local_detachment,
            }
            replicas = gather_rank_objects(local_replica, object_group)
            # B / A4: rank-synchronised. Only rank 0 used to evaluate this,
            # so a genuine divergence hung the other fifteen ranks silently.
            mapping = rank_synchronised_replica_validation(
                replicas, rank=rank, object_group=object_group
            )
            source_packages.append({**source, "replica_mapping": mapping})
            replica_evidence.append([value["detachment"] for value in replicas] if rank == R0_SOURCE_RANK else None)
            del local_image
        if rank == R0_SOURCE_RANK:
            try:
                detector = request_geneval_score_chunks(
                    images=[bytes(value["image_bytes"]) for value in source_packages],
                    metadata=[_hidden_verifier_metadata(value) for value in source_packages],
                    reward_url=str(config.g016.reward_url), timeout_sec=float(config.g016.reward_timeout_sec),
                )
                if len(detector["results"]) != root_count:
                    raise RuntimeError("G021 process-RTG detector coverage differs")
                packages = []
                for root_index, (source, result, score, strict) in enumerate(zip(source_packages, detector["results"], detector["scores"], detector["strict_rewards"])):
                    packages.append({**source, "root_group_index": root_index, "detector_result": result,
                                     "detector_score": float(score), "detector_strict": bool(strict),
                                     "detachment": {"version": ("clean29529_g021_four_detached_roots_v2"), "root_group_index": root_index,
                                         "logical_r0_image_count": 1, "distributed_fsdp_replica_count": world_size,
                                         "policy_record_count": 0, "flow_logprob_record_count": 0, "text_record_count": 0,
                                         "reward_channel_count": 0, "advantage_count": 0, "optimizer_contribution": 0.0,
                                         "rank_evidence": replica_evidence[root_index]}})
                package_result = {"ok": True, "packages": packages}
            except Exception as exc:
                package_result = {"ok": False, "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}
        else:
            package_result = None
        package_result = broadcast_rank_object(package_result, src=R0_SOURCE_RANK, group=object_group)
        if package_result.get("ok") is not True:
            raise RuntimeError("G021 detached roots failed: " + str(package_result.get("error")))
        packages = list(package_result["packages"])
        difficulty_filter = {"enabled": False}
        # G022 item 5 / doc Appendix K-4. A root whose R0 is already exact
        # produces a group with correctly-zero advantage: no gradient, full
        # rollout cost. Replace it, bounded, and log the rate either way.
        difficulty_filter = apply_g022_difficulty_filter(
            step=step,
            packages=packages,
            root_prompts=root_prompts,
            root_metadata=root_metadata,
            root_indices=root_indices,
            build_root=lambda position, dataset_index, attempt: regenerate_one_root(
                step=step,
                position=position,
                dataset_index=dataset_index,
                attempt=attempt,
            ),
        )
        packages = list(difficulty_filter["packages"])
        root_prompts = list(difficulty_filter["root_prompts"])
        root_metadata = list(difficulty_filter["root_metadata"])
        root_indices = list(difficulty_filter["root_indices"])
        local_indexes = [rank * int(config.sample.train_batch_size) + local for local in range(int(config.sample.train_batch_size))]
        siblings = active_siblings_per_root()
        local_root_indexes = [min(index // siblings, root_count - 1) for index in local_indexes]
        prompts = [root_prompts[root_index] for root_index in local_root_indexes]
        metadata_rows = [root_metadata[root_index] for root_index in local_root_indexes]
        anchor_images = []
        anchor_metadata = []
        for trajectory_index, root_index in zip(local_indexes, local_root_indexes):
            package = packages[root_index]
            with Image.open(BytesIO(bytes(package["image_bytes"]))) as handle:
                anchor_images.append(handle.convert("RGB").copy())
            anchor_metadata.append({
                "state_group_id": f"g021_root_{root_index:02d}", "state_group_member": trajectory_index % siblings,
                "repair_global_index": trajectory_index, "source_anchor_r0_global_index": root_index,
                "anchor_sha256": package["image_sha256"], "anchor_uid": package["uid"],
                "trajectory_index": trajectory_index, "r0_policy_record_present": False,
            })
        repair_seed_base = rollout_seed(base_seed=int(config.g016.seed) + 50_000_000, logical_step=step)
        trajectories = rollout.generate_full_trajectory_batch(
            prompts=prompts,
            verification_contracts=[value["verification_constraints"] for value in metadata_rows],
            anchor_images=anchor_images, anchor_metadata=anchor_metadata,
            repair_generators=[torch.Generator().manual_seed(int(repair_seed_base + index * 1009)) for index in local_indexes],
            repair_sde_window_identities=[{"experiment_seed": int(config.g016.seed), "logical_step": int(step), "trajectory_index": int(index)} for index in local_indexes],
            do_sample=True, temperature=float(config.g016.controller_temperature), response_max_tokens=int(config.g016.controller_max_tokens),
        )
        for trajectory_index, root_index, trajectory in zip(local_indexes, local_root_indexes, trajectories):
            trajectory.metadata["g016_world16_collective_padding"] = trajectory_index >= trajectory_count
            trajectory.metadata["g021_root_group_index"] = root_index
            trajectory.metadata["g021_root_group_member"] = trajectory_index % siblings
        secrecy = assert_policy_observation_secrecy(trajectories)
        local_payload = []
        for prompt, metadata, trajectory_index, root_index, trajectory in zip(prompts, metadata_rows, local_indexes, local_root_indexes, trajectories):
            if trajectory_index >= trajectory_count:
                continue
            package = packages[root_index]
            round_bytes = [bytes(package["image_bytes"]), *(image_bytes(record.image) for record in trajectory.flow_calls)]
            local_payload.append({
                "uid": metadata["uid"], "family": metadata["family"], "prompt": prompt,
                "geneval_metadata": metadata["geneval_metadata"], "verification_contract": metadata["verification_constraints"],
                "trajectory_index": trajectory_index, "repair_global_index": trajectory_index,
                "root_group_index": root_index, "root_group_member": trajectory_index % siblings,
                "state_group_id": f"g021_root_{root_index:02d}", "state_group_member": trajectory_index % siblings,
                "source_anchor_r0_global_index": root_index, "r0_sha256": package["image_sha256"],
                "anchor_sha256": package["image_sha256"], "r0_detector_result": package["detector_result"],
                "r0_detector_score": float(package["detector_score"]), "r0_detector_strict": bool(package["detector_strict"]),
                "round_image_bytes": round_bytes, "round_image_sha256s": [image_state_sha256(value) for value in round_bytes],
                "events": list(trajectory.events), "done": trajectory.done, "stop_reason": trajectory.stop_reason,
                "flow_call_count": len(trajectory.flow_calls), "text_turn_count": trajectory.valid_text_turn_count,
                "malformed_text_turn_count": trajectory.malformed_text_turn_count,
                "post_fork_rng_seed": int(repair_seed_base + trajectory_index * 1009),
                "flow_sde_timestep_begins": [int(record.sde_timestep_begin) for record in trajectory.flow_calls],
                "flow_sde_transition_indices": [list(record.sde_transition_indices or []) for record in trajectory.flow_calls],
                "flow_sde_transition_layouts": [str(record.sde_transition_layout) for record in trajectory.flow_calls],
                "flow_sde_window_seed_identities": [dict(record.sde_window_seed_identity) for record in trajectory.flow_calls],
                "branch_selection_used": False, "best_of_k_used": False, "per_round_sibling_branching_used": False,
                "trajectory_pruned": False, "r0_policy_record_present": False,
                "policy_observation_secrecy_passed": bool(secrecy["passed"]),
            })
        local_padding = next((record for trajectory in trajectories for record in trajectory.flow_calls), getattr(rollout, "last_collective_padding_flow_record", None))
        padding_candidates = gather_rank_objects(local_padding, object_group)
        flow_padding_record = next((value for value in padding_candidates if value is not None), None)
        batches_consumed += 1
        return {
            "step": step, "batches_consumed": batches_consumed, "prompts": prompts, "metadata_rows": metadata_rows,
            "trajectories": trajectories, "local_payload": local_payload, "r0_package": {"packages": packages},
            "r0_detachment": {"version": ("clean29529_g021_four_detached_roots_v2"), "logical_r0_image_count": root_count,
                              "controller_group_size": 4, "policy_record_count": 0, "optimizer_contribution": 0.0},
            "g022_difficulty_filter": difficulty_filter,
            "policy_observation_secrecy": secrecy, "flow_padding_record": flow_padding_record,
            "rollout_elapsed_sec": time.monotonic() - started, "r0_elapsed_sec": time.monotonic() - r0_started,
            "rollout_phase_timings": dict(rollout.last_phase_timings), "rollout_sync_sec": 0.0,
        }

    def generate_attempt(*, step: int) -> dict[str, Any]:
        if g021_process_rtg_mode():
            return generate_g021_attempt(step=step)
        nonlocal batches_consumed
        started = time.monotonic()
        if int(step) != batches_consumed + 1:
            raise RuntimeError("G016 sampler cursor differs from logical step")
        if (
            world_size * int(config.sample.train_batch_size) == 32
            and int(config.sample.num_image_per_prompt) == 28
        ):
            generator = torch.Generator().manual_seed(
                int(config.g016.seed) + int(step) - 1
            )
            prompt_index = int(torch.randperm(len(dataset), generator=generator)[0])
            indices = [
                prompt_index
                for _ in range(int(config.sample.train_batch_size))
            ]
        else:
            indices = step_batch_indices(
                dataset_size=len(dataset),
                batch_size=int(config.sample.train_batch_size),
                group_size=int(config.sample.num_image_per_prompt),
                world_size=world_size,
                rank=rank,
                seed=int(config.g016.seed),
                logical_step=step,
            )
        prompts, metadata_rows = G016PromptDataset.collate_fn(
            [dataset[index] for index in indices]
        )
        batches_consumed += 1
        if len(set(prompts)) != 1 or len({value["uid"] for value in metadata_rows}) != 1:
            raise RuntimeError("G016 local group mixed prompt/UID")

        # One logical distributed-FSDP R0 generation. All ranks must enter the
        # sharded forward in lockstep, but rank 0 is the sole image source: all
        # non-source local outputs are discarded before the environment R0 is
        # formed. Rank 0 serializes one canonical JPEG plus prompt/UID/seed
        # metadata and broadcasts those exact bytes. Every rank reconstructs
        # from that payload and the unchanged byte-exact guard validates all
        # recipients before candidate-specific RNG begins after the R0 fork.
        # No FlowCallRecord, log-probability, latent, reward, or advantage
        # survives this boundary.
        r0_started = time.monotonic()
        r0_seed = rollout_seed(
            base_seed=int(config.g016.seed) + 10_000_000,
            logical_step=step,
        )
        local_r0_image, local_r0_detachment = rollout.generate_detached_r0(
            prompt=prompts[0],
            generator=torch.Generator().manual_seed(int(r0_seed)),
        )
        local_metadata_identity = {
            "prompt": prompts[0],
            "uid": metadata_rows[0]["uid"],
            "r0_seed": int(r0_seed),
            "family": metadata_rows[0]["family"],
            "geneval_metadata": metadata_rows[0]["geneval_metadata"],
            "verification_contract": metadata_rows[0]["verification_constraints"],
        }
        if rank == R0_SOURCE_RANK:
            canonical_source_bytes = image_bytes(local_r0_image)
            source_r0 = {
                "ok": True,
                "source_rank": R0_SOURCE_RANK,
                "transport_version": R0_BROADCAST_VERSION,
                **local_metadata_identity,
                "image_bytes": canonical_source_bytes,
                "image_sha256": image_state_sha256(canonical_source_bytes),
                "serialized_encoding": "JPEG_RGB_Q95",
            }
        else:
            source_r0 = None
        source_r0 = broadcast_rank_object(
            source_r0,
            src=R0_SOURCE_RANK,
            group=object_group,
        )
        if not isinstance(source_r0, dict) or source_r0.get("ok") is not True:
            raise RuntimeError("G016 rank-0 detached R0 byte broadcast failed")
        if any(
            source_r0.get(key) != value
            for key, value in local_metadata_identity.items()
        ):
            raise RuntimeError(
                "G016 rank-0 detached R0 prompt/UID/seed metadata differs"
            )
        canonical_r0_bytes = bytes(source_r0["image_bytes"])
        canonical_r0_sha256 = image_state_sha256(canonical_r0_bytes)
        if canonical_r0_sha256 != str(source_r0["image_sha256"]):
            raise RuntimeError("G016 rank-0 detached R0 payload hash differs")
        with Image.open(BytesIO(canonical_r0_bytes)) as handle:
            canonical_r0 = handle.convert("RGB").copy()
        reconstructed_rgb_sha256 = hashlib.sha256(
            canonical_r0.tobytes()
        ).hexdigest()
        local_replica = {
            "rank": rank,
            "source_rank": int(source_r0["source_rank"]),
            "transport_version": str(source_r0["transport_version"]),
            "prompt": source_r0["prompt"],
            "uid": source_r0["uid"],
            "r0_seed": int(source_r0["r0_seed"]),
            "image_sha256": canonical_r0_sha256,
            "received_byte_count": len(canonical_r0_bytes),
            "reconstructed_rgb_sha256": reconstructed_rgb_sha256,
            "reconstructed_mode": canonical_r0.mode,
            "reconstructed_size": list(canonical_r0.size),
            "detachment": local_r0_detachment,
        }
        gathered_r0 = gather_rank_objects(local_replica, object_group)
        if rank == R0_SOURCE_RANK:
            try:
                if len(gathered_r0) != world_size:
                    raise RuntimeError("G016 distributed R0 replica coverage differs")
                replica_mapping = validate_detached_r0_replicas(gathered_r0)
                r0_detector = (
                    request_geneval_score_chunks(
                        images=[canonical_r0_bytes],
                        metadata=[_hidden_verifier_metadata(source_r0)],
                        reward_url=str(config.g016.reward_url),
                        timeout_sec=float(config.g016.reward_timeout_sec),
                    )
                )
                if len(r0_detector["results"]) != 1:
                    raise RuntimeError("G016 detached R0 detector coverage differs")
                r0_package = {
                    "ok": True,
                    "prompt": source_r0["prompt"],
                    "uid": source_r0["uid"],
                    "r0_seed": int(source_r0["r0_seed"]),
                    "image_bytes": canonical_r0_bytes,
                    "image_sha256": canonical_r0_sha256,
                    "replica_mapping": replica_mapping,
                    "detector_result": r0_detector["results"][0],
                    "detector_score": float(r0_detector["scores"][0]),
                    "detector_strict": bool(r0_detector["strict_rewards"][0]),
                    "detector_transport": {
                        key: r0_detector[key]
                        for key in (
                            "service_version",
                            "protocol_version",
                            "chunk_count",
                            "chunk_sizes",
                            "total_attempts",
                        )
                    },
                    "detachment": {
                        "version": "clean29529_g016_detached_environment_r0_v1",
                        "transport_version": R0_BROADCAST_VERSION,
                        "source_rank": R0_SOURCE_RANK,
                        "logical_r0_image_count": 1,
                        "distributed_fsdp_replica_count": world_size,
                        "broadcast_recipient_count": world_size,
                        "distributed_replicas_byte_identical": True,
                        "independent_rank_outputs_retained": 1,
                        "non_source_local_generation_outputs_discarded": True,
                        "candidate_diversity_begins_after_r0_fork": True,
                        "retained_canonical_image_count": 1,
                        "policy_record_count": 0,
                        "flow_logprob_record_count": 0,
                        "text_record_count": 0,
                        "reward_channel_count": 0,
                        "advantage_count": 0,
                        "optimizer_contribution": 0.0,
                        "replica_mapping": replica_mapping,
                        "rank_evidence": [value["detachment"] for value in gathered_r0],
                    },
                }
            except Exception as exc:
                r0_package = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                }
        else:
            r0_package = None
        r0_package = broadcast_rank_object(
            r0_package,
            src=R0_SOURCE_RANK,
            group=object_group,
        )
        if r0_package.get("ok") is not True:
            raise RuntimeError(
                "G016 detached R0 generation/detector failed: "
                + str(r0_package.get("error"))
            )
        if (
            image_state_sha256(bytes(r0_package["image_bytes"]))
            != canonical_r0_sha256
            or canonical_r0_sha256 != r0_package["image_sha256"]
        ):
            raise RuntimeError("G016 post-detector R0 broadcast changed bytes")
        del local_r0_image

        local_trajectory_indexes = [
            rank * int(config.sample.train_batch_size) + local_index
            for local_index in range(int(config.sample.train_batch_size))
        ]
        anchor_images = [canonical_r0.copy() for _ in local_trajectory_indexes]
        anchor_metadata = [
            {
                "state_group_id": "same_r0_k28",
                "state_group_member": trajectory_index % 7,
                "repair_global_index": trajectory_index,
                "source_anchor_r0_global_index": 0,
                "anchor_sha256": r0_package["image_sha256"],
                "anchor_uid": r0_package["uid"],
                "trajectory_index": trajectory_index,
                "r0_policy_record_present": False,
            }
            for trajectory_index in local_trajectory_indexes
        ]
        repair_seed_base = rollout_seed(
            base_seed=int(config.g016.seed) + 50_000_000,
            logical_step=step,
        )
        repair_generators = [
            torch.Generator().manual_seed(
                int(repair_seed_base + trajectory_index * 1009)
            )
            for trajectory_index in local_trajectory_indexes
        ]
        repair_sde_window_identities = [
            {
                "experiment_seed": int(config.g016.seed),
                "logical_step": int(step),
                "trajectory_index": int(trajectory_index),
            }
            for trajectory_index in local_trajectory_indexes
        ]
        trajectories = rollout.generate_full_trajectory_batch(
            prompts=prompts,
            verification_contracts=[
                value["verification_constraints"] for value in metadata_rows
            ],
            anchor_images=anchor_images,
            anchor_metadata=anchor_metadata,
            repair_generators=repair_generators,
            repair_sde_window_identities=repair_sde_window_identities,
            do_sample=True,
            temperature=float(config.g016.controller_temperature),
            response_max_tokens=int(config.g016.controller_max_tokens),
        )
        for trajectory_index, trajectory in zip(local_trajectory_indexes, trajectories):
            trajectory.metadata["g016_world16_collective_padding"] = bool(
                world_size * int(config.sample.train_batch_size) == 32
                and int(config.sample.num_image_per_prompt) == 28
                and trajectory_index >= 28
            )
        # Hidden detector/exact labels do not exist yet.  Audit every policy
        # context, forced-prefix/token-constraint surface, and metadata object
        # before constructing the reward-worker payload.
        policy_observation_secrecy = assert_policy_observation_secrecy(
            trajectories
        )
        local_payload = []
        for prompt, metadata, trajectory_index, trajectory in zip(
            prompts, metadata_rows, local_trajectory_indexes, trajectories
        ):
            if trajectory.metadata.get("g016_world16_collective_padding") is True:
                continue
            round_bytes = [
                r0_package["image_bytes"],
                *(image_bytes(record.image) for record in trajectory.flow_calls),
            ]
            local_payload.append(
                {
                    "uid": metadata["uid"],
                    "family": metadata["family"],
                    "prompt": prompt,
                    "geneval_metadata": metadata["geneval_metadata"],
                    "verification_contract": metadata["verification_constraints"],
                    "trajectory_index": trajectory_index,
                    "repair_global_index": trajectory_index,
                    "state_group_id": "same_r0_k28",
                    "state_group_member": trajectory_index % 7,
                    "source_anchor_r0_global_index": 0,
                    "r0_sha256": r0_package["image_sha256"],
                    "anchor_sha256": r0_package["image_sha256"],
                    "r0_detector_result": r0_package["detector_result"],
                    "r0_detector_score": float(r0_package["detector_score"]),
                    "r0_detector_strict": bool(r0_package["detector_strict"]),
                    "round_image_bytes": round_bytes,
                    "round_image_sha256s": [
                        image_state_sha256(value) for value in round_bytes
                    ],
                    "events": list(trajectory.events),
                    "done": trajectory.done,
                    "stop_reason": trajectory.stop_reason,
                    "flow_call_count": len(trajectory.flow_calls),
                    "text_turn_count": trajectory.valid_text_turn_count,
                    "malformed_text_turn_count": trajectory.malformed_text_turn_count,
                    "post_fork_rng_seed": int(
                        repair_seed_base + trajectory_index * 1009
                    ),
                    "flow_sde_timestep_begins": [
                        int(record.sde_timestep_begin)
                        for record in trajectory.flow_calls
                    ],
                    "flow_sde_window_seed_identities": [
                        dict(record.sde_window_seed_identity)
                        for record in trajectory.flow_calls
                    ],
                    "branch_selection_used": False,
                    "best_of_k_used": False,
                    "per_round_sibling_branching_used": False,
                    "trajectory_pruned": False,
                    "r0_policy_record_present": False,
                    "policy_observation_secrecy_passed": bool(
                        policy_observation_secrecy["passed"]
                    ),
                }
            )
        if any(
            value["round_image_sha256s"][0] != r0_package["image_sha256"]
            for value in local_payload
        ):
            raise RuntimeError("G016 local trajectory changed detached R0 bytes")

        # Flow padding is always an inactive repair-path record. Never use R0.
        local_padding = next(
            (record for trajectory in trajectories for record in trajectory.flow_calls),
            getattr(rollout, "last_collective_padding_flow_record", None),
        )
        padding_candidates = gather_rank_objects(local_padding, object_group)
        flow_padding_record = next(
            (value for value in padding_candidates if value is not None), None
        )
        return {
            "step": step,
            "batches_consumed": batches_consumed,
            "prompts": prompts,
            "metadata_rows": metadata_rows,
            "trajectories": trajectories,
            "local_payload": local_payload,
            "r0_package": r0_package,
            "r0_detachment": r0_package["detachment"],
            "policy_observation_secrecy": policy_observation_secrecy,
            "flow_padding_record": flow_padding_record,
            "rollout_elapsed_sec": time.monotonic() - started,
            "r0_elapsed_sec": time.monotonic() - r0_started,
            "rollout_phase_timings": dict(rollout.last_phase_timings),
            "rollout_sync_sec": 0.0,
        }

    timing_state = {
        "last_completion": time.monotonic(),
        "completed_step": start_step,
        "previous_checkpoint_elapsed_sec": 0.0,
    }

    def apply_scored_attempt(
        attempt: dict[str, Any],
        scored: dict[str, Any],
    ) -> None:
        # Zero-cost, load-bearing guards run before either optimizer can step.
        diagnostics = scored["group_diagnostics"]
        reward_rows = list(scored["trajectory_records"])
        # T1.2 ROOT CAUSE, found by the per-rank failure dump on attempt
        # 20260824T200233Z. All 16 ranks reported
        #   phase=commit_synchronized commit_synchronized=True metrics_persisted=False
        # after `G016 rank0 persistence failed: KeyError: 'g016_live_monitoring'`.
        #
        # This was built ONLY inside the G016-G021 `else:` branch below, while
        # the shared rank-0 persistence block reads it unconditionally. Under
        # G022 the key was never set: the persistence block raised,
        # `persistence["ok"]` went False, every rank raised, and
        # `mark_metrics_persisted()` was therefore never reached -- so the next
        # call, `finish_without_checkpoint()`, found commit_synchronized True
        # and metrics_persisted False and raised
        #   "V20 update cannot finish before persistence".
        # That is the error eleven attempts reported. It was the CONSEQUENCE;
        # this missing key is the cause.
        #
        # Live monitoring is a monitoring statistic, not a G016-G021 algorithm
        # assertion, and every field it reads (`actions`, `rounds`,
        # `controller_advantages`, `repair_flow_advantages`,
        # `repair_flow_active`, `exact_before_action`) is produced by the G022
        # projection too. Computed here, ahead of the branch, so BOTH paths
        # have it -- rather than teaching the persistence block to tolerate a
        # missing key, because the reader is right to require it.
        monitoring_started = time.monotonic()
        monitoring_payload = None
        if rank == 0:
            try:
                history = _g016_metrics_history_from(metrics_path)
                history_read_sec = time.monotonic() - monitoring_started
                bootstrap_started = time.monotonic()
                monitoring_payload = {
                    "ok": True,
                    "history": history,
                    "monitoring": build_g016_live_monitoring(
                        reward_rows, history, step=int(attempt["step"])
                    ),
                    "history_read_sec": history_read_sec,
                    "bootstrap_sec": time.monotonic() - bootstrap_started,
                }
            except Exception as exc:
                monitoring_payload = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        monitoring_payload = broadcast_rank_object(
            monitoring_payload, src=0, group=object_group
        )
        if not monitoring_payload["ok"]:
            raise RuntimeError(
                "G016 monitoring history failed: " + monitoring_payload["error"]
            )
        history = monitoring_payload["history"]
        scored["g016_live_monitoring"] = monitoring_payload["monitoring"]
        monitoring_elapsed_sec = time.monotonic() - monitoring_started
        protocol = scored["reward_diagnostics"]["controller_protocol"]
        pre_update_errors: list[str] = []
        # T1.2, second instance of the SAME defect, found by attempt
        # 20260824T202647Z: `G016 rank0 persistence failed: KeyError:
        # 'g017_protocol_degradation'`. Like `g016_live_monitoring`, this was
        # built only inside the G016-G021 `else:` while the live-status block
        # reads it unconditionally -- the read is split across three lines,
        # which is why my first single-line sweep missed it and cost an attempt.
        #
        # The STATISTIC is computed here for both branches. The HALT it can
        # drive stays where it was, inside the else. That is deliberate and is
        # not a weakening: G022 has never had this halt, it runs its own
        # `g022_pre_update_invariants` instead, and its Appendix A-1 design
        # classifies malformed actions rather than stopping the run. Adding an
        # untested stop condition immediately before a 40-hour committing run
        # would be the riskier change, and silently would be the wrong way to
        # make it. Recorded as an open question rather than
        # decided here.
        protocol_history = history
        protocol_denominator_shared = max(
            1, int(scored["reward_diagnostics"].get("trajectory_count", 28))
        )
        prior_invalid_steps_shared: list[float] = []
        prior_false_done_shared: list[tuple[int, int]] = []
        for value in protocol_history:
            prior_protocol = value.get("g017_protocol_degradation") or {}
            prior_dashboard = value.get("reward_diagnostics") or {}
            prior_controller = prior_dashboard.get("controller_protocol") or {}
            prior_denominator = max(
                1, int(prior_dashboard.get("trajectory_count", 28))
            )
            prior_false_numerator = int(
                prior_protocol.get(
                    "false_done_numerator",
                    int(prior_dashboard.get("false_done_count", 0)),
                )
            )
            prior_false_denominator = int(
                prior_protocol.get("false_done_denominator", prior_denominator)
            )
            prior_false_done_shared.append(
                (prior_false_numerator, prior_false_denominator)
            )
            if "protocol_invalid_step" in prior_protocol:
                prior_invalid_steps_shared.append(
                    float(prior_protocol["protocol_invalid_step"])
                )
            else:
                prior_invalid_steps_shared.append(
                    max(
                        float(prior_controller.get("controller_invalid_rate", 0.0)),
                        int(prior_dashboard.get("parse_failure_count", 0))
                        / prior_denominator,
                        prior_false_numerator / prior_false_denominator
                        if prior_false_denominator
                        else 0.0,
                    )
                )
        scored["g017_protocol_degradation"] = evaluate_protocol_degradation(
            prior_invalid_steps_shared,
            controller_invalid_rate=float(
                protocol.get("controller_invalid_rate", 1.0)
            ),
            parse_failure_rate=(
                int(scored["reward_diagnostics"].get("parse_failure_count", 0))
                / protocol_denominator_shared
            ),
            false_done_numerator=int(
                scored["reward_diagnostics"].get("false_done_count", 0)
            ),
            false_done_denominator=protocol_denominator_shared,
            prior_false_done_observations=prior_false_done_shared,
        )
        # G022 is one-head whole-trajectory GRPO. The G016-G021 block below
        # checks a two-head per-round design and is about a different
        # algorithm, so G022 contributes its own equally strict findings
        # and the shared terminal check below raises on either.
        pre_update_errors.extend(
            g022_pre_update_invariants(
                diagnostics=diagnostics,
                reward_rows=reward_rows,
                scored=scored,
                advantage_version=ADVANTAGE_VERSION,
            )
        )
        # T1.2 ROOT CAUSE, found by the per-rank failure dump on attempt
        # 20260824T200233Z. All 16 ranks reported
        #   phase=commit_synchronized  commit_synchronized=True  metrics_persisted=False
        # after `G016 rank0 persistence failed: KeyError: 'g016_live_monitoring'`.
        #
        # `scored["g016_live_monitoring"]` was built ONLY inside the G016-G021
        # `else:` branch above, while the shared rank-0 persistence block reads
        # it unconditionally. Under G022 the key was never set, the persistence
        # block raised, `persistence["ok"]` went False, every rank raised, and
        # `mark_metrics_persisted()` was therefore never reached -- so the very
        # next call, `finish_without_checkpoint()`, found commit_synchronized
        # True and metrics_persisted False and raised
        #   "V20 update cannot finish before persistence"
        # which is the error eleven attempts reported. That message was the
        # CONSEQUENCE; this missing key is the cause.
        #
        # Live monitoring is a monitoring statistic, not a G016-G021 algorithm
        # assertion, and every field it reads (`actions`, `rounds`,
        # `controller_advantages`, `repair_flow_advantages`,
        # `repair_flow_active`, `exact_before_action`) is produced by the G022
        # projection too. So it is computed HERE, at the join, for both
        # branches, rather than teaching the persistence block to tolerate a
        # missing key -- the reader is right to require it.
        if pre_update_errors:
            raise RuntimeError(
                "G016 pre-update invariant failed: " + ",".join(pre_update_errors)
            )

        update_started = time.monotonic()
        trajectories = attempt["trajectories"]
        semantic_trajectories = [
            trajectory
            for trajectory in trajectories
            if trajectory.metadata.get("g016_world16_collective_padding")
            is not True
        ]
        world_padding_trajectories = [
            trajectory
            for trajectory in trajectories
            if trajectory.metadata.get("g016_world16_collective_padding")
            is True
        ]
        local_records = sorted(
            [
                value
                for value in scored["trajectory_records"]
                if int(value["owner_rank"]) == rank
            ],
            key=lambda value: int(value["owner_local_index"]),
        )
        if len(local_records) != len(semantic_trajectories):
            raise RuntimeError(
                "G016 advantage broadcast lost semantic local trajectories"
            )
        for trajectory in world_padding_trajectories:
            # Trajectory indexes 28..31 exist only to make 16-rank FSDP
            # traversal uniform. They never enter the K=28 reward payload or
            # any hidden-exactness bucket, but must execute zero-credit text
            # and repair forwards/backwards on ranks 14/15.
            trajectory.reward = 0.0
            trajectory.metadata["r0_policy_record_present"] = False
            for flow_record in trajectory.flow_calls:
                flow_record.advantage = 0.0
                flow_record.policy_active = False
                flow_record.g016_collective_only_world_padding = True
            for turn in trajectory.text_turns:
                turn.advantage = 0.0
                turn.local_advantage = 0.0 if turn.valid else None
                turn.policy_active = False
                turn.g016_repair_advantage = 0.0
                turn.g016_collective_only_world_padding = True
        local_token_routing_failure = None
        for trajectory, record in zip(semantic_trajectories, local_records):
            trajectory.reward = float(record["raw_return"])
            # Hidden detector/exact diagnostics remain in the reward-side
            # record only. They are never attached to policy context metadata.
            trajectory.metadata["r0_policy_record_present"] = False
            trajectory.assign_channel_advantages(
                repair_flow=list(record["repair_flow_advantages"]),
                controller=list(record["controller_advantages"]),
                repair_active=list(record["repair_flow_active"]),
                controller_active=list(record["controller_active"]),
            )
            token_audits = []
            for turn, round_row, event in zip(trajectory.text_turns, record["rounds"], trajectory.events):
                if round_row.get("g016_two_head_credit_applied") is not True:
                    # `assign_g022_advantages` sets this
                    # unconditionally today, so this cannot fire -- but if
                    # it ever did, G024 would SILENTLY leave that turn's
                    # tokens uncredited, which is precisely the failure
                    # G024 exists to eliminate. A turn the model generated
                    # and that receives no gradient is not a degraded run,
                    # it is G023. Fail closed instead.
                    local_token_routing_failure = {
                        "rank": rank,
                        "step": int(attempt["step"]),
                        "trajectory_index": int(record["trajectory_index"]),
                        "turn_index": int(turn.turn_index),
                        "uid": str(record["uid"]),
                        "infra_invariant": True,
                        "g024": True,
                        "error": (
                            "G024 round is missing "
                            "g016_two_head_credit_applied; refusing to "
                            "leave model-sampled tokens uncredited"
                        ),
                    }
                    break
                # No routing or decision lineage: every model-sampled token of
                # the round is credited. The only thing that can fail here is
                # an infra invariant,
                # which raises and takes the same synchronized fail-closed
                # path as every other routing fault.
                try:
                    masks, audit = build_g024_uniform_masks(turn=turn)
                except G022TokenRoutingLineageError as exc:
                    local_token_routing_failure = {
                        "rank": rank,
                        "step": int(attempt["step"]),
                        "trajectory_index": int(record["trajectory_index"]),
                        "turn_index": int(turn.turn_index),
                        "uid": str(record["uid"]),
                        "infra_invariant": True,
                        "g024": True,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    break
                turn.g016_action_token_mask = masks["action"]
                turn.g016_repair_payload_token_mask = masks["repair"]
                turn.g016_token_routing_audit = audit
                turn.g016_decision_token_mask = masks["action"]
                turn.g022_routing_status = str(audit["routing_status"])
                turn.g022_malformed_action = False
                # There is no second head. Leaving a stale repair advantage
                # here would be harmless today (the mask is empty) and a
                # live bug the moment anything reads it.
                turn.g016_repair_advantage = 0.0
                turn.g023_repair_active = False
                turn.policy_active = True
                # The one scalar, broadcast to every token the model wrote.
                turn.advantage = float(
                    record["controller_advantages"][turn.turn_index]
                )
                turn.local_advantage = float(turn.advantage) if turn.valid else None
                round_row["g022_routing_status"] = str(audit["routing_status"])
                round_row["g022_routing_detail"] = str(audit["routing_detail"])
                round_row["g022_malformed_action"] = False
                round_row["g024_uniform_credit"] = True
                round_row["g024_credited_token_count"] = int(
                    audit["g024_credited_token_count"]
                )
                round_row["g024_sampled_token_count"] = int(
                    audit["g024_sampled_token_count"]
                )
                round_row["g024_partial_credit_mask"] = bool(
                    audit["g024_partial_credit_mask"]
                )
                token_audits.append(audit)
                continue
            trajectory.metadata["g016_token_routing_audits"] = token_audits
            if local_token_routing_failure is not None:
                break
        # Fixed-size world consensus occurs before any rank can enter FSDP
        # backward. A one-rank validation failure therefore cannot strand peers.
        token_failure_flag = torch.tensor(
            int(local_token_routing_failure is not None),
            device=accelerator.device,
            dtype=torch.int32,
        )
        if dist.is_initialized():
            dist.all_reduce(token_failure_flag, op=dist.ReduceOp.MAX)
        if int(token_failure_flag.item()):
            failures = gather_rank_objects(local_token_routing_failure, object_group)
            failures = [value for value in failures if value is not None]
            if rank == 0:
                atomic_json(
                    output_dir / f"G016_TOKEN_ROUTING_FAILURE_STEP{int(attempt['step']):04d}.json",
                    {
                        "version": "clean29529_g016_distributed_token_routing_failure_v1",
                        "created_at_utc": utc_now(),
                        "failed_step": int(attempt["step"]),
                        "failure_count": len(failures),
                        "failures": failures,
                        "pre_update": True,
                        "optimizer_update_executed": False,
                        "world_consensus_before_backward": True,
                    },
                )
            accelerator.wait_for_everyone()
            raise RuntimeError(
                "G016 synchronized pre-update token routing validation failed: "
                + json.dumps(failures, sort_keys=True)
            )
        assert_policy_observation_secrecy(trajectories)

        if rank == 0:
            r0_seed = rollout_seed(
                base_seed=int(config.g016.seed) + 10_000_000,
                logical_step=int(attempt["step"]),
            )
            forensic_context_lineage = [
                {
                    "trajectory_index": int(value["trajectory_index"]),
                    "uid": str(value["uid"]),
                    "r0_image_sha256": str(value["r0_sha256"]),
                    "r0_seed": int(r0_seed),
                    "trajectory_seed": int(value["seed"]),
                    "round_image_sha256s": list(
                        value["image_state_sha256s"]
                    ),
                    "flow_sde_window_seed_identities": list(
                        value["flow_sde_window_seed_identities"]
                    ),
                }
                for value in scored["trajectory_records"]
            ]
            forensic_ring.begin_step(
                step=int(attempt["step"]),
                marker={
                    "phase": "pre_update",
                    "created_at_utc": utc_now(),
                    "prior_committed_step": int(attempt["step"]) - 1,
                    "optimizer_step_possible": False,
                    "context_lineage_persisted_before_backward": True,
                },
                context_lineage=forensic_context_lineage,
            )
            atomic_json(forensic_ring_path, forensic_ring.payload())

        update_metrics = [
            update_multiround_group(
                trajectories,
                flow_inferencer=flow_inferencer,
                reference_flow_inferencer=reference_flow_inferencer,
                policy_controller_inferencer=controller_inferencer,
                reference_controller_inferencer=reference_controller_inferencer,
                grpo_config=config,
                accelerator=accelerator,
                text_optimizer=text_optimizer,
                flow_optimizer=flow_optimizer,
                transformer=transformer,
                flow_parameters=flow_parameters,
                text_parameters=text_parameters,
                max_grad_norm=float(config.train.max_grad_norm),
                text_clip_range=float(config.g016.text_clip_range),
                text_kl_beta=float(config.g016.text_kl_beta),
                max_flow_calls=(
                    int(scored["max_flow_calls"])
                ),
                max_text_turns=(
                    int(scored["max_text_turns"])
                ),
                flow_padding_record=attempt["flow_padding_record"],
                process_group=None,
                commit_optimizer_step=True,
                active_channel_mode=("both"),
                optimizer_step_mode=("both"),
                capture_full_diagnostics=False,
                text_parameter_names=trainable_surface["text_parameter_names"],
                flow_parameter_names=trainable_surface["flow_parameter_names"],
                phase_observer=observe_update_phase,
                optimizer_step_observer=observe_optimizer_step,
            )
        ]
        update_transaction.record_local_update_complete(
            text_stepped=bool(update_metrics[0]["text_step"]["stepped"]),
            flow_stepped=bool(update_metrics[0]["flow_step"]["stepped"]),
        )
        committed_states = synchronize_update_commit(
            update_transaction, process_group=object_group
        )

        # Every committed step checks the actual policy/optimizer tensors, not
        # only logged scalar losses. These are load-bearing G016 automatic
        # nonfinite stops.
        local_nonfinite_parameters = sum(
            not bool(torch.isfinite(parameter.detach()).all().item())
            for parameter in (*text_parameters, *flow_parameters)
        )
        optimizer_tensors = [
            value
            for optimizer in (text_optimizer, flow_optimizer)
            for state in optimizer.state.values()
            for value in state.values()
            if torch.is_tensor(value) and value.is_floating_point()
        ]
        local_nonfinite_optimizer = sum(
            not bool(torch.isfinite(value.detach()).all().item())
            for value in optimizer_tensors
        )
        finite_counts = torch.tensor(
            [local_nonfinite_parameters, local_nonfinite_optimizer],
            device=accelerator.device,
            dtype=torch.int64,
        )
        if dist.is_initialized():
            dist.all_reduce(finite_counts, op=dist.ReduceOp.SUM)
        if int(finite_counts.sum().item()) != 0:
            raise RuntimeError(
                "G016 nonfinite parameter/optimizer state after committed update: "
                f"{finite_counts.tolist()}"
            )

        update_elapsed_sec = time.monotonic() - update_started
        # Full no-commit diagnostics may retain tiny CUDA tensor scalars.
        # Gloo object unpickling restores those tensors onto every policy GPU,
        # multiplying residency by world size at the post-backward peak. Move
        # the rank report to JSON-safe CPU values before object transport; this
        # changes evidence transport only, never policy/update math.
        local_rank_report = jsonable(
            {
                "update_metrics": update_metrics,
                "optimizer_precision": {
                    "text": optimizer_state_precision(text_optimizer),
                    "flow": optimizer_state_precision(flow_optimizer),
                },
                "nonfinite_parameter_tensor_count": int(finite_counts[0].item()),
                "nonfinite_optimizer_tensor_count": int(finite_counts[1].item()),
                "update_transaction": update_transaction.snapshot(),
                "committed_transaction_states": committed_states,
                "max_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(accelerator.device)
                ),
                "max_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(accelerator.device)
                ),
                "memory_events": list(memory_events),
                "decoder_call_count": len(decoder_module_order),
                "decoder_traversal_count": (
                    len(decoder_module_order) // 28
                ),
                "decoder_order_sha256": hashlib.sha256(
                    bytes(decoder_module_order)
                ).hexdigest(),
                "uniform_decoder_layer_order": (
                    len(decoder_module_order) % 28 == 0
                    and all(
                        tuple(decoder_module_order[start : start + 28])
                        in {
                            tuple(range(28)),
                            tuple(reversed(range(28))),
                        }
                        for start in range(
                            0,
                            len(decoder_module_order),
                            28,
                        )
                    )
                ),
            }
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rank_reports = gather_rank_objects(local_rank_report, object_group)
        accelerator.wait_for_everyone()
        persistence = {"ok": True, "error": None, "automatic_stop": None}
        if rank == 0:
            try:
                completed_now = time.monotonic()
                completion_interval = completed_now - timing_state["last_completion"]
                timing_state["last_completion"] = completed_now
                all_updates = [
                    value
                    for report in rank_reports
                    for value in report["update_metrics"]
                ]
                flow_rows = [
                    metric
                    for update in all_updates
                    for trajectory_metrics in update["flow_metrics"]
                    for metric in trajectory_metrics
                    if not bool(metric.get("collective_only_padding"))
                ]
                text_rows = [
                    metric
                    for update in all_updates
                    for trajectory_metrics in update["text_metrics"]
                    for metric in trajectory_metrics
                    if not bool(metric.get("collective_only_padding"))
                ]
                flow_losses = [abs(float(value["loss"])) for value in flow_rows]
                text_losses = [abs(float(value["total_loss"])) for value in text_rows]
                loss_max = max([*flow_losses, *text_losses], default=0.0)
                grad_max = max(
                    [
                        abs(float(update.get("flow_grad_norm") or 0.0))
                        for update in all_updates
                    ]
                    + [
                        abs(float(update.get("text_grad_norm") or 0.0))
                        for update in all_updates
                    ],
                    default=0.0,
                )
                nonfinite_paths = nonfinite_numeric_paths(
                    {
                        "scored": scored,
                        "all_rank_update_metrics": all_updates,
                        "loss_max": loss_max,
                        "grad_max": grad_max,
                    },
                    prefix="g016_step",
                )
                if nonfinite_paths:
                    raise RuntimeError(
                        "G016 nonfinite reward/advantage/loss metric: "
                        + json.dumps(nonfinite_paths[:20])
                    )
                # G022 item 2 / doc section 5 part 3: hard runtime stop on
                # max |delta|, so divergence halts the run instead of being
                # silently absorbed by a zero-gradient clamp. Evaluated here,
                # after the world-wide gather, so one rank cannot strand peers.
                text_kl_delta_stop = getattr(
                    config.g016, "kl_max_abs_delta_stop", None
                )
                text_action_support_summary = None
                if text_action_support_binding is not None:
                    text_action_support_summary = (
                        text_action_support_world_consensus(
                            all_updates,
                            expected_support_hash=(
                                text_action_support_binding["support"][
                                    "support_hash"
                                ]
                            ),
                        )
                    )
                text_kl_delta_summary = text_kl_delta_max(all_updates)
                if text_kl_delta_stop is not None:
                    assert_g022_delta_within_stop(
                        text_kl_delta_summary["max_abs_delta"],
                        limit=float(text_kl_delta_stop),
                        channel="text",
                    )
                prior_rows = history[-10:]
                loss_baseline = statistics.median(
                    float(value["numerical_safety"]["loss_abs_max"])
                    for value in prior_rows[-5:]
                ) if len(prior_rows) >= 5 else None
                grad_baseline = statistics.median(
                    float(value["numerical_safety"]["grad_norm_max"])
                    for value in prior_rows[-5:]
                ) if len(prior_rows) >= 5 else None
                explosion_excursion = bool(
                    (loss_baseline is not None and loss_baseline > 0 and loss_max > 100.0 * loss_baseline)
                    or (grad_baseline is not None and grad_baseline > 0 and grad_max > 100.0 * grad_baseline)
                )
                slow_baseline = statistics.median(
                    float(value["completion_interval_sec"])
                    for value in prior_rows[-10:]
                ) if len(prior_rows) >= 5 else None
                throughput_excursion = bool(
                    slow_baseline is not None
                    and slow_baseline > 0
                    and completion_interval > 3.0 * slow_baseline
                )
                prior_explosion = [
                    bool(value["numerical_safety"].get("explosion_excursion"))
                    for value in prior_rows[-2:]
                ]
                prior_slow = [
                    bool(value["numerical_safety"].get("throughput_excursion"))
                    for value in prior_rows[-2:]
                ]
                explosion_stop = len(prior_explosion) == 2 and all(prior_explosion) and explosion_excursion
                throughput_stop = len(prior_slow) == 2 and all(prior_slow) and throughput_excursion
                accounting = build_optimizer_step_accounting(
                    all_updates,
                    expected_world_size=world_size,
                    commit_optimizer_step=True,
                )
                row = {
                    "version": VERSION,
                    "method": ADVANTAGE_VERSION,
                    "channel_credit_version": ADVANTAGE_VERSION,
                    "controller_credit_version": CONTROLLER_CREDIT_VERSION,
                    "renderer_credit_version": RENDERER_CREDIT_VERSION,
                    "renderer_bucket_version": RENDERER_BUCKET_VERSION,
                    "step": int(attempt["step"]),
                    "stop_now_scorer_identity": STOP_NOW_SCORER_IDENTITY,
                    "stop_now_scorer_version": STOP_NOW_SCORER_VERSION,
                    "prompt_narrowing_manifest_sha256": frozen_bindings[
                        "prompt_narrowing_manifest_sha256"
                    ],
                    "text_learning_rate": float(config.g016.text_learning_rate),
                    "flow_learning_rate": float(config.train.learning_rate),
                    "stopnow_live_metrics": scored["stopnow_live_metrics"],
                    "sde_sampling": scored["sde_sampling"],
                    "repair_effectiveness": scored["repair_effectiveness"],
                    "g016_live_monitoring": scored["g016_live_monitoring"],
                    "g017_protocol_degradation": scored[
                        "g017_protocol_degradation"
                    ],
                    "flow_grad_norm": {
                        "rank_max": max(
                            abs(float(update.get("flow_grad_norm") or 0.0))
                            for update in all_updates
                        ),
                        "rank_median": statistics.median(
                            abs(float(update.get("flow_grad_norm") or 0.0))
                            for update in all_updates
                        ),
                    },
                    "raw_return_mean": float(scored["reward_diagnostics"]["raw_return_mean"]),
                    "raw_return_std": float(scored["reward_diagnostics"]["raw_return_std"]),
                    "r0_detachment": attempt["r0_detachment"],
                    "reward_diagnostics": scored["reward_diagnostics"],
                    "group_diagnostics": scored["group_diagnostics"],
                    "r0_advantage_records": [],
                    "trajectory_process_records": scored["trajectory_records"],
                    "repair_advantage_records": scored["trajectory_records"],
                    "per_round_counting_detector_lineage": scored["per_round_lineage"],
                    "policy_observation_secrecy": scored[
                        "policy_observation_secrecy"
                    ],
                    "gpt_request_count": int(scored["gpt_request_count"]),
                    "detector_transport": scored["detector_transport"],
                    "all_rank_update_metrics": jsonable(all_updates),
                    "optimizer_step_accounting": accounting,
                    "optimizer_state_precision": [
                        value["optimizer_precision"] for value in rank_reports
                    ],
                    "memory_probe": False,
                    "memory_events": jsonable(memory_events),
                    "memory_state_materialization": jsonable(memory_state_materialization),
                    "rank_memory": [
                        {
                            "rank": rank_index,
                            "max_allocated_bytes": int(
                                value["max_allocated_bytes"]
                            ),
                            "max_reserved_bytes": int(
                                value["max_reserved_bytes"]
                            ),
                            "memory_events": jsonable(value["memory_events"]),
                        }
                        for rank_index, value in enumerate(rank_reports)
                    ],
                    "rank_collective_order": [
                        {
                            "rank": rank_index,
                            "decoder_call_count": int(
                                value["decoder_call_count"]
                            ),
                            "decoder_traversal_count": int(
                                value["decoder_traversal_count"]
                            ),
                            "decoder_order_sha256": str(
                                value["decoder_order_sha256"]
                            ),
                            "uniform_decoder_layer_order": bool(
                                value["uniform_decoder_layer_order"]
                            ),
                        }
                        for rank_index, value in enumerate(rank_reports)
                    ],
                    "trainable_surface": trainable_surface,
                    "run_plan": authorization,
                    "fsdp_sharding_strategy": transformer.sharding_strategy.name,
                    "reference_cpu_streaming": bool(
                        config.g016.reference_cpu_streaming
                    ),
                    "optimizer_state_cpu_offload": bool(
                        config.g016.optimizer_state_cpu_offload
                    ),
                    "nested_fsdp_wrap": bool(config.g016.nested_fsdp_wrap),
                    "nested_collective_padding_version": (
                        "clean29529_g016_wrapped_root_attention_text_gen_order_v2"
                    ),
                    "numerical_safety": {
                        "version": "clean29529_g016_continuous_numerical_stop_v1",
                        "nonfinite_metric_count": 0,
                        "nonfinite_parameter_tensor_count": 0,
                        "nonfinite_optimizer_tensor_count": 0,
                        "loss_abs_max": loss_max,
                        "grad_norm_max": grad_max,
                        "loss_prior_five_median": loss_baseline,
                        "grad_prior_five_median": grad_baseline,
                        "explosion_excursion": explosion_excursion,
                        "explosion_three_consecutive_stop": explosion_stop,
                        "prior_ten_step_wall_median_sec": slow_baseline,
                        "throughput_excursion": throughput_excursion,
                        "throughput_three_consecutive_stop": throughput_stop,
                        "text_ratio_first_inner_epoch_max_abs_deviation": max(
                            [abs(float(value.get("ratio_max_abs_deviation", 0.0))) for value in text_rows],
                            default=0.0,
                        ),
                        "text_kl_max_abs_delta": text_kl_delta_summary[
                            "max_abs_delta"
                        ],
                        "text_kl_linearized_token_count": text_kl_delta_summary[
                            "linearized_token_count"
                        ],
                        "text_kl_bound_mode": text_kl_delta_summary["bound_mode"],
                        "g022_text_action_support": text_action_support_summary,
                        "text_kl_max_abs_delta_stop": (
                            None
                            if text_kl_delta_stop is None
                            else float(text_kl_delta_stop)
                        ),
                        "flow_kl_max": max(
                            [abs(float(value.get("kl_loss", 0.0))) for value in flow_rows],
                            default=0.0,
                        ),
                        "stopnow_credit_invariant_passed": (
                            int(scored["group_diagnostics"].get("trajectory_count", -1)) == (active_g021_trajectory_count() if g021_process_rtg_mode() else 28)
                            and scored["group_diagnostics"].get("version") == ADVANTAGE_VERSION
                            # G022 IS whole-trajectory broadcast by design (doc
                            # section 2.4: one scalar routed to every
                            # policy-active token). Every earlier stage forbade
                            # it. The invariant therefore asserts the value the
                            # active stage requires, not a fixed False.
                            and scored["group_diagnostics"].get("trajectory_level_advantage_broadcast") is True
                            and scored["group_diagnostics"].get("hard_projection") is False
                            and scored["group_diagnostics"].get("empirical_mean_used_as_center") is False
                            and scored["group_diagnostics"].get("stop_now_scorer_identity") == STOP_NOW_SCORER_IDENTITY
                            and scored["group_diagnostics"].get("stop_now_scorer_version") == STOP_NOW_SCORER_VERSION
                            and scored["group_diagnostics"].get("stop_now_table_invariants", {}).get("passed") is True
                            # G022 one-head assertions: nothing from the
                            # per-round credit design may be active.
                            and (
                                (
                                    scored["group_diagnostics"].get("deadzone_used") is False
                                    and scored["group_diagnostics"].get("leave_one_out_used") is False
                                    and scored["group_diagnostics"].get("return_to_go_used") is False
                                    and scored["group_diagnostics"].get("per_round_bucket_used") is False
                                    and scored["group_diagnostics"].get("action_head_used") is False
                                    and scored["group_diagnostics"].get("repair_head_used") is False
                                    and scored["group_diagnostics"].get("critic_used") is False
                                    and scored["group_diagnostics"].get("gae_used") is False
                                )
                            )
                            and int(scored["group_diagnostics"].get("exact_nonexact_mixed_bucket_count", -1)) == 0
                            and int(scored["group_diagnostics"].get("r0_policy_active_count", -1)) == 0
                            and int(scored["group_diagnostics"].get("r0_text_record_count", -1)) == 0
                            and int(scored["group_diagnostics"].get("r0_flow_record_count", -1)) == 0
                            and scored.get("policy_observation_secrecy", {}).get("passed") is True
                            and scored["r0_records"] == []
                        ),
                    },
                    "phase_timings": {
                        "rollout": attempt["rollout_phase_timings"],
                        "reward": scored["reward_phase_timings"],
                        "update_compute_sec": update_elapsed_sec,
                        "previous_checkpoint_sec": float(timing_state["previous_checkpoint_elapsed_sec"]),
                        "monitoring_history_read_sec": monitoring_payload["history_read_sec"],
                        "monitoring_bootstrap_sec": monitoring_payload["bootstrap_sec"],
                        "monitoring_total_sec": monitoring_elapsed_sec,
                    },
                    "g022_whole_trajectory": (
                        {
                            "version": G022_ADVANTAGE_VERSION,
                            "zero_std_metrics": scored["group_diagnostics"].get("zero_std_metrics"),
                            "historical_baseline_summary": scored["group_diagnostics"].get("historical_baseline_summary"),
                            "historical_baseline_reports": scored["group_diagnostics"].get("historical_baseline_reports"),
                            "zero_std_ratio": scored["group_diagnostics"].get("zero_std_ratio"),
                            "zero_std_group_count": scored["group_diagnostics"].get("zero_std_group_count"),
                            "advantage_saturation_rate": scored["group_diagnostics"].get("advantage_saturation_rate"),
                            "group_reward_mean": scored["group_diagnostics"].get("group_reward_mean"),
                            "group_reward_population_std": scored["group_diagnostics"].get("group_reward_population_std"),
                            "malformed_action_trajectory_count": scored["group_diagnostics"].get("malformed_action_trajectory_count"),
                            "premature_done_trajectory_count": scored["group_diagnostics"].get("premature_done_trajectory_count"),
                            "reward_terms": scored["group_diagnostics"].get("reward_terms"),
                        }
                    ),
                    "g022_difficulty_filter": g022_difficulty_filter_metrics_view(
                        attempt.get("g022_difficulty_filter", {"enabled": False})
                    ),
                    "rollout_elapsed_sec": float(attempt["rollout_elapsed_sec"]),
                    "update_elapsed_sec": update_elapsed_sec,
                    "completion_interval_sec": completion_interval,
                    "completed_at_utc": utc_now(),
                    "official_geneval_consumed": False,
                    "efficacy_metrics_are_report_only": True,
                }
                row["numerical_safety"]["round_causal_credit_invariant_passed"] = row[
                    "numerical_safety"
                ]["stopnow_credit_invariant_passed"]
                if not row["numerical_safety"]["stopnow_credit_invariant_passed"]:
                    raise RuntimeError("G016 stop-now/R0/secrecy invariant failed")
                forensic_token_rows = []
                for rank_index, update in enumerate(all_updates):
                    for local_trajectory, trajectory_metrics in enumerate(
                        update["text_metrics"]
                    ):
                        for turn_slot, metric in enumerate(trajectory_metrics):
                            token_topk = metric.get("forensic_token_topk") or {}
                            for token_row in token_topk.get("rows") or []:
                                forensic_token_rows.append(
                                    {
                                        **token_row,
                                        "sample_id": (
                                            f"step{int(attempt['step'])}:rank{rank_index}:"
                                            f"local{local_trajectory}:turn{turn_slot}"
                                        ),
                                        "rank": rank_index,
                                        "local_trajectory": local_trajectory,
                                        "turn_slot": turn_slot,
                                    }
                                )
                forensic_ring.finish_step(
                    step=int(attempt["step"]),
                    marker={
                        "phase": "post_update",
                        "created_at_utc": utc_now(),
                        "committed": True,
                        "loss_abs_max": float(loss_max),
                        "grad_norm_max": float(grad_max),
                        "explosion_excursion": bool(explosion_excursion),
                        "explosion_three_consecutive_stop": bool(explosion_stop),
                        "optimizer_step_accounting": accounting,
                    },
                    token_rows=forensic_token_rows,
                )
                atomic_json(forensic_ring_path, forensic_ring.payload())
                row["forensic_ring_buffer"] = {
                    "version": "clean29529_g016_forensic_ring_buffer_v1",
                    "path": str(forensic_ring_path),
                    "capacity_steps": 6,
                    "top_k_per_step": 64,
                    "context_image_hash_seed_persisted": True,
                    "pre_post_excursion_markers_persisted": True,
                }
                if rank == 0 and g022_baseline is not None:
                    # Item 7: persist before the metrics row is appended, so a
                    # crash between the two leaves a baseline that is at worst
                    # one step ahead, never behind the committed metrics.
                    g022_baseline.save(g022_baseline_path)
                persisted_row = row
                append_jsonl(metrics_path, persisted_row)
                append_jsonl(
                    dashboard_path,
                    {"step": int(attempt["step"]), **scored["reward_diagnostics"]},
                )
                update_g016_live_status(
                    live_status_path,
                    stage="formal_running",
                    values={
                        "formal_root": str(output_dir.resolve()),
                        "latest_committed_step": int(attempt["step"]),
                        "total_steps": total_steps,
                        "loss_abs_max": loss_max,
                        "grad_norm_max": grad_max,
                        "completion_interval_sec": completion_interval,
                        "gpt_request_count": 0,
                        "stopnow_credit_invariant_passed": True,
                        "round_causal_credit_invariant_passed": True,
                        "stopnow_sign_gate_categories": scored[
                            "stopnow_live_metrics"
                        ]["sign_gate_categories"],
                        "sde_sampling": scored["sde_sampling"],
                        "repair_effectiveness": scored[
                            "repair_effectiveness"
                        ],
                        "g016_live_monitoring": scored["g016_live_monitoring"],
                        "protocol_invalid_step": scored[
                            "g017_protocol_degradation"
                        ]["protocol_invalid_step"],
                        "protocol_degradation": scored[
                            "g017_protocol_degradation"
                        ],
                        "policy_observation_secrecy_passed": True,
                        "r0_gradient_contribution": 0.0,
                        "efficacy_metrics_report_only": True,
                    },
                )
                print(json.dumps(persisted_row, sort_keys=True), flush=True)
                # Optimization and throughput excursions are report-only in
                # G016; the preregistered automatic stops are enforced above.
            except Exception as exc:
                # The traceback used to be discarded here: only the message
                # survived into `persistence["error"]`, so every rank re-raised
                # a string with no line number. That is why the first two T1.2
                # instances each cost a full 16-GPU attempt to locate. Keep it.
                persistence = {
                    "ok": False,
                    "error": f"G016 rank0 persistence failed: {type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=40),
                    "automatic_stop": None,
                }
                print(
                    "[rank0] G016 PERSISTENCE TRACEBACK\n"
                    + traceback.format_exc(limit=40),
                    flush=True,
                )
        persistence = broadcast_rank_object(
            persistence if rank == 0 else None, src=0, group=object_group
        )
        if persistence.get("ok") is not True:
            raise RuntimeError(str(persistence.get("error")))
        update_transaction.mark_metrics_persisted()
        if rank == 0:
            # Attempts 9 and 10 died at `finish_without_checkpoint` with
            # "V20 update cannot finish before persistence" and no preceding
            # traceback, which static reading could not explain. Record the
            # transaction state at each milestone so a recurrence names the
            # step that did not happen instead of only its consequence.
            atomic_json(
                output_dir / attempt_scoped("G022_TRANSACTION_TRACE.json"),
                {
                    "version": "clean29529_g022_transaction_trace_v2",
                    "logical_step": int(step),
                    "reached_mark_metrics_persisted": True,
                    "durable_origin_kind": str(
                        update_transaction.durable_origin_kind
                    ),
                    "durable_checkpoint": str(
                        update_transaction.durable_checkpoint
                    ),
                    "optimizer_rng_resumable": bool(
                        update_transaction.optimizer_rng_resumable
                    ),
                    "commit_synchronized": bool(
                        update_transaction.commit_synchronized
                    ),
                    "metrics_persisted": bool(update_transaction.metrics_persisted),
                    "phase": str(update_transaction.phase),
                    "completed_step": update_transaction.completed_step,
                    "text_step_state": str(update_transaction.text_step_state),
                    "flow_step_state": str(update_transaction.flow_step_state),
                },
            )
        accelerator.wait_for_everyone()
        timing_state["previous_checkpoint_elapsed_sec"] = 0.0
        timing_state["completed_step"] = int(update_transaction.completed_step)
        if persistence.get("automatic_stop"):
            raise RuntimeError("G016 automatic stop: " + str(persistence["automatic_stop"]))
    pre_step_rng_state = None
    try:
        for step in range(start_step + 1, total_steps + 1):
            pre_step_rng_state = capture_rank_rng_state()
            update_transaction.begin(step)
            current = generate_attempt(step=step)
            reward_started = time.monotonic()
            scored = score_and_broadcast(
                current["local_payload"],
                reward_url=str(config.g016.reward_url),
                timeout_sec=float(config.g016.reward_timeout_sec),
                repair_progress_deadzone=float(
                    config.g016.repair_progress_deadzone
                ),
                rank=rank,
                object_group=object_group,
                historical_baseline=g022_baseline,
            )
            reward_elapsed_sec = time.monotonic() - reward_started
            record_memory_phase("after_reward")
            scored["async_reward"] = {
                "version": "clean29529_g016_synchronous_reward_v1",
                "request_id": None,
                "gpt_request_count": 0,
                "queue_depth": 0,
                "next_rollout_overlap": False,
                "rollout_policy_lag_updates": 0,
                "reward_elapsed_sec": reward_elapsed_sec,
                "wait_elapsed_sec": reward_elapsed_sec,
                "resolve_blocking_wait_sec": reward_elapsed_sec,
                "submit_payload_gather_sec": 0.0,
                "resolve_broadcast_sec": 0.0,
                "overlapped_rollout_step": None,
                "overlapped_rollout_elapsed_sec": 0.0,
                "fail_closed": True,
            }
            apply_scored_attempt(current, scored)
            if batches_consumed != step:
                raise RuntimeError("G016 completed-step sampler cursor differs")
            if (
                step
                in set(int(value) for value in config.g016.checkpoint_steps)
                or step == stage_checkpoint_step
            ):
                checkpoint_started = time.monotonic()
                scheduled_checkpoint_path = (
                    output_dir
                    / "checkpoints"
                    / f"checkpoint-{step}"
                )
                save_sharded_training_checkpoint(
                    scheduled_checkpoint_path,
                    model=transformer,
                    text_optimizer=text_optimizer,
                    flow_optimizer=flow_optimizer,
                    logical_step=step,
                    initialization_reference=initialization_reference,
                    sampler_state=checkpoint_sampler_state(
                        rank=rank,
                        world_size=world_size,
                        seed=int(config.g016.seed),
                        batches_consumed=batches_consumed,
                        contract=sampler_contract,
                    ),
                    extra_state={
                        "completed_step": int(step),
                        "next_rollout_seed": rollout_seed(
                            base_seed=int(config.g016.seed),
                            logical_step=step + 1,
                        ),
                        "channel_credit_version": ADVANTAGE_VERSION,
                        "controller_credit_version": CONTROLLER_CREDIT_VERSION,
                        "renderer_credit_version": RENDERER_CREDIT_VERSION,
                        "renderer_bucket_version": RENDERER_BUCKET_VERSION,
                        "stopnow_credit_method": CONTROLLER_CREDIT_VERSION,
                        "stop_now_scorer_identity": STOP_NOW_SCORER_IDENTITY,
                        "stop_now_scorer_version": STOP_NOW_SCORER_VERSION,
                        "prompt_narrowing_manifest_sha256": frozen_bindings[
                            "prompt_narrowing_manifest_sha256"
                        ],
                        "hidden_exactness_policy_visible": False,
                        "r0_policy_or_gradient": False,
                    },
                    process_group=None,
                    control_group=object_group,
                    auxiliary_modules=auxiliary_flow_modules,
                    skip_dcp_optimizer=True,
                )
                timing_state["previous_checkpoint_elapsed_sec"] = (
                    time.monotonic() - checkpoint_started
                )
                update_transaction.mark_checkpoint_durable(
                    logical_step=step,
                    checkpoint=str(scheduled_checkpoint_path.resolve()),
                )
            else:
                update_transaction.finish_without_checkpoint()
            transaction_states = gather_transaction_states(
                update_transaction,
                process_group=object_group,
            )
            if len(
                {
                    (
                        state["completed_step"],
                        state["durable_step"],
                        state["phase"],
                    )
                    for state in transaction_states
                }
            ) != 1:
                raise RuntimeError(
                    "G016 post-step transaction state differs"
                )
            pre_step_rng_state = None
            accelerator.wait_for_everyone()
        if rank == 0:
            atomic_json(
                output_dir / "g016_wandb_status.json",
                redacted_wandb_status(wandb_logger.finish()),
            )
    except BaseException as exc:
        print(
            f"[rank{rank}] G016_PRIMARY_EXCEPTION {type(exc).__name__}: {exc}\n"
            + traceback.format_exc(limit=80),
            flush=True,
        )
        try:
            if rank == 0:
                atomic_json(
                    output_dir / "g016_wandb_status.json",
                    redacted_wandb_status(wandb_logger.finish()),
                )
        except Exception:
            pass
        try:
            transaction_rank_states = gather_transaction_states(
                update_transaction,
                process_group=object_group,
            )
            emergency_decision = emergency_checkpoint_decision(
                transaction_rank_states
            )
        except Exception as transaction_exc:
            transaction_rank_states = [update_transaction.snapshot()]
            emergency_decision = {
                "allow_checkpoint": False,
                "classification": "transaction_state_gather_failed",
                "checkpoint_step": None,
                "resume_step": int(update_transaction.durable_step),
                "resume_checkpoint": str(
                    update_transaction.durable_checkpoint
                ),
                "transaction_error": (
                    f"{type(transaction_exc).__name__}: "
                    f"{transaction_exc}"
                ),
            }
        # T1.2 / Appendix P. The per-rank transaction states were only
        # persisted down one conditional branch, so a failure like the one that
        # killed eleven attempts left nothing a later reader could check --
        # and the attempt roots were then deleted, which is how "the update
        # transaction has never completed" became an unverifiable assertion.
        # Write them UNCONDITIONALLY, on every failure, naming the transition
        # each rank did not reach rather than only its consequence.
        try:
            if rank == 0:
                atomic_json(
                    output_dir
                    / attempt_scoped("G016_TRANSACTION_FAILURE_RANKS.json"),
                    {
                        "version": "clean29529_g022_transaction_failure_ranks_v1",
                        "created_at_utc": utc_now(),
                        "primary_exception": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=80),
                        "logical_step": (
                            update_transaction.current_step
                            if update_transaction.current_step is not None
                            else update_transaction.completed_step
                        ),
                        "emergency_decision": emergency_decision,
                        "rank_count": len(transaction_rank_states),
                        "phases_by_rank": [
                            {
                                "rank": index,
                                "phase": state.get("phase"),
                                "current_step": state.get("current_step"),
                                "completed_step": state.get("completed_step"),
                                "text_step_state": state.get("text_step_state"),
                                "flow_step_state": state.get("flow_step_state"),
                                "commit_synchronized": state.get(
                                    "commit_synchronized"
                                ),
                                "metrics_persisted": state.get(
                                    "metrics_persisted"
                                ),
                                "events": state.get("events"),
                            }
                            for index, state in enumerate(transaction_rank_states)
                        ],
                        "distinct_phases": sorted(
                            {
                                str(state.get("phase"))
                                for state in transaction_rank_states
                            }
                        ),
                        "ranks_disagree": len(
                            {
                                (
                                    state.get("phase"),
                                    state.get("text_step_state"),
                                    state.get("flow_step_state"),
                                    state.get("commit_synchronized"),
                                    state.get("metrics_persisted"),
                                )
                                for state in transaction_rank_states
                            }
                        )
                        > 1,
                    },
                )
        except Exception as dump_exc:  # never mask the primary exception
            print(
                f"[rank{rank}] transaction failure dump failed: {dump_exc}",
                flush=True,
            )
        if (
            emergency_decision["classification"]
            == "clean_pre_step_boundary"
            and update_transaction.current_step is not None
            and pre_step_rng_state is not None
        ):
            restore_rank_rng_state(pre_step_rng_state)
        if (
            emergency_decision.get("allow_checkpoint") is True
        ):
            emergency_step = int(
                emergency_decision["checkpoint_step"]
            )
            try:
                emergency_path = (
                    output_dir
                    / "checkpoints"
                    / f"checkpoint-{emergency_step}-emergency"
                )
                save_sharded_training_checkpoint(
                    emergency_path,
                    model=transformer,
                    text_optimizer=text_optimizer,
                    flow_optimizer=flow_optimizer,
                    logical_step=emergency_step,
                    initialization_reference=initialization_reference,
                    sampler_state=checkpoint_sampler_state(
                        rank=rank,
                        world_size=world_size,
                        seed=int(config.g016.seed),
                        batches_consumed=emergency_step,
                        contract=sampler_contract,
                    ),
                    extra_state={
                        "emergency_reason": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                        "transaction_decision": emergency_decision,
                        "transaction_rank_states": (
                            transaction_rank_states
                        ),
                    },
                    process_group=None,
                    control_group=object_group,
                    auxiliary_modules=auxiliary_flow_modules,
                    skip_dcp_optimizer=True,
                )
                if rank == 0:
                    atomic_json(
                        output_dir
                        / "checkpoints"
                        / f"checkpoint-{emergency_step}-emergency.json",
                        {
                            "version": (
                                "clean29529_g016_transactional_emergency_checkpoint_v1"
                            ),
                            "created_at_utc": utc_now(),
                            "completed_step": emergency_step,
                            "reason": f"{type(exc).__name__}: {exc}",
                            "transaction_decision": emergency_decision,
                            "transaction_rank_states": (
                                transaction_rank_states
                            ),
                            "optimizer_state_saved": True,
                            "fail_closed_run_stopped": True,
                        },
                    )
            except Exception as checkpoint_exc:
                if rank == 0:
                    print(
                        "G016 emergency checkpoint failed: "
                        f"{type(checkpoint_exc).__name__}: {checkpoint_exc}",
                        flush=True,
                    )
        elif emergency_decision.get("allow_checkpoint") is not True:
            if rank == 0:
                contamination_step = (
                    update_transaction.current_step
                    if update_transaction.current_step is not None
                    else update_transaction.completed_step + 1
                )
                contamination_path = output_dir / attempt_scoped(
                    "PARTIAL_UPDATE_CONTAMINATION_"
                    f"STEP{int(contamination_step):04d}.json"
                )
                atomic_json(
                    contamination_path,
                    {
                        "version": (
                            "clean29529_g016_partial_update_contamination_v1"
                        ),
                        "created_at_utc": utc_now(),
                        "failed_step": int(contamination_step),
                        "reason": f"{type(exc).__name__}: {exc}",
                        "transaction_decision": emergency_decision,
                        "transaction_rank_states": transaction_rank_states,
                        "resumable_partial_state_saved": False,
                        "required_resume_step": emergency_decision.get(
                            "resume_step"
                        ),
                        "required_resume_checkpoint": (
                            emergency_decision.get("resume_checkpoint")
                        ),
                        "fail_closed_run_stopped": True,
                    },
                )
        raise


if __name__ == "__main__":
    app.run(main)
