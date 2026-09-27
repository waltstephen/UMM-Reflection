"""G008 state-grouped multiround Flow-GRPO runtime primitives."""

from __future__ import annotations

import math
import json
import pickle
import statistics
import time
from io import BytesIO
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.distributed as dist

from unify_rl.train.v20_update_contract import (
    assert_gradients_none_or_zero,
    clear_gradients,
    gradient_inventory,
    step_channel_optimizer,
    update_state_delta,
    update_state_snapshot,
)
from unify_rl.train.v20_release_gate import (
    FLOW_KL_LIMIT,
    TEXT_RATIO_LIMIT,
    nonfinite_numeric_paths,
)

from flow_grpo.bagel.modeling.bagel.bagel import (
    G022_ETA_CLAMP_MODE,
    LEGACY_ETA_CLAMP_MODE,
)
from flow_grpo.g022_text_action_support import (
    UnsupportedTextActionTokenError,
    assert_support_contract_synchronized,
    unsupported_token_evidence,
)
from unify_rl.train.g022_bounded_kl import (
    LINEARIZATION_THRESHOLD as G022_LINEARIZATION_THRESHOLD,
    MAX_ABS_DELTA_STOP as G022_MAX_ABS_DELTA_STOP,
    VERSION as G022_BOUND_VERSION,
    bounded_k3 as g022_bounded_k3,
    bounded_ratio as g022_bounded_ratio,
    delta_diagnostics as g022_delta_diagnostics,
)

# Bound selection for the text channel. "clamp20" is the exact G008-G021
# expression and is the default so every prior run stays reproducible;
# "g022_linear" is the non-vanishing-gradient bound of doc section 5.
LEGACY_BOUND_MODE = "clamp20"
G022_BOUND_MODE = "g022_linear"


def _use_g022_bound(bound_mode: str) -> bool:
    mode = str(bound_mode)
    if mode == LEGACY_BOUND_MODE:
        return False
    if mode == G022_BOUND_MODE:
        return True
    raise ValueError(f"unknown text loss bound mode: {bound_mode!r}")


# The single-head text loss is the G022 marker on the config; `g016_two_head` is
# every legacy stage. Read the same way as `g016_counting_process.py:901`.
G022_TEXT_HEAD_MODE = "g022_single_head"


def g022_single_head_mode(grpo_config: Any) -> bool:
    """True when this config is a G022 run, from `g016.text_head_mode`."""

    return (
        str(
            getattr(
                getattr(grpo_config, "g016", None),
                "text_head_mode",
                "g016_two_head",
            )
        )
        == G022_TEXT_HEAD_MODE
    )


# G008 intentionally reuses the exact frozen G007/V22 SFT-native protocol.
from unify_rl.inference.v22_controller_protocol import (
    FORCED_PREFIX_VERSION,
    SCORE_FIELD_VERSION,
    VERSION as G008_PROTOCOL_VERSION,
    ScoreFieldSchedule,
    canonicalize_response,
    forced_controller_prefix,
    parse_controller_response,
)


CONTRACT_VERSION = "clean29529_g008_state_grouped_multiround_flowgrpo_v1"
REWARD_VERSION = "clean29529_g008_simple_counting_reward_v1"
TEXT_POLICY_VERSION = "clean29529_g008_text_clip_ppo_k3_v1"
PADDING_VERSION = "clean29529_g008_real_path_padding_v1"
CONTROLLER_PROTOCOL_VERSION = G008_PROTOCOL_VERSION
CONTROLLER_FORCED_PREFIX_VERSION = FORCED_PREFIX_VERSION
CONTROLLER_SCORE_FIELD_VERSION = SCORE_FIELD_VERSION
# Invalid/malformed output receives the ordinary failed trajectory return and
# its standard group-centered trajectory advantage. There is no fixed local
# credit in G008.
MALFORMED_TEXT_LOCAL_CREDIT = 0.0
MALFORMED_TEXT_LOCAL_CREDIT_SCALE = "disabled_trajectory_advantage_only"
LEGACY_MALFORMED_TEXT_LOCAL_CREDIT = -0.5
GENEVAL_REQUEST_CHUNK_SIZE = 64
GENEVAL_MAX_ATTEMPTS_PER_CHUNK = 3
GENEVAL_EXPECTED_SERVICE_VERSION = (
    "clean29529_g008_counting_detector_service_v1"
)
GENEVAL_EXPECTED_PROTOCOL_VERSION = "flow_grpo_geneval_pickle_18085_v1"
ROLLOUT_VERSION = "clean29529_g008_exact_anchor_fork_rollout_v1"
MAX_REPAIR_ROUNDS = 3
MAX_CONTROLLER_TURNS = 3
# R2: flow_loss = r0_channel_weight * L_r0 + 1.0 * L_repair. Without it the R0 single-shot objective takes about 67% of the
# flow gradient while the acceptance metric is the paired `final - R0`.
R0_CHANNEL_WEIGHT = 0.25
REPAIR_CHANNEL_WEIGHT = 1.0
CHANNEL_WEIGHT_VERSION = "clean29529_v20_flow_channel_weight_v1"
# Standard trajectory GRPO only: no extra action-token objective.
CONTROLLER_ACTION_AUX_WEIGHT = 0.0
CONTROLLER_ACTION_AUX_VERSION = "clean29529_g008_action_token_aux_disabled_v1"


@dataclass
class FlowCallRecord:
    call_index: int
    call_kind: str
    controller_round_index: int | None
    input_terms: list[Any]
    image: Any
    latents: list[torch.Tensor]
    log_probs: list[torch.Tensor]
    timesteps: torch.Tensor
    sde_timestep_begin: int | None = None
    sde_window_seed_identity: dict[str, int] | None = None
    sde_transition_layout: str = "contiguous_v1"
    sde_transition_indices: list[int] | None = None
    # ------------------------------------------------------------------
    # Replay bindings (review of item 4).
    #
    # `call_kind` is a validated field and the learn pass reads it back, so
    # rollout and replay structurally cannot disagree about the stage. Every
    # other sampler quantity that determines the recorded log-probability was
    # re-derived from `grpo_config` at learn time instead, which is fine within
    # one step and wrong across a resume whose config changed (G021-A was
    # resumed from checkpoint 88, so this is a real path) or an offline replay
    # of archived records. It fails silently: a wrong ratio, no error.
    #
    # These carry the values the rollout actually used. `assert_replay_bindings`
    # compares them with the config at learn time and raises on disagreement.
    # `None` means a record predating this field; the caller then falls back to
    # the config and says so, rather than pretending it verified anything.
    # ------------------------------------------------------------------
    noise_level: float | None = None
    eta_clamp_mode: str | None = None
    replay_bindings: dict[str, Any] | None = None
    advantage: float = 0.0
    policy_active: bool = True

    def validate(self) -> None:
        if self.call_kind not in {"r0", "repair"}:
            raise ValueError(f"unsupported flow call kind: {self.call_kind}")
        if self.call_index < 0:
            raise ValueError("flow call index must be non-negative")
        if self.call_kind == "r0" and self.controller_round_index is not None:
            raise ValueError("R0 flow call cannot have a controller round")
        if self.call_kind == "repair" and self.controller_round_index is None:
            raise ValueError("repair flow call requires a controller round")
        if not self.latents or not self.log_probs:
            raise ValueError("flow call omitted SDE trajectory records")
        if self.sde_transition_layout == "g021_stratified_pairs_v1":
            if len(self.latents) != 2 * len(self.log_probs):
                raise ValueError("G021 paired flow latent/log-prob lengths differ")
            if self.sde_transition_indices is None or len(self.sde_transition_indices) != len(self.log_probs):
                raise ValueError("G021 transition-index coverage differs")
        elif len(self.latents) != len(self.log_probs) + 1:
            raise ValueError("flow latent/log-prob trajectory lengths differ")
        if int(self.timesteps.numel()) != len(self.log_probs):
            raise ValueError("flow timestep/log-prob trajectory lengths differ")
        if not isinstance(self.policy_active, bool):
            raise ValueError("flow policy_active must be boolean")
        if self.sde_timestep_begin is not None and self.sde_timestep_begin < 0:
            raise ValueError("flow SDE timestep begin is negative")
        if self.sde_window_seed_identity is not None:
            required = {
                "experiment_seed",
                "logical_step",
                "trajectory_index",
                "round_index",
            }
            if set(self.sde_window_seed_identity) != required:
                raise ValueError("flow SDE seed identity differs")
        if self.noise_level is not None and not (
            math.isfinite(float(self.noise_level)) and float(self.noise_level) >= 0.0
        ):
            raise ValueError("flow record noise level must be finite and non-negative")
        if self.eta_clamp_mode is not None and self.eta_clamp_mode not in {
            LEGACY_ETA_CLAMP_MODE,
            G022_ETA_CLAMP_MODE,
        }:
            raise ValueError(
                f"flow record eta clamp mode is unknown: {self.eta_clamp_mode!r}"
            )
        if self.replay_bindings is not None:
            missing = REPLAY_BINDING_KEYS - set(self.replay_bindings)
            if missing:
                raise ValueError(
                    f"flow record replay bindings are incomplete: {sorted(missing)}"
                )

    def offload_to_cpu(self) -> None:
        self.latents = [
            value.detach().to(device="cpu", non_blocking=False)
            for value in self.latents
        ]
        self.log_probs = [
            value.detach().to(device="cpu", non_blocking=False)
            for value in self.log_probs
        ]
        self.timesteps = self.timesteps.detach().to(
            device="cpu",
            non_blocking=False,
        )

    def training_sample(
        self,
        *,
        device: torch.device | str | None = None,
    ) -> dict[str, Any]:
        self.validate()
        if self.sde_transition_layout == "g021_stratified_pairs_v1":
            latents = torch.stack(self.latents[0::2], dim=0)
            prev_latents = torch.stack(self.latents[1::2], dim=0)
        else:
            latents = torch.stack(self.latents[:-1], dim=0)
            prev_latents = torch.stack(self.latents[1:], dim=0)
        log_probs = torch.stack(self.log_probs, dim=0)
        timesteps = self.timesteps
        if device is not None:
            latents = latents.to(device=device, non_blocking=False)
            prev_latents = prev_latents.to(
                device=device,
                non_blocking=False,
            )
            log_probs = log_probs.to(device=device, non_blocking=False)
            timesteps = timesteps.to(device=device, non_blocking=False)
        return {
            "latents": latents,
            "prev_latents": prev_latents,
            "log_probs": log_probs,
            "timesteps": timesteps,
            "advantages": torch.tensor(
                float(self.advantage),
                device=timesteps.device,
                dtype=torch.float32,
            ),
            "policy_active": self.policy_active,
            "sde_transition_layout": self.sde_transition_layout,
            "sde_transition_indices": list(self.sde_transition_indices or []),
            "noise_level": self.noise_level,
            "eta_clamp_mode": self.eta_clamp_mode,
            "replay_bindings": (
                None
                if self.replay_bindings is None
                else dict(self.replay_bindings)
            ),
        }


@dataclass
class TextTurnRecord:
    turn_index: int
    context_terms: list[Any]
    forced_prefix_token_ids: list[int]
    sampled_token_ids: list[int]
    old_log_probs: list[float]
    behavior_temperature: float
    canonical_response: str
    action: str
    advantage: float = 0.0
    policy_active: bool = True
    valid: bool = True
    local_advantage: float | None = None
    # Packed old-policy replay captured before any optimizer update. Sampler
    # log-probs remain in `old_log_probs` as behavior evidence; PPO uses this
    # estimator-matched denominator so first-inner-epoch ratio measures policy
    # change/stale context rather than incremental-vs-packed kernel drift.
    replay_old_log_probs: list[float] | None = None
    sampled_token_allowed_ids: list[list[int] | None] | None = None
    sampled_token_policy_active: list[bool] | None = None

    def credited_token_mask(self) -> list[bool]:
        """Positions that receive PPO/KL credit for this turn."""
        if self.sampled_token_policy_active is None:
            return [True] * len(self.sampled_token_ids)
        return [bool(value) for value in self.sampled_token_policy_active]

    def validate(self) -> None:
        if self.turn_index < 0:
            raise ValueError("text turn index must be non-negative")
        if self.action not in {"edit", "done", "invalid"}:
            raise ValueError("text turn action must be EDIT, DONE, or invalid")
        if self.valid and self.action == "invalid":
            raise ValueError("valid text turn cannot use invalid action")
        if not self.valid and self.action != "invalid":
            raise ValueError("malformed text turn must use invalid action")
        if not self.sampled_token_ids:
            raise ValueError("text turn has no model-sampled tokens")
        if any(
            type(value) is not int or value < 0
            for value in (
                *self.forced_prefix_token_ids,
                *self.sampled_token_ids,
            )
        ):
            raise ValueError("text turn token IDs are malformed")
        if len(self.sampled_token_ids) != len(self.old_log_probs):
            raise ValueError("sampled text token/log-prob lengths differ")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in self.old_log_probs
        ):
            raise ValueError("sampled text log-probs are malformed")
        if not math.isfinite(self.behavior_temperature):
            raise ValueError("text behavior temperature must be finite")
        if self.behavior_temperature <= 0.0:
            raise ValueError("text behavior temperature must be positive")
        if not self.canonical_response:
            raise ValueError("text turn canonical response is empty")
        if not isinstance(self.policy_active, bool):
            raise ValueError("text policy_active must be boolean")
        if self.local_advantage is not None and not math.isfinite(
            float(self.local_advantage)
        ):
            raise ValueError("text local advantage must be finite")
        if self.replay_old_log_probs is not None:
            if len(self.replay_old_log_probs) != len(self.sampled_token_ids):
                raise ValueError("packed old replay/token lengths differ")
            if any(not math.isfinite(float(value)) for value in self.replay_old_log_probs):
                raise ValueError("packed old replay log-probs are nonfinite")
        if self.sampled_token_allowed_ids is not None:
            if len(self.sampled_token_allowed_ids) != len(
                self.sampled_token_ids
            ):
                raise ValueError("text constraint mask length differs")
            for allowed, token in zip(
                self.sampled_token_allowed_ids,
                self.sampled_token_ids,
            ):
                if allowed is None:
                    continue
                values = [int(value) for value in allowed]
                if not values or len(set(values)) != len(values):
                    raise ValueError("text constraint set is malformed")
                if int(token) not in values:
                    raise ValueError(
                        "sampled text token is outside its constraint set"
                    )
        if self.sampled_token_policy_active is not None:
            if len(self.sampled_token_policy_active) != len(
                self.sampled_token_ids
            ):
                raise ValueError("text policy mask length differs")
            if any(
                not isinstance(value, bool)
                for value in self.sampled_token_policy_active
            ):
                raise ValueError("text policy mask must be boolean")
            if self.valid and not any(self.sampled_token_policy_active):
                raise ValueError("valid text turn has no credited token")


@dataclass
class MultiroundTrajectory:
    prompt: str
    r0_image: Any
    final_image: Any
    flow_calls: list[FlowCallRecord]
    text_turns: list[TextTurnRecord]
    events: list[dict[str, Any]]
    stop_reason: str
    done: bool
    repair_rounds: int
    reward: float | None = None
    advantage: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    text_padding_turn: TextTurnRecord | None = None

    def validate(self) -> None:
        if not self.prompt:
            raise ValueError("trajectory prompt is empty")
        if self.repair_rounds < 0 or self.repair_rounds > MAX_REPAIR_ROUNDS:
            raise ValueError("trajectory repair count exceeds the G008 cap")
        if any(value.call_kind != "repair" for value in self.flow_calls):
            raise ValueError("G008 repair clone contains an R0 flow record")
        if len(self.flow_calls) != self.repair_rounds:
            raise ValueError("trajectory repair count does not match flow calls")
        if not str(self.metadata.get("anchor_sha256") or ""):
            raise ValueError("G008 repair trajectory omitted anchor identity")
        if len(self.text_turns) > MAX_CONTROLLER_TURNS:
            raise ValueError("trajectory exceeds the controller turn cap")
        for record in self.flow_calls:
            record.validate()
        for record in self.text_turns:
            record.validate()
            if not record.valid and record.local_advantage is not None:
                raise ValueError(
                    "G008 malformed text turn must use trajectory advantage only"
                )
        if self.text_padding_turn is not None:
            self.text_padding_turn.validate()
            if self.text_padding_turn.policy_active:
                raise ValueError("text padding turn must be inactive")

    def assign_channel_advantages(
        self,
        *,
        repair_flow: Sequence[float],
        controller: Sequence[float],
        repair_active: Sequence[bool],
        controller_active: Sequence[bool],
    ) -> None:
        if (
            len(repair_flow) != self.repair_rounds
            or len(repair_active) != self.repair_rounds
            or len(controller) != len(self.text_turns)
            or len(controller_active) != len(self.text_turns)
        ):
            raise ValueError("G008 channel advantage coverage differs")
        values = [
            *(float(value) for value in repair_flow),
            *(float(value) for value in controller),
        ]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("G008 channel advantage is nonfinite")
        for record, value, active in zip(
            self.flow_calls,
            repair_flow,
            repair_active,
        ):
            record.advantage = float(value)
            record.policy_active = bool(active)
        for record, value, active in zip(
            self.text_turns,
            controller,
            controller_active,
        ):
            record.advantage = float(value)
            record.policy_active = bool(active)
            record.local_advantage = None

    @property
    def valid_text_turn_count(self) -> int:
        return sum(record.valid for record in self.text_turns)

    @property
    def malformed_text_turn_count(self) -> int:
        return sum(not record.valid for record in self.text_turns)


def legacy_trajectory_reward_metric(
    *,
    r0_score: float,
    final_score: float,
    done: bool,
    r0_strict_pass: bool,
    strict_pass_final: bool,
    round_cap_hit: bool,
    repair_rounds: int,
    round_scores: Sequence[float] | None = None,
) -> dict[str, Any]:
    r0 = float(r0_score)
    final = float(final_score)
    if not math.isfinite(r0) or not 0.0 <= r0 <= 1.0:
        raise ValueError("R0 graded score must be finite and in [0,1]")
    if not math.isfinite(final) or not 0.0 <= final <= 1.0:
        raise ValueError("final graded score must be finite and in [0,1]")
    normalized_round_scores = (
        [r0, final]
        if round_scores is None
        else [float(value) for value in round_scores]
    )
    if (
        not normalized_round_scores
        or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in normalized_round_scores
        )
        or abs(normalized_round_scores[0] - r0) > 1e-12
        or abs(normalized_round_scores[-1] - final) > 1e-12
    ):
        raise ValueError("per-round graded score curve is invalid")
    progress = max(0.0, final - r0)
    if done:
        terminal_term = 0.2 if strict_pass_final else -0.2
        terminal_reason = (
            "done_strict_pass" if strict_pass_final else "done_not_pass"
        )
    else:
        terminal_term = -0.3
        terminal_reason = (
            "round_cap_without_done"
            if round_cap_hit
            else "non_done_termination"
        )
    meddle_term = (
        -0.1
        if bool(r0_strict_pass)
        and int(repair_rounds) > 0
        and final < r0
        else 0.0
    )
    r0_term = 0.3 * r0
    final_term = final
    progress_term = 0.15 * progress
    total = (
        r0_term
        + final_term
        + progress_term
        + terminal_term
        + meddle_term
    )
    return {
        "version": REWARD_VERSION,
        "r0_score": r0,
        "final_score": final,
        "round_scores": normalized_round_scores,
        "progress": progress,
        "r0_term": r0_term,
        "final_term": final_term,
        "progress_term": progress_term,
        "terminal_term": terminal_term,
        "terminal_reason": terminal_reason,
        "meddle_term": meddle_term,
        "done": bool(done),
        "r0_strict_pass": bool(r0_strict_pass),
        "strict_pass_final": bool(strict_pass_final),
        "round_cap_hit": bool(round_cap_hit),
        "repair_rounds": int(repair_rounds),
        "total": total,
    }


def sandbag_margin(
    *,
    honest_r0_score: float,
    sandbagged_r0_score: float,
    final_score: float,
    done: bool,
    r0_strict_pass: bool,
    strict_pass_final: bool,
    round_cap_hit: bool,
    repair_rounds: int,
    honest_judge_term: float = 0.3,
    sandbagged_judge_term: float = 0.3,
) -> float:
    if sandbagged_r0_score > honest_r0_score:
        raise ValueError("sandbagged R0 score must not exceed honest R0")
    honest = legacy_trajectory_reward_metric(
        r0_score=honest_r0_score,
        final_score=final_score,
        done=done,
        r0_strict_pass=r0_strict_pass,
        strict_pass_final=strict_pass_final,
        round_cap_hit=round_cap_hit,
        repair_rounds=repair_rounds,
    )
    sandbagged = legacy_trajectory_reward_metric(
        r0_score=sandbagged_r0_score,
        final_score=final_score,
        done=done,
        r0_strict_pass=False,
        strict_pass_final=strict_pass_final,
        round_cap_hit=round_cap_hit,
        repair_rounds=repair_rounds,
    )
    return (
        float(honest["total"])
        + float(honest_judge_term)
        - float(sandbagged["total"])
        - float(sandbagged_judge_term)
    )


def release_prompt_index(
    *,
    prompt_indices: Sequence[int],
    step: int,
    resample_attempt: int,
    resample_stride: int,
    dataset_size: int,
) -> int:
    if not prompt_indices:
        raise ValueError("release prompt schedule is empty")
    if step <= 0 or step > len(prompt_indices):
        raise ValueError("release prompt schedule does not cover this step")
    if resample_attempt < 0 or resample_stride <= 0:
        raise ValueError("release resample coordinates are invalid")
    index = (
        int(prompt_indices[step - 1])
        + int(resample_attempt) * int(resample_stride)
    )
    if index < 0 or index >= int(dataset_size):
        raise ValueError("release prompt index is outside the dataset")
    return index


def request_geneval_score_chunks(
    *,
    images: Sequence[bytes],
    metadata: Sequence[Mapping[str, Any]],
    reward_url: str,
    timeout_sec: float,
    chunk_size: int = GENEVAL_REQUEST_CHUNK_SIZE,
    max_attempts: int = GENEVAL_MAX_ATTEMPTS_PER_CHUNK,
    post: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if len(images) != len(metadata) or not images:
        raise ValueError("GenEval image/metadata coverage is invalid")
    if chunk_size <= 0 or max_attempts <= 0:
        raise ValueError("GenEval chunk/retry bounds must be positive")
    if post is None:
        import requests

        post = requests.post
    scores: list[float] = []
    strict_rewards: list[bool] = []
    result_rows: list[dict[str, Any]] = []
    chunk_sizes = []
    total_attempts = 0
    for start in range(0, len(images), int(chunk_size)):
        stop = min(start + int(chunk_size), len(images))
        chunk_sizes.append(stop - start)
        result = None
        last_error = ""
        for attempt in range(1, int(max_attempts) + 1):
            total_attempts += 1
            try:
                response = post(
                    reward_url,
                    data=pickle.dumps(
                        {
                            "images": list(images[start:stop]),
                            "meta_datas": list(metadata[start:stop]),
                            "only_strict": False,
                        }
                    ),
                    timeout=float(timeout_sec),
                )
                status = int(response.status_code)
                if 200 <= status < 300:
                    result = pickle.loads(response.content)
                    break
                try:
                    error_payload = pickle.loads(response.content)
                except Exception:
                    error_payload = {
                        "error": response.text[:1000],
                    }
                last_error = (
                    f"HTTP {status}: "
                    f"{error_payload.get('error') or error_payload}"
                )
                if status < 500 or attempt == int(max_attempts):
                    break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == int(max_attempts):
                    break
            time.sleep(float(attempt))
        if result is None:
            raise RuntimeError(
                f"GenEval chunk {start}:{stop} failed after "
                f"{max_attempts} attempts: {last_error}"
            )
        if (
            result.get("service_version")
            != GENEVAL_EXPECTED_SERVICE_VERSION
            or result.get("protocol_version")
            != GENEVAL_EXPECTED_PROTOCOL_VERSION
        ):
            raise RuntimeError(
                "GenEval response service/protocol identity mismatch"
            )
        chunk_scores = [
            float(value) for value in result.get("scores") or []
        ]
        chunk_strict = [
            bool(value) for value in result.get("strict_rewards") or []
        ]
        chunk_results = list(result.get("results") or [])
        if (
            len(chunk_scores) != stop - start
            or len(chunk_strict) != stop - start
            or len(chunk_results) != stop - start
        ):
            raise RuntimeError("GenEval chunk response coverage mismatch")
        scores.extend(chunk_scores)
        strict_rewards.extend(chunk_strict)
        result_rows.extend(dict(value) for value in chunk_results)
    return {
        "scores": scores,
        "strict_rewards": strict_rewards,
        "results": result_rows,
        "chunk_count": len(chunk_sizes),
        "chunk_sizes": chunk_sizes,
        "total_attempts": total_attempts,
        "service_version": GENEVAL_EXPECTED_SERVICE_VERSION,
        "protocol_version": GENEVAL_EXPECTED_PROTOCOL_VERSION,
    }


def apply_legacy_monitor_scores(
    trajectories: Sequence[MultiroundTrajectory],
    *,
    r0_scores: Sequence[float],
    final_scores: Sequence[float],
    strict_pass_r0: Sequence[bool],
    strict_pass_final: Sequence[bool],
) -> list[dict[str, Any]]:
    count = len(trajectories)
    if not (
        len(r0_scores)
        == len(final_scores)
        == len(strict_pass_r0)
        == len(strict_pass_final)
        == count
    ):
        raise ValueError("trajectory score vectors are misaligned")
    components = []
    for trajectory, r0_score, final_score, r0_strict, strict_pass in zip(
        trajectories,
        r0_scores,
        final_scores,
        strict_pass_r0,
        strict_pass_final,
    ):
        trajectory.validate()
        reward = legacy_trajectory_reward_metric(
            r0_score=float(r0_score),
            final_score=float(final_score),
            done=trajectory.done,
            r0_strict_pass=bool(r0_strict),
            strict_pass_final=bool(strict_pass),
            round_cap_hit=(
                not trajectory.done
                and trajectory.stop_reason == "round_cap"
            ),
            repair_rounds=trajectory.repair_rounds,
        )
        trajectory.reward = float(reward["total"])
        trajectory.metadata["reward"] = reward
        components.append(reward)
    return components


def _image_jpeg_bytes(image: Any) -> bytes:
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError("GenEval request image must be a PIL image")
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def score_trajectories_geneval(
    trajectories: Sequence[MultiroundTrajectory],
    metadatas: Sequence[dict[str, Any]],
    *,
    url: str = "http://127.0.0.1:18085",
    timeout_sec: float = 180.0,
    session: Any = None,
) -> dict[str, Any]:
    import requests

    if len(trajectories) != len(metadatas) or not trajectories:
        raise ValueError("trajectory/metadata score inputs are misaligned")
    images = [
        *(_image_jpeg_bytes(value.r0_image) for value in trajectories),
        *(_image_jpeg_bytes(value.final_image) for value in trajectories),
    ]
    metadata_rows = [
        *(dict(value) for value in metadatas),
        *(dict(value) for value in metadatas),
    ]
    payload = pickle.dumps(
        {
            "images": images,
            "meta_datas": metadata_rows,
            "only_strict": False,
        }
    )
    client = session or requests.Session()
    response = client.post(url, data=payload, timeout=float(timeout_sec))
    response.raise_for_status()
    result = pickle.loads(response.content)
    scores = [float(value) for value in result.get("scores") or []]
    strict = [bool(value) for value in result.get("strict_rewards") or []]
    expected = 2 * len(trajectories)
    if len(scores) != expected or len(strict) != expected:
        raise RuntimeError("GenEval paired score response has invalid coverage")
    split = len(trajectories)
    reward_components = apply_legacy_monitor_scores(
        trajectories,
        r0_scores=scores[:split],
        final_scores=scores[split:],
        strict_pass_r0=strict[:split],
        strict_pass_final=strict[split:],
    )
    return {
        "r0_scores": scores[:split],
        "final_scores": scores[split:],
        "r0_strict_pass": strict[:split],
        "final_strict_pass": strict[split:],
        "reward_components": reward_components,
        "server_group_rewards": result.get("group_rewards") or {},
        "server_group_strict_rewards": (
            result.get("group_strict_rewards") or {}
        ),
    }


# ---------------------------------------------------------------------------
# G022 item 4: stage-split eta (noise_level). Doc Appendix G-2, G-5, E-1.
#
# R3 splits every stage-dependent knob; we fused them. Our R0 generation and our
# repair edits are exactly R3's gen and edit stages, and R3 runs eta 0.7 / 1.0.
# Appendix G-5: the split is adopted for noise_level specifically because it is
# the ONLY lever that increases within-group diversity without touching K or the
# reward -- and after Appendix H-1 (R0 is already shared across a group) the
# shared-R0 sibling group has just two diversity sources left, text sampling and
# SDE noise inside the window. The edit stage is where the diversity is needed.
#
# G008-G021 read a single `sample.noise_level` and are unaffected: when the
# stage keys are absent this returns exactly that value.
# ---------------------------------------------------------------------------

# Every sampler quantity that determines the recorded SDE log-probability and
# that the learn pass would otherwise re-derive from the config. Audited against
# `accumulate_flow_call`'s call into `interleave_inference` and against
# `train_image_with_grpo`'s reconstruction of `original_timesteps`/`dtimesteps`.
REPLAY_BINDING_KEYS = frozenset(
    {
        "num_timesteps",
        "timestep_shift",
        "cfg_text_scale",
        "cfg_img_scale",
        "cfg_interval",
        "cfg_renorm_min",
        "cfg_renorm_type",
        "resolution",
        "sde_window_size",
        "sde_window_range",
    }
)
REPLAY_BINDING_VERSION = "clean29529_g022_flow_replay_binding_v1"


def capture_replay_bindings(
    grpo_config: Any,
    *,
    num_timesteps: int | None = None,
) -> dict[str, Any]:
    """Snapshot, at rollout time, everything the replay must reproduce.

    C1 / finding Q-6: `num_timesteps` used to be read unconditionally from
    `sample.num_steps`, so after T1.5 an evaluation record generated at 50 steps
    recorded 20 and a report could claim both numbers at once. Callers pass the
    **effective** count they actually handed to the sampler; the default keeps
    the legacy value for G008-G021, whose training and only step count it is.
    """

    return {
        "version": REPLAY_BINDING_VERSION,
        "num_timesteps": (
            int(grpo_config.sample.num_steps)
            if num_timesteps is None
            else int(num_timesteps)
        ),
        "timestep_shift": float(grpo_config.train.timestep_shift),
        "cfg_text_scale": float(grpo_config.sample.guidance_scale),
        "cfg_img_scale": 1.5,
        "cfg_interval": [0.4, 1.0],
        "cfg_renorm_min": 0.0,
        "cfg_renorm_type": "global",
        "resolution": int(grpo_config.resolution),
        "sde_window_size": int(grpo_config.sample.sde_window_size),
        "sde_window_range": list(grpo_config.sample.sde_window_range),
    }


def replay_binding_completeness(record: "FlowCallRecord") -> dict[str, Any]:
    """Is this record bound completely, at the exact binding version?

    A2 / finding Q-7. The gate used to be

        if record.replay_bindings is None and record.noise_level is None:

    -- `and` where it needed `or`. A record carrying `noise_level` and **no
    replay dictionary at all** skipped the "unverified" branch, matched the one
    quantity it had, and was reported `verified: True` for all ten. The
    `version` field was never read. This mechanism exists precisely to make a
    silent rollout/replay mismatch impossible, and it was silently passing.

    Verification is atomic: the complete key set, at the exact version, plus
    both eta fields. Anything else is unverified, with the reason named.
    """

    if record.replay_bindings is None:
        return {"complete": False, "reason": "record carries no replay bindings"}
    version = str(record.replay_bindings.get("version"))
    if version != REPLAY_BINDING_VERSION:
        return {
            "complete": False,
            "reason": (
                f"replay binding version {version!r} is not "
                f"{REPLAY_BINDING_VERSION!r}"
            ),
        }
    missing = sorted(REPLAY_BINDING_KEYS - set(record.replay_bindings))
    if missing:
        return {
            "complete": False,
            "reason": "replay bindings are missing " + ", ".join(missing),
        }
    if record.noise_level is None:
        return {"complete": False, "reason": "record carries no noise_level"}
    if record.eta_clamp_mode is None:
        return {"complete": False, "reason": "record carries no eta_clamp_mode"}
    return {"complete": True, "reason": None}


def assert_replay_bindings(
    record: "FlowCallRecord",
    grpo_config: Any,
    *,
    num_timesteps: int | None = None,
) -> dict[str, Any]:
    """Raise if the replay would use different sampler settings than the rollout.

    Review of item 4: `call_kind` is on the record and read back, so the
    stage cannot drift. Nothing else was, so a resume with a changed config, or
    an offline replay of archived records, would recompute the transition under
    different settings and produce a wrong ratio with no error anywhere.

    A2 / Q-7: an incomplete binding is fail-closed under G022 and reported
    `verified: False` for the legacy stages, which legitimately hold pre-binding
    records. Under G022 every record is produced by the G022 rollout and so is
    always completely bound; an incomplete one means the record came from
    somewhere this replay must not trust.
    """

    stage_eta = stage_noise_level(grpo_config, str(record.call_kind))
    config_mode = str(
        getattr(
            getattr(grpo_config, "g016", None),
            "eta_clamp_mode",
            LEGACY_ETA_CLAMP_MODE,
        )
    )
    completeness = replay_binding_completeness(record)
    if not completeness["complete"]:
        if g022_single_head_mode(grpo_config):
            raise RuntimeError(
                "G022 flow replay bindings are incomplete, so the recorded and "
                "replayed log-probabilities cannot be shown to be the same "
                "distribution: " + str(completeness["reason"])
            )
        # Legacy pre-binding record. Fall back, and say so instead of implying
        # a check that did not happen.
        return {
            "version": REPLAY_BINDING_VERSION,
            "verified": False,
            "reason": completeness["reason"],
            "noise_level": (
                stage_eta if record.noise_level is None else float(record.noise_level)
            ),
            "eta_clamp_mode": (
                config_mode
                if record.eta_clamp_mode is None
                else str(record.eta_clamp_mode)
            ),
        }
    mismatched = []
    if record.noise_level is not None and float(record.noise_level) != float(stage_eta):
        mismatched.append(
            {
                "binding": "noise_level",
                "recorded": float(record.noise_level),
                "config": float(stage_eta),
            }
        )
    if record.eta_clamp_mode is not None and str(record.eta_clamp_mode) != config_mode:
        mismatched.append(
            {
                "binding": "eta_clamp_mode",
                "recorded": str(record.eta_clamp_mode),
                "config": config_mode,
            }
        )
    current = capture_replay_bindings(grpo_config, num_timesteps=num_timesteps)
    for key in sorted(REPLAY_BINDING_KEYS):
        recorded = record.replay_bindings[key]
        live = current[key]
        if isinstance(live, list):
            recorded = list(recorded)
        if recorded != live:
            mismatched.append(
                {"binding": key, "recorded": recorded, "config": live}
            )
    if mismatched:
        raise RuntimeError(
            "flow replay bindings differ from the rollout that produced this "
            "record; the recorded log-probability and the replayed one would "
            "not be the same distribution: "
            + json.dumps(mismatched, default=str, sort_keys=True)
        )
    return {
        "version": REPLAY_BINDING_VERSION,
        "verified": True,
        "reason": None,
        "noise_level": (
            stage_eta if record.noise_level is None else float(record.noise_level)
        ),
        "eta_clamp_mode": (
            config_mode if record.eta_clamp_mode is None else str(record.eta_clamp_mode)
        ),
    }


def active_num_timesteps(rollout: Any) -> int:
    """Denoising steps for the rollout's current mode.

    T1.5 / finding O-2. Every generation call read
    `grpo_config.sample.num_steps`, so under G022 evaluation ran at the TRAINING
    step count (20) while `eval_num_steps = 50` was set and never read by any
    generation call. Appendix G-2 and D-4 both require eval at 50 -- official
    BAGEL trains at 15 and evaluates at 50 -- so every comparison against a
    historical 50-step evaluation would have been invalid.

    `evaluation_num_timesteps` is None during training and set to
    `sample.eval_num_steps` for the duration of an evaluation.
    """

    override = getattr(rollout, "evaluation_num_timesteps", None)
    if override is not None:
        return int(override)
    return int(rollout.grpo_config.sample.num_steps)


GENERATION_STAGE = "r0"
EDIT_STAGE = "repair"


def stage_noise_level(grpo_config: Any, stage: str) -> float:
    """eta for one stage. Falls back to the fused `sample.noise_level`."""

    fused = float(grpo_config.sample.noise_level)
    g016 = getattr(grpo_config, "g016", None)
    key = {
        GENERATION_STAGE: "noise_level_gen",
        EDIT_STAGE: "noise_level_edit",
    }.get(str(stage))
    if key is None:
        raise ValueError(f"unknown flow stage for eta selection: {stage!r}")
    value = getattr(g016, key, None) if g016 is not None else None
    return fused if value is None else float(value)


def clipped_text_ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    *,
    advantage: float | torch.Tensor,
    clip_range: float,
    bound_mode: str = LEGACY_BOUND_MODE,
) -> tuple[torch.Tensor, dict[str, float]]:
    if new_log_probs.shape != old_log_probs.shape or new_log_probs.numel() == 0:
        raise ValueError("text PPO log-prob tensors are empty or misaligned")
    if clip_range <= 0.0:
        raise ValueError("text PPO clip range must be positive")
    old = old_log_probs.to(device=new_log_probs.device, dtype=torch.float32)
    new = new_log_probs.float()
    advantage_tensor = torch.as_tensor(
        advantage,
        device=new.device,
        dtype=torch.float32,
    )
    # Appendix M-2: this clamp is a second instance of the section-5
    # zero-gradient defect -- it bounds the PPO *ratio*, not only the KL.
    # G021 keeps the exact legacy expression; G022 uses the C1 linearization.
    if _use_g022_bound(bound_mode):
        ratio = g022_bounded_ratio(new, old)
    else:
        ratio = torch.exp(torch.clamp(new - old, min=-20.0, max=20.0))
    unclipped = -advantage_tensor * ratio
    clipped = -advantage_tensor * torch.clamp(
        ratio,
        min=1.0 - float(clip_range),
        max=1.0 + float(clip_range),
    )
    loss = torch.maximum(unclipped, clipped).mean()
    diagnostics = {
        "ratio_mean": float(ratio.detach().mean().item()),
        "ratio_max_abs_deviation": float(
            torch.abs(ratio.detach() - 1.0).max().item()
        ),
        "clip_fraction": float(
            (torch.abs(ratio.detach() - 1.0) > float(clip_range))
            .float()
            .mean()
            .item()
        ),
        "token_count": int(new.numel()),
        "ppo_bound_mode": str(bound_mode),
        "ratio_max": float(ratio.detach().max().item()),
    }
    return loss, diagnostics


def text_kl_delta_diagnostics(
    new_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
) -> dict[str, float]:
    """Per-token delta statistics, reported whether or not beta is zero.

    The hard runtime stop of doc section 5 needs `max |delta|` even when the KL
    term itself is switched off, so this never short-circuits on beta.
    """

    if new_log_probs.shape != reference_log_probs.shape:
        raise ValueError("text KL log-prob tensors are misaligned")
    return g022_delta_diagnostics(
        new_log_probs.detach().float() - reference_log_probs.detach().float()
    )


def text_k3_kl_loss(
    new_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    *,
    beta: float,
    bound_mode: str = LEGACY_BOUND_MODE,
) -> torch.Tensor:
    if new_log_probs.shape != reference_log_probs.shape:
        raise ValueError("text KL log-prob tensors are misaligned")
    if beta < 0.0:
        raise ValueError("text KL beta must be non-negative")
    if beta == 0.0:
        return new_log_probs.float().sum() * 0.0
    raw = new_log_probs.float() - reference_log_probs.float()
    if _use_g022_bound(bound_mode):
        # Doc section 5 part 3: never clamp inside the loss. The bound is a C1
        # linearization whose gradient past the threshold is the constant
        # exp(T) - 1, so a divergent token is still pulled back.
        return float(beta) * g022_bounded_k3(raw).mean()
    delta = torch.clamp(raw, min=-20.0, max=20.0)
    return float(beta) * (torch.exp(delta) - 1.0 - delta).mean()


def minimal_text_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    *,
    advantage: float | torch.Tensor,
    clip_range: float,
    kl_beta: float,
    bound_mode: str = LEGACY_BOUND_MODE,
) -> tuple[torch.Tensor, dict[str, float]]:
    ppo, diagnostics = clipped_text_ppo_loss(
        new_log_probs,
        old_log_probs,
        advantage=advantage,
        clip_range=clip_range,
        bound_mode=bound_mode,
    )
    kl = text_k3_kl_loss(
        new_log_probs,
        reference_log_probs,
        beta=kl_beta,
        bound_mode=bound_mode,
    )
    total = ppo + kl
    return total, {
        **diagnostics,
        "policy_loss": float(ppo.detach().item()),
        "kl_loss": float(kl.detach().item()),
        "total_loss": float(total.detach().item()),
        "version": TEXT_POLICY_VERSION,
    }


def sampled_token_ids_from_event(
    event: dict[str, Any],
    *,
    eos_token_id: int,
) -> list[int]:
    raw_content = event.get("raw_content_token_ids")
    raw_log_probs = event.get("raw_selected_log_probs")
    if not isinstance(raw_content, list) or not isinstance(
        raw_log_probs,
        list,
    ):
        raise RuntimeError(
            "controller event token evidence is not a list"
        )
    if (
        type(eos_token_id) is not int
        or eos_token_id < 0
        or any(type(value) is not int or value < 0 for value in raw_content)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in raw_log_probs
        )
    ):
        raise RuntimeError("controller event token evidence is malformed")
    content = list(raw_content)
    if bool(event.get("response_eos_model_selected", False)):
        content.append(int(eos_token_id))
    if not content:
        raise RuntimeError("controller event has no sampled tokens")
    if len(content) != len(raw_log_probs):
        raise RuntimeError("controller event token/log-prob lengths differ")
    return content


def sampled_token_constraints_from_event(
    event: dict[str, Any],
    *,
    sampled_token_count: int,
) -> tuple[list[list[int] | None] | None, list[bool] | None]:
    """Align the sampler's constraint/credit masks with sampled tokens.

    A host-forced constrained position (single legal token) never receives
    PPO or KL credit; a branching constrained position keeps credit with its
    exact allowed set. Events without a constrained schedule return
    `(None, None)` and behave exactly as before.
    """
    allowed_ids = event.get("raw_content_allowed_ids")
    policy_active = event.get("raw_content_policy_active")
    if allowed_ids is None and policy_active is None:
        return None, None
    if not isinstance(allowed_ids, list) or not isinstance(
        policy_active,
        list,
    ):
        raise RuntimeError("controller constraint evidence is not a list")
    content_count = len(event.get("raw_content_token_ids") or [])
    if len(allowed_ids) != content_count or len(policy_active) != (
        content_count
    ):
        raise RuntimeError("controller constraint evidence length differs")
    aligned_allowed: list[list[int] | None] = []
    aligned_active: list[bool] = []
    for allowed, active in zip(allowed_ids, policy_active):
        if allowed is None:
            aligned_allowed.append(None)
        else:
            values = [int(value) for value in allowed]
            if not values or len(set(values)) != len(values):
                raise RuntimeError("controller constraint set is malformed")
            aligned_allowed.append(values)
        if not isinstance(active, bool):
            raise RuntimeError("controller credit mask must be boolean")
        aligned_active.append(bool(active))
    while len(aligned_allowed) < int(sampled_token_count):
        aligned_allowed.append(None)
        aligned_active.append(True)
    if len(aligned_allowed) != int(sampled_token_count):
        raise RuntimeError("controller constraint evidence exceeds tokens")
    return aligned_allowed, aligned_active


def malformed_text_turn_from_event(
    event: dict[str, Any],
    *,
    context_terms: Sequence[Any],
    eos_token_id: int,
    preserve_host_forced_mask: bool = False,
) -> TextTurnRecord:
    """Build the TextTurnRecord for a controller response that failed to parse.

    A1 / finding Q-3. This is the **production entry point** for a malformed
    G022 turn; `build_g022_token_masks` is downstream of it. T1.4 excluded
    host-forced tokens inside the router, and this function then undid that
    exclusion one layer up by flipping an all-host-forced mask to all-True, so
    the malformed penalty reached tokens the model never chose. The T1.4 tests
    did not see it because they entered at the router.

    `preserve_host_forced_mask` is the G022 path: the recorded mask is passed
    through untouched. When every token of a malformed response was host-forced
    there is genuinely nothing the model chose, so nothing to penalise, and that
    zero is correct -- the `-0.5` still lives in `R(tau)` and reaches the
    trajectory's other rounds. G008-G021 keep the legacy flip and are
    byte-unchanged.
    """

    try:
        sampled_token_ids = sampled_token_ids_from_event(
            event,
            eos_token_id=eos_token_id,
        )
    except Exception as exc:
        raise RuntimeError(
            "G008 malformed controller token evidence"
        ) from exc
    allowed_ids, policy_active = sampled_token_constraints_from_event(
        event,
        sampled_token_count=len(sampled_token_ids),
    )
    if (
        not preserve_host_forced_mask
        and policy_active is not None
        and not any(policy_active)
    ):
        # A malformed turn still receives its trajectory-level standard-GRPO
        # advantage on sampled policy tokens.
        policy_active = [True] * len(sampled_token_ids)
    turn = TextTurnRecord(
        turn_index=int(event["turn_index"]),
        context_terms=[
            value.copy() if hasattr(value, "copy") else str(value)
            for value in context_terms
        ],
        forced_prefix_token_ids=list(
            event["raw_forced_prefix_token_ids"]
        ),
        sampled_token_ids=sampled_token_ids,
        old_log_probs=list(event["raw_selected_log_probs"]),
        behavior_temperature=float(event["behavior_temperature"]),
        canonical_response=str(event["raw_response"]),
        action="invalid",
        policy_active=True,
        valid=False,
        local_advantage=None,
        sampled_token_allowed_ids=allowed_ids,
        sampled_token_policy_active=policy_active,
    )
    turn.validate()
    return turn


class SharedPrefixReference(torch.nn.Module):
    """Frozen last-four reference using the policy's shared frozen prefix."""

    def __init__(
        self,
        policy_language_model: Any,
        *,
        frozen_decoder_layers: Sequence[torch.nn.Module],
        decoder_layer_start: int,
        initialization: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.frozen_decoder_layers = torch.nn.ModuleList(
            list(frozen_decoder_layers)
        )
        self.decoder_layer_start = int(decoder_layer_start)
        self.initialization = dict(initialization)
        object.__setattr__(
            self,
            "_policy_language_model",
            policy_language_model,
        )
        self.config = policy_language_model.config
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        del mode
        return super().train(False)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("mode") in {"get_embeddings", "get_logits"}:
            with torch.no_grad():
                return self._policy_language_model.forward(*args, **kwargs)
        if kwargs.get("mode") == "collective_noop":
            raise RuntimeError("frozen G008 reference cannot run backward")
        with torch.no_grad():
            return self._policy_language_model.forward(
                *args,
                **kwargs,
                reference_decoder_layers=self.frozen_decoder_layers,
                reference_decoder_layer_start=self.decoder_layer_start,
            )


def assert_frozen_reference_optimizer_isolation(
    reference: torch.nn.Module,
    *,
    optimizers: Sequence[torch.optim.Optimizer],
) -> dict[str, Any]:
    reference_named = list(reference.named_parameters())
    reference_ids = {id(parameter) for _name, parameter in reference_named}
    optimizer_parameters = [
        parameter
        for optimizer in optimizers
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    overlap = reference_ids.intersection(
        id(parameter) for parameter in optimizer_parameters
    )
    trainable = [
        name
        for name, parameter in reference_named
        if parameter.requires_grad
    ]
    if not reference_named or trainable or overlap:
        raise RuntimeError(
            "G008 frozen reference optimizer isolation differs: "
            f"reference_parameter_count={len(reference_named)} "
            f"trainable={trainable[:5]} overlap_count={len(overlap)}"
        )
    return {
        "version": "clean29529_v20_reference_optimizer_isolation_v1",
        "reference_parameter_count": len(reference_named),
        "reference_parameter_numel": sum(
            int(parameter.numel())
            for _name, parameter in reference_named
        ),
        "all_reference_parameters_frozen": True,
        "optimizer_parameter_count": len(optimizer_parameters),
        "reference_optimizer_overlap_count": 0,
    }


def replay_controller_context(
    inferencer: Any,
    context_terms: Sequence[Any],
) -> Any:
    from PIL import Image

    context = inferencer.init_gen_context()
    for term in context_terms:
        if isinstance(term, str):
            context = inferencer.update_context_text(term, context)
        elif isinstance(term, Image.Image):
            context = inferencer.update_context_image(
                term,
                context,
                vae=True,
                vit=True,
            )
        else:
            raise TypeError(f"unsupported controller context term: {type(term)}")
    return context


def teacher_forced_text_log_probs(
    inferencer: Any,
    *,
    context_terms: Sequence[Any],
    forced_prefix_token_ids: Sequence[int],
    sampled_token_ids: Sequence[int],
    temperature: float,
    sampled_token_allowed_ids: Sequence[Sequence[int] | None] | None = None,
    text_action_support: Any = None,
) -> torch.Tensor:
    sampled = [int(value) for value in sampled_token_ids]
    prefix = [int(value) for value in forced_prefix_token_ids]
    if not sampled:
        raise ValueError("teacher-forced text replay requires sampled tokens")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("teacher-forced text temperature must be positive")
    if sampled_token_allowed_ids is not None and len(
        sampled_token_allowed_ids
    ) != len(sampled):
        raise ValueError("teacher-forced constraint mask length differs")
    context = replay_controller_context(inferencer, context_terms)
    if len(context["kv_lens"]) != 1 or len(context["ropes"]) != 1:
        raise NotImplementedError("G008 text replay supports batch size one")
    model = inferencer.model
    wrapped_language_model = model.language_model
    device = model.vae2llm.weight.device
    old_kv_len = int(context["kv_lens"][0])
    old_rope = int(context["ropes"][0])
    bos_token_id = int(inferencer.new_token_ids["bos_token_id"])
    query_token_ids = [
        bos_token_id,
        *prefix,
        *sampled[:-1],
    ]
    query_count = len(query_token_ids)
    packed_text_ids = torch.tensor(
        query_token_ids,
        device=device,
        dtype=torch.long,
    )
    packed_text_embedding = wrapped_language_model.forward(
        mode="get_embeddings",
        input_ids=packed_text_ids,
    )
    extra_inputs = {"mode": "und"} if getattr(model, "use_moe", False) else {}
    output = wrapped_language_model(
        packed_query_sequence=packed_text_embedding,
        query_lens=torch.tensor(
            [query_count],
            device=device,
            dtype=torch.int,
        ),
        packed_query_position_ids=torch.arange(
            old_rope,
            old_rope + query_count,
            device=device,
            dtype=torch.long,
        ),
        packed_query_indexes=torch.arange(
            old_kv_len,
            old_kv_len + query_count,
            device=device,
            dtype=torch.long,
        ),
        past_key_values=context["past_key_values"],
        key_values_lens=torch.tensor(
            [old_kv_len],
            device=device,
            dtype=torch.int,
        ),
        packed_key_value_indexes=torch.arange(
            old_kv_len,
            device=device,
            dtype=torch.long,
        ),
        update_past_key_values=False,
        is_causal=True,
        **extra_inputs,
    )
    logits = wrapped_language_model.forward(
        mode="get_logits",
        hidden_states=output.packed_query_sequence,
    )
    first_sample_logit = len(prefix)
    selected_logits = logits[
        first_sample_logit : first_sample_logit + len(sampled)
    ]
    if selected_logits.shape[0] != len(sampled):
        raise RuntimeError("teacher-forced text replay lost sampled positions")
    # The replay must live in the sampler's probability space, or the PPO
    # ratio and the KL are taken between two different distributions.  The
    # support contract is read off the inferencer so the policy and the
    # frozen reference cannot diverge by a forgotten argument.
    if text_action_support is None:
        text_action_support = getattr(
            inferencer, "g022_text_action_support", None
        )
    if text_action_support is not None:
        selected_logits = text_action_support.mask_logits(selected_logits)
    if sampled_token_allowed_ids is not None and any(
        value is not None for value in sampled_token_allowed_ids
    ):
        masked = selected_logits.float() / float(temperature)
        rows = []
        # Constraint-aware replay uses the identical estimator the sampler
        # used, so old/current/reference ratios stay exact.
        for position, allowed in enumerate(sampled_token_allowed_ids):
            if allowed is None:
                rows.append(
                    torch.nn.functional.log_softmax(masked[position], dim=-1)
                )
                continue
            index = torch.tensor(
                [int(value) for value in allowed],
                device=masked.device,
                dtype=torch.long,
            )
            row = torch.full_like(masked[position], float("-inf"))
            row = row.index_copy(
                0,
                index,
                masked[position].index_select(0, index),
            )
            rows.append(torch.nn.functional.log_softmax(row, dim=-1))
        log_probs = torch.stack(rows, dim=0)
    else:
        log_probs = torch.nn.functional.log_softmax(
            selected_logits.float() / float(temperature),
            dim=-1,
        )
    targets = torch.tensor(sampled, device=device, dtype=torch.long)
    return log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)


def _disabled_controller_action_token_index(
    inferencer: Any,
    turn: TextTurnRecord,
) -> int | None:
    del inferencer, turn
    return None


def controller_action_token_index(
    inferencer: Any,
    turn: TextTurnRecord,
) -> int | None:
    if not turn.valid or turn.action not in {"edit", "done"}:
        return None
    try:
        token_ids = inferencer.tokenizer.encode(
            f" {turn.action}", add_special_tokens=False
        )
    except TypeError:
        token_ids = inferencer.tokenizer.encode(f" {turn.action}")
    if len(token_ids) != 1:
        raise RuntimeError(
            f"G008 G002 ACTION value is not one token: {turn.action} {token_ids}"
        )
    target = int(token_ids[0])
    matches = [
        index
        for index, token_id in enumerate(turn.sampled_token_ids[:16])
        if int(token_id) == target
    ]
    if len(matches) == 1:
        return matches[0]
    if turn.action == "done":
        # Existing P2 semantics route sampled EDIT+[EDIT] None to DONE. Do not
        # give direct DONE auxiliary credit to the sampled EDIT token; this is
        # exactly the alias contamination tracked by edit_none_alias_rate.
        try:
            edit_token = inferencer.tokenizer.encode(
                " edit", add_special_tokens=False
            )
        except TypeError:
            edit_token = inferencer.tokenizer.encode(" edit")
        if len(edit_token) == 1 and any(
            int(token_id) == int(edit_token[0])
            for token_id in turn.sampled_token_ids[:16]
        ):
            return None
    raise RuntimeError(
        "G008 G002 external ACTION token alignment differs: "
        f"action={turn.action} matches={matches}"
    )


def text_turn_policy_loss(
    *,
    policy_inferencer: Any,
    reference_inferencer: Any,
    turn: TextTurnRecord,
    clip_range: float,
    kl_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    turn.validate()
    action_token_original_index = controller_action_token_index(
        policy_inferencer, turn
    )
    new_log_probs = teacher_forced_text_log_probs(
        policy_inferencer,
        context_terms=turn.context_terms,
        forced_prefix_token_ids=turn.forced_prefix_token_ids,
        sampled_token_ids=turn.sampled_token_ids,
        temperature=turn.behavior_temperature,
        sampled_token_allowed_ids=turn.sampled_token_allowed_ids,
    )
    with torch.no_grad():
        reference_log_probs = teacher_forced_text_log_probs(
            reference_inferencer,
            context_terms=turn.context_terms,
            forced_prefix_token_ids=turn.forced_prefix_token_ids,
            sampled_token_ids=turn.sampled_token_ids,
            temperature=turn.behavior_temperature,
            sampled_token_allowed_ids=turn.sampled_token_allowed_ids,
        )
    if not turn.policy_active:
        zero = new_log_probs.float().sum() * 0.0
        return zero, {
            "ratio_mean": 1.0,
            "ratio_max_abs_deviation": 0.0,
            "clip_fraction": 0.0,
            "token_count": 0,
            "policy_loss": 0.0,
            "kl_loss": 0.0,
            "total_loss": 0.0,
            "padding": True,
            "padding_version": PADDING_VERSION,
            "version": TEXT_POLICY_VERSION,
            "reference_log_probs_byte_identical": torch.equal(
                new_log_probs.detach(),
                reference_log_probs.detach(),
            ),
            "reference_log_probs_max_abs_difference": float(
                (
                    new_log_probs.detach().float()
                    - reference_log_probs.detach().float()
                )
                .abs()
                .max()
                .item()
            ),
        }
    effective_advantage = (
        float(turn.local_advantage)
        if turn.local_advantage is not None
        else float(turn.advantage)
    )
    # G008 has exactly one text inner epoch and no optimizer step before this
    # forward. Use the detached same-forward log-prob as the first-epoch old
    # policy: ratio is mathematically 1 while gradients still flow through
    # `new_log_probs`. Separate repeated packed forwards were non-deterministic
    # under real FSDP/Flash (max deviation 1.03 with unchanged weights), so
    # treating that kernel drift as policy change made the 1e-3 guard unusable.
    old_log_probs = new_log_probs.detach().float()
    behavior_old_log_probs = torch.tensor(
        turn.old_log_probs,
        device=new_log_probs.device,
        dtype=torch.float32,
    )
    credited = turn.credited_token_mask()
    credited_indexes = [
        index for index, value in enumerate(credited) if value
    ]
    if (
        action_token_original_index is not None
        and action_token_original_index not in credited_indexes
    ):
        raise RuntimeError("G008 G002 ACTION token is outside PPO credit")
    action_token_index = (
        credited_indexes.index(action_token_original_index)
        if action_token_original_index is not None
        else None
    )
    host_token_count = int(sum(1 for value in credited if not value))
    if host_token_count:
        keep = torch.tensor(
            credited_indexes,
            device=new_log_probs.device,
            dtype=torch.long,
        )
        new_log_probs = new_log_probs.index_select(0, keep)
        reference_log_probs = reference_log_probs.index_select(0, keep)
        old_log_probs = old_log_probs.index_select(0, keep)
        behavior_old_log_probs = behavior_old_log_probs.index_select(0, keep)
    behavior_ratio = torch.exp(
        old_log_probs.detach() - behavior_old_log_probs.detach()
    )
    loss, diagnostics = minimal_text_policy_loss(
        new_log_probs,
        old_log_probs,
        reference_log_probs,
        advantage=effective_advantage,
        clip_range=clip_range,
        kl_beta=kl_beta,
    )
    action_aux_loss = new_log_probs.float().sum() * 0.0
    action_aux_diagnostics = {
        "active": False,
        "token_count": 0,
        "policy_loss": 0.0,
    }
    if CONTROLLER_ACTION_AUX_WEIGHT and action_token_index is not None:
        action_selector = torch.tensor(
            [action_token_index],
            device=new_log_probs.device,
            dtype=torch.long,
        )
        action_aux_loss, action_raw = clipped_text_ppo_loss(
            new_log_probs.index_select(0, action_selector),
            old_log_probs.index_select(0, action_selector),
            advantage=effective_advantage,
            clip_range=clip_range,
        )
        action_aux_diagnostics = {
            "active": True,
            "token_count": 1,
            "policy_loss": float(action_aux_loss.detach().item()),
            "ratio_mean": action_raw["ratio_mean"],
            "ratio_max_abs_deviation": action_raw[
                "ratio_max_abs_deviation"
            ],
            "clip_fraction": action_raw["clip_fraction"],
        }
    total_loss = loss + CONTROLLER_ACTION_AUX_WEIGHT * action_aux_loss
    return total_loss, {
        **diagnostics,
        "total_loss": float(total_loss.detach().item()),
        "effective_advantage": float(effective_advantage),
        "controller_action_aux": {
            "version": CONTROLLER_ACTION_AUX_VERSION,
            "weight": CONTROLLER_ACTION_AUX_WEIGHT,
            "action": turn.action,
            **action_aux_diagnostics,
        },
        "host_forced_token_count": host_token_count,
        "ppo_old_log_prob_source": (
            "first_inner_same_forward_detached"
        ),
        "sampler_vs_packed_ratio_max_abs_deviation": float(
            (behavior_ratio.float() - 1.0).abs().max().item()
        ),
        "reference_log_probs_byte_identical": torch.equal(
            new_log_probs.detach(),
            reference_log_probs.detach(),
        ),
        "reference_log_probs_max_abs_difference": float(
            (
                new_log_probs.detach().float()
                - reference_log_probs.detach().float()
            )
            .abs()
            .max()
            .item()
        ),
    }


def build_flow_contexts(
    inferencer: Any,
    input_terms: Sequence[Any],
) -> dict[str, Any]:
    from PIL import Image
    from flow_grpo.bagel.data.data_utils import pil_img2rgb

    gen_context = inferencer.init_gen_context()
    cfg_text_context = deepcopy(gen_context)
    cfg_img_context = deepcopy(gen_context)
    for term in input_terms:
        if isinstance(term, str):
            cfg_text_context = deepcopy(gen_context)
            gen_context = inferencer.update_context_text(term, gen_context)
            cfg_img_context = inferencer.update_context_text(
                term,
                cfg_img_context,
            )
        elif isinstance(term, Image.Image):
            resized = inferencer.vae_transform.resize_transform(
                pil_img2rgb(term)
            )
            gen_context = inferencer.update_context_image(
                resized,
                gen_context,
                vae=True,
                vit=True,
            )
            cfg_text_context = deepcopy(gen_context)
        else:
            raise TypeError(f"unsupported flow context term: {type(term)}")
    return {
        "past_key_values": gen_context["past_key_values"],
        "cfg_text_past_key_values": cfg_text_context["past_key_values"],
        "cfg_img_past_key_values": cfg_img_context["past_key_values"],
    }


def move_context_cache(
    context: dict[str, Any],
    device: torch.device | str,
) -> dict[str, dict[int, torch.device]]:
    cache = context.get("past_key_values")
    if cache is None:
        return {}
    original_devices: dict[str, dict[int, torch.device]] = {
        "key_cache": {},
        "value_cache": {},
    }
    for cache_name in ("key_cache", "value_cache"):
        values = getattr(cache, cache_name)
        for layer_index, value in values.items():
            if value is None:
                continue
            original_devices[cache_name][layer_index] = value.device
            values[layer_index] = value.to(
                device=device,
                non_blocking=False,
            )
    return original_devices


def restore_context_cache(
    context: dict[str, Any],
    original_devices: dict[str, dict[int, torch.device]],
) -> None:
    cache = context.get("past_key_values")
    if cache is None:
        if original_devices:
            raise ValueError("controller cache disappeared during offload")
        return
    for cache_name, device_by_layer in original_devices.items():
        values = getattr(cache, cache_name)
        for layer_index, device in device_by_layer.items():
            value = values[layer_index]
            if value is None:
                raise ValueError(
                    f"controller {cache_name}[{layer_index}] disappeared"
                )
            values[layer_index] = value.to(
                device=device,
                non_blocking=False,
            )


def clip_trainable_gradients(
    *,
    accelerator: Any,
    parameters: Sequence[torch.nn.Parameter],
    max_grad_norm: float,
) -> float | None:
    if max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    trainable = tuple(
        parameter for parameter in parameters if parameter.requires_grad
    )
    if not trainable:
        raise ValueError("gradient clipping requires trainable parameters")
    if not bool(getattr(accelerator, "sync_gradients", True)):
        return None
    norm = accelerator.clip_grad_norm_(trainable, float(max_grad_norm))
    if torch.is_tensor(norm):
        return float(norm.detach().float().item())
    return float(norm)


def collective_padding_backward(
    *,
    transformer: Any,
    accelerator: Any,
) -> None:
    sentinel = torch.zeros(
        (),
        device=accelerator.device,
        dtype=torch.float32,
        requires_grad=True,
    )
    loss = transformer(mode="collective_noop", sentinel=sentinel)
    accelerator.backward(loss)


def flow_channel_weight(call_kind: str) -> float:
    """R2 flow channel weight: `0.25 * L_r0 + 1.0 * L_repair`."""
    kind = str(call_kind)
    if kind == "r0":
        return float(R0_CHANNEL_WEIGHT)
    if kind == "repair":
        return float(REPAIR_CHANNEL_WEIGHT)
    raise ValueError(f"unknown G008 flow call kind: {call_kind!r}")


def accumulate_flow_call(
    *,
    flow_inferencer: Any,
    reference_flow_inferencer: Any,
    record: FlowCallRecord,
    grpo_config: Any,
    accelerator: Any,
    transformer: Any,
    backward_loss_scale: float,
    max_grad_norm: float = 1.0,
) -> dict[str, Any]:
    record.validate()
    # Review of item 4. Verify, before anything else, that the replay is
    # about to use the same sampler settings the rollout recorded under. A
    # mismatch here is silent otherwise: a wrong ratio, no error.
    replay = assert_replay_bindings(record, grpo_config)
    # Padding must execute the exact same policy/reference Flow-GRPO path as
    # an active record. `train_image_with_grpo` consumes
    # sample["policy_active"], performs policy `_forward_flow` and frozen
    # reference `_forward_flow` at every timestep, zeros policy/KL/loss when
    # inactive, and then backward()s the exact zero. The old early-return ran
    # only `collective_noop`, consumed fewer full-layer all-gathers, and made
    # short ranks leave the FSDP sequence early.
    language_module = flow_inferencer.model.language_model
    while hasattr(language_module, "module"):
        language_module = language_module.module
    language_base = (
        language_module.get_base_model()
        if hasattr(language_module, "get_base_model")
        else language_module
    )
    language_base.model.embed_tokens.to(
        device=accelerator.device, dtype=torch.bfloat16
    )
    flow_inferencer.model.connector.to(
        device=accelerator.device, dtype=torch.bfloat16
    )
    flow_inferencer.model.vit_model.to(
        device=accelerator.device, dtype=torch.bfloat16
    )
    flow_inferencer.vae_model.to(
        device=accelerator.device, dtype=torch.bfloat16
    )
    reference_contexts = build_flow_contexts(
        reference_flow_inferencer,
        record.input_terms,
    )
    outputs = flow_inferencer.interleave_inference(
        record.input_terms,
        think=False,
        understanding_output=False,
        cfg_text_scale=float(grpo_config.sample.guidance_scale),
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=float(grpo_config.train.timestep_shift),
        num_timesteps=int(grpo_config.sample.num_steps),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(
            int(grpo_config.resolution),
            int(grpo_config.resolution),
        ),
        learn=True,
        sample=record.training_sample(device=accelerator.device),
        eta_clamp_mode=replay["eta_clamp_mode"],
        grpo_config=grpo_config,
        accelerator=accelerator,
        optimizer=None,
        transformer=transformer,
        gradient_clip_parameters=(),
        max_grad_norm=float(max_grad_norm),
        noise_level=float(replay["noise_level"]),
        generators=None,
        reference_contexts=reference_contexts,
        isolate_regular_gradients=True,
        perform_optimizer_step=False,
        backward_loss_scale=float(backward_loss_scale),
    )
    if len(outputs) != 1 or not isinstance(outputs[0], dict):
        raise RuntimeError("Flow-GRPO learn call returned an invalid payload")
    result = outputs[0]
    required = {
        "policy_loss",
        "kl_loss",
        "loss",
        "clipfrac",
        "flow_kl_version",
        "velocity_mse",
        "transition_mean_shift_mse",
        "transition_variance",
        "weighted_kl_loss",
        "grad_norm",
        "max_grad_norm",
        "perform_optimizer_step",
        "backward_loss_scale",
        "reference_velocity_all_byte_identical",
        "reference_velocity_max_abs_difference",
    }
    if not required.issubset(result):
        raise RuntimeError("Flow-GRPO learn call omitted loss diagnostics")
    if result["perform_optimizer_step"] is not False:
        raise RuntimeError("G008 flow accumulation stepped internally")
    result["flow_channel_weight"] = (
        flow_channel_weight(record.call_kind) if record.policy_active else 0.0
    )
    result["flow_channel_weight_version"] = CHANNEL_WEIGHT_VERSION
    result["call_kind"] = str(record.call_kind)
    result["replay_bindings_verified"] = bool(replay["verified"])
    result["replay_noise_level"] = float(replay["noise_level"])
    result["replay_eta_clamp_mode"] = str(replay["eta_clamp_mode"])
    result["collective_only_padding"] = not record.policy_active
    result["real_policy_reference_padding"] = not record.policy_active
    result["collective_noop_count"] = 0
    if not record.policy_active and any(
        abs(float(result.get(name, 0.0))) > 1e-12
        for name in ("policy_loss", "kl_loss", "loss")
    ):
        raise RuntimeError("G008 inactive flow real-path loss is nonzero")
    return result


def accumulate_text_turn(
    *,
    policy_inferencer: Any,
    reference_inferencer: Any,
    turn: TextTurnRecord,
    accelerator: Any,
    clip_range: float,
    kl_beta: float,
    backward_loss_scale: float,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    # `text_turn_policy_loss` already executes policy teacher-forced FSDP
    # replay and frozen-reference FSDP replay before returning an exact-zero
    # loss for an inactive turn. Always use it. The old early-return executed
    # one collective-noop forward/backward and skipped both real forwards,
    # causing variable-turn ranks to consume different FSDP all-gathers.
    loss, diagnostics = text_turn_policy_loss(
        policy_inferencer=policy_inferencer,
        reference_inferencer=reference_inferencer,
        turn=turn,
        clip_range=clip_range,
        kl_beta=kl_beta,
    )
    accelerator.backward(
        loss * float(backward_loss_scale)
    )
    return {
        **diagnostics,
        "collective_only_padding": not turn.policy_active,
        "real_policy_reference_padding": not turn.policy_active,
        "collective_noop_count": 0,
        "grad_norm": 0.0,
        "max_grad_norm": float(max_grad_norm),
        "backward_loss_scale": float(backward_loss_scale),
    }


def trajectory_channel_scales(
    local_record_counts: Sequence[int],
    *,
    device: torch.device,
    process_group: Any,
) -> dict[str, Any]:
    counts = [int(value) for value in local_record_counts]
    if any(value < 0 for value in counts):
        raise ValueError("G008 trajectory record counts cannot be negative")
    active = torch.tensor(
        sum(value > 0 for value in counts),
        dtype=torch.int64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(active, op=dist.ReduceOp.SUM, group=process_group)
        world_size = dist.get_world_size(group=process_group)
    else:
        world_size = 1
    global_active_trajectory_count = int(active.item())
    backward_scales = [
        (
            float(world_size)
            / float(global_active_trajectory_count)
            / float(record_count)
            if record_count > 0 and global_active_trajectory_count > 0
            else 0.0
        )
        for record_count in counts
    ]
    effective_record_weights = [
        (
            1.0
            / float(global_active_trajectory_count)
            / float(record_count)
            if record_count > 0 and global_active_trajectory_count > 0
            else 0.0
        )
        for record_count in counts
    ]
    return {
        "local_record_counts": counts,
        "local_active_trajectory_count": sum(value > 0 for value in counts),
        "global_active_trajectory_count": (
            global_active_trajectory_count
        ),
        "backward_scales": backward_scales,
        "world_size": int(world_size),
        "effective_record_weights": effective_record_weights,
        "effective_trajectory_weights": [
            (
                float(record_count) * weight
                if record_count > 0
                else 0.0
            )
            for record_count, weight in zip(
                counts,
                effective_record_weights,
            )
        ],
        "global_channel_weight": (
            1.0 if global_active_trajectory_count > 0 else 0.0
        ),
    }


def synchronized_pre_step_gate(
    *,
    channel: str,
    metrics: Sequence[Sequence[dict[str, Any]]],
    named_parameters: Sequence[
        tuple[str, torch.nn.Parameter]
    ],
    globally_active: bool,
    device: torch.device,
    process_group: Any,
) -> dict[str, Any]:
    if channel == "text":
        metric_key = "ratio_max_abs_deviation"
        limit = TEXT_RATIO_LIMIT
    elif channel == "flow":
        metric_key = "kl_loss"
        limit = FLOW_KL_LIMIT
    else:
        raise ValueError("G008 pre-step gate channel is invalid")
    active_metrics = [
        metric
        for trajectory_metrics in metrics
        for metric in trajectory_metrics
        if not bool(metric.get("collective_only_padding"))
    ]
    nonfinite_paths = nonfinite_numeric_paths(
        active_metrics,
        prefix=f"{channel}_metrics",
    )
    values = []
    missing_required_count = 0
    for metric in active_metrics:
        if metric_key not in metric:
            missing_required_count += 1
            continue
        value = float(metric[metric_key])
        if math.isfinite(value):
            values.append(value)
    nonfinite_gradient_tensors = []
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is not None and not bool(
            torch.isfinite(gradient.detach()).all().item()
        ):
            nonfinite_gradient_tensors.append(str(name))
    local = torch.tensor(
        [
            len(nonfinite_paths),
            missing_required_count,
            len(nonfinite_gradient_tensors),
            len(active_metrics),
        ],
        dtype=torch.int64,
        device=device,
    )
    maximum = torch.tensor(
        max(values, default=0.0),
        dtype=torch.float64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local, op=dist.ReduceOp.SUM, group=process_group)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=process_group)
    report = {
        "version": "clean29529_v20_pre_step_numerical_gate_v1",
        "channel": channel,
        "metric_key": metric_key,
        "limit": float(limit),
        "global_nonfinite_metric_count": int(local[0].item()),
        "global_missing_required_metric_count": int(local[1].item()),
        "global_nonfinite_gradient_tensor_count": int(local[2].item()),
        "global_active_metric_count": int(local[3].item()),
        "global_maximum": float(maximum.item()),
        "local_nonfinite_metric_paths": nonfinite_paths,
        "local_nonfinite_gradient_tensors": nonfinite_gradient_tensors,
        "globally_active": bool(globally_active),
    }
    report["passed"] = (
        report["global_nonfinite_metric_count"] == 0
        and report["global_missing_required_metric_count"] == 0
        and report["global_nonfinite_gradient_tensor_count"] == 0
        and (
            not globally_active
            or report["global_active_metric_count"] > 0
        )
        and report["global_maximum"] <= float(limit)
    )
    if not report["passed"]:
        raise RuntimeError(
            "G008 pre-step numerical gate failed: "
            + json.dumps(report, sort_keys=True)
        )
    return report


def synchronized_post_clip_gate(
    *,
    channel: str,
    grad_norm: float | None,
    named_parameters: Sequence[
        tuple[str, torch.nn.Parameter]
    ],
    globally_active: bool,
    device: torch.device,
    process_group: Any,
) -> dict[str, Any]:
    local_nonfinite_norm = int(
        globally_active
        and (
            grad_norm is None
            or not math.isfinite(float(grad_norm))
        )
    )
    local_nonfinite_gradients = [
        str(name)
        for name, parameter in named_parameters
        if parameter.grad is not None
        and not bool(torch.isfinite(parameter.grad.detach()).all().item())
    ]
    counts = torch.tensor(
        [
            local_nonfinite_norm,
            len(local_nonfinite_gradients),
        ],
        dtype=torch.int64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=process_group)
    report = {
        "version": "clean29529_v20_post_clip_numerical_gate_v1",
        "channel": channel,
        "globally_active": bool(globally_active),
        "grad_norm": None if grad_norm is None else float(grad_norm),
        "global_nonfinite_grad_norm_count": int(counts[0].item()),
        "global_nonfinite_clipped_gradient_tensor_count": int(
            counts[1].item()
        ),
        "local_nonfinite_clipped_gradient_tensors": (
            local_nonfinite_gradients
        ),
    }
    report["passed"] = (
        report["global_nonfinite_grad_norm_count"] == 0
        and report[
            "global_nonfinite_clipped_gradient_tensor_count"
        ]
        == 0
    )
    if not report["passed"]:
        raise RuntimeError(
            "G008 post-clip numerical gate failed: "
            + json.dumps(report, sort_keys=True)
        )
    return report


def update_multiround_group(
    trajectories: Sequence[MultiroundTrajectory],
    *,
    r0_records: Sequence[FlowCallRecord],
    flow_inferencer: Any,
    reference_flow_inferencer: Any,
    policy_controller_inferencer: Any,
    reference_controller_inferencer: Any,
    grpo_config: Any,
    accelerator: Any,
    text_optimizer: Any,
    flow_optimizer: Any,
    transformer: Any,
    text_clip_range: float,
    text_kl_beta: float,
    flow_parameters: Sequence[torch.nn.Parameter],
    text_parameters: Sequence[torch.nn.Parameter],
    max_grad_norm: float,
    max_flow_calls: int,
    max_text_turns: int,
    process_group: Any = None,
    commit_optimizer_step: bool = True,
    active_channel_mode: str = "both",
    optimizer_step_mode: str = "both",
    capture_full_diagnostics: bool = False,
    text_parameter_names: Sequence[str] | None = None,
    flow_parameter_names: Sequence[str] | None = None,
    phase_observer: Callable[[str], None] | None = None,
    optimizer_step_observer: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """One text-before-flow update with separate original-R0 accounting.

    ``r0_records`` contains each of the 28 original same-prompt R0 calls once
    across the world.  Repair trajectories contain no R0 call at all.  The
    flow objective is ``0.25 * mean(L_r0) + mean_per_trajectory(L_repair)``;
    both contributions are accumulated before the single flow optimizer step.
    """
    rows = list(trajectories)
    original_r0 = list(r0_records)
    if not rows or not original_r0:
        raise ValueError("G008 update requires local R0 and repair records")
    if any(record.call_kind != "r0" for record in original_r0):
        raise ValueError("G008 original R0 channel contains a repair call")
    for record in original_r0:
        record.validate()
    for trajectory in rows:
        trajectory.validate()
    modes = {"none", "text", "flow", "both"}
    if active_channel_mode not in modes or optimizer_step_mode not in modes:
        raise ValueError("G008 optimizer channel mode is invalid")
    effective_step_mode = optimizer_step_mode if commit_optimizer_step else "none"
    text_enabled = active_channel_mode in {"text", "both"}
    flow_enabled = active_channel_mode in {"flow", "both"}
    if effective_step_mode in {"text", "both"} and not text_enabled:
        raise ValueError("G008 text step enabled while text channel is inactive")
    if effective_step_mode in {"flow", "both"} and not flow_enabled:
        raise ValueError("G008 flow step enabled while flow channel is inactive")

    text_names = list(text_parameter_names or [f"text_{i}" for i in range(len(text_parameters))])
    flow_names = list(flow_parameter_names or [f"flow_{i}" for i in range(len(flow_parameters))])
    if len(text_names) != len(text_parameters) or len(flow_names) != len(flow_parameters):
        raise ValueError("G008 parameter-name coverage differs")
    text_named = tuple(zip(text_names, text_parameters))
    flow_named = tuple(zip(flow_names, flow_parameters))
    diagnostic = None
    if capture_full_diagnostics:
        diagnostic = {
            "version": "clean29529_g008_full_update_diagnostics_v1",
            "active_channel_mode": active_channel_mode,
            "optimizer_step_mode": effective_step_mode,
            "state_snapshots": {
                "before_update": update_state_snapshot(
                    text_optimizer=text_optimizer,
                    flow_optimizer=flow_optimizer,
                    text_named_parameters=text_named,
                    flow_named_parameters=flow_named,
                )
            },
            "pre_mask_gradient_inventory": {},
        }

    # Text channel: exactly the 28 repair clones, never the original R0 calls.
    local_text_counts = [
        sum(record.policy_active and text_enabled for record in trajectory.text_turns)
        for trajectory in rows
    ]
    text_weights = trajectory_channel_scales(
        local_text_counts, device=accelerator.device, process_group=process_group
    )
    local_text_active = sum(local_text_counts)
    global_text_active = int(text_weights["global_active_trajectory_count"])
    text_optimizer.zero_grad(set_to_none=True)
    flow_optimizer.zero_grad(set_to_none=True)
    text_metrics: list[list[dict[str, Any]]] = []
    for trajectory_index, trajectory in enumerate(rows):
        records = [
            record if text_enabled else replace(record, advantage=0.0, policy_active=False)
            for record in trajectory.text_turns
        ]
        padding_source = records[0] if records else trajectory.text_padding_turn
        if max_text_turns and padding_source is None:
            raise ValueError("G008 text padding source is unavailable")
        records.extend(
            replace(padding_source, advantage=0.0, policy_active=False)
            for _ in range(max_text_turns - len(records))
        )
        text_metrics.append([
            accumulate_text_turn(
                policy_inferencer=policy_controller_inferencer,
                reference_inferencer=reference_controller_inferencer,
                turn=record,
                accelerator=accelerator,
                clip_range=text_clip_range,
                kl_beta=text_kl_beta,
                backward_loss_scale=(
                    text_weights["backward_scales"][trajectory_index]
                    if record.policy_active else 1.0
                ),
                max_grad_norm=max_grad_norm,
            )
            for record in records
        ])
    if diagnostic is not None:
        diagnostic["pre_mask_gradient_inventory"]["text_phase"] = {
            "text": gradient_inventory(text_named),
            "flow": gradient_inventory(flow_named),
        }
    if phase_observer is not None:
        phase_observer("after_text_backward")
    text_pre_step_gate = synchronized_pre_step_gate(
        channel="text", metrics=text_metrics, named_parameters=text_named,
        globally_active=bool(global_text_active), device=accelerator.device,
        process_group=process_group,
    )
    text_grad_norm = (
        clip_trainable_gradients(
            accelerator=accelerator, parameters=text_parameters,
            max_grad_norm=max_grad_norm,
        ) if global_text_active else 0.0
    )
    text_post_clip_gate = synchronized_post_clip_gate(
        channel="text", grad_norm=text_grad_norm, named_parameters=text_named,
        globally_active=bool(global_text_active), device=accelerator.device,
        process_group=process_group,
    )
    assert_gradients_none_or_zero(flow_named)
    if commit_optimizer_step:
        if global_text_active and optimizer_step_observer is not None:
            optimizer_step_observer("text", "possible")
        text_step = step_channel_optimizer(
            text_optimizer, local_active=bool(local_text_active),
            activity_device=accelerator.device, process_group=process_group,
            inactive_named_parameters=flow_named,
        )
    else:
        clear_gradients(text_named)
        text_step = {
            "stepped": False, "optimizer_step_call_count": 0,
            "reason": "no_commit_probe", "globally_active": bool(global_text_active),
        }
    if optimizer_step_observer is not None:
        optimizer_step_observer("text", "stepped" if text_step["stepped"] else "skipped")
    if phase_observer is not None:
        phase_observer("after_text_phase")
    if diagnostic is not None:
        diagnostic["state_snapshots"]["after_text_phase"] = update_state_snapshot(
            text_optimizer=text_optimizer, flow_optimizer=flow_optimizer,
            text_named_parameters=text_named, flow_named_parameters=flow_named,
        )

    # Flow channel A: each original K=28 R0 record exactly once.
    prepared_r0 = [
        record if flow_enabled else replace(record, advantage=0.0, policy_active=False)
        for record in original_r0
    ]
    r0_counts = [int(record.policy_active) for record in prepared_r0]
    r0_weights = trajectory_channel_scales(
        r0_counts, device=accelerator.device, process_group=process_group
    )
    r0_metrics: list[list[dict[str, Any]]] = []
    for index, record in enumerate(prepared_r0):
        r0_metrics.append([
            accumulate_flow_call(
                flow_inferencer=flow_inferencer,
                reference_flow_inferencer=reference_flow_inferencer,
                record=record,
                grpo_config=grpo_config,
                accelerator=accelerator,
                transformer=transformer,
                backward_loss_scale=(
                    r0_weights["backward_scales"][index] * R0_CHANNEL_WEIGHT
                    if record.policy_active else 1.0
                ),
                max_grad_norm=max_grad_norm,
            )
        ])

    # Flow channel B: 4x7 repair clones, with no cloned R0 records.  Each
    # trajectory advantage is averaged over that trajectory's executed calls.
    repair_counts = [
        sum(record.policy_active and flow_enabled for record in trajectory.flow_calls)
        for trajectory in rows
    ]
    repair_weights = trajectory_channel_scales(
        repair_counts, device=accelerator.device, process_group=process_group
    )
    repair_metrics: list[list[dict[str, Any]]] = []
    for trajectory_index, trajectory in enumerate(rows):
        records = [
            record if flow_enabled else replace(record, advantage=0.0, policy_active=False)
            for record in trajectory.flow_calls
        ]
        padding_source = original_r0[trajectory_index % len(original_r0)]
        records.extend(
            replace(padding_source, advantage=0.0, policy_active=False)
            for _ in range(max_flow_calls - len(records))
        )
        repair_metrics.append([
            accumulate_flow_call(
                flow_inferencer=flow_inferencer,
                reference_flow_inferencer=reference_flow_inferencer,
                record=record,
                grpo_config=grpo_config,
                accelerator=accelerator,
                transformer=transformer,
                backward_loss_scale=(
                    repair_weights["backward_scales"][trajectory_index]
                    * REPAIR_CHANNEL_WEIGHT
                    if record.policy_active else 1.0
                ),
                max_grad_norm=max_grad_norm,
            )
            for record in records
        ])
    flow_metrics = [*r0_metrics, *repair_metrics]
    local_flow_active = sum(r0_counts) + sum(repair_counts)
    global_flow_active = (
        int(r0_weights["global_active_trajectory_count"])
        + int(repair_weights["global_active_trajectory_count"])
    )
    if diagnostic is not None:
        diagnostic["pre_mask_gradient_inventory"]["flow_phase"] = {
            "text": gradient_inventory(text_named),
            "flow": gradient_inventory(flow_named),
        }
    if phase_observer is not None:
        phase_observer("after_flow_backward")
    flow_pre_step_gate = synchronized_pre_step_gate(
        channel="flow", metrics=flow_metrics, named_parameters=flow_named,
        globally_active=bool(global_flow_active), device=accelerator.device,
        process_group=process_group,
    )
    flow_grad_norm = (
        clip_trainable_gradients(
            accelerator=accelerator, parameters=flow_parameters,
            max_grad_norm=max_grad_norm,
        ) if global_flow_active else 0.0
    )
    flow_post_clip_gate = synchronized_post_clip_gate(
        channel="flow", grad_norm=flow_grad_norm, named_parameters=flow_named,
        globally_active=bool(global_flow_active), device=accelerator.device,
        process_group=process_group,
    )
    assert_gradients_none_or_zero(text_named)
    if commit_optimizer_step:
        if global_flow_active and optimizer_step_observer is not None:
            optimizer_step_observer("flow", "possible")
        flow_step = step_channel_optimizer(
            flow_optimizer, local_active=bool(local_flow_active),
            activity_device=accelerator.device, process_group=process_group,
            inactive_named_parameters=text_named,
        )
    else:
        clear_gradients(flow_named)
        flow_step = {
            "stepped": False, "optimizer_step_call_count": 0,
            "reason": "no_commit_probe", "globally_active": bool(global_flow_active),
        }
    if optimizer_step_observer is not None:
        optimizer_step_observer("flow", "stepped" if flow_step["stepped"] else "skipped")
    if phase_observer is not None:
        phase_observer("after_flow_phase")
    if diagnostic is not None:
        diagnostic["state_snapshots"]["after_flow_phase"] = update_state_snapshot(
            text_optimizer=text_optimizer, flow_optimizer=flow_optimizer,
            text_named_parameters=text_named, flow_named_parameters=flow_named,
        )
        snapshots = diagnostic["state_snapshots"]
        diagnostic["state_deltas"] = {
            "text_phase": update_state_delta(snapshots["before_update"], snapshots["after_text_phase"]),
            "flow_phase": update_state_delta(snapshots["after_text_phase"], snapshots["after_flow_phase"]),
            "whole_update": update_state_delta(snapshots["before_update"], snapshots["after_flow_phase"]),
        }
    return {
        "trajectory_count": len(rows),
        "original_r0_record_count": len(original_r0),
        "cloned_r0_active_count": 0,
        "r0_and_repair_records_separate": True,
        "padding_version": PADDING_VERSION,
        "flow_metrics": flow_metrics,
        "r0_flow_metrics": r0_metrics,
        "repair_flow_metrics": repair_metrics,
        "text_metrics": text_metrics,
        "text_grad_norm": text_grad_norm,
        "flow_grad_norm": flow_grad_norm,
        "text_step": text_step,
        "flow_step": flow_step,
        "text_pre_step_gate": text_pre_step_gate,
        "flow_pre_step_gate": flow_pre_step_gate,
        "text_post_clip_gate": text_post_clip_gate,
        "flow_post_clip_gate": flow_post_clip_gate,
        "text_before_flow": True,
        "commit_optimizer_step": bool(commit_optimizer_step),
        "active_channel_mode": active_channel_mode,
        "optimizer_step_mode": effective_step_mode,
        "text_weighting": text_weights,
        "flow_weighting": {"r0": r0_weights, "repair": repair_weights},
        "full_update_diagnostics": diagnostic,
    }

def _tensor_image_to_pil(image: torch.Tensor):
    from PIL import Image

    tensor = image.detach().float().clamp(0.0, 1.0)
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("flow image tensor must have shape [3,H,W]")
    array = (
        tensor.mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


class FlowGRPOMultiroundRollout:
    """Frozen SFT/V20 controller around state-grouped Flow-GRPO calls."""

    def __init__(
        self,
        *,
        flow_inferencer: Any,
        controller_inferencer: Any,
        system_prompt: str,
        grpo_config: Any,
        accelerator: Any,
        controller_lockstep_process_group: Any = None,
    ) -> None:
        self.flow_inferencer = flow_inferencer
        self.controller_inferencer = controller_inferencer
        self.system_prompt = str(system_prompt).rstrip("\n")
        self.grpo_config = grpo_config
        self.accelerator = accelerator
        self.controller_lockstep_process_group = controller_lockstep_process_group
        self.last_phase_timings: dict[str, Any] = {}
        self.score_field_schedule = ScoreFieldSchedule.from_tokenizer(
            controller_inferencer.tokenizer
        )
        # G022 text-action support contract; None keeps G008-G021 byte-identical.
        self.text_action_support = None
        self.text_action_support_mask = None
        self.text_action_attempt_id = ""
        self.text_action_evidence_dir = None
        self.text_action_refusal_syncs = 0
        self.text_action_injection = None
        self.text_action_injection_fired = False

    def bind_text_action_support(
        self,
        support: Any,
        *,
        attempt_id: str,
        evidence_dir: Any = None,
        require_lockstep: bool = True,
    ) -> dict[str, Any]:
        """Install the supported-id contract for every G022 controller call.

        Binding is deliberately explicit and validated here rather than read
        opportunistically at each call site: the constrained score schedule is
        proven to be a subset of the support, the contract is proven identical
        on every rank, and the per-turn refusal reduction is proven to have a
        synchronised place to run.  A contract that cannot be enforced is
        refused at bind time instead of silently degrading to no mask.
        """
        if support is None:
            raise ValueError("G022 text-action support contract is required")
        schedule_binding = support.validate_schedule(
            self.score_field_schedule, where="g022_score_field_schedule"
        )
        sync = assert_support_contract_synchronized(
            support,
            process_group=self.controller_lockstep_process_group,
            require_distributed=False,
        )
        if require_lockstep and int(sync["world_size"]) > 1:
            if self.controller_lockstep_process_group is None:
                raise UnsupportedTextActionTokenError(
                    "G022 text-action refusal needs the CPU/Gloo object group"
                )
            # `uniform_collective_order` is what makes the per-turn reduction
            # safe: every rank walks the full controller-turn range with
            # padded dummy calls, so all ranks reach the same reduction at the
            # same turn index. Without it a rank could break out of the loop
            # early and leave its peers blocked -- the A4 hang, reintroduced
            # by the very check meant to prevent a hang.
            if not bool(getattr(self, "uniform_collective_order", False)):
                raise UnsupportedTextActionTokenError(
                    "G022 text-action refusal needs uniform collective order so "
                    "every rank reaches the reduction at the same controller turn"
                )
        self.text_action_support = support
        self.text_action_support_mask = support.additive_mask()
        self.text_action_attempt_id = str(attempt_id)
        self.text_action_evidence_dir = evidence_dir
        for inferencer in (
            self.controller_inferencer,
            getattr(self, "reference_controller_inferencer", None),
        ):
            if inferencer is not None:
                inferencer.g022_text_action_support = support
        return {
            "version": support.version,
            "attempt_id": str(attempt_id),
            "support": support.describe(),
            "rank_synchronized": sync,
            "score_field_schedule": schedule_binding,
        }

    @staticmethod
    def parse_text_action_injection(spec: str) -> dict[str, int] | None:
        """`rank=1,turn=0,row=0,position=3,token_id=151665` -> dict, or None.

        Deliberately explicit and never defaulted: the injected-hole gate has
        to drive the refusal through the production path, and an injection
        that silently no-ops would make a green gate meaningless.
        """
        text = str(spec or "").strip()
        if not text:
            return None
        fields: dict[str, int] = {}
        for chunk in text.split(","):
            if not chunk.strip():
                continue
            key, _, value = chunk.partition("=")
            fields[key.strip()] = int(value.strip())
        missing = {"rank", "turn", "token_id"} - set(fields)
        if missing:
            raise ValueError(
                f"G022 unsupported-token injection is missing {sorted(missing)}"
            )
        return {
            "rank": int(fields["rank"]),
            "turn": int(fields["turn"]),
            "row": int(fields.get("row", 0)),
            "position": int(fields.get("position", 0)),
            "token_id": int(fields["token_id"]),
        }

    def _g022_maybe_inject_unsupported(
        self,
        scratch_rows: Sequence[Mapping[str, Any]],
        *,
        turn_index: int,
    ) -> None:
        """Expected-failure gate only: put one unsupported id on one rank.

        The injection edits the generation *result* rather than the sampler,
        because once the support mask is correct the sampler can no longer
        produce one -- which is exactly the property gate B measures. This
        drives the detector, the evidence, the cross-rank reduction and the
        refusal through the production code path.
        """
        spec = self.text_action_injection
        if not spec or self.text_action_injection_fired:
            return
        if int(spec["turn"]) != int(turn_index):
            return
        rank = (
            int(dist.get_rank())
            if dist.is_available() and dist.is_initialized()
            else 0
        )
        if int(spec["rank"]) != rank:
            return
        row_index = int(spec["row"])
        if row_index >= len(scratch_rows):
            return
        row = scratch_rows[row_index]
        ids = [int(value) for value in (row.get("content_token_ids") or [])]
        if ids:
            position = max(0, min(int(spec["position"]), len(ids) - 1))
            ids[position] = int(spec["token_id"])
        else:
            ids = [int(spec["token_id"])]
            position = 0
        text, unsupported = (
            self.controller_inferencer._decode_content_token_ids_checked(ids)
        )
        row["content_token_ids"] = ids
        row["text"] = text
        row["unsupported_text_action_tokens"] = unsupported
        row["g022_injected_unsupported_token"] = {
            "rank": rank,
            "turn_index": int(turn_index),
            "row_index": row_index,
            "position": int(position),
            "token_id": int(spec["token_id"]),
        }
        self.text_action_injection_fired = True

    def _g022_sync_text_action_refusal(self, local_flag: int) -> int:
        """MAX-reduce the local refusal flag so every rank refuses together.

        A rank-local raise here is the A4 hang: the raising rank leaves the
        collective sequence and its peers wait for the allocation to time out.
        The reduction runs on the CPU/Gloo group so a refusal never depends on
        a healthy NCCL communicator.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return int(local_flag)
        values = torch.tensor([int(local_flag)], dtype=torch.int64, device="cpu")
        dist.all_reduce(
            values,
            op=dist.ReduceOp.MAX,
            group=self.controller_lockstep_process_group,
        )
        self.text_action_refusal_syncs += 1
        return int(values[0].item())

    def _g022_refuse_unsupported_text_actions(
        self,
        scratch_rows: Sequence[Mapping[str, Any]],
        *,
        turn_index: int,
    ) -> None:
        support = self.text_action_support
        if support is None:
            return
        local: list[dict[str, Any]] = []
        for row_index, row in enumerate(scratch_rows):
            entries = row.get("unsupported_text_action_tokens") or []
            if not entries:
                continue
            local.append(
                unsupported_token_evidence(
                    support=support,
                    token_ids=list(row.get("content_token_ids") or []),
                    log_probs=list(row.get("selected_log_probs") or []) or None,
                    attempt_id=self.text_action_attempt_id,
                    where=f"controller_turn_{int(turn_index)}_row_{int(row_index)}",
                    extra={
                        "turn_index": int(turn_index),
                        "row_index": int(row_index),
                        "text_action_support_masked": bool(
                            row.get("text_action_support_masked", False)
                        ),
                        "stop_reason": row.get("stop_reason"),
                    },
                )
            )
        world_flag = self._g022_sync_text_action_refusal(1 if local else 0)
        if not world_flag:
            return
        self._g022_persist_text_action_refusal(local, turn_index=turn_index)
        raise UnsupportedTextActionTokenError(
            "G022 refused an unsupported text-action token before parsing: "
            f"turn={int(turn_index)} local_rows={len(local)} "
            f"support={support.describe()} evidence={local}"
        )

    def _g022_persist_text_action_refusal(
        self,
        local: Sequence[Mapping[str, Any]],
        *,
        turn_index: int,
    ) -> None:
        directory = self.text_action_evidence_dir
        if directory is None:
            return
        try:
            rank = (
                int(dist.get_rank())
                if dist.is_available() and dist.is_initialized()
                else 0
            )
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            attempt = self.text_action_attempt_id or "unknown_attempt"
            target = path / (
                f"{attempt}_G022_UNSUPPORTED_TEXT_ACTION_RANK{rank:03d}.json"
            )
            target.write_text(
                json.dumps(
                    {
                        "version": "clean29529_g022_unsupported_text_action_refusal_v1",
                        "attempt_id": attempt,
                        "rank": rank,
                        "turn_index": int(turn_index),
                        "local_row_count": len(local),
                        "refused_locally": bool(local),
                        "records": [dict(row) for row in local],
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        except Exception:
            # Evidence is best-effort; the refusal itself is not.
            pass

    @torch.no_grad()
    def _flow_call(
        self,
        *,
        input_terms: list[Any],
        call_kind: str,
        controller_round_index: int | None,
        generators: Any = None,
        sde_window_identity: Mapping[str, int] | None = None,
    ) -> FlowCallRecord:
        outputs = self.flow_inferencer.interleave_inference(
            input_terms,
            think=False,
            understanding_output=False,
            cfg_text_scale=float(self.grpo_config.sample.guidance_scale),
            cfg_img_scale=1.5,
            cfg_interval=[0.4, 1.0],
            timestep_shift=float(self.grpo_config.train.timestep_shift),
            num_timesteps=active_num_timesteps(self),
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",
            image_shapes=(
                int(self.grpo_config.resolution),
                int(self.grpo_config.resolution),
            ),
            grpo_config=self.grpo_config,
            accelerator=self.accelerator,
            # A3 / Q-5: this was `EDIT_STAGE` unconditionally, so an R0
            # call ran at edit eta 1.0 while the record below stored the
            # generation eta 0.7. The R3 stage split (G-5, I-1) was
            # therefore not in effect and the record said otherwise.
            noise_level=stage_noise_level(self.grpo_config, call_kind),
            generators=generators,
            sde_window_identities=(
                None if sde_window_identity is None else [sde_window_identity]
            ),
        )
        if len(outputs) != 1 or not isinstance(outputs[0], dict):
            raise RuntimeError("Flow-GRPO image call returned an invalid payload")
        output = outputs[0]
        image_tensor = output.get("image")
        if not torch.is_tensor(image_tensor):
            raise RuntimeError("Flow-GRPO image call returned no image tensor")
        record = FlowCallRecord(
            call_index=0,
            call_kind=call_kind,
            controller_round_index=controller_round_index,
            input_terms=[
                value.copy() if hasattr(value, "copy") else str(value)
                for value in input_terms
            ],
            image=_tensor_image_to_pil(image_tensor),
            latents=list(output.get("all_latents") or []),
            log_probs=list(output.get("all_log_probs") or []),
            timesteps=output.get("timesteps"),
            sde_timestep_begin=output.get("sde_timestep_begin"),
            sde_window_seed_identity=output.get("sde_window_seed_identity"),
            sde_transition_layout=str(output.get("sde_transition_layout") or "contiguous_v1"),
            sde_transition_indices=list(output.get("sde_transition_indices") or []),
            # Review of item 4: bind, on the record, everything the
            # learn pass would otherwise re-derive from a config that may have
            # changed since this record was produced.
            noise_level=stage_noise_level(self.grpo_config, call_kind),
            eta_clamp_mode=str(
                getattr(
                    getattr(self.grpo_config, "g016", None),
                    "eta_clamp_mode",
                    LEGACY_ETA_CLAMP_MODE,
                )
            ),
            # C1 / Q-6: bind the count actually handed to the sampler.
            replay_bindings=capture_replay_bindings(
                self.grpo_config, num_timesteps=active_num_timesteps(self)
            ),
        )
        if not torch.is_tensor(record.timesteps):
            raise RuntimeError("Flow-GRPO image call returned no timesteps")
        record.offload_to_cpu()
        record.validate()
        del image_tensor, output, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return record

    @torch.no_grad()
    def _flow_call_batch(
        self,
        *,
        input_terms_batch: Sequence[Sequence[Any]],
        call_kind: str,
        controller_round_indices: Sequence[int | None],
        generators: Sequence[Any] | None = None,
        sde_window_identities: Sequence[Mapping[str, int]] | None = None,
    ) -> list[FlowCallRecord]:
        rows = [list(row) for row in input_terms_batch]
        batch_size = len(rows)
        if batch_size < 1:
            raise ValueError("flow call batch is empty")
        if len(controller_round_indices) != batch_size:
            raise ValueError("flow/controller batch cardinality mismatch")
        if generators is not None and len(generators) != batch_size:
            raise ValueError("flow/generator batch cardinality mismatch")
        if sde_window_identities is not None and len(sde_window_identities) != batch_size:
            raise ValueError("flow/SDE identity batch cardinality mismatch")
        if batch_size == 1:
            return [
                self._flow_call(
                    input_terms=rows[0],
                    call_kind=call_kind,
                    controller_round_index=controller_round_indices[0],
                    generators=(
                        None
                        if generators is None
                        else [generators[0]]
                    ),
                    sde_window_identity=(
                        None
                        if sde_window_identities is None
                        else sde_window_identities[0]
                    ),
                )
            ]
        outputs = self.flow_inferencer.interleave_inference_batch(
            rows,
            think=False,
            understanding_output=False,
            cfg_text_scale=float(self.grpo_config.sample.guidance_scale),
            cfg_img_scale=1.5,
            cfg_interval=[0.4, 1.0],
            timestep_shift=float(self.grpo_config.train.timestep_shift),
            num_timesteps=active_num_timesteps(self),
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",
            image_shapes=(
                int(self.grpo_config.resolution),
                int(self.grpo_config.resolution),
            ),
            grpo_config=self.grpo_config,
            accelerator=self.accelerator,
            # A3 / Q-5: this was `EDIT_STAGE` unconditionally, so an R0
            # call ran at edit eta 1.0 while the record below stored the
            # generation eta 0.7. The R3 stage split (G-5, I-1) was
            # therefore not in effect and the record said otherwise.
            noise_level=stage_noise_level(self.grpo_config, call_kind),
            generators=generators,
            sde_window_identities=sde_window_identities,
        )
        if len(outputs) != batch_size:
            raise RuntimeError("Flow-GRPO batched image call lost candidates")
        records = []
        for input_terms, controller_round_index, output in zip(
            rows,
            controller_round_indices,
            outputs,
        ):
            if not isinstance(output, dict):
                raise RuntimeError(
                    "Flow-GRPO batched image call returned invalid payload"
                )
            image_tensor = output.get("image")
            if not torch.is_tensor(image_tensor):
                raise RuntimeError(
                    "Flow-GRPO batched image call returned no image tensor"
                )
            record = FlowCallRecord(
                call_index=0,
                call_kind=call_kind,
                controller_round_index=controller_round_index,
                input_terms=[
                    value.copy() if hasattr(value, "copy") else str(value)
                    for value in input_terms
                ],
                image=_tensor_image_to_pil(image_tensor),
                latents=list(output.get("all_latents") or []),
                log_probs=list(output.get("all_log_probs") or []),
                timesteps=output.get("timesteps"),
                sde_timestep_begin=output.get("sde_timestep_begin"),
                sde_window_seed_identity=output.get("sde_window_seed_identity"),
                sde_transition_layout=str(output.get("sde_transition_layout") or "contiguous_v1"),
                sde_transition_indices=list(output.get("sde_transition_indices") or []),
                noise_level=stage_noise_level(self.grpo_config, call_kind),
                eta_clamp_mode=str(
                    getattr(
                        getattr(self.grpo_config, "g016", None),
                        "eta_clamp_mode",
                        LEGACY_ETA_CLAMP_MODE,
                    )
                ),
                # C1 / Q-6: bind the count actually handed to the sampler.
                replay_bindings=capture_replay_bindings(
                    self.grpo_config, num_timesteps=active_num_timesteps(self)
                ),
            )
            if not torch.is_tensor(record.timesteps):
                raise RuntimeError(
                    "Flow-GRPO batched image call returned no timesteps"
                )
            record.offload_to_cpu()
            record.validate()
            records.append(record)
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return records

    def generate_r0_batch(
        self,
        *,
        prompts: Sequence[str],
        generators: Sequence[Any] | None = None,
    ) -> list[FlowCallRecord]:
        prompts = [str(value) for value in prompts]
        if not prompts or any(not value for value in prompts):
            raise ValueError("G008 R0 prompt batch is empty")
        started = time.monotonic()
        records = self._flow_call_batch(
            input_terms_batch=[[value] for value in prompts],
            call_kind="r0",
            controller_round_indices=[None] * len(prompts),
            generators=generators,
        )
        for record in records:
            record.call_index = 0
        self.last_phase_timings = {
            "version": "clean29529_g008_rollout_phase_timing_v1",
            "r0_denoise_sec": time.monotonic() - started,
            "r0_image_count": len(records),
        }
        return records

    def _initial_controller_context(
        self,
        prompt: str,
        anchor_image: Any,
    ) -> tuple[Any, list[Any]]:
        # Exact V20/SFT policy observation: system prompt, raw generation
        # prompt, and one current anchor image.  No verifier contract, budget,
        # diagnosis scaffold, duplicate image, or host image-label text.
        terms: list[Any] = [
            self.system_prompt,
            str(prompt),
            anchor_image.copy(),
        ]
        context = self.controller_inferencer.init_gen_context()
        from PIL import Image
        for term in terms:
            if isinstance(term, str):
                context = self.controller_inferencer.update_context_text(term, context)
            elif isinstance(term, Image.Image):
                context = self.controller_inferencer.update_context_image(
                    term, context, vae=True, vit=True
                )
            else:
                raise TypeError(f"unsupported G008 controller term: {type(term)}")
        return context, terms

    def _collective_pad_decoder_traversals(
        self, count: int, *, modes: Sequence[str] | None = None
    ) -> None:
        if count <= 0:
            return
        traversal_modes = list(modes or ["und"] * int(count))
        if len(traversal_modes) != int(count):
            raise ValueError("collective traversal mode coverage differs")
        language_model = self.controller_inferencer.model.language_model
        sentinel = torch.zeros((), device=self.accelerator.device, dtype=torch.float32)
        for collective_mode in traversal_modes:
            # Enter through the wrapped language-model root. Calling unwrapped
            # decoder layers directly skips the root FSDP handle, so ranks
            # with synthetic context padding enqueue a different nested
            # collective sequence from ranks rebuilding real context.
            language_model(
                mode="collective_noop",
                collective_mode=collective_mode,
                sentinel=sentinel,
            )

    def _distributed_controller_lockstep_sync(
        self, local_continues: bool, local_forced_eos_count: int
    ) -> tuple[bool, int]:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("G016 controller lockstep requires distributed ranks")
        if self.controller_lockstep_process_group is None:
            raise RuntimeError("G016 controller lockstep requires a CPU/Gloo group")
        values = torch.tensor(
            [int(bool(local_continues)), int(local_forced_eos_count)],
            dtype=torch.int64,
            device="cpu",
        )
        dist.all_reduce(
            values,
            op=dist.ReduceOp.MAX,
            group=self.controller_lockstep_process_group,
        )
        return bool(values[0].item()), int(values[1].item())

    def generate_from_anchor_batch(
        self,
        *,
        prompts: Sequence[str],
        verification_contracts: Sequence[Sequence[Mapping[str, Any]]],
        anchor_images: Sequence[Any],
        anchor_metadata: Sequence[Mapping[str, Any]],
        repair_generators: Sequence[Any] | None = None,
        repair_sde_window_identities: Sequence[Mapping[str, int]] | None = None,
        do_sample: bool = True,
        temperature: float = 0.5,
        response_max_tokens: int = 768,
    ) -> list[MultiroundTrajectory]:
        prompts = [str(value) for value in prompts]
        batch_size = len(prompts)
        if not batch_size or not (
            len(verification_contracts)
            == len(anchor_images)
            == len(anchor_metadata)
            == batch_size
        ):
            raise ValueError("G008 anchor-fork batch cardinality differs")
        if repair_generators is not None and len(repair_generators) != batch_size:
            raise ValueError("G008 repair generator coverage differs")
        if repair_sde_window_identities is not None and len(repair_sde_window_identities) != batch_size:
            raise ValueError("G012 repair SDE identity coverage differs")
        rollout_started = time.monotonic()
        self.last_collective_padding_flow_record = None
        phase = {
            "r0_denoise_sec": 0.0,
            "repair_denoise_sec": 0.0,
            "controller_generation_sec": 0.0,
            "controller_context_sec": 0.0,
            "r0_image_count": 0,
            "repair_image_count": 0,
            "controller_turn_count": 0,
            "controller_batch_call_count": 0,
            "controller_single_call_count": 0,
        }
        eos_token_id = int(self.controller_inferencer.new_token_ids["eos_token_id"])
        states = []
        context_started = time.monotonic()
        for local_index, (prompt, clauses, anchor, metadata) in enumerate(
            zip(prompts, verification_contracts, anchor_images, anchor_metadata)
        ):
            rows = [dict(value) for value in clauses]
            clause_ids = [str(value["id"]) for value in rows]
            context, context_terms = self._initial_controller_context(
                prompt, anchor
            )
            padding = TextTurnRecord(
                turn_index=0,
                context_terms=[
                    value.copy() if hasattr(value, "copy") else str(value)
                    for value in context_terms
                ],
                forced_prefix_token_ids=[],
                sampled_token_ids=[eos_token_id],
                old_log_probs=[0.0],
                behavior_temperature=1.0,
                canonical_response="[V20_PADDING]",
                action="done",
                advantage=0.0,
                policy_active=False,
                valid=True,
            )
            padding.validate()
            states.append(
                {
                    "local_index": local_index,
                    "prompt": prompt,
                    "clauses": rows,
                    "clause_ids": clause_ids,
                    "anchor": anchor.copy(),
                    "metadata": dict(metadata),
                    "flow_calls": [],
                    "current_image": anchor.copy(),
                    "context": context,
                    "context_terms": context_terms,
                    "text_turns": [],
                    "events": [],
                    "repair_rounds": 0,
                    "stop_reason": "round_cap",
                    "done": False,
                    "active": True,
                    "padding": padding,
                    "repair_generator": (
                        None if repair_generators is None else repair_generators[local_index]
                    ),
                    "repair_sde_window_identity": (
                        None
                        if repair_sde_window_identities is None
                        else dict(repair_sde_window_identities[local_index])
                    ),
                }
            )
        phase["controller_context_sec"] += time.monotonic() - context_started

        uniform_collective_order = bool(
            getattr(self, "uniform_collective_order", False)
        )
        phase["uniform_collective_order"] = uniform_collective_order
        controller_distributed_lockstep = bool(
            uniform_collective_order
            and getattr(
                getattr(self.grpo_config, "g016", None),
                "controller_distributed_lockstep",
                False,
            )
        )
        phase["controller_distributed_lockstep"] = (
            controller_distributed_lockstep
        )
        phase["controller_lockstep_version"] = (
            "clean29529_g016_distributed_controller_lockstep_v1"
            if controller_distributed_lockstep
            else None
        )
        phase["controller_lockstep_decode_traversals"] = 0
        phase["controller_lockstep_dummy_traversals"] = 0
        phase["controller_lockstep_forced_eos_padding_traversals"] = 0
        phase["collective_padding_controller_turn_count"] = 0
        phase["collective_padding_flow_call_count"] = 0
        for turn_index in range(MAX_CONTROLLER_TURNS):
            genuinely_active_states = [value for value in states if value["active"]]
            active_states = states if uniform_collective_order else genuinely_active_states
            if not active_states:
                break
            pending_repairs = []
            remaining_edits = MAX_REPAIR_ROUNDS - turn_index
            prefix_text = forced_controller_prefix(turn_index, remaining_edits)
            scratch_contexts = [deepcopy(value["context"]) for value in active_states]
            generated_started = time.monotonic()
            if len(active_states) > 1 or controller_distributed_lockstep:
                lockstep_kwargs = (
                    {
                        "distributed_lockstep_sync": self._distributed_controller_lockstep_sync,
                        "distributed_lockstep_noop": lambda: self._collective_pad_decoder_traversals(1),
                    }
                    if controller_distributed_lockstep
                    else {}
                )
                scratch_rows = self.controller_inferencer.gen_text_persistent_batch(
                    scratch_contexts,
                    max_length=int(response_max_tokens),
                    do_sample=bool(do_sample),
                    temperature=float(temperature),
                    return_log_probs=True,
                    forced_prefix_text=prefix_text,
                    constrained_schedule=self.score_field_schedule,
                    support_mask=self.text_action_support_mask,
                    **lockstep_kwargs,
                )
                phase["controller_batch_call_count"] += 1
            else:
                scratch_rows = [
                    self.controller_inferencer.gen_text_persistent(
                        scratch_contexts[0],
                        max_length=int(response_max_tokens),
                        do_sample=bool(do_sample),
                        temperature=float(temperature),
                        return_log_probs=True,
                        forced_prefix_text=prefix_text,
                        constrained_schedule=self.score_field_schedule,
                        support_mask=self.text_action_support_mask,
                    )
                ]
                phase["controller_single_call_count"] += 1
            if uniform_collective_order:
                generated_lengths = [len(row.get("content_token_ids") or []) for row in scratch_rows]
                if not generated_lengths:
                    raise RuntimeError("uniform collective ordering has no local controller rows")
                if controller_distributed_lockstep:
                    if any(
                        row.get("distributed_lockstep_version")
                        != "clean29529_g016_distributed_controller_lockstep_v1"
                        for row in scratch_rows
                    ):
                        raise RuntimeError(
                            "G016 controller lockstep result version differs"
                        )
                    decode_counts = {
                        int(row["distributed_decode_traversals"])
                        for row in scratch_rows
                    }
                    if len(decode_counts) != 1:
                        raise RuntimeError(
                            "G016 local lockstep traversal count differs"
                        )
                    phase["controller_lockstep_decode_traversals"] += next(
                        iter(decode_counts)
                    )
                    phase["controller_lockstep_dummy_traversals"] += int(
                        scratch_rows[0]["distributed_dummy_traversals"]
                    )
                    phase[
                        "controller_lockstep_forced_eos_padding_traversals"
                    ] += int(
                        scratch_rows[0][
                            "distributed_forced_eos_padding_traversals"
                        ]
                    )
                else:
                    missing = int(response_max_tokens) - max(generated_lengths)
                    if missing < 0:
                        raise RuntimeError("controller generation exceeded the fixed traversal budget")
                    self._collective_pad_decoder_traversals(missing)
                    phase.setdefault("collective_padding_decoder_traversals", 0)
                    phase["collective_padding_decoder_traversals"] += missing
            phase["controller_generation_sec"] += time.monotonic() - generated_started
            phase["controller_turn_count"] += len(active_states)
            # Before ANY parse, image action, credit or backward: every rank
            # agrees on whether an unsupported text-action token was produced,
            # and every rank refuses together if one was.
            self._g022_maybe_inject_unsupported(
                scratch_rows, turn_index=turn_index
            )
            self._g022_refuse_unsupported_text_actions(
                scratch_rows, turn_index=turn_index
            )
            phase["g022_text_action_refusal_syncs"] = int(
                self.text_action_refusal_syncs
            )

            for state, scratch in zip(active_states, scratch_rows):
                if not state["active"]:
                    phase["collective_padding_controller_turn_count"] += 1
                    if uniform_collective_order:
                        self._collective_pad_decoder_traversals(1)
                    continue
                raw_response = prefix_text + str(scratch["text"])
                event: dict[str, Any] = {
                    "turn_index": turn_index,
                    "generation_time_sampled_field_spans": None,
                    "remaining_edits": remaining_edits,
                    "raw_response": raw_response,
                    "raw_forced_prefix": prefix_text,
                    "raw_forced_prefix_token_ids": list(scratch.get("forced_prefix_token_ids") or []),
                    "raw_content_token_ids": list(scratch.get("content_token_ids") or []),
                    "raw_selected_log_probs": list(scratch.get("selected_log_probs") or []),
                    "raw_content_allowed_ids": [
                        None if value is None else list(value)
                        for value in (scratch.get("content_token_allowed_ids") or [])
                    ],
                    "raw_content_policy_active": [
                        bool(value) for value in (scratch.get("content_token_policy_active") or [])
                    ],
                    "constrained_schedule_version": scratch.get("constrained_schedule_version"),
                    "constrained_score_token_ids": list(scratch.get("constrained_token_ids") or []),
                    "score": self.score_field_schedule.score_for(
                        scratch.get("constrained_token_ids") or []
                    ),
                    "controller_protocol_version": CONTROLLER_PROTOCOL_VERSION,
                    "forced_prefix_version": CONTROLLER_FORCED_PREFIX_VERSION,
                    "response_eos_model_selected": bool(scratch.get("eos_model_selected", False)),
                    "response_eos_host_forced": bool(scratch.get("eos_host_forced", False)),
                    "behavior_temperature": float(temperature),
                    "route_valid": False,
                    "canonical_response": "",
                    "action": "",
                    "image_generated": False,
                }
                # Capture the exact behavior-action IDs, including the
                # model-selected or host-appended EOS represented by the
                # generation result. Capturing only content_token_ids omitted
                # EOS while TextTurnRecord correctly retained it, making every
                # live lineage check fail despite no retokenization.
                capture_spans = getattr(self, "capture_sampled_field_spans", None)
                if callable(capture_spans):
                    event["generation_time_sampled_field_spans"] = capture_spans(
                        sampled_token_ids_from_event(
                            event, eos_token_id=eos_token_id
                        )
                    )
                try:
                    proposal = parse_controller_response(
                        raw_response, stable_clause_ids=state["clause_ids"]
                    )
                    canonical = canonicalize_response(
                        proposal,
                        round_index=turn_index,
                        remaining_edits=remaining_edits,
                        expected_source_image=f"Image #{state['repair_rounds']}",
                    )
                except Exception as exc:
                    event["error"] = f"{type(exc).__name__}: {exc}"
                    if uniform_collective_order:
                        self._collective_pad_decoder_traversals(1)
                    malformed = malformed_text_turn_from_event(
                        event,
                        context_terms=state["context_terms"],
                        eos_token_id=eos_token_id,
                        # A1 / Q-3: G022 keeps the recorded host-forced mask.
                        preserve_host_forced_mask=g022_single_head_mode(
                            self.grpo_config
                        ),
                    )
                    state["text_turns"].append(malformed)
                    state["events"].append(event)
                    state["stop_reason"] = "parse_error"
                    state["active"] = False
                    continue
                sampled_ids = sampled_token_ids_from_event(event, eos_token_id=eos_token_id)
                allowed_ids, policy_active = sampled_token_constraints_from_event(
                    event, sampled_token_count=len(sampled_ids)
                )
                action = str(proposal["action"])
                turn = TextTurnRecord(
                    turn_index=turn_index,
                    context_terms=[
                        value.copy() if hasattr(value, "copy") else str(value)
                        for value in state["context_terms"]
                    ],
                    forced_prefix_token_ids=list(event["raw_forced_prefix_token_ids"]),
                    sampled_token_ids=sampled_ids,
                    old_log_probs=list(event["raw_selected_log_probs"]),
                    behavior_temperature=float(temperature),
                    canonical_response=canonical,
                    action=action,
                    sampled_token_allowed_ids=allowed_ids,
                    sampled_token_policy_active=policy_active,
                )
                turn.validate()
                state["text_turns"].append(turn)
                state["context"] = self.controller_inferencer.update_context_text(
                    canonical, state["context"]
                )
                state["context_terms"].append(canonical)
                event.update(
                    {
                        "route_valid": True,
                        "canonical_response": canonical,
                        "action": action,
                        "payload": str(proposal.get("payload") or ""),
                        "score": int(proposal["score"]),
                        "raw_action": str(proposal.get("raw_action") or action),
                        "edit_none_alias": bool(proposal.get("edit_none_alias")), 
                    }
                )
                if action == "done":
                    state["done"] = True
                    state["stop_reason"] = "done"
                    state["active"] = False
                    state["events"].append(event)
                    continue
                payload = str(proposal["payload"])
                cache_devices = move_context_cache(state["context"], "cpu")
                pending_repairs.append(
                    {
                        "state": state,
                        "event": event,
                        "payload": payload,
                        "cache_devices": cache_devices,
                    }
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if uniform_collective_order:
                pending_state_ids = {id(value["state"]) for value in pending_repairs}
                for state in states:
                    if id(state) in pending_state_ids:
                        continue
                    pending_repairs.append(
                        {
                            "state": state,
                            "event": None,
                            "payload": "Preserve the image exactly.",
                            "cache_devices": None,
                            "collective_only_padding": True,
                        }
                    )
                    phase["collective_padding_flow_call_count"] += 1
            if pending_repairs:
                repair_started = time.monotonic()
                repairs = self._flow_call_batch(
                    input_terms_batch=[
                        [value["state"]["current_image"], value["payload"]]
                        for value in pending_repairs
                    ],
                    call_kind="repair",
                    controller_round_indices=[turn_index] * len(pending_repairs),
                    generators=[value["state"]["repair_generator"] for value in pending_repairs]
                    if repair_generators is not None else None,
                    sde_window_identities=(
                        None
                        if repair_sde_window_identities is None
                        else [
                            {
                                **value["state"]["repair_sde_window_identity"],
                                "round_index": int(turn_index),
                            }
                            for value in pending_repairs
                        ]
                    ),
                )
                phase["repair_denoise_sec"] += time.monotonic() - repair_started
                phase["repair_image_count"] += len(repairs)
                for item, repair in zip(pending_repairs, repairs):
                    if item.get("collective_only_padding") is True:
                        repair.policy_active = False
                        self.last_collective_padding_flow_record = repair
                        if uniform_collective_order:
                            self._collective_pad_decoder_traversals(
                                2, modes=["gen", "und"]
                            )
                        continue
                    state = item["state"]
                    restore_context_cache(state["context"], item["cache_devices"])
                    repair.call_index = len(state["flow_calls"])
                    state["flow_calls"].append(repair)
                    state["repair_rounds"] += 1
                    state["current_image"] = repair.image
                    state["context"] = self.controller_inferencer.update_context_image(
                        state["current_image"], state["context"], vae=True, vit=True
                    )
                    state["context_terms"].append(state["current_image"].copy())
                    item["event"]["image_generated"] = True
                    item["event"]["flow_call_index"] = repair.call_index
                    state["events"].append(item["event"])
                    if state["repair_rounds"] >= MAX_REPAIR_ROUNDS:
                        state["active"] = False
                        state["stop_reason"] = "round_cap"

        trajectories = []
        for state in states:
            metadata = {
                **state["metadata"],
                "version": ROLLOUT_VERSION,
                "max_repair_rounds": MAX_REPAIR_ROUNDS,
                "max_controller_turns": MAX_CONTROLLER_TURNS,
                "controller_protocol_version": CONTROLLER_PROTOCOL_VERSION,
                "sft_v20_policy_observation_exact": True,
                "verification_contract_visible_to_policy": False,
                "remaining_edits_visible_to_policy": False,
                "diagnosis_visible_to_policy": False,
                "verifier_labels_visible_to_policy": False,
                "cloned_r0_flow_record_present": False,
                "independent_post_fork_rng": True,
            }
            trajectory = MultiroundTrajectory(
                prompt=state["prompt"],
                r0_image=state["anchor"],
                final_image=state["current_image"],
                flow_calls=state["flow_calls"],
                text_turns=state["text_turns"],
                events=state["events"],
                stop_reason=state["stop_reason"],
                done=state["done"],
                repair_rounds=state["repair_rounds"],
                metadata=metadata,
                text_padding_turn=state["padding"],
            )
            trajectory.validate()
            trajectories.append(trajectory)
        total = time.monotonic() - rollout_started
        attributed = sum(
            float(phase[key])
            for key in (
                "repair_denoise_sec",
                "controller_generation_sec",
                "controller_context_sec",
            )
        )
        self.last_phase_timings = {
            "version": "clean29529_g008_rollout_phase_timing_v1",
            **phase,
            "rollout_total_sec": total,
            "rollout_unattributed_sec": max(0.0, total - attributed),
        }
        return trajectories


__all__ = [
    "CHANNEL_WEIGHT_VERSION",
    "CONTRACT_VERSION",
    "CONTROLLER_FORCED_PREFIX_VERSION",
    "CONTROLLER_PROTOCOL_VERSION",
    "CONTROLLER_SCORE_FIELD_VERSION",
    "FlowCallRecord",
    "FlowGRPOMultiroundRollout",
    "GENEVAL_MAX_ATTEMPTS_PER_CHUNK",
    "GENEVAL_REQUEST_CHUNK_SIZE",
    "MAX_CONTROLLER_TURNS",
    "MAX_REPAIR_ROUNDS",
    "LEGACY_MALFORMED_TEXT_LOCAL_CREDIT",
    "MALFORMED_TEXT_LOCAL_CREDIT",
    "MALFORMED_TEXT_LOCAL_CREDIT_SCALE",
    "MultiroundTrajectory",
    "PADDING_VERSION",
    "R0_CHANNEL_WEIGHT",
    "REPAIR_CHANNEL_WEIGHT",
    "REWARD_VERSION",
    "ROLLOUT_VERSION",
    "SharedPrefixReference",
    "assert_frozen_reference_optimizer_isolation",
    "TEXT_POLICY_VERSION",
    "TextTurnRecord",
    "apply_legacy_monitor_scores",
    "build_flow_contexts",
    "clip_trainable_gradients",
    "clipped_text_ppo_loss",
    "collective_padding_backward",
    "accumulate_flow_call",
    "accumulate_text_turn",
    "flow_channel_weight",
    "minimal_text_policy_loss",
    "malformed_text_turn_from_event",
    "move_context_cache",
    "replay_controller_context",
    "sampled_token_constraints_from_event",
    "sampled_token_ids_from_event",
    "ScoreFieldSchedule",
    "sandbag_margin",
    "score_trajectories_geneval",
    "restore_context_cache",
    "release_prompt_index",
    "request_geneval_score_chunks",
    "teacher_forced_text_log_probs",
    "trajectory_channel_scales",
    "synchronized_post_clip_gate",
    "synchronized_pre_step_gate",
    "text_k3_kl_loss",
    "text_turn_policy_loss",
    "legacy_trajectory_reward_metric",
    "update_multiround_group",
]
