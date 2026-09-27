"""G016 full-trajectory counting Flow-GRPO runtime primitives.

This module intentionally reuses the frozen G008/V22 SFT-native controller,
PPO/KL kernels, detector transport, and numerical gates.  It changes only the
G016 topology and credit/update semantics:

* one logical detached R0 is replicated byte-for-byte to all K=28 paths;
* every path is rolled independently to DONE/environment cap;
* no R0 policy record or optimizer contribution exists;
* controller and repair-flow calls consume their own round-causal advantage;
* text is updated before flow, once per complete logical group.
"""

from __future__ import annotations

import math
import pickle
import random
from collections import defaultdict
import os
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.distributed as dist

from flow_grpo.g016_sampled_spans import capture_sampled_field_spans
from flow_grpo.g008_counting import (
    GENERATION_STAGE,
    active_num_timesteps,
    stage_noise_level,
)
from flow_grpo.g008_counting import (
    CONTROLLER_FORCED_PREFIX_VERSION,
    CONTROLLER_PROTOCOL_VERSION,
    CONTROLLER_SCORE_FIELD_VERSION,
    FlowCallRecord,
    FlowGRPOMultiroundRollout as _G008ProtocolRollout,
    MultiroundTrajectory,
    ScoreFieldSchedule,
    SharedPrefixReference,
    TextTurnRecord,
    accumulate_flow_call,
    assert_frozen_reference_optimizer_isolation,
    clip_trainable_gradients,
    release_prompt_index,
    synchronized_post_clip_gate,
    synchronized_pre_step_gate,
    teacher_forced_text_log_probs,
    trajectory_channel_scales,
)
from unify_rl.train.v20_update_contract import (
    assert_gradients_none_or_zero,
    clear_gradients,
    gradient_inventory,
    step_channel_optimizer,
    update_state_delta,
    update_state_snapshot,
)
from unify_rl.train.g016_forensic_ring_buffer import (
    TOKEN_TOPK_VERSION,
    build_token_topk,
    label_sampled_tokens,
    label_sampled_tokens_from_ids,
)


CONTRACT_VERSION = "clean29529_g016_full_trajectory_counting_flowgrpo_v1"
# G016 keeps G009's reward equations and reward identity byte-for-byte.
REWARD_VERSION = "clean29529_g009_counting_process_reward_v1"
TEXT_POLICY_VERSION = "clean29529_g016_text_clip_ppo_k3_v1"
PADDING_VERSION = "clean29529_g016_repair_path_padding_v1"
ROLLOUT_VERSION = "clean29529_g016_same_r0_k28_full_trajectory_rollout_v1"
DETACHED_R0_VERSION = "clean29529_g016_detached_environment_r0_v1"
MAX_REPAIR_ROUNDS = 3
MAX_CONTROLLER_TURNS = 3
REPAIR_CHANNEL_WEIGHT = 1.0
R0_CHANNEL_WEIGHT = 0.0
CHANNEL_WEIGHT_VERSION = "clean29529_g016_no_r0_channel_v1"
CONTROLLER_ACTION_AUX_WEIGHT = 1.0
CONTROLLER_ACTION_AUX_VERSION = "clean29529_g016_alias_aware_token_local_decision_v1"
GENEVAL_REQUEST_CHUNK_SIZE = 64
GENEVAL_MAX_ATTEMPTS_PER_CHUNK = 3
# The trainer asserts the reward service's identity on every response. The
# default is the six-family service in scripts/serve_clean29529_f01_geneval.py.
# The PROTOCOL version below is deliberately NOT selectable: every service must
# speak the same wire format.
GENEVAL_EXPECTED_SERVICE_VERSION = os.environ.get(
    "G016_DETECTOR_SERVICE_VERSION",
    "geneval_sixfamily_graded_detector_service_v1",
)
GENEVAL_EXPECTED_PROTOCOL_VERSION = "flow_grpo_geneval_pickle_18085_v1"


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
    """Bounded chunk transport bound to the G016 detector identity."""

    if len(images) != len(metadata) or not images:
        raise ValueError("G016 detector image/metadata coverage is invalid")
    if chunk_size <= 0 or max_attempts <= 0:
        raise ValueError("G016 detector chunk/retry bounds must be positive")
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
                last_error = f"HTTP {status}: {getattr(response, 'text', '')[:1000]}"
                if status < 500 or attempt == int(max_attempts):
                    break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == int(max_attempts):
                    break
            time.sleep(float(attempt))
        if result is None:
            raise RuntimeError(
                f"G016 detector chunk {start}:{stop} failed after "
                f"{max_attempts} attempts: {last_error}"
            )
        if (
            result.get("service_version") != GENEVAL_EXPECTED_SERVICE_VERSION
            or result.get("protocol_version") != GENEVAL_EXPECTED_PROTOCOL_VERSION
        ):
            raise RuntimeError("G016 detector service/protocol identity mismatch")
        chunk_scores = [float(value) for value in result.get("scores") or []]
        chunk_strict = [bool(value) for value in result.get("strict_rewards") or []]
        chunk_results = list(result.get("results") or [])
        if not (
            len(chunk_scores)
            == len(chunk_strict)
            == len(chunk_results)
            == stop - start
        ):
            raise RuntimeError("G016 detector chunk response coverage mismatch")
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


class FlowGRPOFullTrajectoryRollout(_G008ProtocolRollout):
    """G016 exact protocol with generation-time sampled-token span lineage."""

    # Every rank executes 3 controller calls and 3 repair calls in the same
    # order. Terminated paths use inactive dummy calls whose records are
    # discarded, preventing variable trajectories from reordering FSDP units.
    uniform_collective_order = True

    # G022 item 3 / doc Appendix A-1. When True, a span capture that fails
    # because of what the model wrote returns status "unparseable" instead of
    # raising, and the round is classified and reward-penalised downstream.
    # `g016_sampled_spans._token_span_for_chars` raises on three conditions --
    # empty character span, no exact token-boundary enclosure, ambiguous
    # span/token lineage -- all reachable from model text whose field values do
    # not land on token boundaries. G016-G021 keep the raising behaviour.
    classify_span_capture_failures = False

    def capture_sampled_field_spans(self, sampled_token_ids: Sequence[int]) -> dict[str, Any]:
        if self.classify_span_capture_failures:
            from flow_grpo.g022_token_routing import (
                capture_sampled_field_spans_classified,
            )

            return capture_sampled_field_spans_classified(
                tokenizer=self.controller_inferencer.tokenizer,
                sampled_token_ids=sampled_token_ids,
            )
        return capture_sampled_field_spans(
            tokenizer=self.controller_inferencer.tokenizer,
            sampled_token_ids=sampled_token_ids,
        )

    @torch.no_grad()
    def generate_detached_r0(
        self,
        *,
        prompt: str,
        generator: torch.Generator | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Generate R0 and immediately discard all SDE policy records.

        Distributed FSDP ranks call this with the same prompt/seed.  The
        trainer verifies byte-identical replicas and retains/broadcasts one
        canonical JPEG.  No returned object contains R0 latent, log-probability,
        timestep, advantage, action, or optimizer state.
        """

        started = time.monotonic()
        # Upstream single-image SDE sampling uses the process-global CUDA RNG
        # (unlike its batched path). Save each rank's behavior RNG, pin only
        # this detached environment call to one logical seed, then restore the
        # rank-local stream before controller/repair sampling.
        seed = int(generator.initial_seed()) if generator is not None else 0
        python_rng_state = random.getstate()
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state(self.accelerator.device)
            if torch.cuda.is_available()
            else None
        )
        random.seed(0)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        try:
            outputs = self.flow_inferencer.interleave_inference(
                [str(prompt)],
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
            # R0 is one logical distributed generation. Upstream uses
            # `process_index` only to choose an SDE window; pin that sampler
            # coordinate across FSDP ranks so every replica decodes the exact
            # same bytes while all ranks still participate in model collectives.
            accelerator=SimpleNamespace(
                process_index=0, device=self.accelerator.device
            ),
            noise_level=stage_noise_level(
                    self.grpo_config, GENERATION_STAGE
                ),
                generators=None if generator is None else [generator],
            )
        finally:
            random.setstate(python_rng_state)
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(
                    cuda_rng_state, device=self.accelerator.device
                )
        if len(outputs) != 1 or not isinstance(outputs[0], dict):
            raise RuntimeError("G016 detached R0 call returned invalid payload")
        output = outputs[0]
        image_tensor = output.get("image")
        if not torch.is_tensor(image_tensor):
            raise RuntimeError("G016 detached R0 call returned no image")
        image = _tensor_image_to_pil(image_tensor)
        # The upstream sampler computes transition diagnostics during denoise;
        # G016 destroys them here before constructing any policy record.  Only
        # canonical image bytes cross the environment/policy boundary.
        sampled_latent_count = len(output.get("all_latents") or [])
        sampled_logprob_count = len(output.get("all_log_probs") or [])
        output.clear()
        del outputs, output, image_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        evidence = {
            "version": DETACHED_R0_VERSION,
            "logical_image_count": 1,
            "returned_policy_record_count": 0,
            "returned_flow_logprob_count": 0,
            "returned_flow_latent_count": 0,
            "returned_timestep_count": 0,
            "reward": None,
            "advantage": None,
            "policy_active": False,
            "optimizer_contribution": 0.0,
            "sampler_internal_latents_destroyed": sampled_latent_count,
            "sampler_internal_logprobs_destroyed": sampled_logprob_count,
            "elapsed_sec": time.monotonic() - started,
        }
        return image, evidence

    def generate_full_trajectory_batch(self, **kwargs: Any) -> list[MultiroundTrajectory]:
        trajectories = super().generate_from_anchor_batch(**kwargs)
        for trajectory in trajectories:
            trajectory.metadata.update(
                {
                    "version": ROLLOUT_VERSION,
                    "full_trajectory_before_credit": True,
                    "same_r0_k28": True,
                    "branch_selection_used": False,
                    "best_of_k_used": False,
                    "per_round_sibling_branching_used": False,
                    "r0_policy_record_present": False,
                    "r0_reward_channel_present": False,
                    "trajectory_pruned": False,
                }
            )
        return trajectories


# Expected public name used by the dedicated trainer.
FlowGRPOMultiroundRollout = FlowGRPOFullTrajectoryRollout


def _collective_pad_policy_decoder(inferencer: Any, count: int) -> None:
    if count <= 0:
        return
    if int(count) % 3:
        raise ValueError("G016 context padding must cover complete text/image triples")
    language_model = inferencer.model.language_model
    sentinel = torch.zeros(
        (), device=inferencer.model.vae2llm.weight.device, dtype=torch.float32
    )
    # One prior controller round contributes canonical text (und), image VAE
    # context (gen), then image ViT context (und). Enter through the wrapped
    # root so nested child handles and the root handle have exactly the same
    # order as rebuilding the real context on peer ranks.
    traversal_modes = ("und", "gen", "und") * (int(count) // 3)
    for collective_mode in traversal_modes:
        language_model(
            mode="collective_noop",
            collective_mode=collective_mode,
            sentinel=sentinel,
        )


def _forensic_token_topk_bounded(
    *,
    policy_inferencer: Any,
    turn: Any,
    behavior_old_full: torch.Tensor,
    new_full: torch.Tensor,
    reference_full: torch.Tensor,
    base_mask: Sequence[bool],
    action_mask: Sequence[bool],
    repair_mask: Sequence[bool],
    action_advantage: float,
    repair_advantage: float,
    effective_kl_beta: float,
    bound_mode: str,
    single_head: bool,
) -> dict[str, Any]:
    """Forensic top-k that cannot stop training.

    Two separate defects put a *diagnostic* on the objective-critical path:

    * `label_sampled_tokens` re-decodes the sampled ids, and an LM-head row
      with no tokenizer symbol makes that decode raise -- which is how
      Formal400 attempt B died before its backward pass.  Under G022 the
      labels are derived from token IDs instead, so no model content is
      decoded here at all.
    * any *other* rendering failure would still have propagated.  The whole
      block is therefore bounded: a failure becomes a `diagnostic_error`
      record with no rows.

    Nothing in this function touches autograd, optimizer state or a
    collective, so the bounded and the unbounded outcome produce identical
    loss, gradients and collective order.
    """
    try:
        eos_token_id = int(policy_inferencer.new_token_ids["eos_token_id"])
        labeller = (
            label_sampled_tokens_from_ids if single_head else label_sampled_tokens
        )
        token_labels = labeller(
            tokenizer=policy_inferencer.tokenizer,
            sampled_token_ids=turn.sampled_token_ids,
            action_mask=action_mask,
            repair_mask=repair_mask,
            eos_token_id=eos_token_id,
        )
        payload = build_token_topk(
            sampled_token_ids=turn.sampled_token_ids,
            sampling_logprobs=behavior_old_full,
            train_logprobs=new_full,
            reference_logprobs=reference_full,
            loss_mask=base_mask,
            action_mask=action_mask,
            repair_mask=repair_mask,
            token_labels=token_labels,
            action_advantage=action_advantage,
            repair_advantage=repair_advantage,
            kl_beta=effective_kl_beta,
            backward_loss_scale=1.0,
            top_k=8,
            bound_mode=bound_mode,
        )
        payload["label_source"] = (
            "token_ids" if single_head else "decoded_text"
        )
        payload["diagnostic_error"] = None
        return payload
    except Exception as exc:  # bounded: a diagnostic must never halt a run
        return {
            "version": TOKEN_TOPK_VERSION,
            "top_k": 8,
            "source_token_count": len(turn.sampled_token_ids),
            "selected_token_count": 0,
            "kl_bound_mode": str(bound_mode),
            "label_source": "token_ids" if single_head else "decoded_text",
            "diagnostic_error": f"{type(exc).__name__}: {exc}",
            "rows": [],
        }


def text_turn_policy_loss(
    *,
    policy_inferencer: Any,
    reference_inferencer: Any,
    turn: TextTurnRecord,
    clip_range: float,
    kl_beta: float,
    bound_mode: str = "clamp20",
    text_head_mode: str = "g016_two_head",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Token-local G016 PPO plus full-response frozen-classic KL."""

    turn.validate()
    _collective_pad_policy_decoder(
        policy_inferencer,
        int(getattr(turn, "g016_collective_prefix_padding_traversals", 0)),
    )
    new_full = teacher_forced_text_log_probs(
        policy_inferencer,
        context_terms=turn.context_terms,
        forced_prefix_token_ids=turn.forced_prefix_token_ids,
        sampled_token_ids=turn.sampled_token_ids,
        temperature=turn.behavior_temperature,
        sampled_token_allowed_ids=turn.sampled_token_allowed_ids,
    )
    with torch.no_grad():
        reference_full = teacher_forced_text_log_probs(
            reference_inferencer,
            context_terms=turn.context_terms,
            forced_prefix_token_ids=turn.forced_prefix_token_ids,
            sampled_token_ids=turn.sampled_token_ids,
            temperature=turn.behavior_temperature,
            sampled_token_allowed_ids=turn.sampled_token_allowed_ids,
        )
    # Executed-reachability evidence for the text-action support contract.
    # A mask that is installed but never applied is indistinguishable from no
    # mask at all, so every turn -- INCLUDING collective-padding turns, whose
    # replay ran the same masked forward -- records the hash of the support it
    # actually used, and the world-consensus check downstream refuses a step
    # in which any turn is missing it.
    _support = getattr(policy_inferencer, "g022_text_action_support", None)
    text_action_support_block = None
    if _support is not None:
        _violations = _support.unsupported_positions(turn.sampled_token_ids)
        text_action_support_block = {
            "support_hash": str(_support.support_hash),
            "head_rows": int(_support.head_rows),
            "supported_count": int(_support.supported_count),
            "unsupported_count": int(_support.unsupported_count),
            "policy_replay_masked": True,
            "reference_replay_masked": (
                getattr(reference_inferencer, "g022_text_action_support", None)
                is _support
            ),
            "policy_active": bool(turn.policy_active),
            "unsupported_sampled_token_count": len(_violations),
            "unsupported_sampled_tokens": _violations[:8],
        }
    if not turn.policy_active:
        zero = new_full.float().sum() * 0.0
        difference = new_full.detach().float() - reference_full.detach().float()
        from flow_grpo.g008_counting import text_kl_delta_diagnostics

        return zero, {
            "kl_delta": text_kl_delta_diagnostics(new_full, reference_full),
            "text_action_support": text_action_support_block,
            "kl_bound_mode": str(bound_mode),
            "ratio_mean": 1.0,
            "ratio_max_abs_deviation": 0.0,
            "clip_fraction": 0.0,
            "token_count": 0,
            "policy_token_count": 0,
            "full_response_kl_token_count": 0,
            "policy_loss": 0.0,
            "kl_loss": 0.0,
            "total_loss": 0.0,
            "padding": True,
            "padding_version": PADDING_VERSION,
            "version": TEXT_POLICY_VERSION,
            "controller_action_aux": {
                "version": CONTROLLER_ACTION_AUX_VERSION,
                "weight": CONTROLLER_ACTION_AUX_WEIGHT,
                "active": False,
                "token_count": 0,
                "policy_loss": 0.0,
            },
            "reference_log_probs_byte_identical": torch.equal(
                new_full.detach(), reference_full.detach()
            ),
            "reference_log_probs_max_abs_difference": float(
                difference.abs().max().item()
            ),
        }
    action_advantage = float(turn.local_advantage) if turn.local_advantage is not None else float(turn.advantage)
    repair_advantage = float(getattr(turn, "g016_repair_advantage", 0.0))
    old_full = new_full.detach().float()
    behavior_old_full = torch.tensor(turn.old_log_probs,device=new_full.device,dtype=torch.float32)
    base_mask = turn.credited_token_mask()
    action_mask = getattr(turn, "g016_action_token_mask", None) or list(base_mask)
    repair_mask = getattr(turn, "g016_repair_payload_token_mask", None) or [False]*len(base_mask)
    for label,mask in (("action",action_mask),("repair",repair_mask)):
        if len(mask)!=len(base_mask) or any(x and not y for x,y in zip(mask,base_mask)):
            raise RuntimeError(f"G016 {label}/base token mask lineage differs")
    if any(a and r for a,r in zip(action_mask,repair_mask)):
        raise RuntimeError("G016 action and repair token masks overlap")
    action_indexes=[i for i,x in enumerate(action_mask) if x]
    repair_indexes=[i for i,x in enumerate(repair_mask) if x]
    kl_indexes=[i for i,x in enumerate(base_mask) if x]
    if not action_indexes or not kl_indexes:raise RuntimeError("G016 active text turn has empty action or KL mask")
    from flow_grpo.g008_counting import (
        clipped_text_ppo_loss,
        text_k3_kl_loss,
        text_kl_delta_diagnostics,
    )
    def head(indexes,advantage):
        if not indexes:return new_full.float().sum()*0.0,{"ratio_mean":1.0,"ratio_max_abs_deviation":0.0,"clip_fraction":0.0,"token_count":0}
        keep=torch.tensor(indexes,device=new_full.device,dtype=torch.long)
        return clipped_text_ppo_loss(new_full.index_select(0,keep),old_full.index_select(0,keep),advantage=advantage,clip_range=clip_range,bound_mode=bound_mode)
    # G023 keeps ONE union-mean PPO term (never two independently normalised
    # heads) but routes a PER-TOKEN advantage vector, because on an exact-state
    # round the ACTION token and the payload tokens no longer carry the same
    # number. On exact rounds the payload span is inactive and is excluded from
    # the union rather than included at zero: including ~20 zero-advantage
    # payload tokens would divide the single action token's gradient by 21.
    g023_per_token = str(text_head_mode) == "g023_union_per_token"
    # G024 deliberately adds NO new arithmetic. With an empty repair mask the
    # proven `g022_single_head` branch already computes exactly what G024
    # specifies -- one union (= the action mask), one mean, one scalar
    # advantage. The mode exists only so that contract can be ASSERTED here
    # rather than left to emerge from a degenerate case.
    g024_uniform = str(text_head_mode) == "g024_uniform"
    # G025 = G024's uniform span plus ONE overridden token. The span is
    # identical, so it shares every branch below; the only difference is that
    # the ACTION token of an exact-state EDIT carries its own credit instead of
    # the trajectory scalar.
    g025_override = str(text_head_mode) == "g025_uniform_exact_edit_override"
    # G026 = G024's uniform base objective, UNCHANGED, plus a separate bounded
    # PPO term on the ACTION value span. G025 replaced the base credit; that
    # was measured to cover the entire policy span (overridden_token_count ==
    # policy_token_count in 141/141 live exact-edit rounds), so it painted
    # THINKING / SCORE / protocol tokens negative -- the surface G024 proved
    # must stay positively credited. G026 leaves the base alone.
    g026_auxiliary = str(text_head_mode) == "g026_uniform_plus_action_auxiliary"
    single_head = (
        str(text_head_mode) == "g022_single_head" or g023_per_token or g024_uniform or g025_override or g026_auxiliary
    )
    if g023_per_token:
        repair_credited = bool(
            getattr(turn, "g023_repair_active", True)
        ) and repair_indexes
        single_indexes = sorted(
            set(action_indexes) | (set(repair_indexes) if repair_credited else set())
        )
        action_set = set(action_indexes)
        per_token_advantage = torch.tensor(
            [
                action_advantage if index in action_set else repair_advantage
                for index in single_indexes
            ],
            device=new_full.device,
            dtype=torch.float32,
        )
        action_ppo, action_diag = head(single_indexes, per_token_advantage)
        repair_ppo = new_full.float().sum() * 0.0
        repair_diag = {
            "ratio_mean": 1.0, "ratio_max_abs_deviation": 0.0,
            "clip_fraction": 0.0, "token_count": 0,
            "folded_into_single_head": True,
        }
        action_diag = {
            **action_diag,
            "single_head_token_count": len(single_indexes),
            "g023_per_token_advantage": True,
            "g023_repair_span_credited": bool(repair_credited),
            "g023_distinct_advantage_values": sorted(
                {round(float(v), 6) for v in per_token_advantage.tolist()}
            ),
        }
        single_head_action_aux = {
            "version": CONTROLLER_ACTION_AUX_VERSION,
            "weight": CONTROLLER_ACTION_AUX_WEIGHT,
            "active": True, "single_head_folded": True,
            "covers": "action_and_payload_union_per_token",
            "token_count": len(single_indexes),
            "action_only_token_count": len(action_indexes),
            "payload_only_token_count": len(repair_indexes) if repair_credited else 0,
            "policy_loss": float(action_ppo.detach().item()),
        }
        single_head_repair_block = {
            "version": "clean29529_g016_repair_payload_head_v1",
            "active": False, "single_head_folded": True,
            "folded_into": "controller_action_aux",
            "token_count": 0,
            "payload_token_count_in_single_head": (
                len(repair_indexes) if repair_credited else 0
            ),
            "policy_loss": 0.0, "policy_loss_is_a_separate_term": False,
        }
    elif single_head:
        if g024_uniform or g025_override:
            # The entire G024 contract, checked on every turn:
            #   (1) nothing is routed to a second head, and
            #   (2) the PPO mask IS the credited-token mask -- every position
            #       the model sampled, none it did not.
            # G023 died precisely because (2) was false: THINKING, SCORE and
            # every structural token sat in neither mask, held only by
            # text_kl_beta=1e-4, and the stop protocol drifted from step 11.
            if repair_indexes:
                raise RuntimeError(
                    "G024 uniform credit received a non-empty repair mask "
                    f"({len(repair_indexes)} tokens)"
                )
            if action_indexes != kl_indexes:
                raise RuntimeError(
                    "G024 PPO mask is not identical to the credited-token mask "
                    f"(ppo={len(action_indexes)}, credited={len(kl_indexes)})"
                )
        # ------------------------------------------------------------------
        # G022 item 1: ONE head.
        #
        # Doc section 2.4 requires one advantage broadcast to every
        # policy-active token. Under G022 the action head and the repair head
        # receive the SAME trajectory scalar, so splitting them performs no
        # credit assignment at all -- it only reweights tokens, and badly:
        #
        #   * `mean(action) + mean(payload)` weights a token by 1/N of its own
        #     span. Measured on G021-A the ACTION span is a median of 1 token
        #     and the PAYLOAD span a median of 16, so the single action token
        #     carries ~16x the per-token gradient of a payload token.
        #   * that ratio is N_payload / N_action, which the MODEL controls: a
        #     longer payload dilutes its own payload gradient. Free hack.
        #   * it amplifies exactly the round-count decision that Appendix H-5
        #     measured is already collapsing.
        #
        # One PPO term over the union of the routed spans, one mean, one
        # advantage. The union is used rather than the full credited mask so
        # thinking/SCORE tokens stay outside policy credit, preserving the
        # token-routing contract's `score_or_thinking_broadcast: False`.
        # ------------------------------------------------------------------
        single_indexes = sorted(set(action_indexes) | set(repair_indexes))
        if g026_auxiliary:
            # Base objective: identical to G024's, one mean over the union with
            # the trajectory scalar. Nothing is overridden.
            action_ppo, action_diag = head(single_indexes, action_advantage)
            aux_active = bool(getattr(turn, "g026_action_auxiliary_active", False))
            aux_adv = float(getattr(turn, "g026_action_span_advantage", 0.0))
            eta = float(getattr(turn, "g026_action_auxiliary_eta", 0.0))
            aux_indexes = sorted(set(action_indexes)) if aux_active else []
            if aux_active and aux_indexes and eta > 0.0:
                # A SECOND, bounded term on the same span -- not a second
                # independently normalised head competing with the first: its
                # weight is fixed at eta and it carries a constant +/-0.5, so
                # it cannot dominate. Sized against the worst live case: the
                # shortest exact-edit round was 35 tokens, giving the base
                # objective at most 1/35 = 0.029 on the action token, while
                # eta*0.5 = 0.05 here.
                aux_ppo, aux_diag = head(aux_indexes, aux_adv)
                action_ppo = action_ppo + float(eta) * aux_ppo
            else:
                aux_diag = {"token_count": 0}
            action_diag = {
                **action_diag,
                "g026_action_auxiliary_active": aux_active,
                "g026_action_span_advantage": aux_adv,
                "g026_action_auxiliary_eta": eta,
                "g026_action_auxiliary_token_count": int(aux_diag.get("token_count", 0)),
                "g026_round_class": str(getattr(turn, "g026_round_class", "")),
                "g026_base_credit_overridden": False,
            }
        elif g025_override and getattr(turn, "g025_action_override_applied", False):
            # Exactly one token differs from the scalar. Build a per-token
            # vector rather than a second head: two independently normalised
            # heads is what G023 was told never to reintroduce.
            override = float(getattr(turn, "g025_action_token_credit", action_advantage))
            action_set = set(action_indexes)
            per_token = torch.tensor(
                [
                    override if index in action_set else action_advantage
                    for index in single_indexes
                ],
                device=new_full.device,
                dtype=torch.float32,
            )
            action_ppo, action_diag = head(single_indexes, per_token)
            action_diag = {
                **action_diag,
                "g025_action_override_applied": True,
                "g025_action_token_credit": override,
                "g025_overridden_token_count": len(action_set),
                "g025_round_class": str(getattr(turn, "g025_round_class", "")),
            }
        else:
            action_ppo, action_diag = head(single_indexes, action_advantage)
            if g025_override:
                action_diag = {
                    **action_diag,
                    "g025_action_override_applied": False,
                    "g025_round_class": str(getattr(turn, "g025_round_class", "")),
                }
        repair_ppo = new_full.float().sum() * 0.0
        repair_diag = {
            "ratio_mean": 1.0,
            "ratio_max_abs_deviation": 0.0,
            "clip_fraction": 0.0,
            "token_count": 0,
            "folded_into_single_head": True,
        }
        action_diag = {**action_diag, "single_head_token_count": len(single_indexes)}
        if g024_uniform:
            action_diag = {
                **action_diag,
                "g024_uniform_credit": True,
                "g024_ppo_mask_equals_credited_mask": True,
                "g024_advantage": float(action_advantage),
                "g024_credited_token_count": len(single_indexes),
            }
        # C3 / finding Q-11. The two diagnostic blocks below used to describe
        # the two-head design that no longer runs: `controller_action_aux`
        # reported the COMBINED action+payload loss under an action-only token
        # count, and `repair_payload_head` reported `policy_loss: 0.0` beside a
        # non-zero payload token count -- i.e. it said the payload tokens
        # contributed nothing when in fact they were inside the single term.
        # The gradient was right and the report was wrong, which is the harder
        # kind to notice. Built explicitly here instead.
        single_head_action_aux = {
            "version": CONTROLLER_ACTION_AUX_VERSION,
            "weight": CONTROLLER_ACTION_AUX_WEIGHT,
            "active": True,
            "single_head_folded": True,
            "covers": "action_and_payload_union",
            "token_count": len(single_indexes),
            "action_only_token_count": len(action_indexes),
            "payload_only_token_count": len(repair_indexes),
            "policy_loss": float(action_ppo.detach().item()),
        }
        single_head_repair_block = {
            "version": "clean29529_g016_repair_payload_head_v1",
            "active": False,
            "single_head_folded": True,
            "folded_into": "controller_action_aux",
            "token_count": 0,
            "payload_token_count_in_single_head": len(repair_indexes),
            "policy_loss": 0.0,
            "policy_loss_is_a_separate_term": False,
        }
    else:
        action_ppo,action_diag=head(action_indexes,action_advantage)
        repair_ppo,repair_diag=head(repair_indexes,repair_advantage)
    kl_keep=torch.tensor(kl_indexes,device=new_full.device,dtype=torch.long)
    new_kl=new_full.index_select(0,kl_keep);reference_kl=reference_full.index_select(0,kl_keep)
    effective_kl_beta = (
        0.0 if getattr(turn, "g016_collective_only_world_padding", False)
        else kl_beta
    )
    kl=text_k3_kl_loss(new_kl,reference_kl,beta=effective_kl_beta,bound_mode=bound_mode);loss=action_ppo+repair_ppo+kl
    # Doc section 5 part 3: the hard runtime stop needs max |delta| whether or
    # not beta is zero, so it is measured here and consumed at the post-update
    # world-consensus point rather than raising mid-backward on one rank.
    kl_delta_diagnostics=text_kl_delta_diagnostics(new_kl,reference_kl)
    behavior_keep=torch.tensor(action_indexes,device=new_full.device,dtype=torch.long)
    behavior_ratio=torch.exp(old_full.index_select(0,behavior_keep)-behavior_old_full.index_select(0,behavior_keep))
    difference=new_kl.detach().float()-reference_kl.detach().float();token_audit=getattr(turn,"g016_token_routing_audit",None)
    forensic_token_topk = _forensic_token_topk_bounded(
        policy_inferencer=policy_inferencer,
        turn=turn,
        behavior_old_full=behavior_old_full,
        new_full=new_full,
        reference_full=reference_full,
        base_mask=base_mask,
        action_mask=action_mask,
        repair_mask=repair_mask,
        action_advantage=action_advantage,
        repair_advantage=repair_advantage,
        effective_kl_beta=effective_kl_beta,
        bound_mode=bound_mode,
        single_head=single_head,
    )
    return loss,{
        **action_diag,"version":TEXT_POLICY_VERSION,"policy_loss":float((action_ppo+repair_ppo).detach().item()),"kl_loss":float(kl.detach().item()),"total_loss":float(loss.detach().item()),
        "effective_advantage":action_advantage,"action_head_advantage":action_advantage,"repair_head_advantage":repair_advantage,
        "policy_token_count":len(action_indexes)+len(repair_indexes),"action_token_count":len(action_indexes),"repair_payload_token_count":len(repair_indexes),"full_response_kl_token_count":len(kl_indexes),
        "independent_head_normalization": not single_head,"text_head_mode":str(text_head_mode),"token_local_decision":action_mask!=base_mask,"token_routing_audit":token_audit,
        "controller_action_aux":(single_head_action_aux if single_head else {"version":CONTROLLER_ACTION_AUX_VERSION,"weight":CONTROLLER_ACTION_AUX_WEIGHT,"active":True,"token_count":len(action_indexes),"policy_loss":float(action_ppo.detach().item())}),
        "repair_payload_head":(single_head_repair_block if single_head else {"version":"clean29529_g016_repair_payload_head_v1","active":bool(repair_indexes),"token_count":len(repair_indexes),"policy_loss":float(repair_ppo.detach().item())}),
        "host_forced_token_count":sum(1 for x in base_mask if not x),"ppo_old_log_prob_source":"first_inner_same_forward_detached",
        "sampler_vs_packed_ratio_max_abs_deviation":float((behavior_ratio.float()-1).abs().max().item()),
        "reference_log_probs_byte_identical":torch.equal(new_kl.detach(),reference_kl.detach()),"reference_log_probs_max_abs_difference":float(difference.abs().max().item()),
        "forensic_token_topk": forensic_token_topk,
        "text_action_support": text_action_support_block,
        "kl_delta": kl_delta_diagnostics,
        "kl_bound_mode": str(bound_mode),
    }

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
    bound_mode: str = "clamp20",
    text_head_mode: str = "g016_two_head",
) -> dict[str, Any]:
    loss, diagnostics = text_turn_policy_loss(
        policy_inferencer=policy_inferencer,
        reference_inferencer=reference_inferencer,
        turn=turn,
        clip_range=clip_range,
        kl_beta=kl_beta,
        bound_mode=bound_mode,
        text_head_mode=text_head_mode,
    )
    accelerator.backward(loss * float(backward_loss_scale))
    forensic = diagnostics.get("forensic_token_topk")
    if isinstance(forensic, dict):
        forensic = {**forensic, "backward_loss_scale": float(backward_loss_scale)}
        forensic["rows"] = [
            {
                **row,
                "preclip_dlogp": {
                    key: float(value) * float(backward_loss_scale)
                    for key, value in row["preclip_dlogp"].items()
                },
                "ranking_score": float(row["ranking_score"])
                * float(backward_loss_scale),
            }
            for row in forensic["rows"]
        ]
        diagnostics = {**diagnostics, "forensic_token_topk": forensic}
    return {
        **diagnostics,
        "collective_only_padding": not turn.policy_active,
        "real_policy_reference_padding": not turn.policy_active,
        "collective_noop_count": 0,
        "grad_norm": 0.0,
        "max_grad_norm": float(max_grad_norm),
        "backward_loss_scale": float(backward_loss_scale),
    }


def virtual_adamw_movement(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Evaluate one exact next AdamW update in temporary tensors only.

    Parameters and optimizer state are never mutated. The temporary delta for
    one parameter is released before visiting the next parameter, making this
    suitable for the no-commit probe without another model-sized snapshot.
    """

    groups = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    parameter_sq = 0.0
    gradient_sq = 0.0
    delta_sq = 0.0
    active_parameters = 0
    active_numel = 0
    nonfinite_gradient_tensors = 0
    nonfinite_delta_tensors = 0
    grouped: dict[str, dict[str, float]] = defaultdict(
        lambda: {"parameter_sq": 0.0, "gradient_sq": 0.0, "delta_sq": 0.0, "active_parameters": 0.0, "active_numel": 0.0}
    )
    for _name, parameter in named_parameters:
        if _name.startswith("model.layers."):
            group_name = "layer_" + _name.split(".")[2]
        elif _name.startswith("flow_root."):
            group_name = ".".join(_name.split(".")[:2])
        else:
            group_name = "other"
        detached = parameter.detach().float()
        current_parameter_sq = float(torch.sum(detached * detached).item())
        parameter_sq += current_parameter_sq
        grouped[group_name]["parameter_sq"] += current_parameter_sq
        gradient = parameter.grad
        if gradient is None:
            continue
        grad = gradient.detach().float()
        if not bool(torch.isfinite(grad).all().item()):
            nonfinite_gradient_tensors += 1
            continue
        group = groups[id(parameter)]
        state = optimizer.state.get(parameter, {})
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        eps = float(group.get("eps", 1e-8))
        learning_rate = float(group["lr"])
        weight_decay = float(group.get("weight_decay", 0.0))
        maximize = bool(group.get("maximize", False))
        if maximize:
            grad = -grad
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        old_step = state.get("step", 0)
        step_value = (
            float(old_step.detach().cpu().item())
            if torch.is_tensor(old_step)
            else float(old_step)
        )
        next_step = step_value + 1.0
        previous_avg = (
            torch.zeros_like(grad)
            if exp_avg is None
            else exp_avg.detach().to(device=grad.device, dtype=torch.float32)
        )
        previous_sq = (
            torch.zeros_like(grad)
            if exp_avg_sq is None
            else exp_avg_sq.detach().to(device=grad.device, dtype=torch.float32)
        )
        next_avg = previous_avg.mul(beta1).add(grad, alpha=1.0 - beta1)
        next_sq = previous_sq.mul(beta2).addcmul(
            grad, grad, value=1.0 - beta2
        )
        if bool(group.get("amsgrad", False)):
            previous_max = state.get("max_exp_avg_sq")
            max_sq = (
                next_sq
                if previous_max is None
                else torch.maximum(
                    previous_max.detach().to(device=grad.device, dtype=torch.float32),
                    next_sq,
                )
            )
            variance = max_sq
        else:
            variance = next_sq
        bias1 = 1.0 - beta1**next_step
        bias2 = 1.0 - beta2**next_step
        adaptive = next_avg.div(variance.sqrt().div(math.sqrt(bias2)).add(eps))
        delta = adaptive.mul(-learning_rate / bias1)
        if weight_decay:
            delta = delta.add(detached, alpha=-learning_rate * weight_decay)
        if not bool(torch.isfinite(delta).all().item()):
            nonfinite_delta_tensors += 1
            continue
        current_gradient_sq = float(torch.sum(grad * grad).item())
        current_delta_sq = float(torch.sum(delta * delta).item())
        gradient_sq += current_gradient_sq
        delta_sq += current_delta_sq
        active_parameters += 1
        active_numel += int(parameter.numel())
        grouped[group_name]["gradient_sq"] += current_gradient_sq
        grouped[group_name]["delta_sq"] += current_delta_sq
        grouped[group_name]["active_parameters"] += 1
        grouped[group_name]["active_numel"] += int(parameter.numel())
    parameter_norm = math.sqrt(parameter_sq)
    gradient_norm = math.sqrt(gradient_sq)
    delta_norm = math.sqrt(delta_sq)
    return {
        "version": "clean29529_g016_virtual_adamw_movement_v1",
        "method": "exact_next_adamw_delta_temporary_tensors_no_mutation",
        "parameter_norm_l2": parameter_norm,
        "gradient_norm_l2": gradient_norm,
        "virtual_delta_norm_l2": delta_norm,
        "virtual_relative_movement": (
            delta_norm / parameter_norm if parameter_norm else None
        ),
        "active_parameter_count": active_parameters,
        "active_parameter_numel": active_numel,
        "nonfinite_gradient_tensor_count": nonfinite_gradient_tensors,
        "nonfinite_virtual_delta_tensor_count": nonfinite_delta_tensors,
        "parameter_or_optimizer_mutated": False,
        "surface_groups": {
            name: {
                "parameter_norm_l2": math.sqrt(values["parameter_sq"]),
                "gradient_norm_l2": math.sqrt(values["gradient_sq"]),
                "virtual_delta_norm_l2": math.sqrt(values["delta_sq"]),
                "virtual_relative_movement": (
                    math.sqrt(values["delta_sq"]) / math.sqrt(values["parameter_sq"])
                    if values["parameter_sq"] else None
                ),
                "active_parameter_count": int(values["active_parameters"]),
                "active_parameter_numel": int(values["active_numel"]),
            }
            for name, values in sorted(grouped.items())
        },
    }


def synchronize_unmanaged_gradients(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    prefix: str = "",
    marker: str = "",
    process_group: Any,
) -> dict[str, Any]:
    """Average replicated non-FSDP flow-root gradients across policy ranks."""
    selected = []
    world_size = (
        dist.get_world_size(group=process_group)
        if dist.is_available() and dist.is_initialized()
        else 1
    )
    for name, parameter in named_parameters:
        if (
            (prefix and not name.startswith(prefix))
            or (marker and marker not in name)
            or parameter.grad is None
        ):
            continue
        gradient = parameter.grad
        if world_size > 1:
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=process_group)
            gradient.div_(world_size)
        selected.append(name)
    return {
        "version": "clean29529_g016_replicated_flow_root_grad_sync_v1",
        "prefix": prefix,
        "marker": marker,
        "world_size": world_size,
        "synchronized_tensor_count": len(selected),
        "synchronized_names": selected,
    }


def _global_activity(local_count: int, *, device: torch.device, process_group: Any) -> int:
    count = torch.tensor(int(local_count), device=device, dtype=torch.int64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=process_group)
    return int(count.item())


def bound_world_padding_records(
    records: Sequence[Any], maximum: int
) -> tuple[list[Any], int]:
    """Fit zero-credit world padding to the semantic K=28 collective width.

    ``maximum`` is computed from the real K=28 trajectories. Ranks 28..31
    independently roll padding-only trajectories so they can enter the same
    FSDP forwards, but their sampled number of turns/repairs is not semantic
    data and may exceed the K=28 maximum. Only records already marked as
    collective-only world padding may be dropped. A real semantic overflow
    remains a hard error.
    """
    rows = list(records)
    maximum = int(maximum)
    if maximum < 0:
        raise ValueError("G016 collective padding maximum is invalid")
    if len(rows) <= maximum:
        return rows, 0
    if not all(
        getattr(record, "g016_collective_only_world_padding", False)
        for record in rows
    ):
        raise ValueError("G016 semantic record count exceeds collective maximum")
    return rows[:maximum], len(rows) - maximum


def collective_padding_schedule(real_count: int, maximum: int) -> list[bool]:
    """True for semantic records, False for external collective padding."""
    if not 0 <= int(real_count) <= int(maximum):
        raise ValueError("G016 collective padding count is invalid")
    return [True] * int(real_count) + [False] * (int(maximum) - int(real_count))


def text_prefix_collective_padding(real_count: int, maximum: int) -> list[int]:
    schedule = collective_padding_schedule(real_count, maximum)
    return [0 if is_real else 3 * index for index, is_real in enumerate(schedule)]


def prepare_phase0_semantic_activity(
    trajectories: Sequence[MultiroundTrajectory], *, advantage: float = 0.5
) -> list[dict[str, Any]]:
    """Activate only real records; collective padding stays external."""
    inventory = []
    for local_index, trajectory in enumerate(trajectories):
        trajectory.validate()
        actionable_renders = sum(
            event.get("route_valid") is True
            and str(event.get("action", "")).casefold() == "edit"
            and event.get("image_generated") is True
            for event in trajectory.events
        )
        if not trajectory.repair_rounds == len(trajectory.flow_calls) == actionable_renders:
            raise ValueError("G016 semantic repair/flow/event counts differ")
        for record in trajectory.flow_calls:
            record.advantage = float(advantage)
            record.policy_active = True
        for turn in trajectory.text_turns:
            turn.advantage = float(advantage)
            turn.policy_active = True
        inventory.append({
            "local_index": local_index,
            "stop_reason": trajectory.stop_reason,
            "done": trajectory.done,
            "repair_rounds": trajectory.repair_rounds,
            "real_flow_call_count": len(trajectory.flow_calls),
            "actionable_render_event_count": actionable_renders,
            "actions": [event.get("action") for event in trajectory.events],
            "image_generated": [bool(event.get("image_generated")) for event in trajectory.events],
            "generation_time_sampled_field_spans": [event.get("generation_time_sampled_field_spans") for event in trajectory.events],
            "semantic_count_invariant": True,
        })
    return inventory


def update_multiround_group(
    trajectories: Sequence[MultiroundTrajectory],
    *,
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
    flow_padding_record: FlowCallRecord | None = None,
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
    """One complete-group text-before-flow update, with no R0 channel."""

    # G022 item 2: which text bound to use. Read off the config so no call site
    # has to change; G016-G021 keep the exact legacy clamp(+/-20) expression.
    text_bound_mode = str(
        getattr(getattr(grpo_config, "g016", None), "text_bound_mode", "clamp20")
    )
    text_head_mode = str(
        getattr(
            getattr(grpo_config, "g016", None), "text_head_mode", "g016_two_head"
        )
    )
    # G022 item 3: R3's explicit CE 1.0 / MSE 2.0 loss weights,
    # Appendix E-2(a) and G-2. R3 does not try to equalise gradient norms; it
    # weights the two loss terms directly and weights the diffusion side higher.
    # `None` for G016-G021 leaves both channels at their existing scale, so
    # those runs are numerically unchanged.
    channel_ce_weight = getattr(getattr(grpo_config, "g016", None), "ce_weight", None)
    channel_mse_weight = getattr(getattr(grpo_config, "g016", None), "mse_weight", None)
    text_loss_weight = 1.0 if channel_ce_weight is None else float(channel_ce_weight)
    flow_loss_weight = 1.0 if channel_mse_weight is None else float(channel_mse_weight)
    rows = list(trajectories)
    policy_module = transformer
    while hasattr(policy_module, "module"):
        policy_module = policy_module.module
    policy_base = (
        policy_module.get_base_model()
        if hasattr(policy_module, "get_base_model")
        else policy_module
    )
    policy_base.model.embed_tokens.to(
        device=accelerator.device, dtype=torch.bfloat16
    )
    policy_base.lm_head.to(
        device=accelerator.device, dtype=torch.bfloat16
    )
    if not rows:
        raise ValueError("G016 update requires local full trajectories")
    for trajectory in rows:
        trajectory.validate()
        if trajectory.metadata.get("r0_policy_record_present") is not False:
            raise RuntimeError("G016 trajectory did not prove R0 detachment")
        if any(record.call_kind != "repair" for record in trajectory.flow_calls):
            raise RuntimeError("G016 update received a non-repair flow call")
    modes = {"none", "text", "flow", "both"}
    if active_channel_mode not in modes or optimizer_step_mode not in modes:
        raise ValueError("G016 optimizer channel mode is invalid")
    effective_step_mode = optimizer_step_mode if commit_optimizer_step else "none"
    text_enabled = active_channel_mode in {"text", "both"}
    flow_enabled = active_channel_mode in {"flow", "both"}
    if effective_step_mode in {"text", "both"} and not text_enabled:
        raise ValueError("G016 text step enabled while text channel is inactive")
    if effective_step_mode in {"flow", "both"} and not flow_enabled:
        raise ValueError("G016 flow step enabled while flow channel is inactive")

    text_names = list(
        text_parameter_names or [f"text_{index}" for index in range(len(text_parameters))]
    )
    flow_names = list(
        flow_parameter_names or [f"flow_{index}" for index in range(len(flow_parameters))]
    )
    if len(text_names) != len(text_parameters) or len(flow_names) != len(flow_parameters):
        raise ValueError("G016 parameter-name coverage differs")
    text_named = tuple(zip(text_names, text_parameters))
    flow_named = tuple(zip(flow_names, flow_parameters))
    diagnostic = None
    if capture_full_diagnostics:
        diagnostic = {
            "version": "clean29529_g016_full_update_diagnostics_v1",
            "active_channel_mode": active_channel_mode,
            "optimizer_step_mode": effective_step_mode,
            "r0_channel_present": False,
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

    # Text channel: independently credited policy actions, one optimizer step.
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
    text_world_padding_trimmed_counts: list[int] = []
    for trajectory_index, trajectory in enumerate(rows):
        records = [
            record
            if text_enabled
            else replace(record, advantage=0.0, policy_active=False)
            for record in trajectory.text_turns
        ]
        records, trimmed_count = bound_world_padding_records(
            records, max_text_turns
        )
        text_world_padding_trimmed_counts.append(trimmed_count)
        padding_source = records[0] if records else trajectory.text_padding_turn
        if max_text_turns and padding_source is None:
            raise ValueError("G016 text padding source is unavailable")
        text_schedule = collective_padding_schedule(len(records), max_text_turns)
        prefix_padding = text_prefix_collective_padding(len(records), max_text_turns)
        for slot_index, is_real in enumerate(text_schedule):
            if is_real:
                if int(records[slot_index].turn_index) != slot_index:
                    raise RuntimeError("G016 real text turn/slot index differs")
                continue
            padding_record = replace(
                padding_source,
                turn_index=slot_index,
                advantage=0.0,
                policy_active=False,
            )
            padding_record.g016_collective_only_world_padding = True
            padding_record.g016_collective_prefix_padding_traversals = prefix_padding[slot_index]
            records.append(padding_record)
        text_metrics.append(
            [
                accumulate_text_turn(
                    policy_inferencer=policy_controller_inferencer,
                    reference_inferencer=reference_controller_inferencer,
                    turn=record,
                    accelerator=accelerator,
                    clip_range=text_clip_range,
                    kl_beta=text_kl_beta,
                    bound_mode=text_bound_mode,
                    text_head_mode=text_head_mode,
                    backward_loss_scale=(
                        text_weights["backward_scales"][trajectory_index]
                        * text_loss_weight
                        if record.policy_active
                        else 1.0
                    ),
                    max_grad_norm=max_grad_norm,
                )
                for record in records
            ]
        )
    text_lora_gradient_sync = synchronize_unmanaged_gradients(
        text_named,
        marker=".lora_",
        process_group=process_group,
    )
    if diagnostic is not None:
        diagnostic["pre_mask_gradient_inventory"]["text_phase"] = {
            "text": gradient_inventory(text_named),
            "flow": gradient_inventory(flow_named),
        }
    if phase_observer is not None:
        phase_observer("after_text_backward")
    text_pre_step_gate = synchronized_pre_step_gate(
        channel="text",
        metrics=text_metrics,
        named_parameters=text_named,
        globally_active=bool(global_text_active),
        device=accelerator.device,
        process_group=process_group,
    )
    text_grad_norm = (
        clip_trainable_gradients(
            accelerator=accelerator,
            parameters=text_parameters,
            max_grad_norm=max_grad_norm,
        )
        if global_text_active
        else 0.0
    )
    text_post_clip_gate = synchronized_post_clip_gate(
        channel="text",
        grad_norm=text_grad_norm,
        named_parameters=text_named,
        globally_active=bool(global_text_active),
        device=accelerator.device,
        process_group=process_group,
    )
    assert_gradients_none_or_zero(flow_named)
    if diagnostic is not None:
        diagnostic.setdefault("virtual_in_memory_step", {})["text"] = (
            virtual_adamw_movement(text_named, text_optimizer)
        )
    if commit_optimizer_step:
        if global_text_active and optimizer_step_observer is not None:
            optimizer_step_observer("text", "possible")
        text_step = step_channel_optimizer(
            text_optimizer,
            local_active=bool(local_text_active),
            activity_device=accelerator.device,
            process_group=process_group,
            inactive_named_parameters=flow_named,
        )
    else:
        clear_gradients(text_named)
        text_step = {
            "stepped": False,
            "optimizer_step_call_count": 0,
            "reason": "no_commit_probe",
            "globally_active": bool(global_text_active),
        }
    if optimizer_step_observer is not None:
        optimizer_step_observer(
            "text", "stepped" if text_step["stepped"] else "skipped"
        )
    if phase_observer is not None:
        phase_observer("after_text_phase")
    if diagnostic is not None:
        diagnostic["state_snapshots"]["after_text_phase"] = update_state_snapshot(
            text_optimizer=text_optimizer,
            flow_optimizer=flow_optimizer,
            text_named_parameters=text_named,
            flow_named_parameters=flow_named,
        )
    # Text and gen experts share each FSDP flat buffer even though optimizers
    # are channel-separated. Drop the completed text flat-gradient storage
    # before allocating the flow backward graph; semantic updates are already
    # committed (or virtually audited in no-commit mode).
    transformer.zero_grad(set_to_none=True)
    clear_gradients(text_named)
    clear_gradients(flow_named)
    policy_base.lm_head.to(device="cpu", dtype=torch.bfloat16)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Flow channel: repair calls only. Globally inactive means no flow forward
    # and no optimizer step. Active groups use an inactive repair record for
    # FSDP padding on ranks/trajectories with fewer calls; never an R0 record.
    repair_counts = [
        sum(record.policy_active and flow_enabled for record in trajectory.flow_calls)
        for trajectory in rows
    ]
    global_repair_call_count = _global_activity(
        sum(repair_counts), device=accelerator.device, process_group=process_group
    )
    repair_metrics: list[list[dict[str, Any]]] = []
    flow_world_padding_trimmed_counts = [0] * len(rows)
    flow_optimizer.zero_grad(set_to_none=True)
    if global_repair_call_count:
        local_padding = next(
            (
                record
                for trajectory in rows
                for record in trajectory.flow_calls
                if record.call_kind == "repair"
            ),
            flow_padding_record,
        )
        if local_padding is None or local_padding.call_kind != "repair":
            raise RuntimeError("G016 active flow group lacks repair-path padding")
        repair_weights = trajectory_channel_scales(
            repair_counts, device=accelerator.device, process_group=process_group
        )
        for trajectory_index, trajectory in enumerate(rows):
            records = [
                record
                if flow_enabled
                else replace(record, advantage=0.0, policy_active=False)
                for record in trajectory.flow_calls
            ]
            records, trimmed_count = bound_world_padding_records(
                records, max_flow_calls
            )
            flow_world_padding_trimmed_counts[trajectory_index] = trimmed_count
            flow_schedule = collective_padding_schedule(len(records), max_flow_calls)
            for is_real in flow_schedule:
                if is_real:
                    continue
                padding_record = replace(
                    local_padding, advantage=0.0, policy_active=False
                )
                padding_record.g016_collective_only_world_padding = True
                records.append(padding_record)
            repair_metrics.append(
                [
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
                            * flow_loss_weight
                            if record.policy_active
                            else 1.0
                        ),
                        max_grad_norm=max_grad_norm,
                    )
                    for record in records
                ]
            )
    else:
        repair_weights = {
            "local_record_counts": repair_counts,
            "local_active_trajectory_count": 0,
            "global_active_trajectory_count": 0,
            "backward_scales": [0.0] * len(repair_counts),
            "world_size": (
                dist.get_world_size(group=process_group)
                if dist.is_available() and dist.is_initialized()
                else 1
            ),
            "effective_record_weights": [0.0] * len(repair_counts),
            "effective_trajectory_weights": [0.0] * len(repair_counts),
            "global_channel_weight": 0.0,
        }
    flow_lora_gradient_sync = synchronize_unmanaged_gradients(
        flow_named,
        marker=".lora_",
        process_group=process_group,
    )
    flow_root_gradient_sync = synchronize_unmanaged_gradients(
        flow_named,
        prefix="flow_root.",
        process_group=process_group,
    )
    if diagnostic is not None:
        diagnostic["pre_mask_gradient_inventory"]["flow_phase"] = {
            "text": gradient_inventory(text_named),
            "flow": gradient_inventory(flow_named),
        }
    if phase_observer is not None:
        phase_observer("after_flow_backward")
    flow_pre_step_gate = synchronized_pre_step_gate(
        channel="flow",
        metrics=repair_metrics,
        named_parameters=flow_named,
        globally_active=bool(global_repair_call_count),
        device=accelerator.device,
        process_group=process_group,
    )
    flow_grad_norm = (
        clip_trainable_gradients(
            accelerator=accelerator,
            parameters=flow_parameters,
            max_grad_norm=max_grad_norm,
        )
        if global_repair_call_count
        else 0.0
    )
    flow_post_clip_gate = synchronized_post_clip_gate(
        channel="flow",
        grad_norm=flow_grad_norm,
        named_parameters=flow_named,
        globally_active=bool(global_repair_call_count),
        device=accelerator.device,
        process_group=process_group,
    )
    assert_gradients_none_or_zero(text_named)
    local_flow_active = sum(repair_counts)
    if diagnostic is not None:
        diagnostic.setdefault("virtual_in_memory_step", {})["flow"] = (
            virtual_adamw_movement(flow_named, flow_optimizer)
        )
    if commit_optimizer_step:
        if global_repair_call_count and optimizer_step_observer is not None:
            optimizer_step_observer("flow", "possible")
        flow_step = step_channel_optimizer(
            flow_optimizer,
            local_active=bool(local_flow_active),
            activity_device=accelerator.device,
            process_group=process_group,
            inactive_named_parameters=text_named,
        )
    else:
        clear_gradients(flow_named)
        flow_step = {
            "stepped": False,
            "optimizer_step_call_count": 0,
            "reason": "no_commit_probe",
            "globally_active": bool(global_repair_call_count),
        }
    if optimizer_step_observer is not None:
        optimizer_step_observer(
            "flow", "stepped" if flow_step["stepped"] else "skipped"
        )
    if phase_observer is not None:
        phase_observer("after_flow_phase")
    if diagnostic is not None:
        diagnostic["state_snapshots"]["after_flow_phase"] = update_state_snapshot(
            text_optimizer=text_optimizer,
            flow_optimizer=flow_optimizer,
            text_named_parameters=text_named,
            flow_named_parameters=flow_named,
        )
        snapshots = diagnostic["state_snapshots"]
        diagnostic["state_deltas"] = {
            "text_phase": update_state_delta(
                snapshots["before_update"], snapshots["after_text_phase"]
            ),
            "flow_phase": update_state_delta(
                snapshots["after_text_phase"], snapshots["after_flow_phase"]
            ),
            "whole_update": update_state_delta(
                snapshots["before_update"], snapshots["after_flow_phase"]
            ),
        }

    # Flow replay offloads ignored embeddings/head plus frozen connector,
    # ViT, and VAE to make backward fit. Restore the complete rollout surface
    # before returning so the next logical step receives CUDA inputs and CUDA
    # weights for prompt packing, conditioning, and image decode.
    policy_base.model.embed_tokens.to(
        device=accelerator.device, dtype=torch.bfloat16
    )
    policy_base.lm_head.to(device=accelerator.device, dtype=torch.bfloat16)
    flow_inferencer.model.connector.to(
        device=accelerator.device, dtype=torch.bfloat16
    )
    flow_inferencer.model.vit_model.to(
        device=accelerator.device, dtype=torch.bfloat16
    )
    flow_inferencer.vae_model.to(
        device=accelerator.device, dtype=torch.bfloat16
    )
    connector_parameter = next(flow_inferencer.model.connector.parameters())
    vit_parameter = next(flow_inferencer.model.vit_model.parameters())
    vae_parameter = next(flow_inferencer.vae_model.parameters())
    post_update_rollout_residency = {
        "embed_tokens_device": str(policy_base.model.embed_tokens.weight.device),
        "lm_head_device": str(policy_base.lm_head.weight.device),
        "connector_device": str(connector_parameter.device),
        "vit_device": str(vit_parameter.device),
        "vae_device": str(vae_parameter.device),
        "matches_accelerator_device": (
            policy_base.model.embed_tokens.weight.device == accelerator.device
            and policy_base.lm_head.weight.device == accelerator.device
            and connector_parameter.device == accelerator.device
            and vit_parameter.device == accelerator.device
            and vae_parameter.device == accelerator.device
        ),
    }
    if not post_update_rollout_residency["matches_accelerator_device"]:
        raise RuntimeError("G016 post-update rollout residency differs")

    return {
        "trajectory_count": len(rows),
        "original_r0_record_count": 0,
        "r0_flow_metrics": [],
        "r0_channel_present": False,
        "r0_gradient_contribution": 0.0,
        "cloned_r0_active_count": 0,
        "padding_version": PADDING_VERSION,
        "world_padding_width_normalization": {
            "version": "clean29529_g016_k28_collective_width_normalization_v1",
            "semantic_width_source": "real_k28_only",
            "semantic_overflow_fail_closed": True,
            "text_trimmed_counts": text_world_padding_trimmed_counts,
            "flow_trimmed_counts": flow_world_padding_trimmed_counts,
            "text_trimmed_total": sum(text_world_padding_trimmed_counts),
            "flow_trimmed_total": sum(flow_world_padding_trimmed_counts),
        },
        "flow_metrics": repair_metrics,
        "repair_flow_metrics": repair_metrics,
        "text_metrics": text_metrics,
        "text_grad_norm": text_grad_norm,
        "flow_grad_norm": flow_grad_norm,
        "text_lora_gradient_sync": text_lora_gradient_sync,
        "flow_lora_gradient_sync": flow_lora_gradient_sync,
        "flow_root_gradient_sync": flow_root_gradient_sync,
        "text_step": text_step,
        "flow_step": flow_step,
        "text_pre_step_gate": text_pre_step_gate,
        "flow_pre_step_gate": flow_pre_step_gate,
        "text_post_clip_gate": text_post_clip_gate,
        "flow_post_clip_gate": flow_post_clip_gate,
        "text_before_flow": True,
        "commit_optimizer_step": bool(commit_optimizer_step),
        "text_head_mode": str(text_head_mode),
        "ce_weight": text_loss_weight,
        "mse_weight": flow_loss_weight,
        "active_channel_mode": active_channel_mode,
        "optimizer_step_mode": effective_step_mode,
        "text_weighting": text_weights,
        "flow_weighting": {"repair": repair_weights},
        "global_repair_call_count": global_repair_call_count,
        "post_update_rollout_residency": post_update_rollout_residency,
        "full_update_diagnostics": diagnostic,
    }


__all__ = [
    "CHANNEL_WEIGHT_VERSION",
    "CONTRACT_VERSION",
    "CONTROLLER_ACTION_AUX_VERSION",
    "CONTROLLER_ACTION_AUX_WEIGHT",
    "CONTROLLER_FORCED_PREFIX_VERSION",
    "CONTROLLER_PROTOCOL_VERSION",
    "CONTROLLER_SCORE_FIELD_VERSION",
    "DETACHED_R0_VERSION",
    "FlowCallRecord",
    "FlowGRPOFullTrajectoryRollout",
    "FlowGRPOMultiroundRollout",
    "GENEVAL_EXPECTED_PROTOCOL_VERSION",
    "GENEVAL_EXPECTED_SERVICE_VERSION",
    "GENEVAL_MAX_ATTEMPTS_PER_CHUNK",
    "GENEVAL_REQUEST_CHUNK_SIZE",
    "MAX_CONTROLLER_TURNS",
    "MAX_REPAIR_ROUNDS",
    "MultiroundTrajectory",
    "PADDING_VERSION",
    "R0_CHANNEL_WEIGHT",
    "REPAIR_CHANNEL_WEIGHT",
    "REWARD_VERSION",
    "ROLLOUT_VERSION",
    "ScoreFieldSchedule",
    "SharedPrefixReference",
    "TEXT_POLICY_VERSION",
    "TextTurnRecord",
    "assert_frozen_reference_optimizer_isolation",
    "release_prompt_index",
    "bound_world_padding_records",
    "collective_padding_schedule",
    "text_prefix_collective_padding",
    "prepare_phase0_semantic_activity",
    "request_geneval_score_chunks",
    "update_multiround_group",
]
