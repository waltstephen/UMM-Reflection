# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any

from PIL import Image
import torch

from data.data_utils import pil_img2rgb
from modeling.bagel.qwen2_navit import NaiveCache



VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''

BEHAVIOR_CATEGORICAL_LOG_PROB_V1 = "behavior_categorical_log_prob_v1"
GREEDY_MODEL_LOG_PROB_V1 = "greedy_model_log_prob_v1"
CONSTRAINED_CATEGORICAL_LOG_PROB_V1 = (
    "constrained_categorical_log_prob_v1"
)


def constrained_log_probs(logits, allowed_token_ids, *, temperature=1.0):
    """Renormalize one row of logits over an explicit allowed token set.

    A single-element allowed set yields an exact `0.0` log-probability, so a
    deterministic constrained position is a host token by construction. The
    same function is used by the sampler, the PPO replay and the frozen
    reference so old/current/reference log-probs share one estimator.
    """
    allowed = [int(value) for value in allowed_token_ids]
    if not allowed:
        raise ValueError("constrained decoding needs a non-empty allowed set")
    if len(set(allowed)) != len(allowed):
        raise ValueError("constrained allowed set contains duplicates")
    if float(temperature) <= 0.0:
        raise ValueError("constrained temperature must be positive")
    scaled = logits.float() / float(temperature)
    mask = torch.full_like(scaled, float("-inf"))
    index = torch.tensor(allowed, device=scaled.device, dtype=torch.long)
    if scaled.dim() == 1:
        mask[index] = scaled[index]
    else:
        mask[..., index] = scaled[..., index]
    return torch.nn.functional.log_softmax(mask, dim=-1)


_constrained_log_probs = constrained_log_probs


def prepare_text_action_support_mask(support_mask, logits):
    """Validate a support mask against live logits and place it once.

    `support_mask` is an additive `[vocab]` row of `0.0` (supported) and
    `-inf` (no tokenizer symbol for that LM-head row).  It is converted to the
    logits' device/dtype exactly once per generation call rather than per
    decoded token.
    """
    if support_mask is None:
        return None
    if support_mask.dim() != 1:
        raise AssertionError("text-action support mask must be one row")
    if int(support_mask.shape[0]) != int(logits.shape[-1]):
        raise AssertionError(
            f"text-action support width {int(support_mask.shape[0])} != "
            f"logits width {int(logits.shape[-1])}"
        )
    if (
        support_mask.device != logits.device
        or support_mask.dtype != logits.dtype
    ):
        support_mask = support_mask.to(
            device=logits.device, dtype=logits.dtype
        )
    return support_mask


def decode_content_segment(inferencer, content_token_ids):
    """Decode a generated segment and report ids with no tokenizer symbol.

    Test doubles borrow the generation methods without inheriting from
    `InterleaveInferencer` and only provide `_decode_content_token_ids`.
    They report an empty unsupported list, which is correct for a double
    whose tokenizer is total. Every production controller inferencer
    inherits the checked variant, and the G022 world-consensus gate proves
    independently that the support contract ran on every real text turn --
    so this fallback cannot quietly stand in for the real detector.
    """
    checked = getattr(inferencer, "_decode_content_token_ids_checked", None)
    if callable(checked):
        return checked(content_token_ids)
    return inferencer._decode_content_token_ids(content_token_ids), []


def apply_text_action_support(logits, support_mask):
    """The single place the sampler's probability space is narrowed.

    Every selection and every behaviour log-prob in this file is computed from
    the return value, so sampling, greedy selection and the recorded
    log-probs share one support by construction rather than by convention.
    """
    if support_mask is None:
        return logits
    return logits + support_mask


class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids

    def _wrapped_language_model(self):
        return self.model.language_model

    def _embed_token_ids(self, input_ids):
        return self._wrapped_language_model().forward(
            mode="get_embeddings",
            input_ids=input_ids,
        )

    def _project_logits(self, hidden_states):
        return self._wrapped_language_model().forward(
            mode="get_logits",
            hidden_states=hidden_states,
        )

    def context_cache_signature(self, gen_context):
        past_key_values = gen_context["past_key_values"]
        key_shapes = []
        value_shapes = []
        layer_lengths = []
        for layer_index in range(past_key_values.num_layers):
            key = past_key_values.key_cache[layer_index]
            value = past_key_values.value_cache[layer_index]
            key_shape = None if key is None else tuple(key.shape)
            value_shape = None if value is None else tuple(value.shape)
            if (key is None) != (value is None):
                raise AssertionError(
                    f"layer {layer_index} key/value cache presence mismatch"
                )
            if key is None:
                layer_length = 0
            else:
                if key.shape[0] != value.shape[0]:
                    raise AssertionError(
                        f"layer {layer_index} key/value cache length mismatch"
                    )
                layer_length = int(key.shape[0])
            key_shapes.append(key_shape)
            value_shapes.append(value_shape)
            layer_lengths.append(layer_length)

        kv_lens = [int(value) for value in gen_context["kv_lens"]]
        ropes = [int(value) for value in gen_context["ropes"]]
        if len(kv_lens) != len(ropes):
            raise AssertionError("kv_lens/ropes length mismatch")
        expected_total = sum(kv_lens)
        if any(length != expected_total for length in layer_lengths):
            raise AssertionError(
                f"layer cache lengths {layer_lengths} do not match "
                f"kv_lens total {expected_total}"
            )
        return {
            "kv_lens": kv_lens,
            "ropes": ropes,
            "key_shapes": key_shapes,
            "value_shapes": value_shapes,
            "layer_lengths": layer_lengths,
        }

    def _model_device(self):
        return self.model.vae2llm.weight.device

    @torch.no_grad()
    def update_context_token_ids_batched(
        self,
        token_ids,
        gen_context,
        return_next_token_logits=False,
    ):
        """Append exact token IDs in one batched causal forward.

        This path is retained for diagnostics. BF16 FlashAttention/GEMM
        batching is not bit-equivalent to public tokenwise generation.
        """
        token_ids = [int(token_id) for token_id in token_ids]
        if not token_ids:
            return gen_context
        if len(gen_context["kv_lens"]) != 1:
            raise NotImplementedError(
                "exact token-ID append currently supports batch size 1"
            )
        before = self.context_cache_signature(gen_context)
        old_kv_len = before["kv_lens"][0]
        old_rope = before["ropes"][0]
        device = self._model_device()
        count = len(token_ids)
        generation_input = {
            "packed_text_ids": torch.tensor(
                token_ids,
                dtype=torch.long,
                device=device,
            ),
            "packed_text_position_ids": torch.arange(
                old_rope,
                old_rope + count,
                dtype=torch.long,
                device=device,
            ),
            "text_token_lens": torch.tensor(
                [count],
                dtype=torch.int,
                device=device,
            ),
            "packed_text_indexes": torch.arange(
                old_kv_len,
                old_kv_len + count,
                dtype=torch.long,
                device=device,
            ),
            "packed_key_value_indexes": torch.arange(
                old_kv_len,
                dtype=torch.long,
                device=device,
            ),
            "key_values_lens": torch.tensor(
                [old_kv_len],
                dtype=torch.int,
                device=device,
            ),
        }
        packed_text_embedding = self._embed_token_ids(
            generation_input["packed_text_ids"]
        )
        extra_inputs = {}
        if getattr(self.model, "use_moe", False):
            extra_inputs = {"mode": "und"}
        output = self.model.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=generation_input["text_token_lens"],
            packed_query_position_ids=(
                generation_input["packed_text_position_ids"]
            ),
            packed_query_indexes=generation_input["packed_text_indexes"],
            past_key_values=gen_context["past_key_values"],
            packed_key_value_indexes=(
                generation_input["packed_key_value_indexes"]
            ),
            key_values_lens=generation_input["key_values_lens"],
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        gen_context["past_key_values"] = output.past_key_values
        gen_context["kv_lens"] = [old_kv_len + count]
        gen_context["ropes"] = [old_rope + count]
        after = self.context_cache_signature(gen_context)
        if after["kv_lens"][0] - old_kv_len != count:
            raise AssertionError("token-ID append cache delta mismatch")
        if after["ropes"][0] - old_rope != count:
            raise AssertionError("token-ID append rope delta mismatch")
        if return_next_token_logits:
            next_token_logits = self._project_logits(
                output.packed_query_sequence[-1:]
            )
            return gen_context, next_token_logits
        return gen_context

    @torch.no_grad()
    def update_context_token_ids(
        self,
        token_ids,
        gen_context,
        return_next_token_logits=False,
        replay_mode="stepwise",
    ):
        """Append exact IDs without retokenizing, stepwise by default."""
        token_ids = [int(token_id) for token_id in token_ids]
        if replay_mode == "batched":
            return self.update_context_token_ids_batched(
                token_ids,
                gen_context,
                return_next_token_logits=return_next_token_logits,
            )
        if replay_mode != "stepwise":
            raise ValueError(f"unknown token replay mode: {replay_mode}")
        if not token_ids:
            if return_next_token_logits:
                raise ValueError(
                    "cannot return next logits for an empty token append"
                )
            return gen_context
        before = self.context_cache_signature(gen_context)
        next_token_logits = None
        for token_id in token_ids:
            next_token_logits = self._forward_persistent_token(
                token_id,
                gen_context,
            )
        after = self.context_cache_signature(gen_context)
        expected_delta = len(token_ids)
        if after["kv_lens"][0] - before["kv_lens"][0] != expected_delta:
            raise AssertionError("stepwise token-ID cache delta mismatch")
        if after["ropes"][0] - before["ropes"][0] != expected_delta:
            raise AssertionError("stepwise token-ID rope delta mismatch")
        if return_next_token_logits:
            return gen_context, next_token_logits
        return gen_context

    @torch.no_grad()
    def append_generated_content_token_ids(self, content_token_ids, gen_context):
        full_segment_token_ids = [
            int(self.new_token_ids["bos_token_id"]),
            *[int(token_id) for token_id in content_token_ids],
            int(self.new_token_ids["eos_token_id"]),
        ]
        return self.update_context_token_ids(
            full_segment_token_ids,
            gen_context,
        )

    @torch.no_grad()
    def _forward_persistent_token(self, token_id, gen_context):
        if len(gen_context["kv_lens"]) != 1:
            raise NotImplementedError(
                "persistent text generation currently supports batch size 1"
            )
        return self._forward_persistent_tokens([token_id], gen_context)

    @torch.no_grad()
    def _forward_persistent_tokens(self, token_ids, gen_context):
        token_ids = [int(token_id) for token_id in token_ids]
        batch_size = len(gen_context["kv_lens"])
        if len(token_ids) != batch_size:
            raise ValueError("persistent token/context batch cardinality mismatch")
        if batch_size < 1:
            raise ValueError("persistent token batch is empty")
        old_kv_lens = [int(value) for value in gen_context["kv_lens"]]
        old_ropes = [int(value) for value in gen_context["ropes"]]
        device = self._model_device()
        current_tokens = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=device,
        )
        packed_text_embedding = self._embed_token_ids(current_tokens)
        key_value_indexes = []
        offset = 0
        for sample_index, length in enumerate(old_kv_lens):
            key_value_indexes.extend(
                range(offset + sample_index, offset + sample_index + length)
            )
            offset += length
        extra_inputs = {}
        if getattr(self.model, "use_moe", False):
            extra_inputs = {"mode": "und"}
        output = self.model.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=torch.ones(
                batch_size,
                dtype=torch.int,
                device=device,
            ),
            packed_query_position_ids=torch.tensor(
                old_ropes,
                dtype=torch.long,
                device=device,
            ),
            packed_query_indexes=torch.tensor(
                [
                    sum(old_kv_lens[: index + 1]) + index
                    for index in range(batch_size)
                ],
                dtype=torch.long,
                device=device,
            ),
            past_key_values=gen_context["past_key_values"],
            key_values_lens=torch.tensor(
                old_kv_lens,
                dtype=torch.int,
                device=device,
            ),
            packed_key_value_indexes=torch.tensor(
                key_value_indexes,
                dtype=torch.long,
                device=device,
            ),
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        gen_context["past_key_values"] = output.past_key_values
        gen_context["kv_lens"] = [value + 1 for value in old_kv_lens]
        gen_context["ropes"] = [value + 1 for value in old_ropes]
        self.context_cache_signature(gen_context)
        return self._project_logits(
            output.packed_query_sequence
        )

    def _merge_generation_contexts(self, gen_contexts):
        contexts = list(gen_contexts)
        if not contexts:
            raise ValueError("cannot merge an empty context batch")
        signatures = [self.context_cache_signature(value) for value in contexts]
        if any(len(value["kv_lens"]) != 1 for value in signatures):
            raise ValueError("context merge requires independent batch-1 contexts")
        caches = [value["past_key_values"] for value in contexts]
        num_layers = caches[0].num_layers
        if any(cache.num_layers != num_layers for cache in caches):
            raise ValueError("context cache layer counts differ")
        merged_cache = NaiveCache(num_layers)
        for cache_name in ("key_cache", "value_cache"):
            merged_values = getattr(merged_cache, cache_name)
            per_context = [getattr(cache, cache_name) for cache in caches]
            for layer_index in range(num_layers):
                tensors = [
                    values[layer_index] for values in per_context
                ]
                if all(value is None for value in tensors):
                    merged_values[layer_index] = None
                elif any(value is None for value in tensors):
                    raise ValueError("context cache layer presence differs")
                else:
                    merged_values[layer_index] = torch.cat(tensors, dim=0)
        merged = {
            "kv_lens": [
                int(signature["kv_lens"][0]) for signature in signatures
            ],
            "ropes": [
                int(signature["ropes"][0]) for signature in signatures
            ],
            "past_key_values": merged_cache,
        }
        self.context_cache_signature(merged)
        return merged

    def _split_generation_context(self, gen_context):
        signature = self.context_cache_signature(gen_context)
        lengths = [int(value) for value in signature["kv_lens"]]
        cache = gen_context["past_key_values"]
        contexts = [
            {
                "kv_lens": [length],
                "ropes": [int(signature["ropes"][index])],
                "past_key_values": NaiveCache(cache.num_layers),
            }
            for index, length in enumerate(lengths)
        ]
        for cache_name in ("key_cache", "value_cache"):
            merged_values = getattr(cache, cache_name)
            split_values = [
                getattr(value["past_key_values"], cache_name)
                for value in contexts
            ]
            for layer_index in range(cache.num_layers):
                merged = merged_values[layer_index]
                if merged is None:
                    continue
                if int(merged.shape[0]) != sum(lengths):
                    raise ValueError("packed context cache length differs")
                for index, chunk in enumerate(merged.split(lengths, dim=0)):
                    split_values[index][layer_index] = chunk.clone()
        for value in contexts:
            self.context_cache_signature(value)
        return contexts

    def _decode_content_token_ids_checked(self, content_token_ids):
        """Decode, and return every id the tokenizer has no symbol for.

        The previous implementation dropped `None` symbols silently, so an
        LM-head row outside the tokenizer's support vanished from the text
        while remaining in `content_token_ids`.  The ids are now retained
        exactly and returned alongside the text: nothing is dropped without a
        record, coerced to UNK, or replaced by a parseable placeholder.
        """
        ids = [int(value) for value in content_token_ids]
        symbols = self.tokenizer.convert_ids_to_tokens(list(ids))
        unsupported = [
            {"position": int(index), "token_id": int(ids[index])}
            for index, token in enumerate(symbols)
            if token is None
        ]
        text = self.tokenizer.convert_tokens_to_string(
            [token for token in symbols if token is not None]
        )
        return text, unsupported

    def _decode_content_token_ids(self, content_token_ids):
        text, _ = self._decode_content_token_ids_checked(content_token_ids)
        return text

    @torch.no_grad()
    def gen_text_persistent(
        self,
        gen_context,
        max_length: int = 500,
        do_sample: bool = True,
        temperature: float = 1.0,
        return_log_probs: bool = False,
        return_next_token_logits: bool = False,
        forced_prefix_text: str = "",
        constrained_schedule=None,
        support_mask=None,
    ):
        """Generate one BOS/prefix/sampled-content/EOS segment into live KV.

        `constrained_schedule` optionally forces the first content tokens onto
        a token trie (see `ScoreFieldSchedule`). Constrained positions sample
        from the renormalized allowed set, so their recorded behaviour
        log-probs are constraint-aware and single-successor positions are
        deterministic host tokens with an exact zero log-prob.
        """
        if max_length < 0:
            raise ValueError("max_length must be non-negative")
        if do_sample and temperature <= 0:
            raise ValueError("temperature must be positive when sampling")
        if constrained_schedule is not None and max_length < (
            constrained_schedule.max_length
        ):
            raise ValueError(
                "max_length cannot truncate the constrained schedule"
            )
        before = self.context_cache_signature(gen_context)
        if len(before["kv_lens"]) != 1:
            raise NotImplementedError(
                "persistent text generation currently supports batch size 1"
            )

        bos_token_id = int(self.new_token_ids["bos_token_id"])
        eos_token_id = int(self.new_token_ids["eos_token_id"])
        forced_prefix_text = str(forced_prefix_text or "")
        if forced_prefix_text:
            try:
                encoded_prefix = self.tokenizer.encode(
                    forced_prefix_text,
                    add_special_tokens=False,
                )
            except TypeError:
                encoded_prefix = self.tokenizer.encode(forced_prefix_text)
            forced_prefix_token_ids = [
                int(token_id) for token_id in encoded_prefix
            ]
        else:
            forced_prefix_token_ids = []
        content_token_ids = []
        full_segment_token_ids = [
            bos_token_id,
            *forced_prefix_token_ids,
        ]
        selected_log_probs = []
        logits = self._forward_persistent_token(
            bos_token_id,
            gen_context,
        )
        support_mask = prepare_text_action_support_mask(support_mask, logits)
        for token_id in forced_prefix_token_ids:
            logits = self._forward_persistent_token(
                token_id,
                gen_context,
            )
        stop_reason = None
        eos_model_selected = False
        eos_host_forced = False
        content_token_allowed_ids = []
        content_token_policy_active = []
        constrained_token_ids = []

        while True:
            if max_length == 0:
                eos_host_forced = True
                stop_reason = "max_length"
                logits = self._forward_persistent_token(
                    eos_token_id,
                    gen_context,
                )
                full_segment_token_ids.append(eos_token_id)
                break

            allowed = None
            if constrained_schedule is not None and not (
                constrained_schedule.is_complete(constrained_token_ids)
            ):
                allowed = constrained_schedule.allowed_next(
                    constrained_token_ids
                )
                if not allowed:
                    raise AssertionError(
                        "constrained schedule reached a dead trie node"
                    )

            decision_logits = apply_text_action_support(logits, support_mask)
            if allowed is not None:
                behavior_log_probs = _constrained_log_probs(
                    decision_logits,
                    allowed,
                    temperature=float(temperature) if do_sample else 1.0,
                )
                if do_sample:
                    next_token = int(
                        torch.multinomial(
                            behavior_log_probs.exp(),
                            num_samples=1,
                        ).item()
                    )
                else:
                    next_token = int(
                        torch.argmax(behavior_log_probs, dim=-1).item()
                    )
            elif do_sample:
                behavior_log_probs = torch.nn.functional.log_softmax(
                    decision_logits.float() / float(temperature),
                    dim=-1,
                )
                next_token = int(
                    torch.multinomial(
                        behavior_log_probs.exp(),
                        num_samples=1,
                    ).item()
                )
            else:
                next_token = int(
                    torch.argmax(decision_logits, dim=-1).item()
                )
                behavior_log_probs = torch.nn.functional.log_softmax(
                    decision_logits.float(),
                    dim=-1,
                )

            if return_log_probs:
                selected_log_probs.append(
                    float(behavior_log_probs[0, next_token].item())
                )

            if next_token == eos_token_id:
                eos_model_selected = True
                stop_reason = "eos"
                logits = self._forward_persistent_token(
                    eos_token_id,
                    gen_context,
                )
                full_segment_token_ids.append(eos_token_id)
                break

            content_token_ids.append(next_token)
            content_token_allowed_ids.append(
                list(allowed) if allowed is not None else None
            )
            content_token_policy_active.append(
                True if allowed is None else len(allowed) > 1
            )
            if allowed is not None:
                constrained_token_ids.append(next_token)
            full_segment_token_ids.append(next_token)
            if len(content_token_ids) >= max_length:
                logits = self._forward_persistent_token(
                    next_token,
                    gen_context,
                )
                eos_host_forced = True
                stop_reason = "max_length"
                logits = self._forward_persistent_token(
                    eos_token_id,
                    gen_context,
                )
                full_segment_token_ids.append(eos_token_id)
                break
            logits = self._forward_persistent_token(
                next_token,
                gen_context,
            )

        after = self.context_cache_signature(gen_context)
        expected_delta = (
            1
            + len(forced_prefix_token_ids)
            + len(content_token_ids)
            + 1
        )
        actual_delta = after["kv_lens"][0] - before["kv_lens"][0]
        rope_delta = after["ropes"][0] - before["ropes"][0]
        if actual_delta != expected_delta:
            raise AssertionError(
                f"persistent cache delta {actual_delta} != {expected_delta}"
            )
        if rope_delta != expected_delta:
            raise AssertionError(
                f"persistent rope delta {rope_delta} != {expected_delta}"
            )
        if full_segment_token_ids != [
            bos_token_id,
            *forced_prefix_token_ids,
            *content_token_ids,
            eos_token_id,
        ]:
            raise AssertionError("persistent full segment token mismatch")

        if constrained_schedule is not None and not (
            constrained_schedule.is_complete(constrained_token_ids)
        ):
            raise AssertionError(
                "constrained schedule did not complete before EOS"
            )
        decoded_text, unsupported_tokens = decode_content_segment(
            self, content_token_ids
        )
        result = {
            "context": gen_context,
            "text": decoded_text,
            "text_action_support_masked": support_mask is not None,
            "unsupported_text_action_tokens": unsupported_tokens,
            "forced_prefix_text": forced_prefix_text,
            "forced_prefix_token_ids": forced_prefix_token_ids,
            "content_token_ids": content_token_ids,
            "content_token_allowed_ids": content_token_allowed_ids,
            "content_token_policy_active": content_token_policy_active,
            "constrained_schedule_version": (
                None
                if constrained_schedule is None
                else str(getattr(constrained_schedule, "version", ""))
            ),
            "constrained_token_ids": list(constrained_token_ids),
            "full_segment_token_ids": full_segment_token_ids,
            "stop_reason": stop_reason,
            "eos_model_selected": eos_model_selected,
            "eos_host_forced": eos_host_forced,
            "cache_delta": actual_delta,
            "kv_lens": list(after["kv_lens"]),
            "ropes": list(after["ropes"]),
            "cache_signature": after,
            "behavior_do_sample": bool(do_sample),
            "behavior_temperature": float(temperature),
            "selected_log_prob_definition": (
                BEHAVIOR_CATEGORICAL_LOG_PROB_V1
                if do_sample
                else GREEDY_MODEL_LOG_PROB_V1
            ),
        }
        if return_log_probs:
            result["selected_log_probs"] = selected_log_probs
            result["sum_log_prob"] = float(sum(selected_log_probs))
        if return_next_token_logits:
            result["next_token_logits"] = logits.detach().float().cpu()
        return result

    @torch.no_grad()
    def gen_text_persistent_batch(
        self,
        gen_contexts,
        max_length: int = 500,
        do_sample: bool = True,
        temperature: float = 1.0,
        return_log_probs: bool = False,
        return_next_token_logits: bool = False,
        forced_prefix_text: str = "",
        constrained_schedule=None,
        distributed_lockstep_sync=None,
        distributed_lockstep_noop=None,
        support_mask=None,
    ):
        """Generate aligned controller segments for independent KV contexts."""
        contexts = list(gen_contexts)
        distributed_lockstep = (
            distributed_lockstep_sync is not None
            or distributed_lockstep_noop is not None
        )
        if not contexts or (len(contexts) < 2 and not distributed_lockstep):
            raise ValueError(
                "persistent batch generation requires batch >= 2 unless distributed lockstep is active"
            )
        if max_length < 0:
            raise ValueError("max_length must be non-negative")
        if do_sample and temperature <= 0:
            raise ValueError("temperature must be positive when sampling")
        if constrained_schedule is not None and max_length < (
            constrained_schedule.max_length
        ):
            raise ValueError(
                "max_length cannot truncate the constrained schedule"
            )
        if distributed_lockstep and (
            not callable(distributed_lockstep_sync)
            or not callable(distributed_lockstep_noop)
            or max_length == 0
        ):
            raise ValueError(
                "distributed lockstep requires sync/noop callbacks and positive max_length"
            )
        before = [self.context_cache_signature(value) for value in contexts]
        bos_token_id = int(self.new_token_ids["bos_token_id"])
        eos_token_id = int(self.new_token_ids["eos_token_id"])
        forced_prefix_text = str(forced_prefix_text or "")
        if forced_prefix_text:
            try:
                encoded_prefix = self.tokenizer.encode(
                    forced_prefix_text,
                    add_special_tokens=False,
                )
            except TypeError:
                encoded_prefix = self.tokenizer.encode(forced_prefix_text)
            forced_prefix_token_ids = [
                int(token_id) for token_id in encoded_prefix
            ]
        else:
            forced_prefix_token_ids = []

        active_indexes = list(range(len(contexts)))
        active_context = (
            contexts[0]
            if len(contexts) == 1
            else self._merge_generation_contexts(contexts)
        )
        logits = self._forward_persistent_tokens(
            [bos_token_id] * len(active_indexes),
            active_context,
        )
        support_mask = prepare_text_action_support_mask(support_mask, logits)
        for token_id in forced_prefix_token_ids:
            logits = self._forward_persistent_tokens(
                [token_id] * len(active_indexes),
                active_context,
            )

        content_token_ids = [[] for _ in contexts]
        content_token_allowed_ids = [[] for _ in contexts]
        content_token_policy_active = [[] for _ in contexts]
        constrained_token_ids = [[] for _ in contexts]
        selected_log_probs = [[] for _ in contexts]
        stop_reasons = [None for _ in contexts]
        eos_model_selected = [False for _ in contexts]
        eos_host_forced = [False for _ in contexts]
        final_logits = [None for _ in contexts]
        completed_contexts = [None for _ in contexts]
        distributed_decode_traversals = 0
        distributed_dummy_traversals = 0
        distributed_forced_eos_padding_traversals = 0

        if max_length == 0:
            logits = self._forward_persistent_tokens(
                [eos_token_id] * len(active_indexes),
                active_context,
            )
            split = self._split_generation_context(active_context)
            for row, index in enumerate(active_indexes):
                stop_reasons[index] = "max_length"
                eos_host_forced[index] = True
                final_logits[index] = logits[row : row + 1]
                completed_contexts[index] = split[row]
            active_indexes = []

        while active_indexes or distributed_lockstep:
            if not active_indexes:
                # Peer ranks may still have sampled rows. Execute exactly one
                # wrapped-root dummy traversal before the global continuation
                # reduction, matching their one real token traversal.
                distributed_lockstep_noop()
                distributed_decode_traversals += 1
                distributed_dummy_traversals += 1
                global_continues, global_forced_eos = (
                    distributed_lockstep_sync(False, 0)
                )
                for _ in range(int(global_forced_eos)):
                    distributed_lockstep_noop()
                    distributed_forced_eos_padding_traversals += 1
                if not global_continues:
                    break
                continue
            row_allowed = [None for _ in active_indexes]
            if constrained_schedule is not None:
                for row, index in enumerate(active_indexes):
                    if constrained_schedule.is_complete(
                        constrained_token_ids[index]
                    ):
                        continue
                    allowed = constrained_schedule.allowed_next(
                        constrained_token_ids[index]
                    )
                    if not allowed:
                        raise AssertionError(
                            "constrained schedule reached a dead trie node"
                        )
                    row_allowed[row] = allowed
            decision_logits = apply_text_action_support(logits, support_mask)
            if do_sample:
                behavior_log_probs = torch.nn.functional.log_softmax(
                    decision_logits.float() / float(temperature),
                    dim=-1,
                )
            else:
                behavior_log_probs = torch.nn.functional.log_softmax(
                    decision_logits.float(),
                    dim=-1,
                )
            if any(value is not None for value in row_allowed):
                rows = []
                for row, allowed in enumerate(row_allowed):
                    if allowed is None:
                        rows.append(behavior_log_probs[row])
                        continue
                    rows.append(
                        constrained_log_probs(
                            decision_logits[row],
                            allowed,
                            temperature=(
                                float(temperature) if do_sample else 1.0
                            ),
                        )
                    )
                behavior_log_probs = torch.stack(rows, dim=0)
            if do_sample:
                next_tokens = torch.multinomial(
                    behavior_log_probs.exp(),
                    num_samples=1,
                ).squeeze(1)
            else:
                next_tokens = torch.argmax(behavior_log_probs, dim=-1)
            if return_log_probs:
                values = behavior_log_probs.gather(
                    1,
                    next_tokens.unsqueeze(1),
                ).squeeze(1)
                for row, index in enumerate(active_indexes):
                    selected_log_probs[index].append(float(values[row].item()))

            token_values = [int(value) for value in next_tokens.tolist()]
            logits_after = self._forward_persistent_tokens(
                token_values,
                active_context,
            )
            if distributed_lockstep:
                distributed_decode_traversals += 1
            finishing_rows = []
            max_length_rows = []
            for row, (index, token_id) in enumerate(
                zip(active_indexes, token_values)
            ):
                if token_id == eos_token_id:
                    stop_reasons[index] = "eos"
                    eos_model_selected[index] = True
                    finishing_rows.append(row)
                    continue
                content_token_ids[index].append(token_id)
                allowed = row_allowed[row]
                content_token_allowed_ids[index].append(
                    list(allowed) if allowed is not None else None
                )
                content_token_policy_active[index].append(
                    True if allowed is None else len(allowed) > 1
                )
                if allowed is not None:
                    constrained_token_ids[index].append(token_id)
                if len(content_token_ids[index]) >= max_length:
                    stop_reasons[index] = "max_length"
                    eos_host_forced[index] = True
                    finishing_rows.append(row)
                    max_length_rows.append(row)

            if not finishing_rows:
                logits = logits_after
                if distributed_lockstep:
                    global_continues, global_forced_eos = (
                        distributed_lockstep_sync(True, 0)
                    )
                    for _ in range(int(global_forced_eos)):
                        distributed_lockstep_noop()
                        distributed_forced_eos_padding_traversals += 1
                    if not global_continues:
                        raise RuntimeError(
                            "distributed lockstep stopped while local rows remained"
                        )
                continue

            split = self._split_generation_context(active_context)
            finishing = set(finishing_rows)
            max_length_set = set(max_length_rows)
            next_indexes = []
            next_contexts = []
            next_logits = []
            pending_forced_eos = []
            for row, index in enumerate(active_indexes):
                context = split[row]
                if row not in finishing:
                    next_indexes.append(index)
                    next_contexts.append(context)
                    next_logits.append(logits_after[row])
                    continue
                completed_contexts[index] = context
                if row in max_length_set:
                    if distributed_lockstep:
                        pending_forced_eos.append((index, context))
                    else:
                        final_logits[index] = self._forward_persistent_token(
                            eos_token_id,
                            context,
                        )
                else:
                    final_logits[index] = logits_after[row : row + 1]

            if distributed_lockstep:
                # Every rank completed exactly one sampled/dummy traversal
                # before this reduction. Determine both global continuation
                # and the symmetric number of host-EOS cache traversals.
                global_continues, global_forced_eos = (
                    distributed_lockstep_sync(
                        bool(next_indexes), len(pending_forced_eos)
                    )
                )
                if int(global_forced_eos) < len(pending_forced_eos):
                    raise RuntimeError(
                        "distributed lockstep lost forced-EOS coverage"
                    )
                for slot in range(int(global_forced_eos)):
                    if slot < len(pending_forced_eos):
                        index, context = pending_forced_eos[slot]
                        final_logits[index] = self._forward_persistent_token(
                            eos_token_id,
                            context,
                        )
                    else:
                        distributed_lockstep_noop()
                        distributed_forced_eos_padding_traversals += 1
                active_indexes = next_indexes
                if not global_continues:
                    if active_indexes:
                        raise RuntimeError(
                            "distributed lockstep stopped with local rows active"
                        )
                    break
                if not active_indexes:
                    continue
            else:
                active_indexes = next_indexes
                if not active_indexes:
                    break

            active_context = (
                next_contexts[0]
                if len(next_contexts) == 1
                else self._merge_generation_contexts(next_contexts)
            )
            logits = torch.stack(next_logits, dim=0)

        for index, context in enumerate(completed_contexts):
            if context is None:
                raise AssertionError("persistent batch lost a context")
            contexts[index].clear()
            contexts[index].update(context)

        results = []
        for index, context in enumerate(contexts):
            if constrained_schedule is not None and not (
                constrained_schedule.is_complete(constrained_token_ids[index])
            ):
                raise AssertionError(
                    "constrained schedule did not complete before EOS"
                )
            after = self.context_cache_signature(context)
            expected_delta = (
                1
                + len(forced_prefix_token_ids)
                + len(content_token_ids[index])
                + 1
            )
            actual_delta = (
                int(after["kv_lens"][0]) - int(before[index]["kv_lens"][0])
            )
            rope_delta = (
                int(after["ropes"][0]) - int(before[index]["ropes"][0])
            )
            if actual_delta != expected_delta or rope_delta != expected_delta:
                raise AssertionError("persistent batch cache delta mismatch")
            full_segment_token_ids = [
                bos_token_id,
                *forced_prefix_token_ids,
                *content_token_ids[index],
                eos_token_id,
            ]
            decoded_text, unsupported_tokens = decode_content_segment(
                self, content_token_ids[index]
            )
            result = {
                "context": context,
                "text": decoded_text,
                "text_action_support_masked": support_mask is not None,
                "unsupported_text_action_tokens": unsupported_tokens,
                "forced_prefix_text": forced_prefix_text,
                "forced_prefix_token_ids": list(forced_prefix_token_ids),
                "content_token_ids": list(content_token_ids[index]),
                "content_token_allowed_ids": list(
                    content_token_allowed_ids[index]
                ),
                "content_token_policy_active": list(
                    content_token_policy_active[index]
                ),
                "constrained_schedule_version": (
                    None
                    if constrained_schedule is None
                    else str(getattr(constrained_schedule, "version", ""))
                ),
                "constrained_token_ids": list(constrained_token_ids[index]),
                "full_segment_token_ids": full_segment_token_ids,
                "stop_reason": stop_reasons[index],
                "eos_model_selected": eos_model_selected[index],
                "eos_host_forced": eos_host_forced[index],
                "cache_delta": actual_delta,
                "kv_lens": list(after["kv_lens"]),
                "ropes": list(after["ropes"]),
                "cache_signature": after,
                "behavior_do_sample": bool(do_sample),
                "behavior_temperature": float(temperature),
                "selected_log_prob_definition": (
                    BEHAVIOR_CATEGORICAL_LOG_PROB_V1
                    if do_sample
                    else GREEDY_MODEL_LOG_PROB_V1
                ),
                "distributed_lockstep_version": (
                    "clean29529_g016_distributed_controller_lockstep_v1"
                    if distributed_lockstep
                    else None
                ),
                "distributed_decode_traversals": (
                    int(distributed_decode_traversals)
                    if distributed_lockstep
                    else None
                ),
                "distributed_dummy_traversals": (
                    int(distributed_dummy_traversals)
                    if distributed_lockstep
                    else None
                ),
                "distributed_forced_eos_padding_traversals": (
                    int(distributed_forced_eos_padding_traversals)
                    if distributed_lockstep
                    else None
                ),
            }
            if return_log_probs:
                result["selected_log_probs"] = selected_log_probs[index]
                result["sum_log_prob"] = float(
                    sum(selected_log_probs[index])
                )
            if return_next_token_logits:
                result["next_token_logits"] = (
                    final_logits[index].detach().float().cpu()
                )
            results.append(result)
        return results
        
    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        self.context_cache_signature(gen_context)
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference, 

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        self.context_cache_signature(gen_context)
        
        return gen_context

    @torch.no_grad()
    def update_context_image(
        self,
        image,
        gen_context,
        vae=True,
        vit=True,
        return_last_hidden=False,
    ):
        # used for interleave data, currently only support 1 data inference, 

        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']
        last_hidden = None

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            vae_result = self.model.forward_cache_update_vae(
                self.vae_model,
                past_key_values,
                return_last_hidden=return_last_hidden,
                **generation_input,
            )
            if return_last_hidden:
                past_key_values, last_hidden = vae_result
            else:
                past_key_values = vae_result
        
        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            vit_result = self.model.forward_cache_update_vit(
                past_key_values,
                return_last_hidden=return_last_hidden,
                **generation_input,
            )
            if return_last_hidden:
                past_key_values, last_hidden = vit_result
            else:
                past_key_values = vit_result

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        self.context_cache_signature(gen_context)
        
        if return_last_hidden:
            return gen_context, last_hidden
        return gen_context

    @torch.no_grad()
    def gen_image(
        self, 
        image_shape, 
        gen_context, 
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,

        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        
        num_timesteps=50, 
        timestep_shift=3.0,
        enable_taylorseer=False,
    ):
        # print(cfg_renorm_type)
        context_signatures = {
            "main": self.context_cache_signature(gen_context),
            "text_cfg": self.context_cache_signature(cfg_text_precontext),
            "image_cfg": self.context_cache_signature(cfg_img_precontext),
        }
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
        ) 
        
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            enable_taylorseer=enable_taylorseer,
        )

        image = self.decode_image(unpacked_latent[0], image_shape)
        after_signatures = {
            "main": self.context_cache_signature(gen_context),
            "text_cfg": self.context_cache_signature(cfg_text_precontext),
            "image_cfg": self.context_cache_signature(cfg_img_precontext),
        }
        if after_signatures != context_signatures:
            raise AssertionError("image diffusion mutated persistent KV context")
        return image

        
    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0, return_log_probs: bool = False):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        result = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            return_log_probs=return_log_probs,
            **generation_input,
        )
        if return_log_probs and isinstance(result, tuple):
            unpacked_latent, log_probs = result
            token_ids = unpacked_latent[:, 0].tolist()
            tokens = [token for token in self.tokenizer.convert_ids_to_tokens(token_ids) if token is not None]
            output = self.tokenizer.convert_tokens_to_string(tokens)
            output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
            sum_log_prob = float(log_probs[:, 0].sum()) if log_probs.numel() > 0 else 0.0
            return output, sum_log_prob
        else:
            unpacked_latent = result if not isinstance(result, tuple) else result[0]
            token_ids = unpacked_latent[:, 0].tolist()
            tokens = [token for token in self.tokenizer.convert_ids_to_tokens(token_ids) if token is not None]
            output = self.tokenizer.convert_tokens_to_string(tokens)
            output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
            return output
        
    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,

        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        enable_taylorseer=False,
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT 
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_list.append(gen_text)

            else:
                if think:
                    gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)

                img = self.gen_image(
                    image_shapes, 
                    gen_context, 
                    cfg_text_precontext=cfg_text_context, 
                    cfg_img_precontext=cfg_img_context,

                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    enable_taylorseer=enable_taylorseer,
                )

                output_list.append(img)

        return output_list
    
    def __call__(
        self, 
        image: Optional[Image.Image] = None, 
        text: Optional[str] = None, 
        **kargs
    ) -> Dict[str, Any]:
        output_dict = {'image': None, 'text': None}

        if image is None and text is None:
            print('Please provide at least one input: either an image or text.')
            return output_dict

        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        output_list = self.interleave_inference(input_list, **kargs)

        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict['image'] = i
            elif isinstance(i, str):
                output_dict['text'] = i
        return output_dict
