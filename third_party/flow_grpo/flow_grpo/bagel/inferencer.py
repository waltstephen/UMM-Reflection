# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any, Sequence

from PIL import Image
import torch
from flow_grpo.g021_dense_flow import TRANSITION_INDICES as G021_TRANSITION_INDICES

from flow_grpo.bagel.data.data_utils import pil_img2rgb
from flow_grpo.bagel.modeling.bagel.qwen2_navit import NaiveCache
from flow_grpo.bagel.modeling.bagel.bagel import select_sde_timestep_begin
from flow_grpo.bagel.modeling.bagel.bagel import LEGACY_ETA_CLAMP_MODE


def _eta_clamp_mode(grpo_config, override=None) -> str:
    """G022 item 4 / doc Appendix I: which eta denominator clamp to use.

    G016-G021 keep the exact-t==1 substitution; G022 uses R3's t >= 0.95 floor.
    Read off the config so no call site needs to know which run it is in.
    """

    if override is not None:
        return str(override)
    return str(
        getattr(
            getattr(grpo_config, "g016", None),
            "eta_clamp_mode",
            LEGACY_ETA_CLAMP_MODE,
        )
    )



VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''


class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        
    def init_gen_context(self, batch_size: int = 1):
        if batch_size < 1:
            raise ValueError("generation context batch size must be positive")
        gen_context = {
            'kv_lens': [0] * int(batch_size),
            'ropes': [0] * int(batch_size),
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text_batch(self, texts, gen_context):
        texts = [str(text) for text in texts]
        if len(texts) != len(gen_context['kv_lens']):
            raise ValueError("text/context batch cardinality mismatch")
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=gen_context['kv_lens'],
            curr_rope=gen_context['ropes'],
            prompts=texts,
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        gen_context['past_key_values'] = self.model.forward_cache_update_text(
            gen_context['past_key_values'],
            **generation_input,
        )
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        return gen_context

    @torch.no_grad()
    def update_context_image_batch(
        self,
        images,
        gen_context,
        vae=True,
        vit=True,
    ):
        if not vae and not vit:
            raise ValueError("batched image context requires VAE or ViT")
        images = list(images)
        if len(images) != len(gen_context['kv_lens']):
            raise ValueError("image/context batch cardinality mismatch")
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        if vae:
            generation_input, kv_lens, ropes = (
                self.model.prepare_vae_images(
                    curr_kvlens=kv_lens,
                    curr_rope=ropes,
                    images=images,
                    transforms=self.vae_transform,
                    new_token_ids=self.new_token_ids,
                )
            )
            past_key_values = self.model.forward_cache_update_vae(
                self.vae_model,
                past_key_values,
                **generation_input,
            )
        if vit:
            generation_input, kv_lens, ropes = (
                self.model.prepare_vit_images(
                    curr_kvlens=kv_lens,
                    curr_rope=ropes,
                    images=images,
                    transforms=self.vit_transform,
                    new_token_ids=self.new_token_ids,
                )
            )
            past_key_values = self.model.forward_cache_update_vit(
                past_key_values,
                **generation_input,
            )
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
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
        
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        # used for interleave data, currently only support 1 data inference, 

        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        
        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

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

        # for grpo learn
        learn=False,
        sample=None,
        grpo_config=None,
        accelerator=None,
        optimizer=None,
        transformer=None,
        noise_level=0.7,
        generators=None,
        reference_contexts=None,
        gradient_clip_parameters=None,
        max_grad_norm=1.0,
        isolate_regular_gradients=False,
        perform_optimizer_step=True,
        backward_loss_scale=1.0,
        sde_window_identities=None,
        eta_clamp_mode=None,
    ):
        # Do not set the initial latent to be the same for the same prompt in eval mode
        if noise_level==0:
            generators=None
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
            generators=generators
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
        if learn:
            if bool(
                getattr(
                    grpo_config.train,
                    "offload_frozen_during_backward",
                    True,
                )
            ):
                # Multiround full-rank campaigns trade recompute latency for
                # memory. The upstream direct-T2I path keeps these resident.
                self.model.connector.to(device="cpu", dtype=torch.bfloat16)
                if self.model.vit_model is not None:
                    self.model.vit_model.to(
                        device="cpu",
                        dtype=torch.bfloat16,
                    )
                self.vae_model.to(device="cpu", dtype=torch.bfloat16)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            (
                clipfrac,
                clipfrac_gt_one,
                clipfrac_lt_one,
                policy_loss,
                kl_loss,
                loss,
                flow_diagnostics,
            ) = self.model.generate_image_learn(
                sample=sample,
                grpo_config=grpo_config,
                accelerator=accelerator,
                optimizer=optimizer,
                transformer=transformer,
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
                noise_level=noise_level,
                eta_clamp_mode=_eta_clamp_mode(grpo_config, eta_clamp_mode),
                ref_past_key_values=(
                    reference_contexts.get("past_key_values")
                    if reference_contexts is not None
                    else None
                ),
                ref_cfg_text_past_key_values=(
                    reference_contexts.get("cfg_text_past_key_values")
                    if reference_contexts is not None
                    else None
                ),
                ref_cfg_img_past_key_values=(
                    reference_contexts.get("cfg_img_past_key_values")
                    if reference_contexts is not None
                    else None
                ),
                gradient_clip_parameters=gradient_clip_parameters,
                max_grad_norm=max_grad_norm,
                isolate_regular_gradients=bool(
                    isolate_regular_gradients
                ),
                perform_optimizer_step=bool(perform_optimizer_step),
                backward_loss_scale=float(backward_loss_scale),
            )
            return {
                "clipfrac": clipfrac, 
                "clipfrac_gt_one": clipfrac_gt_one,
                "clipfrac_lt_one": clipfrac_lt_one,
                "policy_loss": policy_loss, 
                "kl_loss": kl_loss,
                "loss": loss,
                **flow_diagnostics,
            }
        else:
            unpacked_latent, all_latents, all_log_probs, timesteps = self.model.generate_image(
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
                noise_level=noise_level,
                sample_sde_window_size=grpo_config.sample.sde_window_size,
                sample_sde_window_range=grpo_config.sample.sde_window_range,
                process_index=getattr(accelerator, 'process_index', 0),
                device=getattr(accelerator, 'device', 'cuda'),
                sde_window_identities=sde_window_identities,
                sample_sde_stratified=bool(getattr(grpo_config.sample, "g021_stratified_sde", False)),
                sample_sde_contract=getattr(grpo_config.sample, "g021_flow_contract", None),
                eta_clamp_mode=_eta_clamp_mode(grpo_config, eta_clamp_mode),
            )
            if sde_window_identities is not None:
                if not (
                    len(all_latents) == len(all_log_probs) == len(timesteps) == 1
                ):
                    raise RuntimeError("G012 single SDE history shape differs")
                all_latents = all_latents[0]
                all_log_probs = all_log_probs[0]
                timesteps = timesteps[0]
                sde_timestep_begin = (
                    G021_TRANSITION_INDICES[0]
                    if bool(getattr(grpo_config.sample, "g021_stratified_sde", False))
                    else select_sde_timestep_begin(
                        grpo_config.sample.sde_window_range,
                        grpo_config.sample.sde_window_size,
                        process_index=getattr(accelerator, 'process_index', 0),
                        seed_identity=list(sde_window_identities)[0],
                    )
                )
            else:
                sde_timestep_begin = None
            all_latents = [
                value.detach().to(device="cpu", non_blocking=False)
                for value in all_latents
            ]
            all_log_probs = [
                value.detach().to(device="cpu", non_blocking=False)
                for value in all_log_probs
            ]
            timesteps = timesteps.detach().to(
                device="cpu",
                non_blocking=False,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            image = self.decode_image(unpacked_latent[0].float(), image_shape)
            return {
                "image": image,
                "all_latents": all_latents,
                "all_log_probs": all_log_probs,
                "timesteps": timesteps,
                "sde_timestep_begin": sde_timestep_begin,
                "sde_window_seed_identity": (
                    None
                    if sde_window_identities is None
                    else dict(list(sde_window_identities)[0])
                ),
                "sde_transition_layout": (
                    "g021_stratified_pairs_v1"
                    if bool(getattr(grpo_config.sample, "g021_stratified_sde", False))
                    else "contiguous_v1"
                ),
                "sde_transition_indices": (
                    list(G021_TRANSITION_INDICES)
                    if bool(getattr(grpo_config.sample, "g021_stratified_sde", False))
                    else list(range(sde_timestep_begin, sde_timestep_begin + len(all_log_probs)))
                    if sde_timestep_begin is not None else []
                ),
            }

    def gen_image_batch(
        self,
        image_shapes: Sequence[tuple[int, int]],
        gen_context,
        *,
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_text_precontext=None,
        cfg_img_precontext=None,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        num_timesteps=50,
        timestep_shift=3.0,
        grpo_config=None,
        accelerator=None,
        noise_level=0.7,
        generators=None,
        sde_window_identities=None,
        eta_clamp_mode=None,
    ):
        image_shapes = [tuple(shape) for shape in image_shapes]
        batch_size = len(image_shapes)
        if batch_size < 2:
            raise ValueError("batched image generation requires two samples")
        if len(gen_context['kv_lens']) != batch_size:
            raise ValueError("image/context batch cardinality mismatch")
        if generators is not None and len(generators) != batch_size:
            raise ValueError("image/generator batch cardinality mismatch")
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=gen_context['kv_lens'],
            curr_rope=gen_context['ropes'],
            image_sizes=image_shapes,
            new_token_ids=self.new_token_ids,
            generators=generators,
        )
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_text_precontext['kv_lens'],
            curr_rope=cfg_text_precontext['ropes'],
            image_sizes=image_shapes,
        )
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img_precontext['kv_lens'],
            curr_rope=cfg_img_precontext['ropes'],
            image_sizes=image_shapes,
        )
        (
            unpacked_latent,
            all_latents,
            all_log_probs,
            timesteps,
        ) = self.model.generate_image(
            past_key_values=gen_context['past_key_values'],
            cfg_text_past_key_values=(
                cfg_text_precontext['past_key_values']
            ),
            cfg_img_past_key_values=cfg_img_precontext['past_key_values'],
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=(
                generation_input_cfg_text['cfg_packed_position_ids']
            ),
            cfg_text_packed_query_indexes=(
                generation_input_cfg_text['cfg_packed_query_indexes']
            ),
            cfg_text_key_values_lens=(
                generation_input_cfg_text['cfg_key_values_lens']
            ),
            cfg_text_packed_key_value_indexes=(
                generation_input_cfg_text[
                    'cfg_packed_key_value_indexes'
                ]
            ),
            cfg_img_packed_position_ids=(
                generation_input_cfg_img['cfg_packed_position_ids']
            ),
            cfg_img_packed_query_indexes=(
                generation_input_cfg_img['cfg_packed_query_indexes']
            ),
            cfg_img_key_values_lens=(
                generation_input_cfg_img['cfg_key_values_lens']
            ),
            cfg_img_packed_key_value_indexes=(
                generation_input_cfg_img[
                    'cfg_packed_key_value_indexes'
                ]
            ),
            noise_level=noise_level,
            sample_sde_window_size=grpo_config.sample.sde_window_size,
            sample_sde_window_range=grpo_config.sample.sde_window_range,
            process_index=getattr(accelerator, 'process_index', 0),
            device=getattr(accelerator, 'device', 'cuda'),
            sde_generators=generators,
            sde_window_identities=sde_window_identities,
            sample_sde_stratified=bool(getattr(grpo_config.sample, "g021_stratified_sde", False)),
            sample_sde_contract=getattr(grpo_config.sample, "g021_flow_contract", None),
            eta_clamp_mode=_eta_clamp_mode(grpo_config, eta_clamp_mode),
        )
        sample_lens = [
            int(length) - 2
            for length in generation_input['packed_seqlens'].tolist()
        ]
        if len(unpacked_latent) != batch_size:
            raise RuntimeError("batched denoise lost final latent samples")
        if sde_window_identities is not None:
            if not (
                len(all_latents)
                == len(all_log_probs)
                == len(timesteps)
                == batch_size
            ):
                raise RuntimeError("G012 batched per-sample SDE history differs")
            per_sample_latents = [
                [value.detach().to(device="cpu", non_blocking=False) for value in rows]
                for rows in all_latents
            ]
            per_sample_log_probs = [
                [value.detach().to(device="cpu", non_blocking=False) for value in rows]
                for rows in all_log_probs
            ]
            per_sample_timesteps = [
                value.detach().to(device="cpu", non_blocking=False)
                for value in timesteps
            ]
            sde_timestep_begins = (
                [G021_TRANSITION_INDICES[0]] * batch_size
                if bool(getattr(grpo_config.sample, "g021_stratified_sde", False))
                else [
                    select_sde_timestep_begin(
                        grpo_config.sample.sde_window_range,
                        grpo_config.sample.sde_window_size,
                        process_index=getattr(accelerator, 'process_index', 0),
                        seed_identity=identity,
                    )
                    for identity in sde_window_identities
                ]
            )
        else:
            per_sample_latents = [[] for _ in range(batch_size)]
            for packed_latent in all_latents:
                chunks = packed_latent.split(sample_lens, dim=0)
                if len(chunks) != batch_size:
                    raise RuntimeError("batched denoise lost latent history")
                for index, value in enumerate(chunks):
                    per_sample_latents[index].append(
                        value.detach().to(device="cpu", non_blocking=False)
                    )
            per_sample_log_probs = [[] for _ in range(batch_size)]
            for log_probs in all_log_probs:
                if log_probs.ndim != 1 or log_probs.shape[0] != batch_size:
                    raise RuntimeError(
                        "batched denoise returned non-candidate log-probs"
                    )
                for index, value in enumerate(log_probs):
                    per_sample_log_probs[index].append(
                        value.detach().to(device="cpu", non_blocking=False)
                    )
            shared_timesteps = timesteps.detach().to(
                device="cpu",
                non_blocking=False,
            )
            per_sample_timesteps = [shared_timesteps.clone() for _ in range(batch_size)]
            sde_timestep_begins = [None] * batch_size
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        images = [
            self.decode_image(latent.float(), image_shape)
            for latent, image_shape in zip(unpacked_latent, image_shapes)
        ]
        return [
            {
                "image": image,
                "all_latents": per_sample_latents[index],
                "all_log_probs": per_sample_log_probs[index],
                "timesteps": per_sample_timesteps[index],
                "sde_timestep_begin": sde_timestep_begins[index],
                "sde_window_seed_identity": (
                    None
                    if sde_window_identities is None
                    else dict(sde_window_identities[index])
                ),
                "sde_transition_layout": (
                    "g021_stratified_pairs_v1"
                    if bool(getattr(grpo_config.sample, "g021_stratified_sde", False))
                    else "contiguous_v1"
                ),
                "sde_transition_indices": (
                    list(G021_TRANSITION_INDICES)
                    if bool(getattr(grpo_config.sample, "g021_stratified_sde", False))
                    else list(range(sde_timestep_begins[index], sde_timestep_begins[index] + len(per_sample_log_probs[index])))
                    if sde_timestep_begins[index] is not None else []
                ),
            }
            for index, image in enumerate(images)
        ]

    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample
        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].float()
        return image

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output
        
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
        learn=False,
        sample=None,
        grpo_config=None,
        accelerator=None,
        optimizer=None,
        transformer=None,
        noise_level=0.7,
        generators=None,
        reference_contexts=None,
        gradient_clip_parameters=None,
        max_grad_norm=1.0,
        isolate_regular_gradients=False,
        perform_optimizer_step=True,
        backward_loss_scale=1.0,
        sde_window_identities=None,
        eta_clamp_mode=None,
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

                    # for grpo learn
                    learn=learn,
                    sample=sample,
                    grpo_config=grpo_config,
                    accelerator=accelerator,
                    optimizer=optimizer,
                    transformer=transformer,
                    noise_level=noise_level,
                    generators=generators,
                    reference_contexts=reference_contexts,
                    gradient_clip_parameters=gradient_clip_parameters,
                    max_grad_norm=max_grad_norm,
                    isolate_regular_gradients=bool(
                        isolate_regular_gradients
                    ),
                    perform_optimizer_step=bool(perform_optimizer_step),
                    backward_loss_scale=float(backward_loss_scale),
                    sde_window_identities=sde_window_identities,
                    eta_clamp_mode=eta_clamp_mode,
                )

                output_list.append(img)

        return output_list

    def interleave_inference_batch(
        self,
        input_lists: Sequence[Sequence[Union[str, Image.Image]]],
        *,
        think=False,
        understanding_output=False,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=(0.4, 1.0),
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        grpo_config=None,
        accelerator=None,
        noise_level=0.7,
        generators=None,
        sde_window_identities=None,
    ):
        if think or understanding_output:
            raise NotImplementedError(
                "V19 batched rollout supports image generation only"
            )
        rows = [list(row) for row in input_lists]
        batch_size = len(rows)
        if batch_size < 2:
            raise ValueError("batched interleave requires two samples")
        term_counts = {len(row) for row in rows}
        if len(term_counts) != 1:
            raise ValueError("batched interleave term counts differ")
        gen_context = self.init_gen_context(batch_size)
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)
        per_sample_image_shapes = [tuple(image_shapes)] * batch_size

        with torch.autocast(
            device_type="cuda",
            enabled=True,
            dtype=torch.bfloat16,
        ):
            for term_index in range(next(iter(term_counts))):
                terms = [row[term_index] for row in rows]
                if all(isinstance(term, str) for term in terms):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text_batch(
                        terms,
                        gen_context,
                    )
                    cfg_img_context = self.update_context_text_batch(
                        terms,
                        cfg_img_context,
                    )
                elif all(isinstance(term, Image.Image) for term in terms):
                    resized_images = [
                        self.vae_transform.resize_transform(
                            pil_img2rgb(term)
                        )
                        for term in terms
                    ]
                    gen_context = self.update_context_image_batch(
                        resized_images,
                        gen_context,
                        vae=True,
                        vit=True,
                    )
                    per_sample_image_shapes = [
                        image.size[::-1] for image in resized_images
                    ]
                    cfg_text_context = deepcopy(gen_context)
                else:
                    raise ValueError(
                        "batched interleave term types are not aligned"
                    )

            return self.gen_image_batch(
                per_sample_image_shapes,
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
                grpo_config=grpo_config,
                accelerator=accelerator,
                noise_level=noise_level,
                generators=generators,
                sde_window_identities=sde_window_identities,
            )
    
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
        return output_list[0]
