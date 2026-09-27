# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import List, Tuple, Optional
import hashlib
import math
import random
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from diffusers.utils.torch_utils import randn_tensor
from flow_grpo.g021_dense_flow import TRANSITION_INDICES as G021_TRANSITION_INDICES
from flow_grpo.bagel.data.data_utils import (
    create_sparse_mask, 
    get_flattened_position_ids_extrapolate, 
    get_flattened_position_ids_interpolate,
    patchify, 
)
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as fsdp
from contextlib import nullcontext
from torch.profiler import profile, ProfilerActivity, record_function
from .qwen2_navit import NaiveCache
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding

from tqdm import tqdm
from functools import partial

tqdm = partial(tqdm, dynamic_ncols=True, leave=False, position=1)
FLOW_KL_VERSION = "clean29529_v19_flow_transition_kl_v2"
G012_SDE_WINDOW_SEED_VERSION = "clean29529_g012_local_sde_window_seed_v1"


G022_FLOW_CONTRACT = "g022_contiguous_uniform_random_flow2_v1"
G022_TRAIN_NUM_TIMESTEPS = 20
G022_EVAL_NUM_TIMESTEPS = 50
G022_SDE_WINDOW_SIZE = 2
G022_SDE_WINDOW_RANGE = (0, G022_TRAIN_NUM_TIMESTEPS // 2)


# ---------------------------------------------------------------------------
# G022 item 4: eta clamp. Doc Appendix I.
#
# R3's `sde_sampler.get_eta` floors the denominator at t >= 0.95:
#
#     torch.sqrt(t / (1 - torch.where(t >= 0.95, 0.95, t))) * constant_eta
#
# Ours substitutes only at *exact float equality* t == 1, replacing the
# denominator's t with `sigma_max = timesteps[1]`. The formula is otherwise
# identical -- we are already running R3's "monotonic" eta mode, and our
# `noise_level` is exactly their `constant_eta`.
#
# The difference bites precisely where the G022 window sits. At num_timesteps=20
# with timestep_shift=3.0 the schedule starts t = 1.0, 0.98182, 0.96226, ... and
# the window range (0, 10) makes start = randint(0, 8), so i = 0 and i = 1 are
# reachable. Appendix I-2, measured:
#
#     i | t       | ours std/eta | R3 std/eta | ratio
#     0 | 1.00000 |     7.42     |    4.47    | 1.66x
#     1 | 0.98182 |     7.35     |    4.43    | 1.66x
#     2 | 0.96226 |     5.05     |    4.39    | 1.15x
#     3 | 0.94118 |     4.00     |    4.00    | 1.00x
#
# On those draws we inject 66% more noise than R3's clamp permits, at the
# highest-variance end of the schedule -- exactly the shape that produces the
# outlier log-probs that fed the G021-B KL blowup. Appendix I-3 adopts R3's
# clamp, and treats eta_edit as the diversity knob instead of relying on an
# uncontrolled singularity at the start of the schedule.
LEGACY_ETA_CLAMP_MODE = "exact_t_equals_one_sigma_max"
G022_ETA_CLAMP_MODE = "r3_t_ge_0p95"
G022_ETA_CLAMP_T = 0.95


def g022_sde_denominator(timestep, *, sigma_max=None, eta_clamp_mode=LEGACY_ETA_CLAMP_MODE):
    """`1 - t`, floored the way the selected clamp mode requires."""

    mode = str(eta_clamp_mode)
    if mode == G022_ETA_CLAMP_MODE:
        floor = torch.full_like(timestep, G022_ETA_CLAMP_T)
        return 1 - torch.where(timestep >= G022_ETA_CLAMP_T, floor, timestep)
    if mode != LEGACY_ETA_CLAMP_MODE:
        raise ValueError(f"unknown SDE eta clamp mode: {eta_clamp_mode!r}")
    if sigma_max is None:
        raise ValueError("the legacy eta clamp requires sigma_max")
    return 1 - torch.where(timestep == 1, sigma_max, timestep)


def validate_g021_flow_contract(
    *,
    contract: Optional[str],
    sample_sde_stratified: bool,
    sample_sde_window_size: int,
    sample_sde_window_range: Tuple[int, int],
    num_timesteps: int,
) -> str | None:
    """Production guard separating G021-A flow4 from G021-B G018 flow2."""
    if contract is None:
        return None
    if contract == "g021a_stratified_flow4_v2":
        if not sample_sde_stratified or sample_sde_window_size != 4 or tuple(sample_sde_window_range) != (0, 49) or num_timesteps != 50:
            raise ValueError("G021-A requires stratified flow4 over 50 generation steps")
        return "stratified_4"
    if contract == "g021b_contiguous_seeded_flow2_v1":
        if sample_sde_stratified or sample_sde_window_size != 2 or tuple(sample_sde_window_range) != (0, 25) or num_timesteps != 50:
            raise ValueError("G021-B requires original G018 contiguous seeded flow2")
        return "contiguous_seeded_2"
    if contract == G022_FLOW_CONTRACT:
        # G022, doc section 7 + Appendix G-2. Official law with the chosen
        # 20 training denoising steps: contiguous window of size 2, range
        # (0, num_steps // 2) = (0, 10), start = randint(0, 8), so the window
        # always lies in the first half of the schedule and has 9 distinct
        # placements.
        #
        # A5 (found while working A3/C1, not a Q-series item). This branch used
        # to require `num_timesteps == G022_TRAIN_NUM_TIMESTEPS` with the
        # comment "Eval stays at 50 steps and never uses this contract." That
        # comment was false: `config/g016.py` sets `g021_flow_contract`
        # unconditionally and `inferencer.py` passes it into every
        # `interleave_inference`, evaluation included. Before T1.5 evaluation
        # ran at 20 and the guard happened to pass; T1.5 correctly moved
        # evaluation to 50 and thereby made this raise on the first evaluation
        # image call. The contract governs the SDE window, which is unchanged
        # between the two modes; the step count is the mode. Accept both, and
        # return which one so a caller can record it.
        if (
            sample_sde_stratified
            or sample_sde_window_size != G022_SDE_WINDOW_SIZE
            or tuple(sample_sde_window_range) != G022_SDE_WINDOW_RANGE
            or num_timesteps
            not in (G022_TRAIN_NUM_TIMESTEPS, G022_EVAL_NUM_TIMESTEPS)
        ):
            raise ValueError(
                "G022 requires contiguous uniform-random flow2 over "
                f"{G022_TRAIN_NUM_TIMESTEPS} training or "
                f"{G022_EVAL_NUM_TIMESTEPS} evaluation denoising steps with "
                f"range {G022_SDE_WINDOW_RANGE}"
            )
        return "contiguous_uniform_random_2"
    raise ValueError(f"unknown G021 flow contract: {contract}")


def select_sde_timestep_begin(
    window_range: Tuple[int, int],
    window_size: int,
    *,
    process_index: int,
    seed_identity=None,
) -> int:
    """Select one SDE start while preserving the retired legacy default.

    G012 passes an explicit run/call identity and uses an isolated RNG. Older
    callers pass ``None`` and retain the original per-rank global reseed.
    """

    low, high = map(int, window_range)
    size = int(window_size)
    if low < 0 or size <= 0 or high - size < low:
        raise ValueError("invalid SDE window range/size")
    if seed_identity is None:
        random.seed(int(process_index))
        return random.randint(low, high - size)
    required = (
        "experiment_seed",
        "logical_step",
        "trajectory_index",
        "round_index",
    )
    if not isinstance(seed_identity, dict) or any(
        key not in seed_identity for key in required
    ):
        raise ValueError("G012 SDE seed identity is incomplete")
    canonical = "|".join(str(int(seed_identity[key])) for key in required)
    seed = int.from_bytes(
        hashlib.sha256(
            (G012_SDE_WINDOW_SEED_VERSION + "|" + canonical).encode("ascii")
        ).digest()[:8],
        "big",
    )
    return random.Random(seed).randint(low, high - size)


def _packed_global_cfg_renorm(
    base: torch.Tensor,
    guided: torch.Tensor,
    sample_lens: List[int],
    *,
    minimum: float,
) -> torch.Tensor:
    if sum(int(length) for length in sample_lens) != base.shape[0]:
        raise ValueError("CFG sample lengths do not cover packed state")
    if base.shape != guided.shape:
        raise ValueError("CFG base/guided shapes differ")
    if len(sample_lens) == 1:
        scale = (
            torch.norm(base) / (torch.norm(guided) + 1e-8)
        ).clamp(min=minimum, max=1.0)
        return guided * scale
    base_chunks = base.split(sample_lens, dim=0)
    guided_chunks = guided.split(sample_lens, dim=0)
    return torch.cat(
        [
            guided_chunk
            * (
                torch.norm(base_chunk)
                / (torch.norm(guided_chunk) + 1e-8)
            ).clamp(min=minimum, max=1.0)
            for base_chunk, guided_chunk in zip(
                base_chunks,
                guided_chunks,
            )
        ],
        dim=0,
    )


def equal_variance_transition_kl(
    policy_mean: torch.Tensor,
    reference_mean: torch.Tensor,
    std_dev_t: torch.Tensor,
    d_timestep: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if policy_mean.shape != reference_mean.shape:
        raise ValueError("policy/reference transition means differ")
    delta_t = torch.clamp(
        -torch.as_tensor(
            d_timestep,
            device=policy_mean.device,
            dtype=torch.float32,
        ),
        min=torch.finfo(torch.float32).eps,
    )
    transition_variance = torch.clamp(
        torch.as_tensor(
            std_dev_t,
            device=policy_mean.device,
            dtype=torch.float32,
        ).square()
        * delta_t,
        min=torch.finfo(torch.float32).eps,
    )
    mean_delta = policy_mean.float() - reference_mean.float()
    mean_shift_mse = mean_delta.square().mean()
    kl = (mean_delta.square() / (2.0 * transition_variance)).mean()
    return kl, {
        "mean_shift_mse": mean_shift_mse.detach(),
        "transition_variance": transition_variance.mean().detach(),
    }


class BagelConfig(PretrainedConfig):
    def __init__(
        self,
        visual_gen=True,
        visual_und=True,
        llm_config=None,
        vit_config=None,
        vae_config=None,
        latent_patch_size=2,
        max_latent_size=32,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift


class Bagel(PreTrainedModel):
    config_class = BagelConfig
    base_model_prefix = 'bagel'

    def __init__(self, language_model, vit_model, config: BagelConfig):
        super().__init__(config)    
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads

        if config.visual_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.latent_downsample = config.vae_config.downsample * config.latent_patch_size
            self.max_latent_size = config.max_latent_size
            self.latent_channel = config.vae_config.z_channels
            self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel
            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)

        if config.visual_und:
            self.vit_model = vit_model
            self.vit_patch_size = config.vit_config.patch_size
            self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
            self.vit_hidden_size = config.vit_config.hidden_size
            self.connector = MLPconnector(self.vit_hidden_size, self.hidden_size, config.connector_act)
            self.vit_pos_embed = PositionEmbedding(self.vit_max_num_patch_per_side, self.hidden_size)

        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config
        self._init_weights()

    def _init_weights(self):
        if self.config.visual_gen:
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)

    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        # for visual generation
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
        mse_loss_indexes: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            sequence_length: length of sequence.
            packed_text_ids: 1-D int tensor, packed text token ids.
            packed_text_indexes: 1-D int tensor, packed text token indexes in sequence.
            sample_lens: A list of N ints, length of each sample in packed_sequence.
            nested_attention_masks: A list of N 2-D float tensor,  where 0.0 means attention and 
                -inf means ignore.
            packed_position_ids: packed 1-D positions, an image has only one global position shared
                by all latent tokens.

            packed_vit_tokens: packed patchified image tokens for vit model.
            packed_vit_position_ids: 1-D int tensor, the position of each token for vit model.
            packed_vit_token_indexes: 1-D int tensor, packed vit token indexes in sequence.
            vit_token_seqlens: 1-D int tensor, the length of each image tokens for vit model.
            packed_label_ids: 1-D int tensor, packed label token ids.
            ce_loss_indexes: 1-D bool tensor, where to compute ce loss.

            padded_latent: padded latent from VAE encoder.
            patchified_vae_latent_shapes: A list of (h, w) tuples, patchfied latent shapes of each image.
            packed_latent_position_ids: 1-D int tensor, the position of each token for latent.
            packed_vae_token_indexes: 1-D int tensor, padded image token indexes in sequence.
            packed_timesteps: 1-D float tensor, flow timesteps. 0 indicates use clean image.
            mse_loss_indexes: 1-D bool tensor, where to compute mse loss.
        """
        packed_text_embedding = self.language_model.forward(mode="get_embeddings", input_ids=packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if nested_attention_masks is None:
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:
            attention_mask = nested_attention_masks

        if self.config.visual_und:
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
            cu_seqlens = cu_seqlens.to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()
            packed_vit_token_embed = self.vit_model(
                packed_pixel_values=packed_vit_tokens, 
                packed_flattened_position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            packed_vit_token_embed = self.connector(packed_vit_token_embed)
            vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
            packed_sequence = packed_sequence.to(packed_vit_token_embed.dtype)
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        if self.config.visual_gen:
            p = self.latent_patch_size
            packed_latent = []
            for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                packed_latent.append(latent)
            packed_latent_clean = torch.cat(packed_latent, dim=0)

            noise = torch.randn_like(packed_latent_clean)
            packed_timesteps = torch.sigmoid(packed_timesteps)
            packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
            packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise
            packed_timestep_embeds = self.time_embedder(packed_timesteps)
            latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
            packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb
            packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes=torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,
            )

        last_hidden_state = self.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )

        mse = None
        if self.config.visual_gen:
            packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
            target = noise - packed_latent_clean # NOTE: v_t=dx_t/dt=x_1-x_0, pointing from data to noise
            has_mse = packed_timesteps > 0
            mse = (packed_mse_preds - target[has_mse]) ** 2

        ce = None
        if ce_loss_indexes is not None:
            packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")

        return dict(mse=mse, ce=ce)


    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.forward(mode="get_embeddings", input_ids=packed_text_ids)
        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vit_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2), 
                self.vit_patch_size, 
                max_num_patches_per_side=self.vit_max_num_patch_per_side
            )
            vit_tokens = patchify(image_tensor, self.vit_patch_size)
            packed_vit_tokens.append(vit_tokens)
            num_img_tokens = vit_tokens.shape[0]
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        return_last_hidden: bool = False,
    ):
        packed_text_embedding = self.language_model.forward(mode="get_embeddings", input_ids=packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        vit_parameter = next(self.vit_model.parameters())
        packed_vit_tokens = packed_vit_tokens.to(
            device=vit_parameter.device,
            dtype=vit_parameter.dtype,
        )
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()
        packed_vit_token_embed = self.vit_model(
            packed_pixel_values=packed_vit_tokens, 
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        connector_parameter = next(self.connector.parameters())
        packed_vit_token_embed = packed_vit_token_embed.to(
            device=connector_parameter.device,
            dtype=connector_parameter.dtype,
        )
        packed_vit_token_embed = self.connector(packed_vit_token_embed)
        pos_emb = self.vit_pos_embed(packed_vit_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + pos_emb
        if packed_vit_token_embed.dtype != packed_sequence.dtype:
            packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        if return_last_hidden:
            return past_key_values, output.packed_query_sequence[-1]
        return past_key_values

    def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, timestep=0):
        patchified_vae_latent_shapes, packed_vae_position_ids = list(), list()
        packed_vae_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        vae_image_tensors = list()
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vae_image_tensors.append(image_tensor)
            vae_posiiton_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2),
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)
            H, W = image_tensor.shape[1:]
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            patchified_vae_latent_shapes.append((h, w))

            num_img_tokens = w * h
            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vae(
        self,
        vae_model,
        past_key_values: NaiveCache,
        padded_images: torch.Tensor,
        patchified_vae_latent_shapes: List,
        packed_vae_position_ids: torch.LongTensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
        return_last_hidden: bool = False,
    ):
        packed_text_embedding = self.language_model.forward(mode="get_embeddings", input_ids=packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        vae_parameter = next(vae_model.parameters())
        padded_images = padded_images.to(
            device=vae_parameter.device,
            dtype=vae_parameter.dtype,
        )
        padded_latent = vae_model.encode(padded_images)

        p = self.latent_patch_size
        packed_latent = list()
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
            packed_latent.append(latent)
        packed_latent = torch.cat(packed_latent, dim=0)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_latent = packed_latent.to(
            device=self.vae2llm.weight.device,
            dtype=self.vae2llm.weight.dtype,
        )
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        if packed_latent.dtype != packed_sequence.dtype:
            packed_latent = packed_latent.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        output = self.language_model.forward(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        if return_last_hidden:
            return past_key_values, output.packed_query_sequence[-1]
        return past_key_values

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids, generators=None):
        packed_text_ids, packed_text_indexes = list(), list()
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = list(), list(), list()
        packed_position_ids, packed_seqlens, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        query_curr = curr = 0
        index=0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            vae_posiiton_ids = self.get_flattened_position_ids(
                H, W,
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            if generators:
                packed_init_noises.append(
                    torch.randn((num_image_tokens, self.latent_channel * self.latent_patch_size ** 2),generator=generators[index])
                )
            else:
                packed_init_noises.append(
                    torch.randn(num_image_tokens, self.latent_channel * self.latent_patch_size ** 2)
                )
            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1
            index+=1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))
            packed_seqlens.append(num_image_tokens + 2)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
        packed_position_ids, packed_indexes, packed_key_value_indexes = list(), list(), list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

        generation_input = {
            "cfg_packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_query_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "cfg_packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    # @torch.no_grad
    def generate_image(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        noise_level: float = 0.7,
        sample_sde_window_size: int = 1,
        sample_sde_window_range: Tuple[int, int] = (0, 5),
        process_index: int = 0,
        device = "cuda",
        sde_generators = None,
        sde_window_identities = None,
        sample_sde_stratified: bool = False,
        sample_sde_contract: Optional[str] = None,
        eta_clamp_mode: str = LEGACY_ETA_CLAMP_MODE,
    ):
        validate_g021_flow_contract(
            contract=sample_sde_contract,
            sample_sde_stratified=sample_sde_stratified,
            sample_sde_window_size=sample_sde_window_size,
            sample_sde_window_range=sample_sde_window_range,
            num_timesteps=num_timesteps,
        )
        x_t = packed_init_noises.to(device)
        sample_lens = (packed_seqlens - 2).tolist()
        corrected_sampling = sde_window_identities is not None
        if corrected_sampling:
            identities = list(sde_window_identities)
            if len(identities) != len(sample_lens):
                raise ValueError("G012 SDE identity/sample cardinality mismatch")
            if sample_sde_stratified:
                if sample_sde_window_size != len(G021_TRANSITION_INDICES) or num_timesteps != 50:
                    raise ValueError("G021 attempt2 requires stratified 4-of-49 transitions")
                per_sample_selected_indices = [G021_TRANSITION_INDICES for _ in identities]
                sde_timestep_begins = [G021_TRANSITION_INDICES[0] for _ in identities]
            else:
                per_sample_selected_indices = None
                sde_timestep_begins = [
                    select_sde_timestep_begin(
                        sample_sde_window_range,
                        sample_sde_window_size,
                        process_index=process_index,
                        seed_identity=identity,
                    )
                    for identity in identities
                ]
            per_sample_latents = [[] for _ in sample_lens]
            per_sample_log_probs = [[] for _ in sample_lens]
            per_sample_timesteps = [[] for _ in sample_lens]
        else:
            # Retired callers preserve the original exact behavior.
            sde_timestep_begin = select_sde_timestep_begin(
                sample_sde_window_range,
                sample_sde_window_size,
                process_index=process_index,
            )

        timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts = timesteps[1:] - timesteps[:-1]
        timesteps = timesteps[:-1]

        all_latents = []
        all_log_probs = []
        all_timesteps = []

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            if corrected_sampling:
                before_chunks = x_t.split(sample_lens, dim=0)
                if sample_sde_stratified:
                    for sample_index, selected in enumerate(per_sample_selected_indices):
                        if i in selected:
                            per_sample_latents[sample_index].append(before_chunks[sample_index])
                    level_chunks = [
                        torch.full(
                            (int(length), *([1] * (x_t.ndim - 1))),
                            float(noise_level if i in selected else 0.0),
                            device=x_t.device,
                            dtype=x_t.dtype,
                        )
                        for length, selected in zip(sample_lens, per_sample_selected_indices)
                    ]
                else:
                    for sample_index, begin in enumerate(sde_timestep_begins):
                        if i == begin:
                            per_sample_latents[sample_index].append(before_chunks[sample_index])
                    level_chunks = [
                        torch.full(
                            (int(length), *([1] * (x_t.ndim - 1))),
                            float(noise_level if begin <= i < begin + sample_sde_window_size else 0.0),
                            device=x_t.device,
                            dtype=x_t.dtype,
                        )
                        for length, begin in zip(sample_lens, sde_timestep_begins)
                    ]
                cur_noise_level = torch.cat(level_chunks, dim=0)
            elif i < sde_timestep_begin:
                cur_noise_level = 0
            elif i == sde_timestep_begin:
                cur_noise_level= noise_level
                all_latents.append(x_t)
            elif i > sde_timestep_begin and i < sde_timestep_begin + sample_sde_window_size:
                cur_noise_level = noise_level
            else:
                cur_noise_level= 0
            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            v_t = self._forward_flow(
                x_t=x_t,
                timestep=timestep, 
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # cfg_text
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                # cfg_img
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
            )
            batched_sample_lens = (
                sample_lens if len(sample_lens) > 1 else None
            )
            x_t, log_prob, _, _ = self._sde_step_with_logprob(
                v_t, 
                timesteps[i], 
                timesteps[i+1] if i+1 < len(timesteps) else timesteps[i]*0, # 最后一个step, timestep是0
                dts[i], 
                x_t, 
                sigma_max=timesteps[1], 
                noise_level=cur_noise_level,
                eta_clamp_mode=eta_clamp_mode,
                generator=sde_generators,
                sample_lens=batched_sample_lens,
            )
            if corrected_sampling:
                after_chunks = x_t.split(sample_lens, dim=0)
                vector_log_prob = (
                    log_prob.reshape(1) if log_prob.ndim == 0 else log_prob
                )
                for sample_index, begin in enumerate(sde_timestep_begins):
                    selected = (
                        i in per_sample_selected_indices[sample_index]
                        if sample_sde_stratified
                        else begin <= i < begin + sample_sde_window_size
                    )
                    if selected:
                        per_sample_latents[sample_index].append(after_chunks[sample_index])
                        per_sample_log_probs[sample_index].append(vector_log_prob[sample_index])
                        per_sample_timesteps[sample_index].append(t)
            elif i >= sde_timestep_begin and i < sde_timestep_begin + sample_sde_window_size:
                all_latents.append(x_t)
                all_log_probs.append(log_prob)
                all_timesteps.append(t)
        unpacked_latent = x_t.split(sample_lens)
        if corrected_sampling:
            if any(
                len(latents) != (2 * sample_sde_window_size if sample_sde_stratified else sample_sde_window_size + 1)
                or len(log_probs) != sample_sde_window_size
                or len(sample_timesteps) != sample_sde_window_size
                for latents, log_probs, sample_timesteps in zip(
                    per_sample_latents,
                    per_sample_log_probs,
                    per_sample_timesteps,
                )
            ):
                raise RuntimeError("G012 per-sample SDE history coverage differs")
            return (
                unpacked_latent,
                per_sample_latents,
                per_sample_log_probs,
                [
                    torch.stack(sample_timesteps).to(log_prob.device)
                    for sample_timesteps in per_sample_timesteps
                ],
            )
        all_timesteps = torch.tensor(all_timesteps, device=log_prob.device)
        return unpacked_latent, all_latents, all_log_probs, all_timesteps.to(log_prob.device)
    
    def generate_image_learn(
        self,
        sample,
        grpo_config,
        accelerator,
        optimizer,
        transformer,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        noise_level: float = 0.7,
        ref_past_key_values: Optional[NaiveCache] = None,
        ref_cfg_text_past_key_values: Optional[NaiveCache] = None,
        ref_cfg_img_past_key_values: Optional[NaiveCache] = None,
        gradient_clip_parameters=None,
        max_grad_norm: float = 1.0,
        isolate_regular_gradients: bool = False,
        perform_optimizer_step: bool = True,
        backward_loss_scale: float = 1.0,
        eta_clamp_mode: Optional[str] = None,
    ):
        # G022 item 4. The learn pass must recompute the SDE transition with the
        # same eta clamp the rollout used, or the recorded log-prob and the
        # replayed one are not the same distribution. Default is read off the
        # config so no caller has to change; an explicit argument wins.
        learn_eta_clamp_mode = str(
            eta_clamp_mode
            if eta_clamp_mode is not None
            else getattr(
                getattr(grpo_config, "g016", None),
                "eta_clamp_mode",
                LEGACY_ETA_CLAMP_MODE,
            )
        )
        replay_device = accelerator.device
        latents = sample["latents"].to(
            replay_device,
            non_blocking=False,
        )
        prev_latents = sample["prev_latents"].to(
            replay_device,
            non_blocking=False,
        )
        timesteps = sample["timesteps"].to(
            replay_device,
            non_blocking=False,
        )
        old_log_probs = sample["log_probs"].to(
            replay_device,
            non_blocking=False,
        )

        original_timesteps = torch.linspace(1, 0, num_timesteps, device=latents.device)
        original_timesteps = timestep_shift * original_timesteps / (1 + (timestep_shift - 1) * original_timesteps)
        dtimesteps = original_timesteps[1:] - original_timesteps[:-1]

        advantages = torch.clamp(
            sample["advantages"].to(
                replay_device,
                non_blocking=False,
            ),
            -grpo_config.train.adv_clip_max,
            grpo_config.train.adv_clip_max,
        )
        policy_active = bool(sample.get("policy_active", True))
        clipfrac=[]
        clipfrac_gt_one=[]
        clipfrac_lt_one=[]
        policy_loss_list=[]
        kl_loss_list=[]
        velocity_mse_list=[]
        reference_velocity_byte_identical_list=[]
        reference_velocity_max_abs_difference_list=[]
        mean_shift_mse_list=[]
        transition_variance_list=[]
        grad_norm_list=[]
        loss_list=[]
        clip_parameters = tuple(
            gradient_clip_parameters
            if gradient_clip_parameters is not None
            else (
                parameter
                for parameter in transformer.parameters()
                if parameter.requires_grad
            )
        )
        if perform_optimizer_step and not clip_parameters:
            raise ValueError("flow gradient clipping parameters are empty")
        if max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if (
            not math.isfinite(float(backward_loss_scale))
            or float(backward_loss_scale) <= 0.0
        ):
            raise ValueError("backward_loss_scale must be finite and positive")
        if perform_optimizer_step and optimizer is None:
            raise ValueError("optimizer is required when stepping flow")
        transition_count = len(timesteps)
        if transition_count <= 0:
            raise ValueError("flow replay has no SDE transitions")
        with torch.no_grad():
            packed_text_embedding_override = self.language_model.forward(
                mode="get_embeddings", input_ids=packed_text_ids
            ).detach()
        language_module = self.language_model
        while hasattr(language_module, "module"):
            language_module = language_module.module
        language_base = (
            language_module.get_base_model()
            if hasattr(language_module, "get_base_model")
            else language_module
        )
        if bool(
            getattr(
                grpo_config.train,
                "offload_text_embeddings_during_backward",
                True,
            )
        ):
            language_base.model.embed_tokens.to(
                device="cpu",
                dtype=torch.bfloat16,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), disable=not accelerator.is_local_main_process):
            timestep = torch.tensor([t] * latents[0].shape[0], device=latents[0].device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            activation_offload_context = (
                torch.autograd.graph.save_on_cpu(pin_memory=False)
                if bool(
                    getattr(
                        grpo_config.train,
                        "activation_cpu_offload",
                        True,
                    )
                )
                else nullcontext()
            )
            def _policy_flow_forward(x_argument, timestep_argument):
                return self._forward_flow(
                    x_t=x_argument,
                    timestep=timestep_argument, 
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_vae_position_ids=packed_vae_position_ids,
                    packed_text_ids=packed_text_ids,
                    packed_text_indexes=packed_text_indexes,
                    packed_text_embedding_override=packed_text_embedding_override,
                    packed_position_ids=packed_position_ids,
                    packed_indexes=packed_indexes,
                    packed_seqlens=packed_seqlens,
                    key_values_lens=key_values_lens,
                    past_key_values=past_key_values,
                    packed_key_value_indexes=packed_key_value_indexes,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    # cfg_text
                    cfg_text_scale=cfg_text_scale_,
                    cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                    cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                    cfg_text_key_values_lens=cfg_text_key_values_lens,
                    cfg_text_past_key_values=cfg_text_past_key_values,
                    cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                    # cfg_img
                    cfg_img_scale=cfg_img_scale_,
                    cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                    cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                    cfg_img_key_values_lens=cfg_img_key_values_lens,
                    cfg_img_past_key_values=cfg_img_past_key_values,
                    cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                    cfg_type=cfg_type,
                    isolate_regular_gradients=bool(
                        isolate_regular_gradients
                    ),
                )
            with activation_offload_context:
                with accelerator.accumulate(transformer):
                    if bool(getattr(grpo_config.g016, "flow_activation_recompute", True)):
                        policy_v_t = checkpoint(
                            _policy_flow_forward,
                            latents[i],
                            timestep,
                            use_reentrant=False,
                            preserve_rng_state=False,
                        )
                    else:
                        policy_v_t = _policy_flow_forward(latents[i], timestep)
            t_index = (original_timesteps == timesteps[i]).nonzero(as_tuple=True)[0]
            if int(t_index.numel()) != 1:
                raise RuntimeError("flow replay timestep does not map to one generation transition")
            t_scalar_index = int(t_index.item())
            transition_next_timestep = (
                original_timesteps[t_scalar_index + 1]
                if t_scalar_index + 1 < len(original_timesteps)
                else timesteps[i] * 0
            )
            _, log_prob, prev_sample_mean, std_dev_t = self._sde_step_with_logprob(
                policy_v_t,
                timesteps[i],
                transition_next_timestep,
                dtimesteps[t_index], 
                latents[i], 
                prev_sample=prev_latents[i], 
                sigma_max=original_timesteps[1], 
                noise_level=noise_level,
                eta_clamp_mode=learn_eta_clamp_mode,
            )
            if grpo_config.train.beta > 0:
                reference_cpu_offload = bool(
                    getattr(
                        grpo_config.train,
                        "reference_cpu_offload",
                        False,
                    )
                )
                reference_model = self.language_model_ref
                reference_parameter = next(reference_model.parameters())
                reference_was_streamed = (
                    reference_cpu_offload
                    and reference_parameter.device != policy_v_t.device
                )
                if reference_was_streamed:
                    reference_model.to(
                        device=policy_v_t.device,
                        dtype=torch.bfloat16,
                    )
                try:
                    with torch.no_grad():
                        reference_v_t = self._forward_flow(
                            x_t=latents[i],
                            timestep=timestep, 
                            packed_vae_token_indexes=packed_vae_token_indexes,
                            packed_vae_position_ids=packed_vae_position_ids,
                            packed_text_ids=packed_text_ids,
                            packed_text_indexes=packed_text_indexes,
                            packed_text_embedding_override=packed_text_embedding_override,
                            packed_position_ids=packed_position_ids,
                            packed_indexes=packed_indexes,
                            packed_seqlens=packed_seqlens,
                            key_values_lens=key_values_lens,
                            past_key_values=(
                                ref_past_key_values
                                if ref_past_key_values is not None
                                else past_key_values
                            ),
                            packed_key_value_indexes=packed_key_value_indexes,
                            cfg_renorm_min=cfg_renorm_min,
                            cfg_renorm_type=cfg_renorm_type,
                            # cfg_text
                            cfg_text_scale=cfg_text_scale_,
                            cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                            cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                            cfg_text_key_values_lens=cfg_text_key_values_lens,
                            cfg_text_past_key_values=(
                                ref_cfg_text_past_key_values
                                if ref_cfg_text_past_key_values is not None
                                else cfg_text_past_key_values
                            ),
                            cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                            # cfg_img
                            cfg_img_scale=cfg_img_scale_,
                            cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                            cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                            cfg_img_key_values_lens=cfg_img_key_values_lens,
                            cfg_img_past_key_values=(
                                ref_cfg_img_past_key_values
                                if ref_cfg_img_past_key_values is not None
                                else cfg_img_past_key_values
                            ),
                            cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                            cfg_type=cfg_type,
                            ref_model=True,
                        )
                finally:
                    if reference_was_streamed:
                        reference_model.to(
                            device="cpu",
                            dtype=torch.bfloat16,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                _, _, prev_sample_mean_ref, _ = self._sde_step_with_logprob(
                    reference_v_t,
                    timesteps[i],
                    transition_next_timestep,
                    dtimesteps[t_index], 
                    latents[i], 
                    prev_sample=prev_latents[i], 
                    sigma_max=original_timesteps[1], 
                    noise_level=noise_level,
                    eta_clamp_mode=learn_eta_clamp_mode,
                )
            
            # grpo logic
            ratio = torch.exp(log_prob - old_log_probs[i])
            # print('ratio', ratio)
            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(
                ratio,
                1.0 - grpo_config.train.clip_range_lt,
                1.0 + grpo_config.train.clip_range_gt,
            )
            policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
            if grpo_config.train.beta > 0:
                kl_loss, kl_diagnostics = equal_variance_transition_kl(
                    prev_sample_mean,
                    prev_sample_mean_ref,
                    std_dev_t,
                    dtimesteps[t_index],
                )
                velocity_mse = (
                    policy_v_t.float() - reference_v_t.float()
                ).square().mean()
                reference_velocity_byte_identical_list.append(
                    torch.tensor(
                        int(torch.equal(policy_v_t, reference_v_t)),
                        device=policy_v_t.device,
                        dtype=torch.int32,
                    )
                )
                reference_velocity_max_abs_difference_list.append(
                    (
                        policy_v_t.float() - reference_v_t.float()
                    )
                    .abs()
                    .max()
                    .detach()
                )
                loss = policy_loss + grpo_config.train.beta * kl_loss
            else:
                loss = policy_loss
            if not policy_active:
                policy_loss = policy_loss * 0.0
                if grpo_config.train.beta > 0:
                    kl_loss = kl_loss * 0.0
                    velocity_mse = velocity_mse * 0.0
                loss = loss * 0.0
            if not torch.isfinite(loss).all():
                raise FloatingPointError(
                    "nonfinite direct Flow-GRPO loss before optimizer update"
                )
            clipfrac.append(
                torch.mean(
                    (
                        ratio - 1.0 > grpo_config.train.clip_range_gt or 1.0 - ratio > grpo_config.train.clip_range_lt
                    ).float()
                )
            )
            clipfrac_gt_one.append(
                torch.mean(
                    (
                        ratio - 1.0 > grpo_config.train.clip_range_gt
                    ).float()
                )
            )
            clipfrac_lt_one.append(
                torch.mean(
                    (
                        1.0 - ratio > grpo_config.train.clip_range_lt
                    ).float()
                )
            )
            policy_loss_list.append(policy_loss)
            if grpo_config.train.beta > 0:
                kl_loss_list.append(kl_loss)
                velocity_mse_list.append(velocity_mse.detach())
                mean_shift_mse_list.append(
                    kl_diagnostics["mean_shift_mse"]
                )
                transition_variance_list.append(
                    kl_diagnostics["transition_variance"]
                )
            loss_list.append(loss)
            backward_loss = (
                loss
                if perform_optimizer_step
                else (
                    loss
                    * float(backward_loss_scale)
                    / float(transition_count)
                )
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            accelerator.backward(backward_loss)
            if perform_optimizer_step and accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(
                    clip_parameters,
                    float(max_grad_norm),
                )
                grad_norm_list.append(
                    torch.as_tensor(grad_norm).detach().float()
                )
            if perform_optimizer_step:
                optimizer.step()
                optimizer.zero_grad()
        policy_loss = torch.cat([t.unsqueeze(0) for t in policy_loss_list]).mean().detach()
        loss = torch.cat([t.unsqueeze(0) for t in loss_list]).mean().detach()
        clipfrac = torch.cat([t.unsqueeze(0) for t in clipfrac]).mean().detach()
        clipfrac_gt_one = torch.cat([t.unsqueeze(0) for t in clipfrac_gt_one]).mean().detach()
        clipfrac_lt_one = torch.cat([t.unsqueeze(0) for t in clipfrac_lt_one]).mean().detach()

        if grpo_config.train.beta > 0:
            kl_loss = torch.cat([t.unsqueeze(0) for t in kl_loss_list]).mean().detach()
            velocity_mse = torch.stack(velocity_mse_list).mean()
            mean_shift_mse = torch.stack(mean_shift_mse_list).mean()
            transition_variance = torch.stack(
                transition_variance_list
            ).mean()
            reference_velocity_all_byte_identical = bool(
                torch.stack(reference_velocity_byte_identical_list)
                .bool()
                .all()
                .item()
            )
            reference_velocity_max_abs_difference = torch.stack(
                reference_velocity_max_abs_difference_list
            ).max()
        else:
            kl_loss = policy_loss*0-1
            velocity_mse = policy_loss * 0.0 - 1
            mean_shift_mse = policy_loss * 0.0 - 1
            transition_variance = policy_loss * 0.0 - 1
            reference_velocity_all_byte_identical = False
            reference_velocity_max_abs_difference = (
                policy_loss * 0.0 - 1
            )
        grad_norm = (
            torch.stack(grad_norm_list).mean()
            if grad_norm_list
            else policy_loss * 0.0
        )
        grad_norm_max = (
            torch.stack(grad_norm_list).max()
            if grad_norm_list
            else policy_loss * 0.0
        )
        flow_diagnostics = {
            "flow_kl_version": FLOW_KL_VERSION,
            "reference_distribution": (
                "frozen_classic_equal_variance_sde_transition"
            ),
            "velocity_mse": velocity_mse.detach(),
            "transition_mean_shift_mse": mean_shift_mse.detach(),
            "transition_variance": transition_variance.detach(),
            "weighted_kl_loss": (
                float(grpo_config.train.beta) * kl_loss
            ).detach(),
            "grad_norm": grad_norm.detach(),
            "grad_norm_max": grad_norm_max.detach(),
            "max_grad_norm": float(max_grad_norm),
            "perform_optimizer_step": bool(perform_optimizer_step),
            "backward_loss_scale": float(backward_loss_scale),
            "transition_count": int(transition_count),
            "reference_velocity_all_byte_identical": (
                reference_velocity_all_byte_identical
            ),
            "reference_velocity_max_abs_difference": (
                reference_velocity_max_abs_difference.detach()
            ),
        }
        return (
            clipfrac,
            clipfrac_gt_one,
            clipfrac_lt_one,
            policy_loss,
            kl_loss,
            loss,
            flow_diagnostics,
        )

    def _sde_step_with_logprob(
        self,
        model_output,
        timestep,
        prev_timestep,
        d_timestep,
        sample,
        prev_sample = None,
        generator = None,
        sigma_max = None,
        noise_level: float = 0.8,
        sde_type: str = "sde",
        sample_lens: Optional[List[int]] = None,
        eta_clamp_mode: str = LEGACY_ETA_CLAMP_MODE,
    ):
        # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
        model_output=model_output.float()
        sample=sample.float()
        if prev_sample is not None:
            prev_sample=prev_sample.float()
        
        if sde_type == "sde":
            std_dev_t = torch.sqrt(
                timestep / g022_sde_denominator(
                    timestep,
                    sigma_max=sigma_max,
                    eta_clamp_mode=eta_clamp_mode,
                )
            ) * noise_level
            prev_sample_mean = sample*(1+std_dev_t**2/(2*timestep)*d_timestep)+model_output*(1+std_dev_t**2*(1-timestep)/(2*timestep))*d_timestep

            if prev_sample is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
                    " `prev_sample` stays `None`."
                )

            if prev_sample is None:
                if isinstance(generator, (list, tuple)):
                    if sample_lens is None:
                        raise ValueError(
                            "per-sample generators require sample_lens"
                        )
                    if len(generator) != len(sample_lens):
                        raise ValueError(
                            "generator/sample_lens cardinality mismatch"
                        )
                    variance_noise = torch.cat(
                        [
                            randn_tensor(
                                (int(length), *model_output.shape[1:]),
                                generator=sample_generator,
                                device=model_output.device,
                                dtype=model_output.dtype,
                            )
                            for length, sample_generator in zip(
                                sample_lens,
                                generator,
                            )
                        ],
                        dim=0,
                    )
                else:
                    variance_noise = randn_tensor(
                        model_output.shape,
                        generator=generator,
                        device=model_output.device,
                        dtype=model_output.dtype,
                    )
                prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*d_timestep) * variance_noise

            log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*d_timestep))**2))

        elif sde_type == "cps":
            std_dev_t = prev_timestep  * math.sin(noise_level * math.pi / 2) # sigma_t in paper
            prev_original_sample = sample - timestep * model_output # predicted x_0 in paper
            noise_estimate = sample + model_output * (1 - timestep) # predicted x_1 in paper
            prev_sample_mean = prev_original_sample * (1 - prev_timestep) + noise_estimate * torch.sqrt(prev_timestep**2 - std_dev_t**2)

            if prev_sample is None:
                if isinstance(generator, (list, tuple)):
                    if sample_lens is None:
                        raise ValueError(
                            "per-sample generators require sample_lens"
                        )
                    if len(generator) != len(sample_lens):
                        raise ValueError(
                            "generator/sample_lens cardinality mismatch"
                        )
                    variance_noise = torch.cat(
                        [
                            randn_tensor(
                                (int(length), *model_output.shape[1:]),
                                generator=sample_generator,
                                device=model_output.device,
                                dtype=model_output.dtype,
                            )
                            for length, sample_generator in zip(
                                sample_lens,
                                generator,
                            )
                        ],
                        dim=0,
                    )
                else:
                    variance_noise = randn_tensor(
                        model_output.shape,
                        generator=generator,
                        device=model_output.device,
                        dtype=model_output.dtype,
                    )
                prev_sample = prev_sample_mean + std_dev_t * variance_noise

            log_prob = -((prev_sample.detach() - prev_sample_mean)**2)

        if sample_lens is None:
            log_prob = log_prob.mean()
        else:
            if sum(int(length) for length in sample_lens) != log_prob.shape[0]:
                raise ValueError("sample_lens do not cover packed SDE state")
            log_prob = torch.stack(
                [
                    value.mean()
                    for value in log_prob.split(
                        [int(length) for length in sample_lens],
                        dim=0,
                    )
                ]
            )

        return prev_sample, log_prob, prev_sample_mean, std_dev_t



    # @torch.no_grad
    def _forward_flow(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        packed_text_embedding_override: Optional[torch.Tensor] = None,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        ref_model: bool = False,
        isolate_regular_gradients: bool = False,
    ):  
        if ref_model:
            forward_model = self.language_model_ref
        else:
            forward_model = self.language_model
        packed_text_embedding = (
            packed_text_embedding_override
            if packed_text_embedding_override is not None
            else forward_model.forward(mode="get_embeddings", input_ids=packed_text_ids)
        )
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)
        x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes,
                "isolate_regular_gradients": bool(
                    isolate_regular_gradients
                ),
            }
        def decoder_branch(
            sequence,
            *,
            position_ids,
            query_indexes,
            branch_past_key_values,
            branch_key_values_lens,
            branch_key_value_indexes,
        ):
            def run(sequence_argument):
                result = forward_model.forward(
                    packed_query_sequence=sequence_argument,
                    query_lens=packed_seqlens,
                    packed_query_position_ids=position_ids,
                    packed_query_indexes=query_indexes,
                    past_key_values=branch_past_key_values,
                    key_values_lens=branch_key_values_lens,
                    packed_key_value_indexes=branch_key_value_indexes,
                    update_past_key_values=False,
                    is_causal=False,
                    **extra_inputs,
                ).packed_query_sequence
                return result
            if not ref_model and torch.is_grad_enabled():
                return checkpoint(
                    run,
                    sequence,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            return run(sequence)

        output_sequence = decoder_branch(
            packed_sequence,
            position_ids=packed_position_ids,
            query_indexes=packed_indexes,
            branch_past_key_values=past_key_values,
            branch_key_values_lens=key_values_lens,
            branch_key_value_indexes=packed_key_value_indexes,
        )
        v_t = self.llm2vae(output_sequence)
        v_t = v_t[packed_vae_token_indexes]
        if cfg_text_scale > 1.0:
            cfg_text_sequence = decoder_branch(
                packed_sequence,
                position_ids=cfg_text_packed_position_ids,
                query_indexes=cfg_text_packed_query_indexes,
                branch_past_key_values=cfg_text_past_key_values,
                branch_key_values_lens=cfg_text_key_values_lens,
                branch_key_value_indexes=cfg_text_packed_key_value_indexes,
            )
            cfg_text_v_t = self.llm2vae(cfg_text_sequence)
            cfg_text_v_t = cfg_text_v_t[packed_vae_token_indexes]
        if cfg_img_scale > 1.0:
            cfg_img_sequence = decoder_branch(
                packed_sequence,
                position_ids=cfg_img_packed_position_ids,
                query_indexes=cfg_img_packed_query_indexes,
                branch_past_key_values=cfg_img_past_key_values,
                branch_key_values_lens=cfg_img_key_values_lens,
                branch_key_value_indexes=cfg_img_packed_key_value_indexes,
            )
            cfg_img_v_t = self.llm2vae(cfg_img_sequence)
            cfg_img_v_t = cfg_img_v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                
                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                if cfg_renorm_type == "global":
                    latent_lens = [
                        int(length) - 2 for length in packed_seqlens.tolist()
                    ]
                    v_t = _packed_global_cfg_renorm(
                        v_t,
                        v_t_,
                        latent_lens,
                        minimum=cfg_renorm_min,
                    )
                    return v_t
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t = v_t_ * scale
        else:
            # No CFG
            pass

        return v_t

    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids):
        packed_start_tokens, packed_key_value_indexes = list(), list()
        packed_query_position_ids = list()

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(new_token_ids['bos_token_id'])
            packed_query_position_ids.append(curr_position_id)
            curr += curr_kvlen

        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_text(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
    ):
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens), 
                device=key_values_lens.device, 
                dtype=key_values_lens.dtype
            )

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            if self.use_moe:
                extra_inputs = {"mode": "und"}

            output = self.language_model.forward(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            pred_logits = self.language_model.lm_head(packed_query_sequence)

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id: # only support batch=1
                break

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)

    # for evaluation
    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        new_token_ids,
        image_transform,
        images,
        prompt,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        device = next(self.parameters()).device

        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(device)
        elif torch.is_tensor(new_token_ids):
            new_token_ids = new_token_ids.to(device)

        # prefill
        past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        # add images
        for image in images:
            generation_input, newlens, new_rope = self.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope, 
                images=[image], 
                transforms=image_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.forward_cache_update_vit(past_key_values, **generation_input)

        # add text
        generation_input, newlens, new_rope = self.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[prompt],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)

        # decode
        generation_input = self.prepare_start_tokens(newlens, new_rope, new_token_ids)
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = self.generate_text(
                past_key_values=past_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=new_token_ids['eos_token_id'],
                **generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]

        return output
