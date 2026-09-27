# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import gc
import json
import os
import wandb
import yaml
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.modeling_utils import no_init_weights
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from modeling.autoencoder import default_ae_params, load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from train.train_utils import create_logger, get_latest_ckpt
from train.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper, 
    fsdp_ema_setup, fsdp_ema_update,
)


import ctypes

def _release_cpu_memory():
    """Force Python to return freed memory to OS."""
    gc.collect()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def qwen2_flop_coefficients(config) -> tuple[float, float]:
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size
    head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)

    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    dense_token_factor = 6.0 * dense_N
    attn_factor = 12.0 * head_dim * num_attention_heads * num_hidden_layers
    return dense_token_factor, attn_factor


def detect_peak_tflops(default_tflops: float) -> float:
    """Guess per-device BF16 TFLOPs from GPU name; fall back to default when unknown."""
    try:
        import torch
        device_name = torch.cuda.get_device_name()
    except (ImportError, RuntimeError):
        return default_tflops

    name = device_name.upper()
    if "MI300X" in name:
        tflops = 1336.0
    elif any(tag in name for tag in ("H100", "H800", "H200")):
        tflops = 989.0
    elif any(tag in name for tag in ("A100", "A800")):
        tflops = 312.0
    elif "L40" in name:
        tflops = 181.05
    elif "L20" in name:
        tflops = 119.5
    elif "H20" in name:
        tflops = 148.0
    elif "910B" in name:
        tflops = 354.0
    elif "RTX 3070 TI" in name:
        tflops = 21.75
    else:
        tflops = default_tflops
    return tflops


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_sample_milestones(value: str, target: int = 0) -> list[int]:
    milestones = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed <= 0:
            raise ValueError(f"sample milestones must be positive: {parsed}")
        milestones.append(parsed)
    if int(target) > 0:
        milestones.append(int(target))
    return sorted(set(milestones))


def newly_crossed_sample_milestones(
    previous: int,
    current: int,
    milestones: list[int],
    already_crossed,
) -> list[int]:
    crossed = set(int(value) for value in (already_crossed or []))
    return [
        milestone
        for milestone in milestones
        if milestone not in crossed and previous < milestone <= current
    ]


def validate_resume_sample_state(
    sample_state,
    *,
    target_global_samples,
    sample_milestones,
    recovery_save_every_samples,
    world_size,
    train_step,
    resume_from,
):
    if not isinstance(sample_state, dict):
        raise RuntimeError("resume sample_state must be a mapping")
    if int(sample_state.get("version", -1)) != 1:
        raise RuntimeError("resume sample_state version mismatch")
    expected_last_step = int(train_step) - 1
    checkpoint_step = int(
        os.path.basename(os.path.normpath(str(resume_from)))
    )
    if checkpoint_step != expected_last_step:
        raise RuntimeError(
            "resume checkpoint directory step does not match train_step"
        )
    if int(sample_state.get("last_completed_step", -1)) != expected_last_step:
        raise RuntimeError("resume sample_state last_completed_step mismatch")
    if int(sample_state.get("world_size", -1)) != int(world_size):
        raise RuntimeError("resume sample_state world_size mismatch")
    if int(sample_state.get("target_global_samples", -1)) != int(
        target_global_samples
    ):
        raise RuntimeError("resume sample_state target mismatch")
    recorded_milestones = [
        int(value) for value in sample_state.get("sample_milestones", [])
    ]
    if recorded_milestones != list(sample_milestones):
        raise RuntimeError("resume sample_state milestones mismatch")
    if int(
        sample_state.get("recovery_save_every_samples", -1)
    ) != int(recovery_save_every_samples):
        raise RuntimeError("resume sample_state recovery cadence mismatch")

    cumulative = int(sample_state.get("cumulative_global_samples", -1))
    if cumulative < 0:
        raise RuntimeError("resume sample_state cumulative samples invalid")
    last_global = int(sample_state.get("global_samples_last_step", -1))
    if last_global <= 0 or last_global > cumulative:
        raise RuntimeError("resume sample_state last global samples invalid")
    expected_crossed = [
        milestone
        for milestone in sample_milestones
        if milestone <= cumulative
    ]
    crossed = [
        int(value)
        for value in sample_state.get("crossed_milestones", [])
    ]
    if crossed != expected_crossed:
        raise RuntimeError("resume sample_state crossed milestones mismatch")
    newly_crossed = {
        int(value)
        for value in sample_state.get("newly_crossed_milestones", [])
    }
    if not newly_crossed.issubset(set(crossed)):
        raise RuntimeError(
            "resume sample_state newly crossed milestones mismatch"
        )

    expected_recovery = []
    if recovery_save_every_samples > 0:
        expected_recovery = list(
            range(
                recovery_save_every_samples,
                cumulative + 1,
                recovery_save_every_samples,
            )
        )
    recorded_recovery = [
        int(value)
        for value in sample_state.get("crossed_recovery_samples", [])
    ]
    if recorded_recovery != expected_recovery:
        raise RuntimeError(
            "resume sample_state crossed recovery samples mismatch"
        )
    new_recovery = {
        int(value)
        for value in sample_state.get(
            "newly_crossed_recovery_samples",
            [],
        )
    }
    if not new_recovery.issubset(set(recorded_recovery)):
        raise RuntimeError(
            "resume sample_state newly crossed recovery mismatch"
        )

    target_reached = (
        target_global_samples > 0
        and cumulative >= target_global_samples
    )
    if bool(sample_state.get("target_reached")) != target_reached:
        raise RuntimeError("resume sample_state target_reached mismatch")
    expected_overshoot = (
        max(0, cumulative - target_global_samples)
        if target_global_samples > 0
        else 0
    )
    if int(sample_state.get("target_overshoot", -1)) != expected_overshoot:
        raise RuntimeError("resume sample_state target overshoot mismatch")
    return {
        "cumulative_global_samples": cumulative,
        "crossed_milestones": crossed,
        "crossed_recovery_samples": recorded_recovery,
    }


def is_fsdp_checkpoint_source_rank(training_args) -> bool:
    """Return ranks that must materialize/load weights before FSDP sync."""
    rank = dist.get_rank()
    if training_args.sharding_strategy == "HYBRID_SHARD":
        expected_world_size = training_args.num_replicate * training_args.num_shard
        if expected_world_size != dist.get_world_size():
            raise ValueError(
                "HYBRID_SHARD requires num_replicate * num_shard == world_size, "
                f"got {training_args.num_replicate} * {training_args.num_shard} "
                f"!= {dist.get_world_size()}"
            )
        # Each shard group has its own rank-0 source. For the usual 2x8 mesh,
        # those are global ranks 0 and 8.
        return rank % training_args.num_shard == 0
    return rank == 0


@dataclass
class ModelArguments:
    model_path: str = field(
        default="hf/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."}
    )
    llm_path: str = field(
        default="hf/Qwen2.5-0.5B-Instruct/",
        metadata={"help": "Path or HuggingFace repo ID of the pretrained Qwen2-style language model."}
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."}
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."}
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."}
    )
    vae_path: str = field(
        default="flux/vae/ae.safetensors",
        metadata={"help": "Path to the pretrained VAE checkpoint for latent-space image generation."}
    )
    vit_path: str = field(
        default="hf/siglip-so400m-14-980-flash-attn2-navit/",
        metadata={"help": "Path or repo ID of the SigLIP Vision Transformer used for image understanding."}
    )
    max_latent_size: int = field(
        default=32,
        metadata={"help": "Maximum latent grid size (patches per side) for the VAE latent tensor."}
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Spatial size (in VAE pixels) covered by each latent patch."}
    )
    vit_patch_size: int = field(
        default=14,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."}
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={"help": "Maximum number of ViT patches along one image side after cropping / resize."}
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function used in the latent-to-text connector MLP."}
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={"help": "Interpolate positional embeddings when image resolution differs from pre-training."}
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={"help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."}
    )
    vit_rope: bool = field(
        default=False,
        metadata={"help": "Replace ViT positional encodings with RoPE."}
    )

    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping text embeddings during training."}
    )
    vae_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping VAE latent inputs during training."}
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping ViT visual features during training."}
    )


@dataclass
class DataArguments:
    dataset_config_file: str = field(
        default="data/configs/example.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."}
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    max_num_tokens_per_sample: int = field(
        default=16384,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."}
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={"help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."}
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={"help": "While batch length is below this, pop from the overflow buffer before new sampling."}
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."}
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Seed used when shuffling / sampling data shards to ensure reproducibility."}
    )


@dataclass
class TrainingArguments:
    # --- modality switches ---
    visual_gen: bool = field(
        default=True,
        metadata={"help": "Train image generation branch."}
    )
    visual_und: bool = field(
        default=True,
        metadata={"help": "Train image understanding branch."}
    )

    # --- bookkeeping & logging ---
    results_dir: str = field(
        default="results",
        metadata={"help": "Root directory for logs."}
    )
    checkpoint_dir: str = field(
        default="results/checkpoints",
        metadata={"help": "Root directory for model checkpoints."}
    )
    wandb_project: str = field(
        default="bagel",
        metadata={"help": "Weights & Biases project name."}
    )
    wandb_name: str = field(
        default="run",
        metadata={"help": "Name shown in the Weights & Biases UI for this run."}
    )
    wandb_runid: str = field(
        default="0",
        metadata={"help": "Unique identifier to resume a previous W&B run, if desired."}
    )
    wandb_resume: str = field(
        default="allow",
        metadata={"help": "W&B resume mode: 'allow', 'must', or 'never'."}
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "Run W&B in offline mode (logs locally, sync later)."}
    )

    # --- reproducibility & resume ---
    global_seed: int = field(
        default=4396,
        metadata={"help": "Base random seed; actual seed is offset by rank for DDP."}
    )
    auto_resume: bool = field(
        default=False,
        metadata={"help": "Automatically pick up the latest checkpoint found in checkpoint_dir."}
    )
    resume_from: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)." }
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only model weights, ignoring optimizer/scheduler states."}
    )
    finetune_from_ema: bool = field(
        default=False,
        metadata={"help": "When resume_model_only=True, load the EMA (exponential moving average) weights instead of raw weights."}
    )
    finetune_from_hf: bool = field(
        default=False,
        metadata={"help": "Whether finetune from HugginFace model."}
    )

    # --- reporting frequency ---
    log_every: int = field(
        default=10,
        metadata={"help": "Print / log every N training steps."}
    )
    save_every: int = field(
        default=2000,
        metadata={"help": "Save a checkpoint every N training steps."}
    )
    total_steps: int = field(
        default=500_000,
        metadata={"help": "Total number of optimizer steps to train for."}
    )
    lr_schedule_steps: int = field(
        default=0,
        metadata={
            "help": (
                "Cosine schedule horizon; zero uses total_steps. This may be "
                "shorter than a separate hard total_steps ceiling."
            )
        },
    )
    target_global_samples: int = field(
        default=0,
        metadata={
            "help": "Opt-in cumulative global sample target; zero disables it."
        },
    )
    sample_milestones: str = field(
        default="",
        metadata={
            "help": "Comma-separated cumulative global sample milestones."
        },
    )
    write_checkpoint_completion_marker: bool = field(
        default=False,
        metadata={
            "help": "Hash checkpoint files and write CHECKPOINT_COMPLETE.json."
        },
    )
    recovery_save_every_samples: int = field(
        default=0,
        metadata={
            "help": (
                "Save a recovery checkpoint at every cumulative sample "
                "multiple; zero disables sample-periodic recovery saves."
            )
        },
    )

    # --- optimization & scheduler ---
    warmup_steps: int = field(
        default=2000,
        metadata={"help": "Linear warm-up steps before applying the main LR schedule."}
    )
    lr_scheduler: str = field(
        default="constant",
        metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."}
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Peak learning rate after warm-up."}
    )
    min_lr: float = field(
        default=1e-7,
        metadata={"help": "Minimum learning rate for cosine schedule (ignored for constant)."}
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW β₁ coefficient."}
    )
    beta2: float = field(
        default=0.95,
        metadata={"help": "AdamW β₂ coefficient."}
    )
    eps: float = field(
        default=1e-15,
        metadata={"help": "AdamW ε for numerical stability."}
    )
    ema: float = field(
        default=0.9999,
        metadata={"help": "Decay rate for the exponential moving average of model weights."}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Gradient clipping threshold (L2 norm)."}
    )
    timestep_shift: float = field(
        default=1.0,
        metadata={"help": "Shift applied to diffusion timestep indices (for latent prediction)."}
    )
    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the image-reconstruction MSE loss term."}
    )
    ce_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )
    ce_loss_reweighting: bool = field(
        default=False,
        metadata={"help": "Reweight CE loss by token importance (provided via ce_loss_weights)."}
    )
    expected_num_tokens: int = field(
        default=32768,
        metadata={"help": "Soft target token count; yield the batch once it reaches or exceeds this size."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    peak_device_tflops: float = field(
        default=0.0,
        metadata={"help": "Per-GPU peak BF16 TFLOPs used to compute MFU; leave at 0 to auto-detect."}
    )

    # --- distributed training / FSDP ---
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )
    num_shard: int = field(
        default=8,
        metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )
    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."}
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."}
    )
    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Enable FSDP parameter offload to CPU."}
    )

    # --- module freezing ---
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_vae: bool = field(
        default=True,
        metadata={"help": "Keep VAE weights fixed; only predict latents, don’t fine-tune encoder/decoder."}
    )
    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze the visual understanding connector layers."}
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."}
    )
    use_flex: bool = field(
        default=False,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Use LoRA for parameter-efficient training."}
    )
    lora_rank: int = field(
        default=64,
        metadata={"help": "LoRA rank."}
    )


def main():
    assert torch.cuda.is_available()
    torch.set_default_dtype(torch.bfloat16)  # reduce CPU mem: 8 ranks × model+EMA must fit in 885GB
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=torch.device(f"cuda:{local_rank}"))
    device = local_rank
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.peak_device_tflops <= 0:
        auto_tflops = detect_peak_tflops(training_args.peak_device_tflops)
        if auto_tflops > 0:
            training_args.peak_device_tflops = auto_tflops

    # Setup logging:
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        wandb.init(
            project=training_args.wandb_project, 
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}", 
            name=training_args.wandb_name, 
            resume=training_args.wandb_resume,
            mode="offline" if training_args.wandb_offline else "online",
            settings=wandb.Settings(init_timeout=120)
        )
        wandb.config.update(training_args)
        wandb.config.update(model_args)
        wandb.config.update(data_args)
        if training_args.peak_device_tflops > 0:
            logger.info(f"Using peak_device_tflops={training_args.peak_device_tflops:.2f} TFLOPs (per GPU).")
        else:
            logger.warning("Peak device TFLOPs not set or auto-detected; MFU will report 0.")
    else:
        logger = create_logger(None, dist.get_rank())
    dist.barrier()
    logger.info(f'Training arguments {training_args}')
    logger.info(f'Model arguments {model_args}')
    logger.info(f'Data arguments {data_args}')
    using_encoded_failaware_cache = (
        os.environ.get("FAILAWARE_USE_CACHE", "0") == "1"
        and os.environ.get("FAILAWARE_CACHE_MODE", "auto").lower() == "encoded"
    )

    # prepare auto resume logic:
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            if resume_model_only:
                finetune_from_ema = training_args.finetune_from_ema
            else:
                finetune_from_ema = False
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        if resume_model_only:
            finetune_from_ema = training_args.finetune_from_ema
        else:
            finetune_from_ema = False

    # Set seed:
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # Setup model:
    # Only FSDP source ranks create/load real CPU weights; other ranks use meta.
    # FULL_SHARD has one source rank. HYBRID_SHARD has one source per shard group.
    local_rank = dist.get_rank() % torch.cuda.device_count()
    is_checkpoint_source_rank = is_fsdp_checkpoint_source_rank(training_args)
    skip_pretrained_init = resume_from is not None

    if training_args.finetune_from_hf:
        llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
    else:
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und

    if is_checkpoint_source_rank:
        with no_init_weights(_enable=skip_pretrained_init):
            if training_args.finetune_from_hf:
                language_model = Qwen2ForCausalLM(llm_config)
            else:
                language_model = Qwen2ForCausalLM.from_pretrained(model_args.llm_path, config=llm_config)
    else:
        # Other ranks: meta device, 0 bytes
        with torch.device('meta'), no_init_weights(_enable=True):
            language_model = Qwen2ForCausalLM(llm_config)

    if training_args.copy_init_moe and resume_from is None and is_checkpoint_source_rank:
        language_model.init_moe()

    if training_args.visual_und:
        if training_args.finetune_from_hf:
            vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_args.model_path, "vit_config.json"))
        else:
            vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
        vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
        vit_config.rope = model_args.vit_rope
        if is_checkpoint_source_rank:
            with no_init_weights(_enable=skip_pretrained_init):
                if training_args.finetune_from_hf:
                    vit_model = SiglipVisionModel(vit_config)
                else:
                    vit_model = SiglipVisionModel.from_pretrained(model_args.vit_path, config=vit_config)
        else:
            with torch.device('meta'), no_init_weights(_enable=True):
                vit_model = SiglipVisionModel(vit_config)

    if training_args.visual_gen:
        if using_encoded_failaware_cache:
            vae_model = None
            vae_config = default_ae_params()
            logger.info("Skipping frozen VAE module load because FAILAWARE_CACHE_MODE=encoded.")
        else:
            vae_model, vae_config = load_ae(
                local_path=os.path.join(model_args.model_path, "ae.safetensors") 
                if training_args.finetune_from_hf else model_args.vae_path
            )

    config = BagelConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config, 
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
    )
    if is_checkpoint_source_rank:
        model = Bagel(language_model, vit_model if training_args.visual_und else None, config)
        if training_args.visual_und:
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
    else:
        with torch.device('meta'):
            model = Bagel(language_model, vit_model if training_args.visual_und else None, config)
        if training_args.visual_und:
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    total_param_count = count_parameters(model)
    lm_param_count = count_parameters(model.language_model)
    logger.info(f"Model parameter count: {total_param_count / 1e9:.2f}B (LM-only: {lm_param_count / 1e9:.2f}B)")

    # Setup tokenizer for model:
    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path if training_args.finetune_from_hf else model_args.llm_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # maybe freeze something:
    if training_args.freeze_vae and training_args.visual_gen and vae_model is not None:
        for param in vae_model.parameters():
            param.requires_grad = False
    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False

    # Setup FSDP and load pretrained model:
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )
    # Source ranks load checkpoints; FSDP sync_module_states broadcasts within each shard group.
    checkpoint_load_result = None
    if is_checkpoint_source_rank:
        model, _, checkpoint_load_result = FSDPCheckpoint.try_load_ckpt(
            resume_from, logger, model, None, resume_from_ema=finetune_from_ema
        )
        if checkpoint_load_result is not None:
            checkpoint_load_message = "[checkpoint-load-gate] " + json.dumps(
                {
                    "missing_keys": checkpoint_load_result["missing_keys"],
                    "rank": dist.get_rank(),
                    "resume_from": resume_from,
                    "unexpected_keys": checkpoint_load_result["unexpected_keys"],
                },
                sort_keys=True,
            )
            logger.info(checkpoint_load_message)
            print(checkpoint_load_message, flush=True)
        _release_cpu_memory()
    dist.barrier()
    if training_args.sharding_strategy == "HYBRID_SHARD":
        logger.info(f"Checkpoint loaded on HYBRID_SHARD source ranks rank % {training_args.num_shard} == 0.")
    else:
        logger.info("Checkpoint loaded on rank 0.")

    # Inject LoRA if requested (before FSDP wrapping)
    if training_args.use_lora:
        from train.lora_utils import inject_lora
        model = inject_lora(model, rank=training_args.lora_rank, alpha=training_args.lora_rank * 2)
        logger.info(f"LoRA injected (rank={training_args.lora_rank}).")

    # EMA disabled for faster init + less memory
    ema_model = None
    logger.info("EMA disabled.")

    fsdp_model = fsdp_wrapper(model, fsdp_config)
    checkpoint_gate_message = (
        f"[checkpoint-gate] rank={dist.get_rank()} ready=1 "
        f"source_rank={int(is_checkpoint_source_rank)} "
        f"resume_from={resume_from} "
        f"resume_model_only={resume_model_only} "
        f"auto_resume={training_args.auto_resume}"
    )
    logger.info(checkpoint_gate_message)
    print(checkpoint_gate_message, flush=True)
    dist.barrier()
    apply_activation_checkpointing(
        fsdp_model, 
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ), 
        check_fn=grad_checkpoint_check_fn
    )

    if dist.get_rank() == 0:
        print(fsdp_model)
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(), 
        lr=training_args.lr, 
        betas=(training_args.beta1, training_args.beta2), 
        eps=training_args.eps, 
        weight_decay=0
    )
    if training_args.lr_scheduler == 'cosine':
        lr_schedule_steps = (
            training_args.lr_schedule_steps
            if training_args.lr_schedule_steps > 0
            else training_args.total_steps
        )
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=lr_schedule_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError

    # maybe resume optimizer, scheduler, and train_steps
    if resume_model_only:
        train_step = 0
        data_status = None
        sample_state = None
    else:
        (
            optimizer,
            scheduler,
            train_step,
            data_status,
            sample_state,
        ) = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config, 
        )

    # Setup packed dataloader
    with open(data_args.dataset_config_file, "r") as stream:
        dataset_text = os.path.expandvars(stream.read())
    if "${" in dataset_text:
        raise ValueError(
            f"unset environment variable in {data_args.dataset_config_file} "
            "(export UNIFY_RL_DATA_ROOT)"
        )
    dataset_meta = yaml.safe_load(dataset_text)
    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if training_args.visual_gen:
        vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
        dataset_config.vae_image_downsample = vae_image_downsample
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
    )
    train_dataset.set_epoch(data_args.data_seed)
    loader_kwargs = dict(
        batch_size=1,
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
    )
    if data_args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = data_args.prefetch_factor
    train_loader = DataLoader(train_dataset, **loader_kwargs)

    # Prepare models for training:
    if training_args.visual_gen and vae_model is not None:
        vae_model.to(device).eval()
    fsdp_model.train()
    if ema_model is not None:
        ema_model.eval()

    # train loop
    start_time = time()
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")
    optimizer.zero_grad()
    total_norm = torch.tensor(0.0, device=device)
    token_window = 0.0
    seqlen_square_window = 0.0
    target_global_samples = int(training_args.target_global_samples)
    sample_milestones = parse_sample_milestones(
        training_args.sample_milestones,
        target_global_samples,
    )
    recovery_save_every_samples = int(
        training_args.recovery_save_every_samples
    )
    if recovery_save_every_samples < 0:
        raise ValueError("recovery_save_every_samples must be non-negative")
    if (
        target_global_samples > 0
        and not resume_model_only
        and resume_from is not None
        and sample_state is None
    ):
        raise RuntimeError(
            "target_global_samples requires sample_state.json on optimizer resume"
        )
    if sample_state is not None:
        validated_sample_state = validate_resume_sample_state(
            sample_state,
            target_global_samples=target_global_samples,
            sample_milestones=sample_milestones,
            recovery_save_every_samples=recovery_save_every_samples,
            world_size=dist.get_world_size(),
            train_step=train_step,
            resume_from=resume_from,
        )
    else:
        validated_sample_state = {
            "cumulative_global_samples": 0,
            "crossed_milestones": [],
            "crossed_recovery_samples": [],
        }
    cumulative_global_samples = int(
        validated_sample_state["cumulative_global_samples"]
    )
    crossed_sample_milestones = set(
        validated_sample_state["crossed_milestones"]
    )
    crossed_recovery_samples = set(
        validated_sample_state["crossed_recovery_samples"]
    )
    pending_global_samples = 0
    last_global_samples = 0
    last_completed_step = train_step - 1
    last_saved_step = None
    curr_step = train_step - 1
    target_reached = (
        target_global_samples > 0
        and cumulative_global_samples >= target_global_samples
    )
    hit_step_ceiling = False
    dataset_rows = int(os.environ.get("FAILAWARE_TRAIN_DATASET_ROWS", "0"))
    dense_token_factor, attn_factor = qwen2_flop_coefficients(model.language_model.config)
    debug_timing = env_flag("FAILAWARE_DEBUG_TIMING", default=False)
    debug_all_ranks = env_flag("FAILAWARE_DEBUG_RANKS", default=False)
    log_batch_uids = env_flag("FAILAWARE_LOG_BATCH_UIDS", default=False)
    log_finite_gate = env_flag("FAILAWARE_LOG_FINITE_GATE", default=False)
    if dist.get_rank() == 0:
        sample_target_message = {
            "target_global_samples": target_global_samples,
            "sample_milestones": sample_milestones,
            "restored_cumulative_global_samples": cumulative_global_samples,
            "restored_crossed_milestones": sorted(crossed_sample_milestones),
            "resume_model_only": bool(resume_model_only),
            "resume_from": resume_from,
        }
        logger.info("[sample-target] " + json.dumps(sample_target_message, sort_keys=True))
        print("[sample-target] " + json.dumps(sample_target_message, sort_keys=True), flush=True)
    for micro_step, data in enumerate(train_loader):
        if target_reached:
            logger.info(
                f"Reached target_global_samples={target_global_samples}, stopping training."
            )
            break
        curr_step = train_step + micro_step // training_args.gradient_accumulation_steps
        if curr_step >= training_args.total_steps:
            logger.info(f"Reached total_steps={training_args.total_steps}, stopping training.")
            hit_step_ceiling = True
            break
        os.environ["FAILAWARE_CURRENT_STEP"] = str(curr_step)
        def debug_phase(message: str) -> None:
            rank = dist.get_rank()
            if debug_timing and (debug_all_ranks or rank == 0):
                phase_message = f"(step={curr_step:07d}, rank={rank:02d}) {message}"
                logger.info(phase_message)
                print(phase_message, flush=True)

        phase_t0 = time()
        data = data.cuda(device).to_dict()
        if debug_timing:
            torch.cuda.synchronize()
            h2d_time = time() - phase_t0
        else:
            h2d_time = 0.0
        data_indexes = data.pop('batch_data_indexes', None)
        ce_loss_weights = data.pop('ce_loss_weights', None)       
        if log_batch_uids:
            local_observed = [
                {
                    "uid": str(item.get("uid") or ""),
                    "source_split": str(item.get("source_split") or ""),
                    "task": str(item.get("task") or ""),
                }
                for item in (data_indexes or [])
                if item.get("uid")
            ]
            if dist.get_rank() == 0:
                gathered_observed = [None] * dist.get_world_size()
            else:
                gathered_observed = None
            dist.gather_object(local_observed, gathered_observed, dst=0)
            if dist.get_rank() == 0:
                observed = [
                    item
                    for rank_items in gathered_observed
                    for item in (rank_items or [])
                ]
                new_uids = [item["uid"] for item in observed if item["source_split"] == "new"]
                replay_uids = [item["uid"] for item in observed if item["source_split"] == "replay"]
                uid_message = (
                    f"[uid-gate] step={curr_step:07d} "
                    f"new_count={len(new_uids)} replay_count={len(replay_uids)} "
                    f"new_uids={','.join(new_uids)} "
                    f"replay_uids={','.join(replay_uids)}"
                )
                logger.info(uid_message)
                print(uid_message, flush=True)
        local_seq_len_tensor = torch.tensor(float(data['sequence_length']), device=device)
        tokens_tensor = local_seq_len_tensor.clone()
        dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
        max_rank_seq_len = local_seq_len_tensor.clone()
        min_rank_seq_len = local_seq_len_tensor.clone()
        dist.all_reduce(max_rank_seq_len, op=dist.ReduceOp.MAX)
        dist.all_reduce(min_rank_seq_len, op=dist.ReduceOp.MIN)
        local_samples_tensor = torch.tensor(float(len(data['sample_lens'])), device=device)
        global_samples_tensor = local_samples_tensor.clone()
        dist.all_reduce(global_samples_tensor, op=dist.ReduceOp.SUM)
        pending_global_samples += int(global_samples_tensor.item())
        max_rank_samples = local_samples_tensor.clone()
        min_rank_samples = local_samples_tensor.clone()
        dist.all_reduce(max_rank_samples, op=dist.ReduceOp.MAX)
        dist.all_reduce(min_rank_samples, op=dist.ReduceOp.MIN)
        local_max_sample_len = torch.tensor(float(max(data['sample_lens']) if data['sample_lens'] else 0), device=device)
        max_sample_len = local_max_sample_len.clone()
        dist.all_reduce(max_sample_len, op=dist.ReduceOp.MAX)
        token_window += tokens_tensor.item()
        # This all_reduce MUST be UNCONDITIONAL across ranks. If a rank's packed batch is empty
        # (sample_lens == []), skipping the collective desyncs ranks and deadlocks multi-node
        # (observed: 16-GPU hang at "Phase loss start" / first loss all_reduce). Use 0.0 when empty.
        if data['sample_lens']:
            sample_lens_tensor = torch.tensor(data['sample_lens'], dtype=torch.float32, device=device)
            sample_square = torch.dot(sample_lens_tensor, sample_lens_tensor)
        else:
            sample_square = torch.tensor(0.0, device=device)
        dist.all_reduce(sample_square, op=dist.ReduceOp.SUM)
        seqlen_square_window += sample_square.item()
        report_mem_metrics = not env_flag("FAILAWARE_SKIP_MEM_METRICS", default=False)
        if report_mem_metrics:
            local_mem_allocated = torch.tensor(torch.cuda.memory_allocated() / 1024**2, device=device)
            local_mem_reserved = torch.tensor(torch.cuda.memory_reserved() / 1024**2, device=device)
            max_mem_allocated_now = local_mem_allocated.clone()
            max_mem_reserved_now = local_mem_reserved.clone()
            dist.all_reduce(max_mem_allocated_now, op=dist.ReduceOp.MAX)
            dist.all_reduce(max_mem_reserved_now, op=dist.ReduceOp.MAX)
        else:
            local_mem_allocated = local_mem_reserved = None
            max_mem_allocated_now = max_mem_reserved_now = None
        if curr_step % training_args.log_every == 0:
            rank_prefix = f"(step={curr_step:07d}, rank={dist.get_rank():02d})" if debug_all_ranks else f"(step={curr_step:07d})"
            batch_message = (
                f"{rank_prefix} Begin Batch "
                f"Local SeqLen: {data['sequence_length']}, "
                f"Local Samples: {len(data['sample_lens'])}, "
                f"Local MaxSampleLen: {local_max_sample_len.item():.0f}, "
                f"Rank SeqLen Min/Max: {min_rank_seq_len.item():.0f}/{max_rank_seq_len.item():.0f}, "
                f"Rank Samples Min/Max: {min_rank_samples.item():.0f}/{max_rank_samples.item():.0f}, "
                f"Max SampleLen: {max_sample_len.item():.0f}, "
                f"Global Tokens: {tokens_tensor.item()/1000:.2f}k"
            )
            if report_mem_metrics:
                batch_message += (
                    f", Local Mem Alloc/Res: {local_mem_allocated.item():.0f}/{local_mem_reserved.item():.0f} MiB, "
                    f"Rank Mem Alloc/Res Max: {max_mem_allocated_now.item():.0f}/{max_mem_reserved_now.item():.0f} MiB"
                )
            logger.info(batch_message)
            if debug_all_ranks or dist.get_rank() == 0:
                print(batch_message, flush=True)

        latent_time = 0.0
        forward_time = 0.0
        loss_time = 0.0
        backward_time = 0.0
        opt_time = 0.0
        optimizer_step_completed = False
        newly_crossed = []
        newly_crossed_recovery = []
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            if training_args.visual_gen:
                debug_phase("Phase latent start")
                phase_t0 = time()
                with torch.no_grad():
                    if 'padded_latent' not in data:
                        if 'padded_vae_moments' in data:
                            moments = data.pop('padded_vae_moments')
                            mean, logvar = torch.chunk(moments, 2, dim=1)
                            latent = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
                            data['padded_latent'] = vae_config.scale_factor * (latent - vae_config.shift_factor)
                        else:
                            if vae_model is None:
                                raise RuntimeError(
                                    "Encoded failaware cache was requested, but the batch did not "
                                    "include padded_vae_moments. Disable FAILAWARE_CACHE_MODE=encoded "
                                    "or rebuild/verify the cache."
                                )
                            data['padded_latent'] = vae_model.encode(data.pop('padded_images'))
                if debug_timing:
                    torch.cuda.synchronize()
                    latent_time = time() - phase_t0
                    debug_phase(f"Phase latent done: {latent_time:.2f}s")
            try:
                debug_phase("Phase forward start")
                phase_t0 = time()
                loss_dict = fsdp_model(**data)
                if debug_timing:
                    torch.cuda.synchronize()
                    forward_time = time() - phase_t0
                    debug_phase(f"Phase forward done: {forward_time:.2f}s")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"CUDA OOM at step {curr_step}: {e}")
                    torch.cuda.empty_cache()
                raise e
        
        debug_phase("Phase loss start")
        phase_t0 = time()
        loss = torch.zeros((), device=device)
        ce = loss_dict["ce"]
        debug_phase("Loss ce-token all_reduce start")
        local_ce_tokens = torch.tensor(
            data['ce_loss_indexes'].numel() if 'ce_loss_indexes' in data else 0,
            dtype=torch.float32,
            device=device,
        )
        total_ce_tokens = local_ce_tokens.clone()
        dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
        debug_phase("Loss ce-token all_reduce done")
        if ce is not None:
            if training_args.ce_loss_reweighting:
                if ce_loss_weights is not None and ce_loss_weights.numel() > 0:
                    ce = ce * ce_loss_weights
                    local_ce_loss_weights = ce_loss_weights.sum()
                else:
                    local_ce_loss_weights = torch.tensor(0.0, device=device)
                total_ce_loss_weights = local_ce_loss_weights.clone()
                debug_phase("Loss ce-weight all_reduce start")
                dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
                debug_phase("Loss ce-weight all_reduce done")
                ce = ce.sum() * dist.get_world_size() / total_ce_loss_weights.clamp_min(1e-8)
            else:
                ce = ce.sum() * dist.get_world_size() / total_ce_tokens.clamp_min(1.0)
            loss_dict["ce"] = ce.detach()
            loss = loss + ce * training_args.ce_weight
        else:
            loss_dict["ce"] = torch.tensor(0, device=device)

        if training_args.visual_gen:
            mse = loss_dict["mse"]
            debug_phase("Loss mse-token all_reduce start")
            total_mse_tokens = torch.tensor(
                data['mse_loss_indexes'].numel() if 'mse_loss_indexes' in data else 0,
                dtype=torch.float32,
                device=device,
            )
            dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            debug_phase("Loss mse-token all_reduce done")
            if mse is not None:
                mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens.clamp_min(1.0)
                loss_dict["mse"] = mse.detach()
                loss = loss + mse * training_args.mse_weight
            else:
                loss_dict["mse"] = torch.tensor(0, device=device)
        else:
            assert not training_args.visual_gen
            loss_dict["mse"] = torch.tensor(0, device=device)
            total_mse_tokens = torch.tensor(0, device=device)

        unscaled_total_loss = loss
        finite_values = torch.stack(
            [
                loss_dict["ce"].detach().float(),
                loss_dict["mse"].detach().float(),
                unscaled_total_loss.detach().float(),
            ]
        )
        if not bool(torch.isfinite(finite_values).all().item()):
            raise FloatingPointError(
                f"non-finite loss at step={curr_step} rank={dist.get_rank()}"
            )
        if log_finite_gate:
            finite_gate_record = {
                "step": int(curr_step),
                "rank": int(dist.get_rank()),
                "raw_ce": float(loss_dict["ce"].detach().item()),
                "weighted_ce_contribution": float(
                    loss_dict["ce"].detach().item() * training_args.ce_weight
                ),
                "raw_mse": float(loss_dict["mse"].detach().item()),
                "weighted_mse_contribution": float(
                    loss_dict["mse"].detach().item() * training_args.mse_weight
                ),
                "total_loss": float(unscaled_total_loss.detach().item()),
                "ce_tokens": int(local_ce_tokens.item()),
                "mse_tokens": int(
                    data['mse_loss_indexes'].numel()
                    if 'mse_loss_indexes' in data
                    else 0
                ),
                "samples": int(len(data["sample_lens"])),
                "sequence_length": int(data["sequence_length"]),
            }
            gate_message = (
                "[finite-loss-gate] "
                + json.dumps(finite_gate_record, sort_keys=True)
            )
            logger.info(gate_message)
            print(gate_message, flush=True)

        loss = loss / training_args.gradient_accumulation_steps
        if debug_timing:
            torch.cuda.synchronize()
            loss_time = time() - phase_t0
            debug_phase(f"Phase loss done: {loss_time:.2f}s")
        debug_phase("Phase backward start")
        phase_t0 = time()
        loss.backward()
        if debug_timing:
            torch.cuda.synchronize()
            backward_time = time() - phase_t0
            debug_phase(f"Phase backward done: {backward_time:.2f}s")

        if (micro_step + 1) % training_args.gradient_accumulation_steps == 0:
            debug_phase("Phase optim start")
            phase_t0 = time()
            total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            if ema_model is not None:
                fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)
            optimizer.zero_grad()
            previous_cumulative_samples = cumulative_global_samples
            last_global_samples = pending_global_samples
            cumulative_global_samples += pending_global_samples
            pending_global_samples = 0
            newly_crossed = newly_crossed_sample_milestones(
                previous_cumulative_samples,
                cumulative_global_samples,
                sample_milestones,
                crossed_sample_milestones,
            )
            crossed_sample_milestones.update(newly_crossed)
            if recovery_save_every_samples > 0:
                first_recovery_index = (
                    previous_cumulative_samples
                    // recovery_save_every_samples
                ) + 1
                last_recovery_index = (
                    cumulative_global_samples
                    // recovery_save_every_samples
                )
                newly_crossed_recovery = [
                    index * recovery_save_every_samples
                    for index in range(
                        first_recovery_index,
                        last_recovery_index + 1,
                    )
                    if index * recovery_save_every_samples
                    not in crossed_recovery_samples
                ]
                crossed_recovery_samples.update(
                    newly_crossed_recovery
                )
            target_reached = (
                target_global_samples > 0
                and cumulative_global_samples >= target_global_samples
            )
            last_completed_step = curr_step
            optimizer_step_completed = True
            if debug_timing:
                torch.cuda.synchronize()
                opt_time = time() - phase_t0
                debug_phase(f"Phase optim done: {opt_time:.2f}s")
        
        # Log loss values:
        if curr_step % training_args.log_every == 0:
            total_samples = global_samples_tensor

            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            elapsed = max(end_time - start_time, 1e-6)
            steps_per_sec = training_args.log_every / elapsed
            tokens_per_sec = token_window / elapsed
            tokens_per_step = token_window / training_args.log_every
            flops_all_token = dense_token_factor * token_window + attn_factor * seqlen_square_window
            actual_tflops = flops_all_token / elapsed / 1e12
            peak_total_tflops = training_args.peak_device_tflops * dist.get_world_size()
            mfu_value = actual_tflops / peak_total_tflops if peak_total_tflops > 0 else 0.0
            message = f"(step={curr_step:07d}) "
            wandb_log = {}
            for key, value in loss_dict.items():
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(value.item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                wandb_log[key] = avg_loss
            message += (
                f"SeqLen: {data['sequence_length']}, Samples: {len(data['sample_lens'])}, "
                f"Global Samples: {int(total_samples.item())}, "
                f"Cumulative Samples: {cumulative_global_samples}, "
                f"Total CE Tokens: {int(total_ce_tokens.item())}, "
                f"Total MSE Tokens: {int(total_mse_tokens.item())}, "
                f"Local MaxSampleLen: {local_max_sample_len.item():.0f}, "
                f"Rank SeqLen Min/Max: {min_rank_seq_len.item():.0f}/{max_rank_seq_len.item():.0f}, "
                f"Rank Samples Min/Max: {min_rank_samples.item():.0f}/{max_rank_samples.item():.0f}, "
                f"Max SampleLen: {max_sample_len.item():.0f}, "
                f"Tokens/Step: {tokens_per_step/1000:.2f}k, "
                f"Train Steps/Sec: {steps_per_sec:.2f}, Tokens/Sec: {tokens_per_sec/1000:.2f}k, "
                f"MFU: {mfu_value*100:.1f}%, "
            )
            avg_total_loss = loss.detach().clone()
            dist.all_reduce(avg_total_loss, op=dist.ReduceOp.SUM)
            avg_total_loss = avg_total_loss.item() / dist.get_world_size()
            message += f"Train Loss total: {avg_total_loss:.4f}, "
            wandb_log["total_loss"] = avg_total_loss
            if dataset_rows > 0:
                message += f"Effective Epochs: {cumulative_global_samples / dataset_rows:.4f}, "
            if debug_timing:
                message += (
                    f"Timing H2D: {h2d_time:.2f}s, Latent: {latent_time:.2f}s, "
                    f"Forward: {forward_time:.2f}s, Loss: {loss_time:.2f}s, "
                    f"Backward: {backward_time:.2f}s, Optim: {opt_time:.2f}s, "
                )

            wandb_log['lr'] = optimizer.param_groups[0]['lr']
            wandb_log['total_mse_tokens'] = total_mse_tokens.item()
            wandb_log['total_ce_tokens'] = total_ce_tokens.item()
            wandb_log['total_norm'] = total_norm.item()
            wandb_log['total_samples'] = total_samples.item()
            wandb_log['cumulative_samples'] = cumulative_global_samples
            if dataset_rows > 0:
                wandb_log['effective_epochs'] = cumulative_global_samples / dataset_rows
            wandb_log['tokens_per_sec'] = tokens_per_sec
            wandb_log['tokens_per_step'] = tokens_per_step
            wandb_log['actual_tflops'] = actual_tflops
            wandb_log['mfu'] = mfu_value

            if report_mem_metrics:
                debug_phase("Log mem_allocated all_reduce start")
                mem_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
                dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
                debug_phase("Log mem_allocated all_reduce done")
                wandb_log['mem_allocated'] = mem_allocated
                debug_phase("Log mem_cache all_reduce start")
                mem_cache = torch.tensor(torch.cuda.max_memory_reserved() / 1024**2, device=device)
                dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
                debug_phase("Log mem_cache all_reduce done")
                wandb_log['mem_cache'] = mem_cache
                message += (
                    f"Peak Mem Alloc/Res: {mem_allocated.item():.0f}/{mem_cache.item():.0f} MiB, "
                )

            logger.info(message)
            if dist.get_rank() == 0:
                print(message, flush=True)

            if dist.get_rank() == 0:
                debug_phase("Log wandb start")
                wandb.log(wandb_log, step=curr_step)
                debug_phase("Log wandb done")
            start_time = time()
            token_window = 0.0
            seqlen_square_window = 0.0

        debug_phase("Data status update start")
        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item['dataset_name'] not in data_status.keys():
                data_status[item['dataset_name']] = {}
            data_status[item['dataset_name']][item['worker_id']] = item['data_indexes']
        debug_phase("Data status update done")

        should_save_checkpoint = optimizer_step_completed and (
            (curr_step > 0 and curr_step % training_args.save_every == 0)
            or bool(newly_crossed)
            or bool(newly_crossed_recovery)
            or target_reached
        )
        if should_save_checkpoint:
            # Clear caches and ensure all CUDA operations complete before checkpoint
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if dist.get_rank() == 0:
                gather_list = [None] * dist.get_world_size()
            else:
                gather_list = None
            try:
                dist.gather_object(data_status, gather_list, dst=0)
            except RuntimeError as e:
                logger.error(f"Error during gather_object at step {curr_step}: {e}")
                gather_list = None if dist.get_rank() != 0 else [data_status] * dist.get_world_size()

            checkpoint_sample_state = {
                "version": 1,
                "cumulative_global_samples": int(cumulative_global_samples),
                "global_samples_last_step": int(last_global_samples),
                "last_completed_step": int(last_completed_step),
                "target_global_samples": int(target_global_samples),
                "target_reached": bool(target_reached),
                "target_overshoot": int(
                    max(0, cumulative_global_samples - target_global_samples)
                    if target_global_samples > 0
                    else 0
                ),
                "sample_milestones": sample_milestones,
                "crossed_milestones": sorted(crossed_sample_milestones),
                "newly_crossed_milestones": newly_crossed,
                "recovery_save_every_samples": int(
                    recovery_save_every_samples
                ),
                "crossed_recovery_samples": sorted(
                    crossed_recovery_samples
                ),
                "newly_crossed_recovery_samples": (
                    newly_crossed_recovery
                ),
                "world_size": int(dist.get_world_size()),
            }
            FSDPCheckpoint.fsdp_save_ckpt(
                ckpt_dir=training_args.checkpoint_dir, 
                train_steps=curr_step, 
                model=fsdp_model, 
                ema_model=ema_model, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                logger=logger,
                fsdp_config=fsdp_config,
                data_status=gather_list,
                sample_state=checkpoint_sample_state,
                write_completion_marker=(
                    training_args.write_checkpoint_completion_marker
                ),
            )
            last_saved_step = curr_step
            checkpoint_message = {
                "step": int(curr_step),
                "cumulative_global_samples": int(cumulative_global_samples),
                "newly_crossed_milestones": newly_crossed,
                "newly_crossed_recovery_samples": (
                    newly_crossed_recovery
                ),
                "target_reached": bool(target_reached),
            }
            logger.info(
                "[sample-checkpoint] "
                + json.dumps(checkpoint_message, sort_keys=True)
            )
            if dist.get_rank() == 0:
                print(
                    "[sample-checkpoint] "
                    + json.dumps(checkpoint_message, sort_keys=True),
                    flush=True,
                )
            # Clear CUDA cache and force garbage collection after checkpoint to free memory
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # comment out as an alternative to save the ema model in pt format
            # ema_state_dict = {}
            # for name, param in ema_model.named_parameters():
            #     ema_state_dict[name] = param.detach().cpu()
            
            # torch.save(
            #     ema_state_dict, 
            #     os.path.join(training_args.checkpoint_dir, f"{curr_step:07d}", "ema_standard.pt")
            # )
        if target_reached:
            break
    
    if target_global_samples > 0 and hit_step_ceiling and not target_reached:
        failure_message = (
            f"fixed step ceiling {training_args.total_steps} reached at "
            f"{cumulative_global_samples} samples before target "
            f"{target_global_samples}"
        )
        logger.error(failure_message)
        if dist.get_rank() == 0:
            wandb.finish()
        dist.destroy_process_group()
        raise RuntimeError(failure_message)

    # Save final checkpoint if not already saved
    skip_final_ckpt = os.environ.get("FAILAWARE_SKIP_FINAL_CKPT", "0").strip().lower() in {"1", "true", "yes", "on"}
    final_save_step = (
        last_completed_step
        if target_global_samples > 0
        else curr_step
    )
    if (
        final_save_step > 0
        and final_save_step != last_saved_step
        and not skip_final_ckpt
    ):
        logger.info(f"Saving final checkpoint at step {final_save_step}...")
        # Clear caches and ensure all CUDA operations complete before final checkpoint
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if dist.get_rank() == 0:
            gather_list = [None] * dist.get_world_size()
        else:
            gather_list = None
        try:
            dist.gather_object(data_status, gather_list, dst=0)
        except RuntimeError as e:
            logger.error(f"Error during final gather_object: {e}")
            gather_list = None if dist.get_rank() != 0 else [data_status] * dist.get_world_size()

        checkpoint_sample_state = {
            "version": 1,
            "cumulative_global_samples": int(cumulative_global_samples),
            "global_samples_last_step": int(last_global_samples),
            "last_completed_step": int(last_completed_step),
            "target_global_samples": int(target_global_samples),
            "target_reached": bool(target_reached),
            "target_overshoot": int(
                max(0, cumulative_global_samples - target_global_samples)
                if target_global_samples > 0
                else 0
            ),
            "sample_milestones": sample_milestones,
            "crossed_milestones": sorted(crossed_sample_milestones),
            "newly_crossed_milestones": [],
            "recovery_save_every_samples": int(
                recovery_save_every_samples
            ),
            "crossed_recovery_samples": sorted(
                crossed_recovery_samples
            ),
            "newly_crossed_recovery_samples": [],
            "world_size": int(dist.get_world_size()),
        }
        FSDPCheckpoint.fsdp_save_ckpt(
            ckpt_dir=training_args.checkpoint_dir, 
            train_steps=final_save_step, 
            model=fsdp_model, 
            ema_model=ema_model, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            logger=logger,
            fsdp_config=fsdp_config,
            data_status=gather_list,
            sample_state=checkpoint_sample_state,
            write_completion_marker=(
                training_args.write_checkpoint_completion_marker
            ),
        )
        # Clear CUDA cache and force garbage collection after final checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(f"Final checkpoint saved at step {final_save_step}")
    elif final_save_step > 0 and skip_final_ckpt:
        logger.info(
            f"Skipping final checkpoint at step {final_save_step} "
            "because FAILAWARE_SKIP_FINAL_CKPT=1"
        )
    
    logger.info("Done!")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
