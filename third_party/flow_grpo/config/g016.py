"""Multi-round reflection GRPO config for BAGEL (six-family GenEval, whole-trajectory reward).

Launch with ``--config config/g016.py:multiround_counting``. Every path comes
from the environment; see ``scripts/rl/train_rl.sh`` for the full set.
"""
from __future__ import annotations

import os

import ml_collections

from config.base import get_config as get_base_config
from unify_rl.train.g022_campaign import (
    validate_launch_bindings as validate_g022_launch_bindings,
    G022_CE_WEIGHT,
    G022_DIFFICULTY_RESAMPLE_MAX_ATTEMPTS,
    G022_KL_BETA_FLOW,
    G022_KL_BETA_TEXT,
    G022_LEARNING_RATE,
    G022_MAX_EXACT_ROOTS_PER_STEP,
    G022_MSE_WEIGHT,
    G022_NOISE_LEVEL_EDIT,
    G022_NOISE_LEVEL_GEN,
    G022_SIBLINGS_PER_ROOT,
    G022_SUSTAINED_PROGRESS_BETA,
    G022_TARGET_EXACT_ROOT_SHARE,
    G022_TEXT_BOUND_MODE,
    G022_TEXT_TEMPERATURE,
)
from unify_rl.train.f01_campaign import F01_REWARD_STAGE, F01_TEXT_HEAD_MODE

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def multiround_counting():
    config = get_base_config()
    config.run_name = os.environ.get("G016_RUN_NAME", "f01-geneval-full-formal1000")
    config.debug = False
    config.num_epochs = 1
    config.logdir = os.environ["G016_OUTPUT_DIR"]
    config.num_checkpoint_limit = 17
    config.mixed_precision = (
        "no" if os.environ.get("G016_BF16_POLICY_CPU_FP32_MASTER", "1") == "1"
        else "bf16"
    )
    config.allow_tf32 = True
    config.use_lora = False
    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = False

    config.pretrained.model = os.environ["G016_FLOWGRPO_MODEL_DIR"]
    config.dataset = os.environ["G016_TRAIN_DATA"]
    config.resolution = 512

    config.sample.num_steps = int(os.environ.get("G016_NUM_STEPS", "50"))
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 3.0
    config.sample.eval_guidance_scale = 3.0
    config.sample.train_batch_size = int(
        os.environ.get("G016_LOCAL_BATCH_SIZE", "2")
    )
    config.sample.num_image_per_prompt = int(
        os.environ.get("G016_GROUP_SIZE", "28")
    )
    config.sample.test_batch_size = 1
    config.sample.num_batches_per_epoch = 1
    config.sample.global_std = False
    config.sample.noise_level = 0.8
    config.sample.same_latent = False
    config.sample.sde_window_size = 2
    config.sample.sde_window_range = (
        0,
        config.sample.num_steps // 2,
    )

    config.train.batch_size = int(
        os.environ.get("G016_TRAIN_MICRO_BATCH_SIZE", str(config.sample.train_batch_size))
    )
    config.train.gradient_accumulation_steps = int(
        os.environ.get("G016_GRADIENT_ACCUMULATION_STEPS", "1")
    )
    config.train.num_inner_epochs = 1
    config.train.learning_rate = float(
        os.environ.get("G016_FLOW_LEARNING_RATE", "1e-5")
    )
    config.train.clip_range = 0.1
    config.train.clip_range_lt = 0.1
    config.train.clip_range_gt = 0.1
    config.train.beta = 0.01
    config.train.max_grad_norm = 1.0
    config.train.ema = False
    config.train.timestep_shift = 3.0

    config.g016 = g016 = ml_collections.ConfigDict()
    g016.total_steps = int(os.environ.get("G016_TOTAL_STEPS", "1000"))
    g016.start_step = int(os.environ.get("G016_START_STEP", "0"))
    g016.resume_checkpoint = os.environ.get("G016_RESUME_CHECKPOINT", "").strip()
    checkpoint_steps = os.environ.get(
        "G016_CHECKPOINT_STEPS",
        "50,100,200,300,400,500,600,700,800,900,1000",
    )
    g016.checkpoint_steps = [
        int(value) for value in checkpoint_steps.split(",") if value
    ]
    g016.stage_checkpoint_step = int(
        os.environ.get("G016_STAGE_CHECKPOINT_STEP", "-1")
    )
    g016.seed = int(os.environ.get("G016_SEED", "20260830"))
    # SHA-256 of the SFT checkpoint the policy is initialised from.
    g016.initialization_sha256 = os.environ.get(
        "G016_INITIALIZATION_SHA256",
        "52e1e11193f26471dc253d0b65089a2aa024a68ce218cd916ea911b38eddc370",
    )
    g016.lora_rank = int(os.environ.get("G016_LORA_RANK", "0"))
    if g016.lora_rank != 0:
        raise ValueError("full-rank training only; LoRA is not supported")
    g016.trainable_layer_start = 0
    g016.trainable_layer_end = 27
    g016.flow_activation_recompute = False
    g016.bf16_policy_cpu_fp32_master = (
        os.environ.get("G016_BF16_POLICY_CPU_FP32_MASTER", "1") == "1"
    )
    g016.fsdp_strategy = os.environ.get("G016_FSDP_STRATEGY", "HYBRID_SHARD")
    if g016.fsdp_strategy not in {"HYBRID_SHARD", "FULL_SHARD"}:
        raise ValueError("FSDP strategy must be HYBRID_SHARD or FULL_SHARD")
    g016.nested_fsdp_wrap = os.environ.get("G016_NESTED_FSDP_WRAP", "1") == "1"
    g016.optimizer_state_cpu_offload = (
        os.environ.get("G016_OPTIMIZER_STATE_CPU_OFFLOAD", "0") == "1"
    )
    g016.reference_cpu_streaming = (
        os.environ.get("G016_REFERENCE_CPU_STREAMING", "0") == "1"
    )
    g016.diagnostic_active_channel_mode = os.environ.get(
        "G016_DIAGNOSTIC_ACTIVE_CHANNEL_MODE", "both"
    ).strip()
    g016.diagnostic_optimizer_step_mode = os.environ.get(
        "G016_DIAGNOSTIC_OPTIMIZER_STEP_MODE", "both"
    ).strip()
    release_prompt_indices = os.environ.get("G016_RELEASE_PROMPT_INDICES", "").strip()
    g016.release_prompt_indices = (
        [int(value) for value in release_prompt_indices.split(",")]
        if release_prompt_indices
        else []
    )
    g016.release_resample_stride = int(
        os.environ.get("G016_RELEASE_RESAMPLE_STRIDE", "6")
    )
    # GenEval detector reward service (scripts/rl/serve_reward.sh).
    g016.reward_url = os.environ.get("G016_REWARD_URL", "http://127.0.0.1:18092")
    g016.reward_timeout_sec = float(os.environ.get("G016_REWARD_TIMEOUT_SEC", "600"))
    g016.text_learning_rate = float(os.environ.get("G016_TEXT_LEARNING_RATE", "1e-5"))
    g016.repair_progress_deadzone = float(
        os.environ.get("G016_REPAIR_PROGRESS_DEADZONE", "0.0")
    )
    if not 0.0 <= g016.repair_progress_deadzone < 0.5:
        raise ValueError("repair progress deadzone must be in [0, 0.5)")
    g016.text_clip_range = 0.2
    g016.text_kl_beta = 0.01
    g016.text_bound_mode = "clamp20"
    g016.text_head_mode = F01_TEXT_HEAD_MODE
    g016.ce_weight = None
    g016.mse_weight = None
    g016.kl_max_abs_delta_stop = None
    g016.noise_level_gen = None
    g016.noise_level_edit = None
    g016.eta_clamp_mode = "exact_t_equals_one_sigma_max"
    g016.g022_max_exact_roots_per_step = -1
    g016.g022_target_exact_root_share = -1.0
    g016.g022_difficulty_resample_max_attempts = 0
    g016.controller_temperature = 0.5
    g016.sustained_progress_beta = 0.0
    g016.max_repair_rounds = int(os.environ.get("G016_MAX_REPAIR_ROUNDS", "3"))
    g016.controller_max_tokens = int(os.environ.get("G016_CONTROLLER_MAX_TOKENS", "512"))
    g016.controller_distributed_lockstep = (
        os.environ.get("G016_CONTROLLER_DISTRIBUTED_LOCKSTEP", "0") == "1"
    )
    g016.system_prompt_file = os.environ.get(
        "G016_SYSTEM_PROMPT_FILE",
        os.path.join(REPO_ROOT, "assets/prompts/controller_system_prompt.txt"),
    )
    g016.ladder_stage = os.environ.get("G016_LADDER_STAGE", "formal")
    g016.wandb_enabled = (
        os.environ.get("G016_WANDB_ENABLED", "0").strip() not in {"0", "false"}
    )
    g016.metrics_jsonl = os.path.join(config.logdir, "g016_metrics.jsonl")
    g016.dashboard_jsonl = os.path.join(
        config.logdir, "g016_self_improvement_dashboard.jsonl"
    )
    g016.done_guard_prefix = os.environ.get("G016_DONE_GUARD_PREFIX", "").strip()
    g016.live_status_path = os.environ.get("G016_LIVE_STATUS_PATH", "").strip() or (
        os.path.join(config.logdir, "live_status.json")
    )
    g016.stop_now_baseline_advantage = True
    g016.stop_now_scorer_identity = "counting_strict_exact_terminal_value"
    g016.stop_now_scorer_version = "clean29529_g011_counting_stop_now_value_v1"
    g016.reward_stage = os.environ.get("G016_REWARD_STAGE", F01_REWARD_STAGE).strip()
    if g016.reward_stage != F01_REWARD_STAGE:
        raise ValueError(f"unknown reward stage: {g016.reward_stage!r}")

    # ------------------------------------------------------------------
    # Whole-trajectory GRPO: one scalar advantage per trajectory, broadcast
    # to every policy-sampled text token and every trained flow transition.
    # ------------------------------------------------------------------
    from flow_grpo.bagel.modeling.bagel.bagel import (
        G022_ETA_CLAMP_MODE,
        G022_EVAL_NUM_TIMESTEPS,
        G022_FLOW_CONTRACT,
        G022_SDE_WINDOW_RANGE,
        G022_SDE_WINDOW_SIZE,
        G022_TRAIN_NUM_TIMESTEPS,
    )
    from flow_grpo.g008_counting import G022_BOUND_MODE
    from unify_rl.reward_models.g022_whole_trajectory_reward import (
        CONTROLLER_CREDIT_VERSION as G022_CONTROLLER_CREDIT_VERSION,
        REPAIR_BUCKET_VERSION as G022_RENDERER_BUCKET_VERSION,
        REPAIR_FLOW_CREDIT_VERSION as G022_RENDERER_CREDIT_VERSION,
        VERSION as G022_ADVANTAGE_VERSION,
    )
    from unify_rl.train.g022_bounded_kl import (
        MAX_ABS_DELTA_STOP as G022_KL_MAX_ABS_DELTA_STOP,
    )

    # The trainer cross-checks these against the reward module's own versions.
    g016.channel_credit_version = G022_ADVANTAGE_VERSION
    g016.controller_credit_version = G022_CONTROLLER_CREDIT_VERSION
    g016.renderer_credit_version = G022_RENDERER_CREDIT_VERSION
    g016.renderer_bucket_version = G022_RENDERER_BUCKET_VERSION
    g016.exact_singleton_action_active = False
    g016.repair_progress_deadzone = 0.0

    # KL: bounded (linearised past a threshold) on both channels.
    if G022_BOUND_MODE != G022_TEXT_BOUND_MODE:
        raise ValueError("text KL bound mode differs between loss and campaign")
    g016.text_kl_beta = float(G022_KL_BETA_TEXT)
    config.train.beta = float(G022_KL_BETA_FLOW)
    g016.text_bound_mode = G022_TEXT_BOUND_MODE
    g016.text_head_mode = F01_TEXT_HEAD_MODE
    g016.kl_max_abs_delta_stop = float(G022_KL_MAX_ABS_DELTA_STOP)

    g016.ce_weight = float(G022_CE_WEIGHT)
    g016.mse_weight = float(G022_MSE_WEIGHT)
    config.train.learning_rate = float(G022_LEARNING_RATE)
    g016.text_learning_rate = float(G022_LEARNING_RATE)

    # Official BAGEL CFG at rollout and evaluation.
    config.sample.guidance_scale = 4.0
    config.sample.eval_guidance_scale = 4.0

    # SDE noise: separate eta for the initial generation and for edits, with
    # the sigma denominator clamped for t >= 0.95.
    g016.noise_level_gen = float(G022_NOISE_LEVEL_GEN)
    g016.noise_level_edit = float(G022_NOISE_LEVEL_EDIT)
    g016.eta_clamp_mode = G022_ETA_CLAMP_MODE
    config.sample.noise_level = float(G022_NOISE_LEVEL_EDIT)

    # Down-sample roots whose first image is already correct (zero-advantage
    # groups are wasted compute).
    g016.g022_max_exact_roots_per_step = int(G022_MAX_EXACT_ROOTS_PER_STEP)
    g016.g022_target_exact_root_share = float(G022_TARGET_EXACT_ROOT_SHARE)
    g016.g022_difficulty_resample_max_attempts = int(
        G022_DIFFICULTY_RESAMPLE_MAX_ATTEMPTS
    )

    # Flow window: 20 training denoising steps, two contiguous SDE steps
    # sampled from the first half; 50 steps at evaluation.
    config.sample.num_steps = G022_TRAIN_NUM_TIMESTEPS
    config.sample.eval_num_steps = G022_EVAL_NUM_TIMESTEPS
    config.sample.sde_window_size = G022_SDE_WINDOW_SIZE
    config.sample.sde_window_range = G022_SDE_WINDOW_RANGE
    config.sample.g021_stratified_sde = False
    config.sample.g021_flow_contract = G022_FLOW_CONTRACT
    config.sample.num_image_per_prompt = G022_SIBLINGS_PER_ROOT
    g016.controller_temperature = G022_TEXT_TEMPERATURE
    g016.sustained_progress_beta = G022_SUSTAINED_PROGRESS_BETA
    if int(os.environ.get("G016_NUM_STEPS", str(G022_TRAIN_NUM_TIMESTEPS))) != (
        G022_TRAIN_NUM_TIMESTEPS
    ):
        raise ValueError("training uses 20 denoising steps")
    if int(os.environ.get("G016_GROUP_SIZE", str(G022_SIBLINGS_PER_ROOT))) != (
        G022_SIBLINGS_PER_ROOT
    ):
        raise ValueError("training uses 16 siblings per root")

    validate_g022_launch_bindings(config)
    g016.renderer_cross_step_pooling = False
    g016.renderer_bucket_fallback_ladder = [
        "exact_q_round_target",
        "exact_q_round",
        "exact_round",
        "disable",
    ]
    g016.prompt_narrowing_manifest = os.environ["G016_PROMPT_NARROWING_MANIFEST"]
    g016.sde_window_seed_mode = "clean29529_g012_local_sde_window_seed_v1"
    g016.prompt_narrowing_manifest_sha256 = os.environ[
        "G016_PROMPT_NARROWING_MANIFEST_SHA256"
    ]
    g016.exact_before_action_policy_visible = False
    g016.singleton_bucket_policy_active = False
    g016.mixed_exactness_fallback = False
    config.reward_fn = {"deterministic_counting_only": 1.0}
    config.per_prompt_stat_tracking = True
    config.save_freq = 1000
    config.eval_freq = 1000
    config.save_dir = os.path.join(config.logdir, "checkpoints")
    return config


def get_config(name):
    return globals()[name]()
