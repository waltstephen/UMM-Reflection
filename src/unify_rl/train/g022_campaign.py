"""Whole-trajectory GRPO hyperparameters shared by the reward, config and trainer.

One scalar advantage per trajectory is broadcast to every policy-sampled text
token and every trained flow transition. The reward is

    R(tau) = alpha * q(R0)
             + sum_t w_t * max(0, q_t - q_{t-1})          (reflection bonus)
             - lambda * sum_t max(0, q_{t-1} - q_t)       (regression penalty)
             + beta * (sum of per-round gains except the largest)
             - premature_done_penalty * [DONE on a wrong image]

and is standardized inside each K-sibling group that shares one root prompt.
"""
from __future__ import annotations

from typing import Any, Mapping

G022_REWARD_STAGE = "g022_whole_trajectory_grpo_v1"

# ---------------------------------------------------------------------------
# Group construction: 2 root prompts x 16 siblings = 32 trajectories per step.
# ---------------------------------------------------------------------------
G022_ROOT_GROUP_COUNT = 2
G022_SIBLINGS_PER_ROOT = 16
G022_TRAJECTORY_COUNT = G022_ROOT_GROUP_COUNT * G022_SIBLINGS_PER_ROOT
G022_MAX_REPAIR_ROUNDS = 3

# SHA-256 of the SFT checkpoint (`ckpts/0002969/model.safetensors`) RL starts from.
G022_CLASSIC_PARENT_SHA256 = (
    "52e1e11193f26471dc253d0b65089a2aa024a68ce218cd916ea911b38eddc370"
)

# ---------------------------------------------------------------------------
# Reward.
# ---------------------------------------------------------------------------
G022_ALPHA = 0.3
# Flat weights: a later round is worth the same as an earlier one, so there is
# no incentive to stall a fix until a higher-weighted round.
G022_REFLECTION_WEIGHTS = (1.0, 1.0, 1.0)
G022_LAMBDA = 0.5
G022_PREMATURE_DONE_PENALTY = 0.5

# Sustained progress: beta times the sum of positive per-round improvements
# except the largest one, so a single big fix earns nothing extra and only
# repeated real progress is rewarded. With beta <= 0.3 a second EDIT is worth
# taking only when its success probability exceeds the break-even below.
G022_SUSTAINED_PROGRESS_BETA = 0.3
G022_SUSTAINED_PROGRESS_FORM = "multi_improvement_except_largest"
G022_SUSTAINED_PROGRESS_BREAK_EVEN_SECOND_EDIT_SUCCESS = 0.735
# Measured on archived rollouts before training; recorded for reference only.
G022_SUSTAINED_PROGRESS_ARCHIVED_FIRE_RATE = 0.0212
G022_SUSTAINED_PROGRESS_RANKING_CHANGE_RATE = 4 / 400

# A malformed controller action is penalized after group normalization, so the
# sign of its advantage is guaranteed negative regardless of its siblings.
G022_MALFORMED_ACTION_PENALTY = 0.5
G022_MALFORMED_ADVANTAGE_PENALTY = 0.5
G022_STD_FLOOR = 0.1
G022_ADVANTAGE_CLIP = 1.0

# Roots whose first image is already correct give near-zero advantage. Keep at
# most one per step, targeting a 15% share, with up to two resample attempts.
G022_MAX_EXACT_ROOTS_PER_STEP = 1
G022_TARGET_EXACT_ROOT_SHARE = 0.15
G022_DIFFICULTY_RESAMPLE_MAX_ATTEMPTS = 2

# ---------------------------------------------------------------------------
# Optimization.
# ---------------------------------------------------------------------------
G022_LEARNING_RATE = 5e-6
G022_CE_WEIGHT = 1.0
G022_MSE_WEIGHT = 2.0
G022_KL_BETA_TEXT = 1e-4
G022_KL_BETA_FLOW = 1e-4
G022_TEXT_BOUND_MODE = "g022_linear"

# ---------------------------------------------------------------------------
# Sampling.
# ---------------------------------------------------------------------------
G022_TEXT_TEMPERATURE = 0.9
G022_NOISE_LEVEL_GEN = 0.7
G022_NOISE_LEVEL_EDIT = 1.0
G022_ETA_CLAMP_MODE_NAME = "r3_t_ge_0p95"
G022_FLOW_CONTRACT = "g022_contiguous_uniform_random_flow2_v1"
# 20 denoising steps during rollout (50 at evaluation); the SDE is applied on
# two contiguous steps drawn uniformly from the first ten.
G022_TRAIN_NUM_TIMESTEPS = 20
G022_EVAL_NUM_TIMESTEPS = 50
G022_SDE_WINDOW_SIZE = 2
G022_SDE_WINDOW_RANGE = (0, 10)


def _expected_flow_learning_rate(g016: Any) -> float:
    del g016
    return G022_LEARNING_RATE


def _expected_text_learning_rate(g016: Any) -> float:
    del g016
    return G022_LEARNING_RATE


def validate_launch_bindings(config: Any) -> Mapping[str, Any]:
    """Check that a resolved config matches the constants above.

    Stages built on top of this reward (e.g. the six-family GenEval stage)
    validate their own bindings and pass through here unchanged.
    """
    g016 = config.g016
    stage = str(g016.reward_stage)
    if stage != G022_REWARD_STAGE:
        return {"validated": False, "reason": "not a G022 run", "reward_stage": stage}
    expected = {
        "sample.num_steps": (config.sample.num_steps, G022_TRAIN_NUM_TIMESTEPS),
        "sample.eval_num_steps": (config.sample.eval_num_steps, G022_EVAL_NUM_TIMESTEPS),
        "sample.num_image_per_prompt": (
            config.sample.num_image_per_prompt, G022_SIBLINGS_PER_ROOT
        ),
        "sample.sde_window_size": (config.sample.sde_window_size, G022_SDE_WINDOW_SIZE),
        "sample.sde_window_range": (
            tuple(config.sample.sde_window_range), G022_SDE_WINDOW_RANGE
        ),
        "g016.text_bound_mode": (g016.text_bound_mode, G022_TEXT_BOUND_MODE),
        "g016.eta_clamp_mode": (g016.eta_clamp_mode, G022_ETA_CLAMP_MODE_NAME),
        "g016.sustained_progress_beta": (
            float(g016.sustained_progress_beta), G022_SUSTAINED_PROGRESS_BETA
        ),
        "g016.controller_temperature": (
            float(g016.controller_temperature), G022_TEXT_TEMPERATURE
        ),
    }
    mismatched = {
        key: {"observed": observed, "expected": want}
        for key, (observed, want) in expected.items()
        if observed != want
    }
    if mismatched:
        raise ValueError(f"G022 launch bindings differ: {mismatched}")
    return {"validated": True, "reward_stage": stage}
