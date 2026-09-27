"""Evaluation-time sampling settings.

``config/g016.py:multiround_counting`` returns the *training* rollout config
(20 denoising steps, CFG 4.0, separate SDE noise for generation and edits,
temperature 0.9). Reported numbers use the evaluation protocol below instead:
50 denoising steps, CFG 3.0, a single SDE noise level, controller temperature
0.5 and at most three repair rounds, identical for every arm.
"""
from __future__ import annotations

from typing import Any

EVAL_NUM_STEPS = 50
EVAL_GUIDANCE_SCALE = 3.0
EVAL_NOISE_LEVEL = 0.8
EVAL_CONTROLLER_TEMPERATURE = 0.5
EVAL_CONTROLLER_MAX_TOKENS = 512
EVAL_MAX_REPAIR_ROUNDS = 3
EVAL_BATCH_SIZE = 2


def _drop(section: Any, key: str) -> None:
    if key in section:
        del section[key]


def apply_eval_overrides(config: Any) -> Any:
    sample, g016 = config.sample, config.g016
    config.resolution = 512
    config.train.timestep_shift = 3.0

    sample.num_steps = EVAL_NUM_STEPS
    sample.eval_num_steps = EVAL_NUM_STEPS
    sample.sde_window_size = 2
    sample.sde_window_range = (0, EVAL_NUM_STEPS // 2)
    sample.guidance_scale = EVAL_GUIDANCE_SCALE
    sample.eval_guidance_scale = EVAL_GUIDANCE_SCALE
    sample.noise_level = EVAL_NOISE_LEVEL
    sample.train_batch_size = EVAL_BATCH_SIZE
    sample.num_image_per_prompt = EVAL_BATCH_SIZE
    # Without the training flow contract the sampler uses the plain SDE window.
    _drop(sample, "g021_flow_contract")
    sample.g021_stratified_sde = False

    # Without per-stage noise levels both generation and edits use
    # ``sample.noise_level``.
    _drop(g016, "noise_level_gen")
    _drop(g016, "noise_level_edit")
    g016.eta_clamp_mode = "exact_t_equals_one_sigma_max"
    g016.controller_temperature = EVAL_CONTROLLER_TEMPERATURE
    g016.controller_max_tokens = EVAL_CONTROLLER_MAX_TOKENS
    g016.controller_distributed_lockstep = False
    g016.max_repair_rounds = EVAL_MAX_REPAIR_ROUNDS
    return config
