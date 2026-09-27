#!/usr/bin/env python3
"""Generate GenEval-553 reflection trajectories for one arm on one GPU.

Each prompt gets a first image (round 0) and up to three controller-driven
repair rounds, rendered at 512x512 with 50 denoising steps. Prompts are
processed in fixed pairs (slots 2k, 2k+1) that share seed ``base_seed + k``, so
every arm sees the same initial noise. Run one process per GPU with
``--shard i --shards N``; completed samples are skipped on restart.

Arms:

* ``base`` -- pretrained BAGEL-7B-MoT (``--model-dir``, default ``$BAGEL_BASE_DIR``).
* ``sft``  -- the SFT overlay built by ``scripts/rl/make_rl_init.sh`` (``--model-dir``,
  default ``$RL_INIT_DIR``).
* ``rl``   -- either a merged RL model directory (``--model-dir``, e.g. the
  released UMM-Reflection-BAGEL-RL weights), or the SFT overlay plus an RL
  checkpoint (``--checkpoint``): the 784 language-model tensors from
  ``<checkpoint>/dcp`` and the nine auxiliary tensors exported by
  ``scripts/rl/export_checkpoint_auxiliary.py``.

Output: ``<output-root>/<label>/samples/<sample_id>/{round_XX.jpg,final.jpg,result.json}``.
Score with ``scripts/eval/score_geneval553.py``.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (
    REPO_ROOT / "third_party/flow_grpo",
    REPO_ROOT / "third_party/flow_grpo/scripts",
    REPO_ROOT / "third_party/Bagel",
    REPO_ROOT / "src",
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts/eval",
):
    sys.path.insert(0, str(entry))

MANIFEST = REPO_ROOT / "assets/geneval/geneval553_eval_manifest.json"
MANIFEST_SHA256 = "c674584f8ec64a997107c95a01c2625a8aa4a96f81ed9a2c5879789895381f2e"
SYSTEM_PROMPT = REPO_ROOT / "assets/prompts/controller_system_prompt.txt"
TRAIN_DATA = REPO_ROOT / "assets/data/f01_geneval_train.jsonl"
PROMPT_MANIFEST = REPO_ROOT / "assets/data/F01_PROMPT_POOL_MANIFEST.json"

PAIR_SIZE = 2
RESOLUTION = (512, 512)
DENOISE_STEPS = 50
CONTROLLER_TEMPERATURE = 0.5
CONTROLLER_MAX_TOKENS = 512
LANGUAGE_KEY_COUNT = 784
AUX_NAMES = {"time_embedder", "vae2llm", "llm2vae", "latent_pos_embed"}
AUX_TENSOR_COUNT = 9
# Logical step that seeds the repair-round SDE window. Reported base/SFT numbers
# use 0 and the reported RL checkpoint uses 500; see README.
DEFAULT_REPAIR_RNG_STEP = {"base": 0, "sft": 0, "rl": 500}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_image(image, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    image.convert("RGB").save(temporary, format="JPEG", quality=95)
    os.replace(temporary, path)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_manifest() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observed = sha256_file(MANIFEST)
    if observed != MANIFEST_SHA256:
        raise SystemExit(f"GenEval-553 manifest changed: sha256 {observed}")
    manifest = read_json(MANIFEST)
    rows = sorted(manifest["rows"], key=lambda row: int(row["slot"]))
    base_seed = int(manifest["base_seed"])
    if [int(row["slot"]) for row in rows] != list(range(int(manifest["n"]))):
        raise SystemExit("manifest slots are not 0..n-1")
    for row in rows:
        if int(row["seed"]) != base_seed + int(row["slot"]) // PAIR_SIZE:
            raise SystemExit(f"slot {row['slot']} does not carry its pair seed")
    return manifest, rows


def load_language_checkpoint(model, checkpoint: Path) -> dict[str, Any]:
    """Load the 784 trained language-model tensors from a DCP checkpoint."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    checkpoint = checkpoint.resolve()
    manifest = read_json(checkpoint / "manifest.json")
    inventory_path = checkpoint / "shard_inventory.json"
    inventory = read_json(inventory_path)
    if manifest.get("inventory_sha256") != sha256_file(inventory_path):
        raise RuntimeError("checkpoint manifest does not match its shard inventory")
    if manifest.get("status") != "complete":
        raise RuntimeError(f"checkpoint is not complete: {checkpoint}")
    keys = list(inventory.get("model_state_keys") or [])
    if len(keys) != LANGUAGE_KEY_COUNT:
        raise RuntimeError(f"expected {LANGUAGE_KEY_COUNT} model keys, got {len(keys)}")
    current = model.language_model.state_dict()
    missing = set(keys) - set(current)
    if missing:
        raise RuntimeError(f"checkpoint keys absent from model: {sorted(missing)[:4]}")
    target = {key: current[key] for key in keys}
    dcp.load({"model": target}, storage_reader=FileSystemReader(str(checkpoint / "dcp")))
    model.language_model.load_state_dict(target, strict=False)
    return {
        "checkpoint": checkpoint.name,
        "logical_step": int(manifest["logical_step"]),
        "world_size": int(manifest["world_size"]),
        "model_state_key_count": len(keys),
    }


def load_auxiliary(model, path: Path) -> int:
    """Load the nine trained auxiliary tensors (time/latent embedders, VAE bridges)."""
    from safetensors.torch import load_file

    modules: dict[str, dict[str, Any]] = {}
    for key, tensor in load_file(str(path), device="cpu").items():
        module, name = key.split(".", 1)
        modules.setdefault(module, {})[name] = tensor
    if set(modules) != AUX_NAMES:
        raise RuntimeError(f"auxiliary modules differ: {sorted(modules)}")
    loaded = 0
    for name, state in modules.items():
        getattr(model, name).load_state_dict(state, strict=True)
        loaded += len(state)
    if loaded != AUX_TENSOR_COUNT:
        raise RuntimeError(f"expected {AUX_TENSOR_COUNT} auxiliary tensors, got {loaded}")
    return loaded


def eval_config(model_dir: Path):
    # config/g016.py reads its paths from the environment; evaluation only
    # needs them to resolve.
    os.environ.setdefault("G016_OUTPUT_DIR", str(REPO_ROOT / "outputs/.eval_config"))
    os.environ.setdefault("G016_FLOWGRPO_MODEL_DIR", str(model_dir))
    os.environ.setdefault("G016_TRAIN_DATA", str(TRAIN_DATA))
    os.environ.setdefault("G016_PROMPT_NARROWING_MANIFEST", str(PROMPT_MANIFEST))
    os.environ.setdefault(
        "G016_PROMPT_NARROWING_MANIFEST_SHA256", sha256_file(PROMPT_MANIFEST)
    )
    from config.g016 import multiround_counting
    from eval_config import apply_eval_overrides

    return apply_eval_overrides(multiround_counting())


def build_rollout(args, model_dir: Path, device):
    import torch
    from flow_grpo.bagel.data.data_utils import add_special_tokens
    from flow_grpo.bagel.data.transforms import ImageTransform
    from flow_grpo.bagel.inferencer import InterleaveInferencer as FlowInterleaveInferencer
    from flow_grpo.g008_counting import active_num_timesteps
    from flow_grpo.g016_counting_process import FlowGRPOMultiroundRollout
    from inferencer import InterleaveInferencer as ControllerInferencer
    from modeling.qwen2 import Qwen2Tokenizer
    from train_bagel_g016_counting import build_bagel

    config = eval_config(model_dir)
    model, vae_model, _ = build_bagel(
        model_dir=model_dir, inference_dtype=torch.bfloat16, device=device
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    checkpoint_load = None
    if args.checkpoint is not None:
        checkpoint_load = load_language_checkpoint(model, Path(args.checkpoint))
        auxiliary = Path(args.auxiliary_safetensors or Path(args.checkpoint) / "auxiliary.safetensors")
        checkpoint_load["auxiliary_tensor_count"] = load_auxiliary(model, auxiliary)

    tokenizer = Qwen2Tokenizer.from_pretrained(str(model_dir))
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    vae_transform, vit_transform = ImageTransform(512, 256, 8), ImageTransform(490, 112, 7)
    shared = dict(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    rollout = FlowGRPOMultiroundRollout(
        flow_inferencer=FlowInterleaveInferencer(**shared),
        controller_inferencer=ControllerInferencer(**shared),
        system_prompt=SYSTEM_PROMPT.read_text(encoding="utf-8").rstrip("\n"),
        grpo_config=config,
        accelerator=SimpleNamespace(process_index=0, device=device),
    )
    steps = int(active_num_timesteps(rollout))
    if steps != DENOISE_STEPS:
        raise SystemExit(f"evaluation renders at {DENOISE_STEPS} steps, sampler says {steps}")
    return rollout, model, vae_model, checkpoint_load


def sample_complete(sample_dir: Path) -> bool:
    result = sample_dir / "result.json"
    return result.is_file() and read_json(result).get("status") == "complete"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arm", choices=("base", "sft", "rl"), required=True)
    parser.add_argument("--model-dir", default=None,
                        help="BAGEL model directory (default: $BAGEL_BASE_DIR or "
                             "pretrained/BAGEL-7B-MoT for base, $RL_INIT_DIR or "
                             "outputs/rl_init for sft/rl)")
    parser.add_argument("--checkpoint", default=None,
                        help="rl only: an RL checkpoint-N directory loaded on top of "
                             "--model-dir; omit when --model-dir is a merged RL model")
    parser.add_argument("--auxiliary-safetensors", default=None,
                        help="rl only: default <checkpoint>/auxiliary.safetensors")
    parser.add_argument("--raw-prompt", action="store_true",
                        help="use the official GenEval prompt instead of the "
                             "instruction-voice rewrite the policy was trained on")
    parser.add_argument("--r0-only", action="store_true",
                        help="generate only the first image, no reflection rounds")
    parser.add_argument("--repair-rng-step", type=int, default=None,
                        help="logical step seeding repair-round SDE windows "
                             "(default: 0 for base/sft, 500 for rl)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N slots (smoke test)")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "outputs/geneval553"))
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    if args.checkpoint is not None and args.arm != "rl":
        raise SystemExit("--checkpoint is only valid with --arm rl")
    if args.arm == "rl" and args.checkpoint is None and args.model_dir is None:
        raise SystemExit("--arm rl needs --checkpoint or a merged RL --model-dir")
    if not 0 <= args.shard < args.shards:
        raise SystemExit("--shard must be in [0, --shards)")
    if args.arm == "base":
        default_dir = os.environ.get("BAGEL_BASE_DIR", str(REPO_ROOT / "pretrained/BAGEL-7B-MoT"))
    else:
        default_dir = os.environ.get("RL_INIT_DIR", str(REPO_ROOT / "outputs/rl_init"))
    model_dir = Path(args.model_dir or default_dir).resolve()
    for required in ("ema.safetensors", "llm_config.json", "vit_config.json", "ae.safetensors"):
        if not (model_dir / required).exists():
            raise SystemExit(f"{model_dir} lacks {required}")
    repair_rng_step = (
        DEFAULT_REPAIR_RNG_STEP[args.arm]
        if args.repair_rng_step is None
        else int(args.repair_rng_step)
    )

    label = args.label
    if label is None:
        label = args.arm
        if args.arm == "rl":
            label += "-" + (Path(args.checkpoint).resolve().name
                            if args.checkpoint is not None else model_dir.name)
        if args.raw_prompt:
            label += "-rawprompt"
        if args.r0_only:
            label += "-r0only"
    samples_root = Path(args.output_root).resolve() / label / "samples"

    manifest, rows = load_manifest()
    base_seed = int(manifest["base_seed"])
    if args.limit is not None:
        rows = rows[: args.limit]
    pairs = [rows[i : i + PAIR_SIZE] for i in range(0, len(rows), PAIR_SIZE)]
    mine = [pair for index, pair in enumerate(pairs) if index % args.shards == args.shard]
    todo = [
        [row for row in pair if not sample_complete(samples_root / row["sample_id"])]
        for pair in mine
    ]
    todo = [pair for pair in todo if pair]
    owned = sum(len(pair) for pair in mine)
    print(f"[{label} shard {args.shard}/{args.shards}] {owned} samples, "
          f"{owned - sum(len(p) for p in todo)} already done", flush=True)
    if not todo:
        return

    import torch
    from accelerate.utils import set_seed
    from scripts.train_bagel import create_generators

    device = torch.device("cuda:0")
    rollout, model, vae_model, checkpoint_load = build_rollout(args, model_dir, device)
    finished = owned - sum(len(p) for p in todo)
    for pair in todo:
        seed = int(pair[0]["seed"])
        if any(int(row["seed"]) != seed for row in pair):
            raise SystemExit("a pair spans two seeds")
        set_seed(seed, device_specific=False)
        prompts = [str(row["official_prompt" if args.raw_prompt else "prompt"]) for row in pair]
        generators = create_generators(prompts, base_seed=seed)
        with torch.inference_mode():
            r0s = rollout.generate_r0_batch(prompts=prompts, generators=generators)
            if args.r0_only:
                trajectories = [
                    SimpleNamespace(flow_calls=[], events=[], done=True,
                                    stop_reason="r0_only", repair_rounds=0)
                    for _ in r0s
                ]
            else:
                trajectories = rollout.generate_from_anchor_batch(
                    prompts=prompts,
                    verification_contracts=[
                        [dict(v) for v in row["verification_contract"]] for row in pair
                    ],
                    anchor_images=[r0.image for r0 in r0s],
                    anchor_metadata=[{
                        "state_group_id": f"official553_slot_{int(row['slot']):05d}",
                        "state_group_member": 0,
                        "repair_global_index": int(row["slot"]),
                        "source_anchor_r0_global_index": int(row["slot"]),
                        "anchor_sha256": "local_forward_only_bound_after_generation",
                        "anchor_uid": str(row["sample_id"]),
                    } for row in pair],
                    repair_generators=generators,
                    repair_sde_window_identities=[{
                        "experiment_seed": base_seed,
                        "logical_step": repair_rng_step,
                        "trajectory_index": int(row["slot"]),
                    } for row in pair],
                    do_sample=True,
                    temperature=CONTROLLER_TEMPERATURE,
                    response_max_tokens=CONTROLLER_MAX_TOKENS,
                )
        for row, r0, trajectory in zip(pair, r0s, trajectories):
            sample_dir = samples_root / row["sample_id"]
            images = [r0.image] + [call.image for call in trajectory.flow_calls]
            bindings = []
            for index, image in enumerate(images):
                if tuple(image.size) != RESOLUTION:
                    raise RuntimeError(f"round {index} rendered at {image.size}")
                bindings.append(save_image(image, sample_dir / f"round_{index:02d}.jpg"))
            final = sample_dir / "final.jpg"
            if final.is_symlink() or final.exists():
                final.unlink()
            final.symlink_to(bindings[-1]["path"])
            events = list(trajectory.events)
            atomic_json(sample_dir / "result.json", {
                **{key: row[key] for key in (
                    "slot", "sample_id", "official_index", "family", "prompt",
                    "official_prompt", "seed", "verification_contract", "geneval_metadata",
                )},
                "status": "complete",
                "arm": args.arm,
                "model_dir": model_dir.name,
                "checkpoint_load": checkpoint_load,
                "raw_prompt": bool(args.raw_prompt),
                "repair_rng_logical_step": repair_rng_step,
                "resolution": list(RESOLUTION),
                "denoise_steps": DENOISE_STEPS,
                "final_image": bindings[-1]["path"],
                "image_bindings": bindings,
                "events": events,
                "done": bool(trajectory.done),
                "stop_reason": str(trajectory.stop_reason),
                "repair_rounds": int(trajectory.repair_rounds),
                "parse_valid": all(event.get("route_valid") is True for event in events),
            })
        finished += len(pair)
        print(f"  {finished}/{owned}", flush=True)

    del rollout, model, vae_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
