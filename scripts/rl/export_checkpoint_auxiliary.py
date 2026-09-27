#!/usr/bin/env python3
"""Export the nine trained auxiliary tensors of an RL checkpoint.

RL trains the full language model plus four small image-side modules
(``time_embedder``, ``vae2llm``, ``llm2vae``, ``latent_pos_embed``). The language
model is saved as DCP under ``checkpoint-N/dcp``; the auxiliary modules live only
in the per-rank resume state. This copies them from rank zero into one small
safetensors file so evaluation does not need the resume state.

    python scripts/rl/export_checkpoint_auxiliary.py \
        --checkpoint outputs/f01_formal1000/checkpoints/checkpoint-1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

AUX_NAMES = {"time_embedder", "vae2llm", "llm2vae", "latent_pos_embed"}
AUX_TENSOR_COUNT = 9
DEFAULT_NAME = "auxiliary.safetensors"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, help="checkpoint-N directory")
    parser.add_argument(
        "--output", default=None, help=f"default: <checkpoint>/{DEFAULT_NAME}"
    )
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    step = int(manifest.get("logical_step", -1))
    world = int(manifest.get("world_size", -1))
    if manifest.get("status") != "complete" or step < 0 or world < 1:
        raise RuntimeError(f"not a complete checkpoint: {checkpoint}")
    state = torch.load(
        checkpoint / "rank_state" / f"rank-00000-of-{world:05d}.pt",
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if int(state.get("logical_step", -1)) != step or int(state.get("world_size", -1)) != world:
        raise RuntimeError("rank-zero state does not match the checkpoint manifest")
    auxiliary = state["auxiliary_module_state"]
    if set(auxiliary) != AUX_NAMES:
        raise RuntimeError(f"auxiliary modules differ: {sorted(auxiliary)}")
    tensors = {
        f"{module}.{name}": tensor.detach().cpu().contiguous().clone()
        for module, values in auxiliary.items()
        for name, tensor in values.items()
    }
    if len(tensors) != AUX_TENSOR_COUNT:
        raise RuntimeError(f"expected {AUX_TENSOR_COUNT} auxiliary tensors, got {len(tensors)}")
    output = Path(args.output) if args.output else checkpoint / DEFAULT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    save_file(tensors, str(temporary), metadata={"checkpoint_step": str(step)})
    temporary.replace(output)
    print(json.dumps({
        "checkpoint": str(checkpoint),
        "logical_step": step,
        "output": str(output),
        "tensor_count": len(tensors),
    }, indent=2))


if __name__ == "__main__":
    main()
