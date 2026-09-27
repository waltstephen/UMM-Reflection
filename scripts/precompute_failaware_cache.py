#!/usr/bin/env python3
"""Precompute failaware image caches for BAGEL SFT.

Stage pixels: stores exact post-resize uint8 tensors for the VAE and VIT input
sizes. Loading them avoids PNG decode and bicubic resize in the DataLoader.

Stage encoded: stores frozen VAE encoder moments and frozen SigLIP hidden tokens.
The train loop samples/scales the moments to preserve AutoEncoder.encode
semantics while skipping the expensive encoder.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:  # pragma: no cover - environment validation path
    raise SystemExit("pyarrow is required to read failaware parquet shards") from exc

try:
    import torch.distributed as dist
except ImportError:  # pragma: no cover
    dist = None


ROOT_DIR = Path(__file__).resolve().parents[1]
BAGEL_DIR = ROOT_DIR / "third_party" / "Bagel"
sys.path.insert(0, str(BAGEL_DIR))

from safetensors import safe_open
from safetensors.torch import save_file

from data.data_utils import get_flattened_position_ids_extrapolate, patchify, pil_img2rgb
from data.failaware_cache import MANIFEST_NAME, tensor_key, u8_chw_to_normalized_float
from data.transforms import ImageTransform
from modeling.autoencoder import load_ae
from modeling.bagel import SiglipVisionConfig, SiglipVisionModel


DATA_ROOT = Path(os.environ.get("UNIFY_RL_DATA_ROOT", ROOT_DIR / "data"))
DEFAULT_MODEL = Path(os.environ.get("BAGEL_BASE_DIR", ROOT_DIR / "pretrained" / "BAGEL-7B-MoT"))
DEFAULT_PARQUET_DIR = DATA_ROOT / "sft" / "trajectory_parquet"
DEFAULT_CACHE_DIR = DATA_ROOT / "sft" / "pixel_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute failaware BAGEL image caches")
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--parquet-info", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vae-path", type=Path, default=None)
    parser.add_argument("--stage", choices=["pixels", "encoded", "both"], default="both")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-shards", type=int, default=None)
    parser.add_argument("--limit-rows-per-shard", type=int, default=None)
    parser.add_argument("--vae-batch-size", type=int, default=8)
    parser.add_argument("--vit-batch-images", type=int, default=64)
    parser.add_argument("--verify", type=int, default=2)
    parser.add_argument("--vit-select-layer", type=int, default=-2)
    parser.add_argument("--vit-rope", action="store_true")
    parser.add_argument("--image-max-size", type=int, default=1024)
    parser.add_argument("--image-min-size", type=int, default=512)
    parser.add_argument("--vit-max-size", type=int, default=364)
    parser.add_argument("--vit-min-size", type=int, default=224)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def init_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if dist is None:
            raise RuntimeError("torch.distributed is unavailable")
        dist.init_process_group("gloo", timeout=timedelta(minutes=60))
    return rank, world_size, local_rank


def barrier(world_size: int) -> None:
    if world_size > 1 and dist is not None and dist.is_initialized():
        dist.barrier()


def load_parquet_paths(parquet_dir: Path, parquet_info: Path | None) -> list[Path]:
    info_path = parquet_info or parquet_dir / "parquet_info.json"
    with info_path.open(encoding="utf-8") as f:
        info = json.load(f)
    paths = [Path(path) for path in sorted(info)]
    if not paths:
        raise ValueError(f"No parquet shards listed in {info_path}")
    return paths


def iter_rows(path: Path, limit_rows: int | None = None) -> Iterable[dict]:
    pf = pq.ParquetFile(str(path))
    yielded = 0
    for row_group_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(row_group_idx)
        for row in table.to_pylist():
            yield row
            yielded += 1
            if limit_rows is not None and yielded >= limit_rows:
                return


def resized_u8_chw(transform: ImageTransform, image) -> torch.Tensor:
    resized = transform.resize_transform(image)
    array = np.asarray(resized, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected RGB image array, got shape={array.shape}")
    return torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()


def build_vit(model_path: Path, device: torch.device, vit_select_layer: int, vit_rope: bool) -> SiglipVisionModel:
    vit_config = SiglipVisionConfig.from_json_file(str(model_path / "vit_config.json"))
    vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + vit_select_layer
    vit_config.rope = vit_rope
    vit_model = SiglipVisionModel(vit_config)
    vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    state_path = model_path / "ema.safetensors"
    state_dict = {}
    with safe_open(str(state_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("vit_model."):
                state_dict[key[len("vit_model."):]] = handle.get_tensor(key)
    missing, unexpected = vit_model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[vit-load] missing={len(missing)} unexpected={len(unexpected)}")
    vit_model.to(device).eval()
    for param in vit_model.parameters():
        param.requires_grad = False
    return vit_model


def encode_vae_moments(items: list[dict], vae_model, device: torch.device, batch_size: int) -> dict[str, torch.Tensor]:
    by_shape: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for item in items:
        by_shape[tuple(item["vae_u8"].shape)].append(item)

    outputs: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for shape, group in sorted(by_shape.items()):
            for start in range(0, len(group), batch_size):
                chunk = group[start:start + batch_size]
                batch = torch.stack([u8_chw_to_normalized_float(item["vae_u8"]) for item in chunk], dim=0).to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                    moments = vae_model.encoder(batch)
                moments = moments.detach().cpu().to(torch.bfloat16)
                for item, moment in zip(chunk, moments, strict=True):
                    outputs[tensor_key(item["uid"], item["step_idx"], "vae_moments")] = moment.contiguous()
    return outputs


def encode_vit_features(items: list[dict], vit_model, device: torch.device, batch_images: int) -> dict[str, torch.Tensor]:
    outputs: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for start in range(0, len(items), batch_images):
            chunk = items[start:start + batch_images]
            token_chunks = []
            position_chunks = []
            seqlens = []
            for item in chunk:
                vit_tensor = u8_chw_to_normalized_float(item["vit_u8"])
                tokens = patchify(vit_tensor, vit_model.config.patch_size)
                height, width = item["vit_u8"].shape[1:]
                positions = get_flattened_position_ids_extrapolate(
                    height,
                    width,
                    vit_model.config.patch_size,
                    max_num_patches_per_side=70,
                )
                token_chunks.append(tokens)
                position_chunks.append(positions)
                seqlens.append(tokens.shape[0])

            packed_tokens = torch.cat(token_chunks, dim=0).to(device)
            packed_positions = torch.cat(position_chunks, dim=0).to(device)
            vit_token_seqlens = torch.tensor(seqlens, dtype=torch.int, device=device)
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0)).to(torch.int32)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                features = vit_model(
                    packed_pixel_values=packed_tokens,
                    packed_flattened_position_ids=packed_positions,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max(seqlens),
                )
            features = features.detach().cpu().to(torch.bfloat16)
            offset = 0
            for item, seqlen in zip(chunk, seqlens, strict=True):
                outputs[tensor_key(item["uid"], item["step_idx"], "vit_features")] = features[offset:offset + seqlen].contiguous()
                height, width = item["vit_u8"].shape[1:]
                outputs[tensor_key(item["uid"], item["step_idx"], "vit_hw")] = torch.tensor([height, width], dtype=torch.int32)
                offset += seqlen
    return outputs


def collect_shard_items(path: Path, vae_transform: ImageTransform, vit_transform: ImageTransform, limit_rows: int | None) -> list[dict]:
    items: list[dict] = []
    for row in iter_rows(path, limit_rows=limit_rows):
        uid = str(row.get("uid", "") or "")
        if not uid:
            continue
        for step_idx, image_bytes in enumerate(row.get("step_image_list") or []):
            if not image_bytes:
                continue
            image = pil_img2rgb(Image.open(io.BytesIO(image_bytes)))
            vae_u8 = resized_u8_chw(vae_transform, image)
            vit_u8 = resized_u8_chw(vit_transform, image)
            items.append({"uid": uid, "step_idx": step_idx, "vae_u8": vae_u8, "vit_u8": vit_u8})
    return items


def save_shard_cache(
    shard_idx: int,
    items: list[dict],
    args: argparse.Namespace,
    vae_model,
    vit_model,
    device: torch.device,
) -> tuple[int, int]:
    pixel_dir = args.cache_dir / "pixels"
    encoded_dir = args.cache_dir / "encoded"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    encoded_dir.mkdir(parents=True, exist_ok=True)

    pixel_path = pixel_dir / f"shard-{shard_idx:05d}.safetensors"
    encoded_path = encoded_dir / f"shard-{shard_idx:05d}.safetensors"

    if args.stage in {"pixels", "both"} and (args.overwrite or not pixel_path.exists()):
        pixel_tensors = {}
        for item in items:
            pixel_tensors[tensor_key(item["uid"], item["step_idx"], "vae_u8")] = item["vae_u8"]
            pixel_tensors[tensor_key(item["uid"], item["step_idx"], "vit_u8")] = item["vit_u8"]
        save_file(pixel_tensors, str(pixel_path), metadata={"stage": "pixels"})

    if args.stage in {"encoded", "both"} and (args.overwrite or not encoded_path.exists()):
        encoded_tensors = {}
        encoded_tensors.update(encode_vae_moments(items, vae_model, device, args.vae_batch_size))
        encoded_tensors.update(encode_vit_features(items, vit_model, device, args.vit_batch_images))
        save_file(encoded_tensors, str(encoded_path), metadata={"stage": "encoded"})

    return len(items), sum(1 for _ in items)


def write_manifest(cache_dir: Path, parquet_paths: list[Path], limit_rows_per_shard: int | None = None) -> dict:
    entries = {}
    image_count = 0
    pixel_count = 0
    encoded_count = 0
    for shard_idx, parquet_path in enumerate(parquet_paths):
        pixel_rel = f"pixels/shard-{shard_idx:05d}.safetensors"
        encoded_rel = f"encoded/shard-{shard_idx:05d}.safetensors"
        pixel_exists = (cache_dir / pixel_rel).exists()
        encoded_exists = (cache_dir / encoded_rel).exists()
        if not pixel_exists and not encoded_exists:
            continue
        for row in iter_rows(parquet_path, limit_rows=limit_rows_per_shard):
            uid = str(row.get("uid", "") or "")
            steps = [idx for idx, image_bytes in enumerate(row.get("step_image_list") or []) if image_bytes]
            if not uid or not steps:
                continue
            image_count += len(steps)
            if pixel_exists:
                pixel_count += len(steps)
            if encoded_exists:
                encoded_count += len(steps)
            entries[uid] = {
                "parquet_shard": str(parquet_path),
                "steps": steps,
            }
            if pixel_exists:
                entries[uid]["pixel_shard"] = pixel_rel
            if encoded_exists:
                entries[uid]["encoded_shard"] = encoded_rel

    manifest = {
        "version": 1,
        "source_parquet_dir": str(parquet_paths[0].parent if parquet_paths else ""),
        "entries": entries,
        "image_count": image_count,
        "pixel_count": pixel_count,
        "encoded_count": encoded_count,
        "pixel_complete": pixel_count == image_count and image_count > 0,
        "encoded_complete": encoded_count == image_count and image_count > 0,
    }
    manifest_path = cache_dir / MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(
        "manifest: "
        f"entries={len(entries)} images={image_count} "
        f"pixel_count={pixel_count} encoded_count={encoded_count} "
        f"pixel_complete={manifest['pixel_complete']} encoded_complete={manifest['encoded_complete']}"
    )
    return manifest


def first_manifest_items(manifest: dict, count: int) -> list[tuple[str, int, str]]:
    selected = []
    for uid, entry in sorted(manifest.get("entries", {}).items()):
        for step_idx in entry.get("steps", []):
            selected.append((uid, int(step_idx), entry["parquet_shard"]))
            if len(selected) >= count:
                return selected
    return selected


def find_image_bytes(parquet_path: str, uid: str, step_idx: int) -> bytes:
    for row in iter_rows(Path(parquet_path)):
        if str(row.get("uid", "") or "") == uid:
            return row["step_image_list"][step_idx]
    raise KeyError(f"{uid} step {step_idx} not found in {parquet_path}")


def verify_cache(
    args: argparse.Namespace,
    manifest: dict,
    vae_transform: ImageTransform,
    vit_transform: ImageTransform,
    vae_model,
    vit_model,
    device: torch.device,
) -> None:
    if args.verify <= 0:
        return
    items = first_manifest_items(manifest, args.verify)
    if not items:
        print("verify: no manifest entries")
        return

    max_pixel_diff = 0.0
    max_moment_diff = 0.0
    max_vit_diff = 0.0
    for uid, step_idx, parquet_path in items:
        entry = manifest["entries"][uid]
        image = pil_img2rgb(Image.open(io.BytesIO(find_image_bytes(parquet_path, uid, step_idx))))

        if entry.get("pixel_shard"):
            with safe_open(str(args.cache_dir / entry["pixel_shard"]), framework="pt", device="cpu") as handle:
                cached_vae = u8_chw_to_normalized_float(handle.get_tensor(tensor_key(uid, step_idx, "vae_u8")))
                cached_vit = u8_chw_to_normalized_float(handle.get_tensor(tensor_key(uid, step_idx, "vit_u8")))
            live_vae = vae_transform(image)
            live_vit = vit_transform(image)
            max_pixel_diff = max(
                max_pixel_diff,
                float((cached_vae - live_vae).abs().max().item()),
                float((cached_vit - live_vit).abs().max().item()),
            )

        if entry.get("encoded_shard") and vae_model is not None and vit_model is not None:
            with safe_open(str(args.cache_dir / entry["encoded_shard"]), framework="pt", device="cpu") as handle:
                cached_moments = handle.get_tensor(tensor_key(uid, step_idx, "vae_moments"))
                cached_features = handle.get_tensor(tensor_key(uid, step_idx, "vit_features"))
            vae_u8 = resized_u8_chw(vae_transform, image)
            vit_u8 = resized_u8_chw(vit_transform, image)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                live_moments = vae_model.encoder(u8_chw_to_normalized_float(vae_u8).unsqueeze(0).to(device))[0].cpu().to(torch.bfloat16)
                vit_tokens = patchify(u8_chw_to_normalized_float(vit_u8), vit_model.config.patch_size)
                height, width = vit_u8.shape[1:]
                vit_positions = get_flattened_position_ids_extrapolate(
                    height,
                    width,
                    vit_model.config.patch_size,
                    max_num_patches_per_side=70,
                )
                live_features = vit_model(
                    packed_pixel_values=vit_tokens.to(device),
                    packed_flattened_position_ids=vit_positions.to(device),
                    cu_seqlens=torch.tensor([0, vit_tokens.shape[0]], dtype=torch.int32, device=device),
                    max_seqlen=vit_tokens.shape[0],
                ).cpu().to(torch.bfloat16)
            max_moment_diff = max(max_moment_diff, float((cached_moments - live_moments).abs().max().item()))
            max_vit_diff = max(max_vit_diff, float((cached_features - live_features).abs().max().item()))

    print(
        "verify: "
        f"checked={len(items)} max_pixel_diff={max_pixel_diff:.6g} "
        f"max_vae_moment_bf16_diff={max_moment_diff:.6g} "
        f"max_vit_feature_bf16_diff={max_vit_diff:.6g}"
    )


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    if args.device is None:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.set_default_dtype(torch.bfloat16)
    else:
        torch.set_default_dtype(torch.float32)

    parquet_paths = load_parquet_paths(args.parquet_dir, args.parquet_info)
    if args.limit_shards is not None:
        parquet_paths = parquet_paths[:args.limit_shards]
    assigned = [(idx, path) for idx, path in enumerate(parquet_paths) if idx % world_size == rank]
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    vae_transform = ImageTransform(args.image_max_size, args.image_min_size, 16)
    vit_transform = ImageTransform(args.vit_max_size, args.vit_min_size, 14)

    vae_model = None
    vit_model = None
    if args.stage in {"encoded", "both"}:
        vae_path = args.vae_path or args.model_path / "ae.safetensors"
        vae_model, _ = load_ae(local_path=str(vae_path))
        vae_model.to(device).eval()
        for param in vae_model.parameters():
            param.requires_grad = False
        vit_model = build_vit(args.model_path, device, args.vit_select_layer, args.vit_rope)

    processed_images = 0
    for shard_idx, parquet_path in assigned:
        pixel_path = args.cache_dir / "pixels" / f"shard-{shard_idx:05d}.safetensors"
        encoded_path = args.cache_dir / "encoded" / f"shard-{shard_idx:05d}.safetensors"
        wants_pixels = args.stage in {"pixels", "both"} and (args.overwrite or not pixel_path.exists())
        wants_encoded = args.stage in {"encoded", "both"} and (args.overwrite or not encoded_path.exists())
        if not wants_pixels and not wants_encoded:
            print(f"[rank {rank}] skip shard {shard_idx:05d}: cache exists")
            continue
        items = collect_shard_items(parquet_path, vae_transform, vit_transform, args.limit_rows_per_shard)
        n_items, _ = save_shard_cache(shard_idx, items, args, vae_model, vit_model, device)
        processed_images += n_items
        print(f"[rank {rank}] wrote shard {shard_idx:05d}: images={n_items}")

    print(f"[rank {rank}] processed_images={processed_images}")
    barrier(world_size)
    manifest = None
    if rank == 0:
        manifest = write_manifest(args.cache_dir, parquet_paths, limit_rows_per_shard=args.limit_rows_per_shard)
        verify_cache(args, manifest, vae_transform, vit_transform, vae_model, vit_model, device)
    barrier(world_size)
    if world_size > 1 and dist is not None and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
