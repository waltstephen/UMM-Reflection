"""Helpers for failaware offline image/encoder caches."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote

import torch
from safetensors import safe_open


DEFAULT_CACHE_DIR = Path(os.environ.get("UNIFY_RL_DATA_ROOT", "data")) / "sft" / "pixel_cache"
MANIFEST_NAME = "manifest.json"


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def cache_uid(uid: object) -> str:
    return quote(str(uid), safe="")


def tensor_key(uid: object, step_idx: int, suffix: str) -> str:
    return f"{cache_uid(uid)}__{int(step_idx):04d}__{suffix}"


def u8_chw_to_normalized_float(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.uint8:
        return tensor.to(torch.get_default_dtype())
    return tensor.to(torch.get_default_dtype()).div(255.0).sub(0.5).div(0.5)


class FailawareCacheReader:
    """Small safetensors-backed reader used inside DataLoader workers."""

    def __init__(self, cache_dir: str | os.PathLike[str], mode: str = "auto") -> None:
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / MANIFEST_NAME
        with manifest_path.open(encoding="utf-8") as f:
            self.manifest = json.load(f)
        self.entries = self.manifest.get("entries", {})
        self._handles = {}
        self._key_sets = {}
        self.mode = self._resolve_mode(mode)

    @classmethod
    def from_env(cls) -> "FailawareCacheReader | None":
        if not env_flag("FAILAWARE_USE_CACHE", default=False):
            return None
        cache_dir = os.environ.get("FAILAWARE_CACHE_DIR", str(DEFAULT_CACHE_DIR))
        mode = os.environ.get("FAILAWARE_CACHE_MODE", "auto")
        manifest_path = Path(cache_dir) / MANIFEST_NAME
        if not manifest_path.exists():
            print(f"FAILAWARE_USE_CACHE=1 but cache manifest is missing: {manifest_path}")
            return None
        return cls(cache_dir, mode=mode)

    def _resolve_mode(self, mode: str) -> str:
        mode = str(mode or "auto").lower()
        encoded_complete = bool(self.manifest.get("encoded_complete", False))
        pixel_complete = bool(self.manifest.get("pixel_complete", False))
        if mode == "auto":
            if encoded_complete:
                return "encoded"
            if pixel_complete:
                return "pixels"
            if self.manifest.get("encoded_count", 0):
                return "encoded"
            return "pixels"
        if mode not in {"pixels", "encoded"}:
            raise ValueError(f"Unknown FAILAWARE_CACHE_MODE={mode!r}")
        return mode

    def _load_tensors(self, relpath: str, keys: list[str]) -> dict[str, torch.Tensor]:
        handle = self._handles.get(relpath)
        if handle is None:
            path = self.cache_dir / relpath
            handle = safe_open(str(path), framework="pt", device="cpu")
            self._handles[relpath] = handle
            self._key_sets[relpath] = set(handle.keys())
        tensors: dict[str, torch.Tensor] = {}
        available = self._key_sets[relpath]
        for key in keys:
            if key not in available:
                raise KeyError(f"{key} missing from {self.cache_dir / relpath}")
            tensors[key] = handle.get_tensor(key)
        return tensors

    def get(self, uid: object, step_idx: int) -> dict[str, torch.Tensor | tuple[int, int] | str] | None:
        entry = self.entries.get(str(uid))
        if entry is None:
            return None
        step_idx = int(step_idx)
        if step_idx not in {int(item) for item in entry.get("steps", [])}:
            return None

        if self.mode == "encoded" and entry.get("encoded_shard"):
            relpath = entry["encoded_shard"]
            keys = [
                tensor_key(uid, step_idx, "vae_moments"),
                tensor_key(uid, step_idx, "vit_features"),
                tensor_key(uid, step_idx, "vit_hw"),
            ]
            try:
                tensors = self._load_tensors(relpath, keys)
            except (FileNotFoundError, KeyError):
                tensors = {}
            if tensors:
                vit_hw_tensor = tensors[tensor_key(uid, step_idx, "vit_hw")].to(torch.int64)
                return {
                    "mode": "encoded",
                    "vae_moments": tensors[tensor_key(uid, step_idx, "vae_moments")],
                    "vit_features": tensors[tensor_key(uid, step_idx, "vit_features")],
                    "vit_hw": (int(vit_hw_tensor[0].item()), int(vit_hw_tensor[1].item())),
                }

        if entry.get("pixel_shard"):
            relpath = entry["pixel_shard"]
            keys = [
                tensor_key(uid, step_idx, "vae_u8"),
                tensor_key(uid, step_idx, "vit_u8"),
            ]
            try:
                tensors = self._load_tensors(relpath, keys)
            except (FileNotFoundError, KeyError):
                return None
            return {
                "mode": "pixels",
                "vae_tensor": u8_chw_to_normalized_float(tensors[tensor_key(uid, step_idx, "vae_u8")]),
                "vit_tensor": u8_chw_to_normalized_float(tensors[tensor_key(uid, step_idx, "vit_u8")]),
            }

        return None
