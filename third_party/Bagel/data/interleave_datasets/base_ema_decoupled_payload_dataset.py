"""Opt-in decoupled CE/MSE readers for the Base-EMA payload SFT."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import torch

from ..existing_exact_edit_relocation_contract import parse_relocated_response
from ..existing_exact_edit_task_v5_contract import system_prompt_for_task
from ..existing_split_allmse_contract import validate_enriched_metadata
from ..failaware_cache import FailawareCacheReader
from ..parquet_utils import get_parquet_data_paths
from .unit_edit_dataset import FailawareTrajIterableDataset


SCHEMA_TAG_WEIGHT = 2.0
ACTION_VALUE_WEIGHT = 3.0
EDIT_MARKER_WEIGHT = 3.0
PAYLOAD_FIRST_TOKEN_WEIGHT = 3.0
PAYLOAD_TOKEN_WEIGHT = 2.0
SOURCE_IMAGE_VALUE_WEIGHT = 2.0

_SCHEMA_TAG_RE = re.compile(
    r"<think>|</think>|\[(?:CURRENT_ROUND|SCORE|ACTION|THINKING|SOURCE_IMAGE)\]"
)
_ACTION_VALUE_RE = re.compile(
    r"^\[ACTION\][ \t]+(?P<value>edit|done)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_SOURCE_IMAGE_VALUE_RE = re.compile(
    r"^\[SOURCE_IMAGE\][ \t]+(?P<value>given|None|Image #[0-9]+)[ \t]*$",
    re.MULTILINE,
)


def normalized_prompt(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def prompt_sha256(value: str) -> str:
    return hashlib.sha256(normalized_prompt(value).encode("utf-8")).hexdigest()


class _StrictSingleCacheDataset(FailawareTrajIterableDataset):
    """Strict projected-parquet reader backed by one verified pixel cache."""

    fail_fast = True

    def __init__(self, *args, cache_dir, **kwargs):
        data_dir_list = [
            os.path.abspath(str(path))
            for path in list(kwargs.get("data_dir_list") or [])
        ]
        if len(data_dir_list) != 1:
            raise ValueError("decoupled reader requires exactly one data root")
        kwargs["data_dir_list"] = data_dir_list
        self.data_root = data_dir_list[0]
        self.cache_dir = os.path.abspath(str(cache_dir))
        manifest_path = Path(self.cache_dir) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("pixel_complete") is not True:
            raise RuntimeError(f"pixel cache is incomplete: {manifest_path}")
        self.cache_steps = {
            str(uid): {int(step) for step in entry.get("steps") or []}
            for uid, entry in (manifest.get("entries") or {}).items()
        }
        self._pixel_cache_reader = None
        super().__init__(*args, **kwargs)

    def get_data_paths(self, data_dir_list, num_used_data, parquet_info):
        if len(data_dir_list) != 1 or len(num_used_data) != 1:
            raise ValueError("decoupled reader data namespace mismatch")
        selected = sorted(
            get_parquet_data_paths(data_dir_list, num_used_data)
        )
        if len(selected) != int(num_used_data[0]):
            raise ValueError(
                f"selected {len(selected)} parquet paths, "
                f"expected {num_used_data[0]}"
            )
        row_groups = []
        seen = set()
        for raw_path in selected:
            data_path = os.path.abspath(str(raw_path))
            if data_path not in parquet_info:
                raise KeyError(f"parquet_info missing exact path {data_path}")
            for row_group_index in range(
                int(parquet_info[data_path]["num_row_groups"])
            ):
                item = (data_path, row_group_index)
                if item in seen:
                    raise ValueError(f"duplicate row group assignment: {item}")
                seen.add(item)
                row_groups.append(item)
        if not row_groups:
            raise ValueError("no row groups selected")
        return row_groups

    def set_epoch(self, seed=42):
        if self.data_paths is None:
            return
        data_paths = sorted(self.data_paths, key=lambda item: (item[0], item[1]))
        self.rng.seed(seed)
        self.rng.shuffle(data_paths)
        self.data_paths_per_rank = data_paths[
            self.local_rank :: self.world_size
        ]
        self.num_files_per_rank = len(self.data_paths_per_rank)

    def get_data_paths_per_worker(self):
        if self.data_paths is None:
            return None
        info = torch.utils.data.get_worker_info()
        if info is None:
            return self.data_paths_per_rank, 0
        return self.data_paths_per_rank[info.id :: info.num_workers], info.id

    def _cache(self):
        if self._pixel_cache_reader is None:
            self._pixel_cache_reader = FailawareCacheReader(
                self.cache_dir,
                mode="pixels",
            )
        return self._pixel_cache_reader

    def _cached_image(self, uid: str, step_index: int):
        cached = self._cache().get(uid, step_index)
        if cached is None:
            raise KeyError(f"cache_miss:{uid}:{step_index}")
        if cached.get("mode") != "pixels":
            raise ValueError(f"non_pixel_cache:{uid}:{step_index}")
        return cached

    def _image_markers(self, uid: str, count: int):
        if uid not in self.cache_steps:
            raise KeyError(f"cache_manifest_uid_missing:{uid}")
        steps = self.cache_steps[uid]
        if any(step < 0 or step >= count for step in steps):
            raise ValueError(f"cache_step_outside_trajectory:{uid}")
        return [b"cached" if step in steps else b"" for step in range(count)]

    def parse_row_from_source(
        self,
        row,
        data_path,
        row_group_id,
        row_idx,
    ):
        return self.parse_row_strict(row, data_path)

    def parse_row(self, row):
        raise RuntimeError("decoupled readers require source-aware parsing")


class Clean29529WeightedControllerIterableDataset(
    _StrictSingleCacheDataset
):
    """Full Clean29K trajectory context with weighted controller CE only."""

    system_prompt_version = "base_ema_decoupled_payload_controller_v1"

    @staticmethod
    def canonical_response(parsed):
        payload = parsed.payload if parsed.action == "edit" else "None"
        return f"{parsed.controller}\n[EDIT] {payload}"

    @staticmethod
    def payload_field_was_canonicalized(parsed):
        raw_field = str(parsed.edit_field)
        if raw_field.endswith("\r\n"):
            raw_field = raw_field[:-2]
        elif raw_field.endswith("\n"):
            raw_field = raw_field[:-1]
        expected = (
            f"[EDIT] {parsed.payload}"
            if parsed.action == "edit"
            else "[EDIT] None"
        )
        return raw_field != expected

    def get_read_columns(self, data_path):
        return [
            "uid",
            "run",
            "task",
            "category",
            "trajectory_subtype",
            "user_prompt",
            "think_list",
            "step_action_list",
            "step_role_list",
            "step_round_list",
            "step_source_image_list",
            "step_image_name_list",
            "step_image_sha256",
            "step_image_size_bytes",
            "step_need_loss",
            "n_steps",
            "n_images",
            "source_image_step_index",
            "final_loss_step_index",
            "final_image_name",
        ]

    def _token_offsets(self, text: str, text_ids: list[int]):
        try:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
        except (NotImplementedError, TypeError):
            encoded = None
        if encoded is not None:
            encoded_ids = encoded.get("input_ids")
            offsets = encoded.get("offset_mapping")
            if encoded_ids == text_ids and offsets is not None:
                if len(offsets) != len(text_ids):
                    raise RuntimeError("tokenizer_offset_length_mismatch")
                return (
                    [(int(start), int(end)) for start, end in offsets],
                    False,
                )

        byte_decoder = getattr(self.tokenizer, "byte_decoder", None)
        if not byte_decoder:
            raise RuntimeError("tokenizer_offsets_unavailable")
        tokens = self.tokenizer.convert_ids_to_tokens(text_ids)
        offsets = []
        cursor = 0
        for token in tokens:
            try:
                token_bytes = bytes(
                    byte_decoder[character] for character in token
                )
            except (KeyError, TypeError) as exc:
                raise RuntimeError("tokenizer_byte_offset_failure") from exc
            offsets.append((cursor, cursor + len(token_bytes)))
            cursor += len(token_bytes)
        if cursor != len(text.encode("utf-8")):
            raise RuntimeError("tokenizer_byte_offset_length_mismatch")
        return offsets, True

    @staticmethod
    def _raise_span(weights, offsets, start, end, value):
        indexes = []
        for token_index, (token_start, token_end) in enumerate(offsets):
            if token_start < end and token_end > start:
                weights[token_index] = max(
                    weights[token_index],
                    float(value),
                )
                indexes.append(token_index)
        return indexes

    @classmethod
    def _raise_text_span(
        cls,
        weights,
        offsets,
        byte_offsets,
        text,
        start,
        end,
        value,
    ):
        if byte_offsets:
            start = len(text[:start].encode("utf-8"))
            end = len(text[:end].encode("utf-8"))
        return cls._raise_span(weights, offsets, start, end, value)

    def response_ce_weights(
        self,
        response: str,
        *,
        task: str,
        expected_round_index: int,
        expected_source_image: str,
    ):
        parsed = parse_relocated_response(
            response,
            task=task,
            expected_round_index=expected_round_index,
            expected_source_image=expected_source_image,
        )
        text_ids = self.tokenizer.encode(response)
        offsets, byte_offsets = self._token_offsets(response, text_ids)
        weights = [1.0] * (len(text_ids) + 1)

        for match in _SCHEMA_TAG_RE.finditer(response):
            self._raise_text_span(
                weights,
                offsets,
                byte_offsets,
                response,
                match.start(),
                match.end(),
                SCHEMA_TAG_WEIGHT,
            )

        action_match = _ACTION_VALUE_RE.search(response)
        source_match = _SOURCE_IMAGE_VALUE_RE.search(response)
        if action_match is None or source_match is None:
            raise ValueError("weighted_response_field_missing")
        self._raise_text_span(
            weights,
            offsets,
            byte_offsets,
            response,
            action_match.start("value"),
            action_match.end("value"),
            ACTION_VALUE_WEIGHT,
        )
        self._raise_text_span(
            weights,
            offsets,
            byte_offsets,
            response,
            source_match.start("value"),
            source_match.end("value"),
            SOURCE_IMAGE_VALUE_WEIGHT,
        )

        edit_marker_start = response.index("[EDIT]", response.index("</think>"))
        close_end = response.index("</think>") + len("</think>")
        self._raise_text_span(
            weights,
            offsets,
            byte_offsets,
            response,
            close_end,
            edit_marker_start,
            SCHEMA_TAG_WEIGHT,
        )
        self._raise_text_span(
            weights,
            offsets,
            byte_offsets,
            response,
            edit_marker_start,
            edit_marker_start + len("[EDIT]"),
            EDIT_MARKER_WEIGHT,
        )

        if parsed.action == "edit":
            payload_start = edit_marker_start + len("[EDIT] ")
            payload_end = len(response)
            payload_indexes = self._raise_text_span(
                weights,
                offsets,
                byte_offsets,
                response,
                payload_start,
                payload_end,
                PAYLOAD_TOKEN_WEIGHT,
            )
            if not payload_indexes:
                raise ValueError("nonempty_payload_has_no_tokens")
            weights[payload_indexes[0]] = max(
                weights[payload_indexes[0]],
                PAYLOAD_FIRST_TOKEN_WEIGHT,
            )
        elif not response.endswith("[EDIT] None"):
            raise ValueError("done_response_not_exact_none")

        # The response EOS and the external None value remain ordinary 1x CE.
        weights[-1] = 1.0
        return weights

    def parse_row_strict(self, row, data_path):
        uid = str(row.get("uid", "") or "")
        task = str(row.get("task", "") or "").lower()
        user_prompt = str(row.get("user_prompt", "") or "")
        if not uid or task not in {"t2i", "edit"} or not user_prompt:
            raise ValueError("invalid_controller_identity")

        think_list = [str(value or "") for value in row["think_list"]]
        image_markers = self._image_markers(uid, len(think_list))
        metadata = validate_enriched_metadata(row, image_markers)
        if metadata["task"] != task:
            raise ValueError("controller_task_mismatch")

        data = self._init_data()
        data = self._add_text(
            data,
            system_prompt_for_task(task),
            need_loss=False,
        )
        start_index = metadata["start_index"]
        if task == "edit":
            data = self._add_cached_image(
                data,
                self._cached_image(uid, 0),
                need_loss=False,
                need_vae=True,
                need_vit=True,
            )
        data = self._add_text(data, user_prompt, need_loss=False)

        response_count = 0
        canonicalized_payload_fields = 0
        for step_index in range(start_index, len(think_list)):
            parsed = metadata["parsed_controllers"][step_index]
            expected_round = metadata["rounds"][step_index]
            expected_source = metadata["source_refs"][step_index]
            response = self.canonical_response(parsed)
            parse_relocated_response(
                response,
                task=task,
                expected_round_index=expected_round,
                expected_source_image=expected_source,
            )
            canonicalized_payload_fields += int(
                self.payload_field_was_canonicalized(parsed)
            )
            if step_index == start_index:
                expected_first_source = "None" if task == "t2i" else "given"
                if (
                    expected_source != expected_first_source
                    or metadata["recorded_roles"][step_index] != "plan"
                    or parsed.action != "edit"
                ):
                    raise ValueError("invalid_first_response_route")

            data = self._add_text(
                data,
                response,
                need_loss=True,
                enable_cfg=False,
                ce_loss_weights=self.response_ce_weights(
                    response,
                    task=task,
                    expected_round_index=expected_round,
                    expected_source_image=expected_source,
                ),
            )
            response_count += 1
            if parsed.action == "edit":
                data = self._add_cached_image(
                    data,
                    self._cached_image(uid, step_index),
                    need_loss=False,
                    need_vae=True,
                    need_vit=True,
                )

        if any(
            item["type"] == "vae_image" and item["loss"] == 1
            for item in data["sequence_plan"]
        ):
            raise AssertionError("controller sample contains image MSE")
        data["_sample_metadata"] = {
            "uid": uid,
            "task": task,
            "source_family": "clean29529_controller",
            "controller_responses": response_count,
            "canonicalized_payload_fields": canonicalized_payload_fields,
            "system_prompt_version": self.system_prompt_version,
        }
        return data


class BasePromptOnlyAnchorMSEIterableDataset(_StrictSingleCacheDataset):
    """Decontaminated Base prompt-only anchor with one final-image MSE."""

    def __init__(self, *args, anchor_allowlist_path, **kwargs):
        record = json.loads(
            Path(anchor_allowlist_path).read_text(encoding="utf-8")
        )
        if record.get("pass") is not True:
            raise ValueError("anchor allowlist does not pass")
        self.anchor_rows = {
            str(item["uid"]): dict(item) for item in record["rows"]
        }
        if len(self.anchor_rows) != len(record["rows"]):
            raise ValueError("duplicate anchor UID")
        super().__init__(*args, **kwargs)

    def get_read_columns(self, data_path):
        return ["uid", "task", "category", "user_prompt"]

    def parse_row_strict(self, row, data_path):
        uid = str(row.get("uid", "") or "")
        if uid not in self.anchor_rows:
            return {}
        item = self.anchor_rows[uid]
        prompt = str(row.get("user_prompt", "") or "")
        if (
            str(row.get("task", "") or "").lower() != "t2i"
            or prompt_sha256(prompt) != item["normalized_prompt_sha256"]
        ):
            raise ValueError(f"anchor_row_mismatch:{uid}")
        target_step = int(item["target_step_index"])
        data = self._init_data()
        data = self._add_text(
            data,
            prompt,
            need_loss=False,
            enable_cfg=True,
        )
        data = self._add_cached_image(
            data,
            self._cached_image(uid, target_step),
            need_loss=True,
            need_vae=False,
            need_vit=False,
        )
        data["_sample_metadata"] = {
            "uid": f"anchor::{uid}",
            "task": "t2i",
            "source_family": "base_prompt_only_anchor",
            "target_step_index": target_step,
        }
        return data


__all__ = [
    "ACTION_VALUE_WEIGHT",
    "BasePromptOnlyAnchorMSEIterableDataset",
    "Clean29529WeightedControllerIterableDataset",
    "EDIT_MARKER_WEIGHT",
    "PAYLOAD_FIRST_TOKEN_WEIGHT",
    "PAYLOAD_TOKEN_WEIGHT",
    "SCHEMA_TAG_WEIGHT",
    "SOURCE_IMAGE_VALUE_WEIGHT",
    "normalized_prompt",
    "prompt_sha256",
]
