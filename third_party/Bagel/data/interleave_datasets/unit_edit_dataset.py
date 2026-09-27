import io
import os
import re
from pathlib import Path

from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..failaware_cache import MANIFEST_NAME, FailawareCacheReader
from ..data_utils import pil_img2rgb


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class UNiTEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    """Dataset for UNiT multi-turn edit SFT with CE (reasoning) + MSE (image) loss.

    Parquet schema:
        image_list: list[bytes]   — round images (round_00, round_01, ...)
        instruction_list: list[str] — <think>...</think> HELP reasoning per round
        user_prompt: str          — prompt + constraints (conditioning text)
        system_prompt: str        — system prompt (conditioning text)

    Sequence structure per sample:
        [TEXT: system_prompt]       need_loss=False
        [TEXT: user_prompt]         need_loss=False
        [VIT+VAE: round_0]         need_loss=False  (condition input)
        [TEXT: think_0]             need_loss=True   (CE loss)
        [VAE: round_1]             need_loss=True   (MSE loss)
        [VIT: round_1]             need_loss=False   (understanding for next round)
        [TEXT: think_1]             need_loss=True   (CE loss)
        ...
        [VAE: round_N]             need_loss=True   (MSE loss, final)
        [VAE+VIT: round_N]         need_loss=False  (clean state for DONE)
        [TEXT: think_N (DONE)]      need_loss=True   (CE loss)
    """

    def __init__(
        self,
        *args,
        terminal_action_weight=10.0,
        terminal_edit_weight=10.0,
        terminal_score_weight=2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.terminal_action_weight = float(os.environ.get("UNIT_TERMINAL_ACTION_WEIGHT", terminal_action_weight))
        self.terminal_edit_weight = float(os.environ.get("UNIT_TERMINAL_EDIT_WEIGHT", terminal_edit_weight))
        self.terminal_score_weight = float(os.environ.get("UNIT_TERMINAL_SCORE_WEIGHT", terminal_score_weight))

    def _add_terminal_text(
        self,
        data,
        text,
        need_loss=True,
        enable_cfg=True,
    ):
        return self._add_text(
            data,
            text,
            need_loss=need_loss,
            enable_cfg=enable_cfg,
            ce_loss_weights=self._terminal_ce_loss_weights(text) if need_loss else None,
        )

    def _terminal_ce_loss_weights(self, text):
        text = str(text)
        text_ids = self.tokenizer.encode(text)
        weights = [1.0] * (len(text_ids) + 1)

        if not re.search(r"\[ACTION\]\s*done\b", text, flags=re.IGNORECASE):
            return None

        token_offsets = self._token_offsets(text, text_ids)
        matches = [
            (re.finditer(r"(\[ACTION\]\s*)(done\b)", text, flags=re.IGNORECASE), 2, self.terminal_action_weight),
            (re.finditer(r"(\[EDIT\]\s*)(none\b)", text, flags=re.IGNORECASE), 2, self.terminal_edit_weight),
            (
                re.finditer(r"(\[SCORE\]\s*)((?:9|10)(?:\.0+)?)(\s*/\s*10)", text, flags=re.IGNORECASE),
                2,
                self.terminal_score_weight,
            ),
        ]

        for iterator, group_idx, weight in matches:
            if weight <= 1.0:
                continue
            for match in iterator:
                if token_offsets is not None:
                    self._raise_char_span_weights(weights, token_offsets, match.start(group_idx), match.end(group_idx), weight)
                else:
                    self._raise_token_subsequence_weights(weights, text_ids, match.group(group_idx), weight)

        return weights if any(weight > 1.0 for weight in weights) else None

    def _token_offsets(self, text, text_ids):
        try:
            encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        except Exception:
            return None
        encoded_ids = encoded.get("input_ids") if hasattr(encoded, "get") else None
        offsets = encoded.get("offset_mapping") if hasattr(encoded, "get") else None
        if encoded_ids == text_ids and offsets is not None and len(offsets) == len(text_ids):
            return offsets
        return None

    @staticmethod
    def _raise_char_span_weights(weights, token_offsets, char_start, char_end, weight):
        for token_idx, (token_start, token_end) in enumerate(token_offsets):
            if token_start < char_end and token_end > char_start:
                weights[token_idx] = max(weights[token_idx], float(weight))

    def _raise_token_subsequence_weights(self, weights, text_ids, needle_text, weight):
        needle_ids = self.tokenizer.encode(needle_text)
        if not needle_ids:
            return
        limit = len(text_ids) - len(needle_ids) + 1
        for start in range(max(limit, 0)):
            if text_ids[start:start + len(needle_ids)] == needle_ids:
                for token_idx in range(start, start + len(needle_ids)):
                    weights[token_idx] = max(weights[token_idx], float(weight))

    def parse_row(self, row):
        image_list = row["image_list"]      # list[bytes]
        think_list = row["instruction_list"]  # list[str], HELP reasoning per round
        n_images = len(image_list)

        if n_images < 2 or len(think_list) < 2:
            return {}

        data = self._init_data()

        # System prompt (conditioning, no loss)
        system_prompt = row.get("system_prompt", "")
        if system_prompt:
            data = self._add_text(data, system_prompt, need_loss=False)

        user_prompt = row.get("user_prompt", "")
        context_order = str(row.get("_bagel_context_order", "") or "")
        prompt_after_image = context_order != "prompt_image"

        # BAGEL edit inference uses system -> input image -> prompt -> think -> image.
        # Rows can explicitly request the legacy prompt-first order if needed.
        if user_prompt and not prompt_after_image:
            data = self._add_text(data, user_prompt, need_loss=False)

        # Round 0: initial image as VIT+VAE condition input (no loss)
        img_0 = pil_img2rgb(Image.open(io.BytesIO(image_list[0])))
        data = self._add_image(data, img_0, need_loss=False, need_vae=True, need_vit=True)

        if user_prompt and prompt_after_image:
            data = self._add_text(data, user_prompt, need_loss=False)

        # Think text for round 0 → CE loss
        data = self._add_text(data, think_list[0], need_loss=True)

        # Rounds 1 to N-1 (intermediate editing rounds)
        for r in range(1, n_images):
            img_r = pil_img2rgb(Image.open(io.BytesIO(image_list[r])))

            if r < n_images - 1:
                # Intermediate image: MSE loss + VIT re-encode for next round understanding
                data = self._add_image(data, img_r, need_loss=True, need_vae=True, need_vit=True)
                # Think text for this round → CE loss
                data = self._add_text(data, think_list[r], need_loss=True)
            else:
                # Final image: MSE target plus clean VAE/VIT state.
                # The noisy MSE split is masked from later text; the clean
                # condition is what lets DONE be learned from the final image.
                data = self._add_image(data, img_r, need_loss=True, need_vae=True, need_vit=True)
                # Final think text (DONE) -> CE loss
                data = self._add_terminal_text(data, think_list[r], need_loss=True)

        return data


class FailawareTrajIterableDataset(UNiTEditIterableDataset):
    """Full-trajectory failaware SFT reader with think-first ordering."""

    edit_system_prompt = os.environ.get(
        "FAILAWARE_EDIT_SYSTEM_PROMPT",
        (
            "You are an honest source-image editing and verification agent.\n"
            "You will see the original user edit request and, for edit trajectories, a given source image before the first think.\n"
            "Treat [SOURCE_IMAGE] given as the source condition, never as a generated target.\n"
            "First verify the visible current/source image, then choose [ACTION] edit or done.\n"
            "When editing, write one concrete [EDIT] instruction that applies the requested change while preserving unrelated regions, layout, identities, and background.\n"
            "When done, use [EDIT] None. Output exactly one <think> block and no text before or after it.\n"
            "Use these exact tag spellings with no spaces inside brackets: [CURRENT_ROUND], [SCORE], [ACTION], [THINKING], [SOURCE_IMAGE], [EDIT]."
        ),
    )

    def _get_failaware_cache(self, task=None):
        if str(task or "").lower() == "edit" and os.environ.get("FAILAWARE_EDIT_CACHE_DIR"):
            if not hasattr(self, "_failaware_edit_cache"):
                cache_dir = os.environ["FAILAWARE_EDIT_CACHE_DIR"]
                mode = os.environ.get("FAILAWARE_EDIT_CACHE_MODE", os.environ.get("FAILAWARE_CACHE_MODE", "auto"))
                manifest_path = Path(cache_dir) / MANIFEST_NAME
                self._failaware_edit_cache = (
                    FailawareCacheReader(cache_dir, mode=mode) if manifest_path.exists() else None
                )
                if self._failaware_edit_cache is not None:
                    print(f"failaware edit cache enabled: mode={self._failaware_edit_cache.mode}")
            return self._failaware_edit_cache
        if not hasattr(self, "_failaware_cache"):
            self._failaware_cache = FailawareCacheReader.from_env()
            if self._failaware_cache is not None:
                print(f"failaware cache enabled: mode={self._failaware_cache.mode}")
        return self._failaware_cache

    def _add_pixel_cached_image(self, data, cached, need_loss, need_vae, need_vit, enable_cfg=True):
        vae_tensor = cached["vae_tensor"]
        vit_tensor = cached["vit_tensor"]
        assert need_loss or need_vae or need_vit

        if need_loss:
            data["sequence_plan"].append(
                {
                    "type": "vae_image",
                    "enable_cfg": 0,
                    "loss": 1,
                    "special_token_loss": 0,
                    "special_token_label": None,
                }
            )
            height, width = vae_tensor.shape[1:]
            data["num_tokens"] += width * height // self.transform.stride ** 2
            data["image_tensor_list"].append(vae_tensor)

        if need_vae:
            data["sequence_plan"].append(
                {
                    "type": "vae_image",
                    "enable_cfg": int(enable_cfg),
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                }
            )
            height, width = vae_tensor.shape[1:]
            data["num_tokens"] += width * height // self.transform.stride ** 2
            data["image_tensor_list"].append(vae_tensor.clone())

        if need_vit:
            data["sequence_plan"].append(
                {
                    "type": "vit_image",
                    "enable_cfg": int(enable_cfg),
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                }
            )
            height, width = vit_tensor.shape[1:]
            data["num_tokens"] += width * height // self.vit_transform.stride ** 2
            data["image_tensor_list"].append(vit_tensor)

        return data

    def _add_encoded_cached_image(self, data, cached, need_loss, need_vae, need_vit, enable_cfg=True):
        vae_moments = cached["vae_moments"]
        vit_features = cached["vit_features"]
        vit_hw = cached["vit_hw"]
        assert need_loss or need_vae or need_vit

        latent_patch_size = 2
        latent_h, latent_w = int(vae_moments.shape[1]), int(vae_moments.shape[2])
        patchified_shape = (latent_h // latent_patch_size, latent_w // latent_patch_size)
        num_vae_tokens = patchified_shape[0] * patchified_shape[1]

        if need_loss:
            data["sequence_plan"].append(
                {
                    "type": "vae_image",
                    "enable_cfg": 0,
                    "loss": 1,
                    "special_token_loss": 0,
                    "special_token_label": None,
                    "precomputed_vae_moments": 1,
                    "patchified_vae_latent_shape": patchified_shape,
                }
            )
            data["num_tokens"] += num_vae_tokens
            data["image_tensor_list"].append(vae_moments)

        if need_vae:
            data["sequence_plan"].append(
                {
                    "type": "vae_image",
                    "enable_cfg": int(enable_cfg),
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                    "precomputed_vae_moments": 1,
                    "patchified_vae_latent_shape": patchified_shape,
                }
            )
            data["num_tokens"] += num_vae_tokens
            data["image_tensor_list"].append(vae_moments.clone())

        if need_vit:
            data["sequence_plan"].append(
                {
                    "type": "vit_image",
                    "enable_cfg": int(enable_cfg),
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                    "precomputed_vit_features": 1,
                    "vit_hw": vit_hw,
                },
            )
            data["num_tokens"] += int(vit_features.shape[0])
            data["image_tensor_list"].append(vit_features)

        return data

    def _add_cached_image(self, data, cached, need_loss, need_vae, need_vit, enable_cfg=True):
        if cached.get("mode") == "encoded":
            return self._add_encoded_cached_image(data, cached, need_loss, need_vae, need_vit, enable_cfg=enable_cfg)
        return self._add_pixel_cached_image(data, cached, need_loss, need_vae, need_vit, enable_cfg=enable_cfg)

    def _add_failaware_image(self, data, image_bytes, uid, step_idx, cache, need_loss, need_vae=True, need_vit=True):
        cached = cache.get(uid, step_idx) if cache is not None and uid else None
        if cached is not None:
            return self._add_cached_image(
                data,
                cached,
                need_loss=need_loss,
                need_vae=need_vae,
                need_vit=need_vit,
            )
        image = pil_img2rgb(Image.open(io.BytesIO(image_bytes)))
        return self._add_image(
            data,
            image,
            need_loss=need_loss,
            need_vae=need_vae,
            need_vit=need_vit,
        )

    def parse_row(self, row):
        think_list = [str(item) for item in row["think_list"]]
        step_image_list = list(row["step_image_list"])
        step_need_loss = [bool(item) for item in row["step_need_loss"]]
        uid = str(row.get("uid", "") or "")
        task = str(row.get("task", "t2i") or "t2i").lower()
        cache = self._get_failaware_cache(task=task)

        if not (len(think_list) == len(step_image_list) == len(step_need_loss)):
            return {}
        if len(think_list) < 1 or sum(1 for image in step_image_list if image) < 1:
            return {}

        data = self._init_data()

        if task == "edit":
            system_prompt = str(row.get("system_prompt", "") or self.edit_system_prompt)
            if system_prompt:
                data = self._add_text(data, system_prompt, need_loss=False)

            user_prompt = row.get("user_prompt", "")
            if user_prompt:
                data = self._add_text(data, user_prompt, need_loss=False)

            source_image = step_image_list[0]
            if not source_image:
                return {}
            data = self._add_failaware_image(
                data,
                source_image,
                uid,
                0,
                cache,
                need_loss=False,
                need_vae=True,
                need_vit=True,
            )

            for step_idx in range(1, len(think_list)):
                think_text = think_list[step_idx]
                image_bytes = step_image_list[step_idx]
                need_loss_image = step_need_loss[step_idx]

                if self._is_done_think(think_text):
                    data = self._add_terminal_text(data, think_text, need_loss=True)
                else:
                    data = self._add_text(data, think_text, need_loss=True)

                if image_bytes:
                    data = self._add_failaware_image(
                        data,
                        image_bytes,
                        uid,
                        step_idx,
                        cache,
                        need_loss=need_loss_image,
                        need_vae=True,
                        need_vit=True,
                    )
            return data

        system_prompt = row.get("system_prompt", "")
        if system_prompt:
            data = self._add_text(data, system_prompt, need_loss=False)

        user_prompt = row.get("user_prompt", "")
        if user_prompt:
            data = self._add_text(data, user_prompt, need_loss=False)

        for step_idx, (think_text, image_bytes, need_loss_image) in enumerate(zip(think_list, step_image_list, step_need_loss)):
            if self._is_done_think(think_text):
                data = self._add_terminal_text(data, think_text, need_loss=True)
            else:
                data = self._add_text(data, think_text, need_loss=True)

            if image_bytes:
                data = self._add_failaware_image(
                    data,
                    image_bytes,
                    uid,
                    step_idx,
                    cache,
                    need_loss=need_loss_image,
                    need_vae=True,
                    need_vit=True,
                )

        return data

    @staticmethod
    def _is_done_think(text):
        return bool(re.search(r"\[ACTION\]\s*done\b", str(text or ""), flags=re.IGNORECASE))
