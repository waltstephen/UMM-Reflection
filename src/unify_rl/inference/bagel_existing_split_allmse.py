"""Host-routed split-controller/payload inference with true persistent KV."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image


BAGEL_NUM_TIMESTEPS = 50
_ACTION_RE = re.compile(
    r"^[ \t]*\[ACTION\][ \t]+(?P<action>edit|done)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_FORBIDDEN_PAYLOAD_RE = re.compile(
    r"(?i)(?:<think|</think|\[(?:ACTION|EDIT|THINKING|SOURCE_IMAGE|SCORE)\])"
)


@dataclass
class HostRoutedResult:
    final_image: Image.Image | None
    events: list[dict[str, Any]]
    stop_reason: str
    context_signatures: dict[str, Any]


def parse_split_controller(text: str) -> str:
    value = str(text or "").strip()
    if (
        not value.startswith("<think>")
        or not value.endswith("</think>")
        or value.count("<think>") != 1
        or value.count("</think>") != 1
        or re.search(r"\[EDIT\]", value, flags=re.IGNORECASE)
    ):
        raise ValueError("malformed_split_controller")
    actions = list(_ACTION_RE.finditer(value))
    if len(actions) != 1:
        raise ValueError(f"controller_action_field_count:{len(actions)}")
    return actions[0].group("action").lower()


def validate_payload(text: str) -> str:
    payload = str(text or "").strip()
    if not payload:
        raise ValueError("empty_payload")
    if _FORBIDDEN_PAYLOAD_RE.search(payload):
        raise ValueError("payload_contains_protocol_text")
    return payload


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


class ExistingSplitHostRoutedInferencer:
    """Matching inference protocol for the existing-split all-MSE reader."""

    def __init__(
        self,
        inferencer,
        *,
        system_prompt: str,
        raw_text_sink: Callable[[str, int, str, dict[str, Any]], None] | None = None,
    ):
        self.inferencer = inferencer
        self.system_prompt = str(system_prompt)
        self.raw_text_sink = raw_text_sink

    def _signature(self, context):
        return self.inferencer.context_cache_signature(context)

    def _persist_raw(
        self,
        kind: str,
        round_index: int,
        text: str,
        generation: dict[str, Any],
    ) -> None:
        if self.raw_text_sink is not None:
            self.raw_text_sink(kind, round_index, text, generation)

    def init_contexts(
        self,
        *,
        prompt: str,
        source_image: Image.Image | None,
    ):
        from data.data_utils import pil_img2rgb

        main = self.inferencer.init_gen_context()
        text_unconditional = self.inferencer.init_gen_context()
        image_unconditional = self.inferencer.init_gen_context()

        main = self.inferencer.update_context_text(
            self.system_prompt,
            main,
        )
        image_unconditional = self.inferencer.update_context_text(
            self.system_prompt,
            image_unconditional,
        )
        if source_image is not None:
            resized_source = self.inferencer.vae_transform.resize_transform(
                pil_img2rgb(source_image.convert("RGB"))
            )
            main = self.inferencer.update_context_image(
                resized_source,
                main,
                vae=True,
                vit=True,
            )
            text_unconditional = self.inferencer.update_context_image(
                resized_source,
                text_unconditional,
                vae=True,
                vit=True,
            )
        main = self.inferencer.update_context_text(str(prompt), main)
        image_unconditional = self.inferencer.update_context_text(
            str(prompt),
            image_unconditional,
        )
        return {
            "main": main,
            "text_unconditional": text_unconditional,
            "image_unconditional": image_unconditional,
        }

    def generate(
        self,
        *,
        prompt: str,
        source_image: Image.Image | None = None,
        image_shape: tuple[int, int] = (1024, 1024),
        max_rounds: int = 4,
        controller_max_tokens: int = 512,
        payload_max_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 0.6,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 2.0,
        cfg_interval: tuple[float, float] = (0.4, 1.0),
        timestep_shift: float = 3.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
    ) -> HostRoutedResult:
        from data.data_utils import pil_img2rgb

        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        contexts = self.init_contexts(
            prompt=prompt,
            source_image=source_image,
        )
        events = []
        current_image = (
            source_image.convert("RGB") if source_image is not None else None
        )
        stop_reason = "round_budget"

        for round_index in range(max_rounds):
            controller = self.inferencer.gen_text_persistent(
                contexts["main"],
                max_length=controller_max_tokens,
                do_sample=do_sample,
                temperature=temperature,
                return_log_probs=True,
            )
            controller_text = str(controller["text"])
            self._persist_raw(
                "controller",
                round_index,
                controller_text,
                controller,
            )
            self.inferencer.update_context_token_ids(
                controller["full_segment_token_ids"],
                contexts["image_unconditional"],
            )
            action = parse_split_controller(controller_text)
            event = {
                "round_index": round_index,
                "raw_controller": controller_text,
                "controller_content_token_ids": controller[
                    "content_token_ids"
                ],
                "controller_full_segment_token_ids": controller[
                    "full_segment_token_ids"
                ],
                "controller_stop_reason": controller["stop_reason"],
                "controller_eos_model_selected": controller[
                    "eos_model_selected"
                ],
                "controller_eos_host_forced": controller["eos_host_forced"],
                "action": action,
                "raw_payload": None,
                "payload_content_token_ids": [],
                "payload_full_segment_token_ids": [],
                "image_generated": False,
            }
            if action == "done":
                stop_reason = "done"
                events.append(event)
                break

            payload = self.inferencer.gen_text_persistent(
                contexts["main"],
                max_length=payload_max_tokens,
                do_sample=do_sample,
                temperature=temperature,
                return_log_probs=True,
            )
            payload_text = str(payload["text"])
            self._persist_raw(
                "payload",
                round_index,
                payload_text,
                payload,
            )
            self.inferencer.update_context_token_ids(
                payload["full_segment_token_ids"],
                contexts["image_unconditional"],
            )
            executable_payload = validate_payload(payload_text)
            event.update(
                {
                    "raw_payload": payload_text,
                    "payload": executable_payload,
                    "payload_content_token_ids": payload[
                        "content_token_ids"
                    ],
                    "payload_full_segment_token_ids": payload[
                        "full_segment_token_ids"
                    ],
                    "payload_stop_reason": payload["stop_reason"],
                    "payload_eos_model_selected": payload[
                        "eos_model_selected"
                    ],
                    "payload_eos_host_forced": payload["eos_host_forced"],
                }
            )

            pre_diffusion = {
                name: self._signature(context)
                for name, context in contexts.items()
            }
            image = self.inferencer.gen_image(
                image_shape,
                contexts["main"],
                cfg_text_precontext=contexts["text_unconditional"],
                cfg_img_precontext=contexts["image_unconditional"],
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=list(cfg_interval),
                timestep_shift=timestep_shift,
                num_timesteps=BAGEL_NUM_TIMESTEPS,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
            ).convert("RGB")
            post_diffusion = {
                name: self._signature(context)
                for name, context in contexts.items()
            }
            if post_diffusion != pre_diffusion:
                raise AssertionError("diffusion changed persistent contexts")

            resized_image = self.inferencer.vae_transform.resize_transform(
                pil_img2rgb(image)
            )
            image_unconditional_before = self._signature(
                contexts["image_unconditional"]
            )
            contexts["main"] = self.inferencer.update_context_image(
                resized_image,
                contexts["main"],
                vae=True,
                vit=True,
            )
            contexts[
                "text_unconditional"
            ] = self.inferencer.update_context_image(
                resized_image,
                contexts["text_unconditional"],
                vae=True,
                vit=True,
            )
            if self._signature(
                contexts["image_unconditional"]
            ) != image_unconditional_before:
                raise AssertionError(
                    "clean image feedback changed text-only CFG context"
                )
            current_image = image
            event.update(
                {
                    "image_generated": True,
                    "image_size": list(image.size),
                    "num_timesteps": BAGEL_NUM_TIMESTEPS,
                    "pre_diffusion_contexts": pre_diffusion,
                    "post_feedback_contexts": {
                        name: self._signature(context)
                        for name, context in contexts.items()
                    },
                }
            )
            events.append(event)

        return HostRoutedResult(
            final_image=current_image,
            events=events,
            stop_reason=stop_reason,
            context_signatures={
                name: self._signature(context)
                for name, context in contexts.items()
            },
        )


class DirectoryRawTextSink:
    """Persist raw generated text and token metadata before strict parsing."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def __call__(
        self,
        kind: str,
        round_index: int,
        text: str,
        generation: dict[str, Any],
    ) -> None:
        stem = f"round_{round_index:02d}_{kind}"
        atomic_write_text(self.root / f"{stem}.txt", text)
        record = {
            "kind": kind,
            "round_index": int(round_index),
            "content_token_ids": generation["content_token_ids"],
            "full_segment_token_ids": generation[
                "full_segment_token_ids"
            ],
            "stop_reason": generation["stop_reason"],
            "eos_model_selected": generation["eos_model_selected"],
            "eos_host_forced": generation["eos_host_forced"],
            "cache_delta": generation["cache_delta"],
            "kv_lens": generation["kv_lens"],
            "ropes": generation["ropes"],
        }
        atomic_write_text(
            self.root / f"{stem}.json",
            json.dumps(record, indent=2, sort_keys=True) + "\n",
        )
