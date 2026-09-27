from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


BAGEL_NATIVE_NUM_TIMESTEPS = 50


def apply_transition_rows(model: Any, path: str) -> dict[str, Any]:
    from safetensors.torch import load_file

    tensors = load_file(path, device="cpu")
    token_ids = tensors["token_ids"].to(torch.long)
    trained_rows = tensors["trained_rows"]
    if trained_rows.ndim != 2 or trained_rows.shape[0] != token_ids.numel():
        raise ValueError("Transition-row artifact has inconsistent token IDs and rows")
    lm_head = model.language_model.lm_head
    with torch.no_grad():
        lm_head.weight[token_ids.to(lm_head.weight.device)] = trained_rows.to(
            device=lm_head.weight.device,
            dtype=lm_head.weight.dtype,
        )
    return {
        "path": path,
        "token_ids": token_ids.tolist(),
        "row_count": int(token_ids.numel()),
        "hidden_size": int(trained_rows.shape[1]),
    }


class BagelTransitionRouter:
    LABELS = ("vision_start", "im_start", "im_end", "other")

    def __init__(
        self,
        *,
        token_ids: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_bias: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_bias: torch.Tensor,
        device: torch.device,
        threshold: float,
        text_decode_threshold: float | None,
        source_path: str,
    ):
        self.token_ids = token_ids.to(device=device, dtype=torch.long)
        self.fc1_weight = fc1_weight.to(device=device, dtype=torch.float32)
        self.fc1_bias = fc1_bias.to(device=device, dtype=torch.float32)
        self.fc2_weight = fc2_weight.to(device=device, dtype=torch.float32)
        self.fc2_bias = fc2_bias.to(device=device, dtype=torch.float32)
        self.device = device
        self.threshold = float(threshold)
        self.text_decode_threshold = (
            self.threshold
            if text_decode_threshold is None
            else float(text_decode_threshold)
        )
        self.source_path = source_path

    @torch.no_grad()
    def decide(self, hidden: torch.Tensor, route_site: str) -> dict[str, Any]:
        normalized = F.layer_norm(hidden.float(), (hidden.numel(),))
        features = F.gelu(
            F.linear(normalized, self.fc1_weight, self.fc1_bias),
            approximate="tanh",
        )
        logits = F.linear(features, self.fc2_weight, self.fc2_bias)
        probabilities = torch.softmax(logits, dim=-1)
        class_index = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[class_index].item())
        allowed = {0, 2} if route_site in {"text_decode", "text_end"} else {0, 1, 2}
        decision_threshold = (
            self.text_decode_threshold
            if route_site == "text_decode"
            else self.threshold
        )
        override = class_index in allowed and confidence >= decision_threshold
        selected_token_id = (
            int(self.token_ids[class_index].item()) if override else None
        )
        return {
            "source": "transition_router",
            "artifact": self.source_path,
            "route_site": route_site,
            "threshold": decision_threshold,
            "boundary_threshold": self.threshold,
            "text_decode_threshold": self.text_decode_threshold,
            "class_index": class_index,
            "class_name": self.LABELS[class_index],
            "confidence": confidence,
            "class_probabilities": {
                name: float(probabilities[index].item())
                for index, name in enumerate(self.LABELS)
            },
            "override": override,
            "selected_token_id": selected_token_id,
        }


def load_transition_router(
    path: str,
    *,
    device: torch.device,
    threshold: float | None = None,
    text_decode_threshold: float | None = None,
) -> BagelTransitionRouter:
    from safetensors.torch import load_file

    tensors = load_file(path, device="cpu")
    artifact_threshold = float(tensors["threshold"].item())
    return BagelTransitionRouter(
        token_ids=tensors["token_ids"],
        fc1_weight=tensors["fc1_weight"],
        fc1_bias=tensors["fc1_bias"],
        fc2_weight=tensors["fc2_weight"],
        fc2_bias=tensors["fc2_bias"],
        device=device,
        threshold=artifact_threshold if threshold is None else threshold,
        text_decode_threshold=text_decode_threshold,
        source_path=path,
    )


@dataclass
class TextSegment:
    text: str
    token_ids: list[int]
    stop_reason: str
    stop_token_id: int | None
    stop_logits: torch.Tensor | None
    stop_router: dict[str, Any] | None = None


@dataclass
class NativeInterleaveResult:
    rendered_text: str
    text_segments: list[str]
    images: list[Image.Image]
    events: list[dict[str, Any]]
    stop_reason: str


class BagelNativeInterleaveInferencer:
    """Token-routed BAGEL interleave inference without an action controller.

    A generated ``<|vision_start|>`` token is the only image trigger. The
    generated image is encoded back into the same main context before text
    decoding resumes. BAGEL's text and image CFG contexts remain separate.
    """

    def __init__(
        self,
        inferencer: Any,
        transition_router: BagelTransitionRouter | None = None,
    ):
        self.inferencer = inferencer
        self.model = inferencer.model
        self.tokenizer = inferencer.tokenizer
        self.new_token_ids = inferencer.new_token_ids
        self.device = next(self.model.parameters()).device
        self.transition_router = transition_router

    @torch.no_grad()
    def generate(
        self,
        *,
        prompt: str,
        system_prompt: str = "",
        image_shape: tuple[int, int] = (1024, 1024),
        max_text_segments: int = 8,
        max_images: int = 4,
        max_tokens_per_segment: int = 768,
        text_do_sample: bool = False,
        text_temperature: float = 0.3,
        text_top_p: float = 0.95,
        route_policy: str = "unrestricted_greedy",
        route_temperature: float = 1.0,
        vision_threshold: float = 0.5,
        teacher_force_first_vision_start: bool = False,
        teacher_force_image_to_text: bool = False,
        image_feedback: str = "vae_vit",
        context_strategy: str = "persistent",
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: tuple[float, float] = (0.4, 1.0),
        timestep_shift: float = 3.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        seed: int = 0,
    ) -> NativeInterleaveResult:
        if max_text_segments < 1:
            raise ValueError("max_text_segments must be positive")
        if max_images < 0:
            raise ValueError("max_images must be non-negative")
        if image_feedback not in {"vit", "vae", "vae_vit"}:
            raise ValueError(f"Unsupported image feedback mode: {image_feedback}")
        if context_strategy not in {"persistent", "rebuild"}:
            raise ValueError(f"Unsupported context strategy: {context_strategy}")

        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        main_context = self.inferencer.init_gen_context()
        cfg_img_context = self.inferencer.init_gen_context()
        if system_prompt:
            main_context = self.inferencer.update_context_text(system_prompt, main_context)
            cfg_img_context = self.inferencer.update_context_text(system_prompt, cfg_img_context)
        main_context = self.inferencer.update_context_text(prompt, main_context)
        cfg_img_context = self.inferencer.update_context_text(prompt, cfg_img_context)

        text_segments: list[str] = []
        images: list[Image.Image] = []
        events: list[dict[str, Any]] = []
        rendered_parts: list[str] = []
        history: list[dict[str, Any]] = []
        stop_reason = "text_segment_budget"

        for segment_index in range(max_text_segments):
            cfg_text_context = deepcopy(main_context)
            segment = self._generate_text_segment(
                main_context,
                max_tokens=max_tokens_per_segment,
                do_sample=text_do_sample,
                temperature=text_temperature,
                top_p=text_top_p,
            )
            text_segments.append(segment.text)
            if segment.text:
                rendered_parts.append(segment.text)

            inline_image = segment.stop_token_id == self.new_token_ids["start_of_image"]
            include_eos = not inline_image
            post_logits, post_router = self._append_text_tokens(
                main_context,
                segment.token_ids,
                include_eos=include_eos,
                return_router_decision=True,
            )
            self._append_text_tokens(
                cfg_img_context,
                segment.token_ids,
                include_eos=include_eos,
            )
            history.append(
                {
                    "type": "text",
                    "token_ids": list(segment.token_ids),
                    "include_eos": include_eos,
                }
            )

            if inline_image:
                route = self._route_snapshot(
                    segment.stop_logits,
                    selected_token_id=self.new_token_ids["start_of_image"],
                    policy="inline_generation",
                    source="inline_text_decode",
                )
                if segment.stop_router is not None:
                    route["transition_router"] = segment.stop_router
            elif segment.stop_reason == "eos":
                route = self._select_post_text_route(
                    post_logits,
                    policy=route_policy,
                    temperature=route_temperature,
                    vision_threshold=vision_threshold,
                    router_decision=post_router,
                )
            else:
                route = self._route_snapshot(
                    post_logits,
                    selected_token_id=self.new_token_ids["eos_token_id"],
                    policy="max_tokens_stop",
                    source="text_budget",
                )
                route["host_forced_stop"] = True
                route["model_selected_token_id"] = None
                route["model_selected_token"] = None
                route["model_selected_probability"] = None

            if teacher_force_first_vision_start and segment_index == 0 and not images:
                model_selected_token_id = route.get("model_selected_token_id", route["selected_token_id"])
                model_selected_token = route.get("model_selected_token", route["selected_token"])
                model_selected_probability = route.get("model_selected_probability", route.get("selected_probability"))
                route = {
                    **route,
                    "teacher_forced": True,
                    "model_selected_token_id": model_selected_token_id,
                    "model_selected_token": model_selected_token,
                    "model_selected_probability": model_selected_probability,
                    "selected_token_id": int(self.new_token_ids["start_of_image"]),
                    "selected_token": self._token_text(int(self.new_token_ids["start_of_image"])),
                    "selected_probability": route.get("vision_start_probability"),
                    "policy": "teacher_forced_first_vision_start",
                }

            events.append(
                {
                    "type": "text",
                    "segment_index": segment_index,
                    "text": segment.text,
                    "token_count": len(segment.token_ids),
                    "decode_stop_reason": segment.stop_reason,
                    "route": route,
                    "kv_tokens_after": int(main_context["kv_lens"][0]),
                }
            )

            if route["selected_token_id"] != self.new_token_ids["start_of_image"]:
                stop_reason = f"route:{route['selected_token']}"
                break

            while route["selected_token_id"] == self.new_token_ids["start_of_image"]:
                if len(images) >= max_images:
                    stop_reason = "image_budget"
                    events.append(
                        {
                            "type": "budget_stop",
                            "reason": stop_reason,
                            "requested_image_index": len(images),
                        }
                    )
                    break

                image = self.inferencer.gen_image(
                    image_shape,
                    main_context,
                    cfg_text_precontext=cfg_text_context,
                    cfg_img_precontext=cfg_img_context,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=list(cfg_interval),
                    timestep_shift=timestep_shift,
                    num_timesteps=BAGEL_NATIVE_NUM_TIMESTEPS,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                ).convert("RGB")
                images.append(image)
                rendered_parts.append(f"<image_{len(images)}>")
                history.append({"type": "image", "image": image.copy()})

                main_context, image_hidden = self._append_feedback_image(
                    main_context,
                    image,
                    image_feedback=image_feedback,
                )
                if context_strategy == "rebuild":
                    main_context, image_hidden = self._rebuild_main_context(
                        system_prompt=system_prompt,
                        prompt=prompt,
                        history=history,
                        image_feedback=image_feedback,
                    )
                    cfg_img_context = self._rebuild_cfg_image_context(
                        system_prompt=system_prompt,
                        prompt=prompt,
                        history=history,
                    )

                image_logits = self.model.language_model.lm_head(image_hidden).float()
                image_router = self._router_decision(image_hidden, "image_end")
                image_route = self._select_route(
                    image_logits,
                    policy=route_policy,
                    temperature=route_temperature,
                    vision_threshold=vision_threshold,
                    source="post_image_end",
                    route_site="image_end",
                    router_decision=image_router,
                )
                if teacher_force_image_to_text:
                    model_selected_token_id = image_route["selected_token_id"]
                    model_selected_token = image_route["selected_token"]
                    model_selected_probability = image_route["selected_probability"]
                    bos_id = int(self.new_token_ids["bos_token_id"])
                    image_route = {
                        **image_route,
                        "teacher_forced": True,
                        "model_selected_token_id": model_selected_token_id,
                        "model_selected_token": model_selected_token,
                        "model_selected_probability": model_selected_probability,
                        "selected_token_id": bos_id,
                        "selected_token": self._token_text(bos_id),
                        "selected_probability": image_route.get("bos_probability"),
                        "policy": "teacher_forced_image_to_text",
                    }

                events.append(
                    {
                        "type": "image",
                        "image_index": len(images) - 1,
                        "shape": list(image.size),
                        "num_timesteps": BAGEL_NATIVE_NUM_TIMESTEPS,
                        "feedback": image_feedback,
                        "context_strategy": context_strategy,
                        "route": image_route,
                        "kv_tokens_after": int(main_context["kv_lens"][0]),
                    }
                )

                if image_route["selected_token_id"] == self.new_token_ids["bos_token_id"]:
                    route = image_route
                    break
                if image_route["selected_token_id"] == self.new_token_ids["start_of_image"]:
                    route = image_route
                    cfg_text_context = deepcopy(main_context)
                    continue

                stop_reason = f"image_route:{image_route['selected_token']}"
                route = image_route
                break

            if stop_reason == "image_budget" or route["selected_token_id"] != self.new_token_ids["bos_token_id"]:
                break

        return NativeInterleaveResult(
            rendered_text="\n".join(part for part in rendered_parts if part),
            text_segments=text_segments,
            images=images,
            events=events,
            stop_reason=stop_reason,
        )

    def _generate_text_segment(
        self,
        context: dict[str, Any],
        *,
        max_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> TextSegment:
        local_context = deepcopy(context)
        past_key_values = local_context["past_key_values"]
        kv_len = int(local_context["kv_lens"][0])
        rope = int(local_context["ropes"][0])
        bos_id = int(self.new_token_ids["bos_token_id"])
        eos_id = int(self.new_token_ids["eos_token_id"])
        vision_id = int(self.new_token_ids["start_of_image"])
        current_id = bos_id
        content_ids: list[int] = []
        last_logits: torch.Tensor | None = None

        while True:
            current = torch.tensor([current_id], dtype=torch.long, device=self.device)
            embedding = self.model.language_model.model.embed_tokens(current)
            output = self.model.language_model.forward_inference(
                packed_query_sequence=embedding,
                query_lens=torch.ones(1, dtype=torch.long, device=self.device),
                packed_query_position_ids=torch.tensor([rope], dtype=torch.long, device=self.device),
                packed_query_indexes=torch.tensor([kv_len], dtype=torch.long, device=self.device),
                past_key_values=past_key_values,
                key_values_lens=torch.tensor([kv_len], dtype=torch.int, device=self.device),
                packed_key_value_indexes=torch.arange(kv_len, dtype=torch.long, device=self.device),
                update_past_key_values=True,
                is_causal=True,
                mode="und",
            )
            past_key_values = output.past_key_values
            hidden = output.packed_query_sequence[-1]
            last_logits = self.model.language_model.lm_head(hidden).float()
            router_decision = self._router_decision(hidden, "text_decode")
            if router_decision is not None and router_decision["override"]:
                next_id = int(router_decision["selected_token_id"])
            else:
                next_id = self._sample_token(
                    last_logits,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                )
            kv_len += 1
            rope += 1

            if next_id == eos_id:
                return TextSegment(
                    text=self.tokenizer.decode(content_ids, skip_special_tokens=False),
                    token_ids=content_ids,
                    stop_reason="eos",
                    stop_token_id=next_id,
                    stop_logits=last_logits,
                    stop_router=router_decision
                    if router_decision is not None and router_decision["override"]
                    else None,
                )
            if next_id == vision_id:
                return TextSegment(
                    text=self.tokenizer.decode(content_ids, skip_special_tokens=False),
                    token_ids=content_ids,
                    stop_reason="vision_start",
                    stop_token_id=next_id,
                    stop_logits=last_logits,
                    stop_router=router_decision
                    if router_decision is not None and router_decision["override"]
                    else None,
                )

            content_ids.append(next_id)
            current_id = next_id
            if len(content_ids) >= max_tokens:
                return TextSegment(
                    text=self.tokenizer.decode(content_ids, skip_special_tokens=False),
                    token_ids=content_ids,
                    stop_reason="max_tokens",
                    stop_token_id=None,
                    stop_logits=last_logits,
                )

    def _append_text_tokens(
        self,
        context: dict[str, Any],
        content_ids: list[int],
        *,
        include_eos: bool,
        return_last_hidden: bool = False,
        return_hidden_sequence: bool = False,
        return_router_decision: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any] | None]:
        if sum(
            int(value)
            for value in (
                return_last_hidden,
                return_hidden_sequence,
                return_router_decision,
            )
        ) > 1:
            raise ValueError("Choose only one specialized text append return mode")
        token_ids = [int(self.new_token_ids["bos_token_id"]), *content_ids]
        if include_eos:
            token_ids.append(int(self.new_token_ids["eos_token_id"]))

        kv_len = int(context["kv_lens"][0])
        rope = int(context["ropes"][0])
        ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        embeddings = self.model.language_model.model.embed_tokens(ids)
        output = self.model.language_model.forward_inference(
            packed_query_sequence=embeddings,
            query_lens=torch.tensor([len(token_ids)], dtype=torch.long, device=self.device),
            packed_query_position_ids=torch.arange(
                rope,
                rope + len(token_ids),
                dtype=torch.long,
                device=self.device,
            ),
            packed_query_indexes=torch.arange(
                kv_len,
                kv_len + len(token_ids),
                dtype=torch.long,
                device=self.device,
            ),
            past_key_values=context["past_key_values"],
            key_values_lens=torch.tensor([kv_len], dtype=torch.int, device=self.device),
            packed_key_value_indexes=torch.arange(kv_len, dtype=torch.long, device=self.device),
            update_past_key_values=True,
            is_causal=True,
            mode="und",
        )
        context["past_key_values"] = output.past_key_values
        context["kv_lens"] = [kv_len + len(token_ids)]
        context["ropes"] = [rope + len(token_ids)]
        last_hidden = output.packed_query_sequence[-1]
        if return_hidden_sequence:
            return output.packed_query_sequence
        if return_last_hidden:
            return last_hidden
        logits = self.model.language_model.lm_head(last_hidden).float()
        if return_router_decision:
            return logits, self._router_decision(last_hidden, "text_end")
        return logits

    def _select_post_text_route(
        self,
        logits: torch.Tensor,
        *,
        policy: str,
        temperature: float,
        vision_threshold: float,
        router_decision: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._select_route(
            logits,
            policy=policy,
            temperature=temperature,
            vision_threshold=vision_threshold,
            source="post_text_eos",
            route_site="text_end",
            router_decision=router_decision,
        )

    def _select_route(
        self,
        logits: torch.Tensor,
        *,
        policy: str,
        temperature: float,
        vision_threshold: float,
        source: str,
        route_site: str,
        router_decision: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        vision_id = int(self.new_token_ids["start_of_image"])
        eos_id = int(self.new_token_ids["eos_token_id"])
        bos_id = int(self.new_token_ids["bos_token_id"])
        if router_decision is not None and router_decision["override"]:
            selected = int(router_decision["selected_token_id"])
            selected_policy = "transition_router"
        elif policy == "unrestricted_greedy":
            selected = int(torch.argmax(logits).item())
            selected_policy = policy
        elif policy == "unrestricted_sample":
            selected = self._sample_token(
                logits,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
            )
            selected_policy = policy
        elif policy == "special_greedy":
            candidates = (vision_id, eos_id) if route_site == "text_end" else (bos_id, vision_id, eos_id)
            selected = max(candidates, key=lambda token_id: float(logits[token_id].item()))
            selected_policy = policy
        elif policy == "vision_threshold":
            probabilities = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
            fallback = eos_id if route_site == "text_end" else bos_id
            selected = vision_id if float(probabilities[vision_id].item()) >= vision_threshold else fallback
            selected_policy = policy
        else:
            raise ValueError(f"Unsupported route policy: {policy}")
        snapshot = self._route_snapshot(
            logits,
            selected_token_id=selected,
            policy=selected_policy,
            source=source,
            temperature=temperature,
        )
        if router_decision is not None:
            snapshot["transition_router"] = router_decision
        return snapshot

    def _router_decision(
        self,
        hidden: torch.Tensor,
        route_site: str,
    ) -> dict[str, Any] | None:
        if self.transition_router is None:
            return None
        return self.transition_router.decide(hidden, route_site)

    def _append_feedback_image(
        self,
        context: dict[str, Any],
        image: Image.Image,
        *,
        image_feedback: str,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        from data.data_utils import pil_img2rgb

        resized = self.inferencer.vae_transform.resize_transform(pil_img2rgb(image))
        result = self.inferencer.update_context_image(
            resized,
            context,
            vae=image_feedback in {"vae", "vae_vit"},
            vit=image_feedback in {"vit", "vae_vit"},
            return_last_hidden=True,
        )
        updated_context, last_hidden = result
        if last_hidden is None:
            raise RuntimeError("Image feedback did not return a final hidden state")
        return updated_context, last_hidden

    def _rebuild_main_context(
        self,
        *,
        system_prompt: str,
        prompt: str,
        history: list[dict[str, Any]],
        image_feedback: str,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        context = self.inferencer.init_gen_context()
        if system_prompt:
            context = self.inferencer.update_context_text(system_prompt, context)
        context = self.inferencer.update_context_text(prompt, context)
        last_hidden = None
        for item in history:
            if item["type"] == "text":
                last_hidden = self._append_text_tokens(
                    context,
                    item["token_ids"],
                    include_eos=item["include_eos"],
                    return_last_hidden=True,
                )
            else:
                context, last_hidden = self._append_feedback_image(
                    context,
                    item["image"],
                    image_feedback=image_feedback,
                )
        if last_hidden is None:
            raise RuntimeError("Cannot rebuild an empty interleave history")
        return context, last_hidden

    def _rebuild_cfg_image_context(
        self,
        *,
        system_prompt: str,
        prompt: str,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        context = self.inferencer.init_gen_context()
        if system_prompt:
            context = self.inferencer.update_context_text(system_prompt, context)
        context = self.inferencer.update_context_text(prompt, context)
        for item in history:
            if item["type"] == "text":
                self._append_text_tokens(
                    context,
                    item["token_ids"],
                    include_eos=item["include_eos"],
                )
        return context

    def _route_snapshot(
        self,
        logits: torch.Tensor | None,
        *,
        selected_token_id: int,
        policy: str,
        source: str,
        temperature: float = 1.0,
        top_k: int = 12,
    ) -> dict[str, Any]:
        vision_id = int(self.new_token_ids["start_of_image"])
        eos_id = int(self.new_token_ids["eos_token_id"])
        bos_id = int(self.new_token_ids["bos_token_id"])
        if logits is None:
            return {
                "source": source,
                "policy": policy,
                "selected_token_id": selected_token_id,
                "selected_token": self._token_text(selected_token_id),
                "vision_start_probability": None,
                "vision_start_rank": None,
                "eos_probability": None,
                "bos_probability": None,
                "top_tokens": [],
            }

        scaled = logits / max(temperature, 1e-6)
        probabilities = torch.softmax(scaled, dim=-1)
        k = min(top_k, logits.numel())
        top_values, top_ids = torch.topk(probabilities, k=k)
        vision_logit = logits[vision_id]
        vision_rank = int((logits > vision_logit).sum().item()) + 1
        bos_logit = logits[bos_id]
        eos_logit = logits[eos_id]
        special_ids = {
            vision_id,
            eos_id,
            bos_id,
            int(self.new_token_ids["end_of_image"]),
        }
        masked_logits = logits.clone()
        masked_logits[list(special_ids)] = -torch.inf
        best_text_id = int(torch.argmax(masked_logits).item())
        best_text_logit = logits[best_text_id]
        return {
            "source": source,
            "policy": policy,
            "selected_token_id": selected_token_id,
            "selected_token": self._token_text(selected_token_id),
            "selected_probability": float(probabilities[selected_token_id].item()),
            "vision_start_probability": float(probabilities[vision_id].item()),
            "vision_start_rank": vision_rank,
            "eos_probability": float(probabilities[eos_id].item()),
            "bos_probability": float(probabilities[bos_id].item()),
            "bos_rank": int((logits > bos_logit).sum().item()) + 1,
            "eos_rank": int((logits > eos_logit).sum().item()) + 1,
            "best_text_token_id": best_text_id,
            "best_text_token": self._token_text(best_text_id),
            "best_text_probability": float(probabilities[best_text_id].item()),
            "vision_vs_eos_logit_margin": float((vision_logit - eos_logit).item()),
            "vision_vs_bos_logit_margin": float((vision_logit - bos_logit).item()),
            "vision_vs_best_text_logit_margin": float((vision_logit - best_text_logit).item()),
            "bos_vs_eos_logit_margin": float((bos_logit - eos_logit).item()),
            "bos_vs_best_text_logit_margin": float((bos_logit - best_text_logit).item()),
            "top_tokens": [
                {
                    "token_id": int(token_id),
                    "token": self._token_text(int(token_id)),
                    "probability": float(value),
                }
                for value, token_id in zip(top_values.tolist(), top_ids.tolist())
            ],
        }

    def _sample_token(
        self,
        logits: torch.Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> int:
        if not do_sample:
            return int(torch.argmax(logits).item())

        probabilities = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_ids = torch.sort(probabilities, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative - sorted_probs > top_p
            sorted_probs = sorted_probs.masked_fill(remove, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum().clamp_min(1e-12)
            sampled_index = int(torch.multinomial(sorted_probs, num_samples=1).item())
            return int(sorted_ids[sampled_index].item())
        return int(torch.multinomial(probabilities, num_samples=1).item())

    def _token_text(self, token_id: int) -> str:
        token = self.tokenizer.convert_ids_to_tokens(int(token_id))
        return str(token if token is not None else f"<token:{token_id}>")
