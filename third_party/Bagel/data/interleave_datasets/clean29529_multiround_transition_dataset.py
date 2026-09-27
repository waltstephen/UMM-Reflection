"""Opt-in true multi-round Clean29K controller, transition, and verifier readers."""

from __future__ import annotations

from .base_ema_decoupled_payload_dataset import (
    BasePromptOnlyAnchorMSEIterableDataset,
    Clean29529WeightedControllerIterableDataset,
    _StrictSingleCacheDataset,
)
from ..existing_exact_edit_relocation_contract import parse_relocated_response


SOURCE_SYSTEM_PROMPT_VERSION = "clean29529_full_trajectory_v1"
SYSTEM_PROMPT_VERSION = "clean29529_multiround_external_edit_v1"


def validate_response(
    response: str,
    *,
    task: str,
    round_index: int,
    source_image: str,
):
    return parse_relocated_response(
        response,
        task=task,
        expected_round_index=int(round_index),
        expected_source_image=str(source_image),
    )


class Clean29529MultiroundControllerIterableDataset(
    Clean29529WeightedControllerIterableDataset
):
    """Full persistent-history controller CE from projected metadata."""

    system_prompt_version = SYSTEM_PROMPT_VERSION

    def get_read_columns(self, data_path):
        return [
            "uid",
            "task",
            "trajectory_subtype",
            "user_prompt",
            "system_prompt",
            "response_list",
            "response_action_list",
            "response_round_list",
            "response_source_image_list",
            "response_image_step_list",
            "source_image_step_index",
            "milestone_plan_text",
        ]

    def parse_row_strict(self, row, data_path):
        uid = str(row.get("uid", "") or "")
        task = str(row.get("task", "") or "").lower()
        user_prompt = str(row.get("user_prompt", "") or "")
        system_prompt = str(row.get("system_prompt", "") or "")
        responses = [str(value or "") for value in row["response_list"]]
        actions = [str(value or "").lower() for value in row["response_action_list"]]
        rounds = [int(value) for value in row["response_round_list"]]
        sources = [str(value or "") for value in row["response_source_image_list"]]
        image_steps = [int(value) for value in row["response_image_step_list"]]
        lengths = {
            len(responses),
            len(actions),
            len(rounds),
            len(sources),
            len(image_steps),
        }
        if (
            not uid
            or task not in {"t2i", "edit"}
            or not user_prompt
            or not system_prompt
            or lengths != {len(responses)}
            or not responses
        ):
            raise ValueError("invalid_multiround_controller_row")

        source_step = int(row["source_image_step_index"])
        if source_step != (0 if task == "edit" else -1):
            raise ValueError("multiround_controller_source_step_mismatch")

        data = self._init_data()
        data = self._add_text(data, system_prompt, need_loss=False)
        if task == "edit":
            data = self._add_cached_image(
                data,
                self._cached_image(uid, source_step),
                need_loss=False,
                need_vae=True,
                need_vit=True,
                enable_cfg=False,
            )
        data = self._add_text(data, user_prompt, need_loss=False)

        edit_count = 0
        done_count = 0
        for response, action, round_index, source_image, image_step in zip(
            responses,
            actions,
            rounds,
            sources,
            image_steps,
        ):
            parsed = validate_response(
                response,
                task=task,
                round_index=round_index,
                source_image=source_image,
            )
            if parsed.action != action:
                raise ValueError("multiround_controller_action_mismatch")
            data = self._add_text(
                data,
                response,
                need_loss=True,
                enable_cfg=False,
                ce_loss_weights=self.response_ce_weights(
                    response,
                    task=task,
                    expected_round_index=round_index,
                    expected_source_image=source_image,
                ),
            )
            if action == "edit":
                if image_step < 0:
                    raise ValueError("multiround_controller_edit_without_image")
                data = self._add_cached_image(
                    data,
                    self._cached_image(uid, image_step),
                    need_loss=False,
                    need_vae=True,
                    need_vit=True,
                    enable_cfg=False,
                )
                edit_count += 1
            else:
                if image_step != -1:
                    raise ValueError("multiround_controller_done_with_image")
                done_count += 1

        if done_count != 1 or actions[-1] != "done":
            raise ValueError("multiround_controller_terminal_mismatch")
        if any(
            item["type"] == "vae_image" and item["loss"] == 1
            for item in data["sequence_plan"]
        ):
            raise AssertionError("multiround controller contains image MSE")
        data["_sample_metadata"] = {
            "uid": uid,
            "task": task,
            "source_family": "clean29529_multiround_controller",
            "controller_responses": len(responses),
            "edit_responses": edit_count,
            "done_responses": done_count,
            "milestone_plan_present": bool(row.get("milestone_plan_text")),
            "system_prompt_version": self.system_prompt_version,
        }
        return data


class Clean29529TransitionMSEIterableDataset(_StrictSingleCacheDataset):
    """One payload-conditioned MSE target for every accepted edit action."""

    PAIR_TYPES = {
        "t2i_initial",
        "t2i_transition",
        "edit_transition",
    }

    def get_read_columns(self, data_path):
        return [
            "pair_uid",
            "uid",
            "task",
            "pair_type",
            "payload",
            "source_step_index",
            "target_step_index",
            "target_image_sha256",
        ]

    def parse_row_strict(self, row, data_path):
        pair_uid = str(row.get("pair_uid", "") or "")
        uid = str(row.get("uid", "") or "")
        task = str(row.get("task", "") or "").lower()
        pair_type = str(row.get("pair_type", "") or "")
        payload = str(row.get("payload", "") or "")
        source_step = int(row.get("source_step_index", -1))
        target_step = int(row.get("target_step_index", -1))
        if (
            not pair_uid
            or not uid
            or task not in {"t2i", "edit"}
            or pair_type not in self.PAIR_TYPES
            or not payload
            or target_step < 0
        ):
            raise ValueError("invalid_multiround_transition_pair")

        if pair_type == "t2i_initial":
            if task != "t2i" or source_step != -1:
                raise ValueError("invalid_t2i_initial_route")
        elif pair_type == "t2i_transition":
            if task != "t2i" or source_step < 0:
                raise ValueError("invalid_t2i_transition_route")
        elif task != "edit" or source_step < 0:
            raise ValueError("invalid_external_edit_transition_route")
        if source_step >= target_step:
            raise ValueError("transition_source_not_earlier_than_target")

        data = self._init_data()
        if source_step >= 0:
            data = self._add_cached_image(
                data,
                self._cached_image(uid, source_step),
                need_loss=False,
                need_vae=True,
                need_vit=True,
                enable_cfg=True,
            )
        data = self._add_text(
            data,
            payload,
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
        if any(
            item["type"] == "text" and item["loss"] == 1
            for item in data["sequence_plan"]
        ):
            raise AssertionError("transition MSE contains text CE")
        data["_sample_metadata"] = {
            "uid": pair_uid,
            "task": task,
            "source_family": "clean29529_transition_mse",
            "pair_type": pair_type,
            "source_step_index": source_step,
            "target_step_index": target_step,
        }
        return data


class Clean29529VerifierStateIterableDataset(
    Clean29529WeightedControllerIterableDataset
):
    """Runtime-faithful prefix with CE only on the next response."""

    source_family = "clean29529_verifier_state"

    def get_read_columns(self, data_path):
        return [
            "state_uid",
            "uid",
            "task",
            "trajectory_subtype",
            "user_prompt",
            "system_prompt",
            "prefix_response_list",
            "prefix_round_list",
            "prefix_source_image_list",
            "prefix_image_step_list",
            "target_response",
            "target_action",
            "target_round_index",
            "target_source_image",
            "current_step_index",
            "is_penultimate",
        ]

    def accept_row(self, row) -> bool:
        return True

    def parse_row_strict(self, row, data_path):
        if not self.accept_row(row):
            return {}
        state_uid = str(row.get("state_uid", "") or "")
        uid = str(row.get("uid", "") or "")
        task = str(row.get("task", "") or "").lower()
        user_prompt = str(row.get("user_prompt", "") or "")
        system_prompt = str(row.get("system_prompt", "") or "")
        prefix_responses = [
            str(value or "") for value in row["prefix_response_list"]
        ]
        prefix_rounds = [int(value) for value in row["prefix_round_list"]]
        prefix_sources = [
            str(value or "") for value in row["prefix_source_image_list"]
        ]
        prefix_images = [int(value) for value in row["prefix_image_step_list"]]
        if (
            not state_uid
            or not uid
            or task not in {"t2i", "edit"}
            or not user_prompt
            or not system_prompt
            or not prefix_responses
            or not (
                len(prefix_responses)
                == len(prefix_rounds)
                == len(prefix_sources)
                == len(prefix_images)
            )
        ):
            raise ValueError("invalid_verifier_state_row")

        data = self._init_data()
        data = self._add_text(data, system_prompt, need_loss=False)
        if task == "edit":
            data = self._add_cached_image(
                data,
                self._cached_image(uid, 0),
                need_loss=False,
                need_vae=True,
                need_vit=True,
                enable_cfg=False,
            )
        data = self._add_text(data, user_prompt, need_loss=False)

        for response, round_index, source_image, image_step in zip(
            prefix_responses,
            prefix_rounds,
            prefix_sources,
            prefix_images,
        ):
            parsed = validate_response(
                response,
                task=task,
                round_index=round_index,
                source_image=source_image,
            )
            if parsed.action != "edit" or image_step < 0:
                raise ValueError("verifier_prefix_is_not_edit_image_pair")
            data = self._add_text(
                data,
                response,
                need_loss=False,
                enable_cfg=False,
            )
            data = self._add_cached_image(
                data,
                self._cached_image(uid, image_step),
                need_loss=False,
                need_vae=True,
                need_vit=True,
                enable_cfg=False,
            )

        target = str(row.get("target_response", "") or "")
        target_action = str(row.get("target_action", "") or "").lower()
        target_round = int(row.get("target_round_index", -1))
        target_source = str(row.get("target_source_image", "") or "")
        parsed_target = validate_response(
            target,
            task=task,
            round_index=target_round,
            source_image=target_source,
        )
        if parsed_target.action != target_action:
            raise ValueError("verifier_target_action_mismatch")
        data = self._add_text(
            data,
            target,
            need_loss=True,
            enable_cfg=False,
            ce_loss_weights=self.response_ce_weights(
                target,
                task=task,
                expected_round_index=target_round,
                expected_source_image=target_source,
            ),
        )

        supervised = [
            item
            for item in data["sequence_plan"]
            if item["type"] == "text" and item["loss"] == 1
        ]
        if len(supervised) != 1:
            raise AssertionError("verifier must supervise exactly one response")
        if any(
            item["type"] == "vae_image" and item["loss"] == 1
            for item in data["sequence_plan"]
        ):
            raise AssertionError("verifier state contains image MSE")
        data["_sample_metadata"] = {
            "uid": state_uid,
            "task": task,
            "source_family": self.source_family,
            "target_action": target_action,
            "current_step_index": int(row["current_step_index"]),
            "is_penultimate": bool(row["is_penultimate"]),
        }
        return data


class Clean29529PenultimateVerifierStateIterableDataset(
    Clean29529VerifierStateIterableDataset
):
    """Oversampling view of states with exactly one accepted edit remaining."""

    source_family = "clean29529_penultimate_verifier_state"

    def accept_row(self, row) -> bool:
        return bool(row.get("is_penultimate"))


__all__ = [
    "BasePromptOnlyAnchorMSEIterableDataset",
    "Clean29529MultiroundControllerIterableDataset",
    "Clean29529PenultimateVerifierStateIterableDataset",
    "Clean29529TransitionMSEIterableDataset",
    "Clean29529VerifierStateIterableDataset",
    "SOURCE_SYSTEM_PROMPT_VERSION",
    "SYSTEM_PROMPT_VERSION",
    "validate_response",
]
