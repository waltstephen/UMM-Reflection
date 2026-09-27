"""Strict text contract for existing-split all-MSE trajectories."""

from __future__ import annotations

import re
from dataclasses import dataclass


SYSTEM_PROMPT_V1_SHORT = (
    "Inspect the current image against the request. Reason inside one <think> "
    "block\nand choose exactly one [ACTION] edit or done. Do not include "
    "[EDIT] inside\n<think>. For edit, emit the executable instruction as the "
    "next assistant text\nsegment; for done, emit no payload."
)
SYSTEM_PROMPT_V2_FULL_SCHEMA = """Inspect the current image against the request.
Return exactly one controller using this complete schema:
<think>
[CURRENT_ROUND] Round#N
[SCORE] N/10
[ACTION] edit or done
[THINKING] Ground the decision in the visible image and list what is complete,
pending, or incorrect.
[SOURCE_IMAGE] given, None, or Image #N
</think>
Do not emit [EDIT] in the controller. If [ACTION] is edit, end the controller
segment after </think> and emit only the executable edit instruction as the
next assistant text segment. If [ACTION] is done, emit no payload."""
SYSTEM_PROMPT_VERSION = "existing_split_allmse_full_schema_v2"
SYSTEM_PROMPT = SYSTEM_PROMPT_V2_FULL_SCHEMA

_ACTION_FIELD_RE = re.compile(
    r"^[ \t]*\[ACTION\][ \t]+(?P<action>edit|done)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_EDIT_FIELD_RE = re.compile(
    r"^[ \t]*\[EDIT\][ \t]*(?P<payload>.*?)"
    r"(?:\r?\n)?"
    r"(?=^[ \t]*\[[A-Z _-]+\]|[ \t]*</think>[ \t]*$|\Z)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
_NULL_PAYLOADS = {"", "none", "null", "n/a", "na"}


@dataclass(frozen=True)
class ParsedController:
    action: str
    controller: str
    payload: str
    raw_controller: str
    edit_field: str


def parse_controller(text: str) -> ParsedController:
    raw = str(text or "")
    controller = raw.strip()
    if (
        not controller.startswith("<think>")
        or not controller.endswith("</think>")
        or controller.count("<think>") != 1
        or controller.count("</think>") != 1
    ):
        raise ValueError("controller_not_single_think_block")

    action_matches = list(_ACTION_FIELD_RE.finditer(controller))
    if len(action_matches) != 1:
        raise ValueError(f"controller_action_field_count:{len(action_matches)}")
    action = action_matches[0].group("action").lower()

    edit_matches = list(_EDIT_FIELD_RE.finditer(controller))
    if len(edit_matches) != 1:
        raise ValueError(f"controller_edit_field_count:{len(edit_matches)}")
    edit_match = edit_matches[0]
    edit_field = edit_match.group(0)
    payload = edit_match.group("payload").strip()
    split_controller = (
        controller[: edit_match.start()] + controller[edit_match.end() :]
    ).strip()
    if re.search(r"\[EDIT\]", split_controller, flags=re.IGNORECASE):
        raise ValueError("controller_edit_field_leak")

    if action == "edit":
        if payload.lower() in _NULL_PAYLOADS:
            raise ValueError("edit_action_missing_payload")
    elif payload.lower() not in _NULL_PAYLOADS:
        raise ValueError("done_action_has_payload")
    else:
        payload = ""

    return ParsedController(
        action=action,
        controller=split_controller,
        payload=payload,
        raw_controller=controller,
        edit_field=edit_field,
    )


def validate_enriched_metadata(row, image_presence):
    names = (
        "think_list",
        "step_action_list",
        "step_role_list",
        "step_round_list",
        "step_source_image_list",
        "step_image_name_list",
        "step_image_sha256",
        "step_image_size_bytes",
        "step_need_loss",
    )
    values = {name: list(row[name]) for name in names}
    lengths = {name: len(value) for name, value in values.items()}
    if len(set(lengths.values())) != 1 or not values["think_list"]:
        raise ValueError(f"inconsistent_enriched_list_lengths:{lengths}")
    count = len(values["think_list"])
    image_presence = [bool(value) for value in image_presence]
    if len(image_presence) != count:
        raise ValueError("enriched_cache_presence_length_mismatch")

    task = str(row.get("task", "") or "").lower()
    if task not in {"t2i", "edit"}:
        raise ValueError("unknown_task")
    actions = [
        str(value or "").lower()
        for value in values["step_action_list"]
    ]
    roles = [
        str(value or "").lower() for value in values["step_role_list"]
    ]
    rounds = [int(value) for value in values["step_round_list"]]
    source_refs = [
        str(value or "") for value in values["step_source_image_list"]
    ]
    image_names = [
        str(value or "") for value in values["step_image_name_list"]
    ]
    image_sha256 = [
        str(value or "") for value in values["step_image_sha256"]
    ]
    image_sizes = [
        int(value or 0) for value in values["step_image_size_bytes"]
    ]
    old_need_loss = [
        bool(value) for value in values["step_need_loss"]
    ]
    think_list = [str(value or "") for value in values["think_list"]]

    if int(row.get("n_steps", -1)) != count:
        raise ValueError("enriched_n_steps_mismatch")
    if int(row.get("n_images", -1)) != sum(image_presence):
        raise ValueError("enriched_n_images_mismatch")
    expected_source_index = 0 if task == "edit" else -1
    if int(row.get("source_image_step_index", -999)) != expected_source_index:
        raise ValueError("enriched_source_image_step_index_mismatch")

    for step_index, present in enumerate(image_presence):
        name = image_names[step_index]
        sha256 = image_sha256[step_index]
        size = image_sizes[step_index]
        if present:
            if name != f"img_{step_index}.png":
                raise ValueError(
                    f"enriched_image_name_mismatch:{step_index}"
                )
            if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
                raise ValueError(
                    f"enriched_image_sha256_invalid:{step_index}"
                )
            if size <= 0:
                raise ValueError(
                    f"enriched_image_size_invalid:{step_index}"
                )
        elif name or sha256 or size != 0:
            raise ValueError(
                f"enriched_absent_image_metadata_present:{step_index}"
            )

    start_index = 0
    last_image_step = None
    if task == "edit":
        if (
            actions[0] != "none"
            or roles[0] != "source"
            or rounds[0] != 0
            or source_refs[0].lower() != "given"
            or think_list[0]
            or not image_presence[0]
            or old_need_loss[0]
        ):
            raise ValueError("invalid_enriched_edit_source_metadata")
        start_index = 1
        last_image_step = 0
    elif any(
        action == "none" or role == "source"
        for action, role in zip(actions, roles)
    ):
        raise ValueError("enriched_t2i_has_source_metadata")

    parsed_controllers = [None] * count
    done_steps = 0
    generated_steps = []
    for step_index in range(start_index, count):
        relative_round = step_index - start_index
        parsed = parse_controller(think_list[step_index])
        parsed_controllers[step_index] = parsed
        if actions[step_index] != parsed.action:
            raise ValueError(
                f"recorded_action_mismatch:"
                f"{actions[step_index]}!={parsed.action}"
            )
        if rounds[step_index] != relative_round:
            raise ValueError(
                f"enriched_round_mismatch:{step_index}"
            )
        expected_source_ref = (
            "None"
            if last_image_step is None
            else (
                "given"
                if task == "edit" and relative_round == 0
                else f"Image #{last_image_step}"
            )
        )
        if source_refs[step_index] != expected_source_ref:
            raise ValueError(
                f"enriched_source_reference_mismatch:{step_index}"
            )
        if parsed.action == "edit":
            expected_role = "plan" if relative_round == 0 else "edit"
            if roles[step_index] != expected_role:
                raise ValueError(
                    f"enriched_edit_role_mismatch:{step_index}"
                )
            if not image_presence[step_index]:
                raise ValueError(f"edit_missing_image:{step_index}")
            generated_steps.append(step_index)
            last_image_step = step_index
        else:
            done_steps += 1
            if roles[step_index] != "done":
                raise ValueError(
                    f"enriched_done_role_mismatch:{step_index}"
                )
            if image_presence[step_index] or step_index != count - 1:
                raise ValueError(f"malformed_done_pair:{step_index}")

    if done_steps != 1:
        raise ValueError(f"terminal_done_count:{done_steps}")
    if not generated_steps:
        raise ValueError("no_generated_image")
    loss_steps = [
        index for index, value in enumerate(old_need_loss) if value
    ]
    if len(loss_steps) != 1:
        raise ValueError("enriched_final_loss_count_mismatch")
    final_loss_step = int(row.get("final_loss_step_index", -1))
    if (
        loss_steps[0] != final_loss_step
        or final_loss_step != generated_steps[-1]
    ):
        raise ValueError("enriched_final_loss_step_index_mismatch")
    if str(row.get("final_image_name", "") or "") != image_names[
        final_loss_step
    ]:
        raise ValueError("enriched_final_image_name_mismatch")

    return {
        "task": task,
        "think_list": think_list,
        "image_presence": image_presence,
        "old_need_loss": old_need_loss,
        "recorded_actions": actions,
        "recorded_roles": roles,
        "rounds": rounds,
        "source_refs": source_refs,
        "image_names": image_names,
        "image_sha256": image_sha256,
        "image_sizes": image_sizes,
        "parsed_controllers": parsed_controllers,
        "start_index": start_index,
        "source_images": int(task == "edit"),
        "generated_steps": generated_steps,
        "done_steps": done_steps,
        "final_loss_step_index": final_loss_step,
    }
