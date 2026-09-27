#!/usr/bin/env python3
"""Build the F01 full-GenEval mixed prompt pool (all six core families).

Method follows flow_grpo's own ``dataset/merge_genevaltask.py``: pick tasks,
assign weights, draw ``weight_i * total`` rows per task, shuffle.  Their weights
tilt hard toward the difficult families (position 41%, counting 29%) and that
tilt is kept.

Three documented deviations from flow_grpo's merge, each for a measured reason:

1. **counting comes from our own narrowed pool, not theirs.**  Their counting
   slice has only 160 distinct prompts (14,706 rows are 160 prompts resampled
   with replacement) and covers targets {2,3,4}.  Ours has 752 distinct prompts
   over targets [2,6] with a verified detector contract.
2. **single_object is added at a small weight.**  flow_grpo excludes it from
   training entirely -- it is nearly saturated -- but we train on all six
   families, so it gets a deliberately small slice.
3. **Rows are deduplicated before sampling.**  flow_grpo samples with
   replacement; here each of the 1000 training steps draws independent
   prompts, so every emitted uid is distinct.

Prompt text reuses flow_grpo's own noun phrases verbatim -- only the leading
``a photo of`` is rewritten into this project's instruction voice -- so the
articles and pluralization stay grammatical without a heuristic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "assets/data/source"
FLOW_GRPO_POOL = SOURCE / "flow_grpo_geneval_train_metadata.jsonl"
COUNTING_TRAIN = SOURCE / "counting_train252.jsonl"
COUNTING_DEV600 = SOURCE / "counting_dev600.jsonl"
COUNTING_FROZEN_DEV100 = SOURCE / "counting_dev100.jsonl"

POOL_VERSION = "clean29529_f01_geneval_pool_v1"
INSTRUCTION_PREFIX = "Create an image with "

# flow_grpo's weights, verbatim, plus the single_object slice.
FLOW_GRPO_WEIGHTS = {
    "position": 0.7,
    "color_attr": 0.3,
    "colors": 0.1,
    "counting": 0.5,
    "two_object": 0.1,
}
# single_object has a hard ceiling of 80 prompts in the world: GenEval defines
# it as one prompt per COCO class.  It therefore gets a fixed small quota rather
# than a weight, and the leftover is redistributed over the weighted families.
SINGLE_OBJECT_TRAIN = 60
FAMILIES = (
    "position", "color_attr", "colors", "counting", "two_object", "single_object",
)
DEV_PER_FAMILY = {
    "position": 50, "color_attr": 50, "colors": 50,
    "counting": 50, "two_object": 50, "single_object": 20,
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def instruction(prompt: str) -> str:
    """Rewrite ``a photo of X`` into this project's instruction voice."""
    text = prompt.strip()
    lowered = text.lower()
    for lead in ("a photo of ", "an photo of "):
        if lowered.startswith(lead):
            text = text[len(lead):]
            break
    else:
        raise ValueError(f"unexpected flow_grpo prompt form: {prompt!r}")
    return f"{INSTRUCTION_PREFIX}{text.strip().rstrip('.')}."


def noun_phrases(prompt: str) -> list[str]:
    """Split ``a carrot and a book`` into grammatical singular noun phrases."""
    body = instruction(prompt)[len(INSTRUCTION_PREFIX):].rstrip(".")
    return [part.strip() for part in body.split(" and ") if part.strip()]


def uid_for(family: str, prompt: str) -> str:
    digest = hashlib.sha256(f"{family}\x00{prompt}".encode()).hexdigest()[:16]
    return f"f01_{family}_{digest}"


def weights() -> dict[str, float]:
    """flow_grpo's weights, normalized.  single_object is quota-based, not
    weighted, so it does not appear here."""
    total = sum(FLOW_GRPO_WEIGHTS.values())
    scaled = {family: value / total for family, value in FLOW_GRPO_WEIGHTS.items()}
    assert abs(sum(scaled.values()) - 1.0) < 1e-9
    return scaled


def distribute(total: int, share: dict[str, float]) -> dict[str, int]:
    """flow_grpo's largest-remainder allocation, verbatim in behaviour."""
    exact = {family: share[family] * total for family in share}
    whole = {family: int(value) for family, value in exact.items()}
    remainder = total - sum(whole.values())
    order = sorted(share, key=lambda f: exact[f] - whole[f], reverse=True)
    for family in order[:remainder]:
        whole[family] += 1
    return whole


def _allocate_within_capacity(
    total: int, share: dict[str, float], capacity: dict[str, int]
) -> dict[str, int]:
    """flow_grpo's weighted allocation, but no family is asked for more rows
    than it distinctly has.

    Deduplication (deviation 3) makes capacity a real constraint: our counting
    pool holds 752 prompts and position holds 20,588, so at flow_grpo's weights
    counting saturates long before position does.  A saturated family is capped
    and its deficit is re-shared among the families that still have room, which
    keeps the requested total exact and every emitted uid distinct.
    """

    remaining = dict(share)
    allocated: dict[str, int] = {}
    budget = total
    while remaining and budget > 0:
        scale = sum(remaining.values())
        draft = distribute(budget, {f: w / scale for f, w in remaining.items()})
        saturated = {f: c for f, c in capacity.items() if f in draft and draft[f] > c}
        if not saturated:
            for family, count in draft.items():
                allocated[family] = allocated.get(family, 0) + count
            break
        for family, cap in saturated.items():
            allocated[family] = cap
            budget -= cap
            remaining.pop(family)
    for family in share:
        allocated.setdefault(family, 0)
    return allocated


def context_for(family: str, include: list[dict]) -> dict:
    """Family context in the shape geneval_family_registry_v1 expects."""
    if family == "single_object":
        return {"object": include[0]["class"]}
    if family == "two_object":
        return {"objects": [row["class"] for row in include]}
    if family == "colors":
        return {"object": include[0]["class"], "color": include[0]["color"]}
    if family == "color_attr":
        return {"attributes": [
            {"object": row["class"], "color": row["color"]} for row in include
        ]}
    if family == "position":
        subject = include[1]
        relation, anchor = subject["position"]
        return {
            "subject": subject["class"],
            "relation": relation,
            "object": include[anchor]["class"],
        }
    raise ValueError(f"no context builder for {family!r}")


def make_row(family: str, prompt_text: str, include: list[dict], split: str) -> dict:
    context = context_for(family, include)
    prompt = instruction(prompt_text)
    return {
        "uid": uid_for(family, prompt_text),
        "prompt": prompt,
        "prompt_text": prompt,
        "user_prompt": prompt,
        # `G016PromptDataset` requires all three of these on every row, and the
        # reward service reads `geneval_metadata` verbatim: its `tag` selects the
        # family and its `include` carries flow_grpo's own clause shape.
        "geneval_metadata": {
            "tag": family,
            "include": include,
            "prompt": prompt,
        },
        "constraints": [{
            "id": f"{family}_v1",
            "kind": family,
            "context": context,
        }],
        "official_geneval_prompt_used": False,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "task": "t2i",
        "family": family,
        "prompt_family": f"geneval_{family}",
        "split": split,
        "source": "flow_grpo_geneval_train_metadata",
        "source_prompt": prompt_text,
        "verifier_registry_key": f"geneval_family_{family}_v1",
        "verifier_context": context,
        "verifier_spec": {
            "family": family,
            "registry_key": f"geneval_family_{family}_v1",
            "context": context,
            "graded": True,
            "terminal_image_only": True,
            "reward_range": [-1.0, 1.0],
        },
        "geneval_include": include,
        "pool_version": POOL_VERSION,
    }


def counting_rows(split: str) -> list[dict]:
    """Reuse our narrowed counting rows unchanged, minus the frozen dev100."""
    frozen = {row["uid"] for row in read_jsonl(COUNTING_FROZEN_DEV100)}
    rows = []
    for source in (COUNTING_TRAIN, COUNTING_DEV600):
        for row in read_jsonl(source):
            if row["uid"] in frozen:
                continue
            row = dict(row)
            row["split"] = split
            row["pool_version"] = POOL_VERSION
            row["verifier_context"] = {
                "object": row["constraints"][0]["class"],
                "count": int(row["target_count"]),
            }
            row["verifier_spec"] = dict(row.get("verifier_spec") or {})
            row["verifier_spec"]["graded"] = True
            rows.append(row)
    return rows


def build(total_train: int, seed: int) -> dict:
    rng = random.Random(seed)
    source = read_jsonl(FLOW_GRPO_POOL)

    # dedup by prompt within family -- flow_grpo resamples with replacement
    by_family: dict[str, dict[str, dict]] = {}
    for row in source:
        by_family.setdefault(row["tag"], {}).setdefault(row["prompt"], row)

    # single_object is synthesised from the grammatical noun phrases that
    # already appear in two_object prompts, so articles stay correct.
    phrases: dict[str, list[dict]] = {}
    for prompt, row in by_family.get("two_object", {}).items():
        for phrase, include in zip(noun_phrases(prompt), row["include"]):
            phrases.setdefault(f"a photo of {phrase}", [{"class": include["class"], "count": 1}])
    by_family["single_object"] = {
        prompt: {"tag": "single_object", "prompt": prompt, "include": include}
        for prompt, include in phrases.items()
    }

    available = {f: sorted(by_family.get(f, {})) for f in FAMILIES if f != "counting"}
    counting = counting_rows("train")

    share = weights()
    capacity = {
        family: max(0, len(available[family]) - DEV_PER_FAMILY[family])
        for family in FAMILIES if family != "counting"
    }
    capacity["counting"] = max(0, len(counting) - DEV_PER_FAMILY["counting"])

    single_train = min(SINGLE_OBJECT_TRAIN, capacity["single_object"])
    train_alloc = _allocate_within_capacity(
        max(0, total_train - single_train), share, capacity
    )
    train_alloc["single_object"] = single_train
    report = {"weights": share, "train_allocation": train_alloc, "available": {}, "shortfall": {}}

    train, dev = [], []
    for family in FAMILIES:
        if family == "counting":
            pool = list(counting)
            rng.shuffle(pool)
            report["available"][family] = len(pool)
            dev_n = DEV_PER_FAMILY[family]
            need = train_alloc[family] + dev_n
            if len(pool) < need:
                report["shortfall"][family] = need - len(pool)
            dev.extend(row | {"split": "dev"} for row in pool[:dev_n])
            train.extend(pool[dev_n:dev_n + train_alloc[family]])
            continue
        prompts = list(available[family])
        rng.shuffle(prompts)
        report["available"][family] = len(prompts)
        dev_n = DEV_PER_FAMILY[family]
        need = train_alloc[family] + dev_n
        if len(prompts) < need:
            report["shortfall"][family] = need - len(prompts)
        chosen_dev = prompts[:dev_n]
        chosen_train = prompts[dev_n:dev_n + train_alloc[family]]
        for prompt in chosen_dev:
            dev.append(make_row(family, prompt, by_family[family][prompt]["include"], "dev"))
        for prompt in chosen_train:
            train.append(make_row(family, prompt, by_family[family][prompt]["include"], "train"))

    rng.shuffle(train)
    rng.shuffle(dev)
    return {"train": train, "dev": dev, "report": report}


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-train", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "assets/data"
    )
    args = parser.parse_args()

    built = build(args.total_train, args.seed)
    train, dev, report = built["train"], built["dev"], built["report"]

    train_uids = {row["uid"] for row in train}
    dev_uids = {row["uid"] for row in dev}
    if train_uids & dev_uids:
        raise SystemExit("train and dev overlap -- refusing to write")
    if len(train_uids) != len(train) or len(dev_uids) != len(dev):
        raise SystemExit("duplicate uid in an emitted split -- refusing to write")

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("dev", dev)):
        path = args.out / f"f01_geneval_{name}.jsonl"
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
            encoding="utf-8",
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        report[f"{name}_rows"], report[f"{name}_sha256"] = len(rows), digest
        print(f"{path.name}: {len(rows)} rows  sha256={digest[:16]}")

    # The trainer binds the pool by the manifest's own sha256 and row count, so
    # the manifest is written HERE, from the same rows, in the same run. Writing
    # it separately guarantees a drift that only fails at launch.
    import collections
    train_sha = report["train_sha256"]
    manifest = {
        "version": "clean29529_f01_geneval_full_prompt_pool_v1",
        "role": (
            "F01 full-GenEval training prompts: six core families, weighted by "
            "flow_grpo's own merge_genevaltask.py"
        ),
        "output_path": _rel(args.out / "f01_geneval_train.jsonl"),
        "output_row_count": len(train),
        "output_sha256": train_sha,
        "output_unique_uid_count": len({r["uid"] for r in train}),
        "family_counts": dict(sorted(collections.Counter(
            r["family"] for r in train).items())),
        "source_flow_grpo_pool": _rel(FLOW_GRPO_POOL),
        "source_counting_pool": _rel(COUNTING_TRAIN),
        "flow_grpo_weights": report["weights"],
        "train_allocation": report["train_allocation"],
        "available_distinct": report["available"],
        "deviations_from_flow_grpo_merge": [
            "counting drawn from our narrowed pool: theirs holds 160 distinct "
            "prompts resampled to 14706 rows over targets {2,3,4}",
            "single_object on a fixed quota: only 80 exist in the world, one "
            "per COCO class",
            "deduplicated before sampling: every step must draw prompts it has "
            "never seen",
        ],
        "steps_supported_without_repetition": len(train) // 2,
        "policy_visible_change": "task text only",
        "system_prompt_changed": False,
        "rows": [
            {"uid": r["uid"], "family": r["family"],
             "prompt_sha256": r.get("prompt_sha256")} for r in train
        ],
    }
    manifest_path = args.out / "F01_PROMPT_POOL_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"{manifest_path.name}: {len(train)} rows  "
          f"sha256={hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:16]}")

    report["pool_version"] = POOL_VERSION
    report["seed"] = args.seed
    (args.out / "F01_POOL_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in ("available", "train_allocation", "shortfall")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
