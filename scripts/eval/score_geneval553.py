#!/usr/bin/env python3
"""Score a GenEval-553 generation run with the six-family GenEval verifiers.

Runs in the GenEval environment (mmdet / open_clip), not the trainer one:
generation needs torch 2.5 and the detector needs mmcv, so the two stages are
separate processes that meet only through ``result.json`` and the saved JPEGs.

Every round image of every sample is scored, so the report carries the full
test-time-scaling curve. A trajectory that stopped at round j keeps its round-j
image for every later budget k (``rounds[min(k, j)]``), so the denominator is
all 553 prompts at every k. ``accuracy_macro`` is the GenEval convention (mean
over the six families); ``accuracy_micro`` is the plain mean over prompts.

``*_parse_gated`` additionally counts a k >= 1 image as wrong when any
controller turn of the trajectory failed to parse (R0 is never gated). The
paper's tables use the parse-gated macro numbers; for a trained policy the two
agree to within a few prompts, but for base BAGEL, which rarely follows the
controller format, the gate turns its reflection curve sharply negative.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval/score_geneval553.py \\
      --run-dir outputs/geneval553/rl-checkpoint-1000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from serve_clean29529_f01_geneval import (  # noqa: E402
    FAMILIES,
    MAX_BATCH_SIZE,
    SERVICE_VERSION,
    BLIPVQACrossCheckBackend,
    GenevalFamilyRegistry,
    Mask2FormerGenevalDetector,
    OpenClipGenevalColorBackend,
    VerifierBackends,
    score_batch,
    set_offline_environment,
)

EXPECTED_ROWS = 553
FAMILY_SIZES = {
    "single_object": 80,
    "two_object": 99,
    "counting": 80,
    "colors": 94,
    "position": 100,
    "color_attr": 100,
}
MAX_BUDGET = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def load_results(samples_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows = []
    for result_path in sorted(samples_root.glob("*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") == "complete":
            rows.append((result_path.parent, result))
    return rows


def score_pending(
    rows: list[tuple[Path, dict[str, Any]]], *, device: str, rescore: bool
) -> None:
    pending = [
        (sample_dir, result)
        for sample_dir, result in rows
        if rescore or not (sample_dir / "scores.json").is_file()
    ]
    if not pending:
        return
    set_offline_environment()
    detector = Mask2FormerGenevalDetector(device=device)
    registry = GenevalFamilyRegistry(
        backends=VerifierBackends(
            detector=detector,
            color=OpenClipGenevalColorBackend(device=device),
            vqa=BLIPVQACrossCheckBackend(device=device),
        )
    )
    for done, (sample_dir, result) in enumerate(pending, start=1):
        paths = []
        for binding in result["image_bindings"]:
            path = sample_dir / binding["path"]
            if sha256_file(path) != binding["sha256"]:
                raise SystemExit(f"image changed after generation: {path}")
            paths.append(str(path))
        if len(paths) > MAX_BATCH_SIZE:
            raise SystemExit(f"{sample_dir.name}: more than {MAX_BATCH_SIZE} rounds")
        metadata = dict(result["geneval_metadata"])
        out = score_batch(registry, detector, paths, [metadata] * len(paths))
        atomic_json(
            sample_dir / "scores.json",
            {
                "sample_id": result["sample_id"],
                "slot": int(result["slot"]),
                "family": result["family"],
                "parse_valid": bool(result.get("parse_valid", True)),
                "done": result.get("done"),
                "rounds": [
                    {"exact": bool(strict), "q": float(q)}
                    for strict, q in zip(out["strict_rewards"], out["scores"])
                ],
                "service_version": SERVICE_VERSION,
            },
        )
        if done % 25 == 0 or done == len(pending):
            print(f"scored {done}/{len(pending)}", flush=True)


def held(rounds: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    return rounds[min(budget, len(rounds) - 1)]


def exact_at(row: dict[str, Any], budget: int, *, parse_gated: bool) -> bool:
    exact = bool(held(row["rounds"], budget)["exact"])
    if parse_gated and budget > 0 and not row["parse_valid"]:
        return False
    return exact


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = {family: [] for family in FAMILIES}
    for row in scored:
        by_family[row["family"]].append(row)

    def accuracy(rows: list[dict[str, Any]], budget: int, gated: bool) -> float:
        return sum(exact_at(r, budget, parse_gated=gated) for r in rows) / len(rows)

    curve = []
    for budget in range(MAX_BUDGET + 1):
        point: dict[str, Any] = {"edits": budget}
        for gated, suffix in ((False, ""), (True, "_parse_gated")):
            per_family = {
                family: accuracy(rows, budget, gated)
                for family, rows in by_family.items()
                if rows
            }
            point[f"accuracy_macro{suffix}"] = sum(per_family.values()) / len(per_family)
            point[f"accuracy_micro{suffix}"] = accuracy(scored, budget, gated)
            point[f"per_family{suffix}"] = per_family
        point["mean_q"] = sum(held(r["rounds"], budget)["q"] for r in scored) / len(scored)
        curve.append(point)

    wrong_at_r0 = [r for r in scored if not r["rounds"][0]["exact"]]
    right_at_r0 = [r for r in scored if r["rounds"][0]["exact"]]
    repaired = sum(1 for r in wrong_at_r0 if r["rounds"][-1]["exact"])
    damaged = sum(1 for r in right_at_r0 if not r["rounds"][-1]["exact"])
    return {
        "row_count": len(scored),
        "r0": curve[0],
        "final": curve[-1],
        "by_budget": curve,
        "repair": {
            "wrong_at_r0": len(wrong_at_r0),
            "repaired": repaired,
            "repair_rate": repaired / len(wrong_at_r0) if wrong_at_r0 else None,
        },
        "damage": {
            "right_at_r0": len(right_at_r0),
            "damaged": damaged,
            "damage_rate": damaged / len(right_at_r0) if right_at_r0 else None,
        },
        "protocol_valid_rate": sum(r["parse_valid"] for r in scored) / len(scored),
        "mean_edits": sum(len(r["rounds"]) - 1 for r in scored) / len(scored),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="directory written by generate_geneval553.py (holds samples/)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="summarize whatever is complete instead of requiring all 553 rows",
    )
    args = parser.parse_args()

    samples_root = args.run_dir / "samples"
    rows = load_results(samples_root)
    if not args.allow_partial and len(rows) != EXPECTED_ROWS:
        raise SystemExit(
            f"{len(rows)}/{EXPECTED_ROWS} samples complete under {samples_root}; "
            "finish generation or pass --allow-partial"
        )
    score_pending(rows, device=args.device, rescore=args.rescore)

    scored = [
        json.loads((sample_dir / "scores.json").read_text(encoding="utf-8"))
        for sample_dir, _ in rows
    ]
    if not args.allow_partial:
        counts = {family: 0 for family in FAMILIES}
        for row in scored:
            counts[row["family"]] += 1
        if counts != FAMILY_SIZES:
            raise SystemExit(f"family sizes differ from GenEval-553: {counts}")

    summary = summarize(scored)
    summary["run"] = args.run_dir.name
    summary["partial"] = len(rows) != EXPECTED_ROWS
    atomic_json(args.run_dir / "summary.json", summary)

    print(f"\n{args.run_dir.name}  (n={summary['row_count']})")
    header = "".join(f"{family[:10]:>12}" for family in FAMILIES)
    print(f"{'edits':<6}{header}{'macro':>9}{'micro':>9}{'gated':>9}")
    for point in summary["by_budget"]:
        cells = "".join(
            f"{point['per_family'].get(family, float('nan')):>12.3f}" for family in FAMILIES
        )
        print(
            f"{point['edits']:<6}{cells}"
            f"{point['accuracy_macro']:>9.4f}{point['accuracy_micro']:>9.4f}"
            f"{point['accuracy_macro_parse_gated']:>9.4f}"
        )
    print("gated = parse-gated macro (the paper's convention)")
    repair, damage = summary["repair"], summary["damage"]
    print(
        f"repaired {repair['repaired']}/{repair['wrong_at_r0']}, "
        f"damaged {damage['damaged']}/{damage['right_at_r0']}, "
        f"protocol valid {summary['protocol_valid_rate']:.3f}, "
        f"mean edits {summary['mean_edits']:.2f}"
    )
    print(f"wrote {args.run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
