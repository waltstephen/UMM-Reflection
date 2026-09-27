#!/usr/bin/env python3
"""Six-family GenEval reward service for reflection RL.

Scores images over HTTP with the frozen GenEval verifiers (Mask2Former
detector, OpenCLIP colour classifier, BLIP VQA cross-check). ``score`` is the
**graded** q from :mod:`geneval_graded_evidence_v1` rather than a binary pass
flag, because the whole-trajectory reward is built on per-round changes in q
and a binary q makes every process term identically zero.

``strict_correct`` is the frozen registry's own verdict and is never derived
from the graded score, so the pass rate reported here matches official GenEval.

The request body is a pickle: bind to localhost (the default) or a trusted
network only.

Requires ``$GENEVAL_ASSETS_DIR`` (verifier weights) and an ``$HF_HOME`` that
already holds ``bert-base-uncased``, which BLIP loads by name.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageOps  # noqa: E402

from unify_rl.reward_models.g011_stopnow_reward import counting_score  # noqa: E402
from unify_rl.reward_models.geneval_family_registry_v1 import (
    COUNTING_THRESHOLD,
    DETECTOR_THRESHOLD,
    EXPECTED_ASSET_SHA256,
    BLIPVQACrossCheckBackend,
    GenevalFamilyRegistry,
    Mask2FormerGenevalDetector,
    OpenClipGenevalColorBackend,
    VerifierBackends,
)
from unify_rl.reward_models.geneval_graded_evidence_v1 import (
    GRADED_FAMILIES,
    assert_verdict_preserved,
    graded_score,
)

SERVICE_VERSION = "geneval_sixfamily_graded_detector_service_v1"
PROTOCOL_VERSION = "flow_grpo_geneval_pickle_18085_v1"
FAMILIES = ("single_object", "two_object", "counting", "colors", "position", "color_attr")
MAX_BATCH_SIZE = 64
MAX_REQUEST_BYTES = 256 * 1024 * 1024


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def set_offline_environment() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def family_context(metadata: Mapping[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Derive the registry context from flow_grpo's own ``include`` metadata."""

    family = str(metadata.get("tag") or "")
    if family not in FAMILIES:
        raise ValueError(f"service rejects family {family!r}")
    prompt = str(metadata.get("prompt") or "").strip()
    include = [dict(value) for value in metadata.get("include") or []]
    if not prompt or not include:
        raise ValueError("metadata requires a prompt and an include list")

    if family == "single_object":
        if len(include) != 1:
            raise ValueError("single_object requires one include row")
        context: dict[str, Any] = {"object": str(include[0]["class"])}
    elif family == "two_object":
        if len(include) != 2:
            raise ValueError("two_object requires two include rows")
        context = {"objects": [str(row["class"]) for row in include]}
    elif family == "counting":
        if len(include) != 1:
            raise ValueError("counting requires one include row")
        target = int(include[0].get("count", 0))
        if target not in range(2, 7):
            raise ValueError("counting target must be in [2,6]")
        context = {"object": str(include[0]["class"]), "count": target}
    elif family == "colors":
        if len(include) != 1 or "color" not in include[0]:
            raise ValueError("colors requires one coloured include row")
        context = {"object": str(include[0]["class"]), "color": str(include[0]["color"])}
    elif family == "position":
        if len(include) != 2 or "position" not in include[1]:
            raise ValueError("position requires a positioned second include row")
        relation, anchor = include[1]["position"]
        context = {
            "subject": str(include[1]["class"]),
            "relation": str(relation),
            "object": str(include[int(anchor)]["class"]),
        }
    else:  # color_attr
        if len(include) != 2 or any("color" not in row for row in include):
            raise ValueError("color_attr requires two coloured include rows")
        context = {"attributes": [
            {"object": str(row["class"]), "color": str(row["color"])} for row in include
        ]}

    normalized = {
        "tag": family,
        "prompt": prompt,
        "include": include,
        "exclude": [dict(value) for value in metadata.get("exclude") or []],
        "verification_constraints": [
            dict(value) for value in metadata.get("verification_constraints") or []
        ],
    }
    return family, context, normalized


def score_batch(
    registry: GenevalFamilyRegistry,
    detector: Mask2FormerGenevalDetector,
    image_paths: Sequence[str],
    metadatas: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not image_paths or len(image_paths) != len(metadatas):
        raise ValueError("detector image/metadata lengths differ")
    if len(image_paths) > MAX_BATCH_SIZE:
        raise ValueError("detector request exceeds 64 images")

    results: list[dict[str, Any]] = []
    scores: list[float] = []
    strict_rewards: list[float] = []
    by_family: dict[str, list[float]] = {family: [] for family in FAMILIES}
    strict_by_family: dict[str, list[float]] = {family: [] for family in FAMILIES}

    for image_path, metadata in zip(image_paths, metadatas):
        family, context, normalized = family_context(metadata)

        if family == "counting":
            # counting keeps its own frozen graded equation and its own 0.9
            # detector threshold; it is not re-derived here.
            object_name = str(context["object"])
            target = int(context["count"])
            detected = detector.detect(
                image_path, [object_name], threshold=COUNTING_THRESHOLD
            )[object_name]
            observed = len(detected)
            q = counting_score(target, observed)
            strict = observed == target
            detected_counts = {object_name: observed}
            detections = [value.public_dict() for value in detected]
            extra = {
                "target_count": target,
                "count_object": object_name,
                "count_error": observed - target,
                "threshold": COUNTING_THRESHOLD,
            }
        else:
            verified = registry.verify(family, image_path, context)
            strict = bool(verified.passed)
            q = graded_score(
                family=family,
                passed=strict,
                diagnostics=verified.diagnostics,
                context=context,
            )
            assert_verdict_preserved(passed=strict, score=q)
            detected_counts, detections = _presence_summary(family, verified.diagnostics)
            extra = {
                "threshold": DETECTOR_THRESHOLD,
                "verifier_version": verified.verifier_version,
                "binary_raw_score": float(verified.raw_score),
                "diagnostics": dict(verified.diagnostics),
            }

        row = {
            "tag": family,
            "prompt": normalized["prompt"],
            "correct": strict,
            "strict_correct": strict,
            "score": float(q),
            "graded_score": float(q),
            "detected_counts": detected_counts,
            "detections": detections,
            "clause_results": [
                {"clause_id": str(clause.get("id") or family), "passed": strict}
                for clause in normalized["verification_constraints"]
            ] or [{"clause_id": family, "passed": strict}],
            "metadata": normalized,
            "score_field_visible_to_detector": False,
            **extra,
        }
        results.append(row)
        scores.append(float(q))
        strict_rewards.append(float(strict))
        by_family[family].append(float(q))
        strict_by_family[family].append(float(strict))

    return {
        "scores": scores,
        "rewards": strict_rewards,
        "strict_rewards": strict_rewards,
        "group_rewards": {k: v for k, v in by_family.items() if v},
        "group_strict_rewards": {k: v for k, v in strict_by_family.items() if v},
        "results": results,
        "service_version": SERVICE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "gpt_request_count": 0,
    }


def _presence_summary(
    family: str, diagnostics: Mapping[str, Any]
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Per-class detection counts, in the shape the trainer already consumes."""

    counts: dict[str, int] = {}
    flat: list[dict[str, Any]] = []
    if family == "single_object":
        rows = list(diagnostics.get("detections") or [])
        counts[str(diagnostics.get("object") or "")] = len(rows)
        flat.extend(rows)
    elif family == "two_object":
        for name, rows in (diagnostics.get("detections") or {}).items():
            counts[str(name)] = len(rows or [])
            flat.extend(rows or [])
    elif family == "colors":
        detection = diagnostics.get("detection")
        counts[str(diagnostics.get("object") or "")] = 1 if detection else 0
        if detection:
            flat.append(detection)
    elif family == "position":
        for key, name_key in (
            ("subject_detection", "subject"), ("object_detection", "object")
        ):
            detection = diagnostics.get(key)
            counts[str(diagnostics.get(name_key) or key)] = 1 if detection else 0
            if detection:
                flat.append(detection)
    else:  # color_attr
        for row in diagnostics.get("attributes") or []:
            detection = row.get("detection")
            counts[str(row.get("object") or "")] = 1 if detection else 0
            if detection:
                flat.append(detection)
    return counts, flat


def process_payload(
    payload: Mapping[str, Any],
    registry: GenevalFamilyRegistry,
    detector: Mask2FormerGenevalDetector,
) -> dict[str, Any]:
    images = payload.get("images")
    metadatas = payload.get("meta_datas")
    if (
        not isinstance(images, list)
        or not isinstance(metadatas, list)
        or len(images) != len(metadatas)
        or not images
    ):
        raise ValueError("invalid detector payload")
    with tempfile.TemporaryDirectory(prefix="f01_geneval_detector_") as root:
        paths = []
        for index, value in enumerate(images):
            if not isinstance(value, (bytes, bytearray, memoryview)):
                raise TypeError("image payload is not encoded bytes")
            with Image.open(io.BytesIO(bytes(value))) as handle:
                image = ImageOps.exif_transpose(handle).convert("RGB")
            path = Path(root) / f"{index:03d}.png"
            image.save(path)
            paths.append(str(path))
        return score_batch(registry, detector, paths, metadatas)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18092)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--idle-offload", action="store_true")
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--audit-log", type=Path)
    args = parser.parse_args()
    set_offline_environment()

    detector = Mask2FormerGenevalDetector(device=args.device)
    color = OpenClipGenevalColorBackend(device=args.device)
    vqa = BLIPVQACrossCheckBackend(device=args.device)
    registry = GenevalFamilyRegistry(
        backends=VerifierBackends(detector=detector, color=color, vqa=vqa)
    )
    import torch

    execution_lock = threading.Lock()
    resident = [
        ("detector", detector.model),
        ("color", getattr(color, "model", None)),
        ("vqa", getattr(vqa, "model", None)),
    ]

    def move(where: str) -> None:
        for _, module in resident:
            if module is not None:
                module.to(where)
        if where == "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def offload_if_configured() -> None:
        if args.idle_offload:
            move("cpu")

    def score_with_lifecycle(payload: Mapping[str, Any]) -> dict[str, Any]:
        # The policy ranks share this GPU only while scoring is synchronous, so
        # every backend goes back to CPU before the reply.
        with execution_lock:
            if args.idle_offload:
                move(args.device)
            try:
                return process_payload(payload, registry, detector)
            finally:
                offload_if_configured()

    manifest = {
        "ok": True,
        "service_version": SERVICE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "host": args.host,
        "port": args.port,
        "device": args.device,
        "idle_offload": bool(args.idle_offload),
        "idle_device": "cpu" if args.idle_offload else args.device,
        "score_device": args.device,
        "families": list(FAMILIES),
        "graded_families": list(GRADED_FAMILIES),
        "counting_threshold": COUNTING_THRESHOLD,
        "detector_threshold": DETECTOR_THRESHOLD,
        "target_range": [2, 6],
        "score_equation": (
            "counting: 1 exact else 0.5*max(0,1-abs(detected-target)/target); "
            "others: 1 on the frozen GenEval verdict else graded partial in [0,0.5)"
        ),
        "strict_source": "frozen geneval_family_registry_v1 verdict, never derived from the graded score",
        "detector": detector.manifest(),
        "color": color.manifest(),
        "vqa": vqa.manifest(),
        "expected_asset_sha256": EXPECTED_ASSET_SHA256,
        "offline": True,
        "gpt_enabled": False,
        "started_at_utc": utc_now(),
    }
    offload_if_configured()
    if args.manifest_output:
        atomic_json(args.manifest_output, manifest)
    audit_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self.send_error(404)
                return
            body = json.dumps(manifest, sort_keys=True).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in {"/", "/score"}:
                self.send_error(404)
                return
            started = time.monotonic()
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("request body size is invalid")
                payload = pickle.loads(self.rfile.read(length))
                if not isinstance(payload, Mapping):
                    raise TypeError("request payload must be a mapping")
                result = score_with_lifecycle(payload)
                result["latency_sec"] = time.monotonic() - started
                if args.audit_log:
                    row = {
                        "captured_at_utc": utc_now(),
                        "batch_size": len(result["scores"]),
                        "scores": result["scores"],
                        "strict_rewards": result["strict_rewards"],
                        "tags": [r["tag"] for r in result["results"]],
                        "gpt_request_count": 0,
                        "latency_sec": result["latency_sec"],
                    }
                    with audit_lock:
                        args.audit_log.parent.mkdir(parents=True, exist_ok=True)
                        with args.audit_log.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                self._send(result, 200)
            except Exception as exc:
                self._send(
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=20),
                    },
                    500,
                )

        def _send(self, value: Mapping[str, Any], status: int) -> None:
            body = pickle.dumps(dict(value), protocol=pickle.HIGHEST_PROTOCOL)
            self.send_response(status)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    print(json.dumps(manifest, sort_keys=True), flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
