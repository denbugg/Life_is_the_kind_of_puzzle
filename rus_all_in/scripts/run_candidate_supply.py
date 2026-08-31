"""Measure exact and content-aware neighbour candidate recall on frozen train splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import cv2

from aiijc_puzzle.candidate_supply import (
    DEFAULT_K,
    DEFAULT_RMSE_THRESHOLDS,
    DEFAULT_VIEWS,
    evaluate_board,
    merge_records,
    split_tiles,
)
from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    compute_protocol_digest,
    select_manifest_records,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "candidate-supply" / "calibration48.json"
INPUTS_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "inputs"
TARGETS_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "targets"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", choices=("calibration", "holdout"), default="calibration")
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--sample-seed", type=int, default=20260829)
    parser.add_argument("--sample-namespace", default=EXPERIMENT_SUBSET_NAMESPACE)
    parser.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS))
    parser.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_K))
    parser.add_argument(
        "--rmse-thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_RMSE_THRESHOLDS),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = compute_protocol_digest(manifest)
    if manifest.get("protocol_digest") != expected:
        raise ValueError(f"validation manifest digest mismatch: {path}")
    return manifest


def _load_rgb(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image[:, :, ::-1]


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _print_compact(rows: list[dict[str, object]]) -> None:
    print("emitter scope dir k candidates exact rmse<=10 rmse<=20 rmse<=30 best_rmse")
    for row in rows:
        if row["scope"] != "trusted" or row["k"] not in (1, 5, 32):
            continue
        values = [
            str(row["emitter"]),
            str(row["scope"]),
            str(row["direction"]),
            str(row["k"]),
            f"{float(row['mean_candidates']):.1f}",
            f"{float(row['exact_recall']):.4f}",
            f"{float(row.get('content_recall_rmse_le_10', 0.0)):.4f}",
            f"{float(row.get('content_recall_rmse_le_20', 0.0)):.4f}",
            f"{float(row.get('content_recall_rmse_le_30', 0.0)):.4f}",
            f"{float(row['mean_best_rmse']):.2f}",
        ]
        print(" ".join(values))


def main() -> None:
    args = _parser().parse_args()
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    selected = select_manifest_records(
        manifest,
        args.split,
        limit=args.limit,
        seed=args.sample_seed,
        namespace=args.sample_namespace,
    )
    selection_digest = hashlib.sha256(
        "\n".join(str(record["filename"]) for record in selected).encode()
    ).hexdigest()
    summary = {
        "mode": "run" if args.run else "dry-run",
        "split": args.split,
        "limit": args.limit,
        "views": args.views,
        "ks": args.ks,
        "rmse_thresholds": args.rmse_thresholds,
        "sample_seed": args.sample_seed,
        "sample_namespace": args.sample_namespace,
        "selection_digest": selection_digest,
        "filenames": [record["filename"] for record in selected],
        "protocol_digest": manifest["protocol_digest"],
        "manifest_path": str(manifest_path),
        "output_path": str(args.output.resolve()),
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    started = time.perf_counter()
    record_groups = []
    board_metrics = []
    mappings = []
    board_times = []
    for index, record in enumerate(selected, start=1):
        name = record["filename"]
        input_path = INPUTS_DIR / name
        target_path = TARGETS_DIR / name
        if sha256_file(input_path) != record["input_sha256"]:
            raise ValueError(f"input hash mismatch: {name}")
        if sha256_file(target_path) != record["target_sha256"]:
            raise ValueError(f"target hash mismatch: {name}")
        board_started = time.perf_counter()
        dirty = split_tiles(_load_rgb(input_path))
        clean = split_tiles(_load_rgb(target_path))
        records, mapping = evaluate_board(
            dirty,
            clean,
            views=args.views,
            ks=args.ks,
            rmse_thresholds=args.rmse_thresholds,
        )
        elapsed = time.perf_counter() - board_started
        record_groups.append(records)
        board_metrics.append({"filename": name, "metrics": merge_records([records])})
        mappings.append({"filename": name, **mapping})
        board_times.append(elapsed)
        print(f"{index:03d}/{len(selected):03d} {name} {elapsed:.2f}s", flush=True)

    rows = merge_records(record_groups)
    payload = {
        "schema_version": 1,
        "experiment": "content-aware-candidate-supply",
        "inference_inputs": ["corrupted shuffled tiles"],
        "evaluation_only_inputs": ["clean train target", "target-assisted Hungarian labels"],
        "configuration": summary,
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "boards": len(selected),
            "total_seconds": time.perf_counter() - started,
            "mean_board_seconds": sum(board_times) / len(board_times),
        },
        "mapping_diagnostics": mappings,
        "board_metrics": board_metrics,
        "metrics": rows,
        "code_sha256": {
            "candidate_supply": sha256_file(
                PROJECT_ROOT / "src" / "aiijc_puzzle" / "candidate_supply.py"
            ),
            "runner": sha256_file(Path(__file__).resolve()),
        },
    }
    _write_json_atomic(payload, args.output.resolve())
    _print_compact(rows)
    print(f"saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
