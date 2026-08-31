#!/usr/bin/env python3
"""Audit historical S1 artifacts or replay its exact tail on input-only boards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from aiijc_puzzle.s1_anchor import (
    S1ArtifactPaths,
    apply_s1_tail,
    assemble_board,
    audit_artifacts,
    default_artifact_paths,
    deterministic_zip,
    load_r5_checkpoint,
    load_rgb,
    save_rgb,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    defaults = default_artifact_paths(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranker", type=Path, default=defaults.ranker)
    parser.add_argument("--affinity-primary", type=Path, default=defaults.affinity_primary)
    parser.add_argument("--affinity-secondary", type=Path, default=defaults.affinity_secondary)
    parser.add_argument("--r5-checkpoint", type=Path, default=defaults.r5)
    parser.add_argument("--r5-sha256", default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data/raw/test")
    parser.add_argument(
        "--boards-dir",
        type=Path,
        help="input-only slot-to-tile .npy boards named like each PNG stem",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/s1-anchor/png")
    parser.add_argument(
        "--output-zip", type=Path, default=PROJECT_ROOT / "outputs/s1-anchor/submission.zip"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument(
        "--report", type=Path, default=PROJECT_ROOT / "outputs/s1-anchor/report.json"
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    paths = S1ArtifactPaths(
        ranker=args.ranker,
        affinity_primary=args.affinity_primary,
        affinity_secondary=args.affinity_secondary,
        r5=args.r5_checkpoint,
    )
    audit = audit_artifacts(paths, expected_r5_sha256=args.r5_sha256)
    if args.audit_only or args.boards_dir is None:
        print(json.dumps(audit, indent=2))
        return
    if not args.r5_checkpoint.is_file():
        raise FileNotFoundError(f"R5 checkpoint is required for tail replay: {args.r5_checkpoint}")

    names = sorted(path.name for path in args.input_dir.glob("*.png"))
    if args.limit:
        names = names[: args.limit]
    if not names:
        raise RuntimeError(f"no PNG inputs found in {args.input_dir}")
    device = resolve_device(args.device)
    model = load_r5_checkpoint(args.r5_checkpoint, device)
    rows: list[dict[str, object]] = []
    for ordinal, name in enumerate(names, start=1):
        input_path = args.input_dir / name
        board_path = args.boards_dir / f"{Path(name).stem}.npy"
        if not board_path.is_file():
            raise FileNotFoundError(f"missing input-only board: {board_path}")
        board = np.load(board_path, allow_pickle=False)
        raw_layout = assemble_board(load_rgb(input_path), board)
        r5_layout, final = apply_s1_tail(raw_layout, model, device)
        output_path = args.output_dir / name
        save_rgb(output_path, final)
        rows.append(
            {
                "ordinal": ordinal,
                "name": name,
                "input_sha256": sha256_file(input_path),
                "board_sha256": hashlib.sha256(
                    np.asarray(board, dtype="<i2").tobytes()
                ).hexdigest(),
                "r5_layout_sha256": hashlib.sha256(r5_layout.tobytes()).hexdigest(),
                "output_sha256": sha256_file(output_path),
            }
        )
        print(json.dumps(rows[-1]), flush=True)

    zip_hash = None
    if not args.no_zip and args.limit == 0:
        zip_hash = deterministic_zip(args.output_dir, names, args.output_zip)
    report = {
        "schema": "aiijc-s1-tail-replay-v1",
        "scope": "input-only supplied boards; no clean references, overrides, or targets",
        "artifact_audit": audit,
        "device": str(device),
        "count": len(rows),
        "rows": rows,
        "output_zip": str(args.output_zip) if zip_hash else None,
        "output_zip_sha256": zip_hash,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.report), "count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
