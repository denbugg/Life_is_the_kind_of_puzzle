#!/usr/bin/env python3
"""Fit the fixed portable TASKA nonlinear edge calibrator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn

from aiijc_puzzle.taska_nonlinear_calibrator import (
    NONLINEAR_CALIBRATOR_PARAMETERS,
    NONLINEAR_CALIBRATOR_SCHEMA,
    fit_taska_nonlinear_calibrator,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING = (
    PROJECT_ROOT / "outputs/taska-edge-calibrator/train256-v1/training-features.npz"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-nonlinear-calibrator/train256-v1"
TRAINING_SHA256 = "2d1ef6267daab67d74971d625d2d446e7dfb8dc30a6165bd3459ab969e34f373"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    training = args.training.resolve()
    if _sha256(training) != TRAINING_SHA256:
        raise ValueError("training feature artifact SHA-256 mismatch")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "calibrator.npz"
    metadata_path = output / "metadata.json"
    if model_path.exists() or metadata_path.exists():
        raise FileExistsError("refusing to overwrite nonlinear calibrator artifacts")
    with np.load(training, allow_pickle=False) as archive:
        features = archive["features"]
        labels = archive["labels"]
    model = fit_taska_nonlinear_calibrator(features, labels)
    model.save_npz(model_path)
    metadata = {
        "schema": NONLINEAR_CALIBRATOR_SCHEMA,
        "training": {"path": str(training), "sha256": TRAINING_SHA256},
        "parameters": NONLINEAR_CALIBRATOR_PARAMETERS,
        "sklearn_version": sklearn.__version__,
        "training_rows": len(features),
        "positive_fraction": float(labels.mean()),
        "tree_count": len(model.tree_offsets) - 1,
        "node_count": len(model.values),
        "artifact": {"path": str(model_path), "sha256": _sha256(model_path)},
    }
    try:
        with metadata_path.open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {metadata_path}") from error
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
