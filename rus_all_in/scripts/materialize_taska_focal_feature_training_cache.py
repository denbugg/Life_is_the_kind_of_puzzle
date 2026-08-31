#!/usr/bin/env python3
"""Materialize the audited train96 22-feature cache for fixed stacker arms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.taska_focal_feature_stacker import stack_taska_focal_features
from aiijc_puzzle.taska_focal_verifier import load_taska_focal_verifier

try:
    from scripts import run_taska_focal_feature_stacker as parent
except ModuleNotFoundError:  # Direct ``python scripts/*.py`` execution.
    import run_taska_focal_feature_stacker as parent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    PROJECT_ROOT
    / "outputs/taska-focal-feature-stacker/train96-v1/training-stacked-features.npz"
)
METADATA = OUTPUT.with_suffix(".json")
CHECKPOINT = PROJECT_ROOT / "artifacts/prior-taska/ckpt/verify_pair_best.pt"


def main() -> None:
    parent._require_frozen_inputs()
    if OUTPUT.exists() or METADATA.exists():
        raise FileExistsError("refusing to overwrite materialized training cache")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_taska_focal_verifier(CHECKPOINT, device=device)
    with (
        np.load(parent.TRAIN_EDGE_ARCHIVE, allow_pickle=False) as edge,
        np.load(parent.TRAIN_FOCAL_ARCHIVE, allow_pickle=False) as focal,
    ):
        offsets = np.asarray(focal["offsets"], dtype=np.int32)
        if not np.array_equal(edge["offsets"][:97], offsets):
            raise RuntimeError("train96 offsets differ")
        stop = int(offsets[-1])
        labels = np.asarray(focal["labels"], dtype=np.uint8)
        if not np.array_equal(labels, edge["labels"][:stop]):
            raise RuntimeError("train96 labels differ")
        sources = np.asarray(focal["source_filenames"])
        if not np.array_equal(sources, edge["source_filenames"][:96]):
            raise RuntimeError("train96 source roster differs")
        edge_features = np.asarray(edge["features"][:stop], dtype=np.float32)
        focal_features = np.asarray(focal["features"], dtype=np.float32)
        patches = np.asarray(focal["patches_uint8"], dtype=np.uint8)
    logits = parent._score_cached_training_patches(
        model,
        patches,
        focal_features,
        device=device,
    )
    stacked = stack_taska_focal_features(edge_features, logits, focal_features)
    parent._write_npz(
        OUTPUT,
        {
            "schema": np.asarray("aiijc-taska-focal-stacked-training-cache-v1"),
            "features": stacked,
            "labels": labels,
            "offsets": offsets,
            "source_filenames": sources,
            "focal_logits": logits,
        },
    )
    parent._write_json(
        METADATA,
        {
            "schema": "aiijc-taska-focal-stacked-training-cache-metadata-v1",
            "board_count": 96,
            "edge_count": len(labels),
            "feature_count": stacked.shape[1],
            "source_names_offsets_and_labels_aligned": True,
            "inference_inputs_target_free": True,
            "labels_offline_training_only": True,
            "artifacts": {
                "cache": parent._record(OUTPUT),
                "edge_source_cache": parent._record(parent.TRAIN_EDGE_ARCHIVE),
                "focal_source_cache": parent._record(parent.TRAIN_FOCAL_ARCHIVE),
                "recovered_checkpoint": parent._record(CHECKPOINT),
                "materializer": parent._record(Path(__file__)),
            },
        },
    )
    print(
        json.dumps(
            {
                "cache": str(OUTPUT.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(OUTPUT),
                "metadata_sha256": sha256_file(METADATA),
                "device": str(device),
            }
        )
    )


if __name__ == "__main__":
    main()

