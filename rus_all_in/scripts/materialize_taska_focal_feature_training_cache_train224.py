#!/usr/bin/env python3
"""Materialize the fixed train224 focal-feature stacker training cache.

The selected organizer-train boards are frozen train256 indices ``0:96`` and
``128:256``.  Indices ``96:128`` are the local gate and are deliberately never
opened while fitting.  The second block is rerun through the verified TASKA
matcher and accepted only when every harvested 15-feature row and exact label
matches the pre-existing train256 cache in the same order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.taska_edge_calibrator import extract_taska_edge_features
from aiijc_puzzle.taska_focal_current_finetune import make_focal_training_board
from aiijc_puzzle.taska_focal_feature_stacker import stack_taska_focal_features
from aiijc_puzzle.taska_pair_pipeline import (
    MATCHER_CONFIG,
    TaskaPairArtifactPaths,
    load_taska_pair_pipeline_resources,
)
from aiijc_puzzle.taska_seam_matcher import match_taska_tiles

try:
    from scripts import run_taska_focal_current_finetune as finetune
    from scripts import run_taska_focal_feature_stacker as train96
except ModuleNotFoundError:  # Direct ``python scripts/*.py`` execution.
    import run_taska_focal_current_finetune as finetune
    import run_taska_focal_feature_stacker as train96

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-focal-feature-stacker/train224-v1"
EDGE_CACHE = PROJECT_ROOT / "outputs/taska-edge-calibrator/train256-v1/training-features.npz"
TRAIN96_CACHE = (
    PROJECT_ROOT
    / "outputs/taska-focal-feature-stacker/train96-v1/training-stacked-features.npz"
)
EXTENSION_ARCHIVE_NAME = "extension128-focal-harvest.npz"
EXTENSION_METADATA_NAME = "extension128-focal-harvest.json"
COMBINED_ARCHIVE_NAME = "training-stacked-features.npz"
COMBINED_METADATA_NAME = "training-stacked-features.json"
EDGE_CACHE_SHA256 = "2d1ef6267daab67d74971d625d2d446e7dfb8dc30a6165bd3459ab969e34f373"
TRAIN96_CACHE_SHA256 = "bb7db2caa09e8305f090da35ace1f9ba11e1c9cc27816c53f324cbf5ab09fc2a"
TRAIN96_COUNT = 96
LOCAL_START = 96
LOCAL_STOP = 128
EXTENSION_START = 128
TRAIN256_COUNT = 256
TRAIN224_COUNT = 224


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=finetune.DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Any) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    try:
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _record(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve().relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(path),
    }


def _edge_arrays(matched: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray([edge.source for edge in matched.candidate_edges], dtype=np.int32)
    target = np.asarray([edge.target for edge in matched.candidate_edges], dtype=np.int32)
    axis = np.asarray(
        [edge.axis == "down" for edge in matched.candidate_edges], dtype=np.uint8
    )
    return source, target, axis


def _edge_digest(source: np.ndarray, target: np.ndarray, axis: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (source, target, axis):
        digest.update(np.asarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _matcher_edge_features(matched: Any) -> np.ndarray:
    records = {record.edge: record for record in matched.vote_records}
    if len(records) != len(matched.vote_records) or set(records) != set(
        matched.candidate_edges
    ):
        raise RuntimeError("matcher vote records and candidate edges differ")
    margins = np.asarray(
        [records[edge].minimum_margin for edge in matched.candidate_edges],
        dtype=np.float64,
    )
    votes = np.asarray(
        [records[edge].vote_count for edge in matched.candidate_edges],
        dtype=np.float64,
    )
    return np.asarray(
        extract_taska_edge_features(
            matched.cost_right,
            matched.cost_down,
            matched.right_log,
            matched.down_log,
            matched.candidate_edges,
            margins,
            votes,
            grid=24,
        ).values,
        dtype=np.float32,
    )


def _combine_offsets(
    first_offsets: np.ndarray,
    extension_offsets: np.ndarray,
) -> np.ndarray:
    first = np.asarray(first_offsets, dtype=np.int64)
    extension = np.asarray(extension_offsets, dtype=np.int64)
    if first.shape != (TRAIN96_COUNT + 1,) or extension.shape != (
        TRAIN256_COUNT - EXTENSION_START + 1,
    ):
        raise ValueError("training offset block sizes differ from fixed train224 split")
    if first[0] != 0 or extension[0] != 0:
        raise ValueError("training offsets must begin at zero")
    if np.any(np.diff(first) <= 0) or np.any(np.diff(extension) <= 0):
        raise ValueError("each selected board must contain harvested edges")
    combined = np.concatenate((first, extension[1:] + first[-1]))
    return np.asarray(combined, dtype=np.int32)


def run(args: argparse.Namespace) -> dict[str, Any]:
    train96._require_frozen_inputs()
    if sha256_file(EDGE_CACHE) != EDGE_CACHE_SHA256:
        raise ValueError("frozen train256 edge cache changed")
    if sha256_file(TRAIN96_CACHE) != TRAIN96_CACHE_SHA256:
        raise ValueError("audited train96 stacked cache changed")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        torch.use_deterministic_algorithms(False)
    else:
        torch.use_deterministic_algorithms(True)
    started = perf_counter()
    config, train_names, local_names = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(config)
    cache = finetune.CleanTileCache(args.targets.resolve(), maximum_boards=2)
    resources = load_taska_pair_pipeline_resources(
        TaskaPairArtifactPaths(), device=args.device
    )

    with (
        np.load(EDGE_CACHE, allow_pickle=False) as edge,
        np.load(TRAIN96_CACHE, allow_pickle=False) as first,
    ):
        source_names = tuple(str(value) for value in edge["source_filenames"])
        draw_indices = np.asarray(edge["draw_indices"], dtype=np.uint8)
        vote_thresholds = np.asarray(edge["vote_thresholds"], dtype=np.int16)
        offsets = np.asarray(edge["offsets"], dtype=np.int64)
        if len(source_names) != TRAIN256_COUNT or offsets.shape != (TRAIN256_COUNT + 1,):
            raise ValueError("frozen train256 cache roster shape changed")
        if source_names[:TRAIN96_COUNT] != train_names:
            raise ValueError("train96 roster differs from the frozen train256 prefix")
        if source_names[LOCAL_START:LOCAL_STOP] != local_names:
            raise ValueError("excluded local32 differs from fixed indices 96:128")
        if not np.array_equal(draw_indices, np.zeros(TRAIN256_COUNT, dtype=np.uint8)):
            raise ValueError("train256 draw-index contract changed")
        first_features = np.asarray(first["features"], dtype=np.float64)
        first_labels = np.asarray(first["labels"], dtype=np.uint8)
        first_logits = np.asarray(first["focal_logits"], dtype=np.float32)
        first_offsets = np.asarray(first["offsets"], dtype=np.int64)
        first_sources = tuple(str(value) for value in first["source_filenames"])
        if first_sources != source_names[:TRAIN96_COUNT]:
            raise ValueError("audited train96 source order changed")
        if not np.array_equal(first_offsets, offsets[: TRAIN96_COUNT + 1]):
            raise ValueError("audited train96 offsets differ from train256")
        if not np.array_equal(first_labels, edge["labels"][: offsets[TRAIN96_COUNT]]):
            raise ValueError("audited train96 labels differ from train256")

        patches: list[np.ndarray] = []
        focal_features: list[np.ndarray] = []
        edge_features: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        edge_sources: list[np.ndarray] = []
        edge_targets: list[np.ndarray] = []
        edge_axes: list[np.ndarray] = []
        extension_offsets = [0]
        rows: list[dict[str, Any]] = []
        extension_started = perf_counter()
        for position, train256_index in enumerate(
            range(EXTENSION_START, TRAIN256_COUNT)
        ):
            name = source_names[train256_index]
            draw = int(draw_indices[train256_index])
            dirty = finetune._dirty_case(cache, lookup[name], name, draw)
            matched = match_taska_tiles(
                dirty.dirty_tiles,
                resources.matchers,
                config=MATCHER_CONFIG,
                device=resources.device,
                require_verified=True,
            )
            actual_features = _matcher_edge_features(matched)
            expected_start = int(offsets[train256_index])
            expected_stop = int(offsets[train256_index + 1])
            expected_features = np.asarray(
                edge["features"][expected_start:expected_stop], dtype=np.float32
            )
            if not np.array_equal(actual_features, expected_features):
                difference = float(np.max(np.abs(actual_features - expected_features)))
                raise RuntimeError(
                    f"train256 edge-feature alignment failed at index {train256_index}; "
                    f"maximum difference {difference}"
                )
            if matched.chosen_vote_threshold != int(vote_thresholds[train256_index]):
                raise RuntimeError(
                    f"train256 vote threshold differs at index {train256_index}"
                )
            reference = finetune._reference(
                cache, lookup[name], name, draw, dirty.dirty_tiles
            )
            board = make_focal_training_board(
                dirty.dirty_tiles,
                matched.cost_right,
                matched.cost_down,
                matched.candidate_edges,
                reference,
                source_filename=name,
            )
            expected_labels = np.asarray(
                edge["labels"][expected_start:expected_stop], dtype=np.uint8
            )
            if not np.array_equal(board.labels, expected_labels):
                raise RuntimeError(
                    f"train256 exact-label alignment failed at index {train256_index}"
                )
            source, target, axis = _edge_arrays(matched)
            patches.append(np.asarray(board.patches, dtype=np.uint8))
            focal_features.append(np.asarray(board.features, dtype=np.float32))
            edge_features.append(actual_features)
            labels.append(expected_labels)
            edge_sources.append(source)
            edge_targets.append(target)
            edge_axes.append(axis)
            extension_offsets.append(extension_offsets[-1] + len(expected_labels))
            rows.append(
                {
                    "extension_position": position,
                    "train256_index": train256_index,
                    "source_filename": name,
                    "draw_index": draw,
                    "dirty_sha256": finetune._dirty_sha256(dirty.dirty_tiles),
                    "edge_count": len(expected_labels),
                    "positive_count": int(expected_labels.sum()),
                    "vote_threshold": int(matched.chosen_vote_threshold),
                    "candidate_edge_digest": _edge_digest(source, target, axis),
                    "edge_features_rowwise_equal_to_train256": True,
                    "labels_rowwise_equal_to_train256": True,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "train224_extension_case_aligned",
                        "case": position + 1,
                        "case_count": TRAIN256_COUNT - EXTENSION_START,
                        "train256_index": train256_index,
                        "source": name,
                        "edges": len(expected_labels),
                    }
                ),
                flush=True,
            )

    patch_matrix = np.concatenate(patches)
    focal_matrix = np.concatenate(focal_features)
    edge_matrix = np.concatenate(edge_features)
    label_vector = np.concatenate(labels)
    source_vector = np.concatenate(edge_sources)
    target_vector = np.concatenate(edge_targets)
    axis_vector = np.concatenate(edge_axes)
    extension_offset_array = np.asarray(extension_offsets, dtype=np.int32)
    focal_logits = train96._score_cached_training_patches(
        resources.focal_verifier,
        patch_matrix,
        focal_matrix,
        device=resources.device,
        chunk_size=4096,
    )
    extension_stacked = stack_taska_focal_features(
        edge_matrix, focal_logits, focal_matrix
    )
    extension_archive = output_dir / EXTENSION_ARCHIVE_NAME
    _write_npz(
        extension_archive,
        {
            "schema": np.asarray("aiijc-taska-focal-train224-extension128-v1"),
            "patches_uint8": patch_matrix,
            "focal_features": focal_matrix,
            "focal_logits": focal_logits,
            "edge_features": edge_matrix,
            "stacked_features": extension_stacked,
            "labels": label_vector,
            "offsets": extension_offset_array,
            "source_filenames": np.asarray(source_names[EXTENSION_START:]),
            "draw_indices": draw_indices[EXTENSION_START:],
            "train256_indices": np.arange(
                EXTENSION_START, TRAIN256_COUNT, dtype=np.int16
            ),
            "edge_source": source_vector,
            "edge_target": target_vector,
            "edge_axis": axis_vector,
        },
    )
    extension_metadata = output_dir / EXTENSION_METADATA_NAME
    _write_json(
        extension_metadata,
        {
            "schema": "aiijc-taska-focal-train224-extension128-metadata-v1",
            "source_count": TRAIN256_COUNT - EXTENSION_START,
            "edge_count": len(label_vector),
            "positive_count": int(label_vector.sum()),
            "runtime_seconds": perf_counter() - extension_started,
            "candidate_alignment": (
                "rowwise exact equality of all 15 matcher features and labels "
                "against frozen train256, plus recorded directed edge identities"
            ),
            "rows": rows,
            "artifacts": {
                "archive": _record(extension_archive),
                "train256_edge_cache": _record(EDGE_CACHE),
                "matcher_v3": _record(TaskaPairArtifactPaths().matcher_v3),
                "matcher_local": _record(TaskaPairArtifactPaths().matcher_local),
                "recovered_focal_checkpoint": _record(
                    TaskaPairArtifactPaths().focal_verifier
                ),
                "materializer": _record(Path(__file__).resolve()),
            },
        },
    )

    combined_features = np.concatenate((first_features, extension_stacked))
    combined_labels = np.concatenate((first_labels, label_vector))
    combined_logits = np.concatenate((first_logits, focal_logits))
    combined_offsets = _combine_offsets(first_offsets, extension_offset_array)
    selected_indices = np.concatenate(
        (
            np.arange(TRAIN96_COUNT, dtype=np.int16),
            np.arange(EXTENSION_START, TRAIN256_COUNT, dtype=np.int16),
        )
    )
    selected_sources = np.asarray(
        source_names[:TRAIN96_COUNT] + source_names[EXTENSION_START:]
    )
    if len(selected_indices) != TRAIN224_COUNT or combined_offsets.shape != (
        TRAIN224_COUNT + 1,
    ):
        raise RuntimeError("combined train224 selection size changed")
    if np.intersect1d(
        selected_indices, np.arange(LOCAL_START, LOCAL_STOP, dtype=np.int16)
    ).size:
        raise RuntimeError("excluded local32 leaked into train224")
    if int(combined_offsets[-1]) != len(combined_labels):
        raise RuntimeError("combined train224 offsets are malformed")
    combined_archive = output_dir / COMBINED_ARCHIVE_NAME
    _write_npz(
        combined_archive,
        {
            "schema": np.asarray("aiijc-taska-focal-stacked-training-cache-train224-v1"),
            "features": combined_features,
            "labels": combined_labels,
            "offsets": combined_offsets,
            "source_filenames": selected_sources,
            "draw_indices": np.zeros(TRAIN224_COUNT, dtype=np.uint8),
            "train256_indices": selected_indices,
            "focal_logits": combined_logits,
        },
    )
    combined_metadata = output_dir / COMBINED_METADATA_NAME
    report = {
        "schema": "aiijc-taska-focal-stacked-training-cache-train224-metadata-v1",
        "board_count": TRAIN224_COUNT,
        "edge_count": len(combined_labels),
        "positive_count": int(combined_labels.sum()),
        "feature_count": int(combined_features.shape[1]),
        "selection": {
            "train256_indices": "0:96 + 128:256",
            "excluded_local32_indices": "96:128",
            "excluded_local32_source_count": len(local_names),
            "excluded_local32_absent": True,
            "competition_test_accessed": False,
        },
        "alignment": {
            "source_names_equal": True,
            "draw_indices_equal": True,
            "offsets_equal": True,
            "edge_features_rowwise_equal": True,
            "labels_rowwise_equal": True,
        },
        "runtime_seconds": perf_counter() - started,
        "artifacts": {
            "combined_cache": _record(combined_archive),
            "extension_cache": _record(extension_archive),
            "extension_metadata": _record(extension_metadata),
            "train96_cache": _record(TRAIN96_CACHE),
            "train256_edge_cache": _record(EDGE_CACHE),
        },
    }
    _write_json(combined_metadata, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run(parse_args())
