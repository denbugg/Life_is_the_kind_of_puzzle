from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from aiijc_puzzle.protocol import compute_protocol_digest
from aiijc_puzzle.synthetic_socket_evaluation import (
    exact_local_retrieval_metrics,
    freeze_topk_candidates,
    load_checkpoint_with_lineage,
    make_exact_synthetic_case,
    names_digest,
    select_source_disjoint_train_records,
)


def _manifest(train_names: list[str]) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": 1,
        "protocol": {"test": True},
        "splits": {
            "train": [
                {
                    "filename": name,
                    "input_sha256": f"input-{name}",
                    "target_sha256": f"target-{name}",
                }
                for name in train_names
            ],
            "calibration": [{"filename": "calibration.png"}],
            "holdout": [{"filename": "holdout.png"}],
        },
    }
    manifest["protocol_digest"] = compute_protocol_digest(manifest)
    return manifest


def _selection(names: list[str], *, lineage: list[str] | None = None) -> dict[str, object]:
    selection: dict[str, object] = {
        "train_filenames": names,
        "train_digest": names_digest(names),
    }
    if lineage is not None:
        selection["lineage_train_filenames"] = lineage
        selection["lineage_train_digest"] = names_digest(lineage, sort_names=True)
    return selection


def test_checkpoint_lineage_recurses_and_verifies_declared_union(tmp_path: Path) -> None:
    ancestor = tmp_path / "ancestor.pt"
    current = tmp_path / "current.pt"
    torch.save(
        {
            "selection": _selection(["a.png", "b.png"]),
            "continued_from": None,
        },
        ancestor,
    )
    torch.save(
        {
            "selection": _selection(
                ["c.png"],
                lineage=["a.png", "b.png", "c.png"],
            ),
            "continued_from": str(ancestor),
        },
        current,
    )

    payload, lineage = load_checkpoint_with_lineage(current, project_root=tmp_path)

    assert payload["continued_from"] == str(ancestor)
    assert lineage.filenames == ("a.png", "b.png", "c.png")
    assert lineage.checkpoint_paths == (str(ancestor.resolve()), str(current.resolve()))


def test_checkpoint_lineage_fails_closed_on_bad_digest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bad.pt"
    torch.save(
        {
            "selection": {
                "train_filenames": ["a.png"],
                "train_digest": "0" * 64,
            },
            "continued_from": None,
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="digest"):
        load_checkpoint_with_lineage(checkpoint, project_root=tmp_path)


def test_source_selection_uses_only_train_and_excludes_lineage() -> None:
    manifest = _manifest([f"train-{index}.png" for index in range(8)])
    selected = select_source_disjoint_train_records(
        manifest,
        excluded_filenames=["train-0.png", "train-3.png"],
        limit=5,
        seed=19,
    )
    names = {record["filename"] for record in selected}
    assert len(names) == 5
    assert not names & {"train-0.png", "train-3.png"}
    assert "calibration.png" not in names
    assert "holdout.png" not in names

    tampered = dict(manifest)
    tampered["protocol_digest"] = "bad"
    with pytest.raises(ValueError, match="protocol digest"):
        select_source_disjoint_train_records(
            tampered,
            excluded_filenames=[],
            limit=1,
            seed=19,
        )


def test_exact_synthetic_case_is_deterministic_and_keeps_label_separate() -> None:
    clean = np.zeros((16, 20, 20, 3), dtype=np.uint8)
    for tile in range(16):
        clean[tile] = 8 + 14 * tile

    first_input, first_reference = make_exact_synthetic_case(
        clean,
        source_filename="source.png",
        draw_index=0,
        seed=23,
    )
    repeat_input, repeat_reference = make_exact_synthetic_case(
        clean,
        source_filename="source.png",
        draw_index=0,
        seed=23,
    )
    other_input, other_reference = make_exact_synthetic_case(
        clean,
        source_filename="source.png",
        draw_index=1,
        seed=23,
    )

    assert first_input.tiles.shape == (16, 20, 20, 3)
    assert first_input.tiles.dtype == np.uint8
    assert np.array_equal(first_input.tiles, repeat_input.tiles)
    assert np.array_equal(
        first_reference.tile_at_position,
        repeat_reference.tile_at_position,
    )
    assert np.array_equal(np.sort(first_reference.tile_at_position), np.arange(16))
    assert first_input.case_id == first_reference.case_id
    assert other_input.case_id == other_reference.case_id
    assert other_input.case_id != first_input.case_id
    assert not np.array_equal(other_input.tiles, first_input.tiles)


def test_frozen_topk_and_exact_local_metrics_recover_known_neighbours() -> None:
    grid = 4
    count = grid * grid
    reference = np.random.default_rng(7).permutation(count).astype(np.int32)
    right_candidates = np.tile(np.arange(count, dtype=np.int32), (count, 1))
    down_candidates = right_candidates.copy()
    for position, anchor in enumerate(reference):
        if position % grid != grid - 1:
            truth = reference[position + 1]
            right_candidates[anchor, 0], right_candidates[anchor, truth] = (
                right_candidates[anchor, truth],
                right_candidates[anchor, 0],
            )
        if position < count - grid:
            truth = reference[position + grid]
            down_candidates[anchor, 0], down_candidates[anchor, truth] = (
                down_candidates[anchor, truth],
                down_candidates[anchor, 0],
            )

    metrics = exact_local_retrieval_metrics(
        right_candidates,
        down_candidates,
        reference,
        ks=(1, 3),
    )
    assert metrics["right_r1"] == 1.0
    assert metrics["down_r1"] == 1.0
    assert metrics["pooled_r1"] == 1.0

    scores = np.zeros((count, count), dtype=np.float32)
    np.fill_diagonal(scores, -100.0)
    frozen = freeze_topk_candidates(scores, max_k=3)
    assert frozen.shape == (count, 3)
    assert frozen.dtype == np.int32
    assert frozen[0].tolist() == [1, 2, 3]

