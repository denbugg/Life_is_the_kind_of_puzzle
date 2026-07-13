from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from puzzle_denoise_v2.real_pairs import RealPairSampler, RealPairTable
from puzzle_denoise_v2.tiles import merge_tiles_numpy


def _write_fixture(tmp_path: Path) -> dict:
    data_root = tmp_path / "puzzle"
    input_dir = data_root / "train" / "inputs"
    target_dir = data_root / "train" / "targets"
    test_dir = data_root / "test"
    input_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    names = ["img_000100.png", "img_000101.png"]
    permutations = []
    generator = np.random.default_rng(91)
    for source_index, name in enumerate(names):
        tile_ids = np.arange(576, dtype=np.uint16)
        tiles = np.empty((576, 20, 20, 3), dtype=np.uint8)
        tiles[..., 0] = (tile_ids % 256)[:, None, None]
        tiles[..., 1] = (tile_ids // 256)[:, None, None]
        tiles[..., 2] = ((tile_ids * 13 + source_index * 37) % 256)[:, None, None]
        permutation = generator.permutation(576).astype(np.int64)
        permutations.append(permutation)
        Image.fromarray(merge_tiles_numpy(tiles), mode="RGB").save(target_dir / name)
        Image.fromarray(merge_tiles_numpy(tiles[permutation]), mode="RGB").save(input_dir / name)

    manifest = {
        "schema_version": 1,
        "seed": 123,
        "policy": {"exclude_all_test_filename_overlaps": True},
        "excluded_test_overlap": [],
        "splits": {"train": names, "val": [], "audit": []},
    }
    manifest_path = tmp_path / "splits.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    source_index = np.asarray([0, 0, 0, 0, 1], dtype=np.uint16)
    input_slot = np.asarray([0, 1, 2, 3, 0], dtype=np.uint16)
    clean_tile_index = np.asarray(
        [
            permutations[0][0],
            permutations[0][1],
            permutations[0][2],
            permutations[0][3],
            permutations[1][0],
        ],
        dtype=np.uint16,
    )
    confidence = np.asarray([0.50, 0.80, 0.90, 0.60, 0.95], dtype=np.float32)
    pair_count = len(source_index)
    metadata = {
        "schema_version": 1,
        "kind": "high_purity_real_tile_pairs",
        "manifest_sha256": manifest_sha,
        "split": "train",
        "source_count": len(names),
        "total_tiles": len(names) * 576,
        "selected_pairs": pair_count,
        "selected_coverage": pair_count / (len(names) * 576),
        "thresholds": {
            "coarse_min_margin": 1e-6,
            "structural_min_margin": 1e-6,
            "joint_min_confidence": 0.45,
        },
        "source_name_encoding": "source_names[source_index]",
        "old_q90_used_as_ground_truth": False,
    }
    artifact_path = tmp_path / "real_pairs.npz"
    ones = np.ones(pair_count, dtype=np.uint8)
    np.savez_compressed(
        artifact_path,
        meta=np.asarray(json.dumps(metadata, sort_keys=True)),
        source_names=np.asarray(names),
        source_index=source_index,
        input_slot=input_slot,
        clean_tile_index=clean_tile_index,
        coarse_cost=np.full(pair_count, 0.1, dtype=np.float32),
        structural_cost=np.full(pair_count, 0.2, dtype=np.float32),
        coarse_row_margin=np.full(pair_count, 0.1, dtype=np.float32),
        coarse_column_margin=np.full(pair_count, 0.1, dtype=np.float32),
        structural_row_margin=np.full(pair_count, 0.2, dtype=np.float32),
        structural_column_margin=np.full(pair_count, 0.2, dtype=np.float32),
        joint_confidence=confidence,
        consensus=ones,
        coarse_mutual_cycle=ones,
        structural_mutual_cycle=ones,
        source_consensus_count=np.asarray([4, 1], dtype=np.uint16),
        source_both_mutual_count=np.asarray([4, 1], dtype=np.uint16),
        source_selected_count=np.asarray([4, 1], dtype=np.uint16),
    )
    return {
        "data_root": data_root,
        "manifest_path": manifest_path,
        "artifact_path": artifact_path,
        "permutations": permutations,
    }


def _load(fixture: dict, min_confidence: float = 0.45) -> RealPairTable:
    return RealPairTable.load(
        fixture["artifact_path"],
        manifest_path=fixture["manifest_path"],
        data_root=fixture["data_root"],
        expected_split="train",
        min_confidence=min_confidence,
    )


def test_loader_filters_confidence_and_materializes_same_slot_pairs(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    table = _load(fixture, min_confidence=0.75)
    assert table.active_pair_indices.tolist() == [1, 2, 4]
    assert table.active_source_indices.tolist() == [0, 1]

    sampler = RealPairSampler(table, seed=7, cache_size=1)
    batch = sampler.materialize_validation()
    assert batch.pair_row.tolist() == [1, 2, 4]
    assert batch.source_index.tolist() == [0, 0, 1]
    assert batch.input_slot.tolist() == [1, 2, 0]
    assert batch.clean_tile_index.tolist() == [
        int(fixture["permutations"][0][1]),
        int(fixture["permutations"][0][2]),
        int(fixture["permutations"][1][0]),
    ]
    assert torch.equal(batch.corrupt, batch.clean)
    assert bool((batch.confidence >= 0.75).all())

    # The all-source materialization leaves source 1 in a capacity-one cache.
    before = sampler.cache_info()
    sampler.materialize_validation(source_indices=[0])
    after_miss = sampler.cache_info()
    sampler.materialize_validation(source_indices=[0])
    after_hit = sampler.cache_info()
    assert before.misses == 2 and before.size == 1
    assert after_miss.misses == before.misses + 1
    assert after_hit.hits == after_miss.hits + 1


def test_grouped_sampling_is_seeded_source_uniform_and_pair_uniform(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    table = _load(fixture)
    first = RealPairSampler(table, seed=42, cache_size=2)
    second = RealPairSampler(table, seed=42, cache_size=2)

    first_rows = first.draw_pair_rows(10_000)
    second_rows = second.draw_pair_rows(10_000)
    assert np.array_equal(first_rows, second_rows)

    sampled_sources = table.source_index[first_rows]
    source_zero_fraction = float(np.mean(sampled_sources == 0))
    assert 0.47 < source_zero_fraction < 0.53
    source_zero_rows = first_rows[sampled_sources == 0]
    row_counts = np.asarray([(source_zero_rows == row).sum() for row in table.source_rows(0)])
    assert float(row_counts.max() - row_counts.min()) < 0.15 * float(row_counts.mean())

    # Tensor batches and balanced validation panels are deterministic too.
    batch_a = RealPairSampler(table, seed=99, cache_size=2).sample(32)
    batch_b = RealPairSampler(table, seed=99, cache_size=2).sample(32)
    assert torch.equal(batch_a.pair_row, batch_b.pair_row)
    assert torch.equal(batch_a.corrupt, batch_b.corrupt)
    assert torch.equal(batch_a.clean, batch_b.clean)

    panel_a = first.materialize_validation(pairs_per_source=1, seed=5)
    panel_b = second.materialize_validation(pairs_per_source=1, seed=5)
    assert panel_a.source_index.tolist() == [0, 1]
    assert torch.equal(panel_a.pair_row, panel_b.pair_row)


def test_grouped_batches_use_expected_unique_sources_and_are_deterministic(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    table = _load(fixture)
    first = RealPairSampler(table, seed=314, cache_size=2)
    second = RealPairSampler(table, seed=314, cache_size=2)

    rows_a = first.draw_grouped_pair_rows(batch_size=8, pairs_per_source=4)
    rows_b = second.draw_grouped_pair_rows(batch_size=8, pairs_per_source=4)
    assert len(rows_a) == 8
    assert np.array_equal(rows_a, rows_b)
    sources = table.source_index[rows_a]
    assert len(np.unique(sources)) == 2
    assert sorted(np.bincount(sources, minlength=2).tolist()) == [4, 4]

    batch_a = RealPairSampler(table, seed=2718, cache_size=2).sample_grouped(7, 4)
    batch_b = RealPairSampler(table, seed=2718, cache_size=2).sample_grouped(7, 4)
    assert len(batch_a) == 7
    assert len(torch.unique(batch_a.source_index)) == 2
    assert torch.equal(batch_a.pair_row, batch_b.pair_row)
    assert torch.equal(batch_a.corrupt, batch_b.corrupt)
    assert torch.equal(batch_a.clean, batch_b.clean)


def test_loader_rejects_failed_gate_and_manifest_mismatch(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    with np.load(fixture["artifact_path"], allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    arrays["consensus"][0] = 0
    bad_flags = tmp_path / "bad_flags.npz"
    np.savez_compressed(bad_flags, **arrays)
    fixture["artifact_path"] = bad_flags
    with pytest.raises(ValueError, match="consensus"):
        _load(fixture)

    fixture = _write_fixture(tmp_path / "sha_case")
    fixture["manifest_path"].write_text(
        fixture["manifest_path"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest SHA256"):
        _load(fixture)
