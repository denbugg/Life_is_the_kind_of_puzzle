from __future__ import annotations

import hashlib

import numpy as np

from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    TILE_COUNT,
    TILE_SIZE,
    SplitCounts,
    assemble_tiles,
    assign_splits,
    build_validation_manifest,
    collect_declared_source_filenames,
    compute_protocol_digest,
    contest_ssim,
    select_manifest_records,
    split_name_sets,
    split_tiles,
)


def test_split_is_deterministic_and_disjoint_for_all_7000_names() -> None:
    filenames = [f"img_{index:06d}.png" for index in range(7_000)]
    counts = SplitCounts(train=5_600, calibration=700, holdout=700)

    first = assign_splits(filenames, seed=20260829, counts=counts)
    second = assign_splits(reversed(filenames), seed=20260829, counts=counts)
    other_seed = assign_splits(filenames, seed=20260830, counts=counts)

    assert first == second
    assert first != other_seed
    assert {name: len(values) for name, values in first.items()} == counts.as_dict()
    assert set(first["train"]).isdisjoint(first["calibration"])
    assert set(first["train"]).isdisjoint(first["holdout"])
    assert set(first["calibration"]).isdisjoint(first["holdout"])
    assert set().union(*map(set, first.values())) == set(filenames)


def test_tile_split_and_assembly_round_trip() -> None:
    image = np.arange(IMAGE_SIZE * IMAGE_SIZE * 3, dtype=np.uint32).reshape(
        IMAGE_SIZE, IMAGE_SIZE, 3
    )

    tiles = split_tiles(image)

    assert tiles.shape == (TILE_COUNT, TILE_SIZE, TILE_SIZE, 3)
    assert np.array_equal(tiles[0], image[:TILE_SIZE, :TILE_SIZE])
    assert np.array_equal(tiles[24], image[TILE_SIZE : 2 * TILE_SIZE, :TILE_SIZE])
    assert np.array_equal(assemble_tiles(tiles), image)


def test_contest_ssim_is_one_for_identical_rgb_images() -> None:
    image = np.random.default_rng(7).integers(
        0,
        256,
        size=(IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8,
    )

    assert contest_ssim(image, image) == 1.0


def test_manifest_contains_content_hashes_and_self_verifying_digest(tmp_path) -> None:
    inputs_dir = tmp_path / "inputs"
    targets_dir = tmp_path / "targets"
    inputs_dir.mkdir()
    targets_dir.mkdir()
    contents: dict[str, tuple[bytes, bytes]] = {}
    for index in range(6):
        name = f"img_{index:06d}.png"
        pair = (f"input-{index}".encode(), f"target-{index}".encode())
        contents[name] = pair
        (inputs_dir / name).write_bytes(pair[0])
        (targets_dir / name).write_bytes(pair[1])

    kwargs = {
        "seed": 17,
        "counts": SplitCounts(train=4, calibration=1, holdout=1),
        "expected_pairs": 6,
    }
    first = build_validation_manifest(inputs_dir, targets_dir, **kwargs)
    second = build_validation_manifest(inputs_dir, targets_dir, **kwargs)

    assert first == second
    assert first["protocol_digest"] == compute_protocol_digest(first)
    names = split_name_sets(first)
    assert set().union(*names.values()) == set(contents)
    assert all(
        names[left].isdisjoint(names[right])
        for left, right in (
            ("train", "calibration"),
            ("train", "holdout"),
            ("calibration", "holdout"),
        )
    )
    for records in first["splits"].values():
        for record in records:
            input_bytes, target_bytes = contents[record["filename"]]
            assert record["input_sha256"] == hashlib.sha256(input_bytes).hexdigest()
            assert record["target_sha256"] == hashlib.sha256(target_bytes).hexdigest()


def test_shared_experiment_subset_is_stable_and_split_scoped() -> None:
    records = [{"filename": f"img_{index:06d}.png"} for index in range(20)]
    manifest = {
        "splits": {
            "train": records[:10],
            "calibration": records[10:15],
            "holdout": records[15:],
        }
    }
    selected = select_manifest_records(manifest, "calibration", limit=3)
    reordered = {
        "splits": {
            "train": records[:10],
            "calibration": list(reversed(records[10:15])),
            "holdout": records[15:],
        }
    }
    assert selected == select_manifest_records(reordered, "calibration", limit=3)
    assert {record["filename"] for record in selected} <= {
        record["filename"] for record in records[10:15]
    }
    assert selected != select_manifest_records(
        manifest,
        "calibration",
        limit=3,
        namespace="different-experiment-panel",
    )


def test_report_source_collector_covers_fit_and_confirmation_panels() -> None:
    report = {
        "selection": {
            "train_filenames": ["train.png"],
            "fit_source_filenames": ["fit-a.png", "fit-b.png"],
            "confirm_source_filenames": ["confirm.png"],
        },
        "nested": [{"source_filenames": ["source.png"]}],
        "checkpoint": {"lineage_filenames": ["ancestral-training-only.png"]},
    }
    assert collect_declared_source_filenames(report) == {
        "train.png",
        "fit-a.png",
        "fit-b.png",
        "confirm.png",
        "source.png",
    }
