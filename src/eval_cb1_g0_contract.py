"""P1/CB1 G0: matched-corruption and directed-label contract.

This target-free structural test validates the pre-registered CB1 data geometry
and the existing challenge-matched independent per-tile distortion implementation.
It creates no training example cache, model, score matrix, layout, or target output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from config import BRIGHT, CONTRAST, FS, JPEG_Q, NFRAG, NOISE_SIGMA
from distort import distort_frags

GRID = 24
WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P1_CB1_boundary_buddies\g0_contract")


def directed_neighbour_ids() -> dict[str, np.ndarray]:
    ids = np.arange(NFRAG, dtype=np.int32).reshape(GRID, GRID)
    right = np.full((NFRAG,), -1, dtype=np.int32)
    down = np.full((NFRAG,), -1, dtype=np.int32)
    right[ids[:, :-1].reshape(-1)] = ids[:, 1:].reshape(-1)
    down[ids[:-1, :].reshape(-1)] = ids[1:, :].reshape(-1)
    return {"right": right, "down": down}


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def assert_contract() -> dict[str, object]:
    if (GRID, NFRAG, FS) != (24, 576, 20):
        raise AssertionError(f"fixed tile geometry changed: {(GRID, NFRAG, FS)}")
    if float(BRIGHT) != 30.0:
        raise AssertionError(f"brightness magnitude contract changed: {BRIGHT}")
    if tuple(map(float, CONTRAST)) != (0.70, 1.30):
        raise AssertionError(f"contrast contract changed: {CONTRAST}")
    if tuple(map(float, NOISE_SIGMA)) != (40.0, 55.0):
        raise AssertionError(f"noise contract changed: {NOISE_SIGMA}")
    if tuple(map(int, JPEG_Q)) != (35, 50):
        raise AssertionError(f"JPEG contract changed: {JPEG_Q}")

    # Identical non-constant tile inputs make final per-fragment output variation
    # attributable only to independently sampled corruption parameters/noise/JPEG.
    yy, xx = np.mgrid[0:FS, 0:FS]
    base = np.stack(((17 * xx + 11 * yy) % 256, (7 * xx + 19 * yy) % 256, (29 * xx + 5 * yy) % 256), axis=-1).astype(np.uint8)
    tiles = np.repeat(base[None, :, :, :], NFRAG, axis=0)
    first = distort_frags(tiles, np.random.default_rng(20260814))
    second = distort_frags(tiles, np.random.default_rng(20260815))
    if first.shape != (NFRAG, FS, FS, 3) or first.dtype != np.uint8:
        raise AssertionError("distortion returned malformed fragment array")
    if np.array_equal(first, tiles):
        raise AssertionError("distortion unexpectedly left all tiles unchanged")
    means = first.astype(np.float32).mean(axis=(1, 2, 3))
    if int(np.unique(np.round(means, 3)).size) < 32:
        raise AssertionError("independent per-tile corruption variation was not observed")
    if np.array_equal(first, second):
        raise AssertionError("different RNG seeds yielded the same full distorted bag")

    neighbors = directed_neighbour_ids()
    expected_edges = GRID * (GRID - 1)
    for direction, values in neighbors.items():
        valid = values >= 0
        if int(valid.sum()) != expected_edges:
            raise AssertionError(f"{direction} directed label count is incorrect")
        anchors = np.flatnonzero(valid)
        if np.any(values[anchors] == anchors) or len(np.unique(np.stack((anchors, values[anchors]), axis=1), axis=0)) != expected_edges:
            raise AssertionError(f"{direction} labels contain self or duplicate directed pairs")
    return {
        "distortion_output_sha256": sha256_array(first),
        "second_seed_output_sha256": sha256_array(second),
        "distinct_rounded_tile_means": int(np.unique(np.round(means, 3)).size),
        "right_label_sha256": sha256_array(neighbors["right"]),
        "down_label_sha256": sha256_array(neighbors["down"]),
        "directed_true_neighbours_per_direction": expected_edges,
    }


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    cfg = args()
    observed = assert_contract()
    report = {
        "experiment": "P1_CB1_boundary_buddies",
        "gate": "G0_target_free_corruption_and_label_contract",
        "geometry": {"grid": GRID, "tiles": NFRAG, "tile_px": FS, "fixed_orientation": True},
        "corruption": {
            "brightness": [-float(BRIGHT), float(BRIGHT)], "contrast": list(map(float, CONTRAST)),
            "noise_sigma": list(map(float, NOISE_SIGMA)), "gaussian_blur": "3x3_reflect",
            "jpeg_quality": list(map(int, JPEG_Q)), "independent_per_tile": True,
            "transform_order": "affine_then_noise_then_blur_then_JPEG",
        },
        "observed": observed,
        "targets_opened": False,
        "models_loaded": False,
        "layouts_assembled": False,
        "passes_G0": True,
        "decision": "advance_to_CB1_G1_capacity",
    }
    cfg.work.mkdir(parents=True, exist_ok=True)
    destination = cfg.report or cfg.work / "cb1_g0_report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
