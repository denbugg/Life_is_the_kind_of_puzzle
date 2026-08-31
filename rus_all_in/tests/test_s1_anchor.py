from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch

from aiijc_puzzle.s1_anchor import (
    IMAGE_SIZE,
    S1_TAIL_CONTRACT,
    RestoreNet,
    S1ArtifactPaths,
    assemble_board,
    audit_artifacts,
    canonical_historical_nlm,
    restore_r5,
    split_tiles,
)


def _canvas(seed: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


def test_identity_board_round_trip() -> None:
    image = _canvas()
    board = np.arange(24 * 24)

    assert np.array_equal(assemble_board(image, board), image)
    assert split_tiles(image).shape == (576, 20, 20, 3)


def test_zeroed_restore_net_is_identity_under_uint8_contract() -> None:
    model = RestoreNet(base=32, depth=4)
    for parameter in model.parameters():
        parameter.data.zero_()
    image = _canvas()

    restored = restore_r5(image, model, torch.device("cpu"))

    assert np.array_equal(restored, image)


def test_historical_nlm_matches_direct_opencv_call() -> None:
    image = _canvas(9)
    cv2.setNumThreads(1)
    expected = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    actual = canonical_historical_nlm(image)

    assert S1_TAIL_CONTRACT["opencv_input_contract"].endswith("no_channel_swap")
    assert np.array_equal(actual, expected)


def test_artifact_audit_distinguishes_verified_unpinned_and_missing(tmp_path: Path) -> None:
    ranker = tmp_path / "ranker.pt"
    ranker.write_bytes(b"ranker")
    r5 = tmp_path / "r5.pt"
    r5.write_bytes(b"r5")
    paths = S1ArtifactPaths(
        ranker=ranker,
        affinity_primary=tmp_path / "primary.pt",
        affinity_secondary=tmp_path / "secondary.pt",
        r5=r5,
    )

    report = audit_artifacts(
        paths,
        expected_rank96={"ranker": hashlib.sha256(b"ranker").hexdigest()},
    )

    statuses = {record["role"]: record for record in report["records"]}
    assert statuses["ranker"]["integrity"] == "verified"
    assert statuses["r5"]["integrity"] == "present_unpinned"
    assert statuses["affinity_primary"]["integrity"] == "missing"
    assert not report["exact_s1_ready"]
    assert report["tail_runnable"]
