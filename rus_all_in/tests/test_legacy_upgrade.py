from __future__ import annotations

import importlib.util
import inspect
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from aiijc_puzzle.legacy_upgrade import (
    atomic_write_png,
    constant_prediction,
    deterministic_submission_zip,
    layout_digest,
    low_frequency_prediction,
    validate_layout,
)
from aiijc_puzzle.protocol import assemble_tiles, split_tiles

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_legacy_upgrade.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("run_legacy_upgrade", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)


def synthetic_image() -> np.ndarray:
    row = np.arange(480, dtype=np.uint16)[:, None]
    column = np.arange(480, dtype=np.uint16)[None, :]
    image = np.empty((480, 480, 3), dtype=np.uint8)
    image[..., 0] = (row + column) % 251
    image[..., 1] = (2 * row + column) % 253
    image[..., 2] = (row + 3 * column) % 255
    return image


def test_constant_median_is_channelwise_and_shuffle_invariant() -> None:
    image = synthetic_image()
    expected_color = np.rint(np.median(image.reshape(-1, 3), axis=0)).astype(np.uint8)
    prediction = constant_prediction(image, statistic="median", per_channel=True)
    assert prediction.shape == image.shape
    assert prediction.dtype == np.uint8
    assert np.array_equal(prediction[0, 0], expected_color)
    assert np.all(prediction == expected_color)

    tiles = split_tiles(image)
    shuffled = assemble_tiles(tiles[np.random.default_rng(7).permutation(len(tiles))])
    assert np.array_equal(
        prediction,
        constant_prediction(shuffled, statistic="median", per_channel=True),
    )


def test_champion_builder_has_no_target_argument_or_hidden_variant() -> None:
    assert tuple(inspect.signature(runner.build_predictions).parameters) == (
        "input_image",
        "suite",
    )
    image = synthetic_image()
    predictions, diagnostics = runner.build_predictions(image, suite="champion")
    assert tuple(predictions) == (runner.CHAMPION_VARIANT,)
    assert np.array_equal(
        predictions[runner.CHAMPION_VARIANT],
        constant_prediction(image, statistic="median", per_channel=True),
    )
    assert diagnostics["layout"] is None


def test_low_frequency_and_layout_contracts() -> None:
    image = synthetic_image()
    blurred = low_frequency_prediction(image, sigma=10.0)
    assert blurred.shape == image.shape
    assert blurred.dtype == np.uint8
    assert not np.array_equal(blurred, image)

    layout = np.arange(576, dtype=np.int32)[::-1]
    assert np.array_equal(validate_layout(layout), layout)
    assert layout_digest(layout) == layout_digest(layout.copy())
    invalid = layout.copy()
    invalid[-1] = invalid[0]
    try:
        validate_layout(invalid)
    except ValueError as error:
        assert "permutation" in str(error)
    else:
        raise AssertionError("duplicate layout was accepted")


def test_submission_zip_is_deterministic_and_root_only(tmp_path: Path) -> None:
    output_dir = tmp_path / "predictions"
    names = ["img_000001.png", "img_000002.png"]
    output_dir.mkdir()
    for index, name in enumerate(names):
        image = np.full((480, 480, 3), 40 + index, dtype=np.uint8)
        atomic_write_png(output_dir / name, image)

    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    first_hash = deterministic_submission_zip(output_dir, names, first)
    second_hash = deterministic_submission_zip(output_dir, names, second)
    assert first_hash == second_hash
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == names
        assert all("/" not in name and "\\" not in name for name in archive.namelist())
        for name in names:
            with archive.open(name) as stream, Image.open(stream) as image:
                assert image.mode == "RGB"
                assert image.size == (480, 480)
