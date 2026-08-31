from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path

import pytest

RUNNER = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/run_fullres_boundary_denoiser.py")
)


def test_recursive_lineage_collector_covers_all_filename_variants() -> None:
    collect = RUNNER["collect_declared_filenames"]
    assert isinstance(collect, Callable)
    payload = {
        "selection": {
            "fit_source_filenames": ["nested/img_000001.png"],
            "confirm_source_filenames": ["img_000002.png"],
            "terminal_filenames": ["img_000003.png"],
        },
        "cases": [{"source_filename": "img_000004.png"}],
        "notes": ["img_000005.png"],
    }
    assert collect(payload) == {
        "img_000001.png",
        "img_000002.png",
        "img_000003.png",
        "img_000004.png",
    }


def test_exact_source_split_accepts_only_disjoint_terminal_pool() -> None:
    validate = RUNNER["validate_source_split"]
    assert isinstance(validate, Callable)
    validate(
        ("train-a.png", "train-b.png"),
        ("eval-a.png",),
        ("terminal-a.png",),
        {"old.png"},
    )
    with pytest.raises(ValueError, match="must be disjoint"):
        validate(
            ("train-a.png",),
            ("train-a.png",),
            ("terminal-a.png",),
            set(),
        )
    with pytest.raises(ValueError, match="excluded"):
        validate(
            ("train-a.png",),
            ("eval-a.png",),
            ("old.png",),
            {"old.png"},
        )
