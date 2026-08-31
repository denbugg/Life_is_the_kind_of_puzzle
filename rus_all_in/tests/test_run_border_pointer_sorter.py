from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path

RUNNER = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/run_border_pointer_sorter.py")
)


def test_collect_declared_filenames_finds_explicit_rosters_only() -> None:
    collect = RUNNER["collect_declared_filenames"]
    assert isinstance(collect, Callable)
    payload = {
        "selection": {
            "fit_filenames": ["img_000001.png"],
            "evaluation_source_filenames": ["img_000002.png"],
            "fit_digest": "not-a-filename.png-but-parent-key-does-not-match",
        },
        "boards": [{"source_filename": "img_000003.png"}],
        "notes": ["img_000004.png"],
    }
    assert collect(payload) == {
        "img_000001.png",
        "img_000002.png",
        "img_000003.png",
    }
