from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any

RUNNER = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/run_restored_border_ranker_oof.py")
)


def test_exclusion_collector_covers_fit_confirm_and_nested_names() -> None:
    collect = RUNNER["_collect_declared_filenames"]
    assert isinstance(collect, Callable)
    payload: dict[str, Any] = {
        "selection": {
            "fit_source_filenames": ["nested/img_000001.png"],
            "confirm_source_filenames": ["img_000002.png"],
        },
        "boards": [{"source_filename": "img_000003.png"}],
        "notes": ["img_000004.png"],
    }
    assert collect(payload) == {
        "img_000001.png",
        "img_000002.png",
        "img_000003.png",
    }
