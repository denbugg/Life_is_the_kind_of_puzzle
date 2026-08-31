from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any

RUNNER = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/run_corruption_border_encoder.py")
)


def test_exclusion_collector_accepts_plural_and_nested_single_filenames() -> None:
    collect = RUNNER["_collect_declared_filenames"]
    assert isinstance(collect, Callable)
    payload = {
        "protocol": {
            "selected_filenames": ["nested/img_000010.png", "img_000011.png"],
            "checkpoint_filenames": ["model.pt"],
        },
        "boards": [
            {"filename": "img_000012.png"},
            {"filename": "not_an_image.json"},
        ],
        "unrelated": "img_000013.png",
    }
    observed = collect(payload)
    assert observed == {
        "img_000010.png",
        "img_000011.png",
        "img_000012.png",
    }


def test_exclusion_collector_does_not_execute_or_infer_undeclared_strings() -> None:
    collect: Callable[..., set[str]] = RUNNER["_collect_declared_filenames"]
    payload: dict[str, Any] = {
        "notes": ["img_000020.png"],
        "path": "img_000021.png",
        "eval_filenames": [],
    }
    assert collect(payload) == set()
