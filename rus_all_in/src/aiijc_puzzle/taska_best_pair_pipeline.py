"""Confirmed production adapter for the best pair-oriented TASKA layout solver.

This layout-only entrypoint wraps the unchanged selective-target500 solver that
passed a preregistered, source-disjoint 16-source x 2-draw confirmation.  It
accepts exactly the 576 original upright RGB tiles and emits only a strict
``tile_at_position`` permutation.  It never renders or changes pixels.

The legacy :mod:`aiijc_puzzle.taska_pair_pipeline` remains untouched.  Its
resource loader is reused here so every matcher, verifier, calibrator, and the
raw solver are SHA-256 gated before deserialization.  This adapter additionally
byte-gates the confirmed selective solver module itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import aiijc_puzzle.taska_selective_vote500 as selective_vote500
from aiijc_puzzle.taska_pair_pipeline import (
    EXPECTED_ARTIFACT_SHA256,
    TILE_COUNT,
    TaskaPairPipelineResources,
    load_taska_pair_pipeline_resources,
)

SELECTIVE_SOLVER_SHA256 = "8bb23f6ff6402bfde3a2ec8701ea8ddffff86711fbd71e48993eb6d29a8e1fbc"
CONFIRMATION_CONFIG_SHA256 = "181d562e2d3cc337404608508e8fca6c25bbebf89d584e67744d30a961656628"
CONFIRMATION_REPORT_SHA256 = "981d2ac218671bee4faaae090e24ebddaf7f075d1129ad9e562d218eec12bfc4"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_taska_best_pair_solver() -> str:
    """Fail closed if the confirmed selective solver source changed."""

    path = Path(selective_vote500.__file__).resolve()
    digest = _sha256_file(path)
    if digest != SELECTIVE_SOLVER_SHA256:
        raise RuntimeError(
            "confirmed selective solver SHA-256 mismatch: "
            f"expected {SELECTIVE_SOLVER_SHA256}, got {digest}"
        )
    return digest


@dataclass(frozen=True)
class TaskaBestPairPipelineResult:
    """Strict layout and compact target-free provenance for one board."""

    layout: np.ndarray
    selected_arm: str
    costs: tuple[tuple[str, float], ...]
    diagnostics: Mapping[str, Any]
    artifact_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        raw = np.asarray(self.layout)
        if raw.shape != (TILE_COUNT,) or raw.dtype.kind not in "iu":
            raise ValueError("best-pair layout must be one 576-element integer vector")
        layout = np.ascontiguousarray(raw, dtype=np.int32)
        if not np.array_equal(np.sort(layout), np.arange(TILE_COUNT)):
            raise ValueError("best-pair layout must use every original tile exactly once")
        if tuple(name for name, _ in self.costs) != (
            "raw",
            "logistic",
            "focal_top5",
            "nonlinear",
            selective_vote500.SELECTIVE_VOTE500_ARM,
        ):
            raise ValueError("best-pair five-arm cost roster changed")
        if self.selected_arm not in {name for name, _ in self.costs}:
            raise ValueError("selected arm is outside the fixed five-arm roster")
        if tuple(self.artifact_sha256) != EXPECTED_ARTIFACT_SHA256:
            raise ValueError("resource provenance differs from the SHA-gated manifest")
        layout = layout.copy()
        layout.setflags(write=False)
        object.__setattr__(self, "layout", layout)

    @property
    def layout_sha256(self) -> str:
        return hashlib.sha256(self.layout.astype("<i4", copy=False).tobytes(order="C")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "aiijc-taska-best-pair-pipeline-result-v1",
            "layout_sha256": self.layout_sha256,
            "selected_arm": self.selected_arm,
            "costs": dict(self.costs),
            "diagnostics": dict(self.diagnostics),
            "artifact_sha256": dict(self.artifact_sha256),
            "selective_solver_sha256": SELECTIVE_SOLVER_SHA256,
            "confirmation_config_sha256": CONFIRMATION_CONFIG_SHA256,
            "confirmation_report_sha256": CONFIRMATION_REPORT_SHA256,
            "layout_only": True,
            "original_upright_tile_permutation": True,
        }


def solve_taska_best_pair_pipeline(
    dirty_tiles: Any,
    resources: TaskaPairPipelineResources,
    *,
    focal_chunk_size: int = 8192,
) -> TaskaBestPairPipelineResult:
    """Run the unchanged confirmed solver with already SHA-gated resources."""

    verify_taska_best_pair_solver()
    if resources.artifact_sha256 != EXPECTED_ARTIFACT_SHA256:
        raise ValueError("resources do not match the fixed production manifest")
    solved = selective_vote500.solve_selective_vote500(
        dirty_tiles,
        resources,
        focal_chunk_size=focal_chunk_size,
    )
    return TaskaBestPairPipelineResult(
        layout=solved.candidate_layout,
        selected_arm=solved.candidate_choice,
        costs=solved.five_arm_costs,
        diagnostics=solved.diagnostics(),
        artifact_sha256=resources.artifact_sha256,
    )


def _write_npy_exclusive(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            np.save(stream, value, allow_pickle=False)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve one 576x20x20x3 original upright tile bag with the confirmed "
            "best pair-oriented pipeline; emit a layout only."
        )
    )
    parser.add_argument("tiles", type=Path, help="input .npy tile bag")
    parser.add_argument("--output-layout", type=Path, required=True)
    parser.add_argument("--diagnostics-json", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--focal-chunk-size", type=int, default=8192)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Layout-only one-board CLI; output files are created exclusively."""

    arguments = parse_args(argv)
    loaded = np.load(arguments.tiles, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        loaded.close()
        raise ValueError("tiles must be one .npy array, not an .npz archive")
    tiles = np.asarray(loaded)
    resources = load_taska_pair_pipeline_resources(device=arguments.device)
    result = solve_taska_best_pair_pipeline(
        tiles,
        resources,
        focal_chunk_size=arguments.focal_chunk_size,
    )
    _write_npy_exclusive(arguments.output_layout, result.layout)
    if arguments.diagnostics_json is not None:
        _write_json_exclusive(arguments.diagnostics_json, result.as_dict())
    print(json.dumps(result.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI shim.
    raise SystemExit(main())


__all__ = [
    "CONFIRMATION_CONFIG_SHA256",
    "CONFIRMATION_REPORT_SHA256",
    "SELECTIVE_SOLVER_SHA256",
    "TaskaBestPairPipelineResult",
    "load_taska_pair_pipeline_resources",
    "main",
    "solve_taska_best_pair_pipeline",
    "verify_taska_best_pair_solver",
]
