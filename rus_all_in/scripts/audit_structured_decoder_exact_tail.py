#!/usr/bin/env python3
"""Audit exact-delta concentration from an existing signed oracle report.

This is a read-only metric consumer: it never reconstructs FIT cases or opens
cache labels.  It validates the completed structured-decoder report and its
recorded artifact hashes, computes the preregistered exact-tail diagnostics,
and writes one separate exclusive JSON artifact.  It cannot modify the signed
runner, config, head, controls or primary report.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from aiijc_puzzle.protocol import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    PROJECT_ROOT / "outputs/compatibility-structured-decoder-fit-oracle/v1/report.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/compatibility-structured-decoder-fit-oracle/v1/exact-tail-audit.json"
)
PRIMARY_REPORT_SCHEMA = "aiijc-structured-decoder-fit-oracle-report-v1"
TAIL_SCHEMA = "aiijc-structured-decoder-exact-tail-audit-v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        label = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        label = str(resolved)
    return {"path": label, "sha256": sha256_file(resolved)}


def _verify_record(record: Mapping[str, Any], *, name: str) -> None:
    path = _project_path(str(record.get("path", "")))
    if not path.is_file() or record.get("sha256") != sha256_file(path):
        raise RuntimeError(f"primary report {name} artifact changed")


def compute_exact_tail(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute fixed exact-tail diagnostics from validated report rows."""

    if not rows:
        raise ValueError("exact-tail audit requires at least one row")
    deltas = np.asarray(
        [int(row["delta"]["exact_tiles"]) for row in rows], dtype=np.int64
    )
    control = np.asarray(
        [int(row["control"]["exact_tiles"]) for row in rows], dtype=np.int64
    )
    ceiling = np.asarray(
        [int(row["ceiling"]["exact_tiles"]) for row in rows], dtype=np.int64
    )
    if not np.array_equal(ceiling - control, deltas):
        raise RuntimeError("reported exact deltas do not match absolute exact counts")
    if np.any(control < 0) or np.any(control > 576):
        raise RuntimeError("control absolute exact count is out of range")
    if np.any(ceiling < 0) or np.any(ceiling > 576):
        raise RuntimeError("ceiling absolute exact count is out of range")

    count = len(rows)
    wins = int(np.count_nonzero(deltas > 0))
    ties = int(np.count_nonzero(deltas == 0))
    losses = int(np.count_nonzero(deltas < 0))
    positive = np.maximum(deltas, 0)
    total_positive = int(positive.sum())
    largest_positive = int(positive.max())
    largest_index = int(np.argmax(positive)) if largest_positive > 0 else None
    if largest_index is None:
        leave_largest_mean = float(np.mean(deltas))
        removed: dict[str, Any] | None = None
    else:
        keep = np.ones(count, dtype=bool)
        keep[largest_index] = False
        leave_largest_mean = (
            float(np.mean(deltas[keep])) if np.any(keep) else 0.0
        )
        removed = {
            "row_index": largest_index,
            "prefix": rows[largest_index].get("prefix"),
            "case_id": rows[largest_index].get("case_id"),
            "exact_delta": int(deltas[largest_index]),
        }

    def sparse_absolute(values: np.ndarray) -> dict[str, int | float]:
        zero = int(np.count_nonzero(values == 0))
        at_most_one = int(np.count_nonzero(values <= 1))
        return {
            "zero_count": zero,
            "zero_share": zero / count,
            "at_most_one_count": at_most_one,
            "at_most_one_share": at_most_one / count,
        }

    return {
        "case_count": count,
        "exact_delta": {
            "mean": float(np.mean(deltas)),
            "median": float(np.median(deltas)),
            "q25": float(np.quantile(deltas, 0.25, method="linear")),
            "q75": float(np.quantile(deltas, 0.75, method="linear")),
            "quantile_method": "linear",
        },
        "absolute_exact": {
            "control": sparse_absolute(control),
            "pair_safe_ceiling": sparse_absolute(ceiling),
        },
        "win_tie_loss": {
            "win_count": wins,
            "tie_count": ties,
            "loss_count": losses,
            "win_share": wins / count,
            "tie_share": ties / count,
            "loss_share": losses / count,
        },
        "positive_concentration": {
            "total_positive_exact_gain": total_positive,
            "largest_positive_exact_gain": largest_positive,
            "largest_positive_share": (
                largest_positive / total_positive if total_positive else 0.0
            ),
            "leave_largest_positive_mean_exact_delta": leave_largest_mean,
            "removed_largest_positive": removed,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    report_path = args.report.resolve()
    output_path = args.output.resolve()
    if output_path == report_path:
        raise RuntimeError("tail audit cannot overwrite the primary signed report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != PRIMARY_REPORT_SCHEMA:
        raise RuntimeError("structured-decoder primary report schema changed")
    rows = tuple(report.get("rows", ()))
    if report.get("case_count") != 64 or len(rows) != 64:
        raise RuntimeError("structured-decoder exact-tail audit requires 64 FIT cases")
    prefixes = [row.get("prefix") for row in rows]
    if prefixes != [f"case_{index:04d}" for index in range(64)]:
        raise RuntimeError("structured-decoder report row order changed")
    if any(row.get("strict_original_upright_permutation") is not True for row in rows):
        raise RuntimeError("structured-decoder report contains a non-strict layout")
    _verify_record(report.get("config", {}), name="config")
    artifacts = report.get("artifacts", {})
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise RuntimeError("structured-decoder report omits artifact provenance")
    for name, artifact in artifacts.items():
        if not isinstance(artifact, Mapping):
            raise RuntimeError("structured-decoder report artifact record changed")
        _verify_record(artifact, name=str(name))

    payload = {
        "schema": TAIL_SCHEMA,
        "status": "complete-read-only-primary-report-tail-audit",
        "scope": "derived only from the frozen primary FIT oracle report; no labels opened",
        "primary_report": _record(report_path),
        "metrics": compute_exact_tail(rows),
        "reference_or_cache_labels_opened": False,
        "primary_report_or_signed_artifacts_modified": False,
        "weco_logged": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
