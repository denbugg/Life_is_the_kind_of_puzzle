"""Evaluator-only E20 union truth-coverage gate; truth never enters scoring."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(HERE)]

import kaggle_e14_solver as e14
from e20_common import candidate_union, coverage_counts, validate_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()
    data, sidecar, provenance, cache_hash, sidecar_hash = validate_inputs(
        args.cache, args.sidecar
    )
    stop = min(args.start + args.limit, len(data["stems"]))
    totals = {direction: {"e14": 0, "union": 0, "count": 0}
              for direction in (0, 1)}
    rows = []
    for index in range(args.start, stop):
        raw = np.asarray(data["tiles"][index], np.uint8)
        restored = np.asarray(sidecar["restored"][index], np.uint8)
        classical_right, classical_down = e14.classical_mgc_ssd_scores(raw)
        e14_right = e14.fuse_scores(data["right"][index], classical_right)
        e14_down = e14.fuse_scores(data["down"][index], classical_down)
        row = {"index": index, "stem": str(data["stems"][index])}
        for direction, scores, label in (
            (0, e14_right, "right"), (1, e14_down, "down")
        ):
            union, e14_ids, _ = candidate_union(scores, restored, direction)
            e14_candidates = [ids for ids in e14_ids]
            e14_hits, count = coverage_counts(
                e14_candidates, data["truth"][index], direction
            )
            union_hits, _ = coverage_counts(union, data["truth"][index], direction)
            totals[direction]["e14"] += e14_hits
            totals[direction]["union"] += union_hits
            totals[direction]["count"] += count
            row[label] = {
                "e14_hits": e14_hits,
                "union_hits": union_hits,
                "count": count,
                "mean_union_size": float(np.mean([len(ids) for ids in union])),
            }
        rows.append(row)
        print(json.dumps({"done": index - args.start + 1,
                          "total": stop - args.start, "stem": row["stem"]}), flush=True)

    coverage = {}
    gate = True
    for direction, label in ((0, "right"), (1, "down")):
        item = totals[direction]
        baseline = item["e14"] / item["count"]
        union = item["union"] / item["count"]
        delta = union - baseline
        coverage[label] = {
            "e14_top32": baseline,
            "union_top32_top32": union,
            "delta": delta,
            "hits_e14": item["e14"],
            "hits_union": item["union"],
            "edges": item["count"],
        }
        gate &= delta >= 0.05
    report = {
        "experiment": "E20 evaluator-only candidate-union truth coverage",
        "cache_sha256": cache_hash,
        "sidecar_sha256": sidecar_hash,
        "sidecar_provenance": provenance,
        "cases": stop - args.start,
        "start": args.start,
        "coverage": coverage,
        "predeclared_coverage_gate": bool(gate),
        "truth_usage": "coverage reporting only; never candidate construction",
        "images": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "images"},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
