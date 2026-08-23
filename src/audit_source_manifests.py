"""Read-only targeted audit for source-forensics JSON manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SHOW = (
    "test", "accepted", "rank", "source", "saved_original", "report", "override",
    "assignment_mean_correlation", "assignment_q10_correlation", "bag_distance",
    "sift_identity_matches", "sift_identity_fraction", "sift_ransac_inliers",
    "sift_median_displacement", "decision", "reason", "note", "url", "record_id",
)


def compact(value: Any, depth: int = 0) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        if depth >= 1:
            return f"list[{len(value)}]"
        return [compact(item, depth + 1) for item in value[:3]]
    if isinstance(value, dict):
        if depth >= 2:
            return {key: compact(val, depth + 1) for key, val in list(value.items())[:8]}
        relevant = {key: val for key, val in value.items() if key in SHOW}
        if not relevant:
            relevant = dict(list(value.items())[:8])
        return {key: compact(val, depth + 1) for key, val in relevant.items()}
    return str(value)


def emit(label: str, item: Any) -> None:
    print(label + "=" + json.dumps(compact(item), ensure_ascii=False, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--sample", type=int, default=30)
    args = parser.parse_args()
    for path in args.paths:
        print(f"\n=== {path} ===")
        if not path.is_file():
            print("NOT_FOUND")
            continue
        with path.open("r", encoding="utf-8") as fh:
            obj = json.load(fh)
        if not isinstance(obj, dict):
            print("ROOT_NOT_DICT")
            continue
        print("top_keys=" + ",".join(obj.keys()))
        for key, value in obj.items():
            if isinstance(value, list):
                print(f"top.{key}=list[{len(value)}]")
            elif isinstance(value, dict):
                print(f"top.{key}=dict[{len(value)}]")
            else:
                print(f"top.{key}={compact(value)}")
        for key in ("accepted", "accepted_overrides", "rows"):
            value = obj.get(key)
            if not isinstance(value, list):
                continue
            print(f"section={key} count={len(value)}")
            if key == "rows":
                accepted_rows = [row for row in value if isinstance(row, dict) and row.get("accepted") is not None]
                source_rows = [row for row in value if isinstance(row, dict) and row.get("source")]
                print(f"rows_accepted_nonnull={len(accepted_rows)} rows_source_nonempty={len(source_rows)}")
                candidates = accepted_rows or source_rows
            else:
                candidates = value
            for index, row in enumerate(candidates[:args.sample]):
                emit(f"{key}[{index}]", row)


if __name__ == "__main__":
    main()
