"""Read-only extractor for verified test source records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def source_summary(source: Any) -> tuple[Any, Any]:
    if not isinstance(source, dict):
        return None, None
    return source.get("record_id"), source.get("url")


def line(prefix: str, row: dict[str, Any]) -> str:
    sid, url = source_summary(row.get("source"))
    accepted = row.get("accepted")
    if isinstance(accepted, dict):
        accept_type = "object"
    else:
        accept_type = repr(accepted)
    fields = {
        "test": row.get("test"),
        "accepted": accept_type,
        "rank": row.get("rank"),
        "source_id": sid,
        "url": url,
        "saved_original": row.get("saved_original"),
        "override": row.get("override"),
        "corr": row.get("assignment_mean_correlation"),
        "q10": row.get("assignment_q10_correlation"),
        "bag": row.get("bag_distance"),
        "sift_frac": row.get("sift_identity_fraction"),
        "sift_inliers": row.get("sift_ransac_inliers"),
        "sift_disp": row.get("sift_median_displacement"),
    }
    return prefix + " " + " ".join(f"{key}={value!r}" for key, value in fields.items() if value is not None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        print(f"### {path}")
        if not path.is_file():
            print("NOT_FOUND")
            continue
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            print(f"ROOT={type(data).__name__}")
            continue
        for name, value in data.items():
            if isinstance(value, list):
                print(f"KEY {name} LIST {len(value)}")
            elif isinstance(value, dict):
                print(f"KEY {name} DICT {len(value)}")
            else:
                print(f"KEY {name} SCALAR {value!r}")
        for key in ("accepted", "accepted_overrides"):
            value = data.get(key)
            if isinstance(value, list):
                print(f"SECTION {key} {len(value)}")
                for index, row in enumerate(value):
                    if isinstance(row, dict):
                        print(line(f"{key}[{index}]", row))
                    else:
                        print(f"{key}[{index}] {row!r}")
        rows = data.get("rows")
        if isinstance(rows, list):
            special = [row for row in rows if isinstance(row, dict) and (row.get("accepted") is not None or row.get("source") is not None)]
            print(f"ROWS_SPECIAL {len(special)} of {len(rows)}")
            for index, row in enumerate(special):
                print(line(f"row_special[{index}]", row))


if __name__ == "__main__":
    main()
