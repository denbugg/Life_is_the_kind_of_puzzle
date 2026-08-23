"""Read-only compact extractor for source-forensics benchmark JSON reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def brief(value: Any, depth: int = 0) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        if depth >= 2:
            return f"list[{len(value)}]"
        return [brief(item, depth + 1) for item in value[:5]]
    if isinstance(value, dict):
        if depth >= 4:
            return f"dict[{len(value)}]"
        interesting = {}
        for key, item in value.items():
            lower = key.lower()
            if any(token in lower for token in ("summary", "metric", "recall", "precision", "accuracy", "coverage", "top", "rank", "accepted", "count", "total", "mean", "median", "q10", "threshold", "fold", "sample")):
                interesting[key] = brief(item, depth + 1)
        if not interesting:
            interesting = {key: brief(item, depth + 1) for key, item in list(value.items())[:10]}
        return interesting
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        print(f"\n=== {path} ===")
        if not path.is_file():
            print("NOT_FOUND")
            continue
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        print(json.dumps(brief(data), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
