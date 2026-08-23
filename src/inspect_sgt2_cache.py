"""CPU-only schema inspector for the pre-existing SGT candidate graph cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def describe(value: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    if value.size and np.issubdtype(value.dtype, np.number):
        sampled = value.reshape(-1)[: min(10000, value.size)]
        result["sample_min"] = float(np.min(sampled))
        result["sample_max"] = float(np.max(sampled))
        result["sample_mean"] = float(np.mean(sampled))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    with np.load(args.cache, allow_pickle=False) as archive:
        report = {"path": str(args.cache), "arrays": {name: describe(archive[name]) for name in archive.files}}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
