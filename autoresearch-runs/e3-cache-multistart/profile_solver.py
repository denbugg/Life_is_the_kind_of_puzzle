"""Profile one frozen case through the baseline solver."""
from __future__ import annotations

import argparse
import cProfile
import pstats
from pathlib import Path

import numpy as np

from global_solver_candidate import solve_layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.cache, mmap_mode="r")
    profiler = cProfile.Profile()
    profiler.enable()
    solve_layout(data["right"][0], data["down"][0], data["pos"][0], 20260818)
    profiler.disable()
    with args.output.open("w") as handle:
        stats = pstats.Stats(profiler, stream=handle).sort_stats("cumulative")
        stats.print_stats(30)


if __name__ == "__main__":
    main()
