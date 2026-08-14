"""P9 G0b synthetic contract for canonical direction-indexed candidate lists."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p9_directed_loop_reweight import (
    DOWN,
    RIGHT,
    directed_to_dense_rd,
    reweight_directed_2x2_loops,
)

SENTINEL = -1.0e9
OUT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P9_loop_decoder\g0_structural")


def query_rows() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Required canonical queries for a 2x2 square: 0-R-1, 0-D-2, 1-D-3, 2-R-3.
    # Candidate lists deliberately differ by direction for the same anchor.
    anchors = np.array([0, 0, 1, 2, 1, 2], dtype=np.int64)
    directions = np.array([RIGHT, DOWN, DOWN, RIGHT, RIGHT, DOWN], dtype=np.int64)
    members = np.array(
        [
            [1, 4, 5],
            [2, 5, 4],
            [3, 4, 5],
            [3, 4, 5],
            [4, 0, 5],
            [4, 0, 5],
        ],
        dtype=np.int64,
    )
    scores = np.full((6, 3), SENTINEL, dtype=np.float64)
    scores[0] = [10.0, 6.0, 4.0]
    scores[1] = [9.0, 5.0, 3.0]
    scores[2] = [8.0, 5.0, 2.0]
    scores[3] = [7.0, 5.0, 2.0]
    scores[4] = [2.0, 1.0, 0.0]
    scores[5] = [2.0, 1.0, 0.0]
    return anchors, directions, members, scores


def main() -> None:
    a, d, m, s = query_rows()
    zero, zero_report = reweight_directed_2x2_loops(a, d, m, s, n_tiles=6, loop_k=3, lambda_value=0.0, sentinel=SENTINEL)
    assert np.array_equal(zero, s)
    x1, report1 = reweight_directed_2x2_loops(a, d, m, s, n_tiles=6, loop_k=3, lambda_value=0.5, sentinel=SENTINEL)
    x2, report2 = reweight_directed_2x2_loops(a, d, m, s, n_tiles=6, loop_k=3, lambda_value=0.5, sentinel=SENTINEL)
    assert np.array_equal(x1, x2) and report1 == report2
    assert report1.accepted_loops >= 2, report1
    assert report1.supported_horizontal_edges >= 1 and report1.supported_vertical_edges >= 1, report1
    assert np.all(x1[s <= SENTINEL / 2.0] == SENTINEL), "must not create absent candidate edges"
    r, down = directed_to_dense_rd(a, d, m, x1, n_tiles=6, sentinel=SENTINEL)
    assert np.all(np.diag(r) == SENTINEL) and np.all(np.diag(down) == SENTINEL)
    # Direction-specific candidate lists must remain direction-specific in dense output.
    assert r[0, 1] > SENTINEL / 2.0 and down[0, 2] > SENTINEL / 2.0
    assert down[0, 1] == SENTINEL and r[0, 2] == SENTINEL
    result = {
        "experiment": "P9_loop_decoder",
        "gate": "G0b_directed_loop_contract",
        "decision": "PASS",
        "zero_lambda_bit_identical": True,
        "deterministic": True,
        "absent_edges_preserved": True,
        "direction_specific_lists_preserved": True,
        "loop_report": report1.__dict__,
        "CAL_target_opened": False,
        "DEV_targets_opened": False,
        "test_accessed": False,
        "layouts_assembled": False,
        "restorer_used": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p9_g0b_directed_structural_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
