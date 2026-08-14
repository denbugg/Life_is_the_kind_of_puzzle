"""Synthetic structural G0 for P9 loop reweighting.

No data files, models, targets, layouts, or GPU are used.  The test encodes a
single 2×2 loop with tile IDs 0,1,2,3 plus distractors and verifies decoder
invariants required before source-disjoint FIT evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p9_loop_reweight import (
    DOWN,
    RIGHT,
    assert_valid_loop_report,
    reweight_2x2_loops,
    sparse_to_dense_rd,
)

SENTINEL = -1.0e9
OUT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P9_loop_decoder\g0_structural")


def fixture() -> tuple[np.ndarray, np.ndarray]:
    # Candidate rows are anchor-indexed and shared across directions, matching rank96/P8.
    c = np.array(
        [
            [1, 2, 4, 5],  # 0: R->1 and D->2 build the intended square
            [3, 0, 4, 5],  # 1: D->3
            [3, 0, 4, 5],  # 2: R->3
            [0, 1, 2, 5],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ],
        dtype=np.int64,
    )
    s = np.full((4, 6, 4), SENTINEL, dtype=np.float64)
    # One high-confidence 2x2 clockwise relation; nonused directions deliberately absent.
    s[RIGHT, 0, 0] = 10.0  # 0 -> 1
    s[DOWN, 0, 1] = 9.0    # 0 -> 2
    s[DOWN, 1, 0] = 8.0    # 1 -> 3
    s[RIGHT, 2, 0] = 7.0   # 2 -> 3
    # Distractors that cannot form a valid 2x2 loop.
    s[RIGHT, 0, 2] = 6.0
    s[DOWN, 0, 3] = 5.0
    return c, s


def main() -> None:
    candidates, scores = fixture()
    zero, zero_report = reweight_2x2_loops(candidates, scores, loop_k=4, lambda_value=0.0, sentinel=SENTINEL)
    assert np.array_equal(zero, scores), "lambda=0 must be bit-identical"
    out_a, report_a = reweight_2x2_loops(candidates, scores, loop_k=4, lambda_value=0.75, sentinel=SENTINEL)
    out_b, report_b = reweight_2x2_loops(candidates, scores, loop_k=4, lambda_value=0.75, sentinel=SENTINEL)
    assert np.array_equal(out_a, out_b), "reweighting must be deterministic"
    assert report_a == report_b
    assert_valid_loop_report(report_a)
    # Existing unsupported/inexistent relations may not become edges.
    assert np.all(out_a[RIGHT][scores[RIGHT] <= SENTINEL / 2.0] == SENTINEL)
    assert np.all(out_a[DOWN][scores[DOWN] <= SENTINEL / 2.0] == SENTINEL)
    # The fixture must exercise real loop support and the reported edge support.
    assert report_a.accepted_loops >= 2, report_a
    assert report_a.supported_edges >= 2, report_a
    r, d = sparse_to_dense_rd(candidates, out_a, sentinel=SENTINEL)
    assert r.shape == (6, 6) and d.shape == (6, 6)
    assert np.all(np.diag(r) == SENTINEL) and np.all(np.diag(d) == SENTINEL)
    result = {
        "experiment": "P9_loop_decoder",
        "gate": "G0b_structural_loop_contract",
        "decision": "PASS",
        "zero_lambda_bit_identical": True,
        "deterministic": True,
        "absent_edges_preserved": True,
        "valid_dense_shapes": [int(r.shape[0]), int(r.shape[1])],
        "loop_report": report_a.__dict__,
        "CAL_target_opened": False,
        "DEV_targets_opened": False,
        "test_accessed": False,
        "layouts_assembled": False,
        "restorer_used": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p9_g0b_structural_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
