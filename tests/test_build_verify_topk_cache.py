import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_verify_topk_cache import candidate_labels, shortlist


def test_shortlist_excludes_diagonal_and_orders_scores():
    cost = np.array([[0, 4, 1, 3], [2, 0, 1, 5],
                     [2, 4, 0, 1], [3, 2, 1, 0]], np.float32)
    ids, scores = shortlist(cost, 2)
    assert np.array_equal(ids[0], [2, 3])
    assert 0 not in ids[0]
    assert np.all(scores[:, 0] >= scores[:, 1])


def test_candidate_labels_distinguish_missing_and_boundary():
    ids = np.array([[1, 2], [3, 0], [3, 0], [0, 1]], np.uint16)
    valid = np.array([True, True, True, False])
    labels = candidate_labels(ids, 1, valid)
    assert np.array_equal(labels, [0, 2, 0, -1])
