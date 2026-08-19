import unittest

import numpy as np

import global_solver_candidate as solver


class RelaxationSolverTests(unittest.TestCase):
    def test_sparse_compatibility_is_normalized_and_has_no_self_edges(self):
        rng = np.random.default_rng(3)
        scores = rng.normal(size=(solver.N, solver.N))
        np.fill_diagonal(scores, -1e4)
        outgoing, incoming = solver._topk_compatibility(scores, 7)
        self.assertTrue(np.all(outgoing.getnnz(axis=1) == 7))
        self.assertTrue(np.allclose(np.asarray(outgoing.sum(axis=1)).ravel(), 1.0))
        self.assertTrue(np.allclose(np.asarray(incoming.sum(axis=1)).ravel(), 1.0))
        self.assertTrue(np.all(outgoing.diagonal() == 0))

    def test_solver_returns_a_valid_permutation(self):
        rng = np.random.default_rng(5)
        right = rng.normal(size=(solver.N, solver.N)).astype(np.float32)
        down = rng.normal(size=(solver.N, solver.N)).astype(np.float32)
        np.fill_diagonal(right, -1e4)
        np.fill_diagonal(down, -1e4)
        pos = rng.normal(size=(solver.N, solver.N)).astype(np.float32)
        layout = np.asarray(solver.solve_layout(right, down, pos, 11))
        self.assertTrue(np.array_equal(np.sort(layout), np.arange(solver.N)))


if __name__ == "__main__":
    unittest.main()
