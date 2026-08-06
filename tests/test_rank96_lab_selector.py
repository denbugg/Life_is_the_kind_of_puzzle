from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
from skimage.color import rgb2lab


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import rank96_lab_selector as selector  # noqa: E402


def _gradient_tiles() -> np.ndarray:
    tiles = np.empty(
        (selector.NUM_TILES, selector.TILE_SIZE, selector.TILE_SIZE, 3),
        dtype=np.uint8,
    )
    for index in range(selector.NUM_TILES):
        row, column = divmod(index, selector.GRID)
        tiles[index, ..., 0] = 16 + 7 * row
        tiles[index, ..., 1] = 16 + 7 * column
        tiles[index, ..., 2] = 12 + 4 * (row + column)
    return tiles


def _scrambled_board() -> np.ndarray:
    return np.random.default_rng(20_260_808).permutation(selector.NUM_TILES)


class LabDepth1ScoreTests(unittest.TestCase):
    def test_matches_literal_assembled_canvas_lab_formula(self) -> None:
        rng = np.random.default_rng(17)
        tiles = rng.integers(
            0,
            256,
            size=(selector.NUM_TILES, selector.TILE_SIZE, selector.TILE_SIZE, 3),
            dtype=np.uint8,
        )
        board = rng.permutation(selector.NUM_TILES)
        canvas = (
            tiles[board]
            .reshape(
                selector.GRID,
                selector.GRID,
                selector.TILE_SIZE,
                selector.TILE_SIZE,
                3,
            )
            .transpose(0, 2, 1, 3, 4)
            .reshape(
                selector.GRID * selector.TILE_SIZE,
                selector.GRID * selector.TILE_SIZE,
                3,
            )
        )
        lab = rgb2lab(canvas.astype(np.float32) / 255.0).astype(np.float32)
        lab /= selector.LAB_SCALE
        blocked = lab.reshape(
            selector.GRID,
            selector.TILE_SIZE,
            selector.GRID,
            selector.TILE_SIZE,
            3,
        )
        horizontal = (
            blocked[:, :, :-1, selector.INNER_HIGH, :]
            - blocked[:, :, 1:, selector.INNER_LOW, :]
        )
        vertical = (
            blocked[:-1, selector.INNER_HIGH, :, :, :]
            - blocked[1:, selector.INNER_LOW, :, :, :]
        )
        expected = -0.5 * (
            np.square(horizontal).mean(dtype=np.float64)
            + np.square(vertical).mean(dtype=np.float64)
        )
        self.assertEqual(selector.lab_depth1_board_score(tiles, board), float(expected))

    def test_smooth_upright_board_beats_scrambled_board(self) -> None:
        tiles = _gradient_tiles()
        identity = np.arange(selector.NUM_TILES)
        scrambled = _scrambled_board()
        self.assertGreater(
            selector.lab_depth1_board_score(tiles, identity),
            selector.lab_depth1_board_score(tiles, scrambled),
        )
        result = selector.select_lab_depth1_board(tiles, identity, scrambled)
        self.assertEqual(result.selected_arm, selector.RANK96_ARM)
        self.assertTrue(np.array_equal(result.selected_board, identity))

    def test_selector_can_choose_rank512_and_tie_is_rank96(self) -> None:
        tiles = _gradient_tiles()
        identity = np.arange(selector.NUM_TILES)
        scrambled = _scrambled_board()
        selected = selector.select_lab_depth1_board(tiles, scrambled, identity)
        self.assertEqual(selected.selected_arm, selector.RANK512_ARM)
        self.assertTrue(np.array_equal(selected.selected_board, identity))

        tied = selector.select_lab_depth1_board(tiles, identity, identity.copy())
        self.assertEqual(tied.rank96_lab_score, tied.rank512_lab_score)
        self.assertEqual(tied.selected_arm, selector.RANK96_ARM)

    def test_outer_boundary_pixels_do_not_affect_depth1_score(self) -> None:
        tiles = _gradient_tiles()
        board = np.arange(selector.NUM_TILES)
        before = selector.lab_depth1_board_score(tiles, board)
        changed = tiles.copy()
        # Depth is perpendicular to a seam.  Its tangent trace still includes
        # row/column endpoints, so alter only outer pixels outside all four
        # selected inset traces (rows/columns 1 and 18).
        keep = {selector.INNER_LOW, selector.INNER_HIGH}
        for index in range(selector.TILE_SIZE):
            if index not in keep:
                changed[:, 0, index, :] = 255
                changed[:, -1, index, :] = 0
                changed[:, index, 0, :] = 127
                changed[:, index, -1, :] = 63
        after = selector.lab_depth1_board_score(changed, board)
        self.assertEqual(before, after)

    def test_invalid_tiles_and_non_permutation_fail_closed(self) -> None:
        tiles = _gradient_tiles()
        invalid_board = np.arange(selector.NUM_TILES)
        invalid_board[-1] = 0
        with self.assertRaises(selector.LabSelectorError):
            selector.lab_depth1_board_score(tiles, invalid_board)
        with self.assertRaises(selector.LabSelectorError):
            selector.lab_depth1_board_score(tiles.astype(np.float32), np.arange(576))


class SolverContractTests(unittest.TestCase):
    def test_shared_dense_matrices_use_only_budgets_96_and_512(self) -> None:
        tiles = _gradient_tiles()
        right = np.zeros((selector.NUM_TILES, selector.NUM_TILES), dtype=np.float32)
        down = np.zeros_like(right)
        identity = np.arange(selector.NUM_TILES)
        scrambled = _scrambled_board()
        calls: list[dict[str, object]] = []
        matrix_ids: list[tuple[int, int]] = []

        def fake_solver(r: np.ndarray, d: np.ndarray, **kwargs: object):
            self.assertFalse(r.flags.writeable)
            self.assertFalse(d.flags.writeable)
            self.assertTrue(np.array_equal(r, right))
            self.assertTrue(np.array_equal(d, down))
            self.assertFalse(np.shares_memory(r, right))
            self.assertFalse(np.shares_memory(d, down))
            matrix_ids.append((id(r), id(d)))
            calls.append(dict(kwargs))
            if kwargs["max_edges"] == selector.RANK96_MAX_EDGES:
                return identity, 96.0
            return scrambled, 512.0

        result = selector.solve_and_select_lab_depth1(
            tiles, right, down, solver=fake_solver
        )
        self.assertEqual(
            calls,
            [
                {"max_edges": 96, "min_margin": 0.0, "repair_passes": 0},
                {"max_edges": 512, "min_margin": 0.0, "repair_passes": 0},
            ],
        )
        self.assertEqual(result.rank96_objective, 96.0)
        self.assertEqual(result.rank512_objective, 512.0)
        self.assertEqual(result.selected_arm, selector.RANK96_ARM)
        self.assertEqual(matrix_ids[0], matrix_ids[1])

    def test_solver_cannot_mutate_shared_dense_snapshot(self) -> None:
        tiles = _gradient_tiles()
        right = np.zeros((selector.NUM_TILES, selector.NUM_TILES), dtype=np.float32)
        down = np.zeros_like(right)

        def mutating_solver(r: np.ndarray, d: np.ndarray, **_: object):
            r[0, 1] = 1.0
            return np.arange(selector.NUM_TILES), 0.0

        with self.assertRaises(ValueError):
            selector.solve_and_select_lab_depth1(
                tiles, right, down, solver=mutating_solver
            )
        self.assertEqual(float(right[0, 1]), 0.0)

    def test_contract_exposes_no_rotation_or_selector_threshold(self) -> None:
        parameters = inspect.signature(selector.solve_and_select_lab_depth1).parameters
        self.assertNotIn("rotation", parameters)
        self.assertNotIn("depth", parameters)
        self.assertNotIn("threshold", parameters)
        self.assertEqual(selector.DEPTH, 1)
        self.assertEqual((selector.INNER_LOW, selector.INNER_HIGH), (1, 18))


if __name__ == "__main__":
    unittest.main()
