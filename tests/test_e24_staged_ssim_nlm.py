from __future__ import annotations

import copy
import hashlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import eval_e24_staged_ssim_nlm as staged


E24_TMP = Path("E:/pazzle_work/posegraph_e24_selector/tmp")


def digest(index: int) -> str:
    return hashlib.sha256(f"synthetic-{index}".encode("ascii")).hexdigest()


def decode_provenance() -> dict[str, str]:
    return {key: digest(index) for index, key in enumerate(sorted(staged._DECODE_PROVENANCE_KEYS))}


def board_provenance() -> dict[str, str]:
    return {key: digest(100 + index) for index, key in enumerate(sorted(staged._BOARD_PROVENANCE_KEYS))}


def synthetic_decode(image: int = 10) -> staged.FrozenDecode:
    first = SimpleNamespace(
        hypothesis_id=7,
        relation_id=17,
        u=0,
        v=1,
        dr=0,
        dc=1,
        score=0.9,
        none_score=0.1,
        margin=0.8,
        support=2,
    )
    second = SimpleNamespace(
        hypothesis_id=8,
        relation_id=18,
        u=2,
        v=3,
        dr=1,
        dc=0,
        score=0.6,
        none_score=0.1,
        margin=0.5,
        support=1,
    )
    components: list[dict[int, tuple[int, int]]] = [{0: (0, 0), 1: (0, 1)}]
    components.extend({tile: (0, 0)} for tile in range(2, staged.NUM_TILES))
    outcomes = (
        SimpleNamespace(
            selection=first,
            accepted=True,
            reason="tree",
            tree_merge=True,
            cycle=False,
        ),
        SimpleNamespace(
            selection=second,
            accepted=False,
            reason="contact",
            tree_merge=False,
            cycle=False,
        ),
    )
    decoded = SimpleNamespace(
        selected=(first, second),
        attempted=(first, second),
        outcomes=outcomes,
        components=tuple(components),
        attempt_cap=2,
    )
    return staged.freeze_decode_result(
        image=image, base_component_count=576, decoded=decoded
    )


def synthetic_boards(image: int = 10) -> staged.FrozenBoardPair:
    right = np.zeros((staged.NUM_TILES, staged.NUM_TILES), dtype=np.float32)
    down = np.zeros_like(right)
    rr = np.arange(staged.NUM_TILES - 1, -1, -1, dtype=np.int64)
    candidate = np.arange(staged.NUM_TILES, dtype=np.int64)
    rr_solved = np.zeros((staged.IMAGE_SIZE, staged.IMAGE_SIZE, 3), dtype=np.uint8)
    candidate_solved = np.full_like(rr_solved, 10)
    rr_restored = np.full_like(rr_solved, 20)
    candidate_restored = np.full_like(rr_solved, 30)
    return staged.FrozenBoardPair(
        image=image,
        right=right,
        down=down,
        rr96_board=rr,
        candidate_board=candidate,
        rr96_solved=rr_solved,
        candidate_solved=candidate_solved,
        rr96_restored=rr_restored,
        candidate_restored=candidate_restored,
        rr96_objective=1.0,
        candidate_objective=2.0,
    )


def arm_metrics(*, solve: float, final: float, neighbour: float) -> dict[str, object]:
    return {
        "objective": 1.0,
        "placement": 0.0,
        "neighbour": neighbour,
        "right": neighbour,
        "down": neighbour,
        "solve_only_ssim": solve,
        "final_ssim": final,
        "board_sha256": digest(201),
        "solved_corrupted_canvas_sha256": digest(202),
        "restored_canvas_sha256": digest(203),
    }


def staged_row(
    image: int,
    *,
    solve_delta: float,
    final_delta: float,
    neighbour_delta: float,
) -> dict[str, object]:
    solve_base = 0.0 if solve_delta >= 0 else -solve_delta
    solve_candidate = solve_delta if solve_delta >= 0 else 0.0
    final_base = 0.0 if final_delta >= 0 else -final_delta
    final_candidate = final_delta if final_delta >= 0 else 0.0
    neighbour_base = 0.0 if neighbour_delta >= 0 else -neighbour_delta
    neighbour_candidate = neighbour_delta if neighbour_delta >= 0 else 0.0
    rr = arm_metrics(
        solve=solve_base, final=final_base, neighbour=neighbour_base
    )
    candidate = arm_metrics(
        solve=solve_candidate,
        final=final_candidate,
        neighbour=neighbour_candidate,
    )
    return {
        "image": image,
        "fold": next(fold for fold, ids in staged.OOF_FOLDS.items() if image in ids),
        "validation_name": f"synthetic_{image}.png",
        "orientation_degrees": 0,
        "reflection": False,
        "provenance": {"board_commit_sha256": digest(image)},
        "permutation_sha256": digest(300 + image),
        "target_sha256": digest(400 + image),
        "rr96": rr,
        "candidate": candidate,
        "delta": {
            "solve_only_ssim": solve_candidate - solve_base,
            "final_ssim": final_candidate - final_base,
            "neighbour": neighbour_candidate - neighbour_base,
        },
    }


def broker_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for image in staged.CALIBRATION_IDS:
        rr = arm_metrics(
            solve=staged.PINNED_RR96_MEAN_SOLVE_SSIM,
            final=staged.PINNED_RR96_MEAN_FINAL_SSIM,
            neighbour=0.20,
        )
        rr["board_sha256"] = digest(1000 + image)
        rr["solved_corrupted_canvas_sha256"] = digest(1100 + image)
        rr["restored_canvas_sha256"] = digest(1200 + image)
        candidate = dict(rr)
        candidate["solve_only_ssim"] = float(rr["solve_only_ssim"]) + 0.004
        candidate["final_ssim"] = float(rr["final_ssim"]) + 0.003
        candidate["neighbour"] = float(rr["neighbour"]) + 0.006
        candidate["board_sha256"] = digest(1300 + image)
        candidate["solved_corrupted_canvas_sha256"] = digest(1400 + image)
        candidate["restored_canvas_sha256"] = digest(1500 + image)
        rows.append(
            {
                "image": image,
                "fold": next(
                    fold for fold, ids in staged.OOF_FOLDS.items() if image in ids
                ),
                "validation_name": f"synthetic_{image}.png",
                "orientation_degrees": 0,
                "reflection": False,
                "provenance": {
                    "board_barrier_sha256": digest(700),
                    "board_commit_sha256": digest(800 + image),
                    "metric_broker_contract_sha256": staged.METRIC_BROKER_CONTRACT_SHA256,
                    "metric_request_sha256": digest(900 + image),
                    "premetric_seal_sha256": digest(502),
                    "raw_archive_sha256": digest(1600 + image),
                    "e12_report_sha256": staged.PINNED_E12_REPORT_SHA256,
                    "calibration_report_sha256": staged.PINNED_CALIBRATION_REPORT_SHA256,
                    "scene_provenance_digest": staged.PINNED_SCENE_PROVENANCE_DIGEST,
                },
                "permutation_sha256": digest(1700 + image),
                "target_sha256": digest(1800 + image),
                "rr96": rr,
                "candidate": candidate,
                "delta": {
                    "solve_only_ssim": 0.0040000000000000036,
                    "final_ssim": 0.0030000000000000027,
                    "neighbour": 0.006000000000000005,
                },
            }
        )
    return rows


def write_metric_chain(
    root: Path,
    rows: list[dict[str, object]],
    board_hashes: dict[str, str],
) -> None:
    previous_sha = staged.METRIC_CHAIN_GENESIS_SHA256
    previous_path = ""
    for sequence, row in enumerate(rows):
        image = int(row["image"])
        scene_root = root / f"image_{image:04d}"
        raw_path = scene_root / "synthetic_raw.npz"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(f"synthetic-raw-{image}".encode("ascii"))
        raw_sha = staged.sha256_file(raw_path)
        row["provenance"]["raw_archive_sha256"] = raw_sha
        request = {
            "schema": staged.METRIC_REQUEST_SCHEMA,
            "schema_version": staged.SCHEMA_VERSION,
            "image": image,
            "sequence_index": sequence,
            "previous_response_path": previous_path,
            "previous_response_sha256": previous_sha,
            "ledger_sha256": digest(500),
            "run_contract_sha256": digest(501),
            "premetric_seal_sha256": digest(502),
            "structural_report_sha256": digest(503),
            "orchestration_receipt_sha256": digest(504),
            "board_barrier_sha256": digest(700),
            "board_commit_path": str(
                (
                    staged.DEFAULT_STAGED_BOARD_ROOT
                    / f"image_{image:04d}"
                    / "board_nlm.commit.json"
                ).resolve(strict=False)
            ),
            "board_commit_sha256": board_hashes[str(image)],
            "raw_archive_path": str(raw_path.resolve()),
            "raw_archive_sha256": raw_sha,
            "validation_name": row["validation_name"],
            "metric_broker_contract_sha256": staged.METRIC_BROKER_CONTRACT_SHA256,
            "e12_report_sha256": staged.PINNED_E12_REPORT_SHA256,
            "calibration_report_sha256": staged.PINNED_CALIBRATION_REPORT_SHA256,
            "scene_provenance_digest": staged.PINNED_SCENE_PROVENANCE_DIGEST,
            "e25_opened": False,
        }
        request_path = scene_root / "request.json"
        staged.commit_canonical_create_once(request_path, request)
        request_sha = staged.sha256_file(request_path)
        row["provenance"]["metric_request_sha256"] = request_sha
        response = {
            "schema": staged.METRIC_RESPONSE_SCHEMA,
            "schema_version": staged.SCHEMA_VERSION,
            "status": "complete_row_only",
            "image": image,
            "sequence_index": sequence,
            "request_sha256": request_sha,
            "previous_response_sha256": previous_sha,
            "row": row,
            "arrays_exported": False,
            "e25_opened": False,
        }
        response_path = scene_root / "response.json"
        staged.commit_canonical_create_once(response_path, response)
        previous_sha = staged.sha256_file(response_path)
        previous_path = str(response_path.resolve())


class E24StagedSyntheticTests(unittest.TestCase):
    def setUp(self) -> None:
        E24_TMP.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="staged_test_", dir=E24_TMP))

    def tearDown(self) -> None:
        if self.root.is_dir():
            shutil.rmtree(self.root)

    def test_protocol_and_broker_are_exact_and_target_loader_absent(self) -> None:
        self.assertEqual(staged.STAGED_PROTOCOL["candidate_packer"]["repair_passes"], 0)
        self.assertEqual(staged.STAGED_PROTOCOL["candidate_packer"]["restarts"], 1)
        self.assertEqual(staged.STAGED_PROTOCOL["candidate_packer"]["seed"], 1234)
        self.assertEqual(staged.STAGED_PROTOCOL["baseline"]["max_edges"], 96)
        self.assertEqual(staged.STAGED_PROTOCOL["restoration"]["h"], 10)
        self.assertEqual(
            staged.STAGED_PROTOCOL["rr96_metric_source"]["choice"],
            "reuse_exact_pinned_E12_RR_record_no_second_NLM_call",
        )
        self.assertEqual(
            staged.METRIC_BROKER_CONTRACT["raw_archive"]["allowed_member"],
            "permutation.npy",
        )
        self.assertFalse(hasattr(staged, "load_raw_scene"))
        self.assertFalse(hasattr(staged, "load_target"))

    def test_decode_commit_is_canonical_order_preserving_and_create_once(self) -> None:
        value = synthetic_decode()
        artifact = self.root / "decode.npz"
        commit = self.root / "decode.json"
        payload = staged.commit_decode(
            artifact_path=artifact,
            commit_path=commit,
            value=value,
            provenance=decode_provenance(),
        )
        loaded, observed = staged.load_decode(
            commit,
            expected_image=10,
            expected_provenance=decode_provenance(),
        )
        self.assertEqual(observed, payload)
        self.assertEqual(loaded.components, value.components)
        self.assertEqual(loaded.attempted_count, 2)
        self.assertEqual(loaded.components[0], {0: (0, 0), 1: (0, 1)})
        foreign = decode_provenance()
        foreign["premetric_seal_sha256"] = digest(999)
        with self.assertRaises(staged.E24StagedContractError):
            staged.load_decode(
                commit, expected_image=10, expected_provenance=foreign
            )
        with self.assertRaises(staged.E24StagedContractError):
            staged.commit_decode(
                artifact_path=artifact,
                commit_path=commit,
                value=value,
                provenance=decode_provenance(),
            )

    def test_decode_rejects_component_tile_order_drift(self) -> None:
        value = synthetic_decode()
        tiles = value.component_tiles.copy()
        tiles[:2] = tiles[1::-1]
        bad = staged.FrozenDecode(**{**value.__dict__, "component_tiles": tiles})
        with self.assertRaises(staged.E24StagedContractError):
            staged.decode_npz_bytes(bad)

    def test_board_builder_uses_only_exact_solver_kwargs_and_upright_assembly(self) -> None:
        calls: dict[str, object] = {}

        def components(right, down, decoded_components, **kwargs):
            calls["components"] = kwargs
            calls["decoded_components"] = decoded_components
            return np.arange(staged.NUM_TILES, dtype=np.int64), 2.0

        def rr96(right, down, **kwargs):
            calls["rr96"] = kwargs
            return np.arange(staged.NUM_TILES - 1, -1, -1, dtype=np.int64), 1.0

        def dense(_ids, _scores):
            matrix = np.zeros((staged.NUM_TILES, staged.NUM_TILES), dtype=np.float32)
            return matrix, matrix.copy()

        from imgio import assemble

        ids = np.zeros((staged.NUM_TILES, staged.CANDIDATE_WIDTH), dtype=np.int64)
        scores = np.zeros(
            (staged.NUM_DIRECTIONS, staged.NUM_TILES, staged.CANDIDATE_WIDTH),
            dtype=np.float32,
        )
        tiles = np.empty(
            (staged.NUM_TILES, staged.TILE_SIZE, staged.TILE_SIZE, 3), dtype=np.uint8
        )
        for tile in range(staged.NUM_TILES):
            tiles[tile].fill(tile % 256)
        output = staged.build_board_pair(
            image=10,
            candidate_ids=ids,
            raw_logits=scores,
            tiles=tiles,
            decode=synthetic_decode(),
            dense_builder=dense,
            component_solver=components,
            rr96_solver=rr96,
            assembler=assemble,
            restorer=lambda value: value,
        )
        self.assertEqual(calls["components"], {"repair_passes": 0, "restarts": 1, "seed": 1234})
        self.assertEqual(
            calls["rr96"], {"max_edges": 96, "min_margin": 0.0, "repair_passes": 0}
        )
        self.assertEqual(calls["decoded_components"], synthetic_decode().components)
        self.assertEqual(int(output.candidate_solved[0, 0, 0]), 0)
        self.assertEqual(int(output.rr96_solved[0, 0, 0]), 575 % 256)

    def test_board_commit_round_trip_and_create_once(self) -> None:
        value = synthetic_boards()
        artifact = self.root / "board.npz"
        commit = self.root / "board.json"
        payload = staged.commit_board_pair(
            artifact_path=artifact,
            commit_path=commit,
            value=value,
            provenance=board_provenance(),
        )
        loaded, observed = staged.load_board_pair(
            commit,
            expected_image=10,
            expected_provenance=board_provenance(),
        )
        self.assertEqual(observed, payload)
        self.assertTrue(np.array_equal(loaded.candidate_board, value.candidate_board))
        foreign = board_provenance()
        foreign["orchestration_receipt_sha256"] = digest(998)
        with self.assertRaises(staged.E24StagedContractError):
            staged.load_board_pair(
                commit, expected_image=10, expected_provenance=foreign
            )
        with self.assertRaises(staged.E24StagedContractError):
            staged.commit_board_pair(
                artifact_path=artifact,
                commit_path=commit,
                value=value,
                provenance=board_provenance(),
            )

    def test_measure_scene_uses_argsort_permutation_and_committed_nlm_bytes(self) -> None:
        boards = synthetic_boards()
        permutation = np.arange(staged.NUM_TILES - 1, -1, -1, dtype=np.int64)
        target = np.zeros((staged.IMAGE_SIZE, staged.IMAGE_SIZE, 3), dtype=np.uint8)
        seen: list[int] = []

        def ssim(_target, canvas):
            seen.append(int(canvas[0, 0, 0]))
            return float(canvas[0, 0, 0]) / 255.0

        row = staged.measure_scene(
            boards=boards,
            permutation=permutation,
            target=target,
            validation_name="synthetic.png",
            provenance={"board_commit_sha256": digest(1)},
            ssim=ssim,
        )
        self.assertEqual(row["rr96"]["placement"], 1.0)
        self.assertEqual(row["candidate"]["placement"], 0.0)
        self.assertEqual(seen, [0, 20, 10, 30])
        self.assertGreater(row["delta"]["final_ssim"], 0.0)

    def test_pinned_rr96_path_reuses_record_and_rejects_canvas_tamper(self) -> None:
        boards = synthetic_boards()
        permutation = np.arange(staged.NUM_TILES - 1, -1, -1, dtype=np.int64)
        target = np.zeros((staged.IMAGE_SIZE, staged.IMAGE_SIZE, 3), dtype=np.uint8)
        pinned = arm_metrics(
            solve=staged.PINNED_RR96_MEAN_SOLVE_SSIM,
            final=staged.PINNED_RR96_MEAN_FINAL_SSIM,
            neighbour=1.0,
        )
        pinned["placement"] = 1.0
        pinned["right"] = 1.0
        pinned["down"] = 1.0
        pinned["board_sha256"] = staged.array_sha256(boards.rr96_board)
        pinned["solved_corrupted_canvas_sha256"] = staged.array_sha256(
            boards.rr96_solved
        )
        pinned["restored_canvas_sha256"] = staged.array_sha256(
            boards.rr96_restored
        )
        provenance = {
            key: digest(2000 + index)
            for index, key in enumerate(sorted(staged._BROKER_ROW_PROVENANCE_KEYS))
        }
        seen: list[int] = []

        def candidate_ssim(_target, canvas):
            seen.append(int(canvas[0, 0, 0]))
            return float(canvas[0, 0, 0]) / 255.0

        row = staged.measure_scene_with_pinned_rr96(
            boards=boards,
            permutation=permutation,
            target=target,
            validation_name="synthetic.png",
            provenance=provenance,
            pinned_rr96=pinned,
            expected_permutation_sha256=staged.array_sha256(permutation),
            expected_target_sha256=staged.array_sha256(target),
            ssim=candidate_ssim,
        )
        self.assertEqual(seen, [10, 30])
        self.assertEqual(
            row["rr96"]["solve_only_ssim"], staged.PINNED_RR96_MEAN_SOLVE_SSIM
        )
        tampered = dict(pinned)
        tampered["restored_canvas_sha256"] = digest(9999)
        with self.assertRaises(staged.E24StagedContractError):
            staged.measure_scene_with_pinned_rr96(
                boards=boards,
                permutation=permutation,
                target=target,
                validation_name="synthetic.png",
                provenance=provenance,
                pinned_rr96=tampered,
                expected_permutation_sha256=staged.array_sha256(permutation),
                expected_target_sha256=staged.array_sha256(target),
                ssim=candidate_ssim,
            )

    def test_all_five_inclusive_gates_pass_at_literal_boundary(self) -> None:
        final = [0.010, 0.010, 0.010, 0.010, 0.010, -0.020, -0.007, -0.007]
        rows = [
            staged_row(
                image,
                solve_delta=0.003,
                final_delta=final[index],
                neighbour_delta=0.005,
            )
            for index, image in enumerate(staged.CALIBRATION_IDS)
        ]
        summary = staged.summarize_staged(rows)
        decision = staged.staged_decision(summary)
        self.assertEqual(summary["final_ssim_wins"], 5)
        self.assertEqual(summary["worst_final_ssim_delta"], -0.020)
        self.assertTrue(decision["passed"], decision)
        self.assertTrue(all(decision["checks"].values()))

    def test_one_ulp_below_and_zero_final_ties_fail(self) -> None:
        below = float(np.nextafter(0.003, -np.inf))
        rows = [
            staged_row(image, solve_delta=below, final_delta=0.002, neighbour_delta=0.005)
            for image in staged.CALIBRATION_IDS
        ]
        self.assertFalse(staged.staged_decision(staged.summarize_staged(rows))["passed"])
        ties = [
            staged_row(image, solve_delta=0.003, final_delta=0.0, neighbour_delta=0.005)
            for image in staged.CALIBRATION_IDS
        ]
        summary = staged.summarize_staged(ties)
        self.assertEqual(summary["final_ssim_wins"], 0)
        self.assertFalse(staged.staged_decision(summary)["passed"])

    def test_truncated_nonfinite_and_wrong_orientation_fail_closed(self) -> None:
        rows = [
            staged_row(image, solve_delta=0.003, final_delta=0.002, neighbour_delta=0.005)
            for image in staged.CALIBRATION_IDS
        ]
        with self.assertRaises(staged.E24StagedContractError):
            staged.summarize_staged(rows[:-1])
        bad = copy.deepcopy(rows)
        bad[0]["candidate"]["final_ssim"] = float("nan")
        with self.assertRaises(staged.E24StagedContractError):
            staged.summarize_staged(bad)
        rotated = copy.deepcopy(rows)
        rotated[0]["orientation_degrees"] = 90
        with self.assertRaises(staged.E24StagedContractError):
            staged.summarize_staged(rotated)

    def test_report_writer_validator_and_create_once_are_exact(self) -> None:
        rows = broker_rows()
        board_hashes = {
            str(image): digest(800 + image) for image in staged.CALIBRATION_IDS
        }
        metric_root = self.root / "metric_chain"
        write_metric_chain(metric_root, rows, board_hashes)
        report = staged.build_staged_report(
            ledger_sha256=digest(500),
            run_contract_sha256=digest(501),
            premetric_seal_sha256=digest(502),
            structural_report_sha256=digest(503),
            orchestration_receipt_sha256=digest(504),
            board_barrier_sha256=digest(700),
            board_commit_sha256=board_hashes,
            rows=rows,
            rr96_verification=staged.rr96_verification_for_rows(rows),
        )
        path = self.root / "report.json"
        staged.commit_canonical_create_once(path, report)
        observed = staged.validate_staged_report(
            path,
            expected_ledger_sha256=digest(500),
            expected_run_contract_sha256=digest(501),
            expected_premetric_seal_sha256=digest(502),
            expected_structural_report_sha256=digest(503),
            expected_orchestration_receipt_sha256=digest(504),
            expected_board_barrier_sha256=digest(700),
            expected_board_commit_sha256=board_hashes,
            expected_metric_response_root=metric_root,
        )
        self.assertEqual(observed, report)
        self.assertTrue(observed["decision"]["passed"])
        with self.assertRaises(staged.E24StagedContractError):
            staged.commit_canonical_create_once(path, report)

    def test_report_rejects_barrier_tamper_order_and_request_replay(self) -> None:
        rows = broker_rows()
        board_hashes = {
            str(image): digest(800 + image) for image in staged.CALIBRATION_IDS
        }
        with self.assertRaises(staged.E24StagedContractError):
            staged.build_staged_report(
                ledger_sha256=digest(500),
                run_contract_sha256=digest(501),
                premetric_seal_sha256=digest(502),
                structural_report_sha256=digest(503),
                orchestration_receipt_sha256=digest(504),
                board_barrier_sha256=digest(701),
                board_commit_sha256=board_hashes,
                rows=rows,
                rr96_verification=staged.rr96_verification_for_rows(rows),
            )
        reordered = copy.deepcopy(rows)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with self.assertRaises(staged.E24StagedContractError):
            staged.rr96_verification_for_rows(reordered)
        replayed = copy.deepcopy(rows)
        replayed[1]["provenance"]["metric_request_sha256"] = replayed[0]["provenance"][
            "metric_request_sha256"
        ]
        with self.assertRaises(staged.E24StagedContractError):
            staged.build_staged_report(
                ledger_sha256=digest(500),
                run_contract_sha256=digest(501),
                premetric_seal_sha256=digest(502),
                structural_report_sha256=digest(503),
                orchestration_receipt_sha256=digest(504),
                board_barrier_sha256=digest(700),
                board_commit_sha256=board_hashes,
                rows=replayed,
                rr96_verification=staged.rr96_verification_for_rows(replayed),
            )

    def test_writes_outside_e24_root_are_rejected(self) -> None:
        with self.assertRaises(staged.E24StagedContractError):
            staged.commit_canonical_create_once(ROOT / "forbidden.json", {"x": 1})


if __name__ == "__main__":
    unittest.main()
