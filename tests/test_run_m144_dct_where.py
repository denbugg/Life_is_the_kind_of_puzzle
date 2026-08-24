from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import run_m144_dct_where as runner


def raw_fixture(count: int = runner.CAL_COUNT) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "board_id": np.arange(count, dtype=np.int64),
        "source_group_id": np.arange(count, dtype=np.int64),
        "swap_cycle_id": np.repeat(np.arange(count // 2), 2).astype(np.int64),
    }
    arrays.update({key: np.zeros(count, dtype=np.float64) for key in runner.SSIM_KEYS})
    arrays["flat_rgb"] = np.zeros((count, 3), dtype=np.float32)
    for key in ("dct_full_coeff", "dct_blind_coeff", "dct_swapped_coeff"):
        arrays[key] = np.zeros((count, runner.core.DCT_OUTPUT_DIM), dtype=np.float32)
    for key in ("rgb8_full_residual", "rgb8_blind_residual"):
        arrays[key] = np.zeros((count, runner.core.RGB_OUTPUT_DIM), dtype=np.float32)
    return arrays


def tiny_arms(seed: int = 77) -> dict[str, runner.Arm]:
    result: dict[str, runner.Arm] = {}
    for index, name in enumerate(runner.ARM_NAMES):
        torch.manual_seed(seed + index)
        model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.GELU(), torch.nn.Linear(5, 2))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=runner.LEARNING_RATE,
            betas=runner.ADAM_BETAS, weight_decay=runner.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=runner.LEARNING_RATE,
            total_steps=runner.TRAIN_STEPS, pct_start=0.05,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        result[name] = runner.Arm(
            name=name,
            kind="dct" if name.startswith("dct") else "rgb",
            blind=name.endswith("blind"),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
    return result


def tiny_step(arms: dict[str, runner.Arm], step: int) -> None:
    x = torch.tensor(
        [[0.1 + step * 1.0e-5, -0.2, 0.3], [0.4, 0.5, -0.6]], dtype=torch.float32
    )
    for arm in arms.values():
        arm.optimizer.zero_grad(set_to_none=True)
        loss = arm.model(x).square().mean()
        loss.backward()
        arm.optimizer.step()
        arm.scheduler.step()


class M144RunnerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="m144_runner_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_stateless_batch_schedule_is_resume_independent(self) -> None:
        first = [
            runner.stateless_batch_indices(17, 8, step, runner.core.BOOTSTRAP_SEED)
            for step in range(1, 9)
        ]
        repeated = [
            runner.stateless_batch_indices(17, 8, step, runner.core.BOOTSTRAP_SEED)
            for step in range(1, 9)
        ]
        for left, right in zip(first, repeated, strict=True):
            np.testing.assert_array_equal(left, right)
            self.assertEqual(left.dtype, np.int64)
        stream = np.concatenate(first)
        self.assertEqual(set(stream[:17].tolist()), set(range(17)))

    def test_dirty_feature60_exact_float64_dimension_major_formula(self) -> None:
        rng = np.random.default_rng(144)
        tiles = rng.integers(0, 256, size=(576, 20, 20, 3), dtype=np.uint8)
        observed = runner.dirty_feature60(tiles)
        x = tiles.astype(np.float64) / 255.0
        means = x.mean(axis=(1, 2), dtype=np.float64)
        centered = x - means[:, None, None, :]
        rms = np.sqrt(np.mean(centered * centered, axis=(1, 2, 3), dtype=np.float64))
        columns = np.column_stack((means, rms))
        expected = np.concatenate(
            (
                np.quantile(columns, np.linspace(0, 1, 13), axis=0, method="linear").T.reshape(-1),
                columns.mean(axis=0, dtype=np.float64),
                columns.std(axis=0, ddof=0, dtype=np.float64),
            )
        )
        self.assertEqual(observed.dtype, np.float64)
        np.testing.assert_array_equal(observed, expected)

    def test_palette_lsa_and_arbitrary_canonical_cycles(self) -> None:
        donor = np.asarray([1, 0, 3, 4, 2], dtype=np.int64)
        board = np.asarray([10, 20, 30, 40, 50], dtype=np.int64)
        np.testing.assert_array_equal(
            runner.canonical_cycle_ids(donor, board),
            np.asarray([0, 0, 1, 1, 1], dtype=np.int64),
        )
        rng = np.random.default_rng(8)
        feature = rng.normal(size=(10, runner.PALETTE_DIM)).astype(np.float64)
        groups = np.repeat(np.arange(5), 2).astype(np.int64)
        board = (100 + np.arange(10)).astype(np.int64)
        mean = rng.normal(size=runner.PALETTE_DIM).astype(np.float64)
        scale = rng.uniform(0.2, 2.0, size=runner.PALETTE_DIM).astype(np.float64)
        observed_donor, observed_cycle = runner.solve_swap_assignment(
            feature, groups, board, mean, scale
        )
        z = (feature - mean) / scale
        cost = np.vstack(
            [np.sum((z - z[index]) ** 2, axis=1, dtype=np.float64) for index in range(len(z))]
        )
        cost[groups[:, None] == groups[None, :]] = np.inf
        rows, expected_donor = linear_sum_assignment(cost)
        np.testing.assert_array_equal(rows, np.arange(10))
        np.testing.assert_array_equal(observed_donor, expected_donor.astype(np.int64))
        self.assertEqual(len(np.unique(observed_donor)), 10)
        self.assertTrue(np.all(observed_donor != np.arange(10)))
        self.assertTrue(np.all(groups[observed_donor] != groups))
        np.testing.assert_array_equal(
            observed_cycle, runner.canonical_cycle_ids(observed_donor, board)
        )

    def test_cycle_stats_exact_schema_and_minimum_is_fail_closed(self) -> None:
        cycle = np.asarray([0, 0, 1, 1, 1, 2, 2], dtype=np.int64)
        self.assertEqual(
            runner.swap_cycle_stats(cycle),
            {
                "count": 3, "min_size": 2, "max_size": 3,
                "mean_size": 7.0 / 3.0, "median_size": 2.0,
                "size_histogram": [[2, 2], [3, 1]],
            },
        )
        arrays = raw_fixture()
        arrays["swap_cycle_id"] = np.zeros(runner.CAL_COUNT, dtype=np.int64)
        with self.assertRaisesRegex(runner.ContractError, "64 bootstrap"):
            runner.validate_raw_arrays(arrays, count=runner.CAL_COUNT)

    def test_official_uint8_ssim_quantization_is_exact(self) -> None:
        rng = np.random.default_rng(9)
        target = rng.integers(0, 256, size=(2, 11, 13, 3), dtype=np.uint8)
        rendered = torch.from_numpy(np.moveaxis(target, -1, 1)).float().div_(255.0)
        score = runner.official_uint8_ssim(rendered, target)
        np.testing.assert_array_equal(score, np.ones(2, dtype=np.float64))
        np.testing.assert_array_equal(
            runner.quantize_render_uint8(rendered), np.moveaxis(target, -1, 1)
        )

    def test_raw_numeric_orphan_recovery_roundtrip_and_tamper_detection(self) -> None:
        paths = runner.RunPaths(self.root)
        paths.ensure()
        arrays = raw_fixture()
        artifact = paths.artifacts / "m144_cal_raw.npz"
        runner._atomic_npz(artifact, arrays)
        record = runner.write_raw_artifact(
            paths, "cal", arrays, "a" * 64, {"fixture": "authenticated"}
        )
        self.assertTrue((paths.receipts / "cal_raw.json").is_file())
        loaded = runner.load_raw_artifact(record, count=runner.CAL_COUNT)
        for key in runner.RAW_KEYS:
            np.testing.assert_array_equal(loaded[key], arrays[key])
        with Path(record["path"]).open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(runner.ContractError, "drift"):
            runner.load_raw_artifact(record, count=runner.CAL_COUNT)

    def test_cache_manifest_authenticates_palette_and_is_dirty_only(self) -> None:
        paths = runner.RunPaths(self.root)
        paths.ensure()
        files = runner.cache_files(paths, "cal")
        names = ("img_000001.png", "img_000002.png")
        np.save(files.embeddings, np.zeros((2, 576, 128), dtype=np.float16))
        np.save(files.flat, np.zeros((2, 3), dtype=np.float32))
        np.save(files.palette, np.zeros((2, runner.PALETTE_DIM), dtype=np.float64))
        # the manifest binds the representation contract by path record and
        # digest, so the fixture has to provide a real one on disk
        representation = paths.artifacts / "representation_contract.json"
        runner.atomic_json(representation, {"representation_sha256": "b" * 64})
        manifest = {
            "schema": runner.CACHE_SCHEMA, "partition": "cal",
            "contract_sha256": "c" * 64, "count": 2,
            "names_sha256": runner._names_digest(names), "embedding_dim": 128,
            "representation_contract": runner.path_record(representation),
            "representation_sha256": "b" * 64,
            "embeddings": runner.path_record(files.embeddings),
            "flat": runner.path_record(files.flat),
            "palette": runner.path_record(files.palette),
            "boards": [
                {"name": name, "input_bytes": 10, "input_sha256": "d" * 64}
                for name in names
            ],
        }
        runner.atomic_json(files.manifest, manifest)
        runner._verify_cache_manifest(files, "c" * 64, names)
        palette = np.load(files.palette, mmap_mode="r+")
        palette[0, 0] = 1.0
        palette.flush()
        del palette
        with self.assertRaisesRegex(runner.ContractError, "drift"):
            runner._verify_cache_manifest(files, "c" * 64, names)

    def test_checkpoint_orphan_resume_matches_uninterrupted_step(self) -> None:
        paths = runner.RunPaths(self.root)
        paths.ensure()
        contract_sha = "e" * 64
        uninterrupted = tiny_arms()
        for step in range(1, 101):
            tiny_step(uninterrupted, step)
        payload = runner._checkpoint_payload(
            uninterrupted, step=100, contract_sha256=contract_sha,
            losses={name: 0.5 for name in runner.ARM_NAMES},
        )
        runner._atomic_torch(paths.checkpoints / "step_0000100.pt", payload)
        tiny_step(uninterrupted, 101)
        resumed = tiny_arms()
        step, receipt = runner.load_latest_checkpoint(paths, resumed, contract_sha)
        self.assertEqual(step, 100)
        self.assertIsNotNone(receipt)
        self.assertTrue((paths.receipts / "checkpoint_step_0000100.json").is_file())
        tiny_step(resumed, 101)
        for name in runner.ARM_NAMES:
            for expected, observed in zip(
                uninterrupted[name].model.parameters(), resumed[name].model.parameters(), strict=True
            ):
                torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)

    def test_gate_map_uses_canonical_passed_key(self) -> None:
        arrays = raw_fixture()
        arrays["flat_ssim"].fill(0.20)
        arrays["target_oracle_dct_ssim"].fill(0.241)
        arrays["dct_full_ssim"].fill(0.210)
        arrays["dct_blind_ssim"].fill(0.206)
        arrays["dct_swapped_ssim"].fill(0.207)
        arrays["rgb8_full_ssim"].fill(0.204)
        arrays["rgb8_blind_ssim"].fill(0.202)
        metrics = runner.metric_summary(arrays, alpha=0.10)
        gates = runner.gate_map(metrics, "CAL")
        self.assertIn("passed", gates)
        self.assertNotIn("pass", gates)
        self.assertTrue(gates["passed"])
        arrays["target_oracle_dct_ssim"].fill(0.239)
        failed = runner.gate_map(runner.metric_summary(arrays, alpha=0.10), "CAL")
        self.assertFalse(failed["passed"])

    def test_run_cache_never_opens_dev_or_targets(self) -> None:
        paths = runner.RunPaths(self.root)
        paths.ensure()
        split = runner.SplitData(
            names={"fit": ("img_000001.png",), "cal": ("img_000002.png",),
                   "dev": ("img_000003.png",), "reserve": ()},
            group_for_name={}, group_id_for_name={},
        )
        manifest_path = self.root / "source_manifest.json"
        runner.atomic_json(manifest_path, {"files": {
            name: {"sha256": "a" * 64}
            for name in ("img_000001.png", "img_000002.png", "img_000003.png")
        }})
        args = mock.Mock(device="cuda", amp=True, cache_chunk=8,
                         source_manifest=manifest_path)
        built: list[str] = []
        with (
            mock.patch.object(runner, "require_cuda", return_value=torch.device("cpu")),
            mock.patch.object(runner, "enforce_resource_caps", return_value={}),
            mock.patch.object(runner, "run_capacity_smoke", return_value={}),
            mock.patch.object(
                runner, "build_embedding_cache",
                side_effect=lambda **kwargs: built.append(kwargs["partition"]),
            ),
            mock.patch.object(runner, "build_or_load_fit_palette", return_value=({}, {})),
            mock.patch.object(runner, "write_status"),
            mock.patch.object(runner, "train_fit_only_encoder",
                              return_value=(object(), {"checkpoint": {}})),
            mock.patch.object(runner, "evaluate_fit_encoder_cal", return_value={}),
            mock.patch.object(runner, "write_or_load_encoder_gate",
                              return_value=({}, {}, {}, {"passed": True})),
            mock.patch.object(runner, "write_or_load_representation_contract",
                              return_value=({}, {})),
            mock.patch.object(runner, "authenticate_partition_targets") as targets,
        ):
            runner.run_cache(args, paths, split, {"contract_sha256": "f" * 64})
        self.assertEqual(built, ["fit", "cal"])
        # Targets are now PINNED by hash during the cache stage, which the
        # original version of this test predated.  The guard that matters is
        # unchanged and is asserted more tightly here: fit and cal only, and
        # the dev split is never named.
        authenticated = [call.kwargs["partition"] for call in targets.call_args_list]
        self.assertEqual(authenticated, ["fit", "cal"])
        named = [name for call in targets.call_args_list for name in call.kwargs["names"]]
        self.assertNotIn("img_000003.png", named)

    def test_atomic_json_lock_and_e_only_guard_fail_closed(self) -> None:
        path = self.root / "value.json"
        runner.atomic_json(path, {"z": 1, "a": 2})
        self.assertEqual(path.read_bytes(), b'{"a":2,"z":1}')
        with self.assertRaisesRegex(runner.ContractError, "must be on E"):
            runner._require_drive_e(self.root, "test root")
        paths = runner.RunPaths(self.root / "lock")
        paths.ensure()
        runner.acquire_run_lock(paths)
        with self.assertRaisesRegex(runner.ContractError, "already locked"):
            runner.acquire_run_lock(paths)
        runner.release_run_lock(paths)
        self.assertFalse(paths.lock.exists())


if __name__ == "__main__":
    unittest.main()
