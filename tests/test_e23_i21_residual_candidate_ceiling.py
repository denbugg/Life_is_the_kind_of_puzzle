from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import eval_e23_i21_residual_candidate_ceiling as gate  # noqa: E402


E_TEST_ROOT = Path("E:/pazzle_work/posegraph_e23/test_tmp")


def _raw_inputs(*, offset: int = 300) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.zeros((576, 128), dtype=np.int64)
    logits = np.full((4, 576, 128), -np.inf, dtype=np.float32)
    for source in range(576):
        candidates[source, 0] = (source + offset) % 576
        logits[:, source, 0] = np.asarray((1.0, 2.0, 3.0, 4.0), dtype=np.float32)
    return candidates, logits


def _tiles() -> np.ndarray:
    return np.zeros((576, 20, 20, 3), dtype=np.uint8, order="C")


def _spatial(value: float = 0.0) -> np.ndarray:
    return np.full((4, 576, 576), value, dtype=np.float32, order="C")


def _high_diversity_spatial() -> np.ndarray:
    values = np.full((4, 576, 576), -10_000.0, dtype=np.float32)
    for direction in range(4):
        first = direction * 64 + 1
        for source in range(576):
            for rank, delta in enumerate(range(first, first + 64)):
                values[direction, source, (source + delta) % 576] = np.float32(
                    1000 - rank
                )
    return values


def _checkpoint_record() -> dict[str, object]:
    return {
        "path": "E:\\dummy\\checkpoint.pt",
        "size_bytes": gate.EXPECTED_CHECKPOINT_SIZE,
        "sha256": gate.EXPECTED_CHECKPOINT_SHA256,
        "step": gate.EXPECTED_CHECKPOINT_STEP,
        "model_args": dict(gate.EXPECTED_MODEL_ARGS),
        "model_dependencies": {
            "config.py": {"path": "E:\\dummy\\config.py", "sha256": "a" * 64},
            "eval_paired_alignment.py": {
                "path": "E:\\dummy\\eval_paired_alignment.py",
                "sha256": "b" * 64,
            },
            "positional_ddpm.py": {
                "path": "E:\\dummy\\positional_ddpm.py",
                "sha256": "c" * 64,
            },
        },
    }


class FrozenContractTests(unittest.TestCase):
    def test_literal_constants_and_decision_caps(self) -> None:
        self.assertEqual(gate.SPATIAL_K, 64)
        self.assertEqual(gate.SPATIAL_LOGIT_VALUES, 1_327_104)
        self.assertEqual(gate.SPATIAL_SELECTIONS, 147_456)
        self.assertEqual(gate.MAX_UNORDERED_PAIRS, 165_600)
        self.assertEqual(gate.MAX_NEW_LITERAL_CLAIMS, 589_824)
        self.assertEqual(gate.MAX_COMBINED_LITERAL_CLAIMS, 662_400)
        self.assertEqual(gate.CACHE_PAYLOAD_NBYTES, 5_308_416)
        self.assertEqual(gate.CACHE_NPY_HEADER_BYTES, 128)
        self.assertEqual(gate.CACHE_NPY_FILE_BYTES, 5_308_544)
        self.assertEqual(gate.CACHE_METADATA_MAX_BYTES, 65_536)
        self.assertEqual(gate.DECISION_RULE["spatial_new_pairs_max_each"], 100_000)
        self.assertEqual(
            gate.DECISION_RULE["spatial_geometry_valid_hypotheses_max_each"],
            450_000,
        )

    def test_core_signature_and_direction_order_are_exact(self) -> None:
        self.assertEqual(
            tuple(inspect.signature(gate.e23_core.run_i21_residual_candidate_oracle).parameters),
            ("candidate_ids", "raw_logits", "spatial_logits"),
        )
        self.assertEqual(
            (gate.e22_core.UP, gate.e22_core.DOWN, gate.e22_core.LEFT, gate.e22_core.RIGHT),
            (0, 1, 2, 3),
        )
        self.assertEqual(gate.e23_core.DIRECTION_NAMES, ("U", "D", "L", "R"))

    def test_protocol_contains_null_dependencies_runtime_and_no_sweep(self) -> None:
        checkpoint = gate.E23_PROTOCOL["checkpoint"]
        self.assertEqual(
            checkpoint["model_dependency_sha256"],
            {
                "positional_ddpm.py": gate.EXPECTED_POSITIONAL_SOURCE_SHA256,
                "eval_paired_alignment.py": gate.EXPECTED_ALIGNMENT_SOURCE_SHA256,
                "config.py": gate.EXPECTED_CONFIG_SOURCE_SHA256,
            },
        )
        self.assertEqual(
            gate.E23_PROTOCOL["matched_budget_null"]["logits"],
            "rank_0_through_575_gets_exact_float32_575_minus_rank",
        )
        self.assertEqual(
            gate.E23_PROTOCOL["cache"]["existing_cache_policy"],
            "always_recompute_checkpoint_logits_and_byte_compare",
        )
        self.assertIn("K32", gate.E23_PROTOCOL["excluded"])
        self.assertIn("alternate_null", gate.E23_PROTOCOL["excluded"])

    def test_source_has_no_ddim_cuda_autocast_board_or_image_metric_calls(self) -> None:
        source = Path(gate.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
        self.assertTrue(
            {
                "ddim_sample",
                "cuda",
                "autocast",
                "assemble",
                "solve_buddies_from_scores",
                "structural_similarity",
                "fastNlMeansDenoisingColored",
            }.isdisjoint(calls)
        )

    def test_model_inference_calls_only_frozen_two_method_sequence(self) -> None:
        source = inspect.getsource(gate.infer_spatial_logits)
        self.assertLess(source.index("model.encode_tiles"), source.index("model.directional_edge_scores"))
        self.assertNotIn("model(", source)
        self.assertNotIn("predict_noise", source)

    def test_e_drive_guard_rejects_c_and_accepts_e(self) -> None:
        with self.assertRaisesRegex(gate.E23ContractError, "E:"):
            gate._require_e_drive(Path("C:/temp/nope"), label="test")
        self.assertEqual(gate._require_e_drive(Path("E:/pazzle_work"), label="test").drive, "E:")


class RuntimeAndInferenceTests(unittest.TestCase):
    def test_runtime_configuration_is_exact_and_idempotent(self) -> None:
        first = gate._configure_frozen_cpu_runtime()
        second = gate._configure_frozen_cpu_runtime()
        self.assertEqual(first, gate.EXPECTED_CPU_RUNTIME_CONFIGURATION)
        self.assertEqual(second, first)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.backends.mkldnn.enabled)
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertEqual(torch.get_num_interop_threads(), 1)

    def test_runtime_manifest_has_exact_package_and_configuration_sections(self) -> None:
        manifest = gate._runtime_provenance()
        self.assertEqual(set(manifest), {"packages", "configuration"})
        self.assertEqual(manifest["packages"], gate.EXPECTED_E22_RUNTIME_PROVENANCE)
        self.assertEqual(manifest["configuration"], gate.EXPECTED_CPU_RUNTIME_CONFIGURATION)

    def test_cpu_float32_eval_inference_shape_range_and_call_order(self) -> None:
        calls: list[str] = []

        class FakeModel:
            training = False

            def parameters(self):
                return ()

            def buffers(self):
                return ()

            def encode_tiles(self, value: torch.Tensor) -> torch.Tensor:
                calls.append("encode_tiles")
                self.assert_tensor = value
                if value.device.type != "cpu" or value.dtype != torch.float32:
                    raise AssertionError("wrong input device/dtype")
                if tuple(value.shape) != (1, 576, 3, 20, 20) or not value.is_contiguous():
                    raise AssertionError("wrong model input shape/order")
                if float(value.min()) != 0.0 or float(value.max()) != 0.0:
                    raise AssertionError("wrong [0,1] conversion")
                return torch.zeros((1, 576, 2), dtype=torch.float32)

            def directional_edge_scores(self, features: torch.Tensor) -> torch.Tensor:
                calls.append("directional_edge_scores")
                if tuple(features.shape) != (1, 576, 2):
                    raise AssertionError("wrong feature handoff")
                return torch.zeros((1, 4, 576, 576), dtype=torch.float32)

        result = gate.infer_spatial_logits(_tiles(), FakeModel())  # type: ignore[arg-type]
        self.assertEqual(calls, ["encode_tiles", "directional_edge_scores"])
        self.assertEqual(result.shape, (4, 576, 576))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)

    def test_inference_rejects_training_model(self) -> None:
        fake = SimpleNamespace(training=True, parameters=lambda: (), buffers=lambda: ())
        with self.assertRaisesRegex(gate.E23ContractError, "evaluation mode"):
            gate.infer_spatial_logits(_tiles(), fake)  # type: ignore[arg-type]

    def test_inference_rejects_nan_output(self) -> None:
        class FakeModel:
            training = False

            def parameters(self):
                return ()

            def buffers(self):
                return ()

            def encode_tiles(self, value):
                return torch.zeros((1, 576, 1), dtype=torch.float32)

            def directional_edge_scores(self, value):
                output = torch.zeros((1, 4, 576, 576), dtype=torch.float32)
                output[0, 0, 0, 0] = torch.nan
                return output

        with self.assertRaisesRegex(gate.E23ContractError, "output contract"):
            gate.infer_spatial_logits(_tiles(), FakeModel())  # type: ignore[arg-type]


class HashNullTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tiles_sha = "0" * 64
        cls.values = gate.hash_null_spatial_logits(cls.tiles_sha)

    def test_full_tensor_contract_and_exact_rank_set(self) -> None:
        values = self.values
        self.assertEqual(values.shape, (4, 576, 576))
        self.assertEqual(values.dtype, np.float32)
        self.assertTrue(values.flags.c_contiguous)
        expected = np.arange(576, dtype=np.float32)
        for direction, anchor in ((0, 0), (1, 23), (2, 511), (3, 575)):
            self.assertTrue(np.array_equal(np.sort(values[direction, anchor]), expected))

    def test_canonical_ascii_digest_order_replays_rank_zero(self) -> None:
        direction = 2
        anchor = 17
        ordered = sorted(
            range(576),
            key=lambda target: (
                hashlib.sha256(
                    f"E23-hash-null-v1|{self.tiles_sha}|{anchor}|{direction}|{target}".encode(
                        "ascii"
                    )
                ).digest(),
                target,
            ),
        )
        row = self.values[direction, anchor]
        self.assertEqual(int(np.argmax(row)), ordered[0])
        self.assertEqual(float(row[ordered[0]]), 575.0)
        self.assertEqual(float(row[ordered[-1]]), 0.0)

    def test_directions_are_literal_and_change_hash_records(self) -> None:
        self.assertFalse(np.array_equal(self.values[0, 9], self.values[1, 9]))
        self.assertFalse(np.array_equal(self.values[1, 9], self.values[2, 9]))

    def test_uppercase_or_malformed_tiles_hash_fails_closed(self) -> None:
        with self.assertRaisesRegex(gate.E23ContractError, "lowercase"):
            gate.hash_null_spatial_logits("A" * 64)
        with self.assertRaisesRegex(gate.E23ContractError, "lowercase"):
            gate.hash_null_spatial_logits("0" * 63)


class ResidualPreflightTests(unittest.TestCase):
    def test_zero_scores_use_tile_id_tie_break_and_exclude_self_and_base(self) -> None:
        base = ((0, 1, 0, None), (0, 2, 1, None))
        selected, nominations = gate._expected_residual_selection(_spatial(), base)
        self.assertEqual(selected.shape, (4, 576, 64))
        self.assertTrue(np.array_equal(selected[0, 0], np.arange(3, 67, dtype=np.int64)))
        self.assertNotIn(0, selected[0, 0])
        self.assertNotIn(1, selected[0, 0])
        self.assertNotIn(2, selected[0, 0])
        self.assertEqual(sum(nominations.values()), 147_456)

    def test_high_diversity_logits_exceed_100k_independently(self) -> None:
        selected, nominations = gate._expected_residual_selection(
            _high_diversity_spatial(), ()
        )
        self.assertEqual(selected.size, 147_456)
        self.assertEqual(len(nominations), 147_456)

    def test_pre_core_spatial_pair_guard_is_scientific_failure(self) -> None:
        candidates, logits = _raw_inputs(offset=300)
        with self.assertRaises(gate.E23ScientificGuardFailure) as caught:
            gate.preflight_spatial_deployability(
                image_id=10,
                candidate_ids=candidates,
                raw_logits=logits,
                spatial_logits=_high_diversity_spatial(),
            )
        payload = caught.exception.payload
        self.assertEqual(payload["guard"], "spatial_new_pairs_max_each")
        self.assertEqual(payload["phase"], "before_combined_core_construction")
        self.assertGreater(payload["observed"], payload["maximum"])

    def test_spatial_contract_rejects_nan_wrong_dtype_shape_and_layout(self) -> None:
        value = _spatial()
        cases = [
            value.astype(np.float64),
            value[:, :, :-1],
            value[:, :, ::-1],
        ]
        for case in cases:
            with self.subTest(shape=getattr(case, "shape", None)), self.assertRaises(
                gate.E23ContractError
            ):
                gate._validate_spatial_logits(case)
        bad = value.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaises(gate.E23ContractError):
            gate._validate_spatial_logits(bad)


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        E_TEST_ROOT.mkdir(parents=True, exist_ok=True)
        cls.runtime = gate._runtime_provenance()

    def _call(self, directory: Path, infer, *, force: bool = False, tiles=None):
        return gate.load_or_compute_spatial_logits(
            cache_dir=directory,
            image_id=10,
            validation_name="synthetic.png",
            tiles_uint8=_tiles() if tiles is None else tiles,
            model=object(),  # injected infer never consults it
            checkpoint_record=_checkpoint_record(),
            runtime_provenance=self.runtime,
            force_recompute=force,
            infer=infer,
        )

    def test_atomic_npy_json_roundtrip_exact_envelope_and_hit(self) -> None:
        with tempfile.TemporaryDirectory(dir=E_TEST_ROOT) as raw:
            directory = Path(raw)
            calls = 0

            def infer(_tiles_value, _model):
                nonlocal calls
                calls += 1
                return _spatial(1.25)

            first, record1 = self._call(directory, infer)
            second, record2 = self._call(directory, infer)
            self.assertEqual(calls, 1)
            self.assertFalse(record1.hit)
            self.assertTrue(record2.hit)
            self.assertEqual(record1.array_file_bytes, gate.CACHE_NPY_FILE_BYTES)
            self.assertEqual(record1.array_file_sha256, gate.e12.sha256_file(record1.array_path))
            self.assertTrue(np.array_equal(first, second))
            with record1.array_path.open("rb") as handle:
                self.assertEqual(np.lib.format.read_magic(handle), (1, 0))
                np.lib.format.read_array_header_1_0(handle, max_header_size=128)
                self.assertEqual(handle.tell(), 128)

    def test_force_recompute_matches_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(dir=E_TEST_ROOT) as raw:
            directory = Path(raw)
            self._call(directory, lambda *_: _spatial(2.0))
            before = next(directory.glob("*.npy")).read_bytes()
            values, record = self._call(
                directory, lambda *_: _spatial(2.0), force=True
            )
            self.assertTrue(record.hit)
            self.assertTrue(record.verified_by_recompute)
            self.assertEqual(before, record.array_path.read_bytes())
            self.assertEqual(float(values[0, 0, 0]), 2.0)

    def test_force_recompute_mismatch_fails_and_preserves_old_bytes(self) -> None:
        with tempfile.TemporaryDirectory(dir=E_TEST_ROOT) as raw:
            directory = Path(raw)
            _values, record = self._call(directory, lambda *_: _spatial(2.0))
            before_sha = gate.e12.sha256_file(record.array_path)
            with self.assertRaisesRegex(gate.E23ContractError, "differs"):
                self._call(directory, lambda *_: _spatial(3.0), force=True)
            self.assertEqual(before_sha, gate.e12.sha256_file(record.array_path))

    def test_mutating_infer_is_detected_before_cache_write(self) -> None:
        with tempfile.TemporaryDirectory(dir=E_TEST_ROOT) as raw:
            directory = Path(raw)
            tiles = _tiles()

            def mutating(value, _model):
                value[0, 0, 0, 0] = 255
                return _spatial()

            with self.assertRaisesRegex(gate.E23ContractError, "mutated"):
                self._call(directory, mutating, tiles=tiles)
            self.assertEqual(list(directory.iterdir()), [])

    def test_byte_tamper_fails_before_array_use(self) -> None:
        with tempfile.TemporaryDirectory(dir=E_TEST_ROOT) as raw:
            directory = Path(raw)
            _values, record = self._call(directory, lambda *_: _spatial())
            with record.array_path.open("r+b") as handle:
                handle.seek(-1, 2)
                byte = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([byte[0] ^ 1]))
            with self.assertRaisesRegex(gate.E23ContractError, "authentication"):
                self._call(directory, lambda *_: (_ for _ in ()).throw(AssertionError()))

    def test_trailing_npy_bytes_rejected_by_exact_size_before_hash_or_load(self) -> None:
        with tempfile.TemporaryDirectory(dir=E_TEST_ROOT) as raw:
            directory = Path(raw)
            _values, record = self._call(directory, lambda *_: _spatial())
            with record.array_path.open("ab") as handle:
                handle.write(b"trailing")
            with mock.patch.object(gate.e12, "sha256_file", side_effect=AssertionError("hash called")), mock.patch.object(
                gate.np, "load", side_effect=AssertionError("load called")
            ):
                with self.assertRaisesRegex(gate.E23ContractError, "envelope"):
                    self._call(directory, lambda *_: _spatial())

    def test_oversized_metadata_rejected_before_json_read(self) -> None:
        with tempfile.TemporaryDirectory(dir=E_TEST_ROOT) as raw:
            directory = Path(raw)
            _values, record = self._call(directory, lambda *_: _spatial())
            record.metadata_path.write_bytes(b"{" + b" " * gate.CACHE_METADATA_MAX_BYTES + b"}")
            with mock.patch.object(gate, "_load_json", side_effect=AssertionError("JSON read")):
                with self.assertRaisesRegex(gate.E23ContractError, "metadata"):
                    self._call(directory, lambda *_: _spatial())

    def test_partial_cache_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=E_TEST_ROOT) as raw:
            directory = Path(raw)
            _values, record = self._call(directory, lambda *_: _spatial())
            record.metadata_path.unlink()
            with self.assertRaisesRegex(gate.E23ContractError, "partial"):
                self._call(directory, lambda *_: _spatial())

    def test_runtime_manifest_mismatch_changes_no_cache(self) -> None:
        with tempfile.TemporaryDirectory(dir=E_TEST_ROOT) as raw:
            directory = Path(raw)
            bad = copy.deepcopy(self.runtime)
            bad["configuration"]["torch_intraop_threads"] = 2
            with self.assertRaisesRegex(gate.E23ContractError, "runtime manifest"):
                gate.load_or_compute_spatial_logits(
                    cache_dir=directory,
                    image_id=10,
                    validation_name="synthetic.png",
                    tiles_uint8=_tiles(),
                    model=object(),
                    checkpoint_record=_checkpoint_record(),
                    runtime_provenance=bad,
                    infer=lambda *_: _spatial(),
                )
            self.assertEqual(list(directory.iterdir()), [])


class CheckpointAndLineageTests(unittest.TestCase):
    def test_real_checkpoint_and_three_model_dependencies_are_byte_pinned(self) -> None:
        record = gate._checkpoint_record(gate.DEFAULT_CHECKPOINT)
        self.assertEqual(record["size_bytes"], gate.EXPECTED_CHECKPOINT_SIZE)
        self.assertEqual(record["sha256"], gate.EXPECTED_CHECKPOINT_SHA256)
        dependencies = record["model_dependencies"]
        self.assertEqual(
            {name: value["sha256"] for name, value in dependencies.items()},
            {
                "positional_ddpm.py": gate.EXPECTED_POSITIONAL_SOURCE_SHA256,
                "eval_paired_alignment.py": gate.EXPECTED_ALIGNMENT_SOURCE_SHA256,
                "config.py": gate.EXPECTED_CONFIG_SOURCE_SHA256,
            },
        )

    def test_safe_checkpoint_loader_uses_weights_only_cpu_and_exact_args(self) -> None:
        calls: dict[str, object] = {}

        class FakeModel:
            def __init__(self, **kwargs):
                calls["args"] = kwargs
                self.training = True

            def load_state_dict(self, state, strict):
                calls["strict"] = strict
                return SimpleNamespace(missing_keys=[], unexpected_keys=[])

            def to(self, **kwargs):
                calls["to"] = kwargs
                return self

            def eval(self):
                self.training = False
                return self

            def parameters(self):
                return ()

            def buffers(self):
                return ()

        payload = {
            "model": {},
            "optimizer": {},
            "step": 6000,
            "metrics": {},
            "model_args": dict(gate.EXPECTED_MODEL_ARGS),
        }

        def fake_load(path, *, map_location, weights_only):
            calls["load"] = (path, map_location, weights_only)
            return payload

        with mock.patch.object(gate, "_checkpoint_record", return_value=_checkpoint_record()), mock.patch.object(
            gate.torch, "load", side_effect=fake_load
        ), mock.patch.object(gate.positional_ddpm, "PositionalDDPM", FakeModel):
            model, _record = gate.load_frozen_i21_model(Path("E:/dummy/checkpoint.pt"))
        self.assertFalse(model.training)
        self.assertEqual(calls["args"], gate.EXPECTED_MODEL_ARGS)
        self.assertEqual(calls["strict"], True)
        self.assertEqual(calls["load"][1:], ("cpu", True))

    def test_checkpoint_wrong_step_fails_closed(self) -> None:
        payload = {
            "model": {},
            "optimizer": {},
            "step": 5999,
            "metrics": {},
            "model_args": dict(gate.EXPECTED_MODEL_ARGS),
        }
        with mock.patch.object(gate, "_checkpoint_record", return_value=_checkpoint_record()), mock.patch.object(
            gate.torch, "load", return_value=payload
        ):
            with self.assertRaisesRegex(gate.E23ContractError, "step/model"):
                gate.load_frozen_i21_model(Path("E:/dummy/checkpoint.pt"))

    def test_exact_e22_kill_report_authenticates(self) -> None:
        report = gate._verify_e22_kill(gate.DEFAULT_E22_REPORT)
        self.assertEqual(report["stage"], gate.EXPECTED_E22_STAGE)
        self.assertFalse(report["decision"]["passed"])
        self.assertEqual(report["completed_images"], list(range(10, 18)))

    def test_label_free_scene_manifest_never_accesses_permutation_or_target(self) -> None:
        class TrapScene:
            image_id = 10
            validation_name = "synthetic.png"
            cache_path = Path("E:/dummy/raw.npz")
            cache_sha256 = "a" * 64
            candidate_ids, base_scores = _raw_inputs()
            tiles_uint8 = _tiles()

            @property
            def permutation(self):
                raise AssertionError("permutation accessed")

            @property
            def target_uint8(self):
                raise AssertionError("target accessed")

        record = gate._label_free_scene_provenance(TrapScene())
        self.assertEqual(
            set(record),
            {
                "image",
                "validation_name",
                "raw_cache_path",
                "raw_cache_sha256",
                "candidate_ids_sha256",
                "raw_logits_sha256",
                "tiles_uint8_sha256",
            },
        )
        self.assertNotIn("permutation", json.dumps(record).lower())
        self.assertNotIn("target", json.dumps(record).lower())

    def test_both_core_payloads_validate_before_first_permutation_access(self) -> None:
        calls: list[str] = []

        class TrapScene:
            image_id = 10
            validation_name = "synthetic.png"
            cache_sha256 = "a" * 64
            tiles_uint8 = _tiles()

            @property
            def permutation(self):
                calls.append("permutation")
                return np.arange(576, dtype=np.int64)

        context = {
            "component_e22_sha256": "a",
            "base_pairs_e22_sha256": "b",
            "spatial_pair_count": 1,
        }

        def core_payload(*args, **kwargs):
            calls.append("core")
            if calls.count("core") == 2:
                raise gate.E23ContractError("second core rejected")
            return {}, context

        fake_result = SimpleNamespace(hypotheses=())
        record = gate.SpatialCacheRecord(
            "a" * 64,
            Path("E:/a.npy"),
            Path("E:/a.json"),
            "b" * 64,
            gate.CACHE_NPY_FILE_BYTES,
            "c" * 64,
            True,
            False,
        )
        with mock.patch.object(gate, "_core_payload", side_effect=core_payload):
            with self.assertRaisesRegex(gate.E23ContractError, "second core"):
                gate.evaluate_scene_pair(
                    TrapScene(),
                    fake_result,
                    fake_result,
                    candidate_ids=np.zeros((576, 128), dtype=np.int64),
                    raw_logits=np.zeros((4, 576, 128), dtype=np.float32),
                    spatial_logits=_spatial(),
                    null_logits=_spatial(),
                    spatial_cache_record=record,
                    authorized_e22_row={},
                )
        self.assertEqual(calls, ["core", "core"])


def _synthetic_row(image: int) -> dict[str, object]:
    base = 1000
    s_spatial = 500
    s_null = 600
    spatial_incremental = 10
    null_incremental = 5
    spatial_efficiency = spatial_incremental / s_spatial
    null_efficiency = null_incremental / s_null

    def metrics(*, new: int, incremental: int, recall: float, hypotheses: int):
        eligible = 100
        combined_hits = int(round(recall * eligible))
        base_hits = combined_hits - incremental
        return {
            "tile_orientation_degrees": 0,
            "emitter_tiles": 576,
            "directed_valid_memberships": 576,
            "spatial_logit_values": 1_327_104,
            "spatial_selections": 147_456,
            "base_affinity_pairs": base,
            "spatial_pairs": new,
            "combined_pairs": base + new,
            "new_literal_rcce4_preclaims": 4 * new,
            "combined_literal_rcce4_preclaims": 4 * (base + new),
            "cross_component_claims": 4 * (base + new),
            "relation_candidates": 4 * (base + new),
            "geometry_valid_hypotheses": hypotheses,
            "theoretical_bounds_passed": True,
            "spatial_pair_deployability_guard": new <= 100_000,
            "spatial_geometry_deployability_guard": hypotheses <= 450_000,
            "eligible_contacts": 100,
            "e22_base_eligible_hits": base_hits,
            "e22_base_eligible_recall": base_hits / eligible,
            "combined_eligible_hits": combined_hits,
            "incremental_eligible_hits": incremental,
            "incremental_hit_efficiency": incremental / new,
            "postfilter_exact_physical_seam_survivors": combined_hits,
            "postfilter_eligible_true_survival": 1.0,
            "combined_eligible_recall": recall,
            "true_hypotheses": 5,
            "legal_origin_count": 1,
            "selected_exact_connected_coverage": 0.5,
            "selected_exact_connected_tiles": 288,
            "selected_components": 11,
            "selected_cycle_rank": 1,
            "selected_cycle_rank_ratio": 0.1,
        }

    spatial_metrics = metrics(
        new=s_spatial, incremental=spatial_incremental, recall=0.95, hypotheses=100
    )
    null_metrics = metrics(
        new=s_null, incremental=null_incremental, recall=0.90, hypotheses=120
    )
    return {
        "image": image,
        "orientation": "upright_0_degrees_no_rotation_no_reflection",
        "exact_e22_prefix_and_provenance_replay": True,
        "spatial": {"metrics": spatial_metrics},
        "hash_null": {"metrics": null_metrics},
        "comparison": {
            "S_spatial": s_spatial,
            "S_null": s_null,
            "spatial_incremental_eligible_hits": spatial_incremental,
            "null_incremental_eligible_hits": null_incremental,
            "spatial_incremental_hit_efficiency": spatial_efficiency,
            "null_incremental_hit_efficiency": null_efficiency,
            "null_efficiency_denominator_positive": True,
            "incremental_hit_efficiency_ratio": spatial_efficiency / null_efficiency,
            "spatial_minus_null_combined_recall": 0.95 - 0.90,
            "spatial_strict_recall_win": True,
        },
    }


class SummaryDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [_synthetic_row(image) for image in range(10, 18)]

    def test_all_predeclared_checks_pass_in_exact_synthetic_case(self) -> None:
        summary = gate.summarize(self.rows)
        result = gate.decision(summary)
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(summary["strict_spatial_recall_win_scenes"], 8)
        self.assertEqual(summary["null_complete_bounds_survival_scenes"], 8)

    def test_tampered_efficiency_ratio_fails_recomputation(self) -> None:
        self.rows[0]["comparison"]["incremental_hit_efficiency_ratio"] = 99.0
        with self.assertRaisesRegex(gate.E23ContractError, "comparison accounting"):
            gate.summarize(self.rows)

    def test_tampered_efficiency_value_fails_recomputation(self) -> None:
        self.rows[0]["comparison"]["spatial_incremental_hit_efficiency"] = 0.5
        with self.assertRaisesRegex(gate.E23ContractError, "comparison accounting"):
            gate.summarize(self.rows)

    def test_tampered_strict_win_flag_fails_recomputation(self) -> None:
        self.rows[0]["comparison"]["spatial_strict_recall_win"] = False
        with self.assertRaisesRegex(gate.E23ContractError, "comparison accounting"):
            gate.summarize(self.rows)

    def test_zero_null_efficiency_fails_decision(self) -> None:
        for row in self.rows:
            row["hash_null"]["metrics"]["combined_eligible_hits"] = 85
            row["hash_null"]["metrics"]["combined_eligible_recall"] = 0.85
            row["hash_null"]["metrics"]["incremental_eligible_hits"] = 0
            row["hash_null"]["metrics"]["incremental_hit_efficiency"] = 0.0
            row["hash_null"]["metrics"]["postfilter_exact_physical_seam_survivors"] = 85
            row["comparison"]["null_incremental_eligible_hits"] = 0
            row["comparison"]["null_incremental_hit_efficiency"] = 0.0
            row["comparison"]["null_efficiency_denominator_positive"] = False
            row["comparison"]["incremental_hit_efficiency_ratio"] = 0.0
            row["comparison"]["spatial_minus_null_combined_recall"] = 0.95 - 0.85
        result = gate.decision(gate.summarize(self.rows))
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["nonzero_null_efficiency"])
        self.assertFalse(result["checks"]["null_incremental_hits_each"])

    def test_spatial_geometry_guard_is_required(self) -> None:
        summary = gate.summarize(self.rows)
        summary["spatial_deployability_guard_scenes"] = 7
        result = gate.decision(summary)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["spatial_deployability_guards"])

    def test_comparative_recall_win_and_lift_are_all_required(self) -> None:
        for index, row in enumerate(self.rows):
            if index >= 5:
                row["spatial"]["metrics"]["combined_eligible_recall"] = 0.89
                row["spatial"]["metrics"]["combined_eligible_hits"] = 89
                row["spatial"]["metrics"]["incremental_eligible_hits"] = 4
                row["spatial"]["metrics"]["incremental_hit_efficiency"] = 4 / 500
                row["spatial"]["metrics"]["postfilter_exact_physical_seam_survivors"] = 89
                row["comparison"]["spatial_incremental_eligible_hits"] = 4
                row["comparison"]["spatial_incremental_hit_efficiency"] = 4 / 500
                row["comparison"]["incremental_hit_efficiency_ratio"] = (4 / 500) / (5 / 600)
                row["comparison"]["spatial_minus_null_combined_recall"] = 0.89 - 0.90
                row["comparison"]["spatial_strict_recall_win"] = False
        result = gate.decision(gate.summarize(self.rows))
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["strict_spatial_recall_wins"])

    def test_nan_summary_value_fails_closed(self) -> None:
        self.rows[0]["spatial"]["metrics"]["combined_eligible_recall"] = float("nan")
        with self.assertRaises(gate.E23ContractError):
            gate.summarize(self.rows)

    def test_zero_eligible_denominator_is_scientific_fail_not_division_error(self) -> None:
        for row in self.rows:
            for arm in ("spatial", "hash_null"):
                metrics = row[arm]["metrics"]
                metrics["eligible_contacts"] = 0
                metrics["e22_base_eligible_hits"] = 0
                metrics["e22_base_eligible_recall"] = 0.0
                metrics["combined_eligible_hits"] = 0
                metrics["combined_eligible_recall"] = 0.0
                metrics["incremental_eligible_hits"] = 0
                metrics["incremental_hit_efficiency"] = 0.0
                metrics["postfilter_exact_physical_seam_survivors"] = 0
                metrics["postfilter_eligible_true_survival"] = 0.0
            comparison = row["comparison"]
            comparison["spatial_incremental_eligible_hits"] = 0
            comparison["null_incremental_eligible_hits"] = 0
            comparison["spatial_incremental_hit_efficiency"] = 0.0
            comparison["null_incremental_hit_efficiency"] = 0.0
            comparison["null_efficiency_denominator_positive"] = False
            comparison["incremental_hit_efficiency_ratio"] = 0.0
            comparison["spatial_minus_null_combined_recall"] = 0.0
            comparison["spatial_strict_recall_win"] = False
        result = gate.decision(gate.summarize(self.rows))
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["spatial_positive_eligible_denominators"])
        self.assertFalse(result["checks"]["null_positive_eligible_denominators"])


class ScientificTerminalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        E_TEST_ROOT.mkdir(parents=True, exist_ok=True)

    def test_structured_guard_terminal_is_scientific_kill_not_execution_failure(self) -> None:
        payload = {
            "image": 10,
            "guard": "spatial_new_pairs_max_each",
            "observed": 100_001,
            "maximum": 100_000,
            "phase": "before_combined_core_construction",
            "evidence": {"spatial_selection_count": 147_456},
        }
        summary, result = gate._scientific_guard_terminal(payload, completed_scenes=0)
        self.assertEqual(summary["completed_scenes"], 0)
        self.assertFalse(result["passed"])
        self.assertIn("scientific", result["scope"])
        self.assertNotIn("failed", result["status"])

    def test_post_core_geometry_cap_raises_scientific_guard_before_labels(self) -> None:
        calls = 0

        class TrapScene:
            image_id = 10

            @property
            def permutation(self):
                raise AssertionError("labels accessed")

        context = {
            "component_e22_sha256": "a",
            "base_pairs_e22_sha256": "b",
            "spatial_pair_count": 1,
        }

        def fake_core(*args, **kwargs):
            nonlocal calls
            calls += 1
            return {}, context

        spatial = SimpleNamespace(hypotheses=(None,) * 450_001)
        null = SimpleNamespace(hypotheses=())
        with mock.patch.object(gate, "_core_payload", side_effect=fake_core):
            with self.assertRaises(gate.E23ScientificGuardFailure) as caught:
                gate.evaluate_scene_pair(
                    TrapScene(),
                    spatial,
                    null,
                    candidate_ids=np.zeros((576, 128), dtype=np.int64),
                    raw_logits=np.zeros((4, 576, 128), dtype=np.float32),
                    spatial_logits=_spatial(),
                    null_logits=_spatial(),
                    spatial_cache_record=mock.Mock(),
                    authorized_e22_row={},
                )
        self.assertEqual(calls, 1)
        self.assertEqual(
            caught.exception.payload["guard"],
            "spatial_geometry_valid_hypotheses_max_each",
        )

    def test_run_gate_writes_and_replays_complete_guard_kill(self) -> None:
        with tempfile.TemporaryDirectory(dir=E_TEST_ROOT) as raw:
            directory = Path(raw)
            paths = gate.E23Paths(
                raw_cache_dir=directory / "raw",
                calibration_report=directory / "calibration.json",
                e12_report=directory / "e12.json",
                e22_report=directory / "e22.json",
                checkpoint=directory / "checkpoint.pt",
                spatial_cache_dir=directory / "cache",
                report=directory / "report.json",
            )
            scenes = [SimpleNamespace(image_id=image) for image in range(10, 18)]
            e22_report = {"rows": [{"image": image} for image in range(10, 18)]}
            failure = gate.E23ScientificGuardFailure(
                image=10,
                guard="spatial_new_pairs_max_each",
                observed=100_001,
                maximum=100_000,
                phase="before_combined_core_construction",
            )
            stable_runtime = gate._runtime_provenance()
            patches = (
                mock.patch.object(gate, "_runtime_provenance", return_value=stable_runtime),
                mock.patch.object(gate, "_verify_e22_kill", return_value=e22_report),
                mock.patch.object(
                    gate,
                    "_load_verified_raw_inputs",
                    return_value=({}, {}, scenes),
                ),
                mock.patch.object(
                    gate, "load_frozen_i21_model", return_value=(object(), _checkpoint_record())
                ),
                mock.patch.object(gate, "_source_provenance", return_value={"x.py": "a" * 64}),
                mock.patch.object(
                    gate,
                    "_label_free_scene_provenance",
                    side_effect=lambda scene: {"image": scene.image_id},
                ),
                mock.patch.object(gate, "_run_scene_pair", side_effect=failure),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
                first = gate.run_gate(paths)
                second = gate.run_gate(paths)
            self.assertEqual(first, second)
            self.assertEqual(first["status"], "complete")
            self.assertFalse(first["decision"]["passed"])
            self.assertEqual(first["guard_failure"]["observed"], 100_001)
            self.assertNotIn("error", first)
            stored = json.loads(paths.report.read_text(encoding="utf-8"))
            self.assertEqual(stored["stage"], first["stage"])


class MutationBoundaryTests(unittest.TestCase):
    def _scene(self):
        candidates, logits = _raw_inputs()
        return SimpleNamespace(
            image_id=10,
            validation_name="synthetic.png",
            cache_sha256="a" * 64,
            candidate_ids=candidates,
            base_scores=logits,
            tiles_uint8=_tiles(),
        )

    @staticmethod
    def _record() -> gate.SpatialCacheRecord:
        return gate.SpatialCacheRecord(
            "a" * 64,
            Path("E:/a.npy"),
            Path("E:/a.json"),
            "b" * 64,
            gate.CACHE_NPY_FILE_BYTES,
            "c" * 64,
            True,
            True,
        )

    def test_mutating_preflight_is_detected_before_scientific_guard(self) -> None:
        scene = self._scene()
        spatial = _spatial()

        def mutating_preflight(**kwargs):
            kwargs["candidate_ids"][0, 0] = 7
            raise gate.E23ScientificGuardFailure(
                image=10,
                guard="spatial_new_pairs_max_each",
                observed=100_001,
                maximum=100_000,
                phase="before_combined_core_construction",
            )

        with mock.patch.object(
            gate, "load_or_compute_spatial_logits", return_value=(spatial, self._record())
        ), mock.patch.object(gate, "hash_null_spatial_logits", return_value=_spatial()), mock.patch.object(
            gate, "preflight_spatial_deployability", side_effect=mutating_preflight
        ), mock.patch.object(gate.e23_core, "run_i21_residual_candidate_oracle") as core:
            with self.assertRaisesRegex(gate.E23ContractError, "input mutated") as caught:
                gate._run_scene_pair(
                    scene,
                    authorized_e22_row={},
                    model=object(),
                    checkpoint_record={},
                    runtime_provenance={},
                    spatial_cache_dir=Path("E:/dummy"),
                    force_recompute_spatial_cache=True,
                )
        self.assertNotIsInstance(caught.exception, gate.E23ScientificGuardFailure)
        core.assert_not_called()

    def test_mutating_spatial_core_is_detected_before_geometry_guard(self) -> None:
        scene = self._scene()
        spatial = _spatial()

        def mutating_core(candidate_ids, raw_logits, spatial_logits):
            spatial_logits[0, 0, 0] = 99.0
            return SimpleNamespace(
                spatial_pairs=(None,), hypotheses=(None,) * 450_001
            )

        with mock.patch.object(
            gate, "load_or_compute_spatial_logits", return_value=(spatial, self._record())
        ), mock.patch.object(gate, "hash_null_spatial_logits", return_value=_spatial()), mock.patch.object(
            gate, "preflight_spatial_deployability", return_value=1
        ), mock.patch.object(
            gate.e23_core, "run_i21_residual_candidate_oracle", side_effect=mutating_core
        ) as core:
            with self.assertRaisesRegex(gate.E23ContractError, "input mutated") as caught:
                gate._run_scene_pair(
                    scene,
                    authorized_e22_row={},
                    model=object(),
                    checkpoint_record={},
                    runtime_provenance={},
                    spatial_cache_dir=Path("E:/dummy"),
                    force_recompute_spatial_cache=True,
                )
        self.assertNotIsInstance(caught.exception, gate.E23ScientificGuardFailure)
        self.assertEqual(core.call_count, 1)


if __name__ == "__main__":
    unittest.main()
