from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import infer_e24 as e24  # noqa: E402


E_TEST_ROOT = Path("E:/pazzle_work/posegraph_e24_selector/production_test_tmp")
E_TEST_ROOT.mkdir(parents=True, exist_ok=True)


def _tempdir():
    return tempfile.TemporaryDirectory(dir=E_TEST_ROOT)


def _image(offset: int = 0) -> np.ndarray:
    row = np.arange(480, dtype=np.uint16)[:, None]
    col = np.arange(480, dtype=np.uint16)[None, :]
    value = np.empty((480, 480, 3), dtype=np.uint8)
    value[..., 0] = (row + offset) % 251
    value[..., 1] = (col + 2 * offset) % 253
    value[..., 2] = (row + col + 3 * offset) % 255
    return value


def _png_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(value, mode="RGB").save(
        stream, format="PNG", optimize=False, compress_level=6
    )
    return stream.getvalue()


def _save_png(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png_bytes(value))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(e24._canonical_json_bytes(value))


def _write_e24_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(e24._e24_canonical_json_bytes(value))


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _config(root: Path, *, expected_count: int = 1) -> e24.InferenceConfig:
    production = root / "production"
    return e24.InferenceConfig(
        input_dir=root / "inputs",
        output_dir=production / "outputs",
        output_zip=production / "submission_e24.zip",
        baseline_zip=root / "baseline.zip",
        baseline_manifest=root / "baseline_manifest.json",
        ledger=root / "authority/ledger.json",
        canary_gate=root / "authority/canary.json",
        structural_report=root / "authority/structural.json",
        orchestration_receipt=root / "authority/receipt.json",
        ranker_checkpoint=root / "ranker.pt",
        affinity_primary_checkpoint=root / "affinity1.pt",
        affinity_secondary_checkpoint=root / "affinity2.pt",
        i21_checkpoint=root / "i21.pt",
        production_root=production,
        device="cpu",
        expected_count=expected_count,
    )


def _authority_files(config: e24.InferenceConfig, root: Path) -> e24.FrozenOOFAuthority:
    source = root / "frozen_source.py"
    source.write_bytes(b"frozen\n")
    run_sha = "a" * 64
    feature_sha = hashlib.sha256(
        e24._e24_canonical_json_bytes({"feature_names": list(e24.selector.FEATURE_NAMES)})
    ).hexdigest()
    ledger = {
        "schema": e24.LEDGER_SCHEMA,
        "status": "frozen_preflight_only",
        "metrics_opened": False,
        "target_artifacts_created": False,
        "staged_board_ssim_nlm": "sealed",
        "e25": {"opened": False},
        "run_contract_sha256": run_sha,
        "ordered_feature_names": list(e24.selector.FEATURE_NAMES),
        "ordered_feature_names_sha256": feature_sha,
        "core_protocol": json.loads(
            e24._e24_canonical_json_bytes(e24.selector.PROTOCOL).decode("ascii")
        ),
        "lightgbm": {"config": dict(e24.selector.LIGHTGBM_CONFIG)},
        "runtime_versions": {
            "python": e24.platform.python_version(),
            "python_implementation": e24.platform.python_implementation(),
            "numpy": np.__version__,
            "torch": e24.importlib.metadata.version("torch"),
            "scikit-image": e24.importlib.metadata.version("scikit-image"),
            "scipy": e24.importlib.metadata.version("scipy"),
            "opencv-python": e24.importlib.metadata.version("opencv-python"),
            "Pillow": e24.importlib.metadata.version("Pillow"),
            "lightgbm": e24.importlib.metadata.version("lightgbm"),
        },
        "sources": {str(source.resolve()): e24.sha256_file(source)},
    }
    _write_e24_json(config.ledger, ledger)
    ledger_sha = e24.sha256_file(config.ledger)
    canary = {
        "schema": e24.CANARY_SCHEMA,
        "status": "pass",
        "passed": True,
        "ledger_sha256": ledger_sha,
        "run_contract_sha256": run_sha,
        "receipt_sha256": "b" * 64,
        "thresholds": {},
        "observed": {},
        "checks": {"bounded": True},
        "labels_or_metrics_opened": False,
    }
    _write_e24_json(config.canary_gate, canary)
    canary_sha = e24.sha256_file(config.canary_gate)
    structural = {
        "schema": e24.STRUCTURAL_SCHEMA,
        "status": "complete",
        "stage": "go_staged_end_to_end",
        "ledger_sha256": ledger_sha,
        "run_contract_sha256": run_sha,
        "fold_commit_sha256": {"0": "c" * 64},
        "rows": [],
        "summary": {},
        "staged_board_ssim_nlm": "sealed_not_run",
        "e25_opened": False,
        "decision": {
            "passed": True,
            "stage": "go_staged_end_to_end",
            "checks": {"all_frozen_gates": True},
        },
    }
    _write_e24_json(config.structural_report, structural)
    structural_sha = e24.sha256_file(config.structural_report)
    receipt = {
        "schema": e24.ORCHESTRATION_SCHEMA,
        "status": "pass",
        "ledger_sha256": ledger_sha,
        "run_contract_sha256": run_sha,
        "canary_gate_sha256": canary_sha,
        "structural_report_sha256": structural_sha,
        "resource": {},
        "checks": {"resources": True},
    }
    _write_e24_json(config.orchestration_receipt, receipt)
    receipt_sha = e24.sha256_file(config.orchestration_receipt)
    return e24.authenticate_frozen_oof_authority(config)


def _fake_baseline(root: Path, names: list[str], overrides: list[str]) -> tuple[Path, Path]:
    zip_path = root / "baseline.zip"
    payloads = {name: _png_bytes(_image(index + 7)) for index, name in enumerate(names)}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in names:
            archive.writestr(name, payloads[name])
    override_rows = [{"name": name, "sha256": _sha(payloads[name])} for name in overrides]
    completed = {
        name: {
            "source": "verified_source_override",
            "output_sha256": _sha(payloads[name]),
            "override_sha256": _sha(payloads[name]),
        }
        for name in overrides
    }
    manifest = {
        "schema": e24.rank96.MANIFEST_SCHEMA,
        "status": "completed",
        "contract": {
            "pipeline": e24.rank96.RANK96_CONTRACT,
            "overrides": override_rows,
            "inputs": [
                {"name": name, "sha256": hashlib.sha256(name.encode()).hexdigest()}
                for name in names
            ],
            "code": e24.rank96._code_provenance(),
            "checkpoints": {"synthetic": {"sha256": "f" * 64}},
        },
        "completed": completed,
    }
    manifest_path = root / "baseline_manifest.json"
    _write_json(manifest_path, manifest)
    return zip_path, manifest_path


class FrozenProductionContractTests(unittest.TestCase):
    def test_contract_is_literal_upright_streaming_nlm10(self) -> None:
        source = (SRC / "infer_e24.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "PRODUCTION_CONTRACT"
        )
        literal = ast.literal_eval(assignment.value)
        self.assertEqual(literal, e24.PRODUCTION_CONTRACT)
        self.assertEqual(literal["orientation"], "upright_fixed_no_rotation_or_reflection")
        self.assertEqual((literal["repair_passes"], literal["restarts"]), (0, 1))
        self.assertEqual((literal["nlm_h"], literal["nlm_h_color"]), (10, 10))
        self.assertIn("one_image_only", literal["features"])
        self.assertNotIn("np.rot90", source)
        self.assertNotIn("save_feature_table_npz", source)

    def test_baseline_identity_is_fully_pinned(self) -> None:
        self.assertEqual(e24.BASELINE_ZIP_SIZE, 222_050_278)
        self.assertEqual(
            e24.BASELINE_ZIP_SHA256,
            "9a2eaf962507d11f2cad0caf59af40fe9755a6f092051c9d144a5f6aca10965f",
        )
        self.assertEqual(e24.BASELINE_OVERRIDE_COUNT, 18)

    def test_cli_write_defaults_are_on_e(self) -> None:
        args = e24.build_parser().parse_args([])
        for path in (args.output_dir, args.output_zip):
            self.assertEqual(path.drive.upper(), "E:")
        self.assertEqual(args.expected_count, 700)
        self.assertFalse(args.resume)

    def test_smoke_is_upright_and_data_free(self) -> None:
        result = e24.smoke_contract()
        self.assertEqual(result["status"], "smoke_pass")
        self.assertEqual(result["feature_count"], 227)
        self.assertIn("no_rotation", result["orientation"])


class AuthorityTests(unittest.TestCase):
    def test_exact_hash_linked_oof_pass_is_verified_then_production_stays_sealed(self) -> None:
        with _tempdir() as directory:
            root = Path(directory)
            config = _config(root)
            authority = _authority_files(config, root)
            self.assertEqual(authority.ledger_sha256, e24.sha256_file(config.ledger))
            self.assertEqual(
                authority.structural_report_sha256,
                e24.sha256_file(config.structural_report),
            )
            with self.assertRaisesRegex(e24.E24InferenceError, "production remains sealed"):
                e24.authenticate_e24_authority(config)

    def test_failed_structural_decision_is_rejected(self) -> None:
        with _tempdir() as directory:
            root = Path(directory)
            config = _config(root)
            _authority_files(config, root)
            report = json.loads(config.structural_report.read_text(encoding="utf-8"))
            report["decision"]["passed"] = False
            report["decision"]["stage"] = "kill_crs_v1"
            _write_e24_json(config.structural_report, report)
            with self.assertRaisesRegex(e24.E24InferenceError, "structural report"):
                e24.authenticate_e24_authority(config)

    def test_run_refuses_before_target_inventory_when_authority_is_absent(self) -> None:
        with _tempdir() as directory:
            config = _config(Path(directory))
            inventory = Mock(side_effect=AssertionError("target inventory must stay sealed"))
            with (
                patch.object(e24, "authenticate_e24_authority", side_effect=e24.E24InferenceError("no PASS")),
                patch.object(e24, "_list_input_names", inventory),
            ):
                with self.assertRaisesRegex(e24.E24InferenceError, "no PASS"):
                    e24.run_inference(config)
            inventory.assert_not_called()

    def test_opened_scene17_parity_precedes_first_target_directory_access(self) -> None:
        with _tempdir() as directory:
            root = Path(directory)
            config = _config(root)
            model = root / "model.txt"
            model.write_bytes(b"model")
            authority = e24.Authority(
                ledger_sha256="1" * 64,
                run_contract_sha256="2" * 64,
                structural_report_sha256="3" * 64,
                orchestration_receipt_sha256="4" * 64,
                canary_gate_sha256="5" * 64,
                final_model_manifest_sha256="6" * 64,
                ordered_feature_names_sha256="7" * 64,
                final_model_path=model,
                final_model_sha256=e24.sha256_file(model),
            )
            baseline = e24.BaselineBundle(
                zip_path=config.baseline_zip,
                zip_sha256="8" * 64,
                manifest_sha256="9" * 64,
                names=("img_000001.png",),
                overrides={},
                rank96_code={},
                rank96_checkpoints={"ranker": {"sha256": "a" * 64}},
            )
            order: list[str] = []

            def canary(*_args, **_kwargs):
                order.append("scene17")
                return {"status": "pass"}

            def target_inventory(_path):
                self.assertEqual(order, ["scene17"])
                raise e24.E24InferenceError("ordered-stop")

            with (
                patch.object(e24, "authenticate_e24_authority", return_value=authority),
                patch.object(e24, "authenticate_baseline", return_value=baseline),
                patch.object(e24, "_configure_i21_runtime_and_checkpoint", return_value={}),
                patch.object(e24, "_rank96_checkpoint_provenance", return_value=baseline.rank96_checkpoints),
                patch.object(e24.rank96, "resolve_device", return_value="cpu"),
                patch.object(e24, "_code_provenance", return_value={}),
                patch.object(e24, "load_production_models", return_value=object()),
                patch.object(e24, "run_label_free_production_canary", side_effect=canary),
                patch.object(e24, "_list_input_names", side_effect=target_inventory),
            ):
                with self.assertRaisesRegex(e24.E24InferenceError, "ordered-stop"):
                    e24.run_inference(config)


class BaselineAndTailTests(unittest.TestCase):
    def test_all_18_overrides_are_authenticated_against_zip_and_manifest(self) -> None:
        with _tempdir() as directory:
            root = Path(directory)
            names = [f"img_{index:06d}.png" for index in range(18)]
            zip_path, manifest_path = _fake_baseline(root, names, names)
            bundle = e24.authenticate_baseline(
                zip_path,
                manifest_path,
                names,
                expected_zip_size=zip_path.stat().st_size,
                expected_zip_sha256=e24.sha256_file(zip_path),
                expected_manifest_sha256=e24.sha256_file(manifest_path),
            )
            self.assertEqual(len(bundle.overrides), 18)
            first = names[0]
            self.assertEqual(_sha(e24._read_baseline_override(bundle, first)), bundle.overrides[first])

    def test_component_tail_has_exact_no_repair_single_restart_contract(self) -> None:
        tiles = e24.rank96.split_upright_tiles(_image())
        right = np.zeros((576, 576), dtype=np.float32)
        down = np.zeros_like(right)
        components = ({0: (0, 0)},)
        observed: dict[str, object] = {}

        def solver(r, d, c, **kwargs):
            observed.update(kwargs)
            self.assertIs(c, components)
            return np.arange(576), 4.5

        output, board, objective = e24.solve_components_tail(
            tiles,
            right,
            down,
            components,
            solver=solver,
            restorer=lambda value: value,
        )
        self.assertEqual(observed, {"repair_passes": 0, "restarts": 1, "seed": 1234})
        self.assertTrue(np.array_equal(output, _image()))
        self.assertTrue(np.array_equal(board, np.arange(576)))
        self.assertEqual(objective, 4.5)

    def test_pending_zip_rejects_member_changed_after_completed_hash(self) -> None:
        with _tempdir() as directory:
            root = Path(directory)
            output = root / "production/png"
            output.mkdir(parents=True)
            name = "img_000001.png"
            _save_png(output / name, _image(1))
            expected = e24.sha256_file(output / name)
            _save_png(output / name, _image(2))
            baseline_path, _manifest_path = _fake_baseline(root, [name], [])
            baseline = e24.BaselineBundle(
                zip_path=baseline_path,
                zip_sha256=e24.sha256_file(baseline_path),
                manifest_sha256="e" * 64,
                names=(name,),
                overrides={},
                rank96_code={},
                rank96_checkpoints={},
            )
            with self.assertRaisesRegex(e24.E24InferenceError, "completed record"):
                e24._build_verified_pending_zip(
                    output,
                    [name],
                    root / "production/submission.zip.pending",
                    baseline,
                    {name: expected},
                )


class StreamingResumeTests(unittest.TestCase):
    def test_override_plus_generic_stream_resume_and_deterministic_zip(self) -> None:
        with _tempdir() as directory:
            root = Path(directory)
            config = _config(root, expected_count=2)
            config.input_dir.mkdir(parents=True)
            names = ["img_000001.png", "img_000002.png"]
            for index, name in enumerate(names):
                _save_png(config.input_dir / name, _image(index))
            baseline_zip, baseline_manifest = _fake_baseline(root, names, [names[0]])
            baseline_sha = e24.sha256_file(baseline_zip)
            override_content = zipfile.ZipFile(baseline_zip).read(names[0])
            baseline = e24.BaselineBundle(
                zip_path=baseline_zip,
                zip_sha256=baseline_sha,
                manifest_sha256=e24.sha256_file(baseline_manifest),
                names=tuple(names),
                overrides={names[0]: _sha(override_content)},
                rank96_code={},
                rank96_checkpoints={"ranker": {"sha256": "c" * 64}},
            )
            model_path = root / "model.txt"
            model_path.write_bytes(b"model")
            authority = e24.Authority(
                ledger_sha256="1" * 64,
                run_contract_sha256="2" * 64,
                structural_report_sha256="3" * 64,
                orchestration_receipt_sha256="4" * 64,
                canary_gate_sha256="5" * 64,
                final_model_manifest_sha256="6" * 64,
                ordered_feature_names_sha256="7" * 64,
                final_model_path=model_path,
                final_model_sha256=e24.sha256_file(model_path),
            )
            inferred = e24.InferredImage(
                output=_image(30),
                board=np.arange(576),
                objective=8.0,
                candidate_ids_sha256="8" * 64,
                raw_scores_sha256="9" * 64,
                spatial_logits_sha256="a" * 64,
                relation_rows=100,
                relation_queries=20,
                proposed_relations=10,
                accepted_relations=8,
                tree_merges=7,
                cycle_acceptances=1,
            )
            generic = Mock(return_value=inferred)
            common = (
                patch.object(e24, "authenticate_e24_authority", return_value=authority),
                patch.object(e24, "authenticate_baseline", return_value=baseline),
                patch.object(e24, "_configure_i21_runtime_and_checkpoint", return_value={"sha256": "b" * 64}),
                patch.object(e24, "_rank96_checkpoint_provenance", return_value={"ranker": {"sha256": "c" * 64}}),
                patch.object(e24.rank96, "resolve_device", return_value="cpu"),
                patch.object(e24, "_code_provenance", return_value={"infer_e24.py": "d" * 64}),
                patch.object(e24, "load_production_models", return_value=object()),
                patch.object(e24, "run_label_free_production_canary", return_value={"status": "pass"}),
                patch.object(e24, "infer_one_e24", generic),
                patch.object(e24, "BASELINE_ZIP_SIZE", baseline_zip.stat().st_size),
                patch.object(e24, "BASELINE_ZIP_SHA256", baseline_sha),
            )
            with common[0], common[1], common[2], common[3], common[4], common[5], common[6], common[7], common[8], common[9], common[10]:
                first = e24.run_inference(config)
            self.assertEqual(first["status"], "completed")
            self.assertEqual((first["override_count"], first["generic_count"]), (1, 1))
            generic.assert_called_once()
            self.assertEqual((config.output_dir / names[0]).read_bytes(), override_content)
            with zipfile.ZipFile(config.output_zip) as archive:
                self.assertEqual(archive.namelist(), names)
                self.assertIsNone(archive.testzip())
            first_zip_sha = e24.sha256_file(config.output_zip)
            self.assertEqual(e24.sha256_file(config.baseline_zip), baseline_sha)

            generic.reset_mock()
            resumed_config = replace(config, resume=True)
            common2 = (
                patch.object(e24, "authenticate_e24_authority", return_value=authority),
                patch.object(e24, "authenticate_baseline", return_value=baseline),
                patch.object(e24, "_configure_i21_runtime_and_checkpoint", return_value={"sha256": "b" * 64}),
                patch.object(e24, "_rank96_checkpoint_provenance", return_value={"ranker": {"sha256": "c" * 64}}),
                patch.object(e24.rank96, "resolve_device", return_value="cpu"),
                patch.object(e24, "_code_provenance", return_value={"infer_e24.py": "d" * 64}),
                patch.object(e24, "load_production_models", return_value=object()),
                patch.object(e24, "run_label_free_production_canary", return_value={"status": "pass"}),
                patch.object(e24, "infer_one_e24", generic),
                patch.object(e24, "BASELINE_ZIP_SIZE", baseline_zip.stat().st_size),
                patch.object(e24, "BASELINE_ZIP_SHA256", baseline_sha),
            )
            with common2[0], common2[1], common2[2], common2[3], common2[4], common2[5], common2[6], common2[7], common2[8], common2[9], common2[10]:
                resumed = e24.run_inference(resumed_config)
            # A completed publish is immutable: resume returns its original
            # signed report instead of rewriting counters after final ZIP.
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(resumed["completed_count"], 2)
            generic.assert_not_called()
            self.assertEqual(e24.sha256_file(config.output_zip), first_zip_sha)

    def test_new_zip_cannot_alias_immutable_baseline(self) -> None:
        with _tempdir() as directory:
            root = Path(directory)
            config = _config(root)
            with self.assertRaisesRegex(e24.E24InferenceError, "never overwrite"):
                e24._validate_config(replace(config, output_zip=config.baseline_zip))


if __name__ == "__main__":
    unittest.main()
