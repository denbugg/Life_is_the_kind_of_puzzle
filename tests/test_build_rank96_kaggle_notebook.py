from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import build_rank96_kaggle_notebook as builder  # noqa: E402


def _literal_entry(contract: dict[str, object], extra: str = "") -> str:
    return f"RANK96_CONTRACT = {contract!r}\n{extra}\n"


class Rank96KaggleBuilderTests(unittest.TestCase):
    def test_frozen_contract_contains_no_rotation_and_exact_winner(self) -> None:
        contract = builder.EXPECTED_INFERENCE_CONTRACT
        self.assertEqual(contract["orientation"], "fixed")
        self.assertEqual(contract["candidate_k_per_encoder"], 64)
        self.assertEqual(contract["max_edges"], 96)
        self.assertEqual(contract["min_margin"], 0.0)
        self.assertEqual(contract["repair_passes"], 0)
        self.assertEqual(contract["nlm_h"], 10)
        self.assertEqual(contract["nlm_h_color"], 10)
        self.assertLess(builder.MAX_INFERENCE_SECONDS, 2 * 60 * 60)
        self.assertLess(builder.GRACEFUL_RUNTIME_SECONDS, builder.MAX_INFERENCE_SECONDS)
        self.assertEqual(set(builder.EXPECTED_CHECKPOINTS), {
            "ranker", "affinity_primary", "affinity_secondary"
        })

    def test_literal_contract_is_parsed_without_import_and_drift_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            entry = Path(temporary) / "infer_rank96.py"
            entry.write_text(
                _literal_entry(builder.EXPECTED_INFERENCE_CONTRACT, "raise RuntimeError('must not import')"),
                encoding="utf-8",
            )
            actual = builder.extract_rank96_contract(entry)
            builder.validate_inference_contract(actual)
            changed = copy.deepcopy(actual)
            changed["max_edges"] = 512
            with self.assertRaisesRegex(builder.BuildContractError, "differs"):
                builder.validate_inference_contract(changed)
            del changed["orientation"]
            with self.assertRaisesRegex(builder.BuildContractError, "missing"):
                builder.validate_inference_contract(changed)

    def test_source_closure_is_transitive_and_excludes_unrelated_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            src = Path(temporary)
            (src / "infer_rank96.py").write_text("from alpha import thing\n", encoding="utf-8")
            (src / "alpha.py").write_text("import beta\nthing = 1\n", encoding="utf-8")
            (src / "beta.py").write_text("value = 2\n", encoding="utf-8")
            (src / "unrelated.py").write_text("SECRET = 'DO_NOT_EMBED'\n", encoding="utf-8")
            names = [path.name for path in builder.discover_source_closure(src)]
        self.assertEqual(names, ["alpha.py", "beta.py", "infer_rank96.py"])

    def test_real_frozen_proof_and_checkpoint_hashes_validate(self) -> None:
        proof = builder._load_champion_proof(ROOT)
        self.assertEqual(proof["scene_count"], 24)
        self.assertGreater(proof["solve_ssim_delta"], 0.0)
        self.assertGreater(proof["final_ssim_delta"], 0.0)
        records, contents = builder._validate_checkpoints(ROOT)
        self.assertEqual(set(records), set(builder.EXPECTED_CHECKPOINTS))
        for role, record in records.items():
            self.assertEqual(record["sha256"], builder.EXPECTED_CHECKPOINTS[role]["sha256"])
            self.assertEqual(record["bytes"], len(contents[role]))

    def test_override_validation_requires_actual_rgb_480_png(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid"
            valid.mkdir()
            Image.new("RGB", (480, 480), (1, 2, 3)).save(valid / "test_000.png")
            records, contents = builder._validate_overrides(valid)
            self.assertEqual(set(records), {"test_000.png"})
            self.assertEqual(
                records["test_000.png"]["sha256"],
                hashlib.sha256(contents["test_000.png"]).hexdigest(),
            )
            invalid = root / "invalid"
            invalid.mkdir()
            Image.new("L", (480, 480), 0).save(invalid / "bad.png")
            with self.assertRaisesRegex(builder.BuildContractError, "RGB 480x480"):
                builder._validate_overrides(invalid)

    def test_minimal_bundle_is_deterministic_self_contained_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            src = workspace / "src"
            src.mkdir(parents=True)
            (src / "infer_rank96.py").write_text(
                _literal_entry(builder.EXPECTED_INFERENCE_CONTRACT, "from helper import run\n"),
                encoding="utf-8",
            )
            (src / "helper.py").write_text("def run(): return 96\n", encoding="utf-8")
            (src / "legacy_secret.py").write_text("WANDB_API_KEY='DO_NOT_EMBED'\n", encoding="utf-8")
            fake_contents = {
                role: f"checkpoint:{role}".encode("ascii")
                for role in builder.EXPECTED_CHECKPOINTS
            }
            fake_records = {
                role: {
                    "filename": expected["filename"],
                    "sha256": hashlib.sha256(fake_contents[role]).hexdigest(),
                    "bytes": len(fake_contents[role]),
                }
                for role, expected in builder.EXPECTED_CHECKPOINTS.items()
            }
            output = workspace / "bundle" / "rank96.ipynb"
            champion = {
                "path": "proof.json",
                "sha256": "a" * 64,
                "gate_root_sha256": builder.EXPECTED_GATE_ROOT_SHA256,
                "scene_count": 24,
                "solve_ssim_delta": 0.001,
                "final_ssim_delta": 0.005,
            }
            with (
                mock.patch.object(builder, "_load_champion_proof", return_value=champion),
                mock.patch.object(
                    builder,
                    "_validate_checkpoints",
                    return_value=(fake_records, fake_contents),
                ),
            ):
                manifest = builder.build_rank96_notebook(workspace=workspace, output=output)
                first = output.read_bytes()
                second_manifest = builder.build_rank96_notebook(workspace=workspace, output=output)
            self.assertEqual(first, output.read_bytes())
            self.assertEqual(manifest, second_manifest)
            self.assertEqual(manifest["notebook"]["sha256"], hashlib.sha256(first).hexdigest())
            self.assertEqual(set(manifest["sources"]), {"helper.py", "infer_rank96.py"})
            self.assertFalse(manifest["runtime"]["spatial_checkpoint_required"])
            notebook = json.loads(first)
            all_code = "\n".join(
                cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"
            )
            self.assertIn('"--max-runtime-seconds", "6900"', all_code)
            self.assertIn('"--expected-count", str(EXPECTED_IMAGES)', all_code)
            self.assertIn('"--dry-run"', all_code)
            self.assertIn('"--resume"', all_code)
            self.assertIn("submission.zip", all_code)
            self.assertIn("exactly 700", all_code)
            self.assertNotIn("DO_NOT_EMBED", all_code)
            self.assertNotIn("WANDB_API", all_code)
            output.write_text("divergent", encoding="utf-8")
            with (
                mock.patch.object(builder, "_load_champion_proof", return_value=champion),
                mock.patch.object(
                    builder,
                    "_validate_checkpoints",
                    return_value=(fake_records, fake_contents),
                ),
                self.assertRaisesRegex(builder.BuildContractError, "refusing to overwrite"),
            ):
                builder.build_rank96_notebook(workspace=workspace, output=output)


if __name__ == "__main__":
    unittest.main()
