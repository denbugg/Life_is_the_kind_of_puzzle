from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import infer_rank96 as rank96  # noqa: E402


def _image(offset: int = 0) -> np.ndarray:
    row = np.arange(rank96.IMAGE_SIZE, dtype=np.uint16)[:, None]
    col = np.arange(rank96.IMAGE_SIZE, dtype=np.uint16)[None, :]
    value = np.empty((rank96.IMAGE_SIZE, rank96.IMAGE_SIZE, 3), dtype=np.uint8)
    value[..., 0] = (row + offset) % 251
    value[..., 1] = (col + 2 * offset) % 253
    value[..., 2] = (row + col + 3 * offset) % 255
    return value


def _save(path: Path, value: np.ndarray, *, mode: str = "RGB") -> None:
    Image.fromarray(value, mode=mode).save(path, format="PNG")


def _fake_checkpoints() -> dict[str, dict[str, object]]:
    return {
        role: {"sha256": digest, "size": 1}
        for role, digest in rank96.EXPECTED_CHECKPOINT_SHA256.items()
    }


def _config(root: Path, *, resume: bool = False) -> rank96.InferenceConfig:
    return rank96.InferenceConfig(
        input_dir=root / "inputs",
        output_dir=root / "outputs",
        output_zip=None,
        ranker_checkpoint=root / "ranker.pt",
        affinity_primary_checkpoint=root / "affinity1.pt",
        affinity_secondary_checkpoint=root / "affinity2.pt",
        device="cpu",
        resume=resume,
        override_dir=root / "overrides",
        expected_count=1,
    )


class FrozenContractTests(unittest.TestCase):
    def test_contract_is_literal_and_exact(self) -> None:
        source = (SRC / "infer_rank96.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "RANK96_CONTRACT"
        )
        literal = ast.literal_eval(assignment.value)
        self.assertEqual(literal, rank96.RANK96_CONTRACT)
        self.assertEqual(literal["schema"], "pazzle-rank96-inference-v1")
        self.assertEqual(
            (
                literal["grid"],
                literal["tile_size"],
                literal["image_size"],
                literal["num_tiles"],
            ),
            (24, 20, 480, 576),
        )
        self.assertEqual(literal["orientation"], "fixed")
        self.assertEqual(literal["candidate_k_per_encoder"], 64)
        self.assertEqual(
            (literal["max_edges"], literal["min_margin"], literal["repair_passes"]),
            (96, 0.0, 0),
        )
        self.assertEqual((literal["nlm_h"], literal["nlm_h_color"]), (10, 10))
        self.assertNotIn("spatial", json.dumps(literal).lower())

    def test_confirmed_checkpoint_hashes_are_pinned(self) -> None:
        self.assertEqual(
            rank96.EXPECTED_CHECKPOINT_SHA256,
            {
                "ranker": "42685373b1a450a4cb3d7a9b22370dfcfaa2335e9e8ada609f21b7cc64abbfbc",
                "affinity_primary": "708565329c7661a965215d98e85f462a90930071f36a0f75b4813c0c5797ec4f",
                "affinity_secondary": "0fceafdb110bde59149fe1ad1e800a69d116041bc627af369aaecd60be53b6c8",
            },
        )

    def test_ranker_graph_requires_the_exact_dual_k64_union(self) -> None:
        payload = {
            "candidate_graph": {
                "per_encoder_top_k": 64,
                "union": True,
                "max_candidates_per_row": 128,
                "encoders": [
                    {"sha256": rank96.EXPECTED_CHECKPOINT_SHA256["affinity_primary"]},
                    {"sha256": rank96.EXPECTED_CHECKPOINT_SHA256["affinity_secondary"]},
                ],
            }
        }
        self.assertEqual(
            rank96._ranker_graph_hashes(payload),
            [
                rank96.EXPECTED_CHECKPOINT_SHA256["affinity_primary"],
                rank96.EXPECTED_CHECKPOINT_SHA256["affinity_secondary"],
            ],
        )
        payload["candidate_graph"]["per_encoder_top_k"] = 63
        with self.assertRaises(rank96.Rank96Error):
            rank96._ranker_graph_hashes(payload)


class FixedOrientationTests(unittest.TestCase):
    def test_upright_split_and_identity_assembly_are_exact(self) -> None:
        image = _image()
        tiles = rank96.split_upright_tiles(image)
        self.assertEqual(tiles.shape, (576, 20, 20, 3))
        rebuilt = rank96.assemble_upright_tiles(tiles, np.arange(576))
        self.assertTrue(np.array_equal(image, rebuilt))
        self.assertTrue(np.array_equal(tiles[0], image[:20, :20]))
        self.assertTrue(np.array_equal(tiles[1], image[:20, 20:40]))

    def test_assembly_rejects_non_bijection(self) -> None:
        tiles = rank96.split_upright_tiles(_image())
        invalid = np.arange(576)
        invalid[-1] = 0
        with self.assertRaises(rank96.Rank96Error):
            rank96.assemble_upright_tiles(tiles, invalid)

    def test_strict_loader_rejects_mode_and_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            grayscale = root / "gray.png"
            Image.fromarray(np.zeros((480, 480), dtype=np.uint8), mode="L").save(grayscale)
            with self.assertRaisesRegex(rank96.Rank96Error, "mode"):
                rank96.load_rgb_strict(grayscale)
            small = root / "small.png"
            _save(small, np.zeros((32, 32, 3), dtype=np.uint8))
            with self.assertRaisesRegex(rank96.Rank96Error, "must be"):
                rank96.load_rgb_strict(small)

    def test_solver_tail_uses_only_budget96_and_no_repair(self) -> None:
        tiles = rank96.split_upright_tiles(_image())
        right = np.zeros((576, 576), dtype=np.float32)
        down = np.zeros_like(right)
        seen: dict[str, object] = {}

        def solver(r: np.ndarray, d: np.ndarray, **kwargs: object):
            seen.update(kwargs)
            self.assertIs(r.dtype, np.dtype(np.float32))
            self.assertIs(d.dtype, np.dtype(np.float32))
            return np.arange(576), 12.5

        output, board, objective = rank96.solve_dense_tiles(
            tiles,
            right,
            down,
            solver=solver,
            restorer=lambda image: image,
        )
        self.assertEqual(
            seen,
            {"max_edges": 96, "min_margin": 0.0, "repair_passes": 0},
        )
        self.assertTrue(np.array_equal(output, _image()))
        self.assertTrue(np.array_equal(board, np.arange(576)))
        self.assertEqual(objective, 12.5)


class CrashSafeResumeTests(unittest.TestCase):
    def _prepare(self, root: Path) -> None:
        (root / "inputs").mkdir()
        (root / "overrides").mkdir()
        _save(root / "inputs" / "img_000001.png", _image(1))
        _save(root / "overrides" / "img_000001.png", _image(7))

    def _run(self, config: rank96.InferenceConfig) -> dict[str, object]:
        with (
            patch.object(rank96, "resolve_device", return_value="cpu"),
            patch.object(rank96, "_checkpoint_provenance", return_value=_fake_checkpoints()),
            patch.object(rank96, "_code_provenance", return_value={"infer_rank96.py": "c" * 64}),
            patch.object(rank96, "load_models", side_effect=AssertionError("override must skip model load")),
            patch.object(rank96, "infer_one", side_effect=AssertionError("override must skip scoring")),
        ):
            return rank96.run_inference(config)

    def test_override_skips_gpu_path_and_resume_verifies_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._prepare(root)
            config = _config(root)
            first = self._run(config)
            self.assertEqual(first["status"], "completed")
            self.assertEqual(first["override_count"], 1)
            self.assertEqual(first["generic_count"], 0)
            output = rank96.load_rgb_strict(root / "outputs" / "img_000001.png")
            self.assertTrue(np.array_equal(output, _image(7)))
            manifest = json.loads(
                (root / "outputs" / "rank96_manifest.json").read_text(encoding="utf-8")
            )
            record = manifest["completed"]["img_000001.png"]
            self.assertEqual(record["source"], "verified_source_override")
            self.assertEqual(record["output_sha256"], rank96.sha256_file(root / "outputs" / "img_000001.png"))

            resumed = self._run(replace(config, resume=True))
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(resumed["skipped_count"], 1)
            self.assertEqual(resumed["new_count"], 0)
            self.assertEqual(resumed["override_count"], 1)

    def test_resume_fails_closed_on_modified_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._prepare(root)
            config = _config(root)
            self._run(config)
            _save(root / "outputs" / "img_000001.png", _image(11))
            with self.assertRaisesRegex(rank96.Rank96Error, "output hash mismatch"):
                self._run(replace(config, resume=True))

    def test_resume_fails_closed_on_modified_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._prepare(root)
            config = _config(root)
            self._run(config)
            _save(root / "inputs" / "img_000001.png", _image(12))
            with self.assertRaisesRegex(rank96.Rank96Error, "different inputs"):
                self._run(replace(config, resume=True))

    def test_existing_png_without_manifest_is_never_blindly_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._prepare(root)
            (root / "outputs").mkdir()
            _save(root / "outputs" / "img_000001.png", _image(7))
            with self.assertRaisesRegex(rank96.Rank96Error, "no matching manifest"):
                self._run(replace(_config(root), resume=True))


class CliAndSmokeTests(unittest.TestCase):
    def test_cli_defaults_to_complete_700_and_has_runtime_resume_controls(self) -> None:
        args = rank96.build_parser().parse_args([])
        self.assertEqual(args.expected_count, 700)
        self.assertEqual(args.limit, 0)
        self.assertEqual(args.max_runtime_seconds, 0.0)
        self.assertFalse(args.resume)
        self.assertFalse(args.dry_run)

    def test_data_free_smoke_passes(self) -> None:
        result = rank96.smoke_contract()
        self.assertEqual(result["status"], "smoke_pass")
        self.assertEqual(result["tiles_shape"], [576, 20, 20, 3])
        self.assertTrue(result["identity_roundtrip"])
        self.assertEqual(result["orientation"], "fixed")

    def test_complete_zip_is_deterministic_and_uses_only_basenames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "outputs"
            output_dir.mkdir()
            names = ["img_000002.png", "img_000009.png"]
            for ordinal, name in enumerate(names):
                rank96._atomic_write_png(output_dir / name, _image(ordinal))
            first = root / "first.zip"
            second = root / "second.zip"
            first_digest = rank96._deterministic_zip(output_dir, names, first)
            second_digest = rank96._deterministic_zip(output_dir, names, second)
            self.assertEqual(first_digest, second_digest)
            with zipfile.ZipFile(first) as archive:
                self.assertEqual(archive.namelist(), names)
                self.assertTrue(all("/" not in name and "\\" not in name for name in archive.namelist()))


if __name__ == "__main__":
    unittest.main()
