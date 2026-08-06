"""Focused CPU-only contracts for eval_frozen_end_to_end_gate."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image


SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import eval_frozen_end_to_end_gate as gate
import build_source_groups as source_builder


class FrozenGateTest(unittest.TestCase):
    def _make_fixture(self, root: Path) -> tuple[Path, Path, dict[str, Path]]:
        targets = root / "targets"
        targets.mkdir()
        y, x = np.indices((gate.IMAGE_SIZE, gate.IMAGE_SIZE))
        names: list[str] = []
        for index in range(5):
            name = f"img_{index:06d}.png"
            names.append(name)
            image = np.stack(
                (
                    (x + 17 * index) % 256,
                    (y * (index + 1) + 29) % 256,
                    ((x // 3 + y // 5) + 41 * index) % 256,
                ),
                axis=-1,
            ).astype(np.uint8)
            Image.fromarray(image, "RGB").save(targets / name)
        source_manifest = root / "source_groups.json"
        source_manifest.write_text(
            json.dumps(
                {
                    "schema": gate.SOURCE_GROUP_SCHEMA,
                    "complete": True,
                    "method": "unit-test explicit source identities",
                    "images": {
                        name: {"source_group": f"source-{index}"}
                        for index, name in enumerate(names)
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        checkpoints: dict[str, Path] = {}
        for role in ("ranker", "affinity_primary", "affinity_secondary", "spatial"):
            path = root / f"{role}.pt"
            path.write_bytes(f"fake checkpoint: {role}\n".encode("ascii"))
            checkpoints[role] = path
        return targets, source_manifest, checkpoints

    def _write_builder_v2_manifest(self, targets: Path, path: Path) -> dict[str, object]:
        payload = source_builder.build_manifest(
            Namespace(
                targets=str(targets),
                train_count=2,
                val_count=3,
                tune_val_max=0,
                candidate_val_min=1,
                exclude_val_indices="1",
                select_count=1,
                seed="123456",
                phash_threshold=-1,
                dhash_threshold=6,
                mean_rgb_threshold=36.0,
            )
        )
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return payload

    def _freeze_small(self, root: Path, gate_name: str = "gate") -> tuple[Path, dict[str, object]]:
        targets, source_manifest, checkpoints = self._make_fixture(root)
        output = root / gate_name
        result = gate.freeze_gate(
            targets_dir=targets,
            source_groups_path=source_manifest,
            gate_dir=output,
            checkpoints=checkpoints,
            number=1,
            gate_seed=123_456,
            validation_count=3,
            tuning_ranges="0:1",
            minimum_scenes=1,
        )
        return output, result

    def test_source_manifest_is_required_and_must_be_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(gate.SourceGroupManifestError, "complete source-group manifest"):
                gate.load_source_groups(root / "missing.json", ["a.png"])
            incomplete = root / "incomplete.json"
            incomplete.write_text(
                json.dumps(
                    {
                        "schema": gate.SOURCE_GROUP_SCHEMA,
                        "complete": True,
                        "method": "test",
                        "images": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(gate.SourceGroupManifestError, "not complete"):
                gate.load_source_groups(incomplete, ["a.png"])

    def test_checked_in_builder_manifest_schema_is_accepted_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "builder.json"
            payload = {
                "schema_version": 1,
                "algorithms": {"exact": "sha256", "phash_threshold": 4},
                "files": {
                    "a.png": {"source_group": "g_a", "sha256": "a" * 64},
                    "b.png": {"source_group": "g_b", "sha256": "b" * 64},
                },
                "groups": {"g_a": ["a.png"], "g_b": ["b.png"]},
                "stats": {"files": 2},
                "split": {"train_count": 1, "val_count": 1},
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            groups, normalized, _ = gate.load_source_groups(path, ["a.png", "b.png"])
            self.assertEqual(groups, {"a.png": "g_a", "b.png": "g_b"})
            self.assertEqual(normalized["_normalized_schema"], "build-source-groups-v1")
            payload["groups"] = {"g_a": ["a.png", "b.png"]}
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(gate.SourceGroupManifestError, "consistent partition"):
                gate.load_source_groups(path, ["a.png", "b.png"])

    def test_builder_v2_manifest_requires_exact_exclusion_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            targets, _, _ = self._make_fixture(root)
            names = sorted(path.name for path in targets.iterdir())
            path = root / "builder-v2.json"
            original = self._write_builder_v2_manifest(targets, path)
            _, normalized, _ = gate.load_source_groups(path, names)
            self.assertEqual(normalized["_normalized_schema"], "build-source-groups-v2")

            mutations = []
            missing_exclusions = json.loads(json.dumps(original))
            del missing_exclusions["split"]["excluded_val_ids"]
            mutations.append((missing_exclusions, "excluded_val_ids"))
            empty_exclusions = json.loads(json.dumps(original))
            empty_exclusions["split"]["excluded_val_ids"] = []
            mutations.append((empty_exclusions, "non-empty"))
            duplicate_exclusions = json.loads(json.dumps(original))
            duplicate_exclusions["split"]["excluded_val_ids"] = [1, 1]
            mutations.append((duplicate_exclusions, "sorted and unique"))
            out_of_bounds = json.loads(json.dumps(original))
            out_of_bounds["split"]["excluded_val_ids"] = [3]
            mutations.append((out_of_bounds, "in bounds"))
            changed_contract = json.loads(json.dumps(original))
            changed_contract["split"]["selection_contract"]["ranking"] = "different"
            mutations.append((changed_contract, "exact selection_contract"))
            selected_exclusion = json.loads(json.dumps(original))
            selected_exclusion["split"]["selected_confirmation"] = ["img_000003.png"]
            mutations.append((selected_exclusion, "exclusions occur"))
            for payload, message in mutations:
                with self.subTest(message=message):
                    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
                    with self.assertRaisesRegex(gate.SourceGroupManifestError, message):
                        gate.load_source_groups(path, names)

    def test_builder_v2_freeze_requires_matching_tuning_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            targets, _, checkpoints = self._make_fixture(root)
            source_manifest = root / "builder-v2.json"
            self._write_builder_v2_manifest(targets, source_manifest)

            result = gate.freeze_gate(
                targets_dir=targets,
                source_groups_path=source_manifest,
                gate_dir=root / "gate-matching",
                checkpoints=checkpoints,
                number=1,
                gate_seed=123_456,
                validation_count=3,
                tuning_ranges="0:2",
                minimum_scenes=1,
            )
            self.assertEqual(result["scenes"], 1)

            for gate_name, tuning_ranges in (
                ("gate-missing", "0:1"),
                ("gate-changed", "0:1,2:1"),
            ):
                with self.subTest(tuning_ranges=tuning_ranges), self.assertRaisesRegex(
                    gate.SourceGroupManifestError, "tuning ranges do not exactly match"
                ):
                    gate.freeze_gate(
                        targets_dir=targets,
                        source_groups_path=source_manifest,
                        gate_dir=root / gate_name,
                        checkpoints=checkpoints,
                        number=1,
                        gate_seed=123_456,
                        validation_count=3,
                        tuning_ranges=tuning_ranges,
                        minimum_scenes=1,
                    )

            with self.assertRaisesRegex(gate.SourceGroupManifestError, "gate_seed differs"):
                gate.freeze_gate(
                    targets_dir=targets,
                    source_groups_path=source_manifest,
                    gate_dir=root / "gate-wrong-seed",
                    checkpoints=checkpoints,
                    number=1,
                    gate_seed=123_457,
                    validation_count=3,
                    tuning_ranges="0:2",
                    minimum_scenes=1,
                )
            with self.assertRaisesRegex(gate.SourceGroupManifestError, "scene count differs"):
                gate.freeze_gate(
                    targets_dir=targets,
                    source_groups_path=source_manifest,
                    gate_dir=root / "gate-wrong-count",
                    checkpoints=checkpoints,
                    number=2,
                    gate_seed=123_456,
                    validation_count=3,
                    tuning_ranges="0:2",
                    minimum_scenes=1,
                )

    def test_source_disjoint_selection_excludes_training_group(self) -> None:
        names = [f"n{index}" for index in range(6)]
        groups = {
            "n0": "train-a",
            "n1": "train-b",
            "n2": "train-c",
            "n3": "train-a",  # validation near-duplicate of training; must be excluded
            "n4": "fresh-d",
            "n5": "fresh-e",
        }
        selected = gate.select_gate_names(
            names,
            groups,
            validation_count=3,
            tuning_ranges="",
            number=2,
            gate_seed=9,
        )
        self.assertEqual({row["source_group"] for row in selected["selected"]}, {"fresh-d", "fresh-e"})
        with self.assertRaisesRegex(gate.SourceGroupManifestError, "training and tuning"):
            gate.select_gate_names(
                names,
                groups,
                validation_count=3,
                tuning_ranges="0:1",
                number=1,
                gate_seed=9,
            )

    def test_freeze_persists_exact_bytes_and_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            targets, source_manifest, checkpoints = self._make_fixture(root)
            outputs: list[Path] = []
            results: list[dict[str, object]] = []
            for gate_name in ("gate-a", "gate-b"):
                output = root / gate_name
                results.append(
                    gate.freeze_gate(
                        targets_dir=targets,
                        source_groups_path=source_manifest,
                        gate_dir=output,
                        checkpoints=checkpoints,
                        number=1,
                        gate_seed=123_456,
                        validation_count=3,
                        tuning_ranges="0:1",
                        minimum_scenes=1,
                    )
                )
                outputs.append(output)
            self.assertEqual(results[0]["root_sha256"], results[1]["root_sha256"])
            manifest_a, arrays_a, root_a = gate.load_and_verify_gate(outputs[0], minimum_scenes=1)
            manifest_b, arrays_b, root_b = gate.load_and_verify_gate(outputs[1], minimum_scenes=1)
            self.assertEqual(root_a, root_b)
            self.assertEqual(manifest_a, manifest_b)
            name = manifest_a["scenes"][0]["name"]
            self.assertEqual(arrays_a[name]["tiles"].dtype, np.uint8)
            self.assertEqual(arrays_a[name]["tiles"].shape, (576, 20, 20, 3))
            self.assertTrue(np.array_equal(arrays_a[name]["tiles"], arrays_b[name]["tiles"]))
            self.assertTrue(np.array_equal(np.sort(arrays_a[name]["permutation"]), np.arange(576)))
            self.assertFalse(np.any(arrays_a[name]["orientations_quarter_turns"]))
            scene_relative = manifest_a["scenes"][0]["file"]
            self.assertEqual((outputs[0] / scene_relative).read_bytes(), (outputs[1] / scene_relative).read_bytes())
            with self.assertRaises(FileExistsError):
                gate.freeze_gate(
                    targets_dir=targets,
                    source_groups_path=source_manifest,
                    gate_dir=outputs[0],
                    checkpoints=checkpoints,
                    number=1,
                    gate_seed=123_456,
                    validation_count=3,
                    tuning_ranges="0:1",
                    minimum_scenes=1,
                )

    def test_verify_detects_one_byte_scene_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, _ = self._freeze_small(Path(temporary))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            scene = output / manifest["scenes"][0]["file"]
            content = bytearray(scene.read_bytes())
            content[len(content) // 2] ^= 1
            scene.write_bytes(content)
            with self.assertRaisesRegex(gate.IntegrityError, "artifact hash mismatch"):
                gate.load_and_verify_gate(output, minimum_scenes=1)

    def test_score_cache_contract_rejects_wrong_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "scene.scores.npz"
            contract = {
                "schema": gate.SCORE_CACHE_SCHEMA,
                "gate_root_sha256": "a" * 64,
                "scene": "unit",
            }
            candidates = (
                np.arange(gate.NFRAG)[:, None] + 1 + np.arange(128, dtype=np.int64)[None, :]
            ) % gate.NFRAG
            candidates = candidates.astype(np.int16)
            valid = np.ones_like(candidates, dtype=np.bool_)
            scores = np.zeros((gate.NFRAG, gate.NUM_DIRECTIONS, 128), dtype=np.float32)
            gate.write_score_cache(
                cache,
                contract=contract,
                candidate_ids=candidates,
                candidate_valid=valid,
                raw_scores=scores,
                spatial_scores=scores,
            )
            loaded = gate.load_score_cache(cache, contract)
            self.assertTrue(np.array_equal(loaded["candidate_ids"], candidates))
            wrong = dict(contract)
            wrong["gate_root_sha256"] = "b" * 64
            with self.assertRaisesRegex(gate.CacheContractError, "different scene/gate"):
                gate.load_score_cache(cache, wrong)
            content = bytearray(cache.read_bytes())
            content[-10] ^= 1
            cache.write_bytes(content)
            with self.assertRaisesRegex(gate.CacheContractError, "digest mismatch"):
                gate.load_score_cache(cache, contract)

    def test_oracle_metrics_and_edge_r1_are_exact(self) -> None:
        y, x = np.indices((gate.IMAGE_SIZE, gate.IMAGE_SIZE))
        target = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
        tiles = gate._to_fragments(target)
        permutation = np.arange(gate.NFRAG, dtype=np.int16)
        board = np.arange(gate.NFRAG, dtype=np.int64)
        metrics = gate._board_metrics(
            tiles=tiles,
            target=target,
            permutation=permutation,
            board=board,
            restorer=lambda image: image.copy(),
        )
        for metric in ("placement", "neighbour", "solve_ssim", "final_ssim"):
            self.assertAlmostEqual(metrics[metric], 1.0, places=7)

        targets, exists = gate._neighbor_targets_numpy(permutation)
        candidates = np.empty((gate.NFRAG, 4), dtype=np.int64)
        scores = np.full((gate.NFRAG, gate.NUM_DIRECTIONS, 4), -100.0, dtype=np.float32)
        for anchor in range(gate.NFRAG):
            true_values = list(dict.fromkeys(int(value) for value in targets[anchor, exists[anchor]]))
            filler = 0
            while len(true_values) < 4:
                if filler != anchor and filler not in true_values:
                    true_values.append(filler)
                filler += 1
            candidates[anchor] = true_values
            for direction in range(gate.NUM_DIRECTIONS):
                if exists[anchor, direction]:
                    slot = true_values.index(int(targets[anchor, direction]))
                    scores[anchor, direction, slot] = 100.0
                else:
                    scores[anchor, direction, 0] = 100.0
        valid = np.ones_like(candidates, dtype=np.bool_)
        self.assertAlmostEqual(gate.edge_r1(candidates, valid, scores, permutation), 1.0, places=7)
        bad_board = board.copy()
        bad_board[-1] = bad_board[-2]
        with self.assertRaisesRegex(gate.IntegrityError, "not a bijection"):
            gate._board_metrics(
                tiles=tiles,
                target=target,
                permutation=permutation,
                board=bad_board,
                restorer=lambda image: image,
            )

    def test_fixed_arm_contract_is_not_a_sweep(self) -> None:
        self.assertEqual(gate.FIXED_ARMS["i11"]["max_edges"], 512)
        self.assertEqual(gate.FIXED_ARMS["i11"]["repair_passes"], 0)
        self.assertEqual(gate.FIXED_ARMS["i21"]["alpha"], 1.25)
        self.assertEqual(gate.FIXED_ARMS["i21"]["max_edges"], 512)
        self.assertIsNone(gate.FIXED_ARMS["raw_input"]["edge_r1"])
        self.assertEqual(gate.DEFAULT_TUNING_RANGES, "0:100")


if __name__ == "__main__":
    unittest.main()
