from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import eval_frozen_end_to_end_gate as frozen
import freeze_rank96_lab_selector_v3 as freezer
import rank96_lab_selector_v3_core as core


class ExactFreezeContractTests(unittest.TestCase):
    def test_exact_scene_seed_validation_and_exclusion_constants(self) -> None:
        self.assertEqual(freezer.GATE_V3_SCENES, 48)
        self.assertEqual(freezer.GATE_V3_SEED, 20_260_808)
        self.assertEqual(freezer.VALIDATION_COUNT, 300)
        self.assertEqual(freezer.CANDIDATE_VALIDATION_MIN, 100)
        self.assertEqual(
            freezer.GATE_V1_VALIDATION_IDS,
            (
                119, 122, 136, 138, 142, 157, 158, 164, 170, 172, 203, 208,
                215, 218, 219, 228, 229, 248, 252, 253, 256, 263, 279, 295,
            ),
        )
        self.assertEqual(
            freezer.GATE_V2_VALIDATION_IDS,
            (
                100, 102, 105, 121, 174, 206, 207, 211, 212, 221, 223, 224,
                227, 238, 249, 251, 268, 272, 275, 282, 283, 287, 290, 299,
            ),
        )
        self.assertEqual(len(freezer.PRIOR_GATE_VALIDATION_IDS), 48)
        self.assertEqual(len(set(freezer.PRIOR_GATE_VALIDATION_IDS)), 48)
        self.assertEqual(
            frozen._parse_ranges(
                freezer.PREDECLARED_TUNING_RANGES,
                freezer.VALIDATION_COUNT,
            ),
            list(freezer.PREDECLARED_TUNING_IDS),
        )
        self.assertEqual(len(freezer.PREDECLARED_TUNING_IDS), 148)
        self.assertEqual(freezer.PREDECLARED_TUNING_IDS[:100], tuple(range(100)))

    def test_default_v3_artifact_paths_are_canonical_on_e_drive(self) -> None:
        paths = freezer._default_paths()
        root = Path("E:/pazzle_work/rank96_e11_v4")
        self.assertEqual(paths["artifact_root"], root)
        self.assertEqual(paths["source_groups"], root / "source_groups_v4.json")
        self.assertEqual(paths["gate"], root / "gate_v4")
        self.assertEqual(paths["score_cache"], root / "score_cache_v4")
        self.assertEqual(paths["report"], root / "report_rank96_lab_selector_v4.json")
        core_paths = core._default_paths()
        self.assertEqual(paths["source_groups"], core_paths["source_manifest"])
        self.assertEqual(paths["gate"], core_paths["gate"])
        self.assertEqual(paths["score_cache"], core_paths["score_cache"])
        self.assertEqual(paths["report"], core_paths["report"])

    def test_cli_exposes_paths_but_no_experiment_controls(self) -> None:
        parser = freezer._build_parser()
        subparser_action = next(
            action for action in parser._actions if isinstance(action, __import__("argparse")._SubParsersAction)
        )
        freeze_parser = subparser_action.choices["freeze"]
        options = {
            option
            for action in freeze_parser._actions
            for option in action.option_strings
        }
        self.assertTrue(
            {
                "--targets-dir",
                "--source-groups",
                "--gate-dir",
                "--ranker",
                "--affinity-primary",
                "--affinity-secondary",
                "--spatial",
            }
            <= options
        )
        for action in freeze_parser._actions:
            if action.option_strings and "--help" not in action.option_strings:
                self.assertIs(action.type, Path)
        for forbidden in (
            "--n",
            "--number",
            "--seed",
            "--gate-seed",
            "--validation-count",
            "--tuning-ranges",
            "--minimum-scenes",
            "--device",
            "--rotation",
        ):
            self.assertNotIn(forbidden, options)


class CodeProvenanceRegistryTests(unittest.TestCase):
    def test_registry_contains_exact_required_roles_and_excludes_launcher(self) -> None:
        workspace = Path(freezer.__file__).resolve().parent.parent
        paths = freezer._require_hashed_code_registry()
        self.assertEqual(tuple(paths), freezer.REQUIRED_CODE_ROLES)
        launcher = workspace / "src" / "eval_rank96_lab_selector_v3.py"
        self.assertNotIn(launcher.resolve(), {path.resolve() for path in paths.values()})
        self.assertIn("build_source_groups_legacy", paths)
        self.assertIn("build_source_groups_v3", paths)
        self.assertIn("test_build_source_groups_v3", paths)
        self.assertNotIn("eval_rank96_lab_selector_v3", paths)

    def test_generic_harness_bytes_and_registry_remain_legacy_v2(self) -> None:
        workspace = Path(freezer.__file__).resolve().parent.parent
        generic = frozen._default_code_paths(workspace)
        self.assertEqual(tuple(generic), freezer.GENERIC_CODE_ROLES)
        self.assertEqual(
            frozen.sha256_file(generic["frozen_gate_harness"]),
            "8a0a5c05485813121db0dba375491157d8085a41c5f0afe271a9731840610ae8",
        )
        self.assertNotIn("rank96_lab_selector_v3_core", generic)

    def test_registry_fails_closed_when_core_role_is_missing(self) -> None:
        original = freezer.e11_code_registry

        def missing_core(workspace: Path) -> dict[str, Path]:
            paths = dict(original(workspace))
            paths.pop("rank96_lab_selector_v3_core")
            return paths

        with mock.patch.object(freezer, "e11_code_registry", side_effect=missing_core):
            with self.assertRaisesRegex(frozen.FrozenGateError, "exact contract"):
                freezer._require_hashed_code_registry()

    def test_temporary_registry_and_validator_restore_exact_identities(self) -> None:
        workspace = Path(freezer.__file__).resolve().parent.parent
        original_registry = frozen._default_code_paths
        original_validator = frozen._validate_builder_v2_selection
        with freezer._temporary_e11_freeze_contract():
            self.assertIs(frozen._default_code_paths, freezer.e11_code_registry)
            self.assertIs(
                frozen._validate_builder_v2_selection,
                freezer._validate_e11_v3_selection,
            )
            self.assertEqual(tuple(frozen._default_code_paths(workspace)), freezer.REQUIRED_CODE_ROLES)
        self.assertIs(frozen._default_code_paths, original_registry)
        self.assertIs(frozen._validate_builder_v2_selection, original_validator)

        with self.assertRaisesRegex(RuntimeError, "injected"):
            with freezer._temporary_e11_freeze_contract():
                raise RuntimeError("injected")
        self.assertIs(frozen._default_code_paths, original_registry)
        self.assertIs(frozen._validate_builder_v2_selection, original_validator)


class WrappedFreezeTests(unittest.TestCase):
    def _call(self, root: Path) -> dict[str, object]:
        return freezer.freeze_rank96_lab_selector_v3(
            targets_dir=root / "targets",
            source_groups_path=root / "source.json",
            gate_dir=root / "gate_v3",
            ranker_checkpoint=root / "ranker.pt",
            affinity_primary_checkpoint=root / "affinity_primary.pt",
            affinity_secondary_checkpoint=root / "affinity_secondary.pt",
            spatial_checkpoint=root / "spatial.pt",
        )

    def test_wrapper_passes_only_exact_fixed_controls_and_creates_no_bytes_when_mocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_registry = frozen._default_code_paths
            original_validator = frozen._validate_builder_v2_selection

            def fake_freeze(**kwargs: object) -> dict[str, object]:
                roles = frozen._default_code_paths(Path(freezer.__file__).resolve().parent.parent)
                self.assertEqual(tuple(roles), freezer.REQUIRED_CODE_ROLES)
                self.assertIs(
                    frozen._validate_builder_v2_selection,
                    freezer._validate_e11_v3_selection,
                )
                return {"root_sha256": "f" * 64, "scenes": 48}

            with mock.patch.object(freezer.frozen, "freeze_gate", side_effect=fake_freeze) as call:
                result = self._call(root)

            self.assertEqual(result["scenes"], 48)
            self.assertFalse((root / "gate_v3").exists())
            self.assertIs(frozen._default_code_paths, original_registry)
            self.assertIs(frozen._validate_builder_v2_selection, original_validator)
            call.assert_called_once_with(
                targets_dir=root / "targets",
                source_groups_path=root / "source.json",
                gate_dir=root / "gate_v3",
                checkpoints={
                    "ranker": root / "ranker.pt",
                    "affinity_primary": root / "affinity_primary.pt",
                    "affinity_secondary": root / "affinity_secondary.pt",
                    "spatial": root / "spatial.pt",
                },
                number=48,
                gate_seed=20_260_808,
                validation_count=300,
                tuning_ranges=freezer.PREDECLARED_TUNING_RANGES,
                minimum_scenes=48,
            )

    def test_freeze_failure_creates_no_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_registry = frozen._default_code_paths
            original_validator = frozen._validate_builder_v2_selection
            with mock.patch.object(
                freezer.frozen,
                "freeze_gate",
                side_effect=RuntimeError("freeze failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "freeze failed"):
                    self._call(root)
            self.assertFalse((root / "gate_v3").exists())
            self.assertIs(frozen._default_code_paths, original_registry)
            self.assertIs(frozen._validate_builder_v2_selection, original_validator)


if __name__ == "__main__":
    unittest.main()
