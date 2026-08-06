from __future__ import annotations

import copy
import hashlib
import io
import json
import shutil
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

import build_source_groups_v3 as source_v3
import eval_frozen_end_to_end_gate as frozen
import eval_rank96_lab_selector_v3 as launcher
import freeze_rank96_lab_selector_v3 as freezer
import rank96_lab_selector_v3_core as gate_v3


ROOT = Path(__file__).resolve().parents[1]


class FrozenE11ContractTests(unittest.TestCase):
    def test_root_independent_core_and_launcher_boundary(self) -> None:
        self.assertFalse(hasattr(gate_v3, "EXPECTED_GATE_ROOT_SHA256"))
        self.assertIsNone(launcher.EXPECTED_GATE_ROOT_SHA256)
        paths = freezer.e11_code_registry(ROOT)
        self.assertEqual(gate_v3.REQUIRED_E11_CODE_ROLES, set(paths))
        launcher_path = Path(launcher.__file__).resolve()
        self.assertNotIn("eval_rank96_lab_selector_v3", paths)
        self.assertNotIn(launcher_path, {path.resolve() for path in paths.values()})
        self.assertEqual(
            paths["rank96_lab_selector_v3_core"].resolve(),
            Path(gate_v3.__file__).resolve(),
        )

    def test_launcher_passes_only_the_root_and_argv(self) -> None:
        digest = "a" * 64
        with (
            mock.patch.object(launcher, "EXPECTED_GATE_ROOT_SHA256", digest),
            mock.patch.object(launcher, "run_cli", return_value=23) as run,
        ):
            self.assertEqual(launcher.main(["preflight"]), 23)
        run.assert_called_once_with(expected_root=digest, argv=["preflight"])

    def test_unpinned_root_stops_before_any_gate_path_access(self) -> None:
        paths = gate_v3._default_paths()
        with mock.patch.object(gate_v3, "_require_exact_paths") as require_paths:
            with self.assertRaises(gate_v3.GateV3NotPinnedError):
                gate_v3._load_verified_inputs(
                    paths["gate"], paths["score_cache"], expected_root=None
                )
        require_paths.assert_not_called()

    def test_future_pinned_root_is_exact(self) -> None:
        digest = "a" * 64
        self.assertEqual(gate_v3._require_pinned_root_configuration(digest), digest)
        gate_v3._require_expected_root(digest, expected_root=digest)
        with self.assertRaises(frozen.IntegrityError):
            gate_v3._require_expected_root("b" * 64, expected_root=digest)

    def test_v3_large_artifact_paths_are_canonical_on_e_drive(self) -> None:
        paths = gate_v3._default_paths()
        self.assertEqual(str(paths["artifact_root"]).replace("\\", "/"), "E:/pazzle_work/rank96_e11_v4")
        self.assertEqual(paths["source_manifest"].parent, paths["artifact_root"])
        self.assertEqual(paths["gate"].parent, paths["artifact_root"])
        self.assertEqual(paths["score_cache"].parent, paths["artifact_root"])
        self.assertEqual(paths["report"].parent, paths["artifact_root"])
        self.assertEqual(paths["evaluation_started"].parent, paths["artifact_root"])
        self.assertEqual(paths["gate"].name, "gate_v4")
        self.assertEqual(paths["score_cache"].name, "score_cache_v4")

    def test_only_two_reported_arms_and_fixed_internal_candidates(self) -> None:
        self.assertEqual(gate_v3.ARM_ORDER, ("rank96", "selector"))
        self.assertEqual(set(gate_v3.FIXED_ARMS), set(gate_v3.ARM_ORDER))
        baseline = gate_v3.FIXED_ARMS["rank96"]
        selected = gate_v3.FIXED_ARMS["selector"]
        self.assertEqual(baseline["max_edges"], 96)
        self.assertEqual(
            [candidate["max_edges"] for candidate in selected["candidate_solvers"]],
            [96, 512],
        )
        self.assertEqual(selected["selection"]["depth"], 1)
        self.assertEqual(selected["selection"]["tie"], "rank96")
        self.assertTrue(selected["selection"]["label_free"])
        for arm in gate_v3.FIXED_ARMS.values():
            self.assertEqual(arm["orientation"], "fixed_type1_no_rotation")
            self.assertEqual(arm["restoration"]["h"], 10)
            self.assertEqual(arm["restoration"]["h_color"], 10)

    def test_keep_rule_boundaries_are_exact(self) -> None:
        threshold = gate_v3.KEEP_FINAL_DELTA_STRICTLY_GREATER_THAN
        just_above = np.nextafter(threshold, np.inf)
        self.assertFalse(
            gate_v3.keep_rule(mean_final_delta=threshold, mean_solve_delta=0.0)
        )
        self.assertTrue(
            gate_v3.keep_rule(mean_final_delta=just_above, mean_solve_delta=0.0)
        )
        self.assertFalse(
            gate_v3.keep_rule(mean_final_delta=just_above, mean_solve_delta=-1e-12)
        )
        with self.assertRaises(ValueError):
            gate_v3.keep_rule(mean_final_delta=np.nan, mean_solve_delta=0.0)

    def test_cli_exposes_no_report_or_experiment_controls(self) -> None:
        parser = gate_v3._build_parser()
        subparser_action = next(
            action
            for action in parser._actions
            if isinstance(action, __import__("argparse")._SubParsersAction)
        )
        self.assertEqual(
            set(subparser_action.choices),
            {"prepare-cache", "preflight", "evaluate"},
        )
        options = {
            option
            for child in subparser_action.choices.values()
            for action in child._actions
            for option in action.option_strings
        }
        for forbidden in (
            "--report",
            "--gate-dir",
            "--score-cache-dir",
            "--device",
            "--depth",
            "--threshold",
            "--max-edges",
            "--min-margin",
            "--repair-passes",
            "--rotation",
            "--sweep",
        ):
            self.assertNotIn(forbidden, options)
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args([])
            with self.assertRaises(SystemExit):
                parser.parse_args(["evaluate", "--report", "somewhere.json"])

    def test_alternate_report_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            alternate = Path(directory) / "report.json"
            with self.assertRaisesRegex(frozen.IntegrityError, "must be exactly"):
                gate_v3._canonical_report_path(alternate)
        self.assertEqual(
            gate_v3._canonical_report_path(), gate_v3._default_paths()["report"].resolve()
        )


class PriorReportIdentityTests(unittest.TestCase):
    def test_prior_identities_use_only_exact_reports_never_legacy_gate_dirs(self) -> None:
        paths = gate_v3._default_paths()
        self.assertNotIn("gate_v1", paths)
        self.assertNotIn("gate_v2", paths)
        self.assertEqual(
            gate_v3.PRIOR_REPORT_CONTRACTS["gate_v1"]["sha256"],
            "2ea813849d45562d2e5af77ac73fdb1258a2b900dbc6290b645abf12b3810db6",
        )
        self.assertEqual(
            gate_v3.PRIOR_REPORT_CONTRACTS["gate_v2"]["sha256"],
            "6c0d8ecf07b505d85bf7a831d5b31a22e3ccbdf3637ca42fd41ee359a8fc92dc",
        )
        with mock.patch.object(
            gate_v3.frozen,
            "load_and_verify_gate",
            side_effect=AssertionError("legacy gate directory access is forbidden"),
        ) as legacy_gate_load:
            names, groups = gate_v3._load_prior_identities()
        legacy_gate_load.assert_not_called()
        self.assertEqual(len(names), 48)
        self.assertEqual(len(groups), 48)

    def _copied_report_paths(self, root: Path) -> dict[str, Path]:
        defaults = gate_v3._default_paths()
        result = dict(defaults)
        for version in ("v1", "v2"):
            source = defaults[f"prior_report_{version}"]
            target = root / source.name
            shutil.copyfile(source, target)
            shutil.copyfile(
                source.with_suffix(source.suffix + ".sha256"),
                target.with_suffix(target.suffix + ".sha256"),
            )
            result[f"prior_report_{version}"] = target
        return result

    def test_prior_report_or_sidecar_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._copied_report_paths(root)
            with mock.patch.object(gate_v3, "_default_paths", return_value=paths):
                gate_v3._load_prior_identities()
                with paths["prior_report_v1"].open("ab") as stream:
                    stream.write(b"\n")
                with self.assertRaisesRegex(frozen.IntegrityError, "digest changed"):
                    gate_v3._load_prior_identities()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._copied_report_paths(root)
            sidecar = paths["prior_report_v2"].with_suffix(".json.sha256")
            sidecar.write_bytes(b"tampered\n")
            with mock.patch.object(gate_v3, "_default_paths", return_value=paths):
                with self.assertRaisesRegex(frozen.IntegrityError, "sidecar differs"):
                    gate_v3._load_prior_identities()


class SourceIsolationTests(unittest.TestCase):
    @staticmethod
    def _group_id(members: list[str]) -> str:
        digest = hashlib.sha256("\0".join(sorted(members)).encode("utf-8")).hexdigest()
        return f"g_{digest[:16]}"

    def _fixture(
        self, *, merge_prior_candidate: bool = False
    ) -> tuple[dict[str, object], dict[str, object], set[str], set[str]]:
        names = [f"img_{index:06d}.png" for index in range(7000)]
        validation_names = names[source_v3.TRAIN_COUNT :]
        member_sets = [[name] for name in names]
        if merge_prior_candidate:
            prior_absolute = source_v3.TRAIN_COUNT + source_v3.PRIOR_GATE_VALIDATION_IDS[0]
            candidate_absolute = source_v3.TRAIN_COUNT + 101
            member_sets[prior_absolute].append(names[candidate_absolute])
            member_sets[candidate_absolute] = []
        member_sets = [members for members in member_sets if members]

        groups = {self._group_id(members): sorted(members) for members in member_sets}
        group_for_name = {
            name: group
            for group, members in groups.items()
            for name in members
        }
        selection = source_v3.select_confirmation_v3(names, group_for_name)
        prior_names = set(selection.prior_scene_names)
        # These are the old v1/v2 IDs.  They intentionally differ from every
        # membership-derived v3 ID; filename mapping, not old ID equality, is
        # the stable source-isolation contract.
        prior_groups = {f"old_prior_group_{index:02d}" for index in range(48)}
        scenes = [
            {"name": name, "source_group": group_for_name[name]}
            for name in selection.selected
        ]
        code = {
            role: {"path": str(path.resolve()), "sha256": "a" * 64}
            for role, path in freezer.e11_code_registry(ROOT).items()
        }
        manifest: dict[str, object] = {
            "scene_count": 48,
            "gate_seed": gate_v3.EXPECTED_GATE_SEED,
            "geometry": {"orientation": gate_v3.FIXED_ORIENTATION},
            "selection": {"validation_count": 300},
            "source_groups": {
                "target_corpus_sha256": gate_v3.EXPECTED_TARGET_CORPUS_SHA256
            },
            "code": code,
            "scenes": scenes,
            "splits": {
                "training": {
                    "source_groups": sorted(
                        {group_for_name[name] for name in names[: source_v3.TRAIN_COUNT]}
                    )
                }
            },
        }
        archived: dict[str, object] = {
            "schema_version": 2,
            "algorithms": copy.deepcopy(source_v3.ALGORITHMS_CONTRACT),
            "builder_contract": copy.deepcopy(source_v3.BUILDER_CONTRACT),
            "stats": {"files": len(names)},
            "files": {
                name: {"sha256": "0" * 64, "source_group": group_for_name[name]}
                for name in names
            },
            "groups": groups,
            "split": {
                "selection_seed": str(gate_v3.EXPECTED_GATE_SEED),
                "known_tune_val_ids": [0, 99],
                "candidate_val_min": 100,
                "eligible_confirmation": list(selection.eligible),
                "selected_confirmation": list(selection.selected),
                "excluded_val_ids": list(source_v3.PRIOR_GATE_VALIDATION_IDS),
                "selection_contract": copy.deepcopy(source_v3.BASE_SELECTION_CONTRACT),
                "v3_selection_contract": copy.deepcopy(source_v3.V3_SELECTION_CONTRACT),
                "prior_scene_names": list(selection.prior_scene_names),
                "prior_source_groups_v3": list(selection.prior_source_groups_v3),
                "train_count": source_v3.TRAIN_COUNT,
                "val_count": 300,
            },
        }
        return manifest, archived, prior_names, prior_groups

    def _validate(
        self,
        manifest: dict[str, object],
        archived: dict[str, object],
        prior_names: set[str],
        prior_groups: set[str],
    ) -> None:
        with (
            mock.patch.object(
                gate_v3, "_load_prior_identities", return_value=(prior_names, prior_groups)
            ),
            mock.patch.object(gate_v3, "_source_manifest", return_value=archived),
        ):
            gate_v3._validate_v3_isolation(manifest, Path("unused"))

    def test_exact_new_source_disjoint_fixture_passes(self) -> None:
        self._validate(*self._fixture())

    def test_old_group_ids_may_change_after_corrected_v3_grouping(self) -> None:
        manifest, archived, prior_names, old_prior_groups = self._fixture()
        mapped = {
            archived["files"][name]["source_group"]  # type: ignore[index]
            for name in prior_names
        }
        self.assertNotEqual(mapped, old_prior_groups)
        self._validate(manifest, archived, prior_names, old_prior_groups)

    def test_corrected_v3_group_may_merge_prior_identity_and_excludes_candidate(self) -> None:
        manifest, archived, prior_names, old_prior_groups = self._fixture(
            merge_prior_candidate=True
        )
        validation_names = sorted(archived["files"])[source_v3.TRAIN_COUNT :]  # type: ignore[arg-type]
        candidate = validation_names[101]
        self.assertNotIn(candidate, archived["split"]["eligible_confirmation"])  # type: ignore[index]
        self._validate(manifest, archived, prior_names, old_prior_groups)

    def test_candidate_floor_must_be_exact(self) -> None:
        manifest, archived, prior_names, prior_groups = self._fixture()
        archived["split"]["candidate_val_min"] = 101  # type: ignore[index]
        with self.assertRaisesRegex(frozen.IntegrityError, "candidate_val_min"):
            self._validate(manifest, archived, prior_names, prior_groups)

    def test_exclusions_must_be_exact_without_extra_ids(self) -> None:
        manifest, archived, prior_names, prior_groups = self._fixture()
        archived["split"]["excluded_val_ids"].append(299)  # type: ignore[index,union-attr]
        with self.assertRaisesRegex(frozen.IntegrityError, "excluded_val_ids"):
            self._validate(manifest, archived, prior_names, prior_groups)

    def test_missing_prior_exclusion_fails(self) -> None:
        manifest, archived, prior_names, prior_groups = self._fixture()
        archived["split"]["excluded_val_ids"].pop()  # type: ignore[index,union-attr]
        with self.assertRaisesRegex(frozen.IntegrityError, "excluded_val_ids"):
            self._validate(manifest, archived, prior_names, prior_groups)

    def test_independent_seeded_order_rejects_reordered_selection(self) -> None:
        manifest, archived, prior_names, prior_groups = self._fixture()
        selected = archived["split"]["selected_confirmation"]  # type: ignore[index]
        selected[0], selected[1] = selected[1], selected[0]  # type: ignore[index]
        scenes = manifest["scenes"]  # type: ignore[index]
        scenes[0], scenes[1] = scenes[1], scenes[0]  # type: ignore[index]
        with self.assertRaisesRegex(frozen.IntegrityError, "not independently reproducible"):
            self._validate(manifest, archived, prior_names, prior_groups)

    def test_prior_source_group_is_removed_from_eligible_pool(self) -> None:
        manifest, archived, prior_names, prior_groups = self._fixture(
            merge_prior_candidate=True
        )
        validation_names = sorted(archived["files"])[source_v3.TRAIN_COUNT :]  # type: ignore[arg-type]
        candidate = validation_names[101]
        archived["split"]["eligible_confirmation"].append(candidate)  # type: ignore[index,union-attr]
        with self.assertRaisesRegex(frozen.IntegrityError, "eligible_confirmation"):
            self._validate(manifest, archived, prior_names, prior_groups)

    def test_algorithm_contract_drift_fails_closed(self) -> None:
        manifest, archived, prior_names, prior_groups = self._fixture()
        archived["algorithms"]["phash_threshold"] = 5  # type: ignore[index]
        with self.assertRaisesRegex(frozen.IntegrityError, "algorithms"):
            self._validate(manifest, archived, prior_names, prior_groups)

    def test_missing_hashed_selector_role_fails(self) -> None:
        manifest, archived, prior_names, prior_groups = self._fixture()
        manifest["code"].pop("rank96_lab_selector")  # type: ignore[union-attr]
        with self.assertRaisesRegex(frozen.IntegrityError, "code roles"):
            self._validate(manifest, archived, prior_names, prior_groups)


class CachePreparationAndOneShotTests(unittest.TestCase):
    @staticmethod
    def _runtime() -> dict[str, object]:
        return {
            "device": "cuda:0",
            "device_index": 0,
            "device_name": "mock-fixed-gpu",
            "compute_capability": [9, 0],
            "torch_cuda_version": "99.0",
            "cudnn_version": 99999,
        }

    def test_prepare_cache_scores_tiles_only_and_is_resumable_without_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "artifact_root": root,
                "source_manifest": root / "source.json",
                "gate": root / "gate_v4",
                "score_cache": root / "score_cache_v4",
                "report": root / "report.json",
                "evaluation_started": root / "EVALUATION_STARTED.json",
                "prior_report_v1": root / "prior_v1.json",
                "prior_report_v2": root / "prior_v2.json",
            }
            tile_sentinel = np.asarray([17], dtype=np.uint8)
            manifest = {
                "gate_seed": gate_v3.EXPECTED_GATE_SEED,
                "environment": {"python": "fixed"},
                "checkpoints": {},
                "code": {},
                "scenes": [
                    {
                        "name": "blind_scene.png",
                        "file_sha256": "1" * 64,
                        "arrays_sha256": {"tiles": "2" * 64},
                    }
                ],
            }
            arrays = {
                "blind_scene.png": {
                    "tiles": tile_sentinel,
                    "orientations_quarter_turns": np.zeros(frozen.NFRAG, dtype=np.uint8),
                }
            }
            root_digest = "a" * 64
            model = mock.Mock()
            model.score.return_value = {"opaque": np.asarray([1], dtype=np.int8)}

            def fake_write(path: Path, **_: object) -> str:
                path.write_bytes(b"opaque neural scores")
                path.with_suffix(path.suffix + ".sha256").write_bytes(b"opaque sidecar")
                return "f" * 64

            with (
                mock.patch.object(gate_v3, "_default_paths", return_value=paths),
                mock.patch.object(
                    gate_v3.frozen,
                    "load_and_verify_gate",
                    return_value=(manifest, arrays, root_digest),
                ),
                mock.patch.object(gate_v3, "_validate_v3_isolation"),
                mock.patch.object(gate_v3, "_validate_environment_contract"),
                mock.patch.object(gate_v3.frozen, "_verify_external_files"),
                mock.patch.object(gate_v3, "_fixed_cuda_runtime", return_value=self._runtime()),
                mock.patch.object(gate_v3.frozen, "_ScoringModels", return_value=model) as models,
                mock.patch.object(gate_v3.frozen, "write_score_cache", side_effect=fake_write),
                mock.patch.object(gate_v3.frozen, "load_score_cache", return_value={}),
                mock.patch.object(
                    gate_v3.frozen,
                    "verify_score_cache_directory",
                    return_value={"verified": 1, "missing": []},
                ),
                mock.patch.object(gate_v3, "_dense_matrices") as dense,
                mock.patch.object(gate_v3, "solve_and_select_lab_depth1") as solve,
                mock.patch.object(gate_v3.frozen, "edge_r1") as edge_metric,
                mock.patch.object(gate_v3.frozen, "_board_metrics") as board_metrics,
                mock.patch.object(gate_v3.frozen, "_fixed_nlm") as restore,
                redirect_stdout(io.StringIO()) as output,
            ):
                first = gate_v3.prepare_score_cache_v3(expected_root=root_digest)
                second = gate_v3.prepare_score_cache_v3(expected_root=root_digest)

            self.assertEqual(first["created"], 1)
            self.assertEqual(second["reused"], 1)
            self.assertEqual(first["scoring_device"], "cuda:0")
            self.assertFalse(first["label_derived_metrics_computed"])
            models.assert_called_once_with(manifest, "cuda:0")
            model.score.assert_called_once()
            self.assertIs(model.score.call_args.args[0], tile_sentinel)
            self.assertNotIn("blind_scene", output.getvalue())
            dense.assert_not_called()
            solve.assert_not_called()
            edge_metric.assert_not_called()
            board_metrics.assert_not_called()
            restore.assert_not_called()
            self.assertTrue((paths["score_cache"] / "PREPARE_CACHE_STARTED.json").is_file())
            self.assertTrue((paths["score_cache"] / "PREPARE_CACHE_COMPLETE.json").is_file())

    def test_one_shot_claim_is_create_once_and_report_artifacts_fail_early(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "EVALUATION_STARTED.json"
            report = root / "report.json"
            preparation = {"complete_receipt_sha256": "c" * 64}
            claim, digest = gate_v3._claim_evaluation_once(
                path=marker,
                report_path=report,
                root_digest="a" * 64,
                preparation=preparation,
            )
            self.assertEqual(claim["schema"], gate_v3.EVALUATION_STARTED_SCHEMA)
            self.assertEqual(digest, frozen.sha256_file(marker))
            with self.assertRaisesRegex(frozen.IntegrityError, "gate is spent"):
                gate_v3._claim_evaluation_once(
                    path=marker,
                    report_path=report,
                    root_digest="a" * 64,
                    preparation=preparation,
                )

            marker.unlink()
            report.write_text("already exists", encoding="utf-8")
            with self.assertRaisesRegex(frozen.IntegrityError, "report artifact"):
                gate_v3._claim_evaluation_once(
                    path=marker,
                    report_path=report,
                    root_digest="a" * 64,
                    preparation=preparation,
                )
            self.assertFalse(marker.exists())

    def test_evaluation_crash_after_claim_leaves_gate_spent_before_retry_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "artifact_root": root,
                "source_manifest": root / "source.json",
                "gate": root / "gate_v4",
                "score_cache": root / "score_cache_v4",
                "report": root / "report.json",
                "evaluation_started": root / "EVALUATION_STARTED.json",
                "prior_report_v1": root / "prior_v1.json",
                "prior_report_v2": root / "prior_v2.json",
            }
            name = "scene.png"
            manifest = {
                "scenes": [{"name": name, "source_group": "group"}],
                "checkpoints": {},
                "code": {},
                "environment": {},
            }
            scenes = {name: {}}
            caches = {
                name: {
                    "candidate_ids": np.zeros((frozen.NFRAG, 1), dtype=np.int16),
                    "candidate_valid": np.ones((frozen.NFRAG, 1), dtype=np.bool_),
                    "raw_scores": np.zeros(
                        (frozen.NFRAG, frozen.NUM_DIRECTIONS, 1), dtype=np.float32
                    ),
                }
            }
            loaded = (
                manifest,
                scenes,
                caches,
                "a" * 64,
                {"verified": 1, "missing": []},
                {"complete_receipt_sha256": "c" * 64},
            )
            with (
                mock.patch.object(gate_v3, "_default_paths", return_value=paths),
                mock.patch.object(gate_v3, "_load_verified_inputs", return_value=loaded),
                mock.patch.object(
                    gate_v3,
                    "_dense_matrices",
                    side_effect=RuntimeError("crash after claim"),
                ) as dense,
            ):
                with self.assertRaisesRegex(RuntimeError, "crash after claim"):
                    gate_v3.evaluate_rank96_lab_selector_v3(expected_root="a" * 64)
                self.assertTrue(paths["evaluation_started"].is_file())
                dense.reset_mock()
                with self.assertRaisesRegex(frozen.IntegrityError, "gate is spent"):
                    gate_v3.evaluate_rank96_lab_selector_v3(expected_root="a" * 64)
                dense.assert_not_called()


class PreflightAndReportTests(unittest.TestCase):
    def test_runtime_environment_must_exactly_match_frozen_manifest(self) -> None:
        current = frozen._package_versions()
        self.assertEqual(
            gate_v3._validate_environment_contract({"environment": current}),
            current,
        )
        changed = dict(current)
        changed["python"] = "0.0-audited-drift"
        with self.assertRaisesRegex(frozen.IntegrityError, "runtime environment"):
            gate_v3._validate_environment_contract({"environment": changed})
        with self.assertRaisesRegex(frozen.IntegrityError, "runtime environment"):
            gate_v3._validate_environment_contract({})

    def test_preflight_does_not_solve_or_restore(self) -> None:
        manifest = {
            "scenes": [{"name": f"scene_{index}"} for index in range(48)],
            "checkpoints": {},
            "code": {},
            "environment": frozen._package_versions(),
        }
        scenes = {row["name"]: {} for row in manifest["scenes"]}
        caches = {row["name"]: {} for row in manifest["scenes"]}
        loaded = (
            manifest,
            scenes,
            caches,
            "a" * 64,
            {"verified": 48, "missing": []},
            {
                "scoring_device": "cuda:0",
                "complete_receipt_sha256": "b" * 64,
            },
        )
        with (
            mock.patch.object(gate_v3, "_load_verified_inputs", return_value=loaded),
            mock.patch.object(gate_v3, "solve_and_select_lab_depth1") as solve,
            mock.patch.object(gate_v3.frozen, "_fixed_nlm") as restore,
        ):
            result = gate_v3.preflight_rank96_lab_selector_v3(expected_root="a" * 64)
        self.assertEqual(result["status"], "preflight_ok")
        self.assertEqual(result["scene_count"], 48)
        solve.assert_not_called()
        restore.assert_not_called()

    def test_report_is_create_once_or_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            digest, status = gate_v3._write_immutable_report(path, {"e11": "fixed"})
            self.assertEqual(status, "created")
            digest_again, status_again = gate_v3._write_immutable_report(
                path, {"e11": "fixed"}
            )
            self.assertEqual(digest_again, digest)
            self.assertEqual(status_again, "already_identical")
            with self.assertRaises(frozen.IntegrityError):
                gate_v3._write_immutable_report(path, {"e11": "changed"})

    def test_report_only_crash_state_is_recovered_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            report = {"e11": "fixed"}
            content = gate_v3._canonical_report_bytes(report)
            self.assertTrue(gate_v3._publish_exclusive_exact(path, content))
            self.assertFalse(path.with_suffix(".json.sha256").exists())
            digest, status = gate_v3._write_immutable_report(path, report)
            self.assertEqual(status, "recovered")
            self.assertEqual(digest, frozen._sha256_bytes(content))
            self.assertTrue(path.with_suffix(".json.sha256").is_file())

    def test_sidecar_failure_never_unlinks_published_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            real_publish = gate_v3._publish_exclusive_exact
            calls = 0

            def fail_sidecar(target: Path, content: bytes) -> bool:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected sidecar failure")
                return real_publish(target, content)

            with mock.patch.object(gate_v3, "_publish_exclusive_exact", side_effect=fail_sidecar):
                with self.assertRaisesRegex(OSError, "injected"):
                    gate_v3._write_immutable_report(path, {"e11": "fixed"})
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_suffix(".json.sha256").exists())
            _, status = gate_v3._write_immutable_report(path, {"e11": "fixed"})
            self.assertEqual(status, "recovered")

    def test_concurrent_different_reports_never_mix_or_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            barrier = threading.Barrier(2)
            reports = ({"winner": "a"}, {"winner": "b"})

            def write(report: dict[str, str]) -> tuple[str, str, dict[str, str]]:
                barrier.wait()
                try:
                    _, status = gate_v3._write_immutable_report(path, report)
                    return "ok", status, report
                except frozen.IntegrityError:
                    return "error", "different", report

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(write, reports))
            winners = [result for result in results if result[0] == "ok"]
            losers = [result for result in results if result[0] == "error"]
            self.assertEqual(len(winners), 1)
            self.assertEqual(len(losers), 1)
            expected = gate_v3._canonical_report_bytes(winners[0][2])
            self.assertEqual(path.read_bytes(), expected)
            digest = frozen._sha256_bytes(expected)
            self.assertEqual(
                path.with_suffix(".json.sha256").read_bytes(),
                f"{digest}  {path.name}\n".encode("ascii"),
            )

    def test_concurrent_identical_reports_have_one_creator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            barrier = threading.Barrier(2)

            def write(_: int) -> str:
                barrier.wait()
                return gate_v3._write_immutable_report(path, {"e11": "fixed"})[1]

            with ThreadPoolExecutor(max_workers=2) as executor:
                statuses = list(executor.map(write, range(2)))
            self.assertEqual(statuses.count("created"), 1)
            self.assertEqual(len(statuses), 2)
            self.assertTrue(set(statuses) <= {"created", "recovered", "already_identical"})

    def test_report_json_is_strict_and_key_order_independent(self) -> None:
        self.assertEqual(
            gate_v3._canonical_report_bytes({"b": 2, "a": 1}),
            gate_v3._canonical_report_bytes({"a": 1, "b": 2}),
        )
        with self.assertRaises(frozen.IntegrityError):
            gate_v3._canonical_report_bytes({"bad": float("nan")})


if __name__ == "__main__":
    unittest.main()
