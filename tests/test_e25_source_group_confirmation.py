from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import eval_e25_source_group_confirmation as e25


def sha(char: str) -> str:
    return char * 64


def synthetic_records() -> list[dict[str, str]]:
    return [
        {
            "name": name,
            "source_group": f"e25_group_{index:02d}",
            "target_sha256": sha("a"),
        }
        for index, name in enumerate(e25.E25_NAMES)
    ]


def records_digest(records: list[dict[str, str]]) -> str:
    body = json.dumps(
        records, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(body).hexdigest()


def synthetic_manifest() -> dict:
    files = {}
    for index in range(7000):
        name = f"img_{index:06d}.png"
        files[name] = {"source_group": f"ordinary_group_{index:04d}", "sha256": sha("a")}
    for index, name in enumerate(e25.E25_NAMES):
        files[name] = {
            "source_group": f"e25_group_{index:02d}",
            "sha256": sha("a"),
        }
    return {
        "schema_version": 2,
        "files": files,
        "stats": {"files": 7000},
        "split": {
            "train_count": 6700,
            "val_count": 300,
            "eligible_confirmation": list(e25.E25_NAMES),
        },
    }


def structural_rows() -> list[dict]:
    return [
        {
            "image": image,
            "provenance_ok": True,
            "input_ok": True,
            "query_canonical_onehot": True,
            "finite_output": True,
            "dsu_legal": True,
            "legal_origin": True,
            "orientation_degrees": 0,
            "reflection": False,
            "proposal_denominator": 10,
            "accepted_denominator": 10,
            "true_relation_denominator": 10,
            "proposed_precision": 0.70,
            "true_relation_recall": 0.65,
            "exact_connected_coverage": 0.50,
            "cycle_rank_ratio": 0.05,
            "geometry_hypotheses": 450_000,
        }
        for image in e25.E25_IDS
    ]


def staged_rows() -> list[dict]:
    rows = []
    for index, image in enumerate(e25.E25_IDS):
        rows.append(
            {
                "image": image,
                "provenance_ok": True,
                "paired_identity_ok": True,
                "rr96_verified": True,
                "frozen_candidate_ok": True,
                "orientation_degrees": 0,
                "reflection": False,
                "solve_only_ssim_delta": 0.003,
                "final_ssim_delta": 0.004 if index < 30 else 0.0,
                "neighbour_delta": 0.005,
            }
        )
    return rows


class E25ContractSyntheticTests(unittest.TestCase):
    def test_frozen_ids_names_and_gates(self) -> None:
        self.assertEqual(len(e25.E25_IDS), 48)
        self.assertEqual(len(set(e25.E25_IDS)), 48)
        self.assertEqual(e25.E25_CANARY_ID, e25.E25_IDS[0])
        self.assertEqual(e25.STAGED_GATES["strict_positive_final_wins_min"], 30)
        self.assertEqual(e25.RESOURCE_LIMITS["feature_cache_bytes_max"], 24 * 1024**3)
        self.assertEqual(e25.RESOURCE_LIMITS["all_artifact_bytes_max"], 48 * 1024**3)
        self.assertEqual(e25.RESOURCE_LIMITS["peak_ram_bytes_max"], 16 * 1024**3)
        self.assertEqual(e25.RESOURCE_LIMITS["total_cpu_seconds_max"], 48 * 3600)
        names_body = "".join(f"{name}\n" for name in e25.E25_NAMES).encode("ascii")
        self.assertEqual(hashlib.sha256(names_body).hexdigest(), e25.E25_NEWLINE_LIST_SHA256)
        broker = e25.METRIC_BROKER_CONTRACT
        self.assertEqual(
            broker["structural_label_phase"]["allowed_scene_members"],
            ("permutation",),
        )
        self.assertFalse(broker["structural_label_phase"]["clean_target_opened"])
        self.assertEqual(
            broker["staged_image_phase"]["opens_only_after"],
            "authenticated_E25_structural_PASS",
        )

    def test_record_projection_is_ordered_distinct_and_hash_bound(self) -> None:
        records = synthetic_records()
        with mock.patch.object(
            e25, "E25_CANONICAL_RECORDS_SHA256", records_digest(records)
        ):
            self.assertEqual(e25.validate_sealed_records(records), records)
            duplicate = copy.deepcopy(records)
            duplicate[1]["source_group"] = duplicate[0]["source_group"]
            with self.assertRaises(e25.E25ContractError):
                e25.validate_sealed_records(duplicate)
            reordered = copy.deepcopy(records)
            reordered[0], reordered[1] = reordered[1], reordered[0]
            with self.assertRaises(e25.E25ContractError):
                e25.validate_sealed_records(reordered)

    def test_source_manifest_proves_full_disjointness_without_pixels(self) -> None:
        payload = synthetic_manifest()
        records = synthetic_records()
        with mock.patch.object(
            e25, "E25_CANONICAL_RECORDS_SHA256", records_digest(records)
        ):
            self.assertEqual(e25.project_and_validate_source_manifest(payload), records)
            overlap = copy.deepcopy(payload)
            first_name = e25.E25_NAMES[0]
            overlap["files"][first_name]["source_group"] = overlap["files"][
                "img_000000.png"
            ]["source_group"]
            modified = copy.deepcopy(records)
            modified[0]["source_group"] = overlap["files"][first_name]["source_group"]
            with mock.patch.object(
                e25, "E25_CANONICAL_RECORDS_SHA256", records_digest(modified)
            ):
                with self.assertRaisesRegex(e25.E25ContractError, "overlaps training"):
                    e25.project_and_validate_source_manifest(overlap)

    def test_real_pinned_records_literal_hash_is_self_consistent(self) -> None:
        # No file is opened: this is the already-frozen 48-record literal.
        expected = e25.E25_CANONICAL_RECORDS_SHA256
        self.assertEqual(len(expected), 64)
        self.assertTrue(all(char in "0123456789abcdef" for char in expected))

    def test_structural_gates_are_inclusive_and_complete(self) -> None:
        rows = structural_rows()
        summary = e25.summarize_structural(rows)
        decision = e25.structural_decision(summary)
        self.assertTrue(decision["passed"])
        self.assertTrue(all(decision["checks"].values()))
        failed = copy.deepcopy(rows)
        failed[0]["proposed_precision"] = 0.599999
        self.assertFalse(
            e25.structural_decision(e25.summarize_structural(failed))["passed"]
        )
        missing = rows[:-1]
        with self.assertRaises(e25.E25ContractError):
            e25.summarize_structural(missing)

    def test_structural_zero_denominator_nan_and_rotation_fail(self) -> None:
        for key, value in (
            ("proposal_denominator", 0),
            ("proposed_precision", float("nan")),
            ("provenance_ok", False),
            ("orientation_degrees", 90),
            ("reflection", True),
        ):
            rows = structural_rows()
            rows[0][key] = value
            with self.assertRaises(e25.E25ContractError):
                e25.summarize_structural(rows)

    def test_staged_exactly_thirty_strict_wins_passes(self) -> None:
        rows = staged_rows()
        summary = e25.summarize_staged(rows)
        self.assertEqual(summary["strict_positive_final_ssim_wins"], 30)
        decision = e25.staged_decision(summary)
        self.assertTrue(decision["passed"])
        self.assertTrue(all(decision["checks"].values()))

    def test_staged_twenty_nine_wins_fails_even_with_good_mean(self) -> None:
        rows = staged_rows()
        rows[29]["final_ssim_delta"] = 0.0
        rows[0]["final_ssim_delta"] = 0.1
        summary = e25.summarize_staged(rows)
        self.assertGreater(summary["mean_final_ssim_delta"], 0.002)
        self.assertEqual(summary["strict_positive_final_ssim_wins"], 29)
        decision = e25.staged_decision(summary)
        self.assertFalse(decision["passed"])
        self.assertFalse(decision["checks"]["strict_positive_final_ssim_wins"])

    def test_label_free_barrier_is_complete_e_only_and_under_caps(self) -> None:
        sealed = synthetic_records()
        record_sha = records_digest(sealed)
        source_sha, authority_sha, model_sha = sha("b"), sha("c"), sha("d")
        commits = []
        for index, image in enumerate(e25.E25_IDS):
            base = f"E:/pazzle_work/posegraph_e25_confirmation/label_free_v1/image_{image:04d}"
            commits.append(
                {
                    "schema": e25.LABEL_FREE_COMMIT_SCHEMA,
                    "schema_version": e25.SCHEMA_VERSION,
                    "image": image,
                    "name": e25.E25_NAMES[index],
                    "source_group": sealed[index]["source_group"],
                    "source_seal_sha256": source_sha,
                    "authority_sha256": authority_sha,
                    "final_model_sha256": model_sha,
                    "input_receipt_path": f"{base}/input.json",
                    "input_receipt_sha256": sha("e"),
                    "feature_path": f"{base}/features.npz",
                    "feature_sha256": sha("f"),
                    "feature_bytes": 1,
                    "prediction_path": f"{base}/predictions.npz",
                    "prediction_sha256": sha("1"),
                    "prediction_bytes": 1,
                    "worker_receipt_path": f"{base}/commit.json",
                    "worker_receipt_sha256": sha("2"),
                    "labels_targets_metrics_opened": False,
                    "orientation_degrees": 0,
                    "reflection": False,
                }
            )
        with mock.patch.object(e25, "E25_CANONICAL_RECORDS_SHA256", record_sha):
            barrier = e25.build_label_free_barrier(
                records=commits,
                sealed_records=sealed,
                source_seal_sha256=source_sha,
                authority_sha256=authority_sha,
                final_model_sha256=model_sha,
                canary_receipt_sha256=sha("3"),
                child_cpu_seconds=1.0,
                peak_rss_bytes=1,
                aggregate_artifact_bytes=96,
            )
            self.assertEqual(barrier["completed_images"], list(e25.E25_IDS))
            self.assertTrue(barrier["metric_broker_authorized"])
            bad = copy.deepcopy(commits)
            bad[0]["feature_path"] = "C:/forbidden/features.npz"
            with self.assertRaises(e25.E25ContractError):
                e25.build_label_free_barrier(
                    records=bad,
                    sealed_records=sealed,
                    source_seal_sha256=source_sha,
                    authority_sha256=authority_sha,
                    final_model_sha256=model_sha,
                    canary_receipt_sha256=sha("3"),
                    child_cpu_seconds=1.0,
                    peak_rss_bytes=1,
                    aggregate_artifact_bytes=96,
                )

    def test_label_free_barrier_fails_resource_overage(self) -> None:
        sealed = synthetic_records()
        record_sha = records_digest(sealed)
        source_sha, authority_sha, model_sha = sha("b"), sha("c"), sha("d")
        commits = []
        for index, image in enumerate(e25.E25_IDS):
            base = f"E:/pazzle_work/posegraph_e25_confirmation/label_free_v1/{image}"
            commits.append(
                {
                    "schema": e25.LABEL_FREE_COMMIT_SCHEMA,
                    "schema_version": 1,
                    "image": image,
                    "name": e25.E25_NAMES[index],
                    "source_group": sealed[index]["source_group"],
                    "source_seal_sha256": source_sha,
                    "authority_sha256": authority_sha,
                    "final_model_sha256": model_sha,
                    "input_receipt_path": f"{base}/i.json",
                    "input_receipt_sha256": sha("e"),
                    "feature_path": f"{base}/f.npz",
                    "feature_sha256": sha("f"),
                    "feature_bytes": 1,
                    "prediction_path": f"{base}/p.npz",
                    "prediction_sha256": sha("1"),
                    "prediction_bytes": 1,
                    "worker_receipt_path": f"{base}/c.json",
                    "worker_receipt_sha256": sha("2"),
                    "labels_targets_metrics_opened": False,
                    "orientation_degrees": 0,
                    "reflection": False,
                }
            )
        with mock.patch.object(e25, "E25_CANONICAL_RECORDS_SHA256", record_sha):
            with self.assertRaisesRegex(e25.E25ContractError, "resource"):
                e25.build_label_free_barrier(
                    records=commits,
                    sealed_records=sealed,
                    source_seal_sha256=source_sha,
                    authority_sha256=authority_sha,
                    final_model_sha256=model_sha,
                    canary_receipt_sha256=sha("3"),
                    child_cpu_seconds=e25.TOTAL_CPU_SECONDS_MAX + 0.1,
                    peak_rss_bytes=1,
                    aggregate_artifact_bytes=96,
                )


if __name__ == "__main__":
    unittest.main()
