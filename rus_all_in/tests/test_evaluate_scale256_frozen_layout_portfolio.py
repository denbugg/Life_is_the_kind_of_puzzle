from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aiijc_puzzle.protocol import sha256_file
from scripts import evaluate_scale256_frozen_layout_portfolio as evaluator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/scale256_frozen_layout_evaluator_unsigned_template_v1.json"
)


def test_trusted_metrics_cover_exact_pairs_l1_and_radius() -> None:
    reference = np.arange(4, dtype=np.int32)
    perfect = evaluator.evaluate_candidate_layout(reference, reference, grid=2)
    assert perfect == {
        "exact_count": 4,
        "exact_rate": 1.0,
        "satisfied_pairs": 4,
        "satisfied_pairs_rate": 1.0,
        "manhattan_l1_per_tile": 0.0,
        "radius_le_1_rate": 1.0,
        "radius_le_2_rate": 1.0,
    }

    swapped = evaluator.evaluate_candidate_layout(
        np.asarray([1, 0, 2, 3], dtype=np.int32), reference, grid=2
    )
    assert swapped["exact_count"] == 2
    assert swapped["exact_rate"] == 0.5
    assert swapped["satisfied_pairs"] == 1
    assert swapped["satisfied_pairs_rate"] == 0.25
    assert swapped["manhattan_l1_per_tile"] == 0.5
    assert swapped["radius_le_1_rate"] == 1.0
    assert swapped["radius_le_2_rate"] == 1.0


def _tiny_case_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    identity = np.arange(4, dtype=np.int32)
    relation = {"relation_truth_selector_layout": identity}
    relation.update(
        {
            f"relation_arm_{arm}_layout": identity
            for arm in evaluator.FUSION_ARM_NAMES
        }
    )
    portfolio = {
        "portfolio_layouts": np.stack(
            [identity for _ in evaluator.PORTFOLIO_MEMBER_NAMES]
        ),
        "selected_arm_indices": np.zeros(
            len(evaluator.PORTFOLIO_MEMBER_NAMES), dtype=np.int32
        ),
    }
    row = {
        "portfolio_member_names": list(evaluator.PORTFOLIO_MEMBER_NAMES),
        "selected_arms": [
            evaluator.FUSION_ARM_NAMES[0] for _ in evaluator.PORTFOLIO_MEMBER_NAMES
        ],
        "changed_from_incumbent": [
            False for _ in evaluator.PORTFOLIO_MEMBER_NAMES
        ],
    }
    return relation, portfolio, row


def test_candidate_roster_is_fixed_and_portfolio_incumbent_must_match() -> None:
    relation, portfolio, row = _tiny_case_arrays()
    candidates = evaluator.load_case_candidates(
        relation, portfolio, row, grid=2
    )
    assert tuple(candidates) == evaluator.CANDIDATE_NAMES

    portfolio["portfolio_layouts"] = portfolio["portfolio_layouts"].copy()
    portfolio["portfolio_layouts"][0] = np.asarray([1, 0, 2, 3])
    with pytest.raises(RuntimeError, match="incumbent_keep"):
        evaluator.load_case_candidates(relation, portfolio, row, grid=2)


def test_source_clustered_statistics_are_fixed_deterministic_and_directional() -> None:
    sources = [f"source_{index:02d}" for index in range(evaluator.DEV_COUNT)]
    raw = np.asarray([1.0] * 10 + [0.0] * 4 + [-1.0] * 50)
    indices = evaluator._shared_bootstrap_indices(evaluator.DEV_COUNT)
    first = evaluator.summarize_delta_metric(
        sources, raw, direction=1, bootstrap_indices=indices
    )
    second = evaluator.summarize_delta_metric(
        sources, raw, direction=1, bootstrap_indices=indices
    )
    assert first == second
    source = first["source_clustered"]
    assert source["wins_ties_losses"] == {"wins": 10, "ties": 4, "losses": 50}
    assert source["worst_source"] == {
        "source_filename": "source_14",
        "raw_delta": -1.0,
        "benefit_delta": -1.0,
    }
    assert source["positive_mass"]["largest_source_share"] == pytest.approx(0.1)
    assert len(source["raw_mean_bootstrap_95pct_ci"]) == 2

    absolute = evaluator.summarize_absolute_metric(
        sources, raw, bootstrap_indices=indices
    )
    assert len(absolute["source_clustered"]["mean_bootstrap_95pct_ci"]) == 2

    lower_is_better = evaluator.summarize_delta_metric(
        sources, -raw, direction=-1, bootstrap_indices=indices
    )
    assert lower_is_better["source_clustered"]["wins_ties_losses"] == {
        "wins": 10,
        "ties": 4,
        "losses": 50,
    }


def test_layout_multiplicity_is_reported_without_deduplication() -> None:
    cases = []
    for case_index in range(evaluator.DEV_COUNT):
        digests = {
            name: ("same" if name != evaluator.CANDIDATE_NAMES[-1] else "other")
            for name in evaluator.CANDIDATE_NAMES
        }
        cases.append(
            {
                "layout_equivalence_classes": [
                    list(evaluator.CANDIDATE_NAMES[:-1]),
                    [evaluator.CANDIDATE_NAMES[-1]],
                ],
                "layout_sha256": digests,
                "case_index": case_index,
            }
        )
    summary = evaluator.summarize_multiplicity(cases)
    assert summary["declared_candidate_count"] == len(evaluator.CANDIDATE_NAMES)
    assert summary["unique_layout_count_per_case"]["mean"] == 2.0
    assert summary["same_as_incumbent_case_count"][evaluator.CANDIDATE_NAMES[-1]] == 0
    assert summary["deduplicated_for_weighting_or_selection"] is False


def test_receipt_failure_prevents_reference_loader() -> None:
    events: list[str] = []

    def verifier(_: object) -> evaluator.VerifiedInputs:
        events.append("verify")
        raise RuntimeError("receipt mismatch")

    def reference_loader(_: evaluator.VerifiedInputs) -> object:
        events.append("reference")
        return object()

    with pytest.raises(RuntimeError, match="receipt mismatch"):
        evaluator.score_after_verified_inputs(
            {},
            verifier=verifier,
            reference_loader=reference_loader,
            scorer=lambda _verified, _references: None,
        )
    assert events == ["verify"]


def test_individual_receipt_record_is_hash_strict(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"target-free-layout")
    receipt = {
        "artifacts": {
            "archive": {
                "path": str(artifact.resolve()),
                "sha256": "0" * 64,
            }
        }
    }
    with pytest.raises(RuntimeError, match="does not bind"):
        evaluator._verify_receipt_record(
            receipt, "artifacts", "archive", artifact
        )
    receipt["artifacts"]["archive"]["sha256"] = sha256_file(artifact)
    evaluator._verify_receipt_record(receipt, "artifacts", "archive", artifact)


def test_unsigned_template_is_exact_but_fail_closed() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    evaluator._require_exact_contract(config)
    assert config["status"] == evaluator.BLOCKED_STATUS
    assert config["execution_authorized"] is False
    assert tuple(
        config["evaluation_contract"]["candidate_names_in_report_order"]
    ) == evaluator.CANDIDATE_NAMES
    assert config["frozen_inputs"]["joint_archive"]["sha256"] == (
        "b2b153b728227950f1645dab2bf77d581c17a0fcd707c71dd96f1fadc4beb0e3"
    )
    assert config["frozen_inputs"]["relation_roster_archive"]["sha256"] == (
        "d0d31d127b4148068394c203b92c2c51c3e0f85d6ef482c51e38892f0e74216e"
    )
    assert config["frozen_inputs"]["portfolio_archive"]["sha256"] == (
        "476f0dd447b77f851ddd455770faf4152fcf1b7dd64c874bc21d5135ddb5a7f6"
    )
    with pytest.raises(RuntimeError, match="intentionally blocked"):
        evaluator._load_signed_config(CONFIG_PATH)


def test_signed_protocol_requires_explicit_execution_authorization() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["status"] = evaluator.SIGNED_STATUS
    with pytest.raises(RuntimeError, match="did not authorize execution"):
        evaluator._require_exact_contract(config)

    config["execution_authorized"] = True
    evaluator._require_exact_contract(config)


def test_one_shot_report_is_claimed_before_reference_access(tmp_path: Path) -> None:
    report_path = tmp_path / "fixed-report.json"
    events: list[str] = []
    config = {"evaluation_contract": {"one_shot_report_path": str(report_path)}}

    def verifier(_: object) -> object:
        events.append("verify")
        return object()

    def reference_loader(_: object) -> object:
        assert report_path.exists()
        events.append("reference")
        return object()

    def scorer(_: object, _references: object) -> dict[str, int]:
        events.append("score")
        return {"case_count": 0}

    first = evaluator.execute_one_shot_report(
        config,
        "a" * 64,
        report_path,
        verifier=verifier,
        reference_loader=reference_loader,
        scorer=scorer,
    )
    assert first["case_count"] == 0
    assert events == ["verify", "reference", "score"]

    with pytest.raises(FileExistsError):
        evaluator.execute_one_shot_report(
            config,
            "a" * 64,
            report_path,
            verifier=verifier,
            reference_loader=reference_loader,
            scorer=scorer,
        )
    assert events == ["verify", "reference", "score", "verify"]


def test_runtime_report_path_cannot_be_redirected(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    args = evaluator.parse_args(
        [
            "--config",
            str(CONFIG_PATH),
            "--report",
            str(tmp_path / "alternate-report.json"),
        ]
    )
    with pytest.raises(RuntimeError, match="report path differs"):
        evaluator._require_runtime_paths(config, args)


def test_checked_in_target_free_receipts_verify_without_references() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    verified = evaluator.verify_frozen_inputs(config)
    assert len(verified.joint.rows) == evaluator.DEV_COUNT
    assert len(verified.relation.rows) == evaluator.DEV_COUNT
    assert len(verified.portfolio.rows) == evaluator.DEV_COUNT
