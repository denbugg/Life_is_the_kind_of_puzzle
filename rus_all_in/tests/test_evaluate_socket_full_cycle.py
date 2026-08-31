from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiijc_puzzle.synthetic_socket_evaluation import names_digest
from scripts.evaluate_socket_full_cycle import (
    _selection,
    _validate_source_report_binding,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture_report(tmp_path: Path) -> tuple[dict[str, object], SimpleNamespace]:
    arrays = tmp_path / "frozen.npz"
    metadata = tmp_path / "frozen.json"
    arrays.write_bytes(b"arrays")
    metadata.write_bytes(b"metadata")
    exposed = ("train_a.png", "train_b.png")
    contract = {"architecture": "fixture-v2"}
    checkpoint = SimpleNamespace(
        contract=contract,
        lineage=SimpleNamespace(
            exposed_filenames=exposed,
            exposed_digest=names_digest(exposed),
            exposed_count=len(exposed),
        ),
    )
    selected = ["eval_b.png", "eval_a.png"]
    report: dict[str, object] = {
        "checkpoint": {
            "sha256": "c" * 64,
            "architecture_contract": contract,
            "lineage_filenames": list(exposed),
            "lineage_digest": names_digest(exposed),
        },
        "protocol": {
            "manifest_digest": "m" * 64,
            "manifest_split": "train",
            "checkpoint_lineage_source_disjoint": True,
            "target_hashes_verified_before_use": True,
            "dirty_only_predictions_frozen_before_reference_scoring": True,
            "frozen_artifact_contains_exact_references": False,
        },
        "selection": {
            "source_limit": len(selected),
            "case_count": 4,
            "seed": 17,
            "draws_per_source": 2,
            "source_filenames": selected,
            "source_digest": names_digest(selected),
            "sources": [
                {"filename": name, "target_sha256": str(index) * 64}
                for index, name in enumerate(selected, start=1)
            ],
        },
        "fixed_candidates": {
            "decoder_edge_budget_per_axis": 144,
            "decoder_swap_steps": 24,
            "global": ["socket_ot_decoder"],
        },
        "frozen_predictions": {
            "arrays_path": str(arrays),
            "arrays_sha256": _sha(b"arrays"),
            "metadata_path": str(metadata),
            "metadata_sha256": _sha(b"metadata"),
        },
    }
    return report, checkpoint


def test_full_cycle_source_report_binding_recomputes_lineage_and_hashes(
    tmp_path: Path,
) -> None:
    report, checkpoint = _fixture_report(tmp_path)
    names, seed, draws, hashes = _selection(report)
    assert (seed, draws) == (17, 2)
    assert list(hashes) == names
    binding = _validate_source_report_binding(
        report,
        checkpoint=checkpoint,
        checkpoint_sha256="c" * 64,
        manifest_digest="m" * 64,
        selected_names=names,
    )
    assert binding["checkpoint_lineage_source_disjoint_recomputed"] is True
    assert binding["source_report_frozen_artifacts_verified"] is True


def test_full_cycle_source_report_binding_fails_closed_on_tampering(tmp_path: Path) -> None:
    report, checkpoint = _fixture_report(tmp_path)
    bad_digest = copy.deepcopy(report)
    bad_digest["selection"]["source_digest"] = "0" * 64
    with pytest.raises(ValueError, match="source_digest"):
        _selection(bad_digest)

    names, _, _, _ = _selection(report)
    with pytest.raises(ValueError, match="actual checkpoint lineage"):
        _validate_source_report_binding(
            report,
            checkpoint=checkpoint,
            checkpoint_sha256="c" * 64,
            manifest_digest="m" * 64,
            selected_names=[*names, "train_a.png"],
        )

    arrays_path = Path(report["frozen_predictions"]["arrays_path"])
    arrays_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="frozen arrays hash mismatch"):
        _validate_source_report_binding(
            report,
            checkpoint=checkpoint,
            checkpoint_sha256="c" * 64,
            manifest_digest="m" * 64,
            selected_names=names,
        )
