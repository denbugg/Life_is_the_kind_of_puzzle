from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_dense_pair_residual_pilot.py"
SPEC = importlib.util.spec_from_file_location("dense_pair_result_verifier", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verifier
SPEC.loader.exec_module(verifier)

PROTOCOL_SPEC = importlib.util.spec_from_file_location(
    "dense_pair_result_verifier_test_protocol",
    ROOT / "src/puzzle_assembly/protocol.py",
)
assert PROTOCOL_SPEC is not None and PROTOCOL_SPEC.loader is not None
protocol = importlib.util.module_from_spec(PROTOCOL_SPEC)
PROTOCOL_SPEC.loader.exec_module(protocol)


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _authoritative(split: str) -> list[str]:
    return protocol.source_names_for_split(
        split,
        manifest_path=ROOT / "configs/denoise_splits_seed20260710.json",
        quarantine_path=ROOT / "configs/denoise_validation_quarantine_v1.json",
        audit_exclusion_path=ROOT / "configs/assembly_audit_exclusion_v1.json",
    )


def _provenance() -> dict:
    specs = {
        "train": ("edge_train", 4096, 2),
        "selection": ("edge_development", 96, 2),
        "holdout": ("assembly_cal", 112, 2),
        "real_gate": ("assembly_incremental_gate", 128, 2),
        "final_audit": ("assembly_final_audit", 0, 2),
        "confirmation": ("assembly_final_audit", 64, 2),
    }
    result = {
        "kind": "dense_all_pairs_residual_pilot",
        "safe_for_submission": False,
        "all_negatives_contract": (
            "each sampled outgoing row and incoming column scores all 576 slots; "
            "self is masked, hence 575 valid alternatives in both orientations"
        ),
        "gate_contract": {"selection_order": list(verifier.GATE_ORDER)},
        "manifest": {
            "path": "/kaggle/manifest.json",
            "sha256": verifier.sha256(ROOT / "configs/denoise_splits_seed20260710.json"),
        },
        "quarantine": {
            "path": "/kaggle/quarantine.json",
            "sha256": verifier.sha256(
                ROOT / "configs/denoise_validation_quarantine_v1.json"
            ),
        },
        "audit_exclusion": {
            "path": "/kaggle/audit_exclusion.json",
            "sha256": verifier.sha256(
                ROOT / "configs/assembly_audit_exclusion_v1.json"
            ),
        },
    }
    for label, (split, offset, count) in specs.items():
        names = _authoritative(split)[offset : offset + count]
        result[f"{label}_partition"] = f"{split}[{offset}:{offset + count}]"
        result[f"{label}_names"] = names
        result[f"{label}_names_sha256"] = verifier.names_sha256(names)
    result["quick_selection_names"] = result["selection_names"][:1]
    result["quick_selection_names_sha256"] = verifier.names_sha256(
        result["quick_selection_names"]
    )
    return result


def _retrieval(names: list[str], *, passed: bool) -> dict:
    delta = 0.02 if passed else 0.0
    lower = 0.01 if passed else -0.01
    aggregate = {
        "mean_delta_recall_at_1": delta,
        "mean_delta_mrr": delta,
        "mean_delta_recall_at_32": 0.0,
        "bootstrap_95_delta_recall_at_1": [lower, 0.03],
        "panels": {
            "primary_kornia": {"mean_delta_recall_at_1": delta},
            "independent_libjpeg": {"mean_delta_recall_at_1": delta},
        },
    }
    checks = {
        "mean_recall_at_1_delta_ge_0.01": passed,
        "mean_mrr_delta_ge_0.01": passed,
        "mean_recall_at_32_delta_ge_minus_0.005": True,
        "bootstrap_recall_at_1_lower_gt_0": passed,
        "every_panel_recall_at_1_positive": passed,
    }
    records = [{"name": name, "panel": "primary_kornia"} for name in names]
    return {
        "records": records,
        "aggregate": aggregate,
        "gate": {"passed": all(checks.values()), "checks": checks},
    }


def _qap(names: list[str]) -> dict:
    aggregate = {
        "mean_delta_ssim": 0.01,
        "mean_delta_adjacency": 0.02,
        "bootstrap_95_delta_ssim": [0.001, 0.02],
        "panels": {
            "primary_kornia": {"mean_delta_ssim": 0.01},
            "independent_libjpeg": {"mean_delta_ssim": 0.01},
        },
    }
    checks = {
        "mean_qap_ssim_delta_ge_0.005": True,
        "mean_qap_adjacency_delta_ge_0.01": True,
        "bootstrap_qap_ssim_lower_gt_0": True,
        "every_panel_qap_ssim_positive": True,
    }
    return {
        "records": [{"name": name} for name in names],
        "aggregate": aggregate,
        "gate": {"passed": True, "checks": checks},
    }


def _selection(names: list[str], *, passed: bool) -> dict:
    retrieval = _retrieval(names, passed=passed)
    return {
        "split": "cheap_selection_edge_development",
        "names": names,
        "names_sha256": verifier.names_sha256(names),
        "retrieval": {
            "records": retrieval["records"],
            "aggregate": retrieval["aggregate"],
        },
        "retrieval_gate": retrieval["gate"],
        "synthetic_target_files_opened": True,
        "qap_metrics_computed": False,
    }


def _holdout(names: list[str]) -> dict:
    retrieval = _retrieval(names, passed=True)
    qap = _qap(names)
    return {
        "split": "synthetic_transfer_assembly_cal",
        "names": names,
        "names_sha256": verifier.names_sha256(names),
        "retrieval": {
            "records": retrieval["records"],
            "aggregate": retrieval["aggregate"],
        },
        "retrieval_gate": retrieval["gate"],
        "synthetic_target_files_opened": True,
        "qap_metrics_computed": True,
        "qap": {"records": qap["records"], "aggregate": qap["aggregate"]},
        "qap_gate": qap["gate"],
    }


def _write_checkpoints(output: Path, provenance: dict, tag: str) -> tuple[str, dict]:
    model_config = {"fixture": tag}
    metadata = {**copy.deepcopy(provenance), "fixture": tag, "safe_for_submission": False}
    base = {
        "schema_version": 1,
        "kind": "puzzle_dense_pair_residual",
        "safe_for_submission": False,
        "model_config": model_config,
        "model_state": {"weight": torch.tensor([1.0])},
        "metadata": metadata,
    }
    torch.save(base, output / verifier.BEST_CHECKPOINT)
    latest = copy.deepcopy(base)
    latest["metadata"]["latest_completed_epoch"] = 1
    latest["training_state"] = {"capture_point": "epoch_boundary"}
    torch.save(latest, output / verifier.LATEST_CHECKPOINT)
    return verifier.sha256(output / verifier.BEST_CHECKPOINT), metadata


def _write_real_phase(
    output: Path,
    report: dict,
    provenance: dict,
    checkpoint_hash: str,
) -> None:
    names = list(provenance["real_gate_names"])
    frozen = (
        output
        / "frozen_real_predictions"
        / "frozen_original_real_input_gate"
    )
    frozen.mkdir(parents=True)
    records = []
    score_records = []
    for index, name in enumerate(names):
        base_layout = np.arange(576, dtype=np.int32)
        candidate_layout = base_layout[::-1].copy()
        base_path = frozen / f"{Path(name).stem}.base.png"
        candidate_path = frozen / f"{Path(name).stem}.candidate.png"
        Image.new("RGB", (480, 480), (index, 0, 0)).save(base_path)
        Image.new("RGB", (480, 480), (0, index + 1, 0)).save(candidate_path)
        base_hash = hashlib.sha256(base_layout.tobytes()).hexdigest()
        candidate_hash = hashlib.sha256(candidate_layout.tobytes()).hexdigest()
        records.append(
            {
                "name": name,
                "input_pixel_sha256": hashlib.sha256(name.encode()).hexdigest(),
                "base_layout": base_layout.tolist(),
                "base_layout_sha256": base_hash,
                "candidate_layout": candidate_layout.tolist(),
                "candidate_layout_sha256": candidate_hash,
                "base_render": f"/kaggle/working/{base_path.name}",
                "base_render_sha256": verifier.sha256(base_path),
                "candidate_render": f"/kaggle/working/{candidate_path.name}",
                "candidate_render_sha256": verifier.sha256(candidate_path),
                "qap_seed": int.from_bytes(
                    hashlib.sha256(name.encode()).digest()[:4], "little"
                )
                + 7001,
            }
        )
        score_records.append(
            {
                "name": name,
                "base": {"ssim": 0.1},
                "candidate": {"ssim": 0.11},
                "delta_ssim": 0.01,
                "base_layout_sha256": base_hash,
                "candidate_layout_sha256": candidate_hash,
            }
        )

    names_hash = verifier.names_sha256(names)
    payload = {
        "schema_version": 1,
        "kind": "dense_pair_input_only_frozen_predictions",
        "split": "frozen_original_real_input_gate",
        "source_names": names,
        "source_names_sha256": names_hash,
        "candidate_checkpoint_path": "/kaggle/working/dense_pair_residual_best.pt",
        "candidate_checkpoint_sha256": checkpoint_hash,
        "target_files_opened": False,
        "records": records,
    }
    envelope = {
        "payload": payload,
        "payload_sha256": verifier.canonical_json_sha256(payload),
    }
    manifest = frozen / verifier.PHASE_A_NAME
    _json(manifest, envelope)
    manifest_hash = verifier.sha256(manifest)
    event = {
        "schema_version": 1,
        "kind": "dense_pair_target_access_event",
        "split": "frozen_original_real_input_gate",
        "phase_a_manifest_sha256": manifest_hash,
        "phase_a_payload_sha256": envelope["payload_sha256"],
        "candidate_checkpoint_sha256": checkpoint_hash,
        "source_names_sha256": names_hash,
        "target_access_started": True,
        "target_files_may_have_been_opened": True,
    }
    event_path = frozen / verifier.TARGET_EVENT_NAME
    _json(event_path, event)
    aggregate = {
        "mean_delta_ssim": 0.01,
        "bootstrap_95_delta_ssim": [0.001, 0.02],
        "win_rate": 1.0,
    }
    checks = {
        "mean_real_ssim_delta_ge_0.005": True,
        "bootstrap_real_ssim_lower_gt_0": True,
        "real_ssim_win_rate_ge_0.60": True,
    }
    report["real_gate"] = {
        "split": "frozen_original_real_input_gate",
        "source_names": names,
        "source_names_sha256": names_hash,
        "phase_a_manifest": f"/kaggle/working/{verifier.PHASE_A_NAME}",
        "phase_a_manifest_sha256": manifest_hash,
        "phase_a_payload_sha256": envelope["payload_sha256"],
        "target_access_event": f"/kaggle/working/{verifier.TARGET_EVENT_NAME}",
        "target_access_event_sha256": verifier.sha256(event_path),
        "target_opened_after_predictions_frozen": True,
        "records": score_records,
        "aggregate": aggregate,
        "gate": {"passed": True, "checks": checks},
    }


def _write_report_dir(output: Path, tag: str, *, real: bool) -> dict:
    output.mkdir(parents=True)
    provenance = _provenance()
    checkpoint_hash, checkpoint_metadata = _write_checkpoints(output, provenance, tag)
    selection_pass = real
    report = {
        "schema_version": 1,
        "kind": "dense_all_pairs_residual_pilot_report",
        "status": "continue_candidate_only" if real else "stop_cheap_selection_retrieval",
        "safe_for_submission": False,
        "provenance": provenance,
        "model_config": {"fixture": tag},
        "candidate_checkpoint_sha256": checkpoint_hash,
        "checkpoint_metadata": checkpoint_metadata,
        "training": {"fixture": tag},
        "selection": _selection(provenance["selection_names"], passed=selection_pass),
        "holdout": _holdout(provenance["holdout_names"]) if real else None,
        "real_gate": None,
        "gate_opened": {
            "synthetic_transfer": real,
            "original_real_input": real,
            "true_final_audit": False,
            "true_confirmation": False,
        },
        "audit_policy": "assembly_final_audit remains sealed until formal promotion",
        "preflight": {"fixture": True},
    }
    if real:
        _write_real_phase(output, report, provenance, checkpoint_hash)
    _json(output / verifier.REPORT_NAME, report)
    (output / verifier.HASHES_NAME).write_text(
        "".join(
            f"{verifier.sha256(output / name)}  {name}\n"
            for name in verifier.CHECKSUM_ARTIFACTS
        ),
        encoding="utf-8",
    )
    return report


def _report_ref(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": f"/kaggle/working/{path.parent.name}/{path.name}",
        "sha256": verifier.sha256(path),
        "status": report["status"],
        "gate_opened": report["gate_opened"],
    }


def _write_tree(root: Path, *, real: bool = False) -> tuple[Path, Path]:
    smoke_dir = root / "dense_pair_residual_smoke"
    pilot_dir = root / "dense_pair_residual_pilot"
    _write_report_dir(smoke_dir, "smoke", real=False)
    _write_report_dir(pilot_dir, "pilot", real=real)
    logs = []
    for name, text in (
        ("dense_pair_residual_tests.log", "tests passed\n"),
        ("dense_pair_residual_smoke.log", "smoke passed\n"),
        ("dense_pair_residual_pilot.log", "pilot passed\n"),
    ):
        path = root / name
        path.write_text(text, encoding="utf-8")
        logs.append(path)
    labels = [
        "dense-pair unit tests",
        "2xT4 full-model one-step smoke",
        "bounded dense-pair residual pilot",
    ]
    steps = []
    for index, (label, log) in enumerate(zip(labels, logs, strict=True)):
        step = {
            "label": label,
            "command": ["python", f"step-{index}"],
            "returncode": 0,
            "timed_out": False,
            "timeout_seconds": 10,
            "seconds": 1.0,
            "log": f"/kaggle/working/{log.name}",
            "log_sha256": verifier.sha256(log),
        }
        if index == 1:
            step["report"] = _report_ref(smoke_dir / verifier.REPORT_NAME)
        if index == 2:
            step["report"] = _report_ref(pilot_dir / verifier.REPORT_NAME)
        steps.append(step)
    wrapper = {
        "schema_version": 1,
        "kind": "dense_pair_residual_kaggle_wrapper",
        "status": "complete",
        "safe_for_submission": False,
        "started_unix": 1.0,
        "completed_unix": 2.0,
        "base": {"mode": "fixture"},
        "base_hashes": {"src/base.py": "a" * 64},
        "overlay": {
            "mode": "fixture",
            "staged_hashes": {"src/model.py": "b" * 64},
        },
        "data_root": "/kaggle/input/puzzle",
        "assets": {
            "denoiser": {"path": "/kaggle/denoiser.pt", "sha256": "c" * 64},
            "hbt": {"path": "/kaggle/hbt.pt", "sha256": "d" * 64},
            "manifest": {
                "path": "/kaggle/manifest.json",
                "sha256": verifier.sha256(
                    ROOT / "configs/denoise_splits_seed20260710.json"
                ),
            },
            "quarantine": {
                "path": "/kaggle/quarantine.json",
                "sha256": verifier.sha256(
                    ROOT / "configs/denoise_validation_quarantine_v1.json"
                ),
            },
            "audit_exclusion": {
                "path": "/kaggle/audit_exclusion.json",
                "sha256": verifier.sha256(
                    ROOT / "configs/assembly_audit_exclusion_v1.json"
                ),
            },
        },
        "hardware": {
            "torch": "fixture",
            "cuda_runtime": "fixture",
            "device_count": 2,
            "devices": [
                {
                    "index": index,
                    "name": "Tesla T4",
                    "capability": [7, 5],
                    "total_memory": 16_000_000_000,
                    "tensor_probe": 0.5,
                }
                for index in range(2)
            ],
            "nvidia_smi": ["Tesla T4", "Tesla T4"],
        },
        "steps": steps,
        "pilot_report": copy.deepcopy(steps[-1]["report"]),
    }
    _json(root / verifier.WRAPPER_NAME, wrapper)
    return smoke_dir, pilot_dir


def _refresh_report_and_wrapper(root: Path, report_dir: Path) -> None:
    report_path = report_dir / verifier.REPORT_NAME
    (report_dir / verifier.HASHES_NAME).write_text(
        "".join(
            f"{verifier.sha256(report_dir / name)}  {name}\n"
            for name in verifier.CHECKSUM_ARTIFACTS
        ),
        encoding="utf-8",
    )
    wrapper_path = root / verifier.WRAPPER_NAME
    wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    reference = _report_ref(report_path)
    wrapper["steps"][-1]["report"] = reference
    wrapper["pilot_report"] = copy.deepcopy(reference)
    _json(wrapper_path, wrapper)


def test_verifies_complete_fail_closed_stopped_pilot(tmp_path: Path) -> None:
    _write_tree(tmp_path)
    result = verifier.verify_dense_pair_residual_pilot(tmp_path, repo_root=ROOT)
    assert result["verified"] is True
    assert result["safe_for_submission"] is False
    assert result["reports"]["pilot"]["status"] == "stop_cheap_selection_retrieval"
    assert result["reports"]["pilot"]["audit_unopened"] is True
    assert result["reports"]["pilot"]["phase_a"] is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("audit_opened", "illegally opened"),
        ("source_slice", "exact authoritative"),
        ("status_order", "disagrees with sequential gates"),
    ],
)
def test_rejects_audit_slice_and_gate_order_mutations(
    tmp_path: Path, mutation: str, message: str
) -> None:
    _, pilot_dir = _write_tree(tmp_path)
    report_path = pilot_dir / verifier.REPORT_NAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if mutation == "audit_opened":
        report["gate_opened"]["true_final_audit"] = True
    elif mutation == "source_slice":
        report["provenance"]["selection_names"][0] = "forged.png"
        report["provenance"]["selection_names_sha256"] = verifier.names_sha256(
            report["provenance"]["selection_names"]
        )
    elif mutation == "status_order":
        report["status"] = "continue_candidate_only"
    else:  # pragma: no cover
        raise AssertionError(mutation)
    _json(report_path, report)
    _refresh_report_and_wrapper(tmp_path, pilot_dir)
    with pytest.raises(verifier.VerificationError, match=message):
        verifier.verify_dense_pair_residual_pilot(tmp_path, repo_root=ROOT)


def test_verifies_real_phase_a_event_and_render_hashes(tmp_path: Path) -> None:
    _, pilot_dir = _write_tree(tmp_path, real=True)
    result = verifier.verify_dense_pair_residual_pilot(tmp_path, repo_root=ROOT)
    phase_a = result["reports"]["pilot"]["phase_a"]
    assert phase_a["passed"] is True
    assert phase_a["source_count"] == 2
    assert phase_a["render_count"] == 4

    render = next(pilot_dir.rglob("*.candidate.png"))
    render.write_bytes(render.read_bytes() + b"tamper")
    with pytest.raises(verifier.VerificationError, match="render hash mismatch"):
        verifier.verify_dense_pair_residual_pilot(tmp_path, repo_root=ROOT)


def test_rejects_reanchored_but_semantically_forged_target_event(tmp_path: Path) -> None:
    _, pilot_dir = _write_tree(tmp_path, real=True)
    event_path = next(pilot_dir.rglob(verifier.TARGET_EVENT_NAME))
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["target_access_started"] = False
    _json(event_path, event)
    report_path = pilot_dir / verifier.REPORT_NAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["real_gate"]["target_access_event_sha256"] = verifier.sha256(event_path)
    _json(report_path, report)
    _refresh_report_and_wrapper(tmp_path, pilot_dir)
    with pytest.raises(verifier.VerificationError, match="immutable Phase-A anchor"):
        verifier.verify_dense_pair_residual_pilot(tmp_path, repo_root=ROOT)
