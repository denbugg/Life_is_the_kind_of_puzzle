#!/usr/bin/env python3
"""Run the precommitted source-disjoint actual-layout confirmation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from puzzle_assembly.protocol import source_names_for_split  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze-layouts", "score"))
    parser.add_argument(
        "--config", default="configs/postassembly_actual_qap_confirmation_v1.json"
    )
    parser.add_argument("--phase-a-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--classical-chunk-size", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _names_sha256(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_hash(path: Path, expected: str, role: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"{role} hash mismatch: expected {expected}, got {actual}")


def _load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_actual_layout_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[str], Any, Path]:
    confirmation = _json(config_path)
    if confirmation.get("kind") != "postassembly_actual_qap_source_disjoint_confirmation":
        raise RuntimeError("unexpected confirmation protocol kind")
    if confirmation.get("status") != "precommitted_before_confirmation_pixel_or_metric_access":
        raise RuntimeError("confirmation protocol is not precommitted")

    decision = confirmation["decision_basis"]
    actual_config_path = REPO_ROOT / decision["actual_layout_protocol"]
    _require_hash(
        actual_config_path,
        decision["actual_layout_protocol_sha256"],
        "actual-layout protocol",
    )
    _require_hash(
        REPO_ROOT / decision["actual_layout_full32_report"],
        decision["actual_layout_full32_report_sha256"],
        "actual-layout development report",
    )
    implementation = confirmation["implementation"]
    runner_path = REPO_ROOT / implementation["frozen_runner"]
    _require_hash(runner_path, implementation["frozen_runner_sha256"], "frozen runner")
    runner = _load_runner(runner_path)
    if implementation.get("reuse_actual_layout_protocol_without_retuning") is not True:
        raise RuntimeError("confirmation attempts to retune the actual-layout protocol")
    for key in (
        "layout_refinement_forbidden",
        "candidate_routing_forbidden",
        "same_qap_w4_layout_for_all_render_arms",
    ):
        if implementation.get(key) is not True:
            raise RuntimeError(f"confirmation safety drift: {key}")
    if implementation.get("luminance_gain_included") is not False:
        raise RuntimeError("confirmation unexpectedly includes luminance gain")

    actual = _json(actual_config_path)
    verified_actual, _, _ = runner._protocol(actual_config_path, limit=32)
    if verified_actual != actual:
        raise RuntimeError("frozen runner resolved a different actual-layout protocol")
    names = [str(value) for value in confirmation["source_selection"]["names"]]
    if len(names) != 32 or len(set(names)) != 32:
        raise RuntimeError("confirmation must contain 32 unique sources")
    if _names_sha256(names) != confirmation["source_selection"]["names_sha256"]:
        raise RuntimeError("confirmation source hash mismatch")
    manifest_path = REPO_ROOT / actual["assets"]["manifest"]["path"]
    quarantine_path = REPO_ROOT / actual["assets"]["quarantine"]["path"]
    edge_development = source_names_for_split(
        "edge_development",
        manifest_path=manifest_path,
        quarantine_path=quarantine_path,
    )
    indices = [edge_development.index(name) for name in names]
    if indices != sorted(indices) or indices[0] != 201 or indices[-1] != 241:
        raise RuntimeError("confirmation selection indices drifted")
    base = _json(REPO_ROOT / actual["base_harmonizer_protocol"]["path"])
    previous_names = set(base["source_selection"]["names"])
    if previous_names & set(names):
        raise RuntimeError("confirmation overlaps the actual-layout development panel")
    if set(edge_development[128:160]) & set(names):
        raise RuntimeError("confirmation overlaps candidate graph oracle sources")
    for split in (
        "assembly_cal",
        "assembly_incremental_gate",
        "assembly_audit_exposed",
        "assembly_final_audit",
    ):
        values = source_names_for_split(
            split,
            manifest_path=manifest_path,
            quarantine_path=quarantine_path,
            audit_exclusion_path=REPO_ROOT / "configs/assembly_audit_exclusion_v1.json",
        )
        if set(names) & set(values):
            raise RuntimeError(f"confirmation overlaps sealed split: {split}")
    test_names = {path.name for path in (REPO_ROOT / "puzzle/test").glob("*.png")}
    if set(names) & test_names:
        raise RuntimeError("confirmation overlaps test basenames")

    frozen_gate = confirmation["frozen_gate"]
    actual_gate = actual["full32_gate"]
    for key in (
        "per_panel_mean_ssim_delta_minimum",
        "per_panel_paired_bootstrap_lower_must_exceed",
        "per_panel_mean_target_referenced_seam_error_delta_maximum",
    ):
        if frozen_gate[key] != actual_gate[key]:
            raise RuntimeError(f"confirmation gate differs from development gate: {key}")
    effective = copy.deepcopy(actual)
    return confirmation, effective, names, runner, runner_path


def main() -> None:
    args = _args()
    if args.denoise_batch_size <= 0 or args.classical_chunk_size <= 0 or args.torch_threads <= 0:
        raise SystemExit("batch/chunk/thread arguments must be positive")
    config_path = (REPO_ROOT / args.config).resolve()
    confirmation, effective, names, runner, runner_path = _validate(config_path)
    forwarded = argparse.Namespace(
        action=args.action,
        config=str(config_path.relative_to(REPO_ROOT)),
        phase_a_dir=args.phase_a_dir,
        output_dir=args.output_dir,
        limit=32,
        device=args.device,
        denoise_batch_size=args.denoise_batch_size,
        classical_chunk_size=args.classical_chunk_size,
        torch_threads=args.torch_threads,
    )
    phase_root = (REPO_ROOT / args.phase_a_dir).resolve()
    if args.action == "freeze-layouts":
        if args.output_dir:
            raise SystemExit("freeze-layouts does not accept --output-dir")
        runner._freeze_layouts(forwarded, config_path, effective, names)
        manifest_path = phase_root / "manifest.json"
        _write_json(
            phase_root / "CONFIRMATION_BINDING.json",
            {
                "kind": "postassembly_actual_confirmation_phase_a_binding",
                "confirmation_config_sha256": _sha256(config_path),
                "wrapper_sha256": _sha256(Path(__file__).resolve()),
                "frozen_runner_sha256": _sha256(runner_path),
                "phase_a_manifest_sha256": _sha256(manifest_path),
                "source_names_sha256": _names_sha256(names),
            },
        )
        return

    if not args.output_dir:
        raise SystemExit("score requires --output-dir")
    binding_path = phase_root / "CONFIRMATION_BINDING.json"
    binding = _json(binding_path)
    expected_binding = {
        "kind": "postassembly_actual_confirmation_phase_a_binding",
        "confirmation_config_sha256": _sha256(config_path),
        "wrapper_sha256": _sha256(Path(__file__).resolve()),
        "frozen_runner_sha256": _sha256(runner_path),
        "phase_a_manifest_sha256": _sha256(phase_root / "manifest.json"),
        "source_names_sha256": _names_sha256(names),
    }
    if binding != expected_binding:
        raise RuntimeError("confirmation Phase A binding mismatch")
    runner._score(forwarded, config_path, effective, _json(REPO_ROOT / effective["base_harmonizer_protocol"]["path"]), names)

    output_root = (REPO_ROOT / args.output_dir).resolve()
    report_path = output_root / "report.json"
    report = _json(report_path)
    gate = report.get("gate")
    if not isinstance(gate, dict) or gate.get("kind") != "full32_gate":
        raise RuntimeError("confirmation report has no full32 gate")
    passed = gate.get("passed") is True
    decision = {
        "kind": "postassembly_actual_qap_source_disjoint_confirmation_decision",
        "status": (
            "source_disjoint_confirmation_passed_candidate_for_production_integration"
            if passed
            else "source_disjoint_confirmation_failed_retain_current_renderer"
        ),
        "passed": passed,
        "submission_promotion_allowed_by_this_development_confirmation": False,
        "production_integration_eligible": passed,
        "report": str(report_path.relative_to(REPO_ROOT)),
        "report_sha256": _sha256(report_path),
        "confirmation_config_sha256": _sha256(config_path),
        "source_names_sha256": _names_sha256(names),
        "gate": gate,
    }
    _write_json(output_root / "CONFIRMATION_DECISION.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
