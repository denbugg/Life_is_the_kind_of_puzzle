#!/usr/bin/env python3
"""Kaggle wrapper for the frozen dual-LambdaRank QAP diagnostic."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
STAGING = WORKING / "dual_lambdarank_qap_diagnostic_code"
REPORT = WORKING / "dual_lambdarank_qap_diagnostic_report.json"
WRAPPER = WORKING / "dual_lambdarank_qap_diagnostic_wrapper.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one(pattern: str) -> Path:
    matches = sorted(INPUT.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern}, found {matches}")
    return matches[0]


def run(command: list[str], *, env: dict[str, str]) -> dict:
    started = time.time()
    completed = subprocess.run(command, env=env, text=True, capture_output=True)
    print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr, flush=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "seconds": time.time() - started,
        "stdout_tail": completed.stdout[-12000:],
        "stderr_tail": completed.stderr[-12000:],
    }


def main() -> None:
    started = time.time()
    payload = {
        "schema_version": 1,
        "kind": "dual_lambdarank_qap_diagnostic_kaggle_wrapper",
        "status": "running",
        "safe_for_submission": False,
    }
    try:
        bundle_matches = sorted(INPUT.rglob("dual_lambdarank_qap_diagnostic_bundle.zip"))
        if len(bundle_matches) == 1:
            bundle = bundle_matches[0]
            if STAGING.exists():
                shutil.rmtree(STAGING)
            STAGING.mkdir(parents=True)
            with zipfile.ZipFile(bundle) as archive:
                archive.extractall(STAGING)
            code_root = STAGING
            bundle_descriptor = {"mode": "zip", "path": str(bundle), "sha256": sha256(bundle)}
        elif len(bundle_matches) == 0:
            # Kaggle expands uploaded ZIP datasets into direct files at mount.
            script_probe = one("scripts/evaluate_dual_lambdarank_qap_diagnostic.py")
            code_root = script_probe.parents[1]
            bundle_descriptor = {"mode": "direct_expanded_dataset", "path": str(code_root)}
        else:
            raise RuntimeError(f"ambiguous code bundles: {bundle_matches}")
        script = code_root / "scripts/evaluate_dual_lambdarank_qap_diagnostic.py"
        test = code_root / "tests/test_dual_lambdarank_qap_diagnostic.py"
        outgoing = code_root / "models/outgoing_lambdarank.txt"
        incoming = code_root / "models/incoming_lambdarank.txt"
        retrieval_report = code_root / "models/dual_lambdarank_report.json"
        manifest = code_root / "configs/denoise_splits_seed20260710.json"
        quarantine = code_root / "configs/denoise_validation_quarantine_v1.json"
        for path in (script, test, outgoing, incoming, retrieval_report, manifest, quarantine):
            if not path.is_file():
                raise RuntimeError(f"missing bundled file: {path}")

        data_root = one("train/targets").parent.parent
        selected = one("selected_tilenaf_synth_50k.pt")
        embedding = one("hbt_d320_denoised_rgb_sobel.pt")
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{code_root / 'src'}:{code_root / 'scripts'}:{code_root}"
        env["PYTHONHASHSEED"] = "0"

        hardware = {
            "python": sys.version,
            "nvidia_smi": subprocess.run(
                ["nvidia-smi"], text=True, capture_output=True, check=False
            ).stdout,
        }
        try:
            import torch

            hardware.update(
                {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "device_count": torch.cuda.device_count(),
                    "devices": [
                        {
                            "index": index,
                            "name": torch.cuda.get_device_name(index),
                            "capability": list(torch.cuda.get_device_capability(index)),
                        }
                        for index in range(torch.cuda.device_count())
                    ],
                }
            )
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            hardware["tensor_op"] = float((torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).mean())
        except Exception:
            hardware["torch_preflight_error"] = traceback.format_exc()
            raise

        test_result = run(
            [sys.executable, "-m", "pytest", "-q", str(test)], env=env
        )
        if test_result["returncode"] != 0:
            raise RuntimeError("focused tests failed")
        command = [
            sys.executable,
            str(script),
            "--data-root", str(data_root),
            "--denoiser", str(selected),
            "--embedding-checkpoint", str(embedding),
            "--outgoing-model", str(outgoing),
            "--incoming-model", str(incoming),
            "--retrieval-report", str(retrieval_report),
            "--manifest", str(manifest),
            "--quarantine", str(quarantine),
            "--device", "cuda",
            "--denoise-batch-size", "512",
            "--output", str(REPORT),
            "--overwrite",
        ]
        evaluator = run(command, env=env)
        if evaluator["returncode"] != 0 or not REPORT.is_file():
            raise RuntimeError("diagnostic evaluator failed")
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        if report.get("kind") != "frozen_dual_lambdarank_qap_actual_input_development_diagnostic":
            raise RuntimeError("unexpected report kind")
        if report.get("safe_for_submission") is not False or report.get("sealed_paths_opened") is not False:
            raise RuntimeError("diagnostic safety contract drift")
        payload.update(
            {
                "status": "complete",
                "hardware": hardware,
                "bundle": bundle_descriptor,
                "assets": {
                    "selected": {"path": str(selected), "sha256": sha256(selected)},
                    "embedding": {"path": str(embedding), "sha256": sha256(embedding)},
                },
                "test": test_result,
                "evaluator": evaluator,
                "report": {"path": str(REPORT), "sha256": sha256(REPORT)},
                "scientific_status": report["status"],
                "gate": report["gate"],
            }
        )
    except Exception:
        payload.update({"status": "error", "error": traceback.format_exc()})
        raise
    finally:
        payload["seconds"] = time.time() - started
        WRAPPER.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
