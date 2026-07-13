from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from scripts import push_candidate_graph_oracle_phase_a as launch
from scripts import recover_candidate_graph_oracle_v3_launch_from_raw as recovery


class ReadOnlyApi:
    def dataset_status(self, dataset: str, format: str) -> str:
        assert dataset in launch.EXPECTED_DATASETS.values()
        assert format == "json(status,current_version_number)"
        return json.dumps({"status": "ready", "current_version_number": 2})

    def candidate_graph_oracle_current_readback(self):
        return (
            {
                "id": launch.EXPECTED_KERNEL_ID,
                "ref": launch.EXPECTED_KERNEL_SLUG,
                "title": "VSOS Candidate Graph Oracle V3 Phase A T4x2",
                "slug": launch.EXPECTED_KERNEL_SLUG.split("/", 1)[1],
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu_observation": False,
                "enable_internet": False,
                "enable_tpu_observation": False,
                "dataset_sources": list(launch.EXPECTED_DATASETS.values()),
                "kernel_sources": [],
                "competition_sources": [],
                "model_sources": [],
                "current_version_number": 2,
                "docker_image": "gcr.io/kaggle-images/python@sha256:test",
                "machine_shape_observation": None,
            },
            hashlib.sha256(
                (
                    Path(launch.__file__).resolve().parents[1]
                    / "runs/assembly_v1/kaggle/candidate_graph_oracle_v3_phase_a_job/run_phase_a.py"
                ).read_bytes()
            ).hexdigest(),
        )


def test_exact_raw_alias_recovers_without_any_push_endpoint(tmp_path: Path) -> None:
    job = (
        Path(launch.__file__).resolve().parents[1]
        / "runs/assembly_v1/kaggle/candidate_graph_oracle_v3_phase_a_job"
    )
    source_state = job / "candidate_graph_oracle_v3_launch_state"
    state = tmp_path / "state"
    state.mkdir()
    for name in (launch.INTENT_NAME, launch.RAW_RESPONSE_NAME):
        shutil.copy2(source_state / name, state / name)
    raw_before = hashlib.sha256((state / launch.RAW_RESPONSE_NAME).read_bytes()).hexdigest()

    receipt = tmp_path / "receipt.json"
    result = recovery.recover(
        job_dir=job,
        state_dir=state,
        receipt_path=receipt,
        api=ReadOnlyApi(),
    )
    assert result["remote_write_performed"] is False
    assert raw_before == recovery.EXPECTED_RAW_SHA256
    assert hashlib.sha256((state / launch.RAW_RESPONSE_NAME).read_bytes()).hexdigest() == raw_before
    normalization = json.loads((state / recovery.NORMALIZATION_NAME).read_text())
    assert normalization["before"]["ref"].startswith("/code/")
    assert normalization["after"]["ref"] == launch.EXPECTED_KERNEL_SLUG
    derived = json.loads((state / launch.RESPONSE_NAME).read_text())
    assert derived["kind"] == recovery.DERIVED_RESPONSE_KIND
    assert derived["derived_from_raw"] is True
    assert derived["raw_response_sha256"] == raw_before
    envelope = json.loads(receipt.read_text())
    assert envelope["payload"]["response_provenance"] == "derived_from_immutable_raw_sdk_response"
    assert envelope["payload"]["remote_write_performed_by_recovery"] is False


def test_recovery_cli_is_reproducible_in_module_mode() -> None:
    root = Path(launch.__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.recover_candidate_graph_oracle_v3_launch_from_raw",
            "--help",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Derive a v3 launch receipt" in result.stdout
