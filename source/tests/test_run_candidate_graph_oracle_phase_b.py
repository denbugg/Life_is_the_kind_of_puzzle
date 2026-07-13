from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_candidate_graph_oracle_phase_b.py"
SPEC = importlib.util.spec_from_file_location("phase_b_runner_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _args(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    phase_a = tmp_path / "phase-a"
    bundle = tmp_path / "bundle"
    lifecycle = tmp_path / "lifecycle"
    for path in (phase_a, bundle, lifecycle):
        path.mkdir()
    return runner.parse_args(
        [
            "--config",
            str(runner.REPO_ROOT / "configs/candidate_graph_oracle_ceiling_v3.json"),
            "--config-sha256",
            "a" * 64,
            "--phase-a-dir",
            str(phase_a),
            "--phase-a-envelope-sha256",
            "b" * 64,
            "--fixture-manifest",
            str(bundle / "fixture_input/fixture_input_manifest.json"),
            "--fixture-manifest-sha256",
            "c" * 64,
            "--fixture-root",
            str(bundle / "fixture_input"),
            "--fixture-bundle-root",
            str(bundle),
            "--lifecycle-ledger",
            str(lifecycle),
            "--output",
            str(tmp_path / "output"),
        ]
    )


def test_cli_has_one_opaque_bundle_and_rejects_legacy_label_arguments(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    assert args.fixture_bundle_root.endswith("/bundle")
    assert not any(name.startswith("labels") for name in vars(args))
    with pytest.raises(SystemExit):
        runner.parse_args([*runner._reexec_arguments(args), "--labels-root", "/forbidden"])


def test_profile_is_default_deny_and_never_constructs_label_paths(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    profile = runner.build_sandbox_profile(args)
    canonical_bundle = runner._profile_canonical(Path(args.fixture_bundle_root))
    canonical_phase_a = runner._profile_canonical(Path(args.phase_a_dir))
    write_line = next(
        line for line in profile.splitlines() if line.startswith("(allow file-write*")
    )
    assert "(deny default)" in profile
    assert "(deny network*)" in profile
    assert f'(subpath "{canonical_bundle}")' in profile
    assert f"{canonical_bundle}/fixture_label" not in profile
    assert "fixture_label_manifest.json" not in profile
    assert "puzzle/train" not in profile
    assert str(canonical_phase_a) not in write_line
    assert f'(subpath "{runner._profile_canonical(Path(args.output))}")' in profile
    assert f'(literal "{runner.REPO_ROOT}")' in profile
    assert f'(subpath "{runner.EXPECTED_PYTHON.parent.parent}")' in profile


def test_dynamic_roots_cannot_whitelist_forbidden_or_overlapping_paths(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.config = str(runner.REPO_ROOT / "puzzle/train/targets/img_003812.png")
    with pytest.raises(RuntimeError, match="frozen repository config"):
        runner.build_sandbox_profile(args)

    args = _args(tmp_path / "second")
    args.phase_a_dir = str(runner.REPO_ROOT / "puzzle/train")
    with pytest.raises(RuntimeError, match="forbidden puzzle data"):
        runner.build_sandbox_profile(args)

    args = _args(tmp_path / "third")
    args.phase_a_dir = args.fixture_bundle_root
    with pytest.raises(RuntimeError, match="roots overlap"):
        runner.build_sandbox_profile(args)


def test_static_code_allowlist_matches_current_frozen_closure() -> None:
    config = json.loads(
        (runner.REPO_ROOT / "configs/candidate_graph_oracle_ceiling_v3.json").read_text()
    )
    assert runner.KNOWN_CODE_ALLOWLIST == frozenset(
        config["frozen_contract"]["assets"]["known_code_sha256"]
    )


def test_runner_binds_safe_input_phase_a_lifecycle_and_output_roots(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    bindings = runner._filesystem_bindings(args, Path(args.output))
    assert bindings == {
        "phase_a_root": str(Path(args.phase_a_dir).absolute()),
        "phase_a_artifact_envelope_sha256": "b" * 64,
        "fixture_bundle_root": str(Path(args.fixture_bundle_root).absolute()),
        "fixture_input_root": str(Path(args.fixture_root).absolute()),
        "fixture_input_manifest": str(Path(args.fixture_manifest).absolute()),
        "fixture_input_manifest_sha256": "c" * 64,
        "lifecycle_ledger_root": str(Path(args.lifecycle_ledger).absolute()),
        "output_root": str(Path(args.output).absolute()),
    }
    assert not any("label" in key for key in bindings)


def test_self_reexec_uses_sandbox_exec_before_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    captured: dict[str, object] = {}

    class ReexecObserved(Exception):
        pass

    def fake_execve(executable: str, command: list[str], environment: dict[str, str]):
        captured.update(executable=executable, command=command, environment=environment)
        raise ReexecObserved

    def forbidden_parent_read(path: Path) -> bytes:
        raise AssertionError(f"unsandboxed parent opened {path}")

    monkeypatch.setattr(runner, "_assert_expected_python", lambda: None)
    monkeypatch.setattr(runner, "_create_fresh_output_root", lambda path: None)
    monkeypatch.setattr(runner, "_read_regular_file", forbidden_parent_read)
    monkeypatch.setattr(runner.os, "execve", fake_execve)
    with pytest.raises(ReexecObserved):
        runner._self_reexec_in_sandbox(args)

    profile = runner.build_sandbox_profile(args)
    command = captured["command"]
    environment = captured["environment"]
    assert isinstance(command, list) and command[:2] == [str(runner.SANDBOX_EXEC), "-p"]
    assert command[2:5] == [profile, str(runner.EXPECTED_PYTHON), str(SCRIPT)]
    assert "--fixture-bundle-root" in command
    assert not any(str(value).startswith("--labels") for value in command)
    assert isinstance(environment, dict)
    assert environment[runner.SANDBOXED_ENV] == "1"
    assert environment[runner.SANDBOX_PROFILE_TEXT_ENV] == profile
    assert environment[runner.SANDBOX_PROFILE_SHA_ENV] == runner._sha256_bytes(
        profile.encode("utf-8")
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS-only")
def test_real_profile_enforces_read_write_and_network_boundaries(tmp_path: Path) -> None:
    args = _args(tmp_path)
    Path(args.output).mkdir()
    profile = runner.build_sandbox_profile(args)
    probe = r'''
import errno, os, platform, socket, sys
import cv2, kornia, numpy, scipy, skimage, torch
from PIL import __version__ as pillow_version
import puzzle_assembly.panels
import puzzle_denoise_v2.inference
config, forbidden, phase_a, output, expected_platform = sys.argv[1:]
assert platform.platform() == expected_platform
with open(config, "rb") as handle:
    assert handle.read(1) == b"{"
for label, operation in (
    ("forbidden_read", lambda: os.open(forbidden, os.O_RDONLY | os.O_DIRECTORY)),
    ("phase_a_write", lambda: os.open(os.path.join(phase_a, ".probe"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)),
):
    try:
        descriptor = operation()
    except OSError as error:
        assert error.errno in (errno.EACCES, errno.EPERM), (label, error)
    else:
        os.close(descriptor)
        raise AssertionError(label + " unexpectedly permitted")
handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    try:
        handle.connect(("127.0.0.1", 9))
    except OSError as error:
        assert error.errno in (errno.EACCES, errno.EPERM), error
    else:
        raise AssertionError("network unexpectedly permitted")
finally:
    handle.close()
probe_path = os.path.join(output, ".write_probe")
descriptor = os.open(probe_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
os.close(descriptor)
os.unlink(probe_path)
print("sandbox-boundaries-ok")
'''
    environment_lock = json.loads(
        (
            runner.REPO_ROOT
            / "configs/candidate_graph_oracle_environment_lock_v1.json"
        ).read_text()
    )
    completed = subprocess.run(
        [
            str(runner.SANDBOX_EXEC),
            "-p",
            profile,
            str(runner.EXPECTED_PYTHON),
            "-c",
            probe,
            str(runner.REPO_ROOT / "configs/candidate_graph_oracle_ceiling_v3.json"),
            str(runner.REPO_ROOT / "puzzle/train/targets"),
            args.phase_a_dir,
            args.output,
            environment_lock["fixture_preparation_and_phase_b"]["platform"],
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout
    assert completed.stdout.strip() == "sandbox-boundaries-ok"


def test_report_payload_hash_is_verified(tmp_path: Path) -> None:
    payload = {"kind": "candidate_graph_oracle_ceiling_report", "status": "stop"}
    envelope = {
        "payload": payload,
        "payload_sha256": runner._sha256_bytes(runner._canonical_object_bytes(payload)),
    }
    path = tmp_path / runner.REPORT_NAME
    path.write_bytes(runner._canonical_line(envelope))
    actual, raw, digest = runner._verified_report_envelope(path)
    assert actual == payload
    assert raw == runner._canonical_line(envelope)
    assert digest == envelope["payload_sha256"]

    envelope["payload_sha256"] = "0" * 64
    path.write_bytes(runner._canonical_line(envelope))
    with pytest.raises(RuntimeError, match="payload_sha256 mismatch"):
        runner._verified_report_envelope(path)
