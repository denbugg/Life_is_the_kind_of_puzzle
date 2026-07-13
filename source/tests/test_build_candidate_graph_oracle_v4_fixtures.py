from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from scripts import build_candidate_graph_oracle_v4_fixtures as fixtures


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/candidate_graph_oracle_ceiling_v4.json"


def test_v4_wrapper_is_the_pinned_builder_and_binds_historical_source() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["runtime_pins"]["fixture_builder_path"] == (
        "scripts/build_candidate_graph_oracle_v4_fixtures.py"
    )
    assert fixtures.EXPECTED_INSTANCE == config["protocol_instance_id"]
    assert fixtures.HISTORICAL_BUILDER_SHA256 == hashlib.sha256(
        fixtures.HISTORICAL_BUILDER.read_bytes()
    ).hexdigest()
    frozen = config["frozen_contract"]["assets"]
    assert frozen["known_code_import_root"] == fixtures.SNAPSHOT_RELATIVE.as_posix()
    assert frozen["mutable_repository_src_imports_forbidden"] is True


def test_v4_wrapper_loads_project_modules_only_from_frozen_snapshot() -> None:
    source = """
import json
from scripts import build_candidate_graph_oracle_v4_fixtures as wrapper
config = wrapper._load_config(wrapper.CONFIG_PATH)
wrapper._load_historical_builder(config)
origins = {}
for name, module in sorted(__import__('sys').modules.items()):
    if wrapper._is_project_module(name) and getattr(module, '__spec__', None):
        origin = getattr(module.__spec__, 'origin', None)
        if isinstance(origin, str):
            origins[name] = origin
print(json.dumps(origins, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    origins = json.loads(completed.stdout)
    assert origins
    snapshot = str(fixtures.SNAPSHOT_ROOT.resolve()) + "/"
    assert all(str(Path(value).resolve()).startswith(snapshot) for value in origins.values())


def test_pre_code_pin_wrapper_refuses_before_any_fixture_or_ledger_mutation(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    input_root = bundle / "fixture_input"
    label_root = bundle / "fixture_label"
    lock_path = bundle / "fixture_control/fixture_lock.json"
    marker_path = bundle / "fixture_control/FIXTURE_PIXEL_ACCESS_STARTED.json"
    ledger = tmp_path / "ledger"
    completed = subprocess.run(
        [
            sys.executable,
            str(fixtures.__file__),
            "--config",
            str(CONFIG),
            "--data-root",
            str(REPO / "puzzle"),
            "--input-root",
            str(input_root),
            "--label-root",
            str(label_root),
            "--lock-path",
            str(lock_path),
            "--prep-marker-path",
            str(marker_path),
            "--lifecycle-ledger-root",
            str(ledger),
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode != 0
    assert "runtime pin must be set before pixel access" in completed.stderr
    assert not bundle.exists()
    assert not ledger.exists()


def test_wrapper_rejects_any_non_v4_config_before_builder_execution(
    tmp_path: Path,
) -> None:
    wrong = tmp_path / "config.json"
    wrong.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    args = [
        "--config",
        str(wrong),
        "--input-root",
        str(tmp_path / "input"),
        "--label-root",
        str(tmp_path / "label"),
        "--lock-path",
        str(tmp_path / "control/fixture_lock.json"),
        "--lifecycle-ledger-root",
        str(tmp_path / "ledger"),
    ]
    completed = subprocess.run(
        [sys.executable, str(fixtures.__file__), *args],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode != 0
    assert "accepts only the v4 protocol config" in completed.stderr
    assert not (tmp_path / "input").exists()
