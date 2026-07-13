from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from scripts import build_candidate_graph_oracle_v4_kaggle_bundles as bundles


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/candidate_graph_oracle_ceiling_v4.json"
RUNNER = (
    REPO
    / "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_phase_a_job/run_phase_a.py"
)


def test_v4_bundle_identity_is_fresh_and_private() -> None:
    assert bundles.CONFIG_MEMBER == "configs/candidate_graph_oracle_ceiling_v4.json"
    assert {item.key for item in bundles.DATASETS} == {"code", "input", "runtime"}
    for item in bundles.DATASETS:
        assert item.slug.startswith("pasha883/vsos-candidate-graph-oracle-v4-")
        assert " V4 " in item.title
        assert "candidate_graph_oracle_v4_" in item.archive_name
        assert bundles._dataset_metadata(item)["isPrivate"] is True


def test_v4_bundle_keeps_snapshot_paths_and_runner_import_root_coherent() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    known = config["frozen_contract"]["assets"]["known_code_sha256"]
    prefix = (
        "runs/assembly_v1/kaggle/"
        "candidate_graph_oracle_v4_source_snapshot/src/"
    )
    assert known and all(path.startswith(prefix) for path in known)
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    assignments = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "FROZEN_SRC_RELATIVE":
                assignments[target.id] = ast.literal_eval(node.value.args[0])
    assert assignments["FROZEN_SRC_RELATIVE"] == prefix.rstrip("/")
    source = RUNNER.read_text(encoding="utf-8")
    assert "str(code_root / FROZEN_SRC_RELATIVE)" in source
    assert 'str(code_root / "src")' not in source


def test_pre_reservation_config_cannot_build_any_bundle(tmp_path: Path) -> None:
    output = tmp_path / "bundles"
    with pytest.raises(RuntimeError, match="runtime pin evaluator_sha256"):
        bundles.build_bundles(
            repo_root=REPO,
            config_path=CONFIG,
            lifecycle_ledger_root=tmp_path / "missing-ledger",
            fixture_input_root=tmp_path / "missing-input",
            output_root=output,
        )
    assert not output.exists()
