from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/run_taska_selective_fullres_fusion.py"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    specification = importlib.util.spec_from_file_location(
        "run_taska_selective_fullres_fusion_test", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def test_all_frozen_parent_rosters_are_identity_aligned() -> None:
    runner._require_inputs()
    for spec in runner.PANELS.values():
        rows = runner._aligned_rows(spec)
        assert len(rows) == 32
        assert all(len(record) == 4 for record in rows)


def test_fullres_accepted_logits_are_recovered_in_frozen_order() -> None:
    spec = runner.PANELS["local32"]
    with np.load(spec.fullres_archive, allow_pickle=False) as archive:
        accepted, logits = runner._fullres_accepted_with_logits(archive, "case_0000")
        assert len(accepted) == 17
        assert logits.shape == (17,)
        assert np.all(logits >= 0.0)


def test_one_case_freeze_is_target_free_and_control_replays(tmp_path) -> None:
    spec = replace(runner.PANELS["local32"], name="smoke1", case_count=1)
    output = tmp_path / "fusion"
    output.mkdir()
    result = runner._run_panel(
        spec,
        output_dir=output,
        lookup=None,
        cache=None,
        target_free_only=True,
    )
    assert result["target_free_summary"]["control_replay_match_count"] == 1
    metadata = json.loads((output / "smoke1/frozen-target-free-eval.json").read_text())
    assert metadata["contains_exact_references_or_candidate_labels"] is False
    assert metadata["matcher_rerun"] is False
    assert metadata["standalone_fullres_arm_in_selector"] is False
    runner._validate_freeze(output / "smoke1/pre-score-freeze.json")
