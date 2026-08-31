from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from aiijc_puzzle.protocol import sha256_file

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import score_joint_native_head_arm_fit_v2 as scorer  # noqa: E402


def test_target_free_layout_proof_replays_all_immutable_bytes() -> None:
    config = {
        "frozen_inputs_gate_copy": {
            "pair_mean_strictly_positive": True,
            "pair_source_bootstrap_95pct_lower_nonnegative": True,
            "exact_mean_nonnegative": True,
            "manhattan_benefit_mean_nonnegative": True,
            "radius2_mean_nonnegative": True,
        },
        "frozen_inputs": {
            "v1_construction_config": {
                "path": "configs/joint_native_head_arm_fit_v1.json",
            }
        },
        "repair_only": {
            "v1_construction_config_sha256": sha256_file(
                ROOT / "configs/joint_native_head_arm_fit_v1.json"
            ),
            "layout_pair_digest": scorer.EXPECTED_LAYOUT_PAIR_DIGEST,
        },
    }
    frozen, head, candidates, controls = scorer._load_and_prove_frozen_layouts(
        config,
        ROOT / "outputs/joint-native-head-arm-fit/fixed-v1",
    )
    assert len(frozen) == len(head) == len(candidates) == len(controls) == 64


def test_manifest_roster_is_ordered_and_hash_bound(tmp_path: Path) -> None:
    rows = [
        {"filename": f"img_{index:06d}.png", "target_sha256": f"sha-{index}"} for index in range(32)
    ]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"splits": {"train": list(reversed(rows))}}))
    config = {
        "frozen_inputs": {"manifest": {"path": str(path), "sha256": sha256_file(path)}},
        "repair_only": {"source_roster": rows},
    }
    observed = scorer._manifest_records(config, path)
    assert [row["filename"] for row in observed] == [row["filename"] for row in rows]
    broken = json.loads(json.dumps(config))
    broken["repair_only"]["source_roster"][0]["target_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="target hash declaration"):
        scorer._manifest_records(broken, path)
