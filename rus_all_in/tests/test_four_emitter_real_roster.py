from __future__ import annotations

import json
from pathlib import Path

from aiijc_puzzle.four_emitter_real_roster import (
    deterministic_fresh_roster,
    minimal_inventory_blocker,
    recursive_json_roster_inventory,
    singular_source_filenames,
    source_union_from_payload,
)


def test_recursive_source_semantics_include_panels_and_rows_not_generic_names() -> None:
    payload = {
        "fit_filenames": ["img_000001.png", "img_000002.png"],
        "rows": [{"source_filename": "img_000003.png"}],
        "generic": {"filenames": ["img_000004.png"]},
    }
    declared, rows, union = source_union_from_payload(payload)
    assert declared == {"img_000001.png", "img_000002.png"}
    assert rows == {"img_000003.png"}
    assert union == declared | rows
    assert singular_source_filenames(payload) == rows


def test_recursive_inventory_is_deterministic_and_content_addressed(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / "configs/a.json").write_text(
        json.dumps({"source_filenames": ["img_000010.png"]}), encoding="utf-8"
    )
    (tmp_path / "outputs/b.json").write_text(
        json.dumps({"rows": [{"source_filename": "img_000011.png"}]}),
        encoding="utf-8",
    )
    first = recursive_json_roster_inventory(tmp_path)
    second = recursive_json_roster_inventory(tmp_path)
    assert first == second
    artifacts, names, inventory_digest = first
    assert len(artifacts) == 2
    assert names == ("img_000010.png", "img_000011.png")
    assert len(inventory_digest) == 64


def test_fresh_roster_or_inventory_blocker_is_fail_closed() -> None:
    train = tuple(f"img_{index:06d}.png" for index in range(5))
    roster, eligible = deterministic_fresh_roster(
        train,
        train,
        count=2,
        namespace="test",
        seed=7,
    )
    assert roster == () and eligible == ()
    blocker = minimal_inventory_blocker(train, train[:3], train[3:])
    assert blocker["previously_excluded_train_count"] == 3
    assert blocker["final_complement_scored_source_count"] == 2
    assert blocker["prior_final_overlap_count"] == 0
    assert blocker["remaining_fresh_train_count"] == 0
    assert blocker["complete_inventory_blocker"] is True
