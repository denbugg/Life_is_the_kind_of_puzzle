from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import pytest
import torch

from aiijc_puzzle.socket_matcher import (
    BORDER_HEAD_EMBEDDING_V2,
    BORDER_HEAD_SCORE_STATS_V3,
    SocketMatcher,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str) -> ModuleType:
    path = PROJECT_ROOT / relative_path
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _arguments(
    border_head_version: str,
    *,
    checkpoint_in: Path | None = None,
    warmstart_in: Path | None = None,
) -> Namespace:
    return Namespace(
        dimension=8,
        heads=2,
        board_layers=1,
        socket_layers=1,
        sinkhorn_iterations=3,
        synthetic_grid=3,
        raw_rank_weight=0.0,
        border_head_version=border_head_version,
        checkpoint_in=checkpoint_in,
        warmstart_in=warmstart_in,
    )


def test_runner_declares_v2_default_and_loads_v2_to_v3_warmstart(tmp_path: Path) -> None:
    runner = _load_script("run_socket_matcher_border_v3_test", "scripts/run_socket_matcher.py")
    v2_args = _arguments(BORDER_HEAD_EMBEDDING_V2)
    assert runner._socket_contract(v2_args)["architecture"].endswith("-v2")

    v2_model = SocketMatcher(
        dimension=8,
        heads=2,
        board_layers=1,
        socket_layers=1,
        sinkhorn_iterations=3,
    )
    checkpoint = tmp_path / "socket-v2.pt"
    torch.save(
        {
            "contract": runner._socket_contract(v2_args),
            "state_dict": v2_model.state_dict(),
        },
        checkpoint,
    )

    v3_args = _arguments(BORDER_HEAD_SCORE_STATS_V3, warmstart_in=checkpoint)
    model, payload = runner.load_or_create_model(v3_args, torch.device("cpu"))
    assert payload is not None
    assert runner._socket_contract(v3_args)["architecture"].endswith("-v3")
    assert model.border_head_version == BORDER_HEAD_SCORE_STATS_V3
    assert all(
        torch.count_nonzero(head.weight) == 0
        for head in model.border_distribution_heads.values()
    )


def test_exact_synthetic_evaluator_strictly_loads_v3_contract() -> None:
    runner = _load_script("run_socket_matcher_contract_v3_test", "scripts/run_socket_matcher.py")
    evaluator = _load_script(
        "evaluate_socket_matcher_border_v3_test",
        "scripts/evaluate_socket_matcher_synthetic_exact.py",
    )
    args = _arguments(BORDER_HEAD_SCORE_STATS_V3)
    model = SocketMatcher(
        dimension=8,
        heads=2,
        board_layers=1,
        socket_layers=1,
        sinkhorn_iterations=3,
        border_head_version=BORDER_HEAD_SCORE_STATS_V3,
    )
    loaded, contract = evaluator.load_model(
        {
            "contract": runner._socket_contract(args),
            "state_dict": model.state_dict(),
        },
        device=torch.device("cpu"),
    )
    assert contract["architecture"].endswith("-v3")
    assert loaded.border_head_version == BORDER_HEAD_SCORE_STATS_V3


def test_exact_evaluator_reuses_paired_roster_but_rejects_lineage_overlap(
    tmp_path: Path,
) -> None:
    evaluator = _load_script(
        "evaluate_socket_matcher_paired_roster_test",
        "scripts/evaluate_socket_matcher_synthetic_exact.py",
    )
    manifest = {
        "splits": {
            "train": [
                {"filename": "a.png", "target_sha256": "a" * 64},
                {"filename": "b.png", "target_sha256": "b" * 64},
            ]
        }
    }
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"selection": {"source_filenames": ["b.png", "a.png"]}}),
        encoding="utf-8",
    )
    records = evaluator.records_from_selection_report(
        manifest,
        report,
        excluded_filenames=(),
    )
    assert [record["filename"] for record in records] == ["b.png", "a.png"]

    with pytest.raises(ValueError, match="overlaps checkpoint lineage"):
        evaluator.records_from_selection_report(
            manifest,
            report,
            excluded_filenames=("a.png",),
        )
