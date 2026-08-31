from __future__ import annotations

from pathlib import Path

import torch

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.synthetic_socket_evaluation import names_digest
from scripts.run_raw_twin_union_reranker_v2 import (
    DEFAULT_ADDITIONAL_COMMITMENT,
    DEFAULT_CONFIG,
    DEFAULT_REQUIRED_COMMITMENT,
    _fit_eval_commitment_names,
    _load_config,
    _required_exclusion_names,
    _synthetic_capacity_board,
)


def test_amended_preregistration_and_sidecar_are_frozen() -> None:
    config = _load_config(DEFAULT_CONFIG)
    assert config["candidate_roster"]["frozen_raw_hard_projection_edges_per_axis"] == 552
    assert config["candidate_roster"]["restricted_ot_equivalence_to_full_raw_not_claimed"]
    assert config["inference"]["baseline"].startswith("untouched full frozen raw d64")
    sidecar = Path(f"{DEFAULT_CONFIG}.sha256").read_text(encoding="utf-8").split()[0]
    assert sidecar == sha256_file(DEFAULT_CONFIG)


def test_required_fresh64_commitment_is_digest_valid() -> None:
    names, payload = _required_exclusion_names(DEFAULT_REQUIRED_COMMITMENT)
    assert len(names) == 64
    assert names_digest(names) == payload["selection"]["source_order_digest"]
    assert sha256_file(DEFAULT_REQUIRED_COMMITMENT).startswith("6056fcc5")


def test_component_placer_fit_and_eval_commitment_is_explicitly_excluded() -> None:
    names, payload = _fit_eval_commitment_names(DEFAULT_ADDITIONAL_COMMITMENT)
    assert len(payload["fit_filenames"]) == 256
    assert len(payload["evaluation_filenames"]) == 32
    assert len(names) == 288
    assert sha256_file(DEFAULT_ADDITIONAL_COMMITMENT).startswith("2b2b0c90")


def test_procedural_capacity_board_has_strict_layout_and_true_union_edges() -> None:
    board, layout = _synthetic_capacity_board(81, device=torch.device("cpu"))
    assert board.grid == 4
    assert board.values.shape[1] == 280
    assert torch.equal(layout.sort().values, torch.arange(16))
    assert all(len(row) == 15 for axis in board.rows for row in axis)
