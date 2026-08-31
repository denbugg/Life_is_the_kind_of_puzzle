from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.guided_fourth_emitter import extend_with_guided_emitter
from aiijc_puzzle.synthetic_socket_evaluation import make_exact_synthetic_case
from aiijc_puzzle.tri_emitter_edge_verifier import EMITTERS, TOP_K, build_candidate_pool
from scripts import run_guided_fourth_emitter_fit_capacity as runner


def _ranked_matrix(count: int, offset: int) -> np.ndarray:
    matrix = np.full((count, count), -count, dtype=np.float32)
    for source in range(count):
        order = [
            (source + offset + step) % count
            for step in range(count)
            if (source + offset + step) % count != source
        ]
        matrix[source, order] = -np.arange(len(order), dtype=np.float32)
        matrix[source, source] = -1e4
    return matrix


def test_target_free_fit_case_matches_exact_input_without_returning_reference() -> None:
    clean = np.random.default_rng(4).integers(0, 256, size=(4, 20, 20, 3), dtype=np.uint8)
    case_id, dirty = runner.make_target_free_fit_case(
        clean,
        source_filename="fit.png",
        draw_index=1,
    )
    exact, _reference = make_exact_synthetic_case(
        clean,
        source_filename="fit.png",
        draw_index=1,
        seed=runner.FIT_CASE_SEED,
    )
    assert case_id == exact.case_id
    np.testing.assert_array_equal(dirty, exact.tiles)


def test_sidecar_validator_accepts_fixed_shape_and_rejects_digest_drift() -> None:
    count = 80
    legacy_scores = _ranked_matrix(count, 1)
    legacy = build_candidate_pool(
        {emitter: (legacy_scores, legacy_scores.copy()) for emitter in EMITTERS},
        top_k=TOP_K,
    )
    guided = _ranked_matrix(count, 33)
    pool = extend_with_guided_emitter(legacy, (guided, guided.copy()))
    arrays = runner._sidecar_arrays(pool)
    runner.validate_sidecar_arrays(arrays, count=count)
    arrays["identity_digest_ascii"] = np.frombuffer(("0" * 64).encode(), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="identity digest"):
        runner.validate_sidecar_arrays(arrays, count=count)


def test_signed_capacity_config_has_no_real_or_evaluation_mode() -> None:
    config, digest = runner._load_config(runner.DEFAULT_CONFIG)
    assert len(digest) == 64
    assert config["real_protocol_signed"] is False
    assert config["labels_stage_requires_separate_authorization"] is True
    assert runner.parse_args(["--mode", "freeze-fit"]).mode == "freeze-fit"
    with pytest.raises(SystemExit):
        runner.parse_args(["--mode", "dev"])
