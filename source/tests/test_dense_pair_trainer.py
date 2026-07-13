from __future__ import annotations

import importlib.util
import hashlib
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from puzzle_assembly.compatibility import CompatibilityMatrices
from puzzle_assembly.dense_pair_residual import DensePairResidualScorer
from puzzle_assembly.geometry import GRID, TILE_COUNT


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_evaluate_dense_pair_residual.py"
_SPEC = importlib.util.spec_from_file_location("dense_pair_trainer", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
trainer = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = trainer
_SPEC.loader.exec_module(trainer)


def _identity_slot_to_target() -> np.ndarray:
    return np.arange(TILE_COUNT, dtype=np.int32)


def _base() -> CompatibilityMatrices:
    rng = np.random.default_rng(7)
    right = rng.random((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    down = rng.random((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    return CompatibilityMatrices("base", right, down)


def _tiny_model() -> DensePairResidualScorer:
    return DensePairResidualScorer(
        tile_size=2,
        encoder_width=4,
        encoder_depth=1,
        expansion=1,
        side_band=1,
        profile_bins=1,
        embedding_dim=4,
        relation_hidden=4,
        pair_hidden=2,
        dropout=0.0,
        max_residual=0.1,
    )


def test_defaults_precommit_fresh_selection_and_holdout_slices(tmp_path: Path) -> None:
    args = trainer.parse_args(["--output-dir", str(tmp_path)])
    assert args.train_offset == 4096
    assert args.train_sources == 256
    assert args.selection_offset == 96
    assert args.selection_sources == 32
    assert args.holdout_offset == 112
    assert args.holdout_sources == 16
    assert args.real_gate_offset == 128
    assert args.real_gate_sources == 64
    assert args.final_audit_offset == 0
    assert args.final_audit_sources == 64
    assert args.confirmation_offset == 64
    assert args.confirmation_sources == 64
    assert args.quick_sources == 32
    assert args.queries_per_source == 48
    assert args.panels == "primary_kornia,independent_libjpeg"


def test_default_source_slices_match_precommitted_hashes(tmp_path: Path) -> None:
    args = trainer.parse_args(["--output-dir", str(tmp_path)])
    split = lambda name: trainer.source_names_for_split(
        name, manifest_path=args.manifest, quarantine_path=args.quarantine
    )
    cases = {
        "train": split("edge_train")[4096:4352],
        "selection": split("edge_development")[96:128],
        "holdout": split("assembly_cal")[112:128],
        "real_gate": split("assembly_incremental_gate")[128:192],
        "final_audit": split("assembly_final_audit")[0:64],
        "confirmation": split("assembly_final_audit")[64:128],
    }
    expected = {
        "train": "9b7369879c85ae999028287901b8dba7063f70bebab4beeda7057cffd6e15920",
        "selection": "a20a4f638af4c28b807d0d194a6be69cee6b4d8bee7847eefee70fb1817c02de",
        "holdout": "59a10b924dca9ff829a3038b5ba06c15fa93e3da0938456528a934dabd49e8ea",
        "real_gate": "e5fb7fc6b3d24e9c080b4f33224b863c181e72452de4e54e602a80a321c13251",
        "final_audit": "c281ef844bed9fbfd452b896c2961164ded794fca65d010f1b4adea7f58bff33",
        "confirmation": "9aa995c9bc628bd17ae2d55d0ef8852d5b5e19ef241f8834c8d1b8bd0358429a",
    }
    for label, names in cases.items():
        actual = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
        assert actual == expected[label]
    exposed = set(split("assembly_audit_exposed"))
    assert len(exposed) == 32
    assert not exposed.intersection(cases["final_audit"])
    assert not exposed.intersection(cases["confirmation"])
    assert trainer._sha256(args.audit_exclusion) == (
        "772e89ad4f633d2050f8ad3806cd24bffed132bcd8914951b7b8edff3f608ab6"
    )


def test_trainer_model_adapter_matches_separately_owned_api(tmp_path: Path) -> None:
    args = trainer.parse_args(["--output-dir", str(tmp_path)])
    model = trainer._model(args)
    assert isinstance(model, DensePairResidualScorer)
    assert model.encoder_width == args.channels
    assert model.embedding_dim == args.embedding_dim
    assert model.relation_hidden == args.pair_hidden_dim
    assert model.max_residual == pytest.approx(args.bounded_gain)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 2_000_000 <= parameters <= 4_000_000


def test_dense_targets_have_552_valid_queries_per_direction() -> None:
    right, down = trainer.dense_targets(_identity_slot_to_target())
    assert int(np.sum(right >= 0)) == GRID * (GRID - 1)
    assert int(np.sum(down >= 0)) == GRID * (GRID - 1)
    assert right[0] == 1
    assert right[GRID - 1] == -1
    assert down[0] == GRID
    assert down[-1] == -1


def test_query_sampling_is_balanced_valid_and_deterministic() -> None:
    first = trainer.sample_queries(
        _identity_slot_to_target(), total=48, rng=np.random.default_rng(19)
    )
    second = trainer.sample_queries(
        _identity_slot_to_target(), total=48, rng=np.random.default_rng(19)
    )
    for direction in (trainer.RIGHT, trainer.DOWN):
        np.testing.assert_array_equal(first[direction], second[direction])
        assert first[direction].shape == (24,)
        targets = trainer.dense_targets(_identity_slot_to_target())[direction]
        assert np.all(targets[first[direction]] >= 0)


def test_score_query_rows_includes_every_slot_without_topk() -> None:
    torch.manual_seed(23)
    model = _tiny_model().eval()
    raw = torch.rand(TILE_COUNT, 3, 2, 2)
    with torch.inference_mode():
        bank = model.encode_tiles(raw, None)
        queries = torch.as_tensor([0, 17, 575])
        rows = trainer._score_query_rows(
            model, bank, queries, trainer.RIGHT, pair_chunk_size=257
        )
        expected = model.score_dense(bank, trainer.RIGHT, chunk_size=113)[queries]
    assert rows.shape == (3, TILE_COUNT)
    torch.testing.assert_close(rows, expected, atol=1e-7, rtol=1e-6)
    # The trainer scores 576 columns then masks self in the base; there are
    # exactly 575 valid alternatives, never an HBT top-k proposal ceiling.
    assert rows.shape[1] - 1 == 575


def test_incoming_column_scores_include_every_possible_predecessor() -> None:
    torch.manual_seed(24)
    model = _tiny_model().eval()
    raw = torch.rand(TILE_COUNT, 3, 2, 2)
    with torch.inference_mode():
        bank = model.encode_tiles(raw, None)
        seconds = torch.as_tensor([1, 29, 574])
        columns = trainer._score_incoming_columns(
            model, bank, seconds, trainer.DOWN, pair_chunk_size=251
        )
        dense = model.score_dense(bank, trainer.DOWN, chunk_size=101)
    assert columns.shape == (3, TILE_COUNT)
    torch.testing.assert_close(columns, dense[:, seconds].T, atol=1e-7, rtol=1e-6)
    assert columns.shape[1] - 1 == 575


def test_rank_cost_is_scale_invariant_and_masks_self() -> None:
    rng = np.random.default_rng(29)
    values = rng.normal(size=(TILE_COUNT, TILE_COUNT)).astype(np.float32)
    np.fill_diagonal(values, np.inf)
    ranked = trainer._rank_cost(values)
    shifted = trainer._rank_cost(7.0 * values + 31.0)
    off_diagonal = ~np.eye(TILE_COUNT, dtype=bool)
    # Float32 affine rounding can exchange an exact tie by one rank.
    np.testing.assert_allclose(
        ranked[off_diagonal], shifted[off_diagonal], atol=1.1 / (TILE_COUNT - 1), rtol=0.0
    )
    assert np.isposinf(np.diag(ranked)).all()
    assert float(ranked[off_diagonal].min()) >= 0.0
    assert float(ranked[off_diagonal].max()) <= 1.0


def test_zero_initialized_model_leaves_frozen_base_exactly_unchanged() -> None:
    torch.manual_seed(31)
    model = _tiny_model().eval()
    base = _base()
    raw = np.random.default_rng(37).integers(
        0, 256, size=(TILE_COUNT, 2, 2, 3), dtype=np.uint8
    )
    api = trainer._dense_api()
    result = api.dense_pair_residual_compatibility(
        model, raw, base, device="cpu", chunk_size=128
    )
    off_diagonal = ~np.eye(TILE_COUNT, dtype=bool)
    np.testing.assert_array_equal(result.right[off_diagonal], base.right[off_diagonal])
    np.testing.assert_array_equal(result.down[off_diagonal], base.down[off_diagonal])


def test_one_dense_all_negative_training_step_is_finite(tmp_path: Path) -> None:
    args = trainer.parse_args(["--output-dir", str(tmp_path), "--smoke", "--no-amp"])
    args.affine_probability = 0.0
    args.extra_noise_probability = 0.0
    args.blur_probability = 0.0
    args.quantize_probability = 0.0
    args.view_dropout = 0.0
    model = _tiny_model().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = trainer._scheduler(optimizer, total_steps=1, warmup_fraction=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    runtime = trainer.Runtime(0, 1, 0, torch.device("cpu"))
    raw = np.random.default_rng(43).integers(
        0, 256, size=(TILE_COUNT, 2, 2, 3), dtype=np.uint8
    )
    source = trainer.PreparedSource(
        name="synthetic.png",
        panel="primary_kornia",
        replica=0,
        seed=47,
        raw=raw,
        denoised=raw.copy(),
            clean=np.zeros((480, 480, 3), dtype=np.uint8),
            slot_to_target=_identity_slot_to_target(),
            seed_score=_base(),
            base=_base(),
    )
    final = model.relation_head[-1]
    assert isinstance(final, torch.nn.Linear)
    before = final.weight.detach().clone()
    metrics = trainer._train_source_step(
        model,
        source,
        args=args,
        runtime=runtime,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        generator=torch.Generator().manual_seed(53),
        rng=np.random.default_rng(59),
    )
    assert np.isfinite(metrics["loss"])
    assert metrics["pairs"] == 2 * args.queries_per_source * 575
    assert metrics["skipped"] == 0.0
    assert not torch.equal(before, final.weight.detach())


def test_retrieval_gate_is_strict_and_panel_aware() -> None:
    passing = {
        "mean_delta_recall_at_1": 0.02,
        "mean_delta_mrr": 0.011,
        "mean_delta_recall_at_32": -0.001,
        "bootstrap_95_delta_recall_at_1": [0.001, 0.03],
        "panels": {
            "primary_kornia": {"mean_delta_recall_at_1": 0.01},
            "independent_libjpeg": {"mean_delta_recall_at_1": 0.005},
        },
    }
    assert trainer.retrieval_gate(passing)["passed"] is True
    failing = {**passing, "mean_delta_recall_at_1": 0.009}
    assert trainer.retrieval_gate(failing)["passed"] is False
    panel_fail = {
        **passing,
        "panels": {
            **passing["panels"],
            "independent_libjpeg": {"mean_delta_recall_at_1": -1e-4},
        },
    }
    assert trainer.retrieval_gate(panel_fail)["passed"] is False


def test_qap_gate_requires_material_ssim_and_adjacency_gain() -> None:
    passing = {
        "mean_delta_ssim": 0.006,
        "mean_delta_adjacency": 0.012,
        "bootstrap_95_delta_ssim": [0.0002, 0.010],
        "panels": {
            "primary_kornia": {"mean_delta_ssim": 0.006},
            "independent_libjpeg": {"mean_delta_ssim": 0.004},
        },
    }
    assert trainer.qap_gate(passing)["passed"] is True
    assert trainer.qap_gate({**passing, "mean_delta_ssim": 0.0049})["passed"] is False


def test_validate_args_rejects_reused_validation_prefixes(tmp_path: Path) -> None:
    runtime = trainer.Runtime(rank=0, world_size=1, local_rank=0, device=torch.device("cpu"))
    args = trainer.parse_args(["--output-dir", str(tmp_path), "--smoke"])
    args.selection_offset = 95
    with pytest.raises(ValueError, match="reused"):
        trainer._validate_args(args, runtime)
    args.selection_offset = 96
    args.holdout_offset = 111
    with pytest.raises(ValueError, match="assembly_cal"):
        trainer._validate_args(args, runtime)
    args.holdout_offset = 112
    args.real_gate_offset = 127
    with pytest.raises(ValueError, match="incremental_gate"):
        trainer._validate_args(args, runtime)
    args.real_gate_offset = 128
    args.final_audit_offset = -1
    with pytest.raises(ValueError, match="audit"):
        trainer._validate_args(args, runtime)
    args.final_audit_offset = 0
    args.confirmation_offset = 63
    with pytest.raises(ValueError, match="confirmation"):
        trainer._validate_args(args, runtime)


def test_scientific_protocol_hash_binds_panels_seed_and_qap(tmp_path: Path) -> None:
    args = trainer.parse_args(["--output-dir", str(tmp_path)])
    baseline = trainer._scientific_protocol(args)["sha256"]
    args.seed += 1
    assert trainer._scientific_protocol(args)["sha256"] != baseline
    args.seed -= 1
    args.panels = "primary_kornia"
    assert trainer._scientific_protocol(args)["sha256"] != baseline
    args.panels = "primary_kornia,independent_libjpeg"
    args.qap_iterations += 1
    assert trainer._scientific_protocol(args)["sha256"] != baseline
    args.qap_iterations -= 1
    for name, value in (
        ("no_amp", True),
        ("pair_chunk_size", args.pair_chunk_size // 2),
        ("denoise_batch_size", args.denoise_batch_size // 2),
        ("classical_chunk_size", args.classical_chunk_size // 2),
        ("max_amp_skips", args.max_amp_skips + 1),
        ("quick_sources", args.quick_sources // 2),
    ):
        changed = trainer.parse_args(["--output-dir", str(tmp_path)])
        setattr(changed, name, value)
        assert trainer._scientific_protocol(changed)["sha256"] != baseline


def test_real_input_gate_requires_material_paired_gain() -> None:
    records = [
        {
            "name": f"{index}.png",
            "base": {"ssim": 0.20},
            "candidate": {"ssim": 0.21},
            "delta_ssim": 0.01,
        }
        for index in range(16)
    ]
    aggregate = trainer._real_gate_aggregate(records, seed=71)
    assert trainer.real_input_gate(aggregate)["passed"] is True
    weak = [
        {**record, "candidate": {"ssim": 0.202}, "delta_ssim": 0.002}
        for record in records
    ]
    assert trainer.real_input_gate(
        trainer._real_gate_aggregate(weak, seed=71)
    )["passed"] is False


def test_frozen_manifest_rejects_path_hash_layout_and_checkpoint_tampering() -> None:
    payload = {
        "candidate_checkpoint_sha256": "checkpoint",
        "records": [
            {
                "name": "img.png",
                "candidate_render": "/frozen/img.candidate.png",
                "candidate_render_sha256": "render",
                "candidate_layout": list(range(TILE_COUNT)),
            }
        ],
    }
    expected = {
        "payload": payload,
        "payload_sha256": trainer._canonical_json_sha256(payload),
    }
    assert trainer._require_exact_frozen_envelope(copy.deepcopy(expected), expected) == payload
    mutations = []
    for key, value in (
        ("candidate_checkpoint_sha256", "changed"),
    ):
        changed = copy.deepcopy(expected)
        changed["payload"][key] = value
        mutations.append(changed)
    for key, value in (
        ("candidate_render", "/tmp/target.png"),
        ("candidate_render_sha256", "changed"),
        ("candidate_layout", list(reversed(range(TILE_COUNT)))),
    ):
        changed = copy.deepcopy(expected)
        changed["payload"]["records"][0][key] = value
        mutations.append(changed)
    for changed in mutations:
        with pytest.raises(RuntimeError, match="manifest"):
            trainer._require_exact_frozen_envelope(changed, expected)


def test_frozen_png_hashes_and_decodes_the_same_bytes(tmp_path: Path) -> None:
    path = tmp_path / "frozen.png"
    values = np.zeros((480, 480, 3), dtype=np.uint8)
    values[0, 0] = (1, 2, 3)
    trainer._atomic_png(path, values)
    digest = trainer._sha256(path)
    np.testing.assert_array_equal(trainer._read_rgb_hashed(path, digest), values)
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        trainer._read_rgb_hashed(path, digest)


def test_epoch0_qap_uses_one_hbt_seed_and_promoted_filename_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial = np.arange(TILE_COUNT, dtype=np.int32)
    calls: list[tuple[np.ndarray, int]] = []
    monkeypatch.setattr(trainer, "_component_seed", lambda _score: initial.copy())

    def fake_qap(_score, *, initial: np.ndarray, seed: int, args: object) -> np.ndarray:
        del args
        calls.append((initial.copy(), seed))
        return initial.copy()

    monkeypatch.setattr(trainer, "_qap_layout", fake_qap)
    tiles = np.zeros((TILE_COUNT, 20, 20, 3), dtype=np.uint8)
    source = trainer.PreparedSource(
        name="img_000123.png",
        panel="primary_kornia",
        replica=0,
        seed=999999,
        raw=tiles,
        denoised=tiles,
        clean=np.zeros((480, 480, 3), dtype=np.uint8),
        slot_to_target=initial,
        seed_score=_base(),
        base=_base(),
    )
    args = trainer.parse_args(["--output-dir", str(tmp_path), "--smoke"])
    result = trainer._evaluate_qap_source(source, source.base, args=args)
    expected_seed = int.from_bytes(
        hashlib.sha256(source.name.encode("utf-8")).digest()[:4], "little"
    ) + 7001
    assert len(calls) == 2
    for seen_initial, seed in calls:
        np.testing.assert_array_equal(seen_initial, initial)
        assert seed == expected_seed
    assert result["delta"]["ssim"] == pytest.approx(0.0)
    assert result["delta"]["adjacency"] == pytest.approx(0.0)


def test_checkpoint_provenance_rejects_changed_frozen_assets() -> None:
    active = {
        "kind": "dense_all_pairs_residual_pilot",
        "base_contract": "frozen",
        "train_names_sha256": "train",
        "selection_names_sha256": "selection",
        "holdout_names_sha256": "holdout",
        "manifest": {"sha256": "manifest"},
        "quarantine": {"sha256": "quarantine"},
        "hbt": {"checkpoint_sha256": "hbt"},
        "denoiser": {"checkpoint_sha256": "denoiser"},
    }
    trainer._validate_checkpoint_provenance(dict(active), active)
    changed = {**active, "hbt": {"checkpoint_sha256": "changed"}}
    with pytest.raises(RuntimeError, match="hbt.checkpoint_sha256"):
        trainer._validate_checkpoint_provenance(changed, active)


def test_top1_conflict_metrics_detects_collisions_and_two_cycles() -> None:
    right = np.ones((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    down = np.ones_like(right)
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    # Every query picks one of the first slots, with an explicit 0<->1 cycle.
    right[:, 0] = 0.0
    down[:, 0] = 0.0
    right[0, 1] = -1.0
    right[1, 0] = -1.0
    down[0, 1] = -1.0
    down[1, 0] = -1.0
    score = CompatibilityMatrices("conflicted", right, down)
    metrics = trainer.top1_conflict_metrics(score, _identity_slot_to_target())
    assert metrics["top1_collision_excess_rate"] > 0.9
    assert metrics["directed_two_cycle_rate"] > 0.0


def test_bootstrap_is_deterministic_and_singleton_safe() -> None:
    first = trainer.bootstrap_mean_ci([0.1, 0.2, -0.1], seed=41, samples=500)
    second = trainer.bootstrap_mean_ci([0.1, 0.2, -0.1], seed=41, samples=500)
    assert first == second
    assert trainer.bootstrap_mean_ci([0.3], seed=1) == pytest.approx((0.3, 0.3))


def test_aggregate_bootstraps_whole_sources_not_correlated_panels() -> None:
    records = []
    for name, delta in (("a.png", 0.1), ("b.png", -0.1)):
        for panel in ("primary_kornia", "independent_libjpeg"):
            records.append(
                {
                    "name": name,
                    "panel": panel,
                    "delta": {
                        "recall_at_1": delta,
                        "recall_at_5": delta,
                        "recall_at_32": delta,
                        "mrr": delta,
                        "top1_collision_excess_rate": delta,
                        "directed_two_cycle_rate": delta,
                    },
                }
            )
    aggregate = trainer._aggregate(records, kind="retrieval", seed=61)
    assert aggregate["count"] == 4
    assert aggregate["source_count"] == 2
    assert aggregate["bootstrap_unit"] == "whole_source_mean_across_panels_and_replicas"
