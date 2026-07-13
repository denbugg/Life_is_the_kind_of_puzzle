from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from puzzle_assembly.compatibility import CompatibilityMatrices
from puzzle_assembly.geometry import TILE_COUNT


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_qap_weight_confirmation.py"
_SPEC = importlib.util.spec_from_file_location("qap_weight_confirmation", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
evaluator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = evaluator
_SPEC.loader.exec_module(evaluator)


@dataclass(frozen=True)
class FrozenBundle:
    config: Path
    data_root: Path
    shard_dirs: tuple[Path, Path]
    shard_hashes: tuple[str, str]
    finalized_dir: Path
    finalized_hash: str
    names: tuple[str, ...]


def _find_asset(env_name: str, filename: str, local_relative: str) -> Path:
    if os.environ.get(env_name):
        explicit = Path(os.environ[env_name]).expanduser().resolve()
        if not explicit.is_file():
            raise RuntimeError(f"{env_name} does not point to a file: {explicit}")
        return explicit
    repository = Path(__file__).resolve().parents[1]
    local = repository / local_relative
    if local.is_file():
        return local.resolve()
    kaggle_input = Path("/kaggle/input")
    matches = sorted(kaggle_input.glob(f"**/{filename}")) if kaggle_input.is_dir() else []
    if len(matches) != 1:
        raise RuntimeError(f"expected one test asset {filename}, found {matches}")
    return matches[0].resolve()


def _asset_cli() -> list[str]:
    repository = Path(__file__).resolve().parents[1]
    return [
        "--denoiser",
        str(
            _find_asset(
                "QAP_CONFIRMATION_DENOISER",
                "selected_tilenaf_synth_50k.pt",
                "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt",
            )
        ),
        "--hbt-checkpoint",
        str(
            _find_asset(
                "QAP_CONFIRMATION_HBT",
                "hbt_d320_denoised_rgb_sobel.pt",
                "runs/assembly_v1/kaggle/edge2vec_gradient_gpu/hbt_d320_denoised_rgb_sobel.pt",
            )
        ),
        "--manifest",
        str((repository / "configs/denoise_splits_seed20260710.json").resolve()),
        "--quarantine",
        str((repository / "configs/denoise_validation_quarantine_v1.json").resolve()),
    ]


def _base_args(*extra: str) -> object:
    return evaluator.parse_args(
        [
            "--action",
            "phase-a",
            "--config",
            str(Path(__file__).resolve().parents[1] / "configs/qap_weight_confirmation_v1.json"),
            *_asset_cli(),
            *extra,
        ]
    )


def _fake_predictor(name: str, image: np.ndarray) -> object:
    del name
    layout = np.arange(TILE_COUNT, dtype=np.int32)
    digest = evaluator.hashlib.sha256(
        evaluator.split_tiles_numpy(image).tobytes()
    ).hexdigest()
    return evaluator.PhaseAPrediction(
        layouts={"baseline": layout.copy(), "candidate": layout.copy()},
        renders={"baseline": image.copy(), "candidate": image.copy()},
        initial_layout=layout.copy(),
        qap_seed=0,  # Replaced below because the seed is filename-bound.
        denoised_tiles_sha256=digest,
        diagnostics={"baseline": {}, "candidate": {}},
    )


def _predictor_for_name(name: str, image: np.ndarray) -> object:
    prediction = _fake_predictor(name, image)
    return evaluator.PhaseAPrediction(
        layouts=prediction.layouts,
        renders=prediction.renders,
        initial_layout=prediction.initial_layout,
        qap_seed=evaluator._filename_qap_seed(name),
        denoised_tiles_sha256=prediction.denoised_tiles_sha256,
        diagnostics=prediction.diagnostics,
    )


@pytest.fixture(scope="module")
def frozen_bundle(tmp_path_factory: pytest.TempPathFactory) -> FrozenBundle:
    root = tmp_path_factory.mktemp("qap_weight_confirmation")
    config = Path(__file__).resolve().parents[1] / "configs/qap_weight_confirmation_v1.json"
    probe = _base_args()
    protocol, assets = evaluator._validated_protocol_and_assets(probe)
    names = evaluator._expected_names(protocol, assets)
    data_root = root / "input_only"
    input_dir = data_root / "train" / "inputs"
    input_dir.mkdir(parents=True)
    encoded = evaluator._png_bytes(np.zeros((480, 480, 3), dtype=np.uint8))
    for name in names:
        evaluator._atomic_bytes(input_dir / name, encoded)
    assert not (data_root / "train" / "targets").exists()

    shard_dirs = (root / "shard_0", root / "shard_1")
    shard_hashes: list[str] = []
    for rank, shard_dir in enumerate(shard_dirs):
        args = _base_args(
            "--rank",
            str(rank),
            "--world-size",
            "2",
            "--phase-a-dir",
            str(shard_dir),
            "--data-root",
            str(data_root),
        )
        result = evaluator.run_phase_a(args, predictor=_predictor_for_name)
        shard_hashes.append(result["phase_a_envelope_sha256"])
        assert result["target_paths_or_pixels_read"] is False

    finalized_dir = root / "finalized"
    args = evaluator.parse_args(
        [
            "--action",
            "finalize-phase-a",
            "--config",
            str(config),
            *_asset_cli(),
            "--phase-a-dirs",
            *(str(path) for path in shard_dirs),
            "--phase-a-envelope-sha256s",
            *shard_hashes,
            "--finalized-phase-a-dir",
            str(finalized_dir),
        ]
    )
    finalized = evaluator.run_finalize_phase_a(args)
    return FrozenBundle(
        config=config,
        data_root=data_root,
        shard_dirs=shard_dirs,
        shard_hashes=(shard_hashes[0], shard_hashes[1]),
        finalized_dir=finalized_dir,
        finalized_hash=finalized["phase_a_envelope_sha256"],
        names=tuple(names),
    )


def _copied_finalized(bundle: FrozenBundle, destination: Path) -> Path:
    shutil.copytree(bundle.finalized_dir, destination)
    return destination


def _phase_b_args(
    bundle: FrozenBundle, finalized_dir: Path, output: Path, *, config: Path | None = None
) -> object:
    return evaluator.parse_args(
        [
            "--action",
            "phase-b",
            "--config",
            str(config or bundle.config),
            *_asset_cli(),
            "--data-root",
            str(bundle.data_root),
            "--finalized-phase-a-dir",
            str(finalized_dir),
            "--phase-a-envelope-sha256",
            bundle.finalized_hash,
            "--output",
            str(output),
        ]
    )


def test_protocol_is_exactly_pinned_to_fresh_64_source_slice() -> None:
    args = _base_args()
    protocol, assets = evaluator._validated_protocol_and_assets(args)
    names = evaluator._expected_names(protocol, assets)
    assert len(names) == 64
    assert evaluator._names_sha256(names) == evaluator.EXPECTED_NAMES_SHA256
    assert protocol["baseline"]["hbt_weight"] == 4.0
    assert protocol["candidate"]["hbt_weight"] == 1.0
    assert protocol["original_real_confirmation"]["metric"]["bootstrap_resamples"] == 20000
    assert protocol["original_real_confirmation"]["metric"]["bootstrap_seed"] == 20260711


def test_phase_a_refuses_any_topology_other_than_two_fixed_shards(tmp_path: Path) -> None:
    args = _base_args(
        "--rank", "0", "--world-size", "1", "--phase-a-dir", str(tmp_path / "bad")
    )
    with pytest.raises(RuntimeError, match="world-size 2"):
        evaluator.run_phase_a(args, predictor=_predictor_for_name)


def test_default_predictor_uses_identical_hbt_seed_and_qap_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _base_args()
    protocol, assets = evaluator._validated_protocol_and_assets(args)
    model = torch.nn.Identity()
    monkeypatch.setattr(
        evaluator, "load_restorer", lambda *_args, **_kwargs: (model, torch.device("cpu"), {})
    )
    monkeypatch.setattr(
        evaluator,
        "load_embedding_checkpoint",
        lambda *_args, **_kwargs: (torch.nn.Identity(), {}),
    )
    tiles = np.zeros((TILE_COUNT, 20, 20, 3), dtype=np.uint8)
    monkeypatch.setattr(evaluator, "restore_tiles_uint8", lambda *_args, **_kwargs: tiles)
    matrix = np.zeros((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    np.fill_diagonal(matrix, np.inf)

    def score(name: str) -> CompatibilityMatrices:
        return CompatibilityMatrices(name, matrix.copy(), matrix.copy())

    monkeypatch.setattr(
        evaluator,
        "build_classical_score_bank",
        lambda *_args, **_kwargs: {"denoised_c1": score("denoised_c1")},
    )
    monkeypatch.setattr(
        evaluator,
        "learned_compatibility",
        lambda *_args, **_kwargs: (score("denoised_hbt_l1"), {}),
    )
    monkeypatch.setattr(
        evaluator,
        "fuse_ranked_scores",
        lambda _bank, *, name, **_kwargs: score(name),
    )
    identity = np.arange(TILE_COUNT, dtype=np.int32)
    monkeypatch.setattr(
        evaluator,
        "soft_cycle_component_solver",
        lambda *_args, **_kwargs: SimpleNamespace(position_to_slot=identity.copy()),
    )
    calls: list[dict[str, object]] = []

    def qap(score_value: CompatibilityMatrices, **kwargs: object) -> object:
        calls.append({"score": score_value.name, **kwargs})
        return SimpleNamespace(
            position_to_slot=identity.copy(),
            objective=1.0,
            relaxed_objective=1.0,
            restart=0,
            iterations=25,
            converged=True,
        )

    monkeypatch.setattr(evaluator, "directional_qap", qap)
    predictor = evaluator._build_default_predictor(protocol, assets, args)
    name = "img_123456.png"
    prediction = predictor(name, np.zeros((480, 480, 3), dtype=np.uint8))
    assert len(calls) == 2
    expected_seed = evaluator._filename_qap_seed(name)
    for call in calls:
        np.testing.assert_array_equal(call["initial"], identity)
        assert call["seed"] == expected_seed
        assert call["iterations"] == 25
        assert call["restarts"] == 2
        assert call["boundary_weight"] == 0.05
        assert call["refine_swaps"] == 8
        assert call["refine_weak_cells"] == 32
    assert prediction.qap_seed == expected_seed
    assert {call["score"] for call in calls} == {
        "denoised_C1_HBTw4_rank_fusion",
        "denoised_C1_HBTw1_rank_fusion",
    }


def test_two_shards_finalize_in_canonical_global_index_order(
    frozen_bundle: FrozenBundle,
) -> None:
    envelope = evaluator._load_exact_envelope(
        frozen_bundle.finalized_dir / evaluator.FINAL_MANIFEST,
        frozen_bundle.finalized_hash,
    )
    payload = envelope["payload"]
    assert [record["source_index"] for record in payload["records"]] == list(range(64))
    assert [record["name"] for record in payload["records"]] == list(frozen_bundle.names)
    assert payload["source_names_sha256"] == evaluator.EXPECTED_NAMES_SHA256
    assert payload["target_paths_constructed"] is False
    assert payload["target_files_opened"] is False
    assert payload["artifact_root"] == "artifacts"
    assert len(payload["shards"]) == 2
    for record in payload["records"]:
        assert record["input_path"] == f"train/inputs/{record['name']}"
        for variant in record["variants"].values():
            for key in ("layout_path", "render_path"):
                relative = Path(variant[key])
                assert not relative.is_absolute()
                assert relative.parts[0] == "artifacts"
                assert (frozen_bundle.finalized_dir / relative).is_file()


def test_manifest_substitution_fails_before_target_read(
    frozen_bundle: FrozenBundle, tmp_path: Path
) -> None:
    finalized = _copied_finalized(frozen_bundle, tmp_path / "finalized")
    manifest = finalized / evaluator.FINAL_MANIFEST
    envelope = json.loads(manifest.read_text(encoding="utf-8"))
    envelope["payload"]["source_names"][0] = "substituted.png"
    evaluator._atomic_bytes(manifest, evaluator._canonical_bytes(envelope) + b"\n")
    calls = 0

    def target_reader(_path: Path) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.zeros((480, 480, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="anchor mismatch"):
        evaluator.run_phase_b(
            _phase_b_args(frozen_bundle, finalized, tmp_path / "report.json"),
            target_reader=target_reader,
        )
    assert calls == 0
    assert not (finalized / evaluator.TARGET_MARKER).exists()


def test_render_substitution_fails_before_target_read(
    frozen_bundle: FrozenBundle, tmp_path: Path
) -> None:
    finalized = _copied_finalized(frozen_bundle, tmp_path / "finalized")
    envelope = evaluator._load_exact_envelope(
        finalized / evaluator.FINAL_MANIFEST,
        frozen_bundle.finalized_hash,
    )
    render = finalized / envelope["payload"]["records"][0]["variants"]["baseline"][
        "render_path"
    ]
    original = render.read_bytes()
    calls = 0

    def target_reader(_path: Path) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.zeros((480, 480, 3), dtype=np.uint8)

    try:
        render.write_bytes(original + b"tamper")
        with pytest.raises(RuntimeError, match="render hash mismatch"):
            evaluator.run_phase_b(
                _phase_b_args(frozen_bundle, finalized, tmp_path / "report.json"),
                target_reader=target_reader,
            )
    finally:
        render.write_bytes(original)
    assert calls == 0
    assert not (finalized / evaluator.TARGET_MARKER).exists()


def test_config_substitution_fails_before_target_read(
    frozen_bundle: FrozenBundle, tmp_path: Path
) -> None:
    finalized = _copied_finalized(frozen_bundle, tmp_path / "finalized")
    config = tmp_path / "substituted_config.json"
    payload = json.loads(frozen_bundle.config.read_text(encoding="utf-8"))
    payload["candidate"]["hbt_weight"] = 1.01
    evaluator._atomic_bytes(config, evaluator._canonical_bytes(payload) + b"\n")
    calls = 0

    def target_reader(_path: Path) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.zeros((480, 480, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="config SHA256 mismatch"):
        evaluator.run_phase_b(
            _phase_b_args(
                frozen_bundle, finalized, tmp_path / "report.json", config=config
            ),
            target_reader=target_reader,
        )
    assert calls == 0
    assert not (finalized / evaluator.TARGET_MARKER).exists()


def test_marker_durably_precedes_first_target_read_and_report_is_fail_closed(
    frozen_bundle: FrozenBundle, tmp_path: Path
) -> None:
    finalized = _copied_finalized(frozen_bundle, tmp_path / "finalized")
    marker = finalized / evaluator.TARGET_MARKER
    accesses: list[Path] = []

    def target_reader(path: Path) -> np.ndarray:
        assert marker.exists()
        evaluator._load_exact_envelope(marker, evaluator._sha256(marker))
        accesses.append(path)
        return np.zeros((480, 480, 3), dtype=np.uint8)

    output = tmp_path / "report.json"
    hidden: list[tuple[Path, Path]] = []
    try:
        for shard in frozen_bundle.shard_dirs:
            renamed = shard.with_name(shard.name + ".hidden")
            shard.rename(renamed)
            hidden.append((renamed, shard))
        report = evaluator.run_phase_b(
            _phase_b_args(frozen_bundle, finalized, output), target_reader=target_reader
        )
    finally:
        for renamed, original in reversed(hidden):
            renamed.rename(original)
    assert len(accesses) == 64
    assert report["phase_b"]["target_access_count"] == 64
    assert report["phase_b"]["marker_preceded_first_target_path_construction"] is True
    assert report["safe_for_submission"] is False
    assert report["eligible_for_final_audit"] is False
    assert report["gate"]["logic"] == "all_of"
    assert report["sealed_sets"] == {
        "final_audit_opened": False,
        "confirmation_audit_opened": False,
        "must_remain_unopened": True,
    }
    assert output.exists()
    # Keep the evaluator/runner schema contract executable.  The focused
    # overlay used on Kaggle may not contain the local runner source, in which
    # case the runner performs the same strict validation after Phase B.
    runner_path = (
        Path(__file__).resolve().parents[1]
        / "runs/assembly_v1/kaggle/qap_weight_confirmation_job/run_qap_weight_confirmation.py"
    )
    if runner_path.is_file():
        spec = importlib.util.spec_from_file_location(
            "qap_weight_confirmation_runner_validator", runner_path
        )
        assert spec is not None and spec.loader is not None
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)
        verified = runner.validate_final_report(
            output,
            config_path=frozen_bundle.config.resolve(),
            evaluator=_SCRIPT.resolve(),
            asset_records=report["assets"],
            finalized_manifest_path=finalized / evaluator.FINAL_MANIFEST,
            target_marker_path=marker,
            combined_names=list(frozen_bundle.names),
        )
        assert verified["sha256"] == evaluator._sha256(output)
        assert verified["safe_for_submission"] is False


def test_exact_gate_operators_and_bootstrap_are_deterministic() -> None:
    protocol, _ = evaluator._validated_protocol_and_assets(_base_args())
    gate_spec = protocol["original_real_confirmation"]["gate"]
    passing = {
        "mean_ssim_delta": 0.005,
        "bootstrap_95_ci": [np.nextafter(0.0, 1.0), 0.01],
        "wins": 40,
        "large_regressions": 6,
        "valid_permutation_count": 64,
    }
    gate = evaluator._gate(passing, gate_spec)
    assert gate == {
        "logic": "all_of",
        "passed": True,
        "checks": {
            "mean_ssim_delta_ge_0.005": True,
            "bootstrap_95_lower_gt_0": True,
            "wins_ge_40": True,
            "large_regressions_le_6": True,
            "valid_permutation_count_eq_64": True,
        },
    }
    for key, value in (
        ("mean_ssim_delta", np.nextafter(0.005, 0.0)),
        ("bootstrap_95_ci", [0.0, 0.01]),
        ("wins", 39),
        ("large_regressions", 7),
        ("valid_permutation_count", 63),
    ):
        changed = dict(passing)
        changed[key] = value
        assert evaluator._gate(changed, gate_spec)["passed"] is False
    metric = protocol["original_real_confirmation"]["metric"]
    deltas = np.linspace(-0.01, 0.02, 64, dtype=np.float64)
    first = evaluator._bootstrap_ci(deltas, metric)
    second = evaluator._bootstrap_ci(deltas, metric)
    assert first == second
