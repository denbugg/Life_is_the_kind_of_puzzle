#!/usr/bin/env python3
"""Frozen CPU fresh64 confirmation of the raw/twin union reranker v2."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.protocol import compute_protocol_digest, select_manifest_records, sha256_file
from aiijc_puzzle.raw_twin_union_reranker import FEATURE_NAMES, RawTwinUnionReranker
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments
from aiijc_puzzle.socket_sorter_production import load_socket_checkpoint
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)
from aiijc_puzzle.synthetic_socket_evaluation import (
    exact_local_retrieval_metrics,
    load_checkpoint_with_lineage,
    names_digest,
)

try:
    from scripts.run_fullres_twin_side_matcher import (
        _atomic_json,
        _evaluation_exclusion_registry,
        _prepare_boards,
        _project_relative,
        _two_view_case,
    )
    from scripts.run_raw_twin_union_reranker_v2 import (
        COUNT,
        EDGE_BUDGET,
        GRID,
        LOCAL_KS,
        FrozenCasePrediction,
        _adjacency_fraction,
        _load_twin_checkpoint,
        _truth_by_anchor,
        freeze_case_prediction,
    )
    from scripts.run_raw_twin_union_reranker_v2 import (
        _load_config as load_base_config,
    )
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    from run_fullres_twin_side_matcher import (
        _atomic_json,
        _evaluation_exclusion_registry,
        _prepare_boards,
        _project_relative,
        _two_view_case,
    )
    from run_raw_twin_union_reranker_v2 import (
        COUNT,
        EDGE_BUDGET,
        GRID,
        LOCAL_KS,
        FrozenCasePrediction,
        _adjacency_fraction,
        _load_twin_checkpoint,
        _truth_by_anchor,
        freeze_case_prediction,
    )
    from run_raw_twin_union_reranker_v2 import (
        _load_config as load_base_config,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/raw_twin_union_reranker_fresh64_confirmation_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/raw-twin-union-reranker/frozen-v2-fresh64-draw0"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
BASE_CONFIG = PROJECT_ROOT / "configs/raw_twin_union_reranker_v2_preregistered.json"
BASE_REPORT = PROJECT_ROOT / "outputs/raw-twin-union-reranker/v2-fit256-s400-eval24/report.json"
UNION_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/raw-twin-union-reranker/v2-fit256-s400-eval24/raw-twin-union-reranker-v2.pt"
)
UNION_SELECTION = (
    PROJECT_ROOT / "outputs/raw-twin-union-reranker/v2-fit256-s400-eval24/selection-commitment.json"
)
SOCKET_CHECKPOINT = (
    PROJECT_ROOT / "outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/socket_matcher.pt"
)
TWIN_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/fullres-twin-side-matcher/v1-fit256-s400-eval24/fullres-twin-side-matcher.pt"
)
COMPONENT_SELECTION = (
    PROJECT_ROOT / "outputs/component-absolute-placer/v1-selection/selection_commitment.json"
)
EXPECTED_SOURCES = 64
SELECTION_SEED = 20330918
SYNTHETIC_SEED = 20330918
BOOTSTRAP_RESAMPLES = 20_000
SELECTION_NAMESPACE = "aiijc-raw-twin-union-reranker-frozen-fresh64-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("selection", "run"), default="run")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    return parser.parse_args()


def _ordered_roster_names(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = (
        ("fit_filenames", "fit_order_digest"),
        ("evaluation_filenames", "evaluation_order_digest"),
    )
    output: list[str] = []
    for names_key, digest_key in pairs:
        names = payload.get(names_key)
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError(f"{path} has no valid {names_key}")
        if names_digest(names) != payload.get(digest_key):
            raise ValueError(f"{path} {names_key} digest mismatch")
        output.extend(Path(name).name for name in names)
    if len(output) != len(set(output)):
        raise ValueError(f"{path} fit/evaluation rosters overlap")
    return tuple(output)


def _write_config_and_sidecar(path: Path, payload: dict[str, Any]) -> str:
    sidecar = path.with_name(f"{path.name}.sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite frozen fresh64 config")
    _atomic_json(path, payload)
    digest = sha256_file(path)
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def freeze_selection(args: argparse.Namespace) -> None:
    base = load_base_config(BASE_CONFIG)
    expected_hashes = {
        BASE_CONFIG: "6741e92e832a630f1b83bde6edc8a341a348f52daa82313c40a8f32c7c1173d4",
        BASE_REPORT: "0a5f0bb990654a0e191430bbf05796332c2f6fce181d51b05ee8b11ba1477bc4",
        UNION_CHECKPOINT: "a5f882ab3c827e4e3779be3372c62d2a8fb9cd95d3558fd30cc566a9c3137f79",
        UNION_SELECTION: "71ae4f5095489613857fcd25c541fe496da0d6861f6ff604850147dd04b91cd2",
        COMPONENT_SELECTION: "2b2b0c90e559ca2a6c7898d8eaccc4f8944a73c35d7d78722e0bb30939e0ebe6",
        SOCKET_CHECKPOINT: base["frozen_inputs"]["socket_d64"]["sha256"],
        TWIN_CHECKPOINT: base["frozen_inputs"]["fullres_twin"]["sha256"],
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen input hash changed: {path}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("manifest protocol digest mismatch")
    _, socket_lineage = load_checkpoint_with_lineage(
        SOCKET_CHECKPOINT,
        project_root=PROJECT_ROOT,
    )
    _, twin_lineage = load_checkpoint_with_lineage(
        TWIN_CHECKPOINT,
        project_root=PROJECT_ROOT,
    )
    union_payload = torch.load(UNION_CHECKPOINT, map_location="cpu", weights_only=False)
    union_train = union_payload.get("selection", {}).get("train_filenames")
    union_train_digest = union_payload.get("selection", {}).get("train_digest")
    if (
        not isinstance(union_train, list)
        or names_digest(union_train, sort_names=True) != union_train_digest
    ):
        raise ValueError("union checkpoint sorted training-lineage digest is invalid")
    union_commitment_names = json.loads(UNION_SELECTION.read_text(encoding="utf-8"))[
        "fit_filenames"
    ]
    if union_train != union_commitment_names:
        raise ValueError("union checkpoint train roster differs from frozen commitment")
    combined_lineage = tuple(
        sorted(set(socket_lineage.filenames) | set(twin_lineage.filenames) | set(union_train))
    )
    excluded, registry = _evaluation_exclusion_registry(
        combined_lineage,
        output_dir=args.output_dir,
    )
    explicit = {
        UNION_SELECTION: _ordered_roster_names(UNION_SELECTION),
        COMPONENT_SELECTION: _ordered_roster_names(COMPONENT_SELECTION),
    }
    for path, names in explicit.items():
        excluded.update(names)
        registry.append(
            {
                "path": _project_relative(path),
                "sha256": sha256_file(path),
                "panel_filename_count": len(names),
                "panel_filename_digest": names_digest(names, sort_names=True),
                "role": "mandatory-complete-fit-and-evaluation-roster-exclusion",
            }
        )
    train = manifest.get("splits", {}).get("train")
    if not isinstance(train, list):
        raise ValueError("manifest train split missing")
    exclusion_digest = names_digest(tuple(sorted(excluded)))
    namespace = (
        f"{SELECTION_NAMESPACE}\0{expected_hashes[UNION_CHECKPOINT]}"
        f"\0{exclusion_digest}\0{SELECTION_SEED}"
    )
    ranked = select_manifest_records(
        manifest,
        "train",
        limit=len(train),
        seed=SELECTION_SEED,
        namespace=namespace,
    )
    records = tuple(
        record for record in ranked if Path(str(record["filename"])).name not in excluded
    )[:EXPECTED_SOURCES]
    if len(records) != EXPECTED_SOURCES:
        raise ValueError("not enough source-disjoint organizer-train records")
    sources = tuple(str(record["filename"]) for record in records)
    if set(sources) & excluded:
        raise RuntimeError("fresh64 roster overlaps exclusion union")
    payload = {
        "schema": "aiijc-raw-twin-union-reranker-fresh64-confirmation-v1",
        "status": "frozen-before-selected-target-access",
        "registered_before_selected_target_access": True,
        "registered_before_dirty_prediction_generation": True,
        "purpose": "frozen-model CPU fresh64 confirmation for final submission candidate",
        "frozen_inputs": {
            "base_config": _project_relative(BASE_CONFIG),
            "base_config_sha256": expected_hashes[BASE_CONFIG],
            "base_report": _project_relative(BASE_REPORT),
            "base_report_sha256": expected_hashes[BASE_REPORT],
            "union_checkpoint": _project_relative(UNION_CHECKPOINT),
            "union_checkpoint_sha256": expected_hashes[UNION_CHECKPOINT],
            "union_selection": _project_relative(UNION_SELECTION),
            "union_selection_sha256": expected_hashes[UNION_SELECTION],
            "socket_checkpoint": _project_relative(SOCKET_CHECKPOINT),
            "socket_checkpoint_sha256": expected_hashes[SOCKET_CHECKPOINT],
            "twin_checkpoint": _project_relative(TWIN_CHECKPOINT),
            "twin_checkpoint_sha256": expected_hashes[TWIN_CHECKPOINT],
            "component_selection": _project_relative(COMPONENT_SELECTION),
            "component_selection_sha256": expected_hashes[COMPONENT_SELECTION],
            "no_retrain_recalibration_hparam_or_inference_semantics_change": True,
        },
        "selection": {
            "split": "train",
            "namespace": namespace,
            "selection_seed": SELECTION_SEED,
            "synthetic_seed": SYNTHETIC_SEED,
            "draw_indices": [0],
            "source_filenames": list(sources),
            "source_order_digest": names_digest(sources),
            "source_set_digest": names_digest(sources, sort_names=True),
            "excluded_filename_count": len(excluded),
            "excluded_filename_digest": exclusion_digest,
            "exclusion_registry": registry,
            "selected_exclusion_overlap": [],
            "union_fit_eval_excluded": len(explicit[UNION_SELECTION]),
            "component_fit_eval_excluded": len(explicit[COMPONENT_SELECTION]),
        },
        "inference": {
            "device": "cpu",
            "candidate_roster": base["candidate_roster"]["operation"],
            "partial_ot": base["inference"]["partial_ot"],
            "hard_projection": base["inference"]["hard_projection"],
            "decoder": "unchanged decoder144 plus cyclic-border5",
            "comparator": base["inference"]["baseline"],
        },
        "gate": {
            "submission_candidate_all_required": {
                "exact_tiles_delta_mean_minimum": 0.25,
                "adjacency_delta_strictly_positive": True,
                "all_layouts_strict_original_permutations": True,
            },
            "clustered_ci": {
                "cluster": "source; exactly one draw per source",
                "confidence": 0.95,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "seed": SYNTHETIC_SEED + 100,
                "exact_ci_is_reported_but_not_an_additional_gate": True,
            },
            "projected_and_top144_edge_deltas": "reported, not additional gate",
        },
        "legality": {
            "organizer_train_only": True,
            "target_available_to_inference": False,
            "layout": "strict permutation of original upright tile identities",
            "pixel_replacement_or_generation": False,
            "holdout_opened": False,
            "competition_test_opened": False,
        },
    }
    digest = _write_config_and_sidecar(args.config, payload)
    print(
        json.dumps(
            {
                "event": "fresh64_config_frozen",
                "path": str(args.config),
                "sha256": digest,
                "source_order_digest": names_digest(sources),
                "excluded": len(excluded),
                "selected_target_access": False,
            }
        ),
        flush=True,
    )


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    sidecar = path.with_name(f"{path.name}.sha256")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError("fresh64 config/sidecar SHA mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "aiijc-raw-twin-union-reranker-fresh64-confirmation-v1":
        raise ValueError("unsupported fresh64 config schema")
    if payload.get("status") != "frozen-before-selected-target-access":
        raise ValueError("fresh64 config was not frozen before target access")
    return payload, observed


def source_clustered_ci(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("cluster values must be a finite non-empty vector")
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    generator = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    remaining = resamples
    while remaining:
        batch = min(2048, remaining)
        indices = generator.integers(0, len(array), size=(batch, len(array)))
        chunks.append(array[indices].mean(axis=1))
        remaining -= batch
    bootstrap = np.concatenate(chunks)
    lower, upper = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "mean": float(array.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "source_clusters": len(array),
        "bootstrap_resamples": resamples,
    }


def evaluate_gate(metrics: Mapping[str, Any], *, strict_layouts: int) -> dict[str, Any]:
    exact = float(metrics["exact_tiles_delta"]["mean"])
    adjacency = float(metrics["adjacency_delta"]["mean"])
    checks = {
        "exact_mean_at_least_quarter_tile": {
            "observed": exact,
            "required": 0.25,
            "pass": exact >= 0.25,
        },
        "adjacency_strictly_positive": {
            "observed": adjacency,
            "required": ">0",
            "pass": adjacency > 0.0,
        },
        "all_layouts_strict": {
            "observed": strict_layouts,
            "required": 2 * EXPECTED_SOURCES,
            "pass": strict_layouts == 2 * EXPECTED_SOURCES,
        },
    }
    passed = all(bool(check["pass"]) for check in checks.values())
    exact_ci_positive = float(metrics["exact_tiles_delta"]["ci95_lower"]) > 0.0
    return {
        "pass": passed,
        "status": (
            "frozen-fresh64-submission-candidate-confirmed"
            if passed
            else "frozen-fresh64-submission-candidate-not-confirmed"
        ),
        "checks": checks,
        "exact_ci_excludes_zero": exact_ci_positive,
        "exact_ci_role": "reported honestly; not an additional preregistered gate",
    }


def _case_seeds(seed: int, filename: str) -> tuple[int, int]:
    # Same deterministic case construction semantics as v2, with the fresh seed.
    import hashlib

    digest = hashlib.sha256(f"{seed}\0{filename}\0raw-twin-v2".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63), int.from_bytes(
        digest[8:16], "little"
    ) % (2**63)


def _decode_frozen_layouts(
    prediction: FrozenCasePrediction,
) -> dict[str, np.ndarray]:
    decoder_config = SocketDecoderConfig(
        component_edge_budget_per_axis=EDGE_BUDGET,
        swap_edge_budget_per_axis=EDGE_BUDGET,
        max_swap_steps=24,
    )
    cyclic_config = CyclicTranslationConfig(border_weight=5.0)
    output: dict[str, np.ndarray] = {}
    for variant, (right, down) in prediction.assignments.items():
        decoded = decode_socket_assignments(
            right,
            down,
            grid=GRID,
            config=decoder_config,
        )
        layout = select_global_cyclic_translation(
            decoded.layout,
            right,
            down,
            grid=GRID,
            config=cyclic_config,
        ).layout
        if not np.array_equal(np.sort(layout), np.arange(COUNT)):
            raise RuntimeError("fresh64 decode is not a strict original permutation")
        output[variant] = np.ascontiguousarray(layout, dtype=np.int32)
    return output


def _write_frozen(
    predictions: list[FrozenCasePrediction],
    layouts: list[dict[str, np.ndarray]],
    output_dir: Path,
) -> tuple[Path, Path]:
    arrays: dict[str, np.ndarray] = {}
    cases: list[dict[str, Any]] = []
    for index, (prediction, case_layouts) in enumerate(zip(predictions, layouts, strict=True)):
        prefix = f"case_{index:04d}"
        for variant, axes in prediction.variants.items():
            arrays[f"{prefix}__{variant}__layout"] = case_layouts[variant]
            for axis, value in axes.items():
                arrays[f"{prefix}__{variant}__{axis}__candidates"] = value.candidates
                arrays[f"{prefix}__{variant}__{axis}__sources"] = value.sources
                arrays[f"{prefix}__{variant}__{axis}__targets"] = value.targets
                arrays[f"{prefix}__{variant}__{axis}__confidence"] = value.confidence
        cases.append(
            {
                "prefix": prefix,
                "case_id": prediction.case_id,
                "source_filename": prediction.source_filename,
                "runtime_seconds": prediction.runtime_seconds,
            }
        )
    npz_path = output_dir / "frozen-target-free-predictions.npz"
    np.savez_compressed(npz_path, **arrays)
    metadata_path = output_dir / "frozen-target-free-predictions.json"
    _atomic_json(
        metadata_path,
        {
            "schema": "aiijc-raw-twin-union-fresh64-frozen-predictions-v1",
            "contains_exact_references": False,
            "contains_clean_or_generated_pixels": False,
            "contains_dirty_pixels": False,
            "contains_target_free_strict_layouts": True,
            "candidate_union": ("immutable raw32 union twin32 union frozen raw hard projection"),
            "cases": cases,
        },
    )
    return npz_path, metadata_path


def _score_case(
    prediction: FrozenCasePrediction,
    layouts: Mapping[str, np.ndarray],
    reference: np.ndarray,
) -> dict[str, Any]:
    output: dict[str, Any] = {"source_filename": prediction.source_filename}
    for variant in ("socket_d64_frozen", "learned_union"):
        local = exact_local_retrieval_metrics(
            prediction.variants[variant]["right"].candidates,
            prediction.variants[variant]["down"].candidates,
            reference,
            ks=LOCAL_KS,
        )
        projected_correct = 0
        top144_correct = 0
        for axis in ("right", "down"):
            frozen = prediction.variants[variant][axis]
            truth = _truth_by_anchor(reference, axis=axis)
            correct = truth[frozen.sources] == frozen.targets
            projected_correct += int(correct.sum())
            order = np.argsort(-frozen.confidence, kind="stable")[:EDGE_BUDGET]
            top144_correct += int(correct[order].sum())
        layout = layouts[variant]
        output[variant] = {
            "local_r1": float(local["pooled_r1"]),
            "local_r5": float(local["pooled_r5"]),
            "projected_correct": projected_correct,
            "top144_correct": top144_correct,
            "top144_precision": top144_correct / float(2 * EDGE_BUDGET),
            "exact_tiles": int(np.count_nonzero(layout == reference)),
            "adjacency": _adjacency_fraction(layout, reference),
            "strict_permutation": bool(np.array_equal(np.sort(layout), np.arange(COUNT))),
        }
    return output


def _metric_summary(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, float]:
    raw = np.asarray([row["socket_d64_frozen"][metric] for row in rows], dtype=np.float64)
    learned = np.asarray([row["learned_union"][metric] for row in rows], dtype=np.float64)
    return {
        "socket_d64_frozen_mean": float(raw.mean()),
        "learned_union_mean": float(learned.mean()),
        "mean_delta": float((learned - raw).mean()),
    }


def run_confirmation(args: argparse.Namespace) -> None:
    config, config_sha = load_config(args.config)
    frozen_inputs = config["frozen_inputs"]
    frozen_paths = {
        "base_config": BASE_CONFIG,
        "base_report": BASE_REPORT,
        "union_checkpoint": UNION_CHECKPOINT,
        "union_selection": UNION_SELECTION,
        "socket_checkpoint": SOCKET_CHECKPOINT,
        "twin_checkpoint": TWIN_CHECKPOINT,
        "component_selection": COMPONENT_SELECTION,
    }
    for key, path in frozen_paths.items():
        if sha256_file(path) != frozen_inputs[f"{key}_sha256"]:
            raise ValueError(f"frozen input changed: {key}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("manifest protocol digest mismatch")
    names = tuple(config["selection"]["source_filenames"])
    if (
        len(names) != EXPECTED_SOURCES
        or names_digest(names) != config["selection"]["source_order_digest"]
    ):
        raise ValueError("fresh64 source roster contract changed")
    lookup = {str(record["filename"]): dict(record) for record in manifest["splits"]["train"]}
    if not set(names).issubset(lookup):
        raise ValueError("fresh64 roster contains non-train or missing sources")
    records = tuple(lookup[name] for name in names)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    prediction_path = output_dir / "frozen-target-free-predictions.npz"
    if report_path.exists() or prediction_path.exists():
        raise FileExistsError("refusing to overwrite frozen fresh64 confirmation")
    print(
        json.dumps(
            {
                "event": "start",
                "pid": os.getpid(),
                "device": "cpu",
                "sources": EXPECTED_SOURCES,
            }
        ),
        flush=True,
    )
    boards = _prepare_boards(records, args.targets)
    socket = load_socket_checkpoint(SOCKET_CHECKPOINT, device=torch.device("cpu"))
    twin, twin_contract = _load_twin_checkpoint(
        TWIN_CHECKPOINT,
        device=torch.device("cpu"),
    )
    checkpoint = torch.load(UNION_CHECKPOINT, map_location="cpu", weights_only=False)
    contract = checkpoint.get("contract", {})
    if contract.get("feature_dimension") != len(FEATURE_NAMES):
        raise ValueError("union checkpoint feature contract changed")
    model = RawTwinUnionReranker(
        feature_dimension=int(contract["feature_dimension"]),
        hidden_dimension=int(contract["hidden_dimension"]),
        residual_limit=float(contract["residual_limit"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    predictions: list[FrozenCasePrediction] = []
    layouts: list[dict[str, np.ndarray]] = []
    references: dict[str, np.ndarray] = {}
    started = perf_counter()
    seed = int(config["selection"]["synthetic_seed"])
    with torch.inference_mode():
        for index, board in enumerate(boards):
            corruption_seed, permutation_seed = _case_seeds(seed, board.filename)
            dirty, _, reference = _two_view_case(
                board.tiles,
                first_seed=corruption_seed,
                second_seed=corruption_seed + 1,
                permutation_seed=permutation_seed,
            )
            case_id = f"raw-twin-v2-fresh64-{index:03d}-{Path(board.filename).stem}"
            prediction = freeze_case_prediction(
                model,
                dirty,
                case_id=case_id,
                source_filename=board.filename,
                socket=socket,
                twin=twin,
                device=torch.device("cpu"),
            )
            predictions.append(prediction)
            layouts.append(_decode_frozen_layouts(prediction))
            references[case_id] = np.ascontiguousarray(reference, dtype=np.int32)
            print(
                json.dumps({"event": "freeze", "done": index + 1, "total": EXPECTED_SOURCES}),
                flush=True,
            )
    npz_path, metadata_path = _write_frozen(predictions, layouts, output_dir)
    npz_sha = sha256_file(npz_path)
    metadata_sha = sha256_file(metadata_path)
    print(
        json.dumps({"event": "predictions_frozen", "path": str(npz_path), "sha256": npz_sha}),
        flush=True,
    )
    rows = [
        _score_case(prediction, case_layouts, references[prediction.case_id])
        for prediction, case_layouts in zip(predictions, layouts, strict=True)
    ]
    metric_names = (
        "local_r1",
        "local_r5",
        "projected_correct",
        "top144_correct",
        "top144_precision",
        "exact_tiles",
        "adjacency",
    )
    summaries = {name: _metric_summary(rows, name) for name in metric_names}
    ci_seed = int(config["gate"]["clustered_ci"]["seed"])

    def delta_ci(metric: str, offset: int) -> dict[str, float | int]:
        values = [
            float(row["learned_union"][metric]) - float(row["socket_d64_frozen"][metric])
            for row in rows
        ]
        return source_clustered_ci(values, seed=ci_seed + offset)

    metrics = {
        "exact_tiles_delta": delta_ci("exact_tiles", 0),
        "adjacency_delta": delta_ci("adjacency", 1),
        "projected_correct_delta": delta_ci("projected_correct", 2),
        "top144_correct_delta": delta_ci("top144_correct", 3),
        "top144_precision_delta": delta_ci("top144_precision", 4),
        "local_r1_delta": delta_ci("local_r1", 5),
        "local_r5_delta": delta_ci("local_r5", 6),
        "arms": summaries,
    }
    strict_layouts = sum(
        int(row[variant]["strict_permutation"])
        for row in rows
        for variant in ("socket_d64_frozen", "learned_union")
    )
    gate = evaluate_gate(metrics, strict_layouts=strict_layouts)
    report = {
        "schema": "aiijc-raw-twin-union-reranker-fresh64-confirmation-report-v1",
        "status": gate["status"],
        "config": _project_relative(args.config),
        "config_sha256": config_sha,
        "frozen_predictions": {
            "npz": _project_relative(npz_path),
            "npz_sha256": npz_sha,
            "metadata": _project_relative(metadata_path),
            "metadata_sha256": metadata_sha,
            "predictions_and_layouts_frozen_before_reference_scoring": True,
            "contains_exact_references": False,
        },
        "frozen_inputs": config["frozen_inputs"],
        "selection": config["selection"],
        "metrics": metrics,
        "gate": gate,
        "rows": rows,
        "runtime_seconds": perf_counter() - started,
        "strict_original_permutations": strict_layouts,
        "no_retrain_recalibration_hparam_or_inference_semantics_change": True,
        "twin_contract": twin_contract,
        "holdout_opened": False,
        "competition_test_opened": False,
        "pixel_replacement_or_generation": False,
    }
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "gate": gate,
                "metrics": metrics,
            }
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.mode == "selection":
        freeze_selection(args)
    else:
        run_confirmation(args)


if __name__ == "__main__":
    main()
