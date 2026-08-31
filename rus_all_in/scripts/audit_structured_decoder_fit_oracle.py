#!/usr/bin/env python3
"""Run one signed FIT-only structured-decoder capacity audit.

The workflow is deliberately staged and fail-closed:

``preregister``
    Verify the exact target-free fixed-5% reciprocal FIT head, immutable cache
    roster and confirmed six-arm solver bytes, then sign one action/gate
    contract before any reference reconstruction.
``freeze-controls``
    Recreate the exact already-opened FIT corruptions and freeze only the
    confirmed relation-selector layouts.  The exact references are discarded.
``score``
    Verify every target-free hash first, reconstruct organizer-train FIT
    references, and measure the explicitly target-assisted capacity ceiling.

There is no DEV, terminal16, competition-test, submission, Weco or Git mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.structured_decoder_fit_oracle import (
    DirectedEdge,
    evaluate_pair_safe_oracle,
    strict_layout,
    validate_fixed_reciprocal_head,
)
from aiijc_puzzle.synthetic_socket_evaluation import make_exact_synthetic_case
from aiijc_puzzle.taska_relation_selector_pipeline import (
    load_taska_relation_selector_resources,
    solve_taska_relation_selector_pipeline,
    verify_taska_relation_selector_solver,
)

try:
    from scripts import run_tri_emitter_edge_verifier as prior
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import run_tri_emitter_edge_verifier as prior


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/structured_decoder_fit_oracle_v1.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/compatibility-structured-decoder-fit-oracle/v1"
)
DEFAULT_JOINT_EXPERIMENT = (
    PROJECT_ROOT
    / "outputs/joint-reciprocal-tri-emitter-verifier/real-fit32-draw2-dev32-development-v2"
)
DEFAULT_HEAD_ARCHIVE = Path("fit/frozen-target-free-reciprocal-heads.npz")
DEFAULT_HEAD_METADATA = Path("fit/frozen-target-free-reciprocal-heads.json")
DEFAULT_HEAD_FREEZE = Path("fit/reciprocal-heads-pre-score-freeze.json")
TRI_REPORT = (
    PROJECT_ROOT
    / "outputs/tri-emitter-edge-verifier/fit32-draw2-s3-local16-v1/report.json"
)
MODULE_PATH = PROJECT_ROOT / "src/aiijc_puzzle/structured_decoder_fit_oracle.py"
TEST_PATH = PROJECT_ROOT / "tests/test_structured_decoder_fit_oracle.py"
GRID = 24
COUNT = GRID * GRID
REQUESTED_PER_AXIS = math.ceil(0.05 * COUNT)
FIT_CASE_SEED = 20260914
CONFIG_SCHEMA = "aiijc-structured-decoder-fit-oracle-config-v1"
CONTROL_SCHEMA = "aiijc-structured-decoder-fit-oracle-controls-v1"
CONTROL_FREEZE_SCHEMA = "aiijc-structured-decoder-fit-oracle-control-freeze-v1"
REPORT_SCHEMA = "aiijc-structured-decoder-fit-oracle-report-v1"
HEAD_SCHEMA = "aiijc-joint-reciprocal-target-free-fit-heads-v1"
HEAD_FREEZE_SCHEMA = "aiijc-joint-reciprocal-fit-heads-pre-score-freeze-v1"
STRICT_HEAD_LOADER_SCHEMA = (
    "aiijc-joint-reciprocal-strict-target-free-fit-cache-loader-v1"
)
TARGET_FREE_HEAD_INPUT_KEYS = sorted(
    (
        "raw_sides",
        "dino_sides",
        "candidates",
        "valid",
        "auxiliary",
        "raw_baseline",
        "emitter_topk",
    )
)
MINIMUM_TRUE_EDGE_HEADROOM_PER_BOARD = 8.0
MINIMUM_PAIR_SAFE_REALISATION_GAIN_PER_BOARD = 8.0
NO_REPEAT_REPORTS = (
    "outputs/taska-relation-ranked-union/fixed-v1/report.json",
    "outputs/taska-fullres-translation-consensus/fixed-v1/report.json",
    "outputs/taska-dense-contact-consensus-feasibility/fixed-v1/report.json",
    "outputs/taska-consensus-component-arm/opened32-v1/report.json",
    "outputs/taska-cross-arm-component-anchor/local32-v1/report.json",
    "outputs/taska-component-relation-anchor/fixed-v1/report.json",
    "outputs/taska-joint-component-pose/pilot-v1/report.json",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("availability", "preregister", "freeze-controls", "score", "validate-only"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--joint-experiment-dir", type=Path, default=DEFAULT_JOINT_EXPERIMENT)
    parser.add_argument("--head-archive", type=Path)
    parser.add_argument("--head-metadata", type=Path)
    parser.add_argument("--head-freeze", type=Path)
    parser.add_argument("--manifest", type=Path, default=prior.roster.DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=prior.roster.DEFAULT_TARGETS)
    parser.add_argument("--socket-checkpoint", type=Path, default=prior.SOCKET_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument("--focal-chunk-size", type=int, default=8192)
    parser.add_argument("--denoiser-batch-size", type=int, default=576)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.device == "mps" and not args.allow_nondeterministic_mps:
        raise ValueError("MPS control replay requires explicit nondeterminism consent")
    if args.device == "cpu" and args.allow_nondeterministic_mps:
        raise ValueError("MPS consent is incompatible with CPU mode")
    if args.focal_chunk_size <= 0 or args.denoiser_batch_size <= 0:
        raise ValueError("inference batch sizes must be positive")


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": _project_path(resolved), "sha256": sha256_file(resolved)}


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _write_sidecar(path: Path) -> None:
    sidecar = Path(f"{path}.sha256")
    with sidecar.open("x", encoding="utf-8") as stream:
        stream.write(f"{sha256_file(path)}  {_project_path(path)}\n")


def _resolve_head_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    root = args.joint_experiment_dir.resolve()
    return (
        (args.head_archive or root / DEFAULT_HEAD_ARCHIVE).resolve(),
        (args.head_metadata or root / DEFAULT_HEAD_METADATA).resolve(),
        (args.head_freeze or root / DEFAULT_HEAD_FREEZE).resolve(),
    )


def availability(args: argparse.Namespace) -> dict[str, Any]:
    archive, metadata, freeze = _resolve_head_paths(args)
    items = {
        "archive": {"path": _project_path(archive), "exists": archive.is_file()},
        "metadata": {"path": _project_path(metadata), "exists": metadata.is_file()},
        "pre_score_freeze": {"path": _project_path(freeze), "exists": freeze.is_file()},
    }
    available = all(item["exists"] for item in items.values())
    return {
        "schema": "aiijc-structured-decoder-fit-oracle-availability-v1",
        "status": "available-unverified" if available else "blocked-missing-fixed-fit-head",
        "fixed_fit_head": items,
        "available": available,
        "local16_substitution_allowed": False,
        "new_model_or_matcher_inference_run": False,
        "reference_labels_loaded": False,
    }


def _load_head(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, dict[str, Any], tuple[dict[str, Any], ...]]:
    archive, metadata, freeze = _resolve_head_paths(args)
    if not archive.is_file() or not metadata.is_file() or not freeze.is_file():
        raise FileNotFoundError("exact target-free fixed-5% reciprocal FIT head is unavailable")
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    frozen = json.loads(freeze.read_text(encoding="utf-8"))
    if meta.get("schema") != HEAD_SCHEMA:
        raise RuntimeError("fixed FIT head metadata schema changed")
    if meta.get("contains_target_slots_truth_or_reference_labels") is not False:
        raise RuntimeError("fixed FIT head metadata unexpectedly contains labels")
    expected_metadata = {
        "contains_pixels": False,
        "tile_id_space": "immutable-shuffled-tile-bag-identity",
        "candidate_identities_immutable": True,
        "fixed_fraction_per_axis_per_board": 0.05,
        "expected_requested_count_for_576_tiles": REQUESTED_PER_AXIS,
        "strict_target_free_loader_schema": STRICT_HEAD_LOADER_SCHEMA,
        "npz_members_materialised": TARGET_FREE_HEAD_INPUT_KEYS,
        "label_members_materialised": [],
    }
    if any(meta.get(key) != value for key, value in expected_metadata.items()):
        raise RuntimeError("fixed FIT head metadata contract changed")
    if frozen.get("schema") != HEAD_FREEZE_SCHEMA:
        raise RuntimeError("fixed FIT head pre-score schema changed")
    if frozen.get("created_before_fit_head_label_scoring") is not True:
        raise RuntimeError("fixed FIT head was not frozen before label scoring")
    if frozen.get("contains_target_slots_truth_or_reference_labels") is not False:
        raise RuntimeError("fixed FIT head freeze unexpectedly contains labels")
    if (
        frozen.get("strict_target_free_loader_schema")
        != STRICT_HEAD_LOADER_SCHEMA
        or frozen.get("label_cache_members_materialised") is not False
    ):
        raise RuntimeError("fixed FIT head did not use the strict target-free loader")
    if not isinstance(meta.get("config_sha256"), str) or (
        frozen.get("config_sha256") != meta["config_sha256"]
    ):
        raise RuntimeError("fixed FIT head metadata/freeze protocol mismatch")
    expected = frozen.get("artifacts", {})
    for key, path in (("archive", archive), ("metadata", metadata)):
        if key not in expected or expected[key].get("sha256") != sha256_file(path):
            raise RuntimeError(f"fixed FIT head {key} changed after pre-score freeze")
    required_provenance = {"fit_endpoint", "runner", "module"}
    if not required_provenance.issubset(expected):
        raise RuntimeError("fixed FIT head freeze omits required provenance")
    for key, artifact in expected.items():
        if key in {"archive", "metadata"}:
            continue
        if not isinstance(artifact, Mapping):
            raise RuntimeError(f"fixed FIT head freeze omits its {key} provenance")
        provenance_path = Path(str(artifact.get("path", "")))
        if not provenance_path.is_absolute():
            provenance_path = PROJECT_ROOT / provenance_path
        if (
            not provenance_path.is_file()
            or artifact.get("sha256") != sha256_file(provenance_path)
        ):
            raise RuntimeError(f"fixed FIT head {key} provenance changed")
    rows = tuple(meta.get("rows", ()))
    if len(rows) != 64:
        raise RuntimeError("fixed FIT head must contain the exact 64-case FIT roster")
    if [row.get("prefix") for row in rows] != [
        f"case_{index:04d}" for index in range(64)
    ]:
        raise RuntimeError("fixed FIT head case prefixes/order changed")
    forbidden = ("target_slots", "truth", "reference", "correct", "label")
    with np.load(archive, allow_pickle=False) as values:
        if any(any(token in key.lower() for token in forbidden) for key in values.files):
            raise RuntimeError("fixed FIT head archive contains a forbidden label field")
        expected_keys: set[str] = set()
        for row in rows:
            prefix = str(row["prefix"])
            union_key = f"{prefix}__union_identity_digest_ascii"
            expected_keys.add(union_key)
            union = values[union_key]
            if union.ndim != 1 or union.dtype != np.uint8:
                raise RuntimeError("fixed FIT head union digest encoding changed")
            try:
                union_digest = bytes(union).decode("ascii")
            except UnicodeDecodeError as error:
                raise RuntimeError("fixed FIT head union digest is not ASCII") from error
            if union_digest != row.get("union_identity_digest"):
                raise RuntimeError("fixed FIT head union identity digest mismatch")
            edges: list[DirectedEdge] = []
            for axis, name in enumerate(("right", "down")):
                keys = {
                    "sources": f"{prefix}__selected_sources__{name}",
                    "targets": f"{prefix}__selected_targets__{name}",
                    "confidence": f"{prefix}__selected_joint_confidences__{name}",
                    "requested": f"{prefix}__requested_count__{name}",
                    "reciprocal": f"{prefix}__reciprocal_count__{name}",
                }
                expected_keys.update(keys.values())
                sources = values[keys["sources"]]
                targets = values[keys["targets"]]
                confidence = values[keys["confidence"]]
                requested_value = values[keys["requested"]]
                reciprocal_value = values[keys["reciprocal"]]
                if (
                    not np.issubdtype(sources.dtype, np.integer)
                    or not np.issubdtype(targets.dtype, np.integer)
                    or not np.issubdtype(confidence.dtype, np.floating)
                    or requested_value.shape != ()
                    or not np.issubdtype(requested_value.dtype, np.integer)
                    or reciprocal_value.shape != ()
                    or not np.issubdtype(reciprocal_value.dtype, np.integer)
                ):
                    raise RuntimeError("fixed reciprocal head array dtypes changed")
                requested = int(requested_value)
                reciprocal_count = int(reciprocal_value)
                if requested != REQUESTED_PER_AXIS:
                    raise RuntimeError("fixed reciprocal fraction no longer requests 29 per axis")
                if (
                    sources.shape != (REQUESTED_PER_AXIS,)
                    or targets.shape != sources.shape
                    or confidence.shape != sources.shape
                ):
                    raise RuntimeError("fixed reciprocal head arrays are misaligned")
                if not REQUESTED_PER_AXIS <= reciprocal_count <= COUNT:
                    raise RuntimeError("fixed reciprocal head cannot support its selection")
                rank = list(
                    zip(
                        confidence.tolist(),
                        sources.tolist(),
                        targets.tolist(),
                        strict=True,
                    )
                )
                if rank != sorted(rank, key=lambda item: (-item[0], item[1], item[2])):
                    raise RuntimeError("fixed reciprocal head ranking order changed")
                edges.extend(
                    DirectedEdge(axis, int(source), int(target), float(score))
                    for source, target, score in zip(
                        sources, targets, confidence, strict=True
                    )
                )
            validate_fixed_reciprocal_head(
                edges, grid=GRID, requested_per_axis=REQUESTED_PER_AXIS
            )
        if set(values.files) != expected_keys:
            raise RuntimeError("fixed FIT head archive schema has missing or extra fields")
    return archive, metadata, freeze, meta, rows


def _verify_head_matches_config(
    config: Mapping[str, Any],
    archive: Path,
    metadata: Path,
    freeze: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    for key, path in (
        ("archive", archive),
        ("metadata", metadata),
        ("pre_score_freeze", freeze),
    ):
        if _record(path) != config["fixed_head"][key]:
            raise RuntimeError("runtime fixed head differs from signed preregistration")
    fields = (
        "prefix",
        "case_id",
        "source_filename",
        "draw_index",
        "dirty_sha256",
        "fit_cache",
        "union_identity_digest",
    )
    signed = config["panel"]["ordered_cases"]
    if len(rows) != len(signed) or any(
        any(observed.get(field) != expected.get(field) for field in fields)
        for observed, expected in zip(rows, signed, strict=True)
    ):
        raise RuntimeError("runtime fixed-head roster differs from signed cases")


def _tri_cache_inventory() -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    report = json.loads(TRI_REPORT.read_text(encoding="utf-8"))
    rows = report["fit_cache"]["rows"]
    lookup = {(str(row["source_filename"]), int(row["draw_index"])): row for row in rows}
    if len(rows) != 64 or len(lookup) != 64:
        raise RuntimeError("immutable tri-emitter FIT cache roster changed")
    for row in rows:
        path = PROJECT_ROOT / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError("immutable tri-emitter FIT cache bytes changed")
    return report, lookup


def _align_head_to_fit_cache(
    rows: Sequence[Mapping[str, Any]],
    cache_lookup: Mapping[tuple[str, int], Mapping[str, Any]],
) -> None:
    observed: set[tuple[str, int]] = set()
    for row in rows:
        key = (str(row["source_filename"]), int(row["draw_index"]))
        if key in observed or key not in cache_lookup:
            raise RuntimeError("fixed FIT head roster differs from immutable FIT caches")
        observed.add(key)
        cache = cache_lookup[key]
        for field in ("case_id", "dirty_sha256"):
            if row.get(field) != cache.get(field):
                raise RuntimeError(f"fixed FIT head/cache identity mismatch: {field}")
        fit_cache = row.get("fit_cache", {})
        if fit_cache.get("path") != cache.get("path"):
            raise RuntimeError("fixed FIT head/cache path mismatch")
        if fit_cache.get("sha256") != cache.get("sha256"):
            raise RuntimeError("fixed FIT head/cache SHA-256 mismatch")
    if observed != set(cache_lookup):
        raise RuntimeError("fixed FIT head does not cover the immutable FIT roster exactly")


def preregister(args: argparse.Namespace) -> dict[str, Any]:
    if args.config.exists() or Path(f"{args.config}.sha256").exists():
        raise FileExistsError("refusing to replace a structured-decoder preregistration")
    archive, metadata, freeze, head_meta, rows = _load_head(args)
    tri_report, cache_lookup = _tri_cache_inventory()
    _align_head_to_fit_cache(rows, cache_lookup)
    confirmed = dict(verify_taska_relation_selector_solver())
    no_repeat = [_record(PROJECT_ROOT / path) for path in NO_REPEAT_REPORTS]
    roster = [
        {
            "prefix": row["prefix"],
            "case_id": row["case_id"],
            "source_filename": row["source_filename"],
            "draw_index": row["draw_index"],
            "dirty_sha256": row["dirty_sha256"],
            "fit_cache": row["fit_cache"],
            "union_identity_digest": row["union_identity_digest"],
        }
        for row in rows
    ]
    config = {
        "schema": CONFIG_SCHEMA,
        "status": "signed-before-control-inference-or-exact-reference-scoring",
        "created_at": "2026-08-31",
        "purpose": (
            "One target-assisted FIT-only capacity audit for adding the exact fixed "
            "5% joint reciprocal head to the confirmed relation-selector layout."
        ),
        "protocol": {
            "organizer_train_fit_only": True,
            "exact_immutable_opened_fit_roster": True,
            "fixed_head_frozen_and_hashed_before_preregistration": True,
            "control_layouts_frozen_before_exact_reference_scoring": True,
            "local16_terminal16_dev_competition_test_or_submission_access": False,
            "target_assisted_capacity_only_never_deployable": True,
            "new_matcher_or_model_inference": (
                "only the strictly necessary confirmed six-arm control replay; "
                "the reciprocal head is never recomputed"
            ),
            "threshold_action_roster_or_tie_break_sweep": False,
        },
        "panel": {
            "source_count": 32,
            "draw_indices": [0, 1],
            "case_seed": FIT_CASE_SEED,
            "case_count": len(roster),
            "ordered_cases": roster,
        },
        "fixed_head": {
            "schema": head_meta["schema"],
            "fraction_per_axis_per_board": 0.05,
            "requested_per_axis_per_board": REQUESTED_PER_AXIS,
            "reciprocal_source_and_target_unique_per_axis": True,
            "archive": _record(archive),
            "metadata": _record(metadata),
            "pre_score_freeze": _record(freeze),
        },
        "control": {
            "solver": "confirmed TASKA relation-selector over six whole post-tail arms",
            "replay_device": "mps",
            "allow_nondeterministic_mps": True,
            "confirmed_sha256": confirmed,
            "all_576_original_upright_tiles_exactly_once": True,
            "output_pixels": False,
        },
        "oracle_action_contract": {
            "state": "current strict tile-at-position permutation",
            "proposal_roster": "only frozen fixed-head edges proven true after freeze",
            "action": (
                "form oracle rigid components from exact relations already realised "
                "in the state; translate either endpoint component by the edge-implied "
                "integer shift; directly displaced tiles bijectively fill vacated cells"
            ),
            "compatibility": (
                "source/target reciprocal uniqueness, distinct endpoint components, "
                "in-bounds rigid span, no internal collision, strict permutation, "
                "net realised supplied-true-edge gain > 0"
            ),
            "pair_safety": "each accepted edit has incremental satisfied-pair delta >= 0",
            "stop": "always available; return unchanged control if no accepted edit",
            "selection_order": [
                "maximum realised supplied-true-edge gain",
                "maximum satisfied-pair delta",
                "maximum exact delta",
                "maximum absolute-Manhattan improvement",
                "maximum radius2 gain",
                "maximum frozen joint confidence",
                "minimum moved-component size",
                "target-component move before source-component move",
                "axis/current source position/current target position/shift ascending",
            ],
            "termination": "stop when no admissible pair-safe true-edge edit remains",
            "nearby_action_or_budget_sweep": False,
        },
        "metrics_and_gate": {
            "primary_supply": "mean compatible missing true fixed-head edges per board",
            "primary_conversion": "mean pair-safe realised supplied true-edge gain per board",
            "pair_safety": "every ceiling board satisfied-pair delta >= 0",
            "diagnostics": [
                "exact tiles",
                "absolute mean Manhattan per tile",
                "absolute radius2 recall",
            ],
            "minimum_true_edge_headroom_per_board": MINIMUM_TRUE_EDGE_HEADROOM_PER_BOARD,
            "minimum_pair_safe_realisation_gain_per_board": (
                MINIMUM_PAIR_SAFE_REALISATION_GAIN_PER_BOARD
            ),
            "all_boards_nonnegative_pair_delta": True,
        },
        "no_repeat_audit": {
            "material_distinction": (
                "unlike all-edge union or priority/consensus arms, this starts from the "
                "confirmed whole layout, consumes only the frozen reciprocal head, "
                "models an explicit feasible strict edit and has STOP plus pair safety"
            ),
            "audited_reports": no_repeat,
        },
        "frozen_inputs": {
            "tri_fit_report": _record(TRI_REPORT),
            "head_archive": _record(archive),
            "head_metadata": _record(metadata),
            "head_pre_score_freeze": _record(freeze),
            "oracle_module": _record(MODULE_PATH),
            "runner": _record(Path(__file__)),
            "tests": _record(TEST_PATH),
        },
        "tri_fit_contract": {
            "fit_digest": tri_report["protocol"]["fit_digest"],
            "cache_case_count": len(cache_lookup),
        },
        "forbidden": {
            "local16": True,
            "terminal16": True,
            "competition_test": True,
            "submission": True,
            "denoised_or_generated_output_pixels": True,
            "weco_logging": True,
            "git_write": True,
        },
    }
    _write_json_exclusive(args.config, config)
    _write_sidecar(args.config)
    return {"config": _record(args.config), "case_count": len(roster), "status": "signed"}


def _load_signed_config(path: Path) -> tuple[dict[str, Any], str]:
    sidecar = Path(f"{path}.sha256")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("signed structured-decoder config is unavailable")
    digest = sha256_file(path)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise RuntimeError("structured-decoder config sidecar mismatch")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise RuntimeError("structured-decoder config schema changed")
    for artifact in config["frozen_inputs"].values():
        target = Path(artifact["path"])
        if not target.is_absolute():
            target = PROJECT_ROOT / target
        if not target.is_file() or sha256_file(target) != artifact["sha256"]:
            raise RuntimeError(f"signed structured-decoder input changed: {target}")
    if dict(verify_taska_relation_selector_solver()) != config["control"]["confirmed_sha256"]:
        raise RuntimeError("confirmed relation-selector solver bytes changed")
    return config, digest


def _fit_boards(args: argparse.Namespace, config: Mapping[str, Any]) -> list[Any]:
    protocol, boards, _, _ = prior.roster._load_protocol(args)
    signed_names = []
    for row in config["panel"]["ordered_cases"]:
        name = str(row["source_filename"])
        if name not in signed_names:
            signed_names.append(name)
    if signed_names != protocol["fit_filenames"]:
        raise RuntimeError("signed structured-decoder source order changed")
    return boards


def run_freeze_controls(args: argparse.Namespace) -> dict[str, Any]:
    config, config_sha = _load_signed_config(args.config)
    if args.device != config["control"]["replay_device"]:
        raise RuntimeError("control replay device differs from signed config")
    head_archive, head_metadata, head_freeze, _, head_rows = _load_head(args)
    _verify_head_matches_config(
        config, head_archive, head_metadata, head_freeze, head_rows
    )
    signed_rows = config["panel"]["ordered_cases"]
    if list(head_rows) and [row["prefix"] for row in head_rows] != [
        row["prefix"] for row in signed_rows
    ]:
        raise RuntimeError("fixed head row order changed after preregistration")
    boards = _fit_boards(args, config)
    board_by_name = {board.filename: board for board in boards}
    resources = load_taska_relation_selector_resources(device=args.device)
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    started = perf_counter()
    for index, signed in enumerate(signed_rows):
        board = board_by_name[str(signed["source_filename"])]
        item, _ = make_exact_synthetic_case(
            board.tiles,
            source_filename=board.filename,
            draw_index=int(signed["draw_index"]),
            seed=FIT_CASE_SEED,
        )
        observed = {
            "case_id": item.case_id,
            "source_filename": item.source_filename,
            "draw_index": item.draw_index,
            "dirty_sha256": hashlib.sha256(item.tiles.tobytes()).hexdigest(),
        }
        if any(signed[key] != value for key, value in observed.items()):
            raise RuntimeError("control reconstruction differs from signed FIT case")
        result = solve_taska_relation_selector_pipeline(
            item.tiles,
            resources,
            focal_chunk_size=args.focal_chunk_size,
            denoiser_batch_size=args.denoiser_batch_size,
        )
        layout = strict_layout(result.layout, grid=GRID, name="control_layout")
        prefix = str(signed["prefix"])
        arrays[f"{prefix}__control_layout"] = layout
        rows.append(
            {
                **observed,
                "prefix": prefix,
                "layout_sha256": result.layout_sha256,
                "selected_arm": result.selected_arm,
                "control_arm": result.control_arm,
                "strict_original_upright_permutation": True,
            }
        )
        print(
            json.dumps(
                {
                    "event": "freeze_structured_control",
                    "case": index + 1,
                    "count": len(signed_rows),
                    "source": board.filename,
                    "draw": item.draw_index,
                }
            ),
            flush=True,
        )
    output = args.output_dir.resolve()
    archive = output / "frozen-target-free-controls.npz"
    metadata = output / "frozen-target-free-controls.json"
    freeze = output / "pre-score-freeze.json"
    _write_npz_exclusive(archive, arrays)
    _write_json_exclusive(
        metadata,
        {
            "schema": CONTROL_SCHEMA,
            "config_sha256": config_sha,
            "contains_exact_references_or_labels": False,
            "contains_output_pixels": False,
            "all_layouts_strict_original_upright_permutations": True,
            "rows": rows,
        },
    )
    _write_json_exclusive(
        freeze,
        {
            "schema": CONTROL_FREEZE_SCHEMA,
            "created_before_exact_reference_scoring": True,
            "contains_exact_references_or_labels": False,
            "config_sha256": config_sha,
            "artifacts": {
                "archive": _record(archive),
                "metadata": _record(metadata),
                "head_archive": config["fixed_head"]["archive"],
                "head_metadata": config["fixed_head"]["metadata"],
                "config": _record(args.config),
            },
        },
    )
    return {
        "schema": "aiijc-structured-decoder-control-freeze-result-v1",
        "status": "controls-frozen-reference-scoring-not-run",
        "case_count": len(rows),
        "runtime_seconds": perf_counter() - started,
        "archive": _record(archive),
        "metadata": _record(metadata),
        "pre_score_freeze": _record(freeze),
        "local16_or_terminal16_opened": False,
        "competition_test_accessed": False,
    }


def _verify_controls(
    args: argparse.Namespace, config_sha: str
) -> tuple[Path, tuple[dict[str, Any], ...]]:
    output = args.output_dir.resolve()
    archive = output / "frozen-target-free-controls.npz"
    metadata = output / "frozen-target-free-controls.json"
    freeze_path = output / "pre-score-freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema") != CONTROL_FREEZE_SCHEMA:
        raise RuntimeError("control pre-score freeze schema changed")
    if freeze.get("created_before_exact_reference_scoring") is not True:
        raise RuntimeError("controls were not frozen before exact reference scoring")
    if freeze.get("config_sha256") != config_sha:
        raise RuntimeError("control freeze belongs to another signed config")
    for key, path in (("archive", archive), ("metadata", metadata)):
        if not path.is_file() or sha256_file(path) != freeze["artifacts"][key]["sha256"]:
            raise RuntimeError(f"frozen control {key} changed before scoring")
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    if payload.get("schema") != CONTROL_SCHEMA:
        raise RuntimeError("control metadata schema changed")
    if payload.get("contains_exact_references_or_labels") is not False:
        raise RuntimeError("control metadata unexpectedly contains references")
    return archive, tuple(payload["rows"])


def _head_edges(archive: Any, prefix: str) -> tuple[DirectedEdge, ...]:
    edges = []
    for axis, name in enumerate(("right", "down")):
        sources = archive[f"{prefix}__selected_sources__{name}"]
        targets = archive[f"{prefix}__selected_targets__{name}"]
        confidence = archive[f"{prefix}__selected_joint_confidences__{name}"]
        edges.extend(
            DirectedEdge(axis, int(source), int(target), float(score))
            for source, target, score in zip(sources, targets, confidence, strict=True)
        )
    return validate_fixed_reciprocal_head(
        edges, grid=GRID, requested_per_axis=REQUESTED_PER_AXIS
    )


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    config, config_sha = _load_signed_config(args.config)
    head_path, head_metadata, head_freeze, _, head_rows = _load_head(args)
    _verify_head_matches_config(
        config, head_path, head_metadata, head_freeze, head_rows
    )
    control_path, control_rows = _verify_controls(args, config_sha)
    signed_rows = config["panel"]["ordered_cases"]
    prefixes = [str(row["prefix"]) for row in signed_rows]
    if [row["prefix"] for row in head_rows] != prefixes:
        raise RuntimeError("head row order changed before scoring")
    if [row["prefix"] for row in control_rows] != prefixes:
        raise RuntimeError("control row order changed before scoring")
    boards = _fit_boards(args, config)
    board_by_name = {board.filename: board for board in boards}
    result_rows: list[dict[str, Any]] = []
    strict_count = 0
    with (
        np.load(head_path, allow_pickle=False) as head_archive,
        np.load(control_path, allow_pickle=False) as control_archive,
    ):
        for signed in signed_rows:
            prefix = str(signed["prefix"])
            board = board_by_name[str(signed["source_filename"])]
            item, reference = make_exact_synthetic_case(
                board.tiles,
                source_filename=board.filename,
                draw_index=int(signed["draw_index"]),
                seed=FIT_CASE_SEED,
            )
            dirty_sha = hashlib.sha256(item.tiles.tobytes()).hexdigest()
            if item.case_id != signed["case_id"] or dirty_sha != signed["dirty_sha256"]:
                raise RuntimeError("FIT reference reconstruction differs from signed case")
            control = strict_layout(
                control_archive[f"{prefix}__control_layout"],
                grid=GRID,
                name="control_layout",
            )
            oracle = evaluate_pair_safe_oracle(
                control,
                reference.tile_at_position,
                _head_edges(head_archive, prefix),
                grid=GRID,
                requested_per_axis=REQUESTED_PER_AXIS,
            )
            strict_layout(oracle.ceiling_layout, grid=GRID, name="ceiling_layout")
            strict_count += 1
            result_rows.append(
                {
                    "prefix": prefix,
                    "case_id": item.case_id,
                    "source_filename": item.source_filename,
                    "draw_index": item.draw_index,
                    **oracle.as_dict(),
                    "ceiling_layout_sha256": hashlib.sha256(
                        oracle.ceiling_layout.astype("<i4", copy=False).tobytes()
                    ).hexdigest(),
                    "strict_original_upright_permutation": True,
                }
            )
    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in result_rows]))

    mean_headroom = mean("compatible_missing_true_edge_headroom")
    mean_realised = mean("realised_supplied_true_edge_gain")
    pair_deltas = [int(row["delta"]["satisfied_pairs"]) for row in result_rows]
    exact_deltas = [int(row["delta"]["exact_tiles"]) for row in result_rows]
    manhattan_deltas = [
        float(row["delta"]["mean_absolute_manhattan"]) for row in result_rows
    ]
    radius2_deltas = [float(row["delta"]["radius2_recall"]) for row in result_rows]
    control_metrics = {
        key: float(np.mean([row["control"][key] for row in result_rows]))
        for key in (
            "satisfied_pairs",
            "exact_tiles",
            "mean_absolute_manhattan",
            "radius2_recall",
        )
    }
    ceiling_metrics = {
        key: float(np.mean([row["ceiling"][key] for row in result_rows]))
        for key in (
            "satisfied_pairs",
            "exact_tiles",
            "mean_absolute_manhattan",
            "radius2_recall",
        )
    }
    gate = {
        "mean_compatible_missing_true_edge_headroom": mean_headroom,
        "minimum_true_edge_headroom_per_board": MINIMUM_TRUE_EDGE_HEADROOM_PER_BOARD,
        "headroom_passed": mean_headroom >= MINIMUM_TRUE_EDGE_HEADROOM_PER_BOARD,
        "mean_pair_safe_realised_supplied_true_edge_gain": mean_realised,
        "minimum_pair_safe_realisation_gain_per_board": (
            MINIMUM_PAIR_SAFE_REALISATION_GAIN_PER_BOARD
        ),
        "pair_safe_realisation_passed": (
            mean_realised >= MINIMUM_PAIR_SAFE_REALISATION_GAIN_PER_BOARD
        ),
        "all_board_pair_deltas_nonnegative": all(value >= 0 for value in pair_deltas),
    }
    gate["passed"] = bool(
        gate["headroom_passed"]
        and gate["pair_safe_realisation_passed"]
        and gate["all_board_pair_deltas_nonnegative"]
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": (
            "pass-capacity-only-model-still-forbidden"
            if gate["passed"]
            else "fail-stop-before-structured-model"
        ),
        "scope": "target-assisted organizer-train FIT capacity only; never deployable",
        "config": _record(args.config),
        "case_count": len(result_rows),
        "gate": gate,
        "summary": {
            "mean_control": control_metrics,
            "mean_pair_safe_ceiling": ceiling_metrics,
            "mean_selected_edges": mean("selected_edge_count"),
            "mean_selected_true_edges": mean("selected_true_edge_count"),
            "mean_initially_realised_selected_true_edges": mean(
                "initially_realised_selected_true_edges"
            ),
            "mean_missing_selected_true_edges": mean(
                "missing_selected_true_edge_count"
            ),
            "mean_initial_pair_safe_action_count": mean(
                "initial_pair_safe_action_count"
            ),
            "mean_accepted_action_count": mean("accepted_action_count"),
            "mean_pair_delta": float(np.mean(pair_deltas)),
            "minimum_pair_delta": min(pair_deltas),
            "mean_exact_delta": float(np.mean(exact_deltas)),
            "mean_absolute_manhattan_delta": float(np.mean(manhattan_deltas)),
            "mean_radius2_delta": float(np.mean(radius2_deltas)),
        },
        "rows": result_rows,
        "legality": {
            "strict_control_and_ceiling_layout_count": strict_count,
            "all_576_original_upright_tiles_exactly_once": strict_count == len(result_rows),
            "output_pixels": False,
            "local16_or_terminal16_opened": False,
            "competition_test_accessed": False,
            "submission_or_production_modified": False,
            "weco_logged": False,
        },
        "artifacts": {
            "head_archive": _record(head_path),
            "control_archive": _record(control_path),
            "module": _record(MODULE_PATH),
            "runner": _record(Path(__file__)),
            "tests": _record(TEST_PATH),
        },
    }
    _write_json_exclusive(args.output_dir.resolve() / "report.json", report)
    return report


def validate_only(args: argparse.Namespace) -> dict[str, Any]:
    config, config_sha = _load_signed_config(args.config)
    head_path, head_metadata, head_freeze, _, rows = _load_head(args)
    _verify_head_matches_config(
        config, head_path, head_metadata, head_freeze, rows
    )
    _tri_report, lookup = _tri_cache_inventory()
    _align_head_to_fit_cache(rows, lookup)
    controls: dict[str, Any]
    try:
        control_path, control_rows = _verify_controls(args, config_sha)
        controls = {
            "available": True,
            "archive": _record(control_path),
            "case_count": len(control_rows),
        }
    except FileNotFoundError:
        controls = {"available": False}
    return {
        "schema": "aiijc-structured-decoder-fit-oracle-validation-v1",
        "status": "valid",
        "config": _record(args.config),
        "fixed_head_archive": _record(head_path),
        "fixed_head_case_count": len(rows),
        "controls": controls,
        "reference_labels_loaded": False,
        "local16_or_terminal16_opened": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode in ("preregister", "freeze-controls"):
        _validate_args(args)
    if args.mode == "availability":
        result = availability(args)
    elif args.mode == "preregister":
        result = preregister(args)
    elif args.mode == "freeze-controls":
        result = run_freeze_controls(args)
    elif args.mode == "score":
        result = run_score(args)
    else:
        result = validate_only(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.mode == "availability" and not result["available"]:
        return 2
    if args.mode == "score" and not result["gate"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
