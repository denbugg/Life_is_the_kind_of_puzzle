#!/usr/bin/env python3
"""Materialise only the signed scale256 joint-reciprocal FIT cache.

This runner has one deliberately narrow transition: a reviewed, signed metadata
contract becomes 256 organizer-train sources x two exact synthetic draws.  It
cannot train a verifier, open the reserved DEV roster, score a reference, load a
capacity checkpoint, or touch terminal/competition-test data.

The output report keeps the legacy tri-emitter report schema because the frozen
joint-reciprocal trainer already validates and consumes that cache schema.  A
producer-specific schema and explicit scope fields distinguish this cache-only
artifact from the historical train/evaluate runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file
from aiijc_puzzle.synthetic_socket_evaluation import (
    make_exact_synthetic_case,
    names_digest,
    select_source_disjoint_train_records,
)
from aiijc_puzzle.tri_emitter_edge_verifier import (
    TOP_K,
    CandidatePool,
    candidate_pool_digest,
)

try:
    from scripts import run_fullres_boundary_denoiser as boundary
    from scripts import run_tri_emitter_edge_verifier as prior
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import run_fullres_boundary_denoiser as boundary
    import run_tri_emitter_edge_verifier as prior


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/joint_reciprocal_scale256_fit_cache_unsigned_template_v1.json"
)
DEFAULT_MANIFEST = prior.roster.DEFAULT_MANIFEST
DEFAULT_TARGETS = prior.roster.DEFAULT_TARGETS
DEFAULT_SOCKET_CHECKPOINT = prior.SOCKET_CHECKPOINT

CONFIG_SCHEMA = "aiijc-joint-reciprocal-scale-fit-cache-materialization-v1"
PRODUCER_SCHEMA = "aiijc-joint-reciprocal-scale-fit-cache-v1"
COMPATIBLE_REPORT_SCHEMA = "aiijc-tri-emitter-edge-verifier-report-v1"
SIGNED_STATUS = "signed-fixed-protocol"
BLOCKED_STATUS = "unsigned-template-blocked-awaiting-roster-and-final-review"

FIT_SOURCE_COUNT = 256
RESERVED_DEV_SOURCE_COUNT = 64
FIT_DRAWS = (0, 1)
FIT_CASE_COUNT = FIT_SOURCE_COUNT * len(FIT_DRAWS)
FIT_CASE_SEED = 20260914
SELECTION_SEED = 20260913
SELECTION_NAMESPACE = "aiijc-joint-reciprocal-scale256-fit256-dev64-v1"
GRID = 24
TILE_COUNT = GRID * GRID
CANDIDATE_WIDTH = 3 * TOP_K
PARENT_EXCLUSION_COUNT = 1120
PARENT_EXCLUSION_DIGEST = (
    "d93311aa39c3c4ccc349928a3e6269103f540affde585e889e3980d2f21227e2"
)

CACHE_KEYS = (
    "raw_sides",
    "dino_sides",
    "candidates",
    "valid",
    "auxiliary",
    "raw_baseline",
    "emitter_topk",
    "target_slots",
)
REQUIRED_FROZEN_INPUTS = frozenset(
    {
        "manifest",
        "socket_checkpoint",
        "adapter1600_checkpoint",
        "dino_checkpoint",
        "socket_parent_report",
        "adapter_parent_report",
        "feature_runner",
        "synthetic_helper",
        "board_loader",
        "materializer_runner",
    }
)
EXPECTED_FROZEN_PATHS = {
    "manifest": "data/interim/validation_manifest.json",
    "socket_checkpoint": (
        "outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/socket_matcher.pt"
    ),
    "adapter1600_checkpoint": (
        "outputs/fullres-retrieval-adapter/scale1600-local16-v1/adapter_step1600.pt"
    ),
    "dino_checkpoint": (
        "artifacts/foundation-semantics/dinov2-vits14-official/"
        "dinov2_vits14_pretrain.pth"
    ),
    "socket_parent_report": (
        "outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/report.json"
    ),
    "adapter_parent_report": (
        "outputs/fullres-retrieval-adapter/scale1600-local16-v1/report.json"
    ),
    "feature_runner": "scripts/run_tri_emitter_edge_verifier.py",
    "synthetic_helper": "src/aiijc_puzzle/synthetic_socket_evaluation.py",
    "board_loader": "scripts/run_fullres_boundary_denoiser.py",
    "materializer_runner": "scripts/materialize_joint_reciprocal_scale_fit_cache.py",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument(
        "--socket-checkpoint", type=Path, default=DEFAULT_SOCKET_CHECKPOINT
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    return parser.parse_args(argv)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _path_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": _path_label(resolved), "sha256": sha256_file(resolved)}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _filename_list(
    value: Any,
    *,
    field: str,
    expected_count: int,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise RuntimeError(f"{field} must be a list of non-empty filenames")
    result = tuple(value)
    if len(result) != expected_count:
        raise RuntimeError(
            f"{field} must contain exactly {expected_count} filenames, got {len(result)}"
        )
    if len(set(result)) != len(result):
        raise RuntimeError(f"{field} contains duplicate filenames")
    return result


def require_exact_contract(config: Mapping[str, Any]) -> None:
    """Validate the fixed cache-only protocol without touching any data pixels."""

    if config.get("schema") != CONFIG_SCHEMA:
        raise RuntimeError("scale FIT cache config schema changed")
    source = config.get("source_protocol")
    if not isinstance(source, Mapping):
        raise RuntimeError("scale FIT cache config has no source protocol")
    allowed_source_keys = {
        "selection_namespace",
        "selection_seed",
        "fit_source_count",
        "fit_filenames",
        "fit_digest",
        "fit_draw_indices",
        "fit_case_seed",
        "fit_case_count",
        "reserved_dev_source_count",
        "reserved_dev_filenames",
        "reserved_dev_digest",
        "parent_exclusion_count",
        "parent_exclusion_digest",
    }
    if set(source) != allowed_source_keys:
        raise RuntimeError("scale FIT source protocol fields changed")
    expected_scalars = {
        "selection_namespace": SELECTION_NAMESPACE,
        "selection_seed": SELECTION_SEED,
        "fit_source_count": FIT_SOURCE_COUNT,
        "fit_case_seed": FIT_CASE_SEED,
        "fit_case_count": FIT_CASE_COUNT,
        "reserved_dev_source_count": RESERVED_DEV_SOURCE_COUNT,
        "parent_exclusion_count": PARENT_EXCLUSION_COUNT,
        "parent_exclusion_digest": PARENT_EXCLUSION_DIGEST,
    }
    for key, expected in expected_scalars.items():
        if source.get(key) != expected:
            raise RuntimeError(f"fixed scale FIT source field changed: {key}")
    if source.get("fit_draw_indices") != list(FIT_DRAWS):
        raise RuntimeError("scale FIT draw indices changed")
    fit = _filename_list(
        source.get("fit_filenames"),
        field="fit_filenames",
        expected_count=FIT_SOURCE_COUNT,
    )
    reserved_dev = _filename_list(
        source.get("reserved_dev_filenames"),
        field="reserved_dev_filenames",
        expected_count=RESERVED_DEV_SOURCE_COUNT,
    )
    if names_digest(fit) != source.get("fit_digest"):
        raise RuntimeError("scale FIT roster digest mismatch")
    if names_digest(reserved_dev) != source.get("reserved_dev_digest"):
        raise RuntimeError("reserved DEV roster digest mismatch")
    if set(fit) & set(reserved_dev):
        raise RuntimeError("scale FIT and reserved DEV rosters overlap")

    cache = config.get("cache_contract", {})
    expected_cache = {
        "grid": GRID,
        "tile_count": TILE_COUNT,
        "candidate_roster": "raw+adapter1600+DINO-top32-stable-union",
        "candidate_width": CANDIDATE_WIDTH,
        "raw_top32_preserved": True,
        "cache_keys": list(CACHE_KEYS),
        "encoding": "numpy-savez-uncompressed",
        "compatible_report_schema": COMPATIBLE_REPORT_SCHEMA,
        "producer_schema": PRODUCER_SCHEMA,
    }
    if cache != expected_cache:
        raise RuntimeError("scale FIT cache schema/identity contract changed")

    legality = config.get("legality", {})
    required_legality = {
        "organizer_train_only": True,
        "known_synthetic_permutation_labels_only": True,
        "cache_contains_pixels": False,
        "original_upright_tile_identities_only": True,
        "adapter_matcher_view_not_persisted": True,
        "reserved_dev_pixels_or_labels_opened": False,
        "terminal16_opened": False,
        "competition_test_accessed": False,
        "training_or_scoring_in_materializer": False,
    }
    if legality != required_legality:
        raise RuntimeError("scale FIT cache legality contract changed")

    frozen = config.get("frozen_inputs", {})
    if not isinstance(frozen, Mapping) or set(frozen) != REQUIRED_FROZEN_INPUTS:
        raise RuntimeError("scale FIT cache frozen input inventory changed")
    for name, artifact in frozen.items():
        if not isinstance(artifact, Mapping) or set(artifact) != {"path", "sha256"}:
            raise RuntimeError(f"invalid frozen input record: {name}")
        if _project_path(artifact["path"]) != _project_path(EXPECTED_FROZEN_PATHS[name]):
            raise RuntimeError(f"frozen scale FIT input path changed: {name}")

    if config.get("organizer_targets_directory") != "data/raw/train/targets":
        raise RuntimeError("organizer train target directory contract changed")

    output = config.get("output_contract", {})
    if output != {
        "exclusive_output_directory": True,
        "partial_run_has_no_report": True,
        "fit_cache_subdirectory": "fit-cache",
        "report_filename": "report.json",
        "endpoint_written": False,
        "dev_archive_written": False,
        "score_written": False,
    }:
        raise RuntimeError("scale FIT cache output contract changed")


def load_signed_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"scale FIT cache config is missing: {resolved}")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if config.get("status") == BLOCKED_STATUS:
        raise RuntimeError(
            "scale FIT cache template is intentionally blocked until exact FIT256/DEV64 "
            "rosters are reviewed and a separate signed config is created"
        )
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if config.get("status") != SIGNED_STATUS or not sidecar.is_file():
        raise RuntimeError("scale FIT cache config is not signed/fixed")
    digest = sha256_file(resolved)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise RuntimeError("scale FIT cache config sidecar mismatch")
    require_exact_contract(config)
    for name, artifact in config["frozen_inputs"].items():
        target = _project_path(artifact["path"])
        if not target.is_file() or sha256_file(target) != artifact["sha256"]:
            raise RuntimeError(f"frozen scale FIT input changed: {name} ({target})")
    return config, digest


def _parent_source_groups(
    socket_report: Mapping[str, Any],
    adapter_report: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    socket = socket_report.get("selection", {})
    adapter = adapter_report.get("protocol", {})
    specs = {
        "socket_train": (socket.get("train_filenames"), 1024),
        "socket_opened_eval": (socket.get("eval_filenames"), 32),
        "adapter_fit": (adapter.get("fit_filenames"), 32),
        "adapter_opened_local": (adapter.get("local_filenames"), 16),
        "adapter_terminal_owned": (adapter.get("terminal_filenames"), 16),
    }
    return {
        name: _filename_list(value, field=name, expected_count=count)
        for name, (value, count) in specs.items()
    }


def validate_metadata_rosters(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    socket_report: Mapping[str, Any],
    adapter_report: Mapping[str, Any],
) -> tuple[tuple[Mapping[str, Any], ...], dict[str, Any]]:
    """Recompute the fixed selection and return FIT records without opening pixels."""

    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise RuntimeError("validation manifest protocol digest changed")
    source = config["source_protocol"]
    groups = _parent_source_groups(socket_report, adapter_report)
    excluded = set().union(*(set(value) for value in groups.values()))
    if len(excluded) != int(source.get("parent_exclusion_count", -1)):
        raise RuntimeError("parent lineage exclusion count changed")
    if names_digest(tuple(excluded), sort_names=True) != source.get(
        "parent_exclusion_digest"
    ):
        raise RuntimeError("parent lineage exclusion digest changed")

    selected = select_source_disjoint_train_records(
        manifest,
        excluded_filenames=tuple(sorted(excluded)),
        limit=FIT_SOURCE_COUNT + RESERVED_DEV_SOURCE_COUNT,
        seed=SELECTION_SEED,
        namespace=SELECTION_NAMESPACE,
    )
    selected_names = tuple(str(record["filename"]) for record in selected)
    expected_fit = selected_names[:FIT_SOURCE_COUNT]
    expected_dev = selected_names[FIT_SOURCE_COUNT:]
    fit = tuple(source["fit_filenames"])
    reserved_dev = tuple(source["reserved_dev_filenames"])
    if fit != expected_fit or reserved_dev != expected_dev:
        raise RuntimeError("signed scale FIT256/DEV64 roster is not the fixed selection")
    if set(fit) & excluded or set(reserved_dev) & excluded:
        raise RuntimeError("scale roster overlaps a relevant parent lineage")
    for record in selected:
        digest = record.get("target_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError("selected manifest record has no valid target SHA-256")
    return tuple(selected[:FIT_SOURCE_COUNT]), {
        "selection_namespace": SELECTION_NAMESPACE,
        "selection_seed": SELECTION_SEED,
        "parent_exclusion_count": len(excluded),
        "parent_exclusion_digest": names_digest(tuple(excluded), sort_names=True),
        "parent_source_groups": {
            name: {
                "count": len(values),
                "digest": names_digest(values),
            }
            for name, values in groups.items()
        },
        "reserved_dev_source_count": len(reserved_dev),
        "reserved_dev_digest": names_digest(reserved_dev),
        "reserved_dev_opened": False,
    }


def _validate_runtime_paths(config: Mapping[str, Any], args: argparse.Namespace) -> None:
    frozen = config["frozen_inputs"]
    exact = {
        "manifest": args.manifest,
        "socket_checkpoint": args.socket_checkpoint,
        "adapter1600_checkpoint": prior.ADAPTER_CHECKPOINT,
        "dino_checkpoint": prior.DINO_CHECKPOINT,
        "feature_runner": Path(prior.__file__),
        "synthetic_helper": PROJECT_ROOT
        / "src/aiijc_puzzle/synthetic_socket_evaluation.py",
        "board_loader": Path(boundary.__file__),
        "materializer_runner": Path(__file__),
    }
    for name, supplied in exact.items():
        if supplied.resolve() != _project_path(frozen[name]["path"]):
            raise RuntimeError(f"runtime path differs from signed scale FIT input: {name}")
    expected_targets = _project_path(config["organizer_targets_directory"])
    if args.targets.resolve() != expected_targets or not expected_targets.is_dir():
        raise RuntimeError("organizer train target directory differs from signed config")


def _validate_case_arrays(values: Mapping[str, np.ndarray], targets: np.ndarray) -> None:
    expected_shapes = {
        "raw_sides": (4, TILE_COUNT, 20, 6),
        "dino_sides": (4, TILE_COUNT, 14, 16),
        "candidates": (2, TILE_COUNT, CANDIDATE_WIDTH),
        "valid": (2, TILE_COUNT, CANDIDATE_WIDTH),
        "auxiliary": (2, TILE_COUNT, CANDIDATE_WIDTH, 19),
        "raw_baseline": (2, TILE_COUNT, CANDIDATE_WIDTH),
        "emitter_topk": (3, 2, TILE_COUNT, TOP_K),
    }
    for key, shape in expected_shapes.items():
        if key not in values or np.asarray(values[key]).shape != shape:
            raise RuntimeError(f"generated scale FIT array shape changed: {key}")
    candidates = np.asarray(values["candidates"])
    valid = np.asarray(values["valid"])
    if valid.dtype != np.bool_:
        raise RuntimeError("generated scale FIT valid mask is not boolean")
    if np.any(valid & ((candidates < 0) | (candidates >= TILE_COUNT))):
        raise RuntimeError("generated scale FIT candidate identity is out of range")
    slots = np.asarray(targets)
    if slots.shape != (2, TILE_COUNT) or np.any(
        (slots < -1) | (slots >= CANDIDATE_WIDTH)
    ):
        raise RuntimeError("generated scale FIT target slot is invalid")
    present = slots >= 0
    axes, sources = np.nonzero(present)
    if len(axes) and not valid[axes, sources, slots[present]].all():
        raise RuntimeError("generated scale FIT target points to an invalid candidate")
    for key in ("raw_sides", "dino_sides", "auxiliary", "raw_baseline"):
        if not np.isfinite(np.asarray(values[key])).all():
            raise RuntimeError(f"generated scale FIT array is non-finite: {key}")


def _load_one_board(record: Mapping[str, Any], targets_dir: Path) -> Any:
    return boundary._prepare_boards((record,), targets_dir)[0]


def _materialize_one_case(
    *,
    board: Any,
    draw_index: int,
    cache_path: Path,
    socket: Any,
    adapter: Any,
    dino: torch.nn.Module,
    projection: np.ndarray,
    device: torch.device,
    executor: ThreadPoolExecutor,
) -> dict[str, Any]:
    started = perf_counter()
    item, reference = make_exact_synthetic_case(
        board.tiles,
        source_filename=board.filename,
        draw_index=draw_index,
        seed=FIT_CASE_SEED,
    )
    values, feature_runtime = prior._extract_case(
        item,
        socket=socket,
        adapter=adapter,
        dino=dino,
        projection=projection,
        device=device,
        executor=executor,
    )
    observed_identity = candidate_pool_digest(
        values["candidates"], values["valid"], values["emitter_topk"]
    )
    if observed_identity != feature_runtime.get("union_identity_digest"):
        raise RuntimeError("generated candidate identity digest changed in flight")
    pool = CandidatePool(
        candidates=values["candidates"],
        valid=values["valid"],
        auxiliary=values["auxiliary"],
        raw_baseline=values["raw_baseline"],
        emitter_topk=values["emitter_topk"],
        identity_digest=observed_identity,
    )
    target_slots = prior._target_slots(pool, reference.tile_at_position)
    _validate_case_arrays(values, target_slots)
    write_started = perf_counter()
    prior._fit_cache_case(cache_path, values, target_slots)
    write_seconds = perf_counter() - write_started
    return {
        "path": _path_label(cache_path),
        "sha256": sha256_file(cache_path),
        "source_filename": board.filename,
        "draw_index": draw_index,
        "case_id": item.case_id,
        "dirty_sha256": hashlib.sha256(item.tiles.tobytes()).hexdigest(),
        "eligible_queries": int(np.count_nonzero(target_slots >= 0)),
        "candidate_union_identity_digest": observed_identity,
        "cache_bytes": cache_path.stat().st_size,
        "runtime": {
            **feature_runtime,
            "cache_write_seconds": write_seconds,
            "materializer_case_seconds": perf_counter() - started,
        },
    }


def run_materialization(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    config_sha: str,
) -> dict[str, Any]:
    """Create cache files and a report, with no callable next experiment stage."""

    _validate_runtime_paths(config, args)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    socket_report_path = _project_path(
        config["frozen_inputs"]["socket_parent_report"]["path"]
    )
    adapter_report_path = _project_path(
        config["frozen_inputs"]["adapter_parent_report"]["path"]
    )
    socket_report = json.loads(socket_report_path.read_text(encoding="utf-8"))
    adapter_report = json.loads(adapter_report_path.read_text(encoding="utf-8"))
    fit_records, selection_metadata = validate_metadata_rosters(
        config, manifest, socket_report, adapter_report
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache_dir = output / "fit-cache"
    cache_dir.mkdir(exist_ok=False)
    started = perf_counter()
    model_started = perf_counter()
    socket, adapter, dino, projection, device = prior._make_models(args)
    model_load_seconds = perf_counter() - model_started
    rows: list[dict[str, Any]] = []
    source_load_seconds = 0.0
    with ThreadPoolExecutor(max_workers=1) as executor:
        for source_index, record in enumerate(fit_records):
            source_started = perf_counter()
            board = _load_one_board(record, args.targets.resolve())
            source_load_seconds += perf_counter() - source_started
            for draw_index in FIT_DRAWS:
                cache_path = (
                    cache_dir / f"source_{source_index:03d}_draw_{draw_index}.npz"
                )
                row = _materialize_one_case(
                    board=board,
                    draw_index=draw_index,
                    cache_path=cache_path,
                    socket=socket,
                    adapter=adapter,
                    dino=dino,
                    projection=projection,
                    device=device,
                    executor=executor,
                )
                row["source_index"] = source_index
                row["source_target_sha256"] = str(record["target_sha256"])
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "event": "scale_fit_cache",
                            "case": len(rows),
                            "count": FIT_CASE_COUNT,
                            "source": board.filename,
                            "draw": draw_index,
                            "seconds": row["runtime"]["materializer_case_seconds"],
                        }
                    ),
                    flush=True,
                )
    if len(rows) != FIT_CASE_COUNT:
        raise RuntimeError("scale FIT cache materializer produced the wrong case count")

    fit = tuple(config["source_protocol"]["fit_filenames"])
    adapter_protocol = adapter_report["protocol"]
    report = {
        "schema": COMPATIBLE_REPORT_SCHEMA,
        "producer_schema": PRODUCER_SCHEMA,
        "status": "complete-cache-only-ready-for-separate-fit-preregistration",
        "config": {
            "path": _path_label(args.config),
            "sha256": config_sha,
        },
        "protocol": {
            "manifest_digest": compute_protocol_digest(manifest),
            "fit_filenames": list(fit),
            "fit_digest": names_digest(fit),
            "local_filenames": list(adapter_protocol["local_filenames"]),
            "local_digest": names_digest(adapter_protocol["local_filenames"]),
            "terminal_filenames": list(adapter_protocol["terminal_filenames"]),
            "terminal_digest": names_digest(adapter_protocol["terminal_filenames"]),
            "reserved_dev_source_count": RESERVED_DEV_SOURCE_COUNT,
            "reserved_dev_digest": config["source_protocol"]["reserved_dev_digest"],
            "reserved_dev_opened": False,
        },
        "selection_audit": selection_metadata,
        "fit_cache": {
            "case_count": len(rows),
            "draws": list(FIT_DRAWS),
            "rows": rows,
        },
        "runtime": {
            "device": str(device),
            "model_load_seconds": model_load_seconds,
            "source_load_seconds": source_load_seconds,
            "total_seconds": perf_counter() - started,
            "mean_feature_seconds": float(
                np.mean([row["runtime"]["total_seconds"] for row in rows])
            ),
            "mean_cache_write_seconds": float(
                np.mean([row["runtime"]["cache_write_seconds"] for row in rows])
            ),
            "cache_bytes": int(sum(row["cache_bytes"] for row in rows)),
        },
        "scope": {
            "organizer_fit_source_pixels_opened": True,
            "known_synthetic_permutation_labels_materialized": True,
            "organizer_reference_or_hidden_labels_opened": False,
            "reserved_dev_pixels_or_labels_opened": False,
            "training_run": False,
            "checkpoint_written": False,
            "scoring_run": False,
            "decoder_run": False,
            "terminal16_opened": False,
            "competition_test_accessed": False,
        },
        "legality": {
            "organizer_train_only": True,
            "candidate_union_target_blind": True,
            "raw_top32_always_preserved": True,
            "cache_contains_pixels": False,
            "adapter_matcher_view_persisted": False,
            "original_upright_tile_identities_only": True,
            "output_material": "candidate identity features and synthetic target slots only",
        },
        "artifacts": {
            "materializer": _record(Path(__file__)),
            "feature_runner": _record(Path(prior.__file__)),
            "manifest": _record(args.manifest),
            "socket": _record(args.socket_checkpoint),
            "adapter": _record(prior.ADAPTER_CHECKPOINT),
            "dino": _record(prior.DINO_CHECKPOINT),
            "socket_parent_report": _record(socket_report_path),
            "adapter_parent_report": _record(adapter_report_path),
        },
        "next_transition_authorized": False,
        "next_required_review": (
            "hash this completed cache report, then create and separately sign one "
            "joint-reciprocal FIT/DEV protocol"
        ),
    }
    _write_json(output / "report.json", report)
    return report


def main() -> None:
    args = parse_args()
    config, config_sha = load_signed_config(args.config)
    random.seed(SELECTION_SEED)
    np.random.seed(SELECTION_SEED)
    torch.manual_seed(SELECTION_SEED)
    torch.use_deterministic_algorithms(
        True, warn_only=bool(args.allow_nondeterministic_mps)
    )
    report = run_materialization(args, config, config_sha)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
