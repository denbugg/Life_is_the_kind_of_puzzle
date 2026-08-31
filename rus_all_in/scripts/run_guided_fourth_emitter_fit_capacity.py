#!/usr/bin/env python3
"""Freeze and measure a guided fourth-emitter FIT-only sidecar cache.

This runner has no development, local, terminal, test or submission mode.  Its
first stage freezes target-free candidate identities and hashes.  Exact FIT
labels can only be recreated by a later mode after those hashes verify.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from aiijc_puzzle.guided_fourth_emitter import (
    FOURTH_EMITTERS,
    GUIDED_AUXILIARY_DIM,
    extend_with_guided_emitter,
    fixed_guided_standalone_scores,
    guided_fourth_pool_digest,
    pool_from_target_free_legacy_cache,
)
from aiijc_puzzle.protocol import (
    compute_protocol_digest,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.restoration_r6 import distort_tiles
from aiijc_puzzle.synthetic_socket_evaluation import (
    DEFAULT_SYNTHETIC_NAMESPACE,
    make_exact_synthetic_case,
    names_digest,
)
from aiijc_puzzle.tri_emitter_edge_verifier import EMITTERS, TOP_K

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/guided_fourth_emitter_fit_capacity_preregistered_v1.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/guided-fourth-emitter/fit32-draw2-capacity-v1"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
CONFIG_SCHEMA = "aiijc-guided-fourth-emitter-fit-capacity-protocol-v1"
CONFIG_STATUS = "signed-fit-cache-capacity-only-not-real-protocol"
AUDIT_SCHEMA = "aiijc-guided-fourth-emitter-roster-audit-v1"
METADATA_SCHEMA = "aiijc-guided-fourth-emitter-target-free-fit-cache-v1"
FREEZE_SCHEMA = "aiijc-guided-fourth-emitter-pre-label-freeze-v1"
LABEL_SCHEMA = "aiijc-guided-fourth-emitter-separated-fit-labels-v1"
CAPACITY_SCHEMA = "aiijc-guided-fourth-emitter-fit-capacity-report-v1"
FIT_CASE_SEED = 20260914
FIT_DRAWS = (0, 1)
LEGACY_FULL_KEYS = frozenset(
    {
        "raw_sides",
        "dino_sides",
        "candidates",
        "valid",
        "auxiliary",
        "raw_baseline",
        "emitter_topk",
        "target_slots",
    }
)
LEGACY_POOL_KEYS = (
    "candidates",
    "valid",
    "auxiliary",
    "raw_baseline",
    "emitter_topk",
)
SIDECAR_KEYS = frozenset(
    {
        "candidates",
        "valid",
        "legacy_slot",
        "guided_auxiliary",
        "guided_baseline",
        "emitter_topk",
        "legacy_identity_digest_ascii",
        "identity_digest_ascii",
    }
)


@dataclass(frozen=True)
class VerifiedFreeze:
    metadata: dict[str, Any]
    metadata_path: Path
    freeze_path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True, choices=("audit", "freeze-fit", "attach-fit-labels")
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    return parser.parse_args(argv)


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        label = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        label = str(resolved)
    return {"path": label, "sha256": sha256_file(resolved)}


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError("signed FIT-capacity config or sidecar is missing")
    digest = sha256_file(resolved)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise RuntimeError("FIT-capacity config sidecar mismatch")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != CONFIG_STATUS:
        raise RuntimeError("config is not the fixed FIT-cache-capacity-only protocol")
    if config.get("real_protocol_signed") is not False:
        raise RuntimeError("this delegated path must not sign a real protocol")
    fixed = config.get("fixed_candidate_path", {})
    expected = {
        "legacy_emitters": list(EMITTERS),
        "fourth_emitter": FOURTH_EMITTERS[-1],
        "top_k_per_emitter": TOP_K,
        "legacy_slot_width": 3 * TOP_K,
        "extended_slot_width": 4 * TOP_K,
        "guided_auxiliary_dim": GUIDED_AUXILIARY_DIM,
        "guided_recipe": {
            "radius": 2,
            "epsilon_uint8_squared": 1600.0,
            "fusion_weight": 0.5,
            "standalone_recovery": "2*fused-minus-bilateral",
        },
    }
    for key, value in expected.items():
        if fixed.get(key) != value:
            raise RuntimeError(f"fixed fourth-emitter contract changed: {key}")
    source = config.get("source_protocol", {})
    names = tuple(source.get("fit_filenames", ()))
    if len(names) != 32 or len(set(names)) != 32:
        raise RuntimeError("FIT roster must contain exactly 32 unique sources")
    if source.get("fit_digest") != names_digest(names):
        raise RuntimeError("FIT roster digest mismatch")
    if tuple(source.get("fit_draw_indices", ())) != FIT_DRAWS:
        raise RuntimeError("FIT draw roster changed")
    if int(source.get("case_seed", -1)) != FIT_CASE_SEED:
        raise RuntimeError("FIT synthetic seed changed")
    forbidden: set[str] = set()
    for group in source.get("forbidden_rosters", {}).values():
        values = tuple(group.get("filenames", ()))
        if group.get("digest") != names_digest(values):
            raise RuntimeError("forbidden roster digest mismatch")
        if forbidden & set(values):
            raise RuntimeError("forbidden rosters unexpectedly overlap")
        forbidden.update(values)
    if set(names) & forbidden:
        raise RuntimeError("FIT roster overlaps an opened or protected roster")
    for artifact in config.get("frozen_inputs", {}).values():
        target = _project_path(artifact["path"])
        if not target.is_file() or sha256_file(target) != artifact["sha256"]:
            raise RuntimeError(f"frozen input changed: {target}")
    return config, digest


def _legacy_rows(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    artifact = config["frozen_inputs"]["legacy_tri_report"]
    report = json.loads(_project_path(artifact["path"]).read_text(encoding="utf-8"))
    if report.get("schema") != "aiijc-tri-emitter-edge-verifier-report-v1":
        raise RuntimeError("legacy tri report schema changed")
    rows = tuple(report.get("fit_cache", {}).get("rows", ()))
    source = config["source_protocol"]
    expected = tuple(
        (name, draw) for name in source["fit_filenames"] for draw in source["fit_draw_indices"]
    )
    observed = tuple((row.get("source_filename"), row.get("draw_index")) for row in rows)
    if observed != expected:
        raise RuntimeError("legacy cache roster/order differs from fixed FIT32 x draw2")
    if len(rows) != 64:
        raise RuntimeError("legacy FIT cache must contain 64 cases")
    for row in rows:
        path = _project_path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"immutable legacy FIT cache changed: {path}")
    inventory = hashlib.sha256(
        "\n".join(
            "\0".join(
                (
                    str(row["path"]),
                    str(row["sha256"]),
                    str(row["source_filename"]),
                    str(row["draw_index"]),
                    str(row["case_id"]),
                    str(row["dirty_sha256"]),
                    str(row["runtime"]["union_identity_digest"]),
                )
            )
            for row in rows
        ).encode()
    ).hexdigest()
    if inventory != config.get("legacy_cache_inventory_digest"):
        raise RuntimeError("legacy cache normalized inventory digest mismatch")
    return rows


def _manifest_records(
    manifest_path: Path,
    names: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise RuntimeError("validation manifest protocol digest mismatch")
    lookup = {str(record["filename"]): record for record in manifest["splits"]["train"]}
    try:
        return tuple(lookup[name] for name in names)
    except KeyError as error:
        raise RuntimeError("FIT source is absent from manifest train split") from error


def _load_clean_tiles(record: Mapping[str, Any], targets: Path) -> np.ndarray:
    path = targets / str(record["filename"])
    if sha256_file(path) != record.get("target_sha256"):
        raise RuntimeError(f"manifest target hash mismatch: {path.name}")
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    tiles = split_tiles(rgb)
    if tiles.shape != (576, 20, 20, 3):
        raise RuntimeError("FIT source does not produce 576 upright 20x20 tiles")
    return np.ascontiguousarray(tiles)


def make_target_free_fit_case(
    clean_tiles: np.ndarray,
    *,
    source_filename: str,
    draw_index: int,
) -> tuple[str, np.ndarray]:
    """Replay the exact synthetic input without constructing an inverse shuffle."""

    digest = hashlib.sha256(
        f"{DEFAULT_SYNTHETIC_NAMESPACE}\0{FIT_CASE_SEED}\0{source_filename}\0{draw_index}".encode()
    ).digest()
    corruption_seed = int.from_bytes(digest[:8], "little")
    permutation_seed = int.from_bytes(digest[8:16], "little")
    corrupted = distort_tiles(clean_tiles, np.random.default_rng(corruption_seed))
    permutation = np.random.default_rng(permutation_seed).permutation(len(clean_tiles))
    case_digest = hashlib.sha256(
        f"{source_filename}\0{draw_index}\0{FIT_CASE_SEED}".encode()
    ).hexdigest()[:16]
    return (
        f"synthetic-{case_digest}",
        np.ascontiguousarray(corrupted[permutation]),
    )


def _load_legacy_pool_target_free(path: Path) -> Any:
    """Load five safe pool arrays while never materialising ``target_slots``."""

    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != LEGACY_FULL_KEYS:
            raise RuntimeError("legacy cache keys changed")
        safe = {key: np.ascontiguousarray(archive[key]) for key in LEGACY_POOL_KEYS}
    return pool_from_target_free_legacy_cache(safe)


def _sidecar_arrays(pool: Any) -> dict[str, np.ndarray]:
    return {
        "candidates": pool.candidates.astype(np.int32),
        "valid": pool.valid.astype(bool),
        "legacy_slot": pool.legacy_slot.astype(np.int16),
        "guided_auxiliary": pool.guided_auxiliary.astype(np.float16),
        "guided_baseline": pool.guided_baseline.astype(np.float16),
        "emitter_topk": pool.emitter_topk.astype(np.int32),
        "legacy_identity_digest_ascii": np.frombuffer(
            pool.legacy_identity_digest.encode(), dtype=np.uint8
        ),
        "identity_digest_ascii": np.frombuffer(pool.identity_digest.encode(), dtype=np.uint8),
    }


def validate_sidecar_arrays(arrays: Mapping[str, np.ndarray], *, count: int = 576) -> None:
    if set(arrays) != SIDECAR_KEYS:
        raise RuntimeError("guided sidecar keys changed")
    width = 4 * TOP_K
    shapes = {
        "candidates": (2, count, width),
        "valid": (2, count, width),
        "legacy_slot": (2, count, width),
        "guided_auxiliary": (2, count, width, GUIDED_AUXILIARY_DIM),
        "guided_baseline": (2, count, width),
        "emitter_topk": (4, 2, count, TOP_K),
        "legacy_identity_digest_ascii": (64,),
        "identity_digest_ascii": (64,),
    }
    for key, shape in shapes.items():
        if np.asarray(arrays[key]).shape != shape:
            raise RuntimeError(f"guided sidecar {key} shape changed")
    candidates = np.asarray(arrays["candidates"])
    valid = np.asarray(arrays["valid"])
    if candidates.dtype not in (np.int32, np.int64) or valid.dtype != np.bool_:
        raise RuntimeError("guided candidates/valid dtype changed")
    if np.asarray(arrays["legacy_slot"]).dtype not in (np.int16, np.int32, np.int64):
        raise RuntimeError("guided legacy slot dtype changed")
    for key in ("guided_auxiliary", "guided_baseline"):
        value = np.asarray(arrays[key])
        if value.dtype not in (np.float16, np.float32) or not np.isfinite(value).all():
            raise RuntimeError(f"guided sidecar {key} must be finite float16/float32")
    if np.any(valid & ((candidates < 0) | (candidates >= count))):
        raise RuntimeError("guided sidecar contains an invalid tile identity")
    legacy_slot = np.asarray(arrays["legacy_slot"])
    if np.any(legacy_slot[..., 3 * TOP_K :] != -1):
        raise RuntimeError("guided-only slots unexpectedly map to a legacy slot")
    for axis in range(2):
        for source in range(count):
            row = candidates[axis, source, valid[axis, source]]
            if len(row) != len(np.unique(row)):
                raise RuntimeError("guided sidecar contains duplicate candidate identities")
            raw = set(arrays["emitter_topk"][0, axis, source].tolist())
            if not raw.issubset(set(row.tolist())):
                raise RuntimeError("guided sidecar dropped raw top32")
    digest = guided_fourth_pool_digest(
        candidates, valid, legacy_slot, arrays["emitter_topk"]
    )
    if bytes(arrays["identity_digest_ascii"]).decode() != digest:
        raise RuntimeError("guided sidecar identity digest mismatch")


def _load_sidecar(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
    validate_sidecar_arrays(arrays)
    return arrays


def run_audit(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    config_sha: str,
) -> dict[str, Any]:
    rows = _legacy_rows(config)
    source = config["source_protocol"]
    forbidden = {
        name: {
            "source_count": len(group["filenames"]),
            "digest": group["digest"],
            "fit_overlap_count": len(set(source["fit_filenames"]) & set(group["filenames"])),
        }
        for name, group in source["forbidden_rosters"].items()
    }
    report = {
        "schema": AUDIT_SCHEMA,
        "status": "pass-target-free-cache-freeze-authorised",
        "config_sha256": config_sha,
        "scope": {
            "metadata_only": True,
            "organizer_pixels_loaded": False,
            "exact_or_recovered_labels_loaded": False,
            "legacy_npz_contents_loaded": False,
            "models_run": False,
            "dev_local_terminal_test_or_submission_accessed": False,
            "real_protocol_signed": False,
        },
        "fit": {
            "source_count": len(source["fit_filenames"]),
            "draw_indices": source["fit_draw_indices"],
            "case_count": len(rows),
            "ordered_source_digest": source["fit_digest"],
            "legacy_cache_inventory_digest": config["legacy_cache_inventory_digest"],
        },
        "forbidden_overlap_audit": forbidden,
        "no_repeat_decision": {
            "guided_recipe_parameters_closed_to_tuning": True,
            "same_panel_tuning_forbidden": True,
            "new_hypothesis": "guided standalone top32 only as a fourth candidate-supply emitter",
            "confidence_replacement": False,
        },
        "cache_contract": {
            "legacy_npz_byte_immutable": True,
            "new_sidecar_schema": METADATA_SCHEMA,
            "legacy_slots_0_through_95_preserved": True,
            "guided_novel_slots_96_through_127_only": True,
            "raw_top32_retained": True,
            "target_labels_separate_after_pre_label_hash_freeze": True,
        },
        "artifacts": {
            "config": _record(args.config),
            "runner": _record(Path(__file__)),
            "module": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/guided_fourth_emitter.py"
            ),
        },
    }
    if any(value["fit_overlap_count"] for value in forbidden.values()):
        raise RuntimeError("FIT source overlaps a forbidden roster")
    _write_json_exclusive(args.output_dir.resolve() / "roster-audit.json", report)
    return report


def _verify_audit(output: Path, config_sha: str) -> dict[str, Any]:
    path = output / "roster-audit.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("schema") != AUDIT_SCHEMA or audit.get("config_sha256") != config_sha:
        raise RuntimeError("roster audit does not belong to this fixed config")
    scope = audit.get("scope", {})
    if not scope.get("metadata_only") or any(
        scope.get(key) is not False
        for key in (
            "organizer_pixels_loaded",
            "exact_or_recovered_labels_loaded",
            "legacy_npz_contents_loaded",
            "models_run",
            "dev_local_terminal_test_or_submission_accessed",
            "real_protocol_signed",
        )
    ):
        raise RuntimeError("roster audit did not remain metadata-only")
    return audit


def run_freeze_fit(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    config_sha: str,
) -> dict[str, Any]:
    output = args.output_dir.resolve()
    _verify_audit(output, config_sha)
    rows = _legacy_rows(config)
    names = tuple(config["source_protocol"]["fit_filenames"])
    records = _manifest_records(args.manifest, names)
    cache_dir = output / "target-free-cache"
    cache_dir.mkdir(parents=True, exist_ok=False)
    frozen_rows: list[dict[str, Any]] = []
    started = perf_counter()
    row_index = 0
    for source_index, record in enumerate(records):
        clean = _load_clean_tiles(record, args.targets)
        for draw_index in FIT_DRAWS:
            legacy_row = rows[row_index]
            case_id, dirty = make_target_free_fit_case(
                clean,
                source_filename=str(record["filename"]),
                draw_index=draw_index,
            )
            dirty_sha = _array_sha256(dirty)
            if (
                case_id != legacy_row["case_id"]
                or dirty_sha != legacy_row["dirty_sha256"]
                or legacy_row["source_filename"] != record["filename"]
                or int(legacy_row["draw_index"]) != draw_index
            ):
                raise RuntimeError("target-free FIT replay differs from immutable legacy case")
            legacy_path = _project_path(legacy_row["path"])
            legacy_pool = _load_legacy_pool_target_free(legacy_path)
            expected_legacy_digest = legacy_row["runtime"]["union_identity_digest"]
            if legacy_pool.identity_digest != expected_legacy_digest:
                raise RuntimeError("legacy cache identity differs from parent report")
            case_started = perf_counter()
            guided_scores = fixed_guided_standalone_scores(dirty)
            extended = extend_with_guided_emitter(legacy_pool, guided_scores)
            arrays = _sidecar_arrays(extended)
            validate_sidecar_arrays(arrays)
            path = cache_dir / f"source_{source_index:02d}_draw_{draw_index}.npz"
            _write_npz_exclusive(path, arrays)
            legacy_valid = int(np.count_nonzero(legacy_pool.valid))
            extended_valid = int(np.count_nonzero(extended.valid))
            frozen_rows.append(
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "sha256": sha256_file(path),
                    "source_filename": record["filename"],
                    "draw_index": draw_index,
                    "case_id": case_id,
                    "dirty_sha256": dirty_sha,
                    "legacy_cache": {"path": legacy_row["path"], "sha256": legacy_row["sha256"]},
                    "legacy_identity_digest": legacy_pool.identity_digest,
                    "identity_digest": extended.identity_digest,
                    "legacy_valid_candidate_slots": legacy_valid,
                    "extended_valid_candidate_slots": extended_valid,
                    "guided_novel_candidate_slots": extended_valid - legacy_valid,
                    "guided_cpu_seconds": perf_counter() - case_started,
                }
            )
            row_index += 1
            print(
                json.dumps(
                    {
                        "event": "guided_fourth_target_free_fit_cache",
                        "case": row_index,
                        "count": len(rows),
                        "source": record["filename"],
                        "draw": draw_index,
                        "guided_novel_slots": extended_valid - legacy_valid,
                    }
                ),
                flush=True,
            )
    metadata_path = output / "target-free-cache.json"
    metadata = {
        "schema": METADATA_SCHEMA,
        "config_sha256": config_sha,
        "created_before_exact_fit_reference_recreation": True,
        "contains_target_slots_truth_or_reference_labels": False,
        "contains_clean_dirty_or_output_pixels": False,
        "candidate_identities_target_blind": True,
        "legacy_cache_byte_immutable": True,
        "raw_top32_retained": True,
        "case_count": len(frozen_rows),
        "rows": frozen_rows,
    }
    _write_json_exclusive(metadata_path, metadata)
    freeze_path = output / "pre-label-freeze.json"
    freeze = {
        "schema": FREEZE_SCHEMA,
        "status": "target-free-identities-frozen-label-stage-not-run",
        "created_before_exact_fit_reference_recreation": True,
        "contains_target_slots_truth_or_reference_labels": False,
        "config_sha256": config_sha,
        "artifacts": {
            "config": _record(args.config),
            "roster_audit": _record(output / "roster-audit.json"),
            "metadata": _record(metadata_path),
            "runner": _record(Path(__file__)),
            "module": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/guided_fourth_emitter.py"
            ),
        },
        "case_files": [
            {"path": row["path"], "sha256": row["sha256"]} for row in frozen_rows
        ],
        "runtime_seconds": perf_counter() - started,
        "dev_local_terminal_test_or_submission_accessed": False,
        "real_protocol_signed": False,
    }
    _write_json_exclusive(freeze_path, freeze)
    return freeze


def verify_pre_label_freeze(output: Path, config_sha: str) -> VerifiedFreeze:
    freeze_path = output / "pre-label-freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("config_sha256") != config_sha:
        raise RuntimeError("pre-label freeze does not belong to this fixed config")
    if freeze.get("created_before_exact_fit_reference_recreation") is not True:
        raise RuntimeError("candidate identities were not frozen before labels")
    if freeze.get("contains_target_slots_truth_or_reference_labels") is not False:
        raise RuntimeError("pre-label freeze unexpectedly contains labels")
    metadata_path = output / "target-free-cache.json"
    expected = freeze["artifacts"]["metadata"]["sha256"]
    if not metadata_path.is_file() or sha256_file(metadata_path) != expected:
        raise RuntimeError("target-free metadata changed after pre-label freeze")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != METADATA_SCHEMA or metadata.get("config_sha256") != config_sha:
        raise RuntimeError("target-free metadata contract changed")
    if metadata.get("contains_target_slots_truth_or_reference_labels") is not False:
        raise RuntimeError("target-free metadata unexpectedly contains labels")
    files = {(item["path"], item["sha256"]) for item in freeze["case_files"]}
    rows = {(row["path"], row["sha256"]) for row in metadata["rows"]}
    if files != rows or len(files) != 64:
        raise RuntimeError("target-free cache manifest changed")
    for path_value, digest in files:
        path = _project_path(path_value)
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"target-free cache changed after freeze: {path}")
    return VerifiedFreeze(metadata=metadata, metadata_path=metadata_path, freeze_path=freeze_path)


def _truth_by_source(reference: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.int32)
    if reference.shape != (576,) or not np.array_equal(np.sort(reference), np.arange(576)):
        raise RuntimeError("FIT exact reference is not a strict 576-tile permutation")
    truth = np.full((2, 576), -1, dtype=np.int32)
    position = np.arange(576)
    for axis, delta in ((0, 1), (1, 24)):
        valid = position % 24 != 23 if axis == 0 else position < 552
        anchors = reference[position[valid]]
        targets = reference[position[valid] + delta]
        truth[axis, anchors] = targets
    return truth


def _target_slots(candidates: np.ndarray, valid: np.ndarray, truth: np.ndarray) -> np.ndarray:
    slots = np.full((2, 576), -1, dtype=np.int16)
    for axis in range(2):
        for source in range(576):
            if truth[axis, source] < 0:
                continue
            match = np.flatnonzero(
                valid[axis, source] & (candidates[axis, source] == truth[axis, source])
            )
            if len(match) > 1:
                raise RuntimeError("truth identity repeats in a candidate row")
            if len(match) == 1:
                slots[axis, source] = match[0]
    return slots


def _coverage_counts(
    arrays: Mapping[str, np.ndarray],
    truth: np.ndarray,
    slots: np.ndarray,
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for axis, name in enumerate(("right", "down")):
        eligible = truth[axis] >= 0
        legacy = (slots[axis] >= 0) & (slots[axis] < 3 * TOP_K)
        extended = slots[axis] >= 0
        raw = np.any(
            arrays["emitter_topk"][0, axis] == truth[axis, :, None], axis=1
        ) & eligible
        guided = np.any(
            arrays["emitter_topk"][3, axis] == truth[axis, :, None], axis=1
        ) & eligible
        result[name] = {
            "eligible": int(np.count_nonzero(eligible)),
            "raw_top32": int(np.count_nonzero(raw)),
            "guided_top32": int(np.count_nonzero(guided)),
            "legacy_union": int(np.count_nonzero(legacy)),
            "extended_union": int(np.count_nonzero(extended)),
            "guided_unique_recovered": int(np.count_nonzero(extended & ~legacy)),
        }
    return result


def run_attach_fit_labels(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    config_sha: str,
) -> dict[str, Any]:
    output = args.output_dir.resolve()
    verified = verify_pre_label_freeze(output, config_sha)
    names = tuple(config["source_protocol"]["fit_filenames"])
    records = _manifest_records(args.manifest, names)
    labels_dir = output / "separated-fit-labels"
    labels_dir.mkdir(parents=True, exist_ok=False)
    label_rows: list[dict[str, Any]] = []
    totals = {
        axis: {key: 0 for key in (
            "eligible",
            "raw_top32",
            "guided_top32",
            "legacy_union",
            "extended_union",
            "guided_unique_recovered",
        )}
        for axis in ("right", "down")
    }
    row_index = 0
    for source_index, record in enumerate(records):
        clean = _load_clean_tiles(record, args.targets)
        for draw_index in FIT_DRAWS:
            frozen_row = verified.metadata["rows"][row_index]
            item, reference = make_exact_synthetic_case(
                clean,
                source_filename=str(record["filename"]),
                draw_index=draw_index,
                seed=FIT_CASE_SEED,
            )
            if item.case_id != frozen_row["case_id"] or _array_sha256(item.tiles) != frozen_row[
                "dirty_sha256"
            ]:
                raise RuntimeError("FIT reference recreation differs from pre-label freeze")
            arrays = _load_sidecar(_project_path(frozen_row["path"]))
            truth = _truth_by_source(reference.tile_at_position)
            slots = _target_slots(arrays["candidates"], arrays["valid"], truth)
            coverage = _coverage_counts(arrays, truth, slots)
            for axis in totals:
                for key, value in coverage[axis].items():
                    totals[axis][key] += value
            label_path = labels_dir / f"source_{source_index:02d}_draw_{draw_index}.npz"
            _write_npz_exclusive(
                label_path,
                {
                    "truth_by_source": truth.astype(np.int32),
                    "target_slots": slots.astype(np.int16),
                },
            )
            label_rows.append(
                {
                    "path": str(label_path.relative_to(PROJECT_ROOT)),
                    "sha256": sha256_file(label_path),
                    "source_filename": record["filename"],
                    "draw_index": draw_index,
                    "case_id": item.case_id,
                    "target_free_cache": {
                        "path": frozen_row["path"],
                        "sha256": frozen_row["sha256"],
                    },
                    "coverage": coverage,
                }
            )
            row_index += 1
            print(
                json.dumps(
                    {
                        "event": "guided_fourth_separated_fit_labels",
                        "case": row_index,
                        "count": 64,
                        "guided_unique_recovered": sum(
                            coverage[axis]["guided_unique_recovered"] for axis in coverage
                        ),
                    }
                ),
                flush=True,
            )
    labels_metadata = output / "separated-fit-labels.json"
    _write_json_exclusive(
        labels_metadata,
        {
            "schema": LABEL_SCHEMA,
            "config_sha256": config_sha,
            "pre_label_freeze_verified_before_reference_recreation": True,
            "labels_physically_separate_from_target_free_cache": True,
            "case_count": len(label_rows),
            "rows": label_rows,
        },
    )
    pooled = {
        key: totals["right"][key] + totals["down"][key] for key in totals["right"]
    }
    rates = {
        key: pooled[key] / pooled["eligible"]
        for key in ("raw_top32", "guided_top32", "legacy_union", "extended_union")
    }
    gain = pooled["extended_union"] - pooled["legacy_union"]
    total_novel_slots = sum(
        int(row["guided_novel_candidate_slots"]) for row in verified.metadata["rows"]
    )
    passed = bool(
        pooled["extended_union"] >= pooled["legacy_union"]
        and gain == pooled["guided_unique_recovered"]
        and total_novel_slots >= int(config["capacity_gate"]["minimum_total_novel_slots"])
    )
    report = {
        "schema": CAPACITY_SCHEMA,
        "status": "pass-fit-only-cache-capacity" if passed else "fail-stop",
        "config_sha256": config_sha,
        "claim": (
            "FIT-training capacity and cache plumbing only; "
            "no real protocol or promotion evidence"
        ),
        "pre_label_freeze_verified_before_reference_recreation": True,
        "case_count": len(label_rows),
        "directed_true_neighbours_per_case": 1104,
        "coverage_counts": {**totals, "pooled": pooled},
        "pooled_coverage_rates": rates,
        "legacy_to_extended": {
            "additional_true_neighbours": gain,
            "absolute_rate_gain": rates["extended_union"] - rates["legacy_union"],
            "candidate_coverage_nonregression": pooled["extended_union"]
            >= pooled["legacy_union"],
            "total_guided_novel_candidate_slots": total_novel_slots,
        },
        "gate": {
            **config["capacity_gate"],
            "passed": passed,
        },
        "legality": {
            "organizer_train_fit_only": True,
            "guided_pixels_matcher_only": True,
            "output_tiles_modified_or_rendered": False,
            "legacy_tri_cache_modified": False,
            "labels_separate": True,
            "dev_local_terminal_test_or_submission_accessed": False,
            "real_protocol_signed": False,
        },
        "artifacts": {
            "config": _record(args.config),
            "roster_audit": _record(output / "roster-audit.json"),
            "target_free_metadata": _record(verified.metadata_path),
            "pre_label_freeze": _record(verified.freeze_path),
            "separated_labels_metadata": _record(labels_metadata),
            "runner": _record(Path(__file__)),
            "module": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/guided_fourth_emitter.py"
            ),
        },
    }
    _write_json_exclusive(output / "capacity-report.json", report)
    if not passed:
        raise RuntimeError("guided fourth-emitter FIT capacity gate failed")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config, config_sha = _load_config(args.config)
    if args.mode == "audit":
        result = run_audit(args, config, config_sha)
    elif args.mode == "freeze-fit":
        result = run_freeze_fit(args, config, config_sha)
    else:
        result = run_attach_fit_labels(args, config, config_sha)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
