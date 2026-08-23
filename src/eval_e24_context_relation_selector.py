"""Leak-resistant orchestration and evaluation boundary for frozen E24 CRS-v1.

This module deliberately separates three capabilities:

* an input broker may open only two literal members of the authenticated
  historical Rank96 NPZ, then a label-free feature worker receives only the
  sanitized two-array handoff;
* a fold-label broker may open ``permutation.npy`` for exactly the six
  training scenes, and a fold trainer receives only those committed labels;
* the held-out evaluator may load all held-out permutations only after every
  one of the four complete fold model/prediction transactions has passed the
  global atomic-commit barrier.

Importing this module does not read an E24 scene, a permutation, a clean target,
an E23 report, or an OOF metric.  Target execution is intentionally exposed as
small capability-based functions so the boundary can be tested with synthetic
objects before any target metric is opened.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sys
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence, TypeVar


# Keep import bytecode off C: even when a caller supplied an unsafe prefix.
_DEFAULT_PYCACHE = Path("E:/pazzle_work/posegraph_e24_selector/pycache")
if sys.pycache_prefix is None or Path(sys.pycache_prefix).drive.upper() != "E:":
    sys.pycache_prefix = str(_DEFAULT_PYCACHE)

import numpy as np

import e24_context_relation_selector as selector
import e23_i21_residual_candidate_oracle as e23_core


class E24EvaluatorContractError(RuntimeError):
    """The frozen E24 evaluator or its process boundary was violated."""


SCHEMA_VERSION = 1
PROTOCOL_SCHEMA = "pazzle-e24-crs-v1-evaluator-protocol-v1"
PREDICTION_SCHEMA = "pazzle-e24-crs-v1-fold-predictions-v1"
COMMIT_SCHEMA = "pazzle-e24-crs-v1-fold-commit-v1"

CALIBRATION_IDS = tuple(range(10, 18))
OOF_FOLDS: Mapping[int, tuple[int, int]] = MappingProxyType(
    {0: (10, 14), 1: (11, 15), 2: (12, 16), 3: (13, 17)}
)
E25_SEALED_IDS = (
    226,
    262,
    242,
    123,
    103,
    231,
    286,
    296,
    230,
    134,
    118,
    110,
    239,
    269,
    146,
    187,
    183,
    151,
    148,
    247,
    191,
    186,
    193,
    106,
    220,
    274,
    125,
    117,
    115,
    265,
    165,
    257,
    210,
    213,
    132,
    143,
    152,
    137,
    177,
    225,
    113,
    259,
    101,
    178,
    202,
    141,
    273,
    111,
)
E25_NEWLINE_LIST_SHA256 = (
    "407a6326ceeec2e8cc78106b74c2f10c46a55143ea488a30f7bac66e2b373caa"
)
E25_CANONICAL_RECORDS_SHA256 = (
    "76e6b9431de41388e4aebef525ff4a5fd8354f789cf0a5913c1e29d8db148e2e"
)

STORAGE_ROOT = Path("E:/pazzle_work/posegraph_e24_selector")
FEATURE_CACHE_BYTES_MAX = 4 * 1024**3
ALL_ARTIFACT_BYTES_MAX = 8 * 1024**3
PEAK_RAM_BYTES_MAX = 16 * 1024**3
OOF_CPU_SECONDS_MAX = 8 * 60 * 60
FINAL_FIT_CPU_SECONDS_MAX = 2 * 60 * 60
GEOMETRY_HYPOTHESES_MAX_EACH = 450_000

RAW_NPZ_KEYS = frozenset({"candidate_ids", "candidate_scores"})
RAW_NPZ_BYTES_MAX = 8 * 1024**2
ORIGINAL_RAW_MEMBER_BYTES_MAX = 2 * 1024**2
PERMUTATION_MEMBER_BYTES_MAX = 64 * 1024
SANITIZED_RAW_SCHEMA = "pazzle-e24-sanitized-rank96-raw-v1"
SANITIZED_RAW_MANIFEST_BYTES_MAX = 64 * 1024
NUM_TILES = 576
CANDIDATE_WIDTH = 128
NUM_DIRECTIONS = 4

EXPECTED_LIGHTGBM_VERSION = "4.6.0"
LIGHTGBM_BASE_CONFIG: Mapping[str, Any] = MappingProxyType(
    dict(selector.LIGHTGBM_CONFIG)
)

STRUCTURAL_GATES: Mapping[str, float | int] = MappingProxyType(
    {
        "complete_integrity_legal_scenes": 8,
        "proposed_precision_mean_min": 0.70,
        "proposed_precision_worst_min": 0.60,
        "true_relation_recall_mean_min": 0.65,
        "true_relation_recall_worst_min": 0.50,
        "exact_connected_coverage_mean_min": 0.50,
        "exact_connected_coverage_worst_min": 0.35,
        "mean_cycle_rank_ratio_min": 0.05,
        "geometry_hypotheses_max_each": GEOMETRY_HYPOTHESES_MAX_EACH,
    }
)

END_TO_END_GATES: Mapping[str, float | int] = MappingProxyType(
    {
        "solve_ssim_delta_mean_min": 0.003,
        "final_ssim_delta_mean_min": 0.002,
        "final_wins_min": 5,
        "worst_final_delta_min": -0.020,
        "neighbour_delta_mean_min": 0.005,
    }
)

E24_EVALUATOR_PROTOCOL: Mapping[str, Any] = MappingProxyType(
    {
        "schema": PROTOCOL_SCHEMA,
        "role": "independent_discovery_not_confirmation_or_production",
        "calibration_ids": CALIBRATION_IDS,
        "folds": {str(key): value for key, value in OOF_FOLDS.items()},
        "raw_feature_worker_npz_keys": tuple(sorted(RAW_NPZ_KEYS)),
        "feature_worker_forbidden": (
            "label",
            "labels",
            "relevance",
            "RawScene",
            "permutation",
            "target",
            "clean_pixels",
            "report",
            "oracle",
        ),
        "learner": dict(LIGHTGBM_BASE_CONFIG),
        "lightgbm_version": EXPECTED_LIGHTGBM_VERSION,
        "fold_seed": "1234_plus_fold",
        "early_stopping": False,
        "validation_callback": False,
        "label_driven_sampling_or_injection": False,
        "structural_gates": dict(STRUCTURAL_GATES),
        "end_to_end_gates": dict(END_TO_END_GATES),
        "storage": {
            "root": str(STORAGE_ROOT),
            "feature_cache_bytes_max": FEATURE_CACHE_BYTES_MAX,
            "all_artifact_bytes_max": ALL_ARTIFACT_BYTES_MAX,
            "peak_ram_bytes_max": PEAK_RAM_BYTES_MAX,
            "oof_cpu_seconds_max": OOF_CPU_SECONDS_MAX,
            "final_fit_cpu_seconds_max": FINAL_FIT_CPU_SECONDS_MAX,
            "c_artifacts": False,
        },
        "e25_seal": {
            "ids": E25_SEALED_IDS,
            "newline_list_sha256": E25_NEWLINE_LIST_SHA256,
            "canonical_records_sha256": E25_CANONICAL_RECORDS_SHA256,
            "unopened_until_e24_freeze": True,
        },
        "orientation_degrees": (0,),
        "reflection": False,
    }
)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise E24EvaluatorContractError("value is not canonical finite JSON") from exc
    return (encoded + "\n").encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


PROTOCOL_SHA256 = _sha256_bytes(_canonical_json_bytes(dict(E24_EVALUATOR_PROTOCOL)))


def _require_e_drive(path: str | os.PathLike[str], *, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.drive.upper() != "E:":
        raise E24EvaluatorContractError(f"{label} must be an absolute E:-drive path")
    return candidate.resolve(strict=False)


def _require_e24_storage_path(
    path: str | os.PathLike[str], *, label: str
) -> Path:
    candidate = _require_e_drive(path, label=label)
    root = STORAGE_ROOT.resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise E24EvaluatorContractError(
            f"{label} must live under frozen E24 storage root {root}"
        ) from exc
    return candidate


def validate_e24_runtime_paths(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Fail closed unless every Python/temp path resolves below E24 on E:."""

    source = os.environ if environment is None else environment
    required = ("PYTHONPYCACHEPREFIX", "TEMP", "TMP", "TMPDIR")
    normalized: dict[str, str] = {}
    for key in required:
        value = source.get(key)
        if not value:
            raise E24EvaluatorContractError(f"required E24 environment path {key} is absent")
        normalized[key] = str(_require_e24_storage_path(value, label=key))
    runtime_prefix = Path(sys.pycache_prefix) if sys.pycache_prefix else None
    if runtime_prefix is None:
        raise E24EvaluatorContractError("runtime pycache prefix is absent")
    _require_e24_storage_path(runtime_prefix, label="sys.pycache_prefix")
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_lower_hex_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise E24EvaluatorContractError(f"{label} must be a lowercase SHA256")
    if any(char not in "0123456789abcdef" for char in value):
        raise E24EvaluatorContractError(f"{label} must be a lowercase SHA256")
    return value


def _atomic_write_create(path: Path, payload: bytes) -> None:
    target = _require_e24_storage_path(path, label="atomic output")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise E24EvaluatorContractError(f"refusing to overwrite frozen artifact: {target}")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists():
            raise E24EvaluatorContractError(
                f"refusing raced overwrite of frozen artifact: {target}"
            )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_create_or_verify(path: Path, payload: bytes) -> None:
    """Resume an interrupted deterministic transaction without overwriting."""

    target = _require_e24_storage_path(path, label="atomic output")
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise E24EvaluatorContractError(
                f"existing deterministic artifact differs: {target}"
            )
        return
    _atomic_write_create(target, payload)


@dataclass(frozen=True)
class RawCandidateArrays:
    """The complete raw input capability exposed to a feature worker."""

    candidate_ids: np.ndarray
    candidate_scores: np.ndarray
    source_sha256: str


def _validated_raw_arrays(
    candidate_ids: object,
    candidate_scores: object,
    *,
    source_sha256: str,
) -> RawCandidateArrays:
    if not isinstance(candidate_ids, np.ndarray):
        raise E24EvaluatorContractError("candidate_ids must be a numpy array")
    if candidate_ids.shape != (NUM_TILES, CANDIDATE_WIDTH):
        raise E24EvaluatorContractError("candidate_ids must be exactly int64[576,128]")
    if candidate_ids.dtype != np.int64 or not candidate_ids.flags.c_contiguous:
        raise E24EvaluatorContractError("candidate_ids must be C-contiguous int64")
    if not isinstance(candidate_scores, np.ndarray):
        raise E24EvaluatorContractError("candidate_scores must be a numpy array")
    expected_scores_shape = (NUM_DIRECTIONS, NUM_TILES, CANDIDATE_WIDTH)
    if candidate_scores.shape != expected_scores_shape:
        raise E24EvaluatorContractError(
            "candidate_scores must be exactly float32[4,576,128]"
        )
    if candidate_scores.dtype != np.float32 or not candidate_scores.flags.c_contiguous:
        raise E24EvaluatorContractError("candidate_scores must be C-contiguous float32")
    if np.any(np.isnan(candidate_scores)) or np.any(np.isposinf(candidate_scores)):
        raise E24EvaluatorContractError("candidate_scores allow only finite values or -inf")
    finite = np.isfinite(candidate_scores)
    if not np.array_equal(finite[0], finite[1]):
        raise E24EvaluatorContractError("candidate score masks differ across directions")
    if not np.array_equal(finite[0], finite[2]) or not np.array_equal(finite[0], finite[3]):
        raise E24EvaluatorContractError("candidate score masks differ across directions")
    valid = finite[0]
    if not bool(valid.any(axis=1).all()) or not bool(
        np.isneginf(candidate_scores[~finite]).all()
    ):
        raise E24EvaluatorContractError("candidate score padding/row completeness drifted")
    for source in range(NUM_TILES):
        ids = candidate_ids[source, valid[source]]
        if (
            bool((ids < 0).any())
            or bool((ids >= NUM_TILES).any())
            or bool((ids == source).any())
            or np.unique(ids).size != ids.size
        ):
            raise E24EvaluatorContractError(
                "valid candidate IDs contain an out-of-range, self, or duplicate "
                "tile; expected unique non-self tiles 0..575"
            )
    detached_ids = np.array(candidate_ids, dtype=np.int64, copy=True, order="C")
    detached_scores = np.array(
        candidate_scores, dtype=np.float32, copy=True, order="C"
    )
    detached_ids.setflags(write=False)
    detached_scores.setflags(write=False)
    return RawCandidateArrays(
        candidate_ids=detached_ids,
        candidate_scores=detached_scores,
        source_sha256=_validate_lower_hex_sha256(
            source_sha256, label="raw array source SHA"
        ),
    )


def load_feature_worker_raw_npz(path: str | os.PathLike[str]) -> RawCandidateArrays:
    """Whitelist-load the only raw NPZ fields a label-free worker may see."""

    source = _require_e_drive(path, label="raw feature-worker NPZ")
    if source.suffix.lower() != ".npz" or not source.is_file():
        raise E24EvaluatorContractError("raw feature-worker input must be an existing NPZ")
    if source.stat().st_size > RAW_NPZ_BYTES_MAX:
        raise E24EvaluatorContractError("raw feature-worker NPZ exceeds the frozen size cap")

    try:
        with np.load(source, allow_pickle=False) as archive:
            keys = frozenset(archive.files)
            if keys != RAW_NPZ_KEYS:
                extras = sorted(keys - RAW_NPZ_KEYS)
                missing = sorted(RAW_NPZ_KEYS - keys)
                raise E24EvaluatorContractError(
                    f"raw NPZ whitelist mismatch; extras={extras}, missing={missing}"
                )
            candidate_ids = np.array(archive["candidate_ids"], copy=True, order="C")
            candidate_scores = np.array(
                archive["candidate_scores"], copy=True, order="C"
            )
    except E24EvaluatorContractError:
        raise
    except Exception as exc:
        raise E24EvaluatorContractError("raw feature-worker NPZ is unreadable") from exc

    return _validated_raw_arrays(
        candidate_ids,
        candidate_scores,
        source_sha256=_sha256_file(source),
    )


def load_original_raw_candidate_members(
    path: str | os.PathLike[str], *, expected_sha256: str
) -> RawCandidateArrays:
    """Read exactly two allowlisted members from the authenticated raw archive.

    The historical Rank96 archive also contains ``permutation`` and several
    training-only arrays.  Using ``numpy.load`` on that archive would expose an
    unrestricted ``NpzFile`` capability to the input broker.  This reader
    instead opens the two literal NPY members through ``zipfile`` and never
    deserializes, indexes, or returns any other member.  The whole-file digest
    is authenticated before either member is opened.
    """

    source = _require_e_drive(path, label="original Rank96 raw cache")
    expected = _validate_lower_hex_sha256(
        expected_sha256, label="expected original raw-cache SHA"
    )
    if source.suffix.lower() != ".npz" or not source.is_file():
        raise E24EvaluatorContractError(
            "original Rank96 raw cache must be an existing NPZ"
        )
    if _sha256_file(source) != expected:
        raise E24EvaluatorContractError("original raw-cache SHA mismatch")

    literal_members = ("candidate_ids.npy", "candidate_scores.npy")
    try:
        with zipfile.ZipFile(source, mode="r") as archive:
            arrays: dict[str, np.ndarray] = {}
            for member in literal_members:
                info = archive.getinfo(member)
                if (
                    info.is_dir()
                    or info.file_size <= 0
                    or info.file_size > ORIGINAL_RAW_MEMBER_BYTES_MAX
                ):
                    raise E24EvaluatorContractError(
                        f"original raw member {member} is empty or oversized"
                    )
                with archive.open(info, mode="r") as stream:
                    value = np.lib.format.read_array(stream, allow_pickle=False)
                if type(value) is not np.ndarray:
                    raise E24EvaluatorContractError(
                        f"original raw member {member} is not an ndarray"
                    )
                arrays[member] = value
    except E24EvaluatorContractError:
        raise
    except Exception as exc:
        raise E24EvaluatorContractError(
            "allowlisted original raw members are unreadable"
        ) from exc

    candidate_ids = arrays["candidate_ids.npy"]
    flat_scores = arrays["candidate_scores.npy"]
    if (
        candidate_ids.shape != (NUM_TILES, CANDIDATE_WIDTH)
        or candidate_ids.dtype != np.int64
    ):
        raise E24EvaluatorContractError(
            "original candidate_ids must be exactly int64[576,128]"
        )
    if (
        flat_scores.shape != (NUM_TILES * NUM_DIRECTIONS, CANDIDATE_WIDTH)
        or flat_scores.dtype != np.float32
    ):
        raise E24EvaluatorContractError(
            "original candidate_scores must be exactly float32[2304,128]"
        )
    candidate_ids = np.array(candidate_ids, dtype=np.int64, copy=True, order="C")
    candidate_scores = np.ascontiguousarray(
        flat_scores.reshape(NUM_TILES, NUM_DIRECTIONS, CANDIDATE_WIDTH).transpose(
            1, 0, 2
        ),
        dtype=np.float32,
    )
    return _validated_raw_arrays(
        candidate_ids,
        candidate_scores,
        source_sha256=expected,
    )


def load_original_permutation_member(
    path: str | os.PathLike[str], *, expected_sha256: str
) -> np.ndarray:
    """Load only ``permutation.npy`` from an authenticated historical archive."""

    source = _require_e_drive(path, label="permutation-only raw cache")
    expected = _validate_lower_hex_sha256(
        expected_sha256, label="expected permutation raw-cache SHA"
    )
    if source.suffix.lower() != ".npz" or not source.is_file():
        raise E24EvaluatorContractError(
            "permutation-only raw cache must be an existing NPZ"
        )
    if _sha256_file(source) != expected:
        raise E24EvaluatorContractError("permutation raw-cache SHA mismatch")
    try:
        with zipfile.ZipFile(source, mode="r") as archive:
            info = archive.getinfo("permutation.npy")
            if (
                info.is_dir()
                or info.file_size <= 0
                or info.file_size > PERMUTATION_MEMBER_BYTES_MAX
            ):
                raise E24EvaluatorContractError(
                    "permutation member is empty or oversized"
                )
            with archive.open(info, mode="r") as stream:
                value = np.lib.format.read_array(stream, allow_pickle=False)
    except E24EvaluatorContractError:
        raise
    except Exception as exc:
        raise E24EvaluatorContractError(
            "permutation-only raw member is unreadable"
        ) from exc
    if (
        type(value) is not np.ndarray
        or value.shape != (NUM_TILES,)
        or value.dtype != np.int64
        or not np.array_equal(np.sort(value), np.arange(NUM_TILES, dtype=np.int64))
    ):
        raise E24EvaluatorContractError(
            "permutation member must be an int64 tile-to-cell bijection of 0..575"
        )
    result = np.array(value, dtype=np.int64, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SanitizedRawArtifact:
    scene_id: int
    npz_path: Path
    npz_sha256: str
    manifest_path: Path
    manifest_sha256: str
    original_path: Path
    original_sha256: str
    source_scene_contract_sha256: str
    arrays: RawCandidateArrays


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, value, version=(1, 0), allow_pickle=False)
    return stream.getvalue()


def _canonical_raw_npz_bytes(
    candidate_ids: np.ndarray, candidate_scores: np.ndarray
) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in (
            ("candidate_ids", candidate_ids),
            ("candidate_scores", candidate_scores),
        ):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(value))
    return stream.getvalue()


def sanitize_raw_candidate_cache(
    *,
    scene_id: int,
    original_raw_cache_path: str | os.PathLike[str],
    expected_original_sha256: str,
    source_scene_contract_sha256: str,
    candidate_ids: np.ndarray,
    candidate_scores: np.ndarray,
    sanitized_npz_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
) -> SanitizedRawArtifact:
    """Commit an authenticated two-key raw NPZ without opening forbidden keys."""

    if type(scene_id) is not int or scene_id not in CALIBRATION_IDS:
        raise E24EvaluatorContractError("sanitizer scene must be one of E24 IDs 10..17")
    original = Path(original_raw_cache_path).resolve(strict=False)
    if not original.is_absolute() or not original.is_file():
        raise E24EvaluatorContractError("original raw cache must be an existing file")
    original_hash = _validate_lower_hex_sha256(
        expected_original_sha256, label="expected original raw-cache SHA"
    )
    if _sha256_file(original) != original_hash:
        raise E24EvaluatorContractError("original raw-cache SHA mismatch")
    scene_contract_hash = _validate_lower_hex_sha256(
        source_scene_contract_sha256, label="source scene-contract SHA"
    )
    detached = _validated_raw_arrays(
        candidate_ids,
        candidate_scores,
        source_sha256=original_hash,
    )
    destination = _require_e24_storage_path(
        sanitized_npz_path, label="sanitized raw NPZ"
    )
    manifest = _require_e24_storage_path(manifest_path, label="sanitized raw manifest")
    if destination.suffix.lower() != ".npz" or manifest.suffix.lower() != ".json":
        raise E24EvaluatorContractError("sanitized raw outputs require .npz and .json")
    npz_payload = _canonical_raw_npz_bytes(
        detached.candidate_ids, detached.candidate_scores
    )
    if len(npz_payload) > RAW_NPZ_BYTES_MAX:
        raise E24EvaluatorContractError("sanitized raw NPZ exceeds the frozen size cap")
    _atomic_write_create_or_verify(destination, npz_payload)
    npz_hash = _sha256_file(destination)
    candidate_ids_hash = _sha256_bytes(detached.candidate_ids.tobytes(order="C"))
    candidate_scores_hash = _sha256_bytes(
        detached.candidate_scores.tobytes(order="C")
    )
    finite_mask_hash = _sha256_bytes(
        np.isfinite(detached.candidate_scores[0]).astype(np.uint8).tobytes(order="C")
    )
    manifest_payload = {
        "schema": SANITIZED_RAW_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "scene_id": scene_id,
        "source_scene_contract_sha256": scene_contract_hash,
        "original_raw_cache": {
            "path": str(original),
            "bytes": original.stat().st_size,
            "sha256": original_hash,
            "parsed_by_sanitizer": False,
        },
        "sanitized_npz": {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": npz_hash,
            "keys": sorted(RAW_NPZ_KEYS),
            "canonical_zip": True,
        },
        "arrays": {
            "candidate_ids": {
                "dtype": "int64",
                "shape": [NUM_TILES, CANDIDATE_WIDTH],
                "sha256": candidate_ids_hash,
            },
            "candidate_scores": {
                "dtype": "float32",
                "shape": [NUM_DIRECTIONS, NUM_TILES, CANDIDATE_WIDTH],
                "sha256": candidate_scores_hash,
                "finite_mask_sha256": finite_mask_hash,
                "padding": "negative_infinity_only",
            },
        },
    }
    manifest_bytes = _canonical_json_bytes(manifest_payload)
    if len(manifest_bytes) > SANITIZED_RAW_MANIFEST_BYTES_MAX:
        raise AssertionError("internal sanitized manifest exceeded 64 KiB")
    _atomic_write_create_or_verify(manifest, manifest_bytes)
    return verify_sanitized_raw_artifact(manifest)


def verify_sanitized_raw_artifact(
    manifest_path: str | os.PathLike[str],
) -> SanitizedRawArtifact:
    """Authenticate source provenance and the canonical detached two-key NPZ."""

    manifest = _require_e24_storage_path(
        manifest_path, label="sanitized raw manifest"
    )
    if not manifest.is_file() or manifest.stat().st_size > SANITIZED_RAW_MANIFEST_BYTES_MAX:
        raise E24EvaluatorContractError("sanitized raw manifest is absent/oversized")
    payload = _load_canonical_commit(manifest)
    expected_top = {
        "schema",
        "schema_version",
        "status",
        "protocol_sha256",
        "scene_id",
        "source_scene_contract_sha256",
        "original_raw_cache",
        "sanitized_npz",
        "arrays",
    }
    if set(payload) != expected_top or (
        payload["schema"] != SANITIZED_RAW_SCHEMA
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["status"] != "complete"
        or payload["protocol_sha256"] != PROTOCOL_SHA256
    ):
        raise E24EvaluatorContractError("sanitized raw manifest identity drifted")
    scene_id = payload["scene_id"]
    if type(scene_id) is not int or scene_id not in CALIBRATION_IDS:
        raise E24EvaluatorContractError("sanitized raw scene identity drifted")
    scene_contract_hash = _validate_lower_hex_sha256(
        payload["source_scene_contract_sha256"], label="source scene-contract SHA"
    )
    original_record = payload["original_raw_cache"]
    sanitized_record = payload["sanitized_npz"]
    arrays_record = payload["arrays"]
    if type(original_record) is not dict or set(original_record) != {
        "path",
        "bytes",
        "sha256",
        "parsed_by_sanitizer",
    }:
        raise E24EvaluatorContractError("original raw-cache manifest record drifted")
    if original_record["parsed_by_sanitizer"] is not False:
        raise E24EvaluatorContractError("sanitizer may not parse the original raw archive")
    if type(sanitized_record) is not dict or set(sanitized_record) != {
        "path",
        "bytes",
        "sha256",
        "keys",
        "canonical_zip",
    }:
        raise E24EvaluatorContractError("sanitized NPZ manifest record drifted")
    if sanitized_record["keys"] != sorted(RAW_NPZ_KEYS) or sanitized_record[
        "canonical_zip"
    ] is not True:
        raise E24EvaluatorContractError("sanitized NPZ whitelist/canonicality drifted")
    original = Path(original_record["path"]).resolve(strict=False)
    original_hash = _validate_lower_hex_sha256(
        original_record["sha256"], label="original raw-cache SHA"
    )
    if (
        not original.is_file()
        or original.stat().st_size != original_record["bytes"]
        or _sha256_file(original) != original_hash
    ):
        raise E24EvaluatorContractError("original raw-cache provenance mismatch")
    destination = _require_e24_storage_path(
        sanitized_record["path"], label="sanitized raw NPZ"
    )
    npz_hash = _validate_lower_hex_sha256(
        sanitized_record["sha256"], label="sanitized raw NPZ SHA"
    )
    if (
        not destination.is_file()
        or destination.stat().st_size != sanitized_record["bytes"]
        or _sha256_file(destination) != npz_hash
    ):
        raise E24EvaluatorContractError("sanitized raw NPZ provenance mismatch")
    arrays = load_feature_worker_raw_npz(destination)
    expected_arrays = {
        "candidate_ids": {
            "dtype": "int64",
            "shape": [NUM_TILES, CANDIDATE_WIDTH],
            "sha256": _sha256_bytes(arrays.candidate_ids.tobytes(order="C")),
        },
        "candidate_scores": {
            "dtype": "float32",
            "shape": [NUM_DIRECTIONS, NUM_TILES, CANDIDATE_WIDTH],
            "sha256": _sha256_bytes(arrays.candidate_scores.tobytes(order="C")),
            "finite_mask_sha256": _sha256_bytes(
                np.isfinite(arrays.candidate_scores[0])
                .astype(np.uint8)
                .tobytes(order="C")
            ),
            "padding": "negative_infinity_only",
        },
    }
    if arrays_record != expected_arrays:
        raise E24EvaluatorContractError("sanitized raw array manifest drifted")
    canonical_bytes = _canonical_raw_npz_bytes(
        arrays.candidate_ids, arrays.candidate_scores
    )
    if destination.read_bytes() != canonical_bytes:
        raise E24EvaluatorContractError("sanitized raw NPZ bytes are not canonical")
    return SanitizedRawArtifact(
        scene_id=scene_id,
        npz_path=destination,
        npz_sha256=npz_hash,
        manifest_path=manifest,
        manifest_sha256=_sha256_file(manifest),
        original_path=original,
        original_sha256=original_hash,
        source_scene_contract_sha256=scene_contract_hash,
        arrays=arrays,
    )


def extract_label_free_feature_table(
    result: object,
    raw_manifest_path: str | os.PathLike[str],
    spatial_logits: np.ndarray,
    tiles_uint8: np.ndarray,
) -> selector.RelationFeatureTable:
    """Bind the strict raw whitelist to the exact label-free core extractor.

    ``result`` is deliberately checked by the core as the exact E23
    ``CandidatePoolResult``.  A ``RawScene``, report row, mapping, duck-typed
    proxy, permutation, or target cannot cross this API.
    """

    raw = verify_sanitized_raw_artifact(raw_manifest_path).arrays
    try:
        table = selector.extract_relation_features(
            result,
            raw.candidate_ids,
            raw.candidate_scores,
            spatial_logits,
            tiles_uint8,
        )
    except selector.ContextRelationSelectorError as exc:
        raise E24EvaluatorContractError("label-free feature extraction failed") from exc
    if type(table) is not selector.RelationFeatureTable:
        raise E24EvaluatorContractError("feature extractor returned the wrong exact type")
    return table


@dataclass(frozen=True)
class FoldBoundary:
    fold: int
    train_ids: tuple[int, ...]
    heldout_ids: tuple[int, int]


def fold_boundary(fold: int) -> FoldBoundary:
    if type(fold) is not int or fold not in OOF_FOLDS:
        raise E24EvaluatorContractError("fold must be one of 0,1,2,3")
    heldout = OOF_FOLDS[fold]
    train = tuple(image for image in CALIBRATION_IDS if image not in heldout)
    if len(train) != 6 or set(train).intersection(heldout):
        raise AssertionError("internal frozen fold partition drift")
    return FoldBoundary(fold=fold, train_ids=train, heldout_ids=heldout)


def frozen_lightgbm_config(fold: int) -> dict[str, Any]:
    boundary = fold_boundary(fold)
    config = dict(LIGHTGBM_BASE_CONFIG)
    seed = 1234 + boundary.fold
    config.update(
        {
            "random_state": seed,
            "data_random_seed": seed,
            "feature_fraction_seed": seed,
        }
    )
    return config


def validate_lightgbm_runtime_version(version: str | None = None) -> str:
    if version is None:
        try:
            from importlib.metadata import version as package_version

            version = package_version("lightgbm")
        except Exception as exc:
            raise E24EvaluatorContractError("LightGBM runtime is unavailable") from exc
    if version != EXPECTED_LIGHTGBM_VERSION:
        raise E24EvaluatorContractError(
            f"LightGBM must be exactly {EXPECTED_LIGHTGBM_VERSION}, got {version}"
        )
    return version


def validate_fold_training_partition(
    fold: int,
    *,
    feature_scene_ids: Sequence[int],
    label_scene_ids: Sequence[int],
) -> FoldBoundary:
    """Require all features but labels for exactly the six training scenes."""

    boundary = fold_boundary(fold)
    features = tuple(int(value) for value in feature_scene_ids)
    labels = tuple(int(value) for value in label_scene_ids)
    if len(features) != len(set(features)) or set(features) != set(CALIBRATION_IDS):
        raise E24EvaluatorContractError(
            "fold trainer must receive one label-free feature table for every E24 scene"
        )
    if len(labels) != len(set(labels)) or set(labels) != set(boundary.train_ids):
        raise E24EvaluatorContractError(
            "fold trainer labels must be exactly the six training scenes"
        )
    if set(labels).intersection(boundary.heldout_ids):
        raise E24EvaluatorContractError("held-out labels reached the fold trainer")
    if set(features).intersection(E25_SEALED_IDS) or set(labels).intersection(E25_SEALED_IDS):
        raise E24EvaluatorContractError("sealed E25 data reached E24")
    return boundary


@dataclass(frozen=True)
class FoldTrainingBatch:
    """Exact six-scene training capability with no held-out relevance arrays."""

    boundary: FoldBoundary
    table: selector.RelationFeatureTable
    relevance: np.ndarray
    row_weights: np.ndarray
    scene_row_offsets: Mapping[int, tuple[int, int]]


@dataclass(frozen=True)
class LabelOnlyRelationTruth:
    """Teacher values created outside the feature/prediction processes."""

    relevance: np.ndarray
    component_shifts: Mapping[int, tuple[int, int] | None]
    query_true_relations: tuple[tuple[int, int, int, int] | None, ...]
    true_seam_relations: frozenset[tuple[int, int, int, int]]
    pure_pair_queries: int
    true_relation_queries: int
    true_relation_rows_present: int


def _ground_truth_seam_relation_set(
    owner: np.ndarray,
    permutation: np.ndarray,
    shifts: Mapping[int, tuple[int, int] | None],
) -> frozenset[tuple[int, int, int, int]]:
    """Deduplicate component relations induced by physical upright GT seams."""

    if owner.shape != (NUM_TILES,) or owner.dtype != np.int64:
        raise E24EvaluatorContractError("owner must be exact int64[576]")
    truth = _validate_permutation(permutation, image=-1)
    tile_at_cell = np.empty(NUM_TILES, dtype=np.int64)
    tile_at_cell[truth] = np.arange(NUM_TILES, dtype=np.int64)
    relations: set[tuple[int, int, int, int]] = set()
    for row in range(24):
        for col in range(24):
            cell = row * 24 + col
            first = int(tile_at_cell[cell])
            neighbours: list[int] = []
            if col < 23:
                neighbours.append(int(tile_at_cell[cell + 1]))
            if row < 23:
                neighbours.append(int(tile_at_cell[cell + 24]))
            for second in neighbours:
                first_component = int(owner[first])
                second_component = int(owner[second])
                if first_component == second_component:
                    continue
                first_shift = shifts.get(first_component)
                second_shift = shifts.get(second_component)
                if first_shift is None or second_shift is None:
                    continue
                if first_component < second_component:
                    u, v = first_component, second_component
                    left, right = first_shift, second_shift
                else:
                    u, v = second_component, first_component
                    left, right = second_shift, first_shift
                relations.add((u, v, right[0] - left[0], right[1] - left[1]))
    return frozenset(relations)


def _validate_result_table_binding(
    result: object, table: selector.RelationFeatureTable
) -> e23_core.CandidatePoolResult:
    if type(result) is not e23_core.CandidatePoolResult:
        raise E24EvaluatorContractError(
            "label-only evaluator requires the exact E23 CandidatePoolResult"
        )
    if type(table) is not selector.RelationFeatureTable:
        raise E24EvaluatorContractError(
            "label-only evaluator requires the exact RelationFeatureTable"
        )
    try:
        selector._validate_table(table)
    except selector.ContextRelationSelectorError as exc:
        raise E24EvaluatorContractError("feature table contract failed") from exc
    if not np.array_equal(table.scene_offsets, np.asarray((0, table.rows), dtype=np.int64)):
        raise E24EvaluatorContractError("one-scene evaluation table has scene-boundary drift")
    offset_rows = np.flatnonzero(table.row_kind == selector.ROW_OFFSET)
    if not np.array_equal(
        table.hypothesis_ids[offset_rows],
        np.arange(len(result.hypotheses), dtype=np.int64),
    ):
        raise E24EvaluatorContractError(
            "feature table does not contain every E23 hypothesis exactly once"
        )
    for row, hypothesis in zip(offset_rows.tolist(), result.hypotheses):
        if (
            int(table.relation_ids[row]) != int(hypothesis.relation_id)
            or tuple(map(int, table.relations[row])) != hypothesis.relation
        ):
            raise E24EvaluatorContractError("feature row/E23 hypothesis binding drifted")
    return result


def build_label_only_relation_truth(
    result: object,
    table: selector.RelationFeatureTable,
    permutation: object,
) -> LabelOnlyRelationTruth:
    """Build one-hot query labels; never call from a feature/prediction worker."""

    value = _validate_result_table_binding(result, table)
    truth = _validate_permutation(permutation, image=-1)
    shifts: dict[int, tuple[int, int] | None] = {}
    for component in value.components:
        observed = {
            (
                int(truth[int(tile)] // 24) - int(local_row),
                int(truth[int(tile)] % 24) - int(local_col),
            )
            for tile, local_row, local_col in component.entries
        }
        shifts[int(component.component_id)] = (
            next(iter(observed)) if len(observed) == 1 else None
        )

    relevance = np.zeros(table.rows, dtype=np.int8)
    query_truth: list[tuple[int, int, int, int] | None] = []
    pure_pair_queries = 0
    for start, stop in zip(table.query_offsets[:-1], table.query_offsets[1:]):
        start_i, stop_i = int(start), int(stop)
        u, v = map(int, table.relations[start_i, :2])
        left = shifts[u]
        right = shifts[v]
        expected: tuple[int, int, int, int] | None = None
        positive_row: int | None = None
        if left is not None and right is not None:
            expected = (u, v, right[0] - left[0], right[1] - left[1])
            pure_pair_queries += 1
            matches = [
                row
                for row in range(start_i, stop_i - 1)
                if tuple(map(int, table.relations[row])) == expected
            ]
            if len(matches) > 1:
                raise E24EvaluatorContractError("query duplicates its exact true offset")
            if matches:
                positive_row = matches[0]
        relevance[positive_row if positive_row is not None else stop_i - 1] = 1
        query_truth.append(expected)
    if any(
        int(relevance[int(start) : int(stop)].sum()) != 1
        for start, stop in zip(table.query_offsets[:-1], table.query_offsets[1:])
    ):
        raise AssertionError("internal one-hot teacher algebra drift")
    relevance.setflags(write=False)
    seam_relations = _ground_truth_seam_relation_set(
        np.asarray(value.owner), truth, shifts
    )
    present_relations = {
        tuple(map(int, table.relations[row]))
        for row in np.flatnonzero(table.row_kind == selector.ROW_OFFSET).tolist()
    }.intersection(seam_relations)
    return LabelOnlyRelationTruth(
        relevance=relevance,
        component_shifts=MappingProxyType(shifts),
        query_true_relations=tuple(query_truth),
        true_seam_relations=seam_relations,
        pure_pair_queries=pure_pair_queries,
        true_relation_queries=len(seam_relations),
        true_relation_rows_present=len(present_relations),
    )


def _validate_scene_relevance(
    table: selector.RelationFeatureTable,
    relevance: object,
    *,
    image: int,
) -> tuple[np.ndarray, list[bool]]:
    if type(table) is not selector.RelationFeatureTable:
        raise E24EvaluatorContractError(
            f"scene {image} feature table has the wrong exact type"
        )
    labels = np.asarray(relevance)
    if (
        labels.shape != (table.rows,)
        or labels.dtype != np.int8
        or not labels.flags.c_contiguous
        or bool(((labels != 0) & (labels != 1)).any())
    ):
        raise E24EvaluatorContractError(
            f"scene {image} relevance must be contiguous binary int8 aligned to rows"
        )
    categories: list[bool] = []
    for start, stop in zip(table.query_offsets[:-1], table.query_offsets[1:]):
        one = labels[int(start) : int(stop)]
        if int(one.sum()) != 1:
            raise E24EvaluatorContractError(
                f"scene {image} relevance is not exactly one-hot per query"
            )
        categories.append(bool(one[-1] == 1))
    if set(categories) != {False, True}:
        raise E24EvaluatorContractError(
            f"scene {image} must contain both positive-offset and NONE-positive queries"
        )
    result = np.array(labels, dtype=np.int8, copy=True, order="C")
    result.setflags(write=False)
    return result, categories


def _per_scene_balanced_weights(
    table: selector.RelationFeatureTable, categories: Sequence[bool]
) -> np.ndarray:
    counts = {False: categories.count(False), True: categories.count(True)}
    if counts[False] <= 0 or counts[True] <= 0:
        raise E24EvaluatorContractError("both per-scene query categories are required")
    weights = np.empty(table.rows, dtype=np.float64)
    for category, start, stop in zip(
        categories, table.query_offsets[:-1], table.query_offsets[1:]
    ):
        start_i, stop_i = int(start), int(stop)
        weights[start_i:stop_i] = 1.0 / (
            2.0 * counts[category] * (stop_i - start_i)
        )
    if not math.isclose(float(math.fsum(weights.tolist())), 1.0, abs_tol=1.0e-12):
        raise AssertionError("internal per-scene weight algebra drift")
    return weights


def _canonical_float32_fold_weights(
    raw_weights: Sequence[np.ndarray],
) -> np.ndarray:
    """Materialize the frozen float32 mean-normalization exactly once."""

    if not raw_weights:
        raise E24EvaluatorContractError("fold weights require at least one scene")
    expected = np.ascontiguousarray(
        np.concatenate(tuple(raw_weights)), dtype=np.float32
    )
    if not np.isfinite(expected).all() or bool((expected <= 0).any()):
        raise E24EvaluatorContractError("raw fold weights are not finite positive")
    mean = float(expected.mean())
    if not math.isfinite(mean) or mean <= 0.0:
        raise E24EvaluatorContractError("raw fold weight mean is invalid")
    expected /= mean
    if not np.isfinite(expected).all() or bool((expected <= 0).any()):
        raise E24EvaluatorContractError(
            "normalized fold weights are not finite positive"
        )
    return expected


def build_fold_training_batch(
    fold: int,
    *,
    tables_by_scene: Mapping[int, selector.RelationFeatureTable],
    relevance_by_scene: Mapping[int, np.ndarray],
) -> FoldTrainingBatch:
    """Concatenate six scenes while preserving the frozen per-scene balance."""

    boundary = validate_fold_training_partition(
        fold,
        feature_scene_ids=tuple(tables_by_scene),
        label_scene_ids=tuple(relevance_by_scene),
    )
    tables: list[selector.RelationFeatureTable] = []
    labels: list[np.ndarray] = []
    raw_weights: list[np.ndarray] = []
    offsets: dict[int, tuple[int, int]] = {}
    cursor = 0
    for image in boundary.train_ids:
        table = tables_by_scene[image]
        one_labels, categories = _validate_scene_relevance(
            table, relevance_by_scene[image], image=image
        )
        tables.append(table)
        labels.append(one_labels)
        raw_weights.append(_per_scene_balanced_weights(table, categories))
        offsets[image] = (cursor, cursor + table.rows)
        cursor += table.rows
    try:
        combined = selector.concatenate_feature_tables(tables)
    except selector.ContextRelationSelectorError as exc:
        raise E24EvaluatorContractError("fold feature concatenation failed") from exc
    combined_labels = np.ascontiguousarray(np.concatenate(labels), dtype=np.int8)
    expected32 = _canonical_float32_fold_weights(raw_weights)
    try:
        # This exact float32 vector is also recomputed and byte-compared by
        # ``fit_lambdarank``.  Do not duplicate its rounding path here.
        weights = selector.balanced_query_row_weights(combined, combined_labels)
    except selector.ContextRelationSelectorError as exc:
        raise E24EvaluatorContractError("frozen fold weighting failed") from exc
    if (
        weights.dtype != np.float32
        or not weights.flags.c_contiguous
        or not np.isfinite(weights).all()
        or bool((weights <= 0).any())
    ):
        raise E24EvaluatorContractError("fold weights are not finite positive values")
    if not np.array_equal(weights, expected32):
        raise E24EvaluatorContractError(
            "core fold weights violate independent per-scene/category algebra"
        )
    # Every independent float64 per-scene vector has exact raw mass one;
    # the byte-exact comparison above then authenticates the specified
    # float32 materialization and global mean normalization without a tuned
    # tolerance or an ideal-float64 false positive.
    combined_labels.setflags(write=False)
    return FoldTrainingBatch(
        boundary=boundary,
        table=combined,
        relevance=combined_labels,
        row_weights=weights,
        scene_row_offsets=MappingProxyType(offsets),
    )


def fit_oof_fold(
    fold: int,
    *,
    tables_by_scene: Mapping[int, selector.RelationFeatureTable],
    relevance_by_scene: Mapping[int, np.ndarray],
) -> tuple[Any, FoldTrainingBatch]:
    """Train the fixed fold model without ever accepting held-out labels."""

    batch = build_fold_training_batch(
        fold,
        tables_by_scene=tables_by_scene,
        relevance_by_scene=relevance_by_scene,
    )
    try:
        model = selector.fit_lambdarank(
            batch.table,
            batch.relevance,
            fold=fold,
            row_weights=batch.row_weights,
        )
    except selector.ContextRelationSelectorError as exc:
        raise E24EvaluatorContractError("frozen fold training failed") from exc
    return model, batch


@dataclass(frozen=True)
class PredictionRows:
    scene_ids: np.ndarray
    row_indices: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class VerifiedPredictionCommit:
    fold: int
    train_ids: tuple[int, ...]
    heldout_ids: tuple[int, int]
    model_path: Path
    model_sha256: str
    prediction_path: Path
    prediction_sha256: str
    run_provenance: Mapping[str, Any]
    feature_sha256: Mapping[int, str]
    row_counts: Mapping[int, int]
    predictions: PredictionRows


@dataclass(frozen=True)
class VerifiedOOFCommitSet:
    """All four immutable fold transactions, verified before OOF label access."""

    commits: Mapping[int, VerifiedPredictionCommit]


def _normalize_row_counts(
    boundary: FoldBoundary, value: Mapping[int, int]
) -> dict[int, int]:
    try:
        normalized = {int(key): int(count) for key, count in value.items()}
    except Exception as exc:
        raise E24EvaluatorContractError("row counts must be an integer mapping") from exc
    if set(normalized) != set(boundary.heldout_ids):
        raise E24EvaluatorContractError("row counts must cover exactly the held-out scenes")
    if any(type(count) is not int or count <= 0 for count in normalized.values()):
        raise E24EvaluatorContractError("every held-out scene must have a positive row count")
    return normalized


def _normalize_feature_hashes(
    boundary: FoldBoundary, value: Mapping[int, str]
) -> dict[int, str]:
    normalized = {int(key): str(item) for key, item in value.items()}
    if set(normalized) != set(boundary.heldout_ids):
        raise E24EvaluatorContractError(
            "feature hashes must cover exactly the held-out scenes"
        )
    return {
        key: _validate_lower_hex_sha256(item, label=f"feature SHA for scene {key}")
        for key, item in normalized.items()
    }


_RUN_PROVENANCE_KEYS = frozenset(
    {
        "ledger_sha256",
        "run_contract_sha256",
        "core_source_sha256",
        "ordered_feature_schema_sha256",
        "lightgbm_contract_sha256",
        "canary_gate_sha256",
        "train_feature_sha256",
        "train_label_manifest_sha256",
    }
)


def _normalize_scene_hashes(
    value: object,
    *,
    expected_ids: Sequence[int],
    label: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise E24EvaluatorContractError(f"{label} must be a scene/SHA mapping")
    try:
        normalized = {int(key): item for key, item in value.items()}
    except Exception as exc:
        raise E24EvaluatorContractError(f"{label} has invalid scene keys") from exc
    if len(normalized) != len(value) or set(normalized) != set(expected_ids):
        raise E24EvaluatorContractError(
            f"{label} must cover exactly the six training scenes"
        )
    return {
        str(image): _validate_lower_hex_sha256(
            normalized[image], label=f"{label} for scene {image}"
        )
        for image in sorted(normalized)
    }


def normalize_fold_run_provenance(
    fold: int, value: object
) -> dict[str, Any]:
    """Canonicalize the source/config/train-artifact identity of one fold."""

    boundary = fold_boundary(fold)
    if not isinstance(value, Mapping) or set(value) != _RUN_PROVENANCE_KEYS:
        raise E24EvaluatorContractError("fold run-provenance field set drifted")
    output: dict[str, Any] = {
        key: _validate_lower_hex_sha256(value[key], label=key)
        for key in (
            "ledger_sha256",
            "run_contract_sha256",
            "core_source_sha256",
            "ordered_feature_schema_sha256",
            "lightgbm_contract_sha256",
            "canary_gate_sha256",
        )
    }
    output["train_feature_sha256"] = _normalize_scene_hashes(
        value["train_feature_sha256"],
        expected_ids=boundary.train_ids,
        label="training feature SHA",
    )
    output["train_label_manifest_sha256"] = _normalize_scene_hashes(
        value["train_label_manifest_sha256"],
        expected_ids=boundary.train_ids,
        label="training label-manifest SHA",
    )
    return output


def _validate_prediction_rows(
    boundary: FoldBoundary,
    rows: PredictionRows,
    row_counts: Mapping[int, int],
) -> PredictionRows:
    scene_ids = np.asarray(rows.scene_ids)
    row_indices = np.asarray(rows.row_indices)
    scores = np.asarray(rows.scores)
    if scene_ids.dtype != np.int16 or scene_ids.ndim != 1:
        raise E24EvaluatorContractError("prediction scene_ids must be one-dimensional int16")
    if row_indices.dtype != np.int64 or row_indices.ndim != 1:
        raise E24EvaluatorContractError("prediction row_indices must be one-dimensional int64")
    if scores.dtype != np.float64 or scores.ndim != 1:
        raise E24EvaluatorContractError("prediction scores must be one-dimensional float64")
    if not (len(scene_ids) == len(row_indices) == len(scores)):
        raise E24EvaluatorContractError("prediction arrays have different lengths")
    if not np.all(np.isfinite(scores)):
        raise E24EvaluatorContractError("prediction artifact contains a non-finite score")
    if not (
        scene_ids.flags.c_contiguous
        and row_indices.flags.c_contiguous
        and scores.flags.c_contiguous
    ):
        raise E24EvaluatorContractError("prediction arrays must be C-contiguous")

    expected_total = sum(row_counts.values())
    if len(scores) != expected_total:
        raise E24EvaluatorContractError("prediction artifact is incomplete")
    expected_scene_ids = np.concatenate(
        tuple(
            np.full(row_counts[image], image, dtype=np.int16)
            for image in boundary.heldout_ids
        )
    )
    expected_row_indices = np.concatenate(
        tuple(
            np.arange(row_counts[image], dtype=np.int64)
            for image in boundary.heldout_ids
        )
    )
    if not (
        np.array_equal(scene_ids, expected_scene_ids)
        and np.array_equal(row_indices, expected_row_indices)
    ):
        raise E24EvaluatorContractError(
            "prediction rows must be complete contiguous held-out blocks in "
            "the frozen fold order"
        )

    scene_copy = np.array(scene_ids, dtype=np.int16, copy=True, order="C")
    index_copy = np.array(row_indices, dtype=np.int64, copy=True, order="C")
    score_copy = np.array(scores, dtype=np.float64, copy=True, order="C")
    scene_copy.setflags(write=False)
    index_copy.setflags(write=False)
    score_copy.setflags(write=False)
    return PredictionRows(scene_ids=scene_copy, row_indices=index_copy, scores=score_copy)


def _prediction_npz_bytes(
    *, fold: int, model_sha256: str, rows: PredictionRows
) -> bytes:
    stream = io.BytesIO()
    members = (
        ("schema", np.asarray(PREDICTION_SCHEMA)),
        ("fold", np.asarray(fold, dtype=np.int8)),
        ("model_sha256", np.asarray(model_sha256)),
        ("scene_ids", rows.scene_ids),
        ("row_indices", rows.row_indices),
        ("scores", rows.scores),
    )
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in members:
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(value))
    return stream.getvalue()


def commit_fold_predictions(
    *,
    fold: int,
    model_path: str | os.PathLike[str],
    prediction_path: str | os.PathLike[str],
    commit_path: str | os.PathLike[str],
    run_provenance: Mapping[str, Any],
    feature_sha256: Mapping[int, str],
    row_counts: Mapping[int, int],
    rows: PredictionRows,
) -> dict[str, Any]:
    """Create the prediction artifact, then atomically publish its commit record.

    The canonical JSON commit is the transaction point.  Until it exists, no
    held-out label reader is authorized to run.
    """

    boundary = fold_boundary(fold)
    model = _require_e24_storage_path(model_path, label="fold model")
    prediction = _require_e24_storage_path(
        prediction_path, label="prediction artifact"
    )
    commit = _require_e24_storage_path(commit_path, label="prediction commit")
    if not model.is_file() or model.stat().st_size <= 0:
        raise E24EvaluatorContractError("fold model must be a complete nonempty file")
    if prediction == commit or model in {prediction, commit}:
        raise E24EvaluatorContractError("model, prediction, and commit paths must differ")
    normalized_counts = _normalize_row_counts(boundary, row_counts)
    normalized_features = _normalize_feature_hashes(boundary, feature_sha256)
    normalized_provenance = normalize_fold_run_provenance(fold, run_provenance)
    validated_rows = _validate_prediction_rows(boundary, rows, normalized_counts)
    model_hash = _sha256_file(model)
    model_size = model.stat().st_size
    if model_size > ALL_ARTIFACT_BYTES_MAX:
        raise E24EvaluatorContractError("fold model exceeds the E24 artifact cap")

    prediction_payload = _prediction_npz_bytes(
        fold=fold, model_sha256=model_hash, rows=validated_rows
    )
    if model_size + len(prediction_payload) > ALL_ARTIFACT_BYTES_MAX:
        raise E24EvaluatorContractError("fold transaction exceeds the E24 artifact cap")
    _atomic_write_create_or_verify(prediction, prediction_payload)
    prediction_hash = _sha256_file(prediction)
    prediction_size = prediction.stat().st_size

    payload: dict[str, Any] = {
        "schema": COMMIT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "fold": fold,
        "train_ids": list(boundary.train_ids),
        "heldout_ids": list(boundary.heldout_ids),
        "run_provenance": normalized_provenance,
        "model": {
            "path": str(model),
            "bytes": model_size,
            "sha256": model_hash,
        },
        "predictions": {
            "path": str(prediction),
            "bytes": prediction_size,
            "sha256": prediction_hash,
            "finite": True,
            "complete": True,
            "row_counts": {
                str(key): normalized_counts[key] for key in sorted(normalized_counts)
            },
        },
        "feature_sha256": {
            str(key): normalized_features[key] for key in sorted(normalized_features)
        },
    }
    _atomic_write_create_or_verify(commit, _canonical_json_bytes(payload))
    return payload


def _load_canonical_commit(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except Exception as exc:
        raise E24EvaluatorContractError("prediction commit is unreadable") from exc
    if type(value) is not dict or raw != _canonical_json_bytes(value):
        raise E24EvaluatorContractError("prediction commit must be canonical JSON")
    return value


def _load_prediction_npz(path: Path) -> tuple[int, str, PredictionRows]:
    expected_keys = {
        "schema",
        "fold",
        "model_sha256",
        "scene_ids",
        "row_indices",
        "scores",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_keys:
                raise E24EvaluatorContractError("prediction NPZ field set drifted")
            schema = str(np.asarray(archive["schema"]).item())
            fold = int(np.asarray(archive["fold"]).item())
            model_hash = str(np.asarray(archive["model_sha256"]).item())
            rows = PredictionRows(
                scene_ids=np.array(archive["scene_ids"], copy=True, order="C"),
                row_indices=np.array(archive["row_indices"], copy=True, order="C"),
                scores=np.array(archive["scores"], copy=True, order="C"),
            )
    except E24EvaluatorContractError:
        raise
    except Exception as exc:
        raise E24EvaluatorContractError("prediction NPZ is unreadable") from exc
    if schema != PREDICTION_SCHEMA:
        raise E24EvaluatorContractError("prediction NPZ schema drifted")
    if path.read_bytes() != _prediction_npz_bytes(
        fold=fold, model_sha256=model_hash, rows=rows
    ):
        raise E24EvaluatorContractError("prediction NPZ bytes are not canonical")
    return fold, model_hash, rows


def verify_prediction_commit(
    commit_path: str | os.PathLike[str],
    *,
    expected_fold: int | None = None,
    expected_run_provenance: Mapping[str, Any] | None = None,
) -> VerifiedPredictionCommit:
    """Authenticate a complete prediction transaction without reading labels."""

    commit_file = _require_e24_storage_path(commit_path, label="prediction commit")
    if not commit_file.is_file():
        raise E24EvaluatorContractError("complete prediction commit is absent")
    payload = _load_canonical_commit(commit_file)
    expected_top = {
        "schema",
        "schema_version",
        "status",
        "protocol_sha256",
        "fold",
        "train_ids",
        "heldout_ids",
        "run_provenance",
        "model",
        "predictions",
        "feature_sha256",
    }
    if set(payload) != expected_top:
        raise E24EvaluatorContractError("prediction commit field set drifted")
    if (
        payload["schema"] != COMMIT_SCHEMA
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["status"] != "complete"
        or payload["protocol_sha256"] != PROTOCOL_SHA256
    ):
        raise E24EvaluatorContractError("prediction commit identity drifted")
    if type(payload["fold"]) is not int:
        raise E24EvaluatorContractError("prediction fold must be an integer")
    boundary = fold_boundary(payload["fold"])
    if expected_fold is not None and boundary.fold != expected_fold:
        raise E24EvaluatorContractError("prediction commit is for the wrong fold")
    if payload["train_ids"] != list(boundary.train_ids):
        raise E24EvaluatorContractError("prediction commit training split drifted")
    if payload["heldout_ids"] != list(boundary.heldout_ids):
        raise E24EvaluatorContractError("prediction commit held-out split drifted")
    run_provenance = normalize_fold_run_provenance(
        boundary.fold, payload["run_provenance"]
    )
    if expected_run_provenance is not None and run_provenance != normalize_fold_run_provenance(
        boundary.fold, expected_run_provenance
    ):
        raise E24EvaluatorContractError(
            "prediction commit belongs to a stale/different run provenance"
        )

    model_record = payload["model"]
    prediction_record = payload["predictions"]
    if type(model_record) is not dict or set(model_record) != {"path", "bytes", "sha256"}:
        raise E24EvaluatorContractError("model commit record drifted")
    if type(prediction_record) is not dict or set(prediction_record) != {
        "path",
        "bytes",
        "sha256",
        "finite",
        "complete",
        "row_counts",
    }:
        raise E24EvaluatorContractError("prediction artifact record drifted")
    if prediction_record["finite"] is not True or prediction_record["complete"] is not True:
        raise E24EvaluatorContractError("prediction artifact is not committed complete/finite")

    model = _require_e24_storage_path(model_record["path"], label="committed model")
    prediction = _require_e24_storage_path(
        prediction_record["path"], label="committed prediction artifact"
    )
    model_hash = _validate_lower_hex_sha256(model_record["sha256"], label="model SHA")
    prediction_hash = _validate_lower_hex_sha256(
        prediction_record["sha256"], label="prediction SHA"
    )
    if not model.is_file() or model.stat().st_size != model_record["bytes"]:
        raise E24EvaluatorContractError("committed model size mismatch")
    if _sha256_file(model) != model_hash:
        raise E24EvaluatorContractError("committed model SHA mismatch")
    if not prediction.is_file() or prediction.stat().st_size != prediction_record["bytes"]:
        raise E24EvaluatorContractError("committed prediction size mismatch")
    if _sha256_file(prediction) != prediction_hash:
        raise E24EvaluatorContractError("committed prediction SHA mismatch")

    try:
        row_counts_input = {
            int(key): value for key, value in prediction_record["row_counts"].items()
        }
        feature_hashes_input = {
            int(key): value for key, value in payload["feature_sha256"].items()
        }
    except Exception as exc:
        raise E24EvaluatorContractError("commit scene-key mapping is invalid") from exc
    row_counts = _normalize_row_counts(boundary, row_counts_input)
    feature_hashes = _normalize_feature_hashes(boundary, feature_hashes_input)
    npz_fold, embedded_model_hash, raw_rows = _load_prediction_npz(prediction)
    if npz_fold != boundary.fold or embedded_model_hash != model_hash:
        raise E24EvaluatorContractError("prediction NPZ is bound to a different model/fold")
    rows = _validate_prediction_rows(boundary, raw_rows, row_counts)
    return VerifiedPredictionCommit(
        fold=boundary.fold,
        train_ids=boundary.train_ids,
        heldout_ids=boundary.heldout_ids,
        model_path=model,
        model_sha256=model_hash,
        prediction_path=prediction,
        prediction_sha256=prediction_hash,
        run_provenance=MappingProxyType(run_provenance),
        feature_sha256=MappingProxyType(dict(feature_hashes)),
        row_counts=MappingProxyType(dict(row_counts)),
        predictions=rows,
    )


def verify_all_oof_commits(
    commit_paths: Mapping[int, str | os.PathLike[str]],
    *,
    expected_run_provenance: Mapping[int, Mapping[str, Any]] | None = None,
) -> VerifiedOOFCommitSet:
    """Authenticate all four folds and exactly one prediction for every scene."""

    if set(commit_paths) != set(OOF_FOLDS):
        raise E24EvaluatorContractError("OOF barrier requires exactly folds 0,1,2,3")
    if expected_run_provenance is not None and set(expected_run_provenance) != set(
        OOF_FOLDS
    ):
        raise E24EvaluatorContractError(
            "OOF expected run provenance requires exactly folds 0,1,2,3"
        )
    verified: dict[int, VerifiedPredictionCommit] = {}
    seen_scenes: list[int] = []
    seen_models: set[Path] = set()
    seen_predictions: set[Path] = set()
    for fold in sorted(OOF_FOLDS):
        item = verify_prediction_commit(
            commit_paths[fold],
            expected_fold=fold,
            expected_run_provenance=(
                expected_run_provenance[fold]
                if expected_run_provenance is not None
                else None
            ),
        )
        if item.model_path in seen_models or item.prediction_path in seen_predictions:
            raise E24EvaluatorContractError("OOF folds reuse a model or prediction artifact")
        seen_models.add(item.model_path)
        seen_predictions.add(item.prediction_path)
        seen_scenes.extend(item.heldout_ids)
        verified[fold] = item
    if tuple(sorted(seen_scenes)) != CALIBRATION_IDS or len(seen_scenes) != len(
        set(seen_scenes)
    ):
        raise E24EvaluatorContractError(
            "OOF commits do not cover every E24 scene exactly once"
        )
    return VerifiedOOFCommitSet(commits=MappingProxyType(verified))


def _validate_permutation(value: object, *, image: int) -> np.ndarray:
    permutation = np.asarray(value)
    if permutation.shape != (NUM_TILES,) or permutation.dtype != np.int64:
        raise E24EvaluatorContractError(
            f"held-out permutation for scene {image} must be int64[576]"
        )
    if not np.array_equal(np.sort(permutation), np.arange(NUM_TILES, dtype=np.int64)):
        raise E24EvaluatorContractError(
            f"held-out permutation for scene {image} is not a bijection"
        )
    result = np.array(permutation, dtype=np.int64, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class StructuralSceneCounts:
    """Label-only integer evidence for one already-committed OOF scene."""

    image: int
    fold: int
    provenance_ok: bool
    query_canonical_onehot: bool
    orientation_ok: bool
    fold_isolated: bool
    finite_output: bool
    dsu_legal: bool
    legal_origin: bool
    component_count: int
    geometry_hypotheses: int
    proposed_relations: int
    true_proposed_relations: int
    true_relations: int
    accepted_relations: int
    true_accepted_relations: int
    exact_connected_tiles: int
    accepted_graph_vertices: int
    accepted_graph_components: int


def _selection_is_true(
    selection: selector.SelectedRelation,
    true_relations: frozenset[tuple[int, int, int, int]],
) -> bool:
    return selection.relation in true_relations


def _accepted_graph_counts(
    accepted: Sequence[selector.SelectedRelation],
) -> tuple[int, int]:
    vertices = sorted({item.u for item in accepted} | {item.v for item in accepted})
    if not vertices:
        return 0, 0
    parent = {value: value for value in vertices}

    def find(value: int) -> int:
        while parent[value] != value:
            value = parent[value]
        return value

    for item in accepted:
        root_u, root_v = find(item.u), find(item.v)
        if root_u != root_v:
            keep, drop = (root_u, root_v) if root_u < root_v else (root_v, root_u)
            parent[drop] = keep
    components = len({find(value) for value in vertices})
    return len(vertices), components


def evaluate_committed_structural_scene(
    verified: VerifiedPredictionCommit,
    *,
    image: int,
    result: object,
    feature_path: str | os.PathLike[str],
    permutation: object,
    candidate_pool_provenance_ok: bool,
) -> StructuralSceneCounts:
    """Derive one structural row from an authenticated prediction transaction."""

    if type(verified) is not VerifiedPredictionCommit:
        raise E24EvaluatorContractError("structural evaluator requires a verified commit")
    if type(image) is not int or image not in verified.heldout_ids:
        raise E24EvaluatorContractError("scene is not held out by the verified fold")
    if type(candidate_pool_provenance_ok) is not bool:
        raise E24EvaluatorContractError("candidate-pool provenance flag must be boolean")
    feature_file = _require_e24_storage_path(feature_path, label="feature artifact")
    if not feature_file.is_file():
        raise E24EvaluatorContractError("committed feature artifact is absent")
    expected_feature_hash = verified.feature_sha256[image]
    if _sha256_file(feature_file) != expected_feature_hash:
        raise E24EvaluatorContractError("committed feature artifact SHA mismatch")
    if feature_file.stat().st_size > FEATURE_CACHE_BYTES_MAX:
        raise E24EvaluatorContractError("feature cache exceeds the E24 cap")
    try:
        table = selector.load_feature_table_npz(feature_file)
    except (selector.ContextRelationSelectorError, OSError, ValueError) as exc:
        raise E24EvaluatorContractError("feature artifact failed strict loading") from exc
    value = _validate_result_table_binding(result, table)

    mask = verified.predictions.scene_ids == image
    row_indices = verified.predictions.row_indices[mask]
    scores = verified.predictions.scores[mask]
    if not np.array_equal(row_indices, np.arange(table.rows, dtype=np.int64)):
        raise E24EvaluatorContractError("committed scores do not cover the feature table")
    try:
        decoded = selector.decode_relation_scores(value, table, scores)
    except selector.ContextRelationSelectorError as exc:
        raise E24EvaluatorContractError("frozen relation decoder failed") from exc
    teacher = build_label_only_relation_truth(value, table, permutation)

    proposed = tuple(decoded.attempted)
    accepted = tuple(outcome.selection for outcome in decoded.outcomes if outcome.accepted)
    if len(decoded.outcomes) != len(proposed):
        raise E24EvaluatorContractError("decoder did not account for every attempted relation")
    valid_reasons = {"tree", "cycle", "conflict", "contact", "collision", "span"}
    dsu_legal = all(
        outcome.reason in valid_reasons
        and outcome.accepted == (outcome.reason in {"tree", "cycle"})
        and outcome.tree_merge == (outcome.reason == "tree")
        and outcome.cycle == (outcome.reason == "cycle")
        for outcome in decoded.outcomes
    )
    if len(proposed) > 2 * (len(value.components) - 1):
        dsu_legal = False

    truth_permutation = _validate_permutation(permutation, image=image)
    largest_exact = 0
    legal_origin = True
    observed_tiles: set[int] = set()
    for component in decoded.components:
        if type(component) is not dict or not component:
            raise E24EvaluatorContractError("decoded component is empty or mutable-type drifted")
        observed_tiles.update(map(int, component))
        rows = [int(position[0]) for position in component.values()]
        cols = [int(position[1]) for position in component.values()]
        if min(rows) != 0 or min(cols) != 0 or max(rows) >= 24 or max(cols) >= 24:
            legal_origin = False
        offsets = {
            (
                int(truth_permutation[int(tile)] // 24) - int(position[0]),
                int(truth_permutation[int(tile)] % 24) - int(position[1]),
            )
            for tile, position in component.items()
        }
        if len(offsets) == 1:
            largest_exact = max(largest_exact, len(component))
    if observed_tiles != set(range(NUM_TILES)):
        raise E24EvaluatorContractError("decoded components do not partition all tiles")

    vertices, graph_components = _accepted_graph_counts(accepted)
    return StructuralSceneCounts(
        image=image,
        fold=verified.fold,
        provenance_ok=candidate_pool_provenance_ok,
        query_canonical_onehot=all(
            int(teacher.relevance[int(start) : int(stop)].sum()) == 1
            for start, stop in zip(table.query_offsets[:-1], table.query_offsets[1:])
        ),
        orientation_ok=True,
        fold_isolated=image in verified.heldout_ids and image not in verified.train_ids,
        finite_output=bool(np.isfinite(scores).all()),
        dsu_legal=dsu_legal,
        legal_origin=legal_origin,
        component_count=len(value.components),
        geometry_hypotheses=len(value.hypotheses),
        proposed_relations=len(proposed),
        true_proposed_relations=sum(
            _selection_is_true(item, teacher.true_seam_relations) for item in proposed
        ),
        true_relations=teacher.true_relation_queries,
        accepted_relations=len(accepted),
        true_accepted_relations=sum(
            _selection_is_true(item, teacher.true_seam_relations) for item in accepted
        ),
        exact_connected_tiles=largest_exact,
        accepted_graph_vertices=vertices,
        accepted_graph_components=graph_components,
    )


def _exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise E24EvaluatorContractError(f"{label} must be an integer >= {minimum}")
    return value


def structural_scene_metrics(row: StructuralSceneCounts) -> dict[str, Any]:
    """Validate raw counts and derive every frozen per-scene structural metric."""

    if type(row) is not StructuralSceneCounts:
        raise E24EvaluatorContractError("structural row has the wrong exact type")
    if row.image not in CALIBRATION_IDS:
        raise E24EvaluatorContractError("structural row contains a non-E24 scene")
    boundary = fold_boundary(row.fold)
    if row.image not in boundary.heldout_ids:
        raise E24EvaluatorContractError("structural row is assigned to the wrong OOF fold")
    integrity_fields = (
        row.provenance_ok,
        row.query_canonical_onehot,
        row.orientation_ok,
        row.fold_isolated,
        row.finite_output,
        row.dsu_legal,
        row.legal_origin,
    )
    if any(type(value) is not bool for value in integrity_fields):
        raise E24EvaluatorContractError("integrity flags must be exact booleans")

    component_count = _exact_int(row.component_count, label="component count", minimum=1)
    geometry = _exact_int(
        row.geometry_hypotheses, label="geometry hypotheses", minimum=1
    )
    proposed = _exact_int(row.proposed_relations, label="proposed relations", minimum=1)
    true_proposed = _exact_int(
        row.true_proposed_relations, label="true proposed relations"
    )
    true_relations = _exact_int(row.true_relations, label="true relations", minimum=1)
    accepted = _exact_int(row.accepted_relations, label="accepted relations", minimum=1)
    true_accepted = _exact_int(
        row.true_accepted_relations, label="true accepted relations"
    )
    connected_tiles = _exact_int(
        row.exact_connected_tiles, label="exact connected tiles", minimum=1
    )
    vertices = _exact_int(
        row.accepted_graph_vertices, label="accepted graph vertices", minimum=1
    )
    graph_components = _exact_int(
        row.accepted_graph_components,
        label="accepted graph components",
        minimum=1,
    )
    if component_count > NUM_TILES or geometry > GEOMETRY_HYPOTHESES_MAX_EACH:
        raise E24EvaluatorContractError("component/geometry cap failed")
    if proposed > 2 * (component_count - 1):
        raise E24EvaluatorContractError("proposed relation count exceeds the frozen cap")
    if not (0 <= true_proposed <= proposed and true_proposed <= true_relations):
        raise E24EvaluatorContractError("true proposed relation counts are inconsistent")
    if not (0 <= true_accepted <= accepted <= proposed):
        raise E24EvaluatorContractError("accepted relation counts are inconsistent")
    if true_accepted > true_relations or connected_tiles > NUM_TILES:
        raise E24EvaluatorContractError("truth/coverage counts are inconsistent")
    if not (1 <= graph_components <= vertices <= component_count):
        raise E24EvaluatorContractError("accepted graph V/K counts are inconsistent")
    cycle_rank = accepted - vertices + graph_components
    if cycle_rank < 0:
        raise E24EvaluatorContractError("accepted graph has a negative cycle rank")
    cycle_denominator = max(1, vertices - graph_components)

    values: dict[str, Any] = {
        "image": row.image,
        "fold": row.fold,
        "integrity_legal": all(integrity_fields),
        "component_count": component_count,
        "geometry_hypotheses": geometry,
        "proposed_relations": proposed,
        "true_proposed_relations": true_proposed,
        "true_relations": true_relations,
        "accepted_relations": accepted,
        "true_accepted_relations": true_accepted,
        "exact_connected_tiles": connected_tiles,
        "proposed_precision": float(true_proposed / proposed),
        "true_relation_recall": float(true_proposed / true_relations),
        "accepted_precision": float(true_accepted / accepted),
        "exact_connected_coverage": float(connected_tiles / NUM_TILES),
        "cycle_rank": cycle_rank,
        "cycle_rank_ratio": float(cycle_rank / cycle_denominator),
    }
    if not all(
        math.isfinite(value)
        for key, value in values.items()
        if key
        in {
            "proposed_precision",
            "true_relation_recall",
            "accepted_precision",
            "exact_connected_coverage",
            "cycle_rank_ratio",
        }
    ):
        raise E24EvaluatorContractError("derived structural metric is non-finite")
    return values


def summarize_structural(rows: Sequence[StructuralSceneCounts]) -> dict[str, Any]:
    """Aggregate exactly eight OOF rows without constructing a board/image metric."""

    values = [structural_scene_metrics(row) for row in rows]
    if len(values) != 8 or sorted(item["image"] for item in values) != list(
        CALIBRATION_IDS
    ):
        raise E24EvaluatorContractError(
            "structural summary requires each E24 scene exactly once"
        )
    if len({item["image"] for item in values}) != 8:
        raise E24EvaluatorContractError("structural summary duplicates a scene")

    def mean(key: str) -> float:
        return float(math.fsum(float(item[key]) for item in values) / len(values))

    output = {
        "completed_scenes": len(values),
        "complete_integrity_legal_scenes": sum(
            bool(item["integrity_legal"]) for item in values
        ),
        "nonempty_proposal_scenes": sum(item["proposed_relations"] > 0 for item in values),
        "nonempty_accepted_scenes": sum(item["accepted_relations"] > 0 for item in values),
        "all_geometry_caps": all(
            item["geometry_hypotheses"] <= GEOMETRY_HYPOTHESES_MAX_EACH
            for item in values
        ),
        "mean_proposed_precision": mean("proposed_precision"),
        "worst_proposed_precision": min(item["proposed_precision"] for item in values),
        "mean_true_relation_recall": mean("true_relation_recall"),
        "worst_true_relation_recall": min(
            item["true_relation_recall"] for item in values
        ),
        "mean_exact_connected_coverage": mean("exact_connected_coverage"),
        "worst_exact_connected_coverage": min(
            item["exact_connected_coverage"] for item in values
        ),
        "mean_cycle_rank_ratio": mean("cycle_rank_ratio"),
        "mean_accepted_precision": mean("accepted_precision"),
        "rows": values,
    }
    return output


def structural_decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Apply only the predeclared structural gates; diagnostics cannot rescue FAIL."""

    required = {
        "completed_scenes",
        "complete_integrity_legal_scenes",
        "nonempty_proposal_scenes",
        "nonempty_accepted_scenes",
        "all_geometry_caps",
        "mean_proposed_precision",
        "worst_proposed_precision",
        "mean_true_relation_recall",
        "worst_true_relation_recall",
        "mean_exact_connected_coverage",
        "worst_exact_connected_coverage",
        "mean_cycle_rank_ratio",
        "mean_accepted_precision",
        "rows",
    }
    if type(summary) is not dict or set(summary) != required:
        raise E24EvaluatorContractError("structural summary field set drifted")
    finite_fields = (
        "mean_proposed_precision",
        "worst_proposed_precision",
        "mean_true_relation_recall",
        "worst_true_relation_recall",
        "mean_exact_connected_coverage",
        "worst_exact_connected_coverage",
        "mean_cycle_rank_ratio",
        "mean_accepted_precision",
    )
    observed: dict[str, float] = {}
    for key in finite_fields:
        try:
            observed[key] = float(summary[key])
        except (TypeError, ValueError) as exc:
            raise E24EvaluatorContractError(f"{key} is not numeric") from exc
        if not math.isfinite(observed[key]):
            raise E24EvaluatorContractError(f"{key} is non-finite")
    checks = {
        "completed_scenes": summary["completed_scenes"] == 8,
        "complete_integrity_legal_scenes": summary[
            "complete_integrity_legal_scenes"
        ]
        == 8,
        "nonempty_proposal_scenes": summary["nonempty_proposal_scenes"] == 8,
        "nonempty_accepted_scenes": summary["nonempty_accepted_scenes"] == 8,
        "all_geometry_caps": summary["all_geometry_caps"] is True,
        "mean_proposed_precision": observed["mean_proposed_precision"]
        >= float(STRUCTURAL_GATES["proposed_precision_mean_min"]),
        "worst_proposed_precision": observed["worst_proposed_precision"]
        >= float(STRUCTURAL_GATES["proposed_precision_worst_min"]),
        "mean_true_relation_recall": observed["mean_true_relation_recall"]
        >= float(STRUCTURAL_GATES["true_relation_recall_mean_min"]),
        "worst_true_relation_recall": observed["worst_true_relation_recall"]
        >= float(STRUCTURAL_GATES["true_relation_recall_worst_min"]),
        "mean_exact_connected_coverage": observed["mean_exact_connected_coverage"]
        >= float(STRUCTURAL_GATES["exact_connected_coverage_mean_min"]),
        "worst_exact_connected_coverage": observed["worst_exact_connected_coverage"]
        >= float(STRUCTURAL_GATES["exact_connected_coverage_worst_min"]),
        "mean_cycle_rank_ratio": observed["mean_cycle_rank_ratio"]
        >= float(STRUCTURAL_GATES["mean_cycle_rank_ratio_min"]),
    }
    return {
        "stage": "go_staged_end_to_end" if all(checks.values()) else "kill_crs_v1",
        "passed": all(checks.values()),
        "checks": checks,
    }


OOFEvaluationResult = TypeVar("OOFEvaluationResult")


def evaluate_oof_after_all_commits(
    commit_paths: Mapping[int, str | os.PathLike[str]],
    *,
    expected_run_provenance: Mapping[int, Mapping[str, Any]] | None = None,
    permutation_loader: Callable[[int], np.ndarray],
    evaluator: Callable[
        [VerifiedOOFCommitSet, Mapping[int, np.ndarray]], OOFEvaluationResult
    ],
) -> OOFEvaluationResult:
    """Cross the OOF label boundary only after the global four-fold barrier."""

    verified = verify_all_oof_commits(
        commit_paths, expected_run_provenance=expected_run_provenance
    )
    permutations: dict[int, np.ndarray] = {}
    for image in CALIBRATION_IDS:
        permutations[image] = _validate_permutation(
            permutation_loader(image), image=image
        )
    return evaluator(verified, MappingProxyType(permutations))


__all__ = [
    "ALL_ARTIFACT_BYTES_MAX",
    "CALIBRATION_IDS",
    "COMMIT_SCHEMA",
    "E24EvaluatorContractError",
    "E24_EVALUATOR_PROTOCOL",
    "E25_CANONICAL_RECORDS_SHA256",
    "E25_NEWLINE_LIST_SHA256",
    "E25_SEALED_IDS",
    "END_TO_END_GATES",
    "EXPECTED_LIGHTGBM_VERSION",
    "FEATURE_CACHE_BYTES_MAX",
    "FoldBoundary",
    "FoldTrainingBatch",
    "GEOMETRY_HYPOTHESES_MAX_EACH",
    "LIGHTGBM_BASE_CONFIG",
    "OOF_FOLDS",
    "PROTOCOL_SHA256",
    "PredictionRows",
    "RawCandidateArrays",
    "SanitizedRawArtifact",
    "STORAGE_ROOT",
    "STRUCTURAL_GATES",
    "StructuralSceneCounts",
    "LabelOnlyRelationTruth",
    "VerifiedPredictionCommit",
    "VerifiedOOFCommitSet",
    "commit_fold_predictions",
    "build_fold_training_batch",
    "build_label_only_relation_truth",
    "evaluate_oof_after_all_commits",
    "evaluate_committed_structural_scene",
    "extract_label_free_feature_table",
    "fit_oof_fold",
    "fold_boundary",
    "frozen_lightgbm_config",
    "load_feature_worker_raw_npz",
    "normalize_fold_run_provenance",
    "sanitize_raw_candidate_cache",
    "structural_decision",
    "structural_scene_metrics",
    "summarize_structural",
    "validate_e24_runtime_paths",
    "validate_lightgbm_runtime_version",
    "validate_fold_training_partition",
    "verify_prediction_commit",
    "verify_all_oof_commits",
    "verify_sanitized_raw_artifact",
]
