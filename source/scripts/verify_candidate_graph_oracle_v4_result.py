#!/usr/bin/env python3
"""Independent fail-closed verifier for the candidate-graph oracle protocol.

The verifier deliberately does not import ``evaluate_candidate_graph_oracle``.
It has two entry points:

* ``phase-a`` verifies the finalized input-only envelope, its out-of-band
  anchors, Kaggle launch/readback receipt, runner wrapper, every array/render,
  and independently reconstructs the seven-origin candidate union from the
  frozen matrices and layouts.  It never accepts a label argument.
* ``phase-b`` first verifies Phase A and the irreversible lifecycle through
  ``LABEL_ACCESS``.  Only after verifying the durable target-access marker does
  it construct label paths.  It then independently regenerates the opaque
  fixtures, recomputes the scientific result and continuation gate, and binds
  the sandboxed runner's canonical stdout attestation outside the evaluator
  output tree.

Every accepted JSON object has an exact schema and canonical encoding.  Every
listed file is opened relative to an anchored directory descriptor with
``O_NOFOLLOW`` and must be a one-link regular file.  Missing, extra, aliased,
non-finite, or hash-inconsistent evidence is a hard failure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
import hashlib
import hmac
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = (
    REPO_ROOT
    / "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src"
)
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from puzzle_assembly.geometry import (  # noqa: E402
    GRID,
    TILE,
    TILE_COUNT,
    true_neighbour_slots,
    validate_permutation,
)
from puzzle_assembly.metrics import layout_metrics  # noqa: E402
from puzzle_assembly.panels import make_exact_panel  # noqa: E402


EXPECTED_PROTOCOL_INSTANCE_ID = "6c0fe4e8524ce39d830d9a5bee118d8b"
EXPECTED_FROZEN_CONTRACT_SHA256 = (
    "2070c1b4ff0a3ff42c5ffdd6d611c214c02dbb99b2c51a88f38a862bb1f8a05c"
)
EXPECTED_RESERVATION_RUNNER_SHA256 = (
    "adf4d61a528f91ce5a4c282b0f3999f8bcdbe8c18d5429a98210dc6b991ab460"
)
EXPECTED_RESERVATION_RECEIPT_SHA256: str | None = "4b38963476279c7ec7c6f4b17baca9997651aad4b48cf152c9165d2dca8cbf80"
EXPECTED_NAMES_SHA256 = (
    "149ca83873e5e2e79e6458098c5c758b935af5d9131e093f5eb34fef82b76634"
)

PHASE_A_MANIFEST = "FROZEN_CANDIDATE_GRAPH_MANIFEST.json"
PHASE_A_SHARD_MANIFEST = "FROZEN_CANDIDATE_GRAPH_SHARD_MANIFEST.json"
INPUT_MANIFEST = "fixture_input_manifest.json"
LABEL_MANIFEST = "fixture_label_manifest.json"
FIXTURE_LOCK = "fixture_lock.json"
FIXTURE_PREP_MARKER = "FIXTURE_PIXEL_ACCESS_STARTED.json"
MASTER_SECRET = "FIXTURE_MASTER_SECRET.bin"
TARGET_MARKER = "TARGET_ACCESS_STARTED.json"
REPORT_NAME = "candidate_graph_oracle_ceiling_report.json"
RAW_PUSH_RESPONSE_FIELDS = {
    "ref",
    "url",
    "version_number",
    "error",
    "invalid_tags",
    "invalid_dataset_sources",
    "invalid_competition_sources",
    "invalid_kernel_sources",
    "invalid_model_sources",
    "kernel_id",
}

PANELS = ("primary_kornia", "independent_libjpeg")
ORIGIN_BITS: dict[str, int] = {
    "c1_out32": 1,
    "hbt_out32": 2,
    "c1_in8": 4,
    "hbt_in8": 8,
    "softcycle": 16,
    "qap_w4": 32,
    "qap_w1": 64,
}
ORIGIN_NAMES = {
    "c1_out32": "C1_OUT32",
    "hbt_out32": "HBT_OUT32",
    "c1_in8": "C1_IN8",
    "hbt_in8": "HBT_IN8",
    "softcycle": "SOFTCYCLE_LAYOUT",
    "qap_w4": "QAP_W4_LAYOUT",
    "qap_w1": "QAP_W1_LAYOUT",
}
ALL_ORIGIN_BITS = sum(ORIGIN_BITS.values())
EXPECTED_PREDEDUP_COUNTS = {
    "c1_out32": 36_864,
    "hbt_out32": 36_864,
    "c1_in8": 9_216,
    "hbt_in8": 9_216,
    "softcycle": 1_104,
    "qap_w4": 1_104,
    "qap_w1": 1_104,
}

DERIVED_ARRAY_SPECS: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {
    "c1_right": (np.dtype("float32"), (TILE_COUNT, TILE_COUNT)),
    "c1_down": (np.dtype("float32"), (TILE_COUNT, TILE_COUNT)),
    "hbt_right": (np.dtype("float32"), (TILE_COUNT, TILE_COUNT)),
    "hbt_down": (np.dtype("float32"), (TILE_COUNT, TILE_COUNT)),
    "w1_right": (np.dtype("float32"), (TILE_COUNT, TILE_COUNT)),
    "w1_down": (np.dtype("float32"), (TILE_COUNT, TILE_COUNT)),
    "w4_right": (np.dtype("float32"), (TILE_COUNT, TILE_COUNT)),
    "w4_down": (np.dtype("float32"), (TILE_COUNT, TILE_COUNT)),
    "softcycle_layout": (np.dtype("int32"), (TILE_COUNT,)),
    "qap_w4_layout": (np.dtype("int32"), (TILE_COUNT,)),
    "qap_w1_layout": (np.dtype("int32"), (TILE_COUNT,)),
    "denoised_tiles": (
        np.dtype("uint8"),
        (TILE_COUNT, TILE, TILE, 3),
    ),
}
CANDIDATE_ARRAY_SPECS: dict[str, np.dtype[Any]] = {
    "candidate_direction": np.dtype("uint8"),
    "candidate_source": np.dtype("uint16"),
    "candidate_destination": np.dtype("uint16"),
    "candidate_origin_mask": np.dtype("uint8"),
    "candidate_c1_cost": np.dtype("float32"),
    "candidate_hbt_cost": np.dtype("float32"),
    "candidate_w1_cost": np.dtype("float32"),
    "candidate_w4_cost": np.dtype("float32"),
}
GRAPH_ARRAY_SPECS = {**DERIVED_ARRAY_SPECS, **CANDIDATE_ARRAY_SPECS}
INPUT_ARRAY_SPECS = {
    "slot_tiles": (np.dtype("uint8"), (TILE_COUNT, TILE, TILE, 3)),
    "qap_seed": (np.dtype("uint64"), ()),
}
INPUT_ARRAY_SEMANTICS = {
    "slot_tiles": "opaque corrupted input slot tiles",
    "qap_seed": "fixed opaque nuisance QAP seed",
}
LABEL_ARRAY_SPECS = {
    "opaque_slot_permutation": (np.dtype("int32"), (TILE_COUNT,)),
    "composed_slot_to_target": (np.dtype("int32"), (TILE_COUNT,)),
    "clean_target_rgb": (np.dtype("uint8"), (GRID * TILE, GRID * TILE, 3)),
}
LABEL_ARRAY_SEMANTICS = {
    "opaque_slot_permutation": "secret second-stage slot permutation",
    "composed_slot_to_target": "truth mapping after opaque slot permutation",
    "clean_target_rgb": "clean RGB target",
}

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
OPAQUE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
LIFECYCLE_STATES = ("PREP", "SEALED", "PHASE_A", "LABEL_ACCESS")
PHASE_A_FORBIDDEN_PATH_COMPONENTS = {
    "fixture_label",
    "label",
    "labels",
    "target",
    "targets",
    "fixture_master_secret.bin",
    "master_secret",
}
LIFECYCLE_KEYS = {
    "schema_version",
    "kind",
    "protocol_instance_id",
    "state",
    "frozen_contract_sha256",
    "config_sha256_or_null",
    "predecessor_sha256",
}


class VerificationError(RuntimeError):
    """A fail-closed evidence or scientific recomputation failure."""


def _guard_phase_a_read_path(path: str | Path, *, label: str) -> Path:
    supplied = Path(path).expanduser().absolute()
    candidates = [supplied]
    try:
        candidates.append(supplied.resolve(strict=True))
    except OSError:
        pass
    for candidate in candidates:
        lowered = {part.lower() for part in candidate.parts}
        if lowered.intersection(PHASE_A_FORBIDDEN_PATH_COMPONENTS):
            _fail(f"{label} enters forbidden label/target namespace")
    return supplied


def _fail(message: str) -> None:
    raise VerificationError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _canonical_object_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _canonical_file_bytes(value: Any) -> bytes:
    return _canonical_object_bytes(value) + b"\n"


def _reject_json_constant(value: str) -> None:
    _fail(f"non-finite JSON constant is forbidden: {value}")


def _parse_json(payload: bytes, *, label: str, canonical_file: bool) -> Any:
    try:
        value = json.loads(
            payload.decode("utf-8"), parse_constant=_reject_json_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid JSON: {label}") from error
    _assert_json_finite(value, label=label)
    if canonical_file and payload != _canonical_file_bytes(value):
        _fail(f"JSON is not canonical: {label}")
    return value


def _assert_json_finite(value: Any, *, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(f"non-finite number in {label}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _assert_json_finite(child, label=f"{label}[{index}]")
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            _fail(f"non-string JSON key in {label}")
        for key, child in value.items():
            _assert_json_finite(child, label=f"{label}.{key}")
        return
    _fail(f"non-JSON value in {label}: {type(value).__name__}")


def _require_object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: Iterable[str], *, label: str
) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        _fail(
            f"{label} schema drift: missing={sorted(expected_set - actual_set)}, "
            f"extra={sorted(actual_set - expected_set)}"
        )


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        _fail(f"{label} must be lowercase SHA-256")
    return value


def _require_utc(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not UTC_RE.fullmatch(value):
        _fail(f"{label} must be canonical UTC seconds")
    return value


def _require_exact_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        _fail(f"{label} is below {minimum}")
    return value


def _require_finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label} must be finite")
    return result


def _assert_close(actual: Any, expected: Any, *, label: str, atol: float = 1e-12) -> None:
    actual_float = _require_finite_float(actual, label=label)
    expected_float = _require_finite_float(expected, label=f"{label}.expected")
    if not math.isclose(actual_float, expected_float, rel_tol=0.0, abs_tol=atol):
        _fail(f"{label} mismatch: {actual_float} != {expected_float}")


def _valid_relative_path(value: Any, *, parent: str, suffix: str | None = None) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        _fail("artifact path must be a nonempty POSIX relative string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or pure.parts != (parent, pure.name):
        _fail(f"artifact path is not canonical under {parent}: {value!r}")
    if pure.name in {"", ".", ".."} or not SAFE_NAME_RE.fullmatch(pure.name):
        _fail(f"unsafe artifact name: {pure.name!r}")
    if suffix is not None and not pure.name.endswith(suffix):
        _fail(f"artifact suffix drift: {value!r}")
    return value


def _open_absolute_directory(path: Path) -> int:
    absolute = path.expanduser().absolute()
    if not absolute.is_absolute():
        _fail(f"directory anchor is not absolute: {path}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            if component in {"", ".", ".."}:
                _fail(f"invalid directory anchor component: {component!r}")
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            info = os.fstat(next_descriptor)
            if not stat.S_ISDIR(info.st_mode):
                os.close(next_descriptor)
                _fail(f"anchor component is not a directory: {component}")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


@dataclass
class AnchoredRoot:
    path: Path
    descriptor: int

    @classmethod
    def open(cls, path: str | Path) -> "AnchoredRoot":
        absolute = Path(path).expanduser().absolute()
        descriptor = _open_absolute_directory(absolute)
        return cls(absolute, descriptor)

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    def __enter__(self) -> "AnchoredRoot":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _open_parent(self, parts: Sequence[str]) -> tuple[int, str]:
        if not parts:
            _fail("empty relative path")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.dup(self.descriptor)
        try:
            for component in parts[:-1]:
                if component in {"", ".", ".."} or "/" in component:
                    _fail(f"invalid relative component: {component!r}")
                child = os.open(component, flags, dir_fd=descriptor)
                info = os.fstat(child)
                if not stat.S_ISDIR(info.st_mode):
                    os.close(child)
                    _fail(f"relative component is not a directory: {component}")
                os.close(descriptor)
                descriptor = child
            return descriptor, parts[-1]
        except BaseException:
            os.close(descriptor)
            raise

    def read_file(self, relative: str, *, expected_size: int | None = None) -> tuple[bytes, os.stat_result]:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            _fail(f"unsafe relative path: {relative!r}")
        parent_fd, name = self._open_parent(pure.parts)
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        finally:
            os.close(parent_fd)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                _fail(f"file is not a one-link regular file: {relative}")
            if info.st_uid != os.getuid():
                _fail(f"file owner differs from verifier uid: {relative}")
            if expected_size is not None and info.st_size != expected_size:
                _fail(f"file size mismatch: {relative}")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            payload = b"".join(chunks)
            if len(payload) != info.st_size:
                _fail(f"short read: {relative}")
            return payload, info
        finally:
            os.close(descriptor)

    def list_names(self, relative_directory: str | None = None) -> set[str]:
        if relative_directory is None:
            descriptor = os.dup(self.descriptor)
        else:
            pure = PurePosixPath(relative_directory)
            if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
                _fail(f"unsafe directory path: {relative_directory!r}")
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.dup(self.descriptor)
            try:
                for component in pure.parts:
                    child = os.open(component, flags, dir_fd=descriptor)
                    os.close(descriptor)
                    descriptor = child
            except BaseException:
                os.close(descriptor)
                raise
        try:
            return set(os.listdir(descriptor))
        finally:
            os.close(descriptor)

    def assert_exact_tree(
        self,
        *,
        top_files: set[str],
        directories: Mapping[str, set[str]],
    ) -> None:
        expected_top = set(top_files) | set(directories)
        actual_top = self.list_names()
        if actual_top != expected_top:
            _fail(
                f"root tree drift at {self.path}: "
                f"missing={sorted(expected_top - actual_top)}, "
                f"extra={sorted(actual_top - expected_top)}"
            )
        seen_inodes: set[tuple[int, int]] = set()
        for file_name in sorted(top_files):
            _, info = self.read_file(file_name)
            inode = (info.st_dev, info.st_ino)
            if inode in seen_inodes:
                _fail("duplicate inode in exact tree")
            seen_inodes.add(inode)
        for directory, expected_names in sorted(directories.items()):
            if self.list_names(directory) != expected_names:
                _fail(f"directory allowlist mismatch: {directory}")
            for name in sorted(expected_names):
                _, info = self.read_file(f"{directory}/{name}")
                inode = (info.st_dev, info.st_ino)
                if inode in seen_inodes:
                    _fail("hardlink/inode alias across exact tree")
                seen_inodes.add(inode)


def _secure_absolute_file(path: str | Path) -> tuple[bytes, os.stat_result]:
    absolute = Path(path).expanduser().absolute()
    with AnchoredRoot.open(absolute.parent) as root:
        return root.read_file(absolute.name)


def _load_canonical_object_from_root(root: AnchoredRoot, relative: str) -> dict[str, Any]:
    payload, _ = root.read_file(relative)
    return _require_object(
        _parse_json(payload, label=relative, canonical_file=True), label=relative
    )


def _load_envelope_bytes(
    payload: bytes, *, expected_file_sha256: str | None, label: str
) -> dict[str, Any]:
    if expected_file_sha256 is not None and _sha256_bytes(payload) != _require_sha(
        expected_file_sha256, label=f"{label}.out_of_band_sha256"
    ):
        _fail(f"out-of-band envelope hash mismatch: {label}")
    envelope = _require_object(
        _parse_json(payload, label=label, canonical_file=True), label=label
    )
    _require_exact_keys(envelope, {"payload", "payload_sha256"}, label=f"{label}.envelope")
    inner = _require_object(envelope["payload"], label=f"{label}.payload")
    expected_inner = _sha256_bytes(_canonical_object_bytes(inner))
    if envelope["payload_sha256"] != expected_inner:
        _fail(f"payload hash mismatch: {label}")
    return inner


def _load_envelope_from_root(
    root: AnchoredRoot,
    relative: str,
    *,
    expected_file_sha256: str | None,
) -> tuple[dict[str, Any], str]:
    payload, _ = root.read_file(relative)
    return (
        _load_envelope_bytes(
            payload, expected_file_sha256=expected_file_sha256, label=relative
        ),
        _sha256_bytes(payload),
    )


def _load_self_manifest_from_root(
    root: AnchoredRoot,
    relative: str,
    *,
    expected_file_sha256: str,
) -> tuple[dict[str, Any], str]:
    raw, _ = root.read_file(relative)
    expected = _require_sha(
        expected_file_sha256, label=f"{relative}.out_of_band_sha256"
    )
    actual = _sha256_bytes(raw)
    if actual != expected:
        _fail(f"out-of-band self-manifest hash mismatch: {relative}")
    payload = _require_object(
        _parse_json(raw, label=relative, canonical_file=True), label=relative
    )
    _verify_self_sha256(payload, label=relative)
    return payload, actual


def _verify_self_sha256(payload: Mapping[str, Any], *, label: str) -> None:
    expected = _require_sha(payload.get("self_sha256"), label=f"{label}.self_sha256")
    base = {key: value for key, value in payload.items() if key != "self_sha256"}
    actual = _sha256_bytes(_canonical_object_bytes(base))
    if actual != expected:
        _fail(f"self_sha256 mismatch: {label}")


def _strict_npz(payload: bytes, specs: Mapping[str, tuple[np.dtype[Any], tuple[int, ...]] | np.dtype[Any]], *, label: str) -> dict[str, np.ndarray]:
    expected_members = {f"{key}.npy" for key in specs}
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            names = [info.filename for info in archive.infolist()]
            if len(names) != len(set(names)) or set(names) != expected_members:
                _fail(f"NPZ member coverage drift: {label}")
            if any(PurePosixPath(name).parts != (name,) for name in names):
                _fail(f"NPZ member path is unsafe: {label}")
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != set(specs):
                _fail(f"NPZ array coverage drift: {label}")
            result = {key: np.asarray(archive[key]) for key in sorted(archive.files)}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise VerificationError(f"invalid NPZ artifact: {label}") from error
    for key, spec in specs.items():
        value = result[key]
        if value.dtype.hasobject:
            _fail(f"object array is forbidden: {label}.{key}")
        if not value.flags.c_contiguous:
            _fail(f"array must be C-contiguous: {label}.{key}")
        if isinstance(spec, tuple):
            dtype, shape = spec
            if value.dtype != dtype or value.shape != shape:
                _fail(f"array dtype/shape drift: {label}.{key}")
        else:
            if value.dtype != spec or value.ndim != 1:
                _fail(f"candidate array dtype/rank drift: {label}.{key}")
    return result


def _validate_score_matrix(value: np.ndarray, *, label: str) -> None:
    if value.dtype != np.float32 or value.shape != (TILE_COUNT, TILE_COUNT):
        _fail(f"score matrix dtype/shape drift: {label}")
    diagonal_mask = np.eye(TILE_COUNT, dtype=bool)
    if not np.all(np.isposinf(value[diagonal_mask])):
        _fail(f"score diagonal must be +inf: {label}")
    if not np.all(np.isfinite(value[~diagonal_mask])):
        _fail(f"score off-diagonal must be finite: {label}")


def _validate_layout(value: np.ndarray, *, label: str) -> np.ndarray:
    if value.dtype != np.int32:
        _fail(f"layout dtype must be int32: {label}")
    try:
        return validate_permutation(value, name=label)
    except (TypeError, ValueError) as error:
        raise VerificationError(f"invalid layout: {label}") from error


def _merge_tiles(tiles: np.ndarray) -> np.ndarray:
    if tiles.dtype != np.uint8 or tiles.shape != (TILE_COUNT, TILE, TILE, 3):
        _fail("tile tensor dtype/shape drift")
    return (
        tiles.reshape(GRID, GRID, TILE, TILE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(GRID * TILE, GRID * TILE, 3)
    )


def _decode_png(payload: bytes, *, label: str) -> np.ndarray:
    try:
        with Image.open(BytesIO(payload)) as image:
            image.load()
            if image.format != "PNG" or image.mode != "RGB":
                _fail(f"render must be RGB PNG: {label}")
            result = np.asarray(image)
    except OSError as error:
        raise VerificationError(f"invalid PNG: {label}") from error
    if result.dtype != np.uint8 or result.shape != (GRID * TILE, GRID * TILE, 3):
        _fail(f"render dtype/shape drift: {label}")
    return result


def _opaque_qap_seed(opaque_id: str) -> int:
    if not isinstance(opaque_id, str) or not OPAQUE_ID_RE.fullmatch(opaque_id):
        _fail("invalid opaque id")
    return int.from_bytes(
        hashlib.sha256(f"qap:{opaque_id}".encode("utf-8")).digest()[:8], "big"
    )


def _rank_normalize(matrix: np.ndarray) -> np.ndarray:
    _validate_score_matrix(matrix, label="rank_normalize.input")
    order = np.argsort(matrix, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.int32)
    rows = np.arange(TILE_COUNT)[:, None]
    ranks[rows, order] = np.arange(TILE_COUNT, dtype=np.int32)[None, :]
    result = ranks.astype(np.float32) / float(TILE_COUNT - 2)
    np.fill_diagonal(result, np.inf)
    return result


def _rank_fusion(c1: np.ndarray, hbt: np.ndarray, *, hbt_weight: float) -> np.ndarray:
    if hbt_weight not in {1.0, 4.0}:
        _fail("unexpected HBT fusion weight")
    result = np.zeros((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    total = 1.0 + hbt_weight
    result += (1.0 / total) * _rank_normalize(c1)
    result += (hbt_weight / total) * _rank_normalize(hbt)
    np.fill_diagonal(result, np.inf)
    return result


def _stable_top_pairs(matrix: np.ndarray, *, outgoing: int, incoming: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    _validate_score_matrix(matrix, label="candidate_matrix")
    row_order = np.argsort(matrix, axis=1, kind="stable")
    column_order = np.argsort(matrix, axis=0, kind="stable")
    outgoing_pairs: list[tuple[int, int]] = []
    incoming_pairs: list[tuple[int, int]] = []
    for source in range(TILE_COUNT):
        selected = [int(value) for value in row_order[source] if int(value) != source]
        if len(selected) < outgoing:
            _fail("outgoing candidate cardinality underflow")
        outgoing_pairs.extend((source, destination) for destination in selected[:outgoing])
    for destination in range(TILE_COUNT):
        selected = [
            int(value)
            for value in column_order[:, destination]
            if int(value) != destination
        ]
        if len(selected) < incoming:
            _fail("incoming candidate cardinality underflow")
        incoming_pairs.extend((source, destination) for source in selected[:incoming])
    return outgoing_pairs, incoming_pairs


def _layout_edges(layout: np.ndarray, direction: int) -> list[tuple[int, int]]:
    layout = _validate_layout(layout, label="candidate_layout")
    grid = layout.reshape(GRID, GRID)
    if direction == 0:
        first, second = grid[:, :-1].ravel(), grid[:, 1:].ravel()
    elif direction == 1:
        first, second = grid[:-1, :].ravel(), grid[1:, :].ravel()
    else:
        _fail("invalid direction")
    return list(zip(first.tolist(), second.tolist(), strict=True))


def rebuild_candidate_union(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Independently reconstruct the canonical seven-origin candidate graph."""

    for key, (dtype, shape) in DERIVED_ARRAY_SPECS.items():
        if key not in arrays or arrays[key].dtype != dtype or arrays[key].shape != shape:
            _fail(f"missing or malformed derived array: {key}")
    for key in (
        "c1_right",
        "c1_down",
        "hbt_right",
        "hbt_down",
        "w1_right",
        "w1_down",
        "w4_right",
        "w4_down",
    ):
        _validate_score_matrix(arrays[key], label=key)
    for key in ("softcycle_layout", "qap_w4_layout", "qap_w1_layout"):
        _validate_layout(arrays[key], label=key)

    for suffix in ("right", "down"):
        expected_w1 = _rank_fusion(
            arrays[f"c1_{suffix}"], arrays[f"hbt_{suffix}"], hbt_weight=1.0
        )
        expected_w4 = _rank_fusion(
            arrays[f"c1_{suffix}"], arrays[f"hbt_{suffix}"], hbt_weight=4.0
        )
        if not np.array_equal(arrays[f"w1_{suffix}"], expected_w1):
            _fail(f"frozen w1 rank fusion mismatch: {suffix}")
        if not np.array_equal(arrays[f"w4_{suffix}"], expected_w4):
            _fail(f"frozen w4 rank fusion mismatch: {suffix}")

    masks: dict[tuple[int, int, int], int] = {}
    counts = {key: 0 for key in ORIGIN_BITS}

    def add(direction: int, pairs: Iterable[tuple[int, int]], origin: str) -> None:
        bit = ORIGIN_BITS[origin]
        for source, destination in pairs:
            if source == destination or not (0 <= source < TILE_COUNT and 0 <= destination < TILE_COUNT):
                _fail("invalid edge in candidate origin stream")
            key = (direction, int(source), int(destination))
            masks[key] = masks.get(key, 0) | bit
            counts[origin] += 1

    for direction, suffix in ((0, "right"), (1, "down")):
        for prefix in ("c1", "hbt"):
            outgoing, incoming = _stable_top_pairs(
                arrays[f"{prefix}_{suffix}"], outgoing=32, incoming=8
            )
            add(direction, outgoing, f"{prefix}_out32")
            add(direction, incoming, f"{prefix}_in8")
        for layout_key, origin in (
            ("softcycle_layout", "softcycle"),
            ("qap_w4_layout", "qap_w4"),
            ("qap_w1_layout", "qap_w1"),
        ):
            add(direction, _layout_edges(arrays[layout_key], direction), origin)
    if counts != EXPECTED_PREDEDUP_COUNTS:
        _fail(f"candidate pre-dedup counts drift: {counts}")

    keys = sorted(masks)
    direction = np.asarray([key[0] for key in keys], dtype=np.uint8)
    source = np.asarray([key[1] for key in keys], dtype=np.uint16)
    destination = np.asarray([key[2] for key in keys], dtype=np.uint16)
    origin_mask = np.asarray([masks[key] for key in keys], dtype=np.uint8)
    result: dict[str, np.ndarray] = {
        "candidate_direction": direction,
        "candidate_source": source,
        "candidate_destination": destination,
        "candidate_origin_mask": origin_mask,
    }
    matrices = {
        "c1": (arrays["c1_right"], arrays["c1_down"]),
        "hbt": (arrays["hbt_right"], arrays["hbt_down"]),
        "w1": (arrays["w1_right"], arrays["w1_down"]),
        "w4": (arrays["w4_right"], arrays["w4_down"]),
    }
    for label, (right, down) in matrices.items():
        result[f"candidate_{label}_cost"] = np.asarray(
            [
                (right if int(d) == 0 else down)[int(first), int(second)]
                for d, first, second in zip(
                    direction, source, destination, strict=True
                )
            ],
            dtype=np.float32,
        )
    return result


def _verify_array_descriptor(
    descriptor: Any,
    array: np.ndarray,
    *,
    label: str,
    semantic: str | None = None,
) -> None:
    value = _require_object(descriptor, label=label)
    _require_exact_keys(
        value, {"semantic", "dtype", "shape", "c_order_sha256"}, label=label
    )
    if not isinstance(value["semantic"], str) or not value["semantic"]:
        _fail(f"array semantic is invalid: {label}")
    if semantic is not None and value["semantic"] != semantic:
        _fail(f"array semantic mismatch: {label}")
    if value["dtype"] != str(array.dtype) or value["shape"] != list(array.shape):
        _fail(f"array descriptor dtype/shape mismatch: {label}")
    if value["c_order_sha256"] != _array_sha256(array):
        _fail(f"array descriptor C-order hash mismatch: {label}")


def _verify_artifact_descriptor(
    descriptor: Any,
    *,
    root: AnchoredRoot,
    expected_parent: str,
    suffix: str,
    label: str,
) -> tuple[bytes, os.stat_result, str]:
    value = _require_object(descriptor, label=label)
    _require_exact_keys(value, {"path", "bytes", "sha256"}, label=label)
    relative = _valid_relative_path(value["path"], parent=expected_parent, suffix=suffix)
    byte_size = _require_exact_int(value["bytes"], label=f"{label}.bytes", minimum=1)
    expected_sha = _require_sha(value["sha256"], label=f"{label}.sha256")
    payload, info = root.read_file(relative, expected_size=byte_size)
    if _sha256_bytes(payload) != expected_sha:
        _fail(f"artifact hash mismatch: {label}")
    return payload, info, relative


@dataclass(frozen=True)
class ProtocolContext:
    config: dict[str, Any]
    config_path: Path
    config_sha256: str
    repository: Path


def _load_protocol(
    config_path: str | Path,
    *,
    expected_config_sha256: str,
    allow_unpinned_verifier: bool = False,
) -> ProtocolContext:
    config_path = _guard_phase_a_read_path(config_path, label="protocol config")
    raw, _ = _secure_absolute_file(config_path)
    actual_sha = _sha256_bytes(raw)
    if actual_sha != _require_sha(expected_config_sha256, label="config_sha256"):
        _fail("out-of-band final config SHA-256 mismatch")
    config = _require_object(
        _parse_json(raw, label="protocol config", canonical_file=False),
        label="protocol config",
    )
    if config.get("schema_version") != 1 or config.get("kind") != "candidate_graph_oracle_ceiling":
        _fail("wrong protocol identity")
    if config.get("protocol_instance_id") != EXPECTED_PROTOCOL_INSTANCE_ID:
        _fail("protocol instance drift")
    frozen = _require_object(config.get("frozen_contract"), label="frozen_contract")
    frozen_sha = _sha256_bytes(_canonical_object_bytes(frozen))
    if (
        frozen_sha != EXPECTED_FROZEN_CONTRACT_SHA256
        or config.get("frozen_contract_sha256") != frozen_sha
    ):
        _fail("frozen contract hash drift")
    if config.get("safe_for_submission") is not False:
        _fail("oracle protocol is not fail-closed")

    path = Path(config_path).expanduser().absolute()
    repository = path.parent.parent.absolute()
    pins = _require_object(config.get("runtime_pins"), label="runtime_pins")
    policy = _require_object(
        config.get("runtime_pin_mutation_policy"), label="runtime_pin_mutation_policy"
    )
    pairs = policy.get("code_pin_fields")
    if not isinstance(pairs, list) or not pairs:
        _fail("code pin closure is missing")
    for index, pair_value in enumerate(pairs):
        pair = _require_object(pair_value, label=f"code_pin_fields[{index}]")
        _require_exact_keys(pair, {"path_field", "sha256_field"}, label=f"code_pin_fields[{index}]")
        path_field = pair["path_field"]
        sha_field = pair["sha256_field"]
        if not isinstance(path_field, str) or not isinstance(sha_field, str):
            _fail("runtime pin pair fields must be strings")
        relative = pins.get(path_field)
        expected = pins.get(sha_field)
        if not isinstance(relative, str) or not relative or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
            _fail(f"invalid runtime pin path: {path_field}")
        if sha_field == "result_verifier_sha256" and expected is None and allow_unpinned_verifier:
            continue
        expected_sha = _require_sha(expected, label=sha_field)
        with AnchoredRoot.open(repository) as repo_root:
            payload, _ = repo_root.read_file(relative)
        if _sha256_bytes(payload) != expected_sha:
            _fail(f"runtime pin hash mismatch: {sha_field}")
    self_pin = pins.get("result_verifier_sha256")
    if self_pin is not None:
        self_bytes, _ = _secure_absolute_file(Path(__file__).resolve())
        if _sha256_bytes(self_bytes) != self_pin:
            _fail("executing verifier differs from its runtime pin")
    elif not allow_unpinned_verifier:
        _fail("result verifier runtime pin is null")
    context = ProtocolContext(config, path, actual_sha, repository)
    _verify_frozen_static_bindings(context)
    if not allow_unpinned_verifier:
        _verify_imported_snapshot_modules(context)
    return context


def _verify_frozen_static_bindings(context: ProtocolContext) -> None:
    frozen = context.config["frozen_contract"]
    assets = _require_object(frozen.get("assets"), label="frozen assets")
    path_hash_pairs: list[tuple[str, str, str]] = []
    for name in ("denoiser", "hbt"):
        descriptor = _require_object(assets.get(name), label=f"asset.{name}")
        path_hash_pairs.append(
            (str(descriptor["path"]), str(descriptor["sha256"]), f"asset.{name}")
        )
    known_code = _require_object(
        assets.get("known_code_sha256"), label="known_code_sha256"
    )
    for relative, digest in known_code.items():
        path_hash_pairs.append((str(relative), str(digest), f"known_code.{relative}"))
    selection = frozen["source_selection"]
    for path_field, sha_field in (
        ("authoritative_manifest", "authoritative_manifest_sha256"),
        ("quarantine", "quarantine_sha256"),
    ):
        path_hash_pairs.append(
            (str(selection[path_field]), str(selection[sha_field]), path_field)
        )
    sealed = frozen["sealed_sets"]
    path_hash_pairs.append(
        (
            str(sealed["audit_exclusion_ledger"]),
            str(sealed["audit_exclusion_ledger_sha256"]),
            "audit_exclusion_ledger",
        )
    )
    decision = context.config["decision_basis"]
    for path_field, sha_field in (
        ("qap_confirmation_config", "qap_confirmation_config_sha256"),
        ("qap_confirmation_report", "qap_confirmation_report_sha256"),
    ):
        path_hash_pairs.append(
            (str(decision[path_field]), str(decision[sha_field]), path_field)
        )
    seen_paths: set[str] = set()
    with AnchoredRoot.open(context.repository) as root:
        for relative, digest, label in path_hash_pairs:
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                _fail(f"unsafe frozen static path: {label}")
            expected = _require_sha(digest, label=f"{label}.sha256")
            raw, _ = root.read_file(relative)
            if _sha256_bytes(raw) != expected:
                _fail(f"frozen static binding mismatch: {label}")
            if relative in seen_paths:
                # The private components-module pin intentionally repeats one
                # known-code path and is checked by value below.
                continue
            seen_paths.add(relative)
    private = _require_object(
        assets.get("private_function_contract"), label="private_function_contract"
    )
    module = str(private["module"])
    expected_module_sha = _require_sha(
        private["module_sha256"], label="private_function_contract.module_sha256"
    )
    if known_code.get(module) != expected_module_sha:
        _fail("private function module pin differs from known-code pin")


def _verify_imported_snapshot_modules(context: ProtocolContext) -> None:
    known_code = _require_object(
        context.config["frozen_contract"]["assets"]["known_code_sha256"],
        label="known_code_sha256",
    )
    expected_modules = {
        "puzzle_assembly": "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src/puzzle_assembly/__init__.py",
        "puzzle_assembly.geometry": "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src/puzzle_assembly/geometry.py",
        "puzzle_assembly.metrics": "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src/puzzle_assembly/metrics.py",
        "puzzle_assembly.panels": "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src/puzzle_assembly/panels.py",
    }
    for module_name, relative in expected_modules.items():
        loaded = sys.modules.get(module_name)
        source = getattr(loaded, "__file__", None) if loaded is not None else None
        if not isinstance(source, str):
            _fail(f"required snapshot module is not imported: {module_name}")
        actual_path = Path(source).resolve(strict=True)
        expected_path = (context.repository / relative).resolve(strict=True)
        if actual_path != expected_path:
            _fail(f"imported module escaped frozen repository: {module_name}")
        expected_sha = _require_sha(
            known_code.get(relative), label=f"known_code.{relative}"
        )
        raw, _ = _secure_absolute_file(actual_path)
        if _sha256_bytes(raw) != expected_sha:
            _fail(f"imported module hash drift: {module_name}")


def _common_manifest_fields(context: ProtocolContext) -> set[str]:
    contract = context.config["frozen_contract"]["fixture_preparation"]
    fields = contract.get("exact_common_manifest_binding_field_names")
    if not isinstance(fields, list) or len(fields) != 14 or len(set(fields)) != 14:
        _fail("common manifest binding closure drift")
    if not all(isinstance(value, str) for value in fields):
        _fail("common manifest binding names must be strings")
    return set(fields)


def _verify_common_manifest_bindings(
    manifest: Mapping[str, Any], context: ProtocolContext
) -> None:
    pins = context.config["runtime_pins"]
    expected = {
        "protocol_instance_id": context.config["protocol_instance_id"],
        "frozen_contract_sha256": context.config["frozen_contract_sha256"],
        "evaluator_sha256": pins["evaluator_sha256"],
        "tests_sha256": pins["tests_sha256"],
        "fixture_builder_sha256": pins["fixture_builder_sha256"],
        "fixture_builder_tests_sha256": pins["fixture_builder_tests_sha256"],
        "pin_finalizer_sha256": pins["pin_finalizer_sha256"],
        "lifecycle_tool_sha256": pins["lifecycle_tool_sha256"],
        "result_verifier_sha256": pins["result_verifier_sha256"],
        "environment_lock_sha256": pins["environment_lock_sha256"],
        "phase_a_runner_sha256": pins["phase_a_runner_sha256"],
        "phase_a_kernel_metadata_sha256": pins["phase_a_kernel_metadata_sha256"],
        "phase_a_launcher_sha256": pins["phase_a_launcher_sha256"],
        "phase_b_runner_sha256": pins["phase_b_runner_sha256"],
    }
    if set(expected) != _common_manifest_fields(context):
        _fail("local common binding map differs from frozen closure")
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            _fail(f"manifest provenance mismatch: {key}")


@dataclass
class InputRecord:
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]


@dataclass
class InputEvidence:
    root_path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    records: dict[str, InputRecord]


def _reject_input_metadata(value: Any, *, path: str = "input") -> None:
    forbidden = ("source", "panel", "target", "label", "secret", "shuffle", "permutation")
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in forbidden):
                _fail(f"forbidden input-only metadata: {path}.{key}")
            _reject_input_metadata(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_input_metadata(child, path=f"{path}[{index}]")


def _contains_exact_string(value: Any, needle: str) -> bool:
    if isinstance(value, str):
        return hmac.compare_digest(value, needle)
    if isinstance(value, dict):
        return any(_contains_exact_string(child, needle) for child in value.values())
    if isinstance(value, list):
        return any(_contains_exact_string(child, needle) for child in value)
    return False


def _forbid_whole_config_binding(
    value: Any, *, forbidden_hashes: set[str], path: str
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            if "config" in lowered and ("sha" in lowered or "hash" in lowered):
                _fail(f"fixture JSON binds whole config hash field: {path}.{key}")
            _forbid_whole_config_binding(
                child, forbidden_hashes=forbidden_hashes, path=f"{path}.{key}"
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _forbid_whole_config_binding(
                child, forbidden_hashes=forbidden_hashes, path=f"{path}[{index}]"
            )
    elif isinstance(value, str) and value in forbidden_hashes:
        _fail(f"fixture JSON embeds a whole-config SHA-256: {path}")


def verify_input_fixture(
    context: ProtocolContext,
    *,
    fixture_root: str | Path,
    expected_manifest_sha256: str,
) -> InputEvidence:
    expected_sha = _require_sha(expected_manifest_sha256, label="fixture_input_manifest_sha256")
    pinned = context.config["runtime_pins"].get("fixture_input_manifest_sha256")
    if pinned != expected_sha:
        _fail("fixture input manifest differs from final runtime pin")
    with AnchoredRoot.open(fixture_root) as root:
        manifest_bytes, _ = root.read_file(INPUT_MANIFEST)
        if _sha256_bytes(manifest_bytes) != expected_sha:
            _fail("fixture input manifest out-of-band hash mismatch")
        manifest = _require_object(
            _parse_json(manifest_bytes, label=INPUT_MANIFEST, canonical_file=True),
            label=INPUT_MANIFEST,
        )
        expected_keys = {
            "schema_version",
            "created_utc",
            "kind",
            *_common_manifest_fields(context),
            "record_count",
            "opaque_ids_sha256",
            "canonical_record_order",
            "allowed_record_metadata",
            "records",
        }
        _require_exact_keys(manifest, expected_keys, label=INPUT_MANIFEST)
        _verify_common_manifest_bindings(manifest, context)
        if manifest["schema_version"] != 1 or manifest["kind"] != "candidate_graph_oracle_fixture_inputs":
            _fail("fixture input manifest identity drift")
        _require_utc(manifest["created_utc"], label="input.created_utc")
        if manifest["record_count"] != 64 or manifest["canonical_record_order"] != "ascending opaque_id":
            _fail("fixture input count/order drift")
        if manifest["allowed_record_metadata"] != ["opaque_id", "artifact", "arrays"]:
            _fail("fixture input metadata allowlist drift")
        record_values = manifest.get("records")
        if not isinstance(record_values, list) or len(record_values) != 64:
            _fail("fixture input must contain exactly 64 records")
        records: dict[str, InputRecord] = {}
        artifact_names: set[str] = set()
        ordered_ids: list[str] = []
        nuisance_seeds: set[int] = set()
        for index, record_value in enumerate(record_values):
            record = _require_object(record_value, label=f"input.records[{index}]")
            _require_exact_keys(record, {"opaque_id", "artifact", "arrays"}, label=f"input.records[{index}]")
            opaque_id = record["opaque_id"]
            if not isinstance(opaque_id, str) or not OPAQUE_ID_RE.fullmatch(opaque_id):
                _fail("fixture opaque id format drift")
            if opaque_id in records:
                _fail("duplicate fixture opaque id")
            artifact_bytes, _, relative = _verify_artifact_descriptor(
                record["artifact"],
                root=root,
                expected_parent="records",
                suffix=".npz",
                label=f"input.records[{index}].artifact",
            )
            if PurePosixPath(relative).name != f"{opaque_id}.npz":
                _fail("fixture artifact name is not opaque-id-derived")
            artifact_names.add(PurePosixPath(relative).name)
            arrays = _strict_npz(
                artifact_bytes, INPUT_ARRAY_SPECS, label=f"input record {opaque_id}"
            )
            descriptors = _require_object(record["arrays"], label=f"input.records[{index}].arrays")
            _require_exact_keys(descriptors, INPUT_ARRAY_SPECS, label=f"input.records[{index}].arrays")
            for key in INPUT_ARRAY_SPECS:
                _verify_array_descriptor(
                    descriptors[key],
                    arrays[key],
                    semantic=INPUT_ARRAY_SEMANTICS[key],
                    label=f"input.records[{index}].arrays.{key}",
                )
            seed = int(arrays["qap_seed"])
            if seed != _opaque_qap_seed(opaque_id):
                _fail("input opaque nuisance seed derivation mismatch")
            if seed in nuisance_seeds:
                _fail("duplicate opaque nuisance seed")
            nuisance_seeds.add(seed)
            ordered_ids.append(opaque_id)
            records[opaque_id] = InputRecord(dict(record), arrays)
        if ordered_ids != sorted(ordered_ids):
            _fail("fixture input records are not in canonical opaque-id order")
        ids_sha = _sha256_bytes("\n".join(ordered_ids).encode("ascii"))
        if manifest["opaque_ids_sha256"] != ids_sha:
            _fail("fixture opaque-id list hash mismatch")
        _reject_input_metadata(manifest)
        root.assert_exact_tree(
            top_files={INPUT_MANIFEST}, directories={"records": artifact_names}
        )
    return InputEvidence(
        Path(fixture_root).expanduser().absolute(),
        manifest,
        expected_sha,
        records,
    )


@dataclass
class PhaseARecord:
    manifest: dict[str, Any]


@dataclass
class PhaseAEvidence:
    root_path: Path
    payload: dict[str, Any]
    envelope_sha256: str
    records: dict[str, PhaseARecord]
    shard_anchors: tuple[str, str]
    kaggle_attestation: PhaseAKaggleAttestation | None = None


@dataclass(frozen=True)
class PhaseAKaggleAttestation:
    wrapper_path: Path
    wrapper_sha256: str
    launch_receipt_path: Path
    launch_receipt_sha256: str
    wrapper: dict[str, Any]
    launch_receipt: dict[str, Any]


def _phase_a_record_keys() -> set[str]:
    return {
        "opaque_id",
        "qap_seed",
        "input_fixture_sha256",
        "input_slot_tiles_c_sha256",
        "graph_artifact",
        "graph_artifact_byte_size",
        "graph_artifact_sha256",
        "candidate_edge_count",
        "candidate_origin_mask_sha256",
        "origin_pre_dedup_counts",
        "arrays",
        "renders",
        "derivation_diagnostics",
    }


def _verify_phase_a_derivation_diagnostics(value: Any) -> None:
    diagnostics = _require_object(value, label="phase_a.derivation_diagnostics")
    _require_exact_keys(
        diagnostics,
        {"hbt_outside_logits", "softcycle", "qap"},
        label="phase_a.derivation_diagnostics",
    )
    hbt_diagnostics = _require_object(
        diagnostics["hbt_outside_logits"],
        label="phase_a.derivation_diagnostics.hbt_outside_logits",
    )
    _require_exact_keys(
        hbt_diagnostics,
        {"dtype", "shape", "c_order_sha256"},
        label="phase_a.derivation_diagnostics.hbt_outside_logits",
    )
    if (
        hbt_diagnostics["dtype"] != "float32"
        or hbt_diagnostics["shape"] != [TILE_COUNT, 4]
    ):
        _fail("Phase-A HBT outside-logit diagnostics dtype/shape drift")
    _require_sha(
        hbt_diagnostics["c_order_sha256"],
        label="phase_a.derivation_diagnostics.hbt_outside_logits.c_order_sha256",
    )
    softcycle_diagnostics = _require_object(
        diagnostics["softcycle"], label="phase_a.derivation_diagnostics.softcycle"
    )
    _require_exact_keys(
        softcycle_diagnostics,
        {"accepted_edges", "component_sizes"},
        label="phase_a.derivation_diagnostics.softcycle",
    )
    accepted_edges = _require_exact_int(
        softcycle_diagnostics["accepted_edges"],
        label="phase_a.derivation_diagnostics.softcycle.accepted_edges",
        minimum=0,
    )
    if accepted_edges > 1104:
        _fail("Phase-A softcycle accepted-edge count drift")
    component_sizes = softcycle_diagnostics["component_sizes"]
    if (
        not isinstance(component_sizes, list)
        or not component_sizes
        or any(type(item) is not int or item < 1 for item in component_sizes)
        or sum(component_sizes) != TILE_COUNT
        or component_sizes != sorted(component_sizes, reverse=True)
        or accepted_edges < TILE_COUNT - len(component_sizes)
    ):
        _fail("Phase-A softcycle component/edge diagnostics drift")
    qap_diagnostics = _require_object(
        diagnostics["qap"], label="phase_a.derivation_diagnostics.qap"
    )
    _require_exact_keys(
        qap_diagnostics,
        {"qap_w1", "qap_w4"},
        label="phase_a.derivation_diagnostics.qap",
    )
    for qap_label in ("qap_w1", "qap_w4"):
        item = _require_object(
            qap_diagnostics[qap_label],
            label=f"phase_a.derivation_diagnostics.qap.{qap_label}",
        )
        _require_exact_keys(
            item,
            {"objective", "relaxed_objective", "restart", "iterations", "converged"},
            label=f"phase_a.derivation_diagnostics.qap.{qap_label}",
        )
        if (
            type(item["objective"]) not in (int, float)
            or type(item["relaxed_objective"]) not in (int, float)
            or not math.isfinite(float(item["objective"]))
            or not math.isfinite(float(item["relaxed_objective"]))
            or type(item["restart"]) is not int
            or item["restart"] not in (0, 1)
            or type(item["iterations"]) is not int
            or item["iterations"] != 25
            or type(item["converged"]) is not bool
        ):
            _fail(f"Phase-A QAP diagnostics type/value drift: {qap_label}")
    _assert_json_finite(diagnostics, label="phase_a.derivation_diagnostics")


def _verify_phase_a_record(
    record_value: Any,
    *,
    index: int,
    root: AnchoredRoot,
    input_record: InputRecord,
) -> tuple[PhaseARecord, str, set[str]]:
    record = _require_object(record_value, label=f"phase_a.records[{index}]")
    _require_exact_keys(record, _phase_a_record_keys(), label=f"phase_a.records[{index}]")
    opaque_id = record.get("opaque_id")
    if not isinstance(opaque_id, str) or not OPAQUE_ID_RE.fullmatch(opaque_id):
        _fail("Phase-A opaque id drift")
    qap_seed = _require_exact_int(record["qap_seed"], label="phase_a.qap_seed", minimum=0)
    if qap_seed >= 2**64 or qap_seed != int(input_record.arrays["qap_seed"]):
        _fail("Phase-A QAP seed mismatch")
    input_artifact = input_record.manifest["artifact"]
    if record["input_fixture_sha256"] != input_artifact["sha256"]:
        _fail("Phase-A input fixture hash crosslink mismatch")
    if record["input_slot_tiles_c_sha256"] != _array_sha256(input_record.arrays["slot_tiles"]):
        _fail("Phase-A input tile hash crosslink mismatch")

    graph_relative = _valid_relative_path(
        record["graph_artifact"], parent="artifacts", suffix=".graph.npz"
    )
    if PurePosixPath(graph_relative).name != f"{opaque_id}.graph.npz":
        _fail("Phase-A graph artifact name drift")
    byte_size = _require_exact_int(
        record["graph_artifact_byte_size"], label="graph_artifact_byte_size", minimum=1
    )
    graph_bytes, _ = root.read_file(graph_relative, expected_size=byte_size)
    graph_sha = _require_sha(record["graph_artifact_sha256"], label="graph_artifact_sha256")
    if _sha256_bytes(graph_bytes) != graph_sha:
        _fail("Phase-A graph artifact hash mismatch")
    arrays = _strict_npz(graph_bytes, GRAPH_ARRAY_SPECS, label=f"graph {opaque_id}")
    for key in (
        "c1_right",
        "c1_down",
        "hbt_right",
        "hbt_down",
        "w1_right",
        "w1_down",
        "w4_right",
        "w4_down",
    ):
        _validate_score_matrix(arrays[key], label=f"{opaque_id}.{key}")
    for key in ("softcycle_layout", "qap_w4_layout", "qap_w1_layout"):
        _validate_layout(arrays[key], label=f"{opaque_id}.{key}")
    expected_candidate = rebuild_candidate_union(arrays)
    edge_count = len(expected_candidate["candidate_direction"])
    if record["candidate_edge_count"] != edge_count:
        _fail("Phase-A candidate edge count mismatch")
    if record["origin_pre_dedup_counts"] != EXPECTED_PREDEDUP_COUNTS:
        _fail("Phase-A origin pre-dedup count receipt mismatch")
    for key, expected in expected_candidate.items():
        if not np.array_equal(arrays[key], expected):
            _fail(f"Phase-A independently rebuilt candidate mismatch: {key}")
    if _array_sha256(arrays["candidate_origin_mask"]) != record["candidate_origin_mask_sha256"]:
        _fail("Phase-A candidate origin-mask hash mismatch")
    if np.any(arrays["candidate_source"] == arrays["candidate_destination"]):
        _fail("Phase-A candidate self edge")
    if np.any((arrays["candidate_origin_mask"] == 0) | ((arrays["candidate_origin_mask"].astype(np.int64) & ~ALL_ORIGIN_BITS) != 0)):
        _fail("Phase-A invalid candidate origin mask")
    for key in ("candidate_c1_cost", "candidate_hbt_cost", "candidate_w1_cost", "candidate_w4_cost"):
        if not np.all(np.isfinite(arrays[key])):
            _fail(f"Phase-A non-finite candidate cost: {key}")

    descriptors = _require_object(record["arrays"], label=f"phase_a.records[{index}].arrays")
    _require_exact_keys(descriptors, GRAPH_ARRAY_SPECS, label=f"phase_a.records[{index}].arrays")
    for key, array in arrays.items():
        _verify_array_descriptor(
            descriptors[key], array, semantic=key, label=f"phase_a.records[{index}].arrays.{key}"
        )

    renders_value = _require_object(record["renders"], label=f"phase_a.records[{index}].renders")
    _require_exact_keys(renders_value, {"softcycle", "qap_w4", "qap_w1"}, label=f"phase_a.records[{index}].renders")
    render_names: set[str] = set()
    for render_label, layout_key in (
        ("softcycle", "softcycle_layout"),
        ("qap_w4", "qap_w4_layout"),
        ("qap_w1", "qap_w1_layout"),
    ):
        descriptor = _require_object(
            renders_value[render_label], label=f"phase_a.render.{render_label}"
        )
        _require_exact_keys(
            descriptor, {"path", "sha256", "layout_sha256"}, label=f"phase_a.render.{render_label}"
        )
        relative = _valid_relative_path(descriptor["path"], parent="renders", suffix=".png")
        if PurePosixPath(relative).name != f"{opaque_id}__{render_label}.png":
            _fail("Phase-A render name drift")
        render_bytes, _ = root.read_file(relative)
        if _sha256_bytes(render_bytes) != _require_sha(descriptor["sha256"], label="render.sha256"):
            _fail("Phase-A render hash mismatch")
        layout = arrays[layout_key]
        if descriptor["layout_sha256"] != _array_sha256(layout):
            _fail("Phase-A render layout hash mismatch")
        decoded = _decode_png(render_bytes, label=relative)
        expected_render = _merge_tiles(arrays["denoised_tiles"][layout])
        if not np.array_equal(decoded, expected_render):
            _fail("Phase-A rendered pixels differ from frozen tiles/layout")
        render_names.add(PurePosixPath(relative).name)
    _verify_phase_a_derivation_diagnostics(record["derivation_diagnostics"])
    return PhaseARecord(dict(record)), PurePosixPath(graph_relative).name, render_names


def verify_phase_a(
    context: ProtocolContext,
    *,
    phase_a_root: str | Path,
    expected_envelope_sha256: str,
    shard_anchors: Sequence[str],
    input_evidence: InputEvidence,
) -> PhaseAEvidence:
    if len(shard_anchors) != 2:
        _fail("exactly two out-of-band shard anchors are required")
    normalized_anchors = tuple(
        _require_sha(value, label=f"phase_a_shard_anchor[{index}]")
        for index, value in enumerate(shard_anchors)
    )
    if len(set(normalized_anchors)) != 2:
        _fail("Phase-A shard anchors must be distinct")
    expected_envelope = _require_sha(
        expected_envelope_sha256, label="phase_a_envelope_sha256"
    )
    with AnchoredRoot.open(phase_a_root) as root:
        payload, actual_envelope = _load_self_manifest_from_root(
            root, PHASE_A_MANIFEST, expected_file_sha256=expected_envelope
        )
        expected_keys = {
            "schema_version",
            "kind",
            "config_sha256",
            "protocol_instance_id",
            "frozen_contract_sha256",
            "phase_a_lifecycle_sha256",
            "script_sha256",
            "runtime_asset_sha256",
            "runtime_pin_sha256",
            "fixture_manifest_sha256",
            "fixture_manifest_name",
            "shard_envelope_sha256s",
            "record_count",
            "records",
            "target_paths_constructed",
            "target_files_opened",
            "safe_for_submission",
            "self_sha256",
        }
        _require_exact_keys(payload, expected_keys, label="Phase-A finalized payload")
        _verify_self_sha256(payload, label="Phase-A finalized payload")
        expected_values = {
            "schema_version": 1,
            "kind": "frozen_candidate_graph_input_only",
            "config_sha256": context.config_sha256,
            "protocol_instance_id": EXPECTED_PROTOCOL_INSTANCE_ID,
            "frozen_contract_sha256": EXPECTED_FROZEN_CONTRACT_SHA256,
            "script_sha256": context.config["runtime_pins"]["evaluator_sha256"],
            "runtime_asset_sha256": {
                key: context.config["frozen_contract"]["assets"][key]["sha256"]
                for key in ("denoiser", "hbt")
            },
            "runtime_pin_sha256": {
                key: value
                for key, value in sorted(context.config["runtime_pins"].items())
                if key.endswith("_sha256")
            },
            "fixture_manifest_sha256": input_evidence.manifest_sha256,
            "fixture_manifest_name": INPUT_MANIFEST,
            "record_count": 64,
            "target_paths_constructed": False,
            "target_files_opened": False,
            "safe_for_submission": False,
        }
        for key, expected in expected_values.items():
            if payload.get(key) != expected:
                _fail(f"Phase-A finalized invariant drift: {key}")
        phase_a_lifecycle = _require_sha(
            payload["phase_a_lifecycle_sha256"], label="phase_a_lifecycle_sha256"
        )
        del phase_a_lifecycle
        if payload["shard_envelope_sha256s"] != list(normalized_anchors):
            _fail("Phase-A two-shard anchor list differs from out-of-band anchors")
        record_values = payload.get("records")
        if not isinstance(record_values, list) or len(record_values) != 64:
            _fail("Phase-A finalized envelope must contain exactly 64 records")
        ids = [value.get("opaque_id") if isinstance(value, dict) else None for value in record_values]
        if ids != sorted(input_evidence.records) or len(set(ids)) != 64:
            _fail("Phase-A record order/coverage differs from input fixture")
        records: dict[str, PhaseARecord] = {}
        graph_names: set[str] = set()
        render_names: set[str] = set()
        for index, record_value in enumerate(record_values):
            opaque_id = str(record_value["opaque_id"])
            verified, graph_name, names = _verify_phase_a_record(
                record_value,
                index=index,
                root=root,
                input_record=input_evidence.records[opaque_id],
            )
            records[opaque_id] = verified
            if graph_name in graph_names or render_names.intersection(names):
                _fail("duplicate Phase-A artifact path")
            graph_names.add(graph_name)
            render_names.update(names)
        root.assert_exact_tree(
            top_files={PHASE_A_MANIFEST},
            directories={"artifacts": graph_names, "renders": render_names},
        )
    return PhaseAEvidence(
        Path(phase_a_root).expanduser().absolute(),
        payload,
        actual_envelope,
        records,
        normalized_anchors,
    )


def _pinned_kernel_metadata(context: ProtocolContext) -> dict[str, Any]:
    pins = context.config["runtime_pins"]
    relative = pins["phase_a_kernel_metadata_path"]
    expected_sha = _require_sha(
        pins["phase_a_kernel_metadata_sha256"],
        label="phase_a_kernel_metadata_sha256",
    )
    with AnchoredRoot.open(context.repository) as root:
        raw, _ = root.read_file(relative)
    if _sha256_bytes(raw) != expected_sha:
        _fail("pinned Phase-A kernel metadata changed")
    metadata = _require_object(
        _parse_json(raw, label="Phase-A kernel metadata", canonical_file=False),
        label="Phase-A kernel metadata",
    )
    _require_exact_keys(
        metadata,
        {
            "id",
            "id_no",
            "reservation_receipt_sha256",
            "title",
            "code_file",
            "language",
            "kernel_type",
            "is_private",
            "enable_gpu",
            "machine_shape",
            "enable_internet",
            "dataset_sources",
            "oracle_launch_expectation",
        },
        label="Phase-A kernel metadata",
    )
    launch = _require_object(
        metadata["oracle_launch_expectation"], label="oracle_launch_expectation"
    )
    _require_exact_keys(
        launch,
        {
            "kernel_id",
            "kernel_slug",
            "kernel_version",
            "reservation_receipt_sha256",
            "dataset_versions",
        },
        label="oracle_launch_expectation",
    )
    datasets = _require_object(
        launch["dataset_versions"], label="oracle_launch_expectation.dataset_versions"
    )
    _require_exact_keys(datasets, {"code", "input", "runtime"}, label="dataset_versions")
    for label, descriptor_value in datasets.items():
        descriptor = _require_object(
            descriptor_value, label=f"dataset_versions.{label}"
        )
        _require_exact_keys(
            descriptor, {"slug", "version"}, label=f"dataset_versions.{label}"
        )
        if (
            not isinstance(descriptor["slug"], str)
            or not descriptor["slug"].startswith("pasha883/")
            or descriptor["version"] != 2
        ):
            _fail(f"Phase-A dataset launch expectation drift: {label}")
    expected_sources = [
        f"{datasets[label]['slug']}/{datasets[label]['version']}"
        for label in ("code", "input", "runtime")
    ]
    reservation_receipt_sha256 = launch["reservation_receipt_sha256"]
    if EXPECTED_RESERVATION_RECEIPT_SHA256 is None:
        if reservation_receipt_sha256 is not None:
            _fail("pre-reservation receipt hash placeholder drift")
    else:
        _require_sha(
            EXPECTED_RESERVATION_RECEIPT_SHA256,
            label="expected reservation receipt SHA-256",
        )
        if reservation_receipt_sha256 != EXPECTED_RESERVATION_RECEIPT_SHA256:
            _fail("reservation receipt SHA-256 binding drift")
    if (
        metadata["id"] != launch["kernel_slug"]
        or metadata["id_no"] != launch["kernel_id"]
        or metadata["reservation_receipt_sha256"]
        != reservation_receipt_sha256
        or launch["kernel_version"] != 2
        or metadata["dataset_sources"] != expected_sources
        or metadata["code_file"] != "run_phase_a.py"
        or metadata["language"] != "python"
        or metadata["kernel_type"] != "script"
        or metadata["is_private"] is not True
        or metadata["enable_gpu"] is not True
        or metadata["enable_internet"] is not False
        or metadata["machine_shape"] != "NvidiaTeslaT4"
    ):
        _fail("pinned Phase-A kernel launch metadata drift")
    return metadata


def _expected_phase_a_code_mount(context: ProtocolContext) -> dict[str, str]:
    expected = {
        context.config_path.relative_to(context.repository).as_posix(): context.config_sha256
    }
    pins = context.config["runtime_pins"]
    policy = context.config["runtime_pin_mutation_policy"]
    for pair_value in policy["code_pin_fields"]:
        pair = _require_object(pair_value, label="code_pin_fields entry")
        relative = pins[pair["path_field"]]
        digest = _require_sha(pins[pair["sha256_field"]], label=pair["sha256_field"])
        if relative in expected and expected[relative] != digest:
            _fail("Phase-A code-mount path has conflicting hashes")
        expected[relative] = digest
    known_code = _require_object(
        context.config["frozen_contract"]["assets"]["known_code_sha256"],
        label="known_code_sha256",
    )
    for relative, digest_value in known_code.items():
        digest = _require_sha(digest_value, label=f"known_code.{relative}")
        if relative in expected and expected[relative] != digest:
            _fail("Phase-A known-code path conflicts with runtime pin")
        expected[relative] = digest
    return expected


def _verify_phase_a_hardware(
    context: ProtocolContext, hardware_value: Any
) -> None:
    hardware = _require_object(hardware_value, label="Phase-A wrapper.hardware")
    expected_keys = {
        "python",
        "torch",
        "cuda_runtime",
        "numpy",
        "scipy",
        "scikit_image",
        "pillow",
        "opencv",
        "kornia",
        "devices",
    }
    _require_exact_keys(hardware, expected_keys, label="Phase-A wrapper.hardware")
    pins = context.config["runtime_pins"]
    with AnchoredRoot.open(context.repository) as root:
        lock_raw, _ = root.read_file(pins["environment_lock_path"])
    if _sha256_bytes(lock_raw) != pins["environment_lock_sha256"]:
        _fail("environment lock changed before wrapper verification")
    lock = _require_object(
        _parse_json(lock_raw, label="environment lock", canonical_file=False),
        label="environment lock",
    )
    expected = _require_object(lock["kaggle_phase_a"], label="kaggle_phase_a")
    packages = _require_object(expected["packages"], label="kaggle_phase_a.packages")
    expected_scalars = {
        "python": expected["python"],
        "torch": packages["torch"],
        "cuda_runtime": expected["cuda_runtime"],
        "numpy": packages["numpy"],
        "scipy": packages["scipy"],
        "scikit_image": packages["scikit_image"],
        "pillow": packages["pillow"],
        "opencv": packages["opencv"],
        "kornia": packages["kornia"],
    }
    for key, expected_value in expected_scalars.items():
        if hardware[key] != expected_value:
            _fail(f"Phase-A wrapper hardware/environment drift: {key}")
    devices = hardware["devices"]
    expected_devices = expected["devices"]
    if not isinstance(devices, list) or len(devices) != 2 or len(expected_devices) != 2:
        _fail("Phase-A wrapper must attest exactly two devices")
    for index, (device_value, expected_device) in enumerate(
        zip(devices, expected_devices, strict=True)
    ):
        device = _require_object(device_value, label=f"hardware.devices[{index}]")
        _require_exact_keys(
            device,
            {"index", "name", "capability", "tensor_probe"},
            label=f"hardware.devices[{index}]",
        )
        if any(device[key] != expected_device[key] for key in ("index", "name", "capability")):
            _fail(f"Phase-A GPU identity drift at device {index}")
        _require_finite_float(
            device["tensor_probe"], label=f"hardware.devices[{index}].tensor_probe"
        )


def _verify_unversioned_kernel_readback(
    value: Any,
    *,
    pinned_metadata: Mapping[str, Any],
    expected_version: int,
    expected_source_sha256: str,
    expected_dataset_sources: list[str],
    label: str,
) -> None:
    readback = _require_object(value, label=label)
    _require_exact_keys(
        readback,
        {
            "access_mode",
            "version_qualified_pull_used",
            "metadata",
            "metadata_sha256",
            "source_sha256",
        },
        label=label,
    )
    if (
        readback["access_mode"] != "unversioned_private_get_kernel"
        or readback["version_qualified_pull_used"] is not False
        or readback["source_sha256"] != expected_source_sha256
    ):
        _fail(f"{label} access/source binding drift")
    _require_sha(readback["metadata_sha256"], label=f"{label}.metadata_sha256")
    metadata = _require_object(readback["metadata"], label=f"{label}.metadata")
    metadata_keys = {
        "id",
        "ref",
        "title",
        "slug",
        "language",
        "kernel_type",
        "is_private",
        "enable_gpu_observation",
        "enable_internet",
        "enable_tpu_observation",
        "dataset_sources",
        "kernel_sources",
        "competition_sources",
        "model_sources",
        "current_version_number",
        "docker_image",
        "machine_shape_observation",
    }
    _require_exact_keys(metadata, metadata_keys, label=f"{label}.metadata")
    if readback["metadata_sha256"] != _sha256_bytes(
        _canonical_object_bytes(metadata)
    ):
        _fail(f"{label} metadata hash mismatch")
    expected = {
        "id": pinned_metadata["id_no"],
        "ref": pinned_metadata["id"],
        "title": pinned_metadata["title"],
        "slug": str(pinned_metadata["id"]).split("/", 1)[1],
        "language": pinned_metadata["language"],
        "kernel_type": pinned_metadata["kernel_type"],
        "is_private": True,
        "enable_internet": False,
        "dataset_sources": expected_dataset_sources,
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "current_version_number": expected_version,
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            _fail(f"{label} normalized metadata drift: {key}")
    if metadata["enable_gpu_observation"] not in (True, False, None):
        _fail(f"{label} enable_gpu observation type drift")
    if metadata["enable_tpu_observation"] not in (True, False, None):
        _fail(f"{label} enable_tpu observation type drift")
    if metadata["docker_image"] is not None and not isinstance(
        metadata["docker_image"], str
    ):
        _fail(f"{label} docker image observation type drift")
    if metadata["machine_shape_observation"] is not None and not isinstance(
        metadata["machine_shape_observation"], str
    ):
        _fail(f"{label} machine-shape observation type drift")


def verify_phase_a_kaggle_attestation(
    context: ProtocolContext,
    *,
    phase_a: PhaseAEvidence,
    input_evidence: InputEvidence,
    wrapper_path: str | Path,
    expected_wrapper_sha256: str,
    launch_receipt_path: str | Path,
    expected_launch_receipt_sha256: str,
) -> PhaseAKaggleAttestation:
    """Bind downloaded Phase A to the exact Kaggle launch and mounted datasets."""

    wrapper_expected = _require_sha(
        expected_wrapper_sha256, label="phase_a_wrapper_sha256"
    )
    receipt_expected = _require_sha(
        expected_launch_receipt_sha256, label="kaggle_launch_receipt_sha256"
    )
    wrapper_absolute = _guard_phase_a_read_path(
        wrapper_path, label="Phase-A wrapper"
    )
    receipt_absolute = _guard_phase_a_read_path(
        launch_receipt_path, label="Kaggle launch receipt"
    )
    forbidden_components = {
        "fixture_label",
        "labels",
        "target",
        "targets",
        "fixture_master_secret.bin",
    }
    for label, path in (
        ("Phase-A wrapper", wrapper_absolute),
        ("Kaggle launch receipt", receipt_absolute),
    ):
        if {part.lower() for part in path.parts}.intersection(forbidden_components):
            _fail(f"{label} path enters label/target namespace")
        if path == input_evidence.root_path or input_evidence.root_path in path.parents:
            _fail(f"{label} may not be stored inside the input fixture root")
    wrapper_raw, wrapper_info = _secure_absolute_file(wrapper_absolute)
    receipt_raw, receipt_info = _secure_absolute_file(receipt_absolute)
    if (wrapper_info.st_dev, wrapper_info.st_ino) == (
        receipt_info.st_dev,
        receipt_info.st_ino,
    ):
        _fail("Phase-A wrapper and launch receipt alias one file")
    if _sha256_bytes(wrapper_raw) != wrapper_expected:
        _fail("Phase-A Kaggle wrapper out-of-band hash mismatch")
    wrapper = _require_object(
        _parse_json(wrapper_raw, label="Phase-A Kaggle wrapper", canonical_file=True),
        label="Phase-A Kaggle wrapper",
    )
    wrapper_keys = {
        "schema_version",
        "kind",
        "status",
        "safe_for_submission",
        "kernel_slug",
        "config_sha256",
        "runner_sha256",
        "kernel_metadata_sha256",
        "launch_expectation",
        "evaluator_sha256",
        "tests_sha256",
        "fixture_builder_tests_sha256",
        "environment_lock_sha256",
        "input_manifest_sha256",
        "runtime_assets",
        "dataset_mounts",
        "exact_code_mount_sha256",
        "hardware",
        "shards",
        "finalized_phase_a_manifest",
        "finalized_phase_a_manifest_sha256",
        "seconds",
    }
    _require_exact_keys(wrapper, wrapper_keys, label="Phase-A Kaggle wrapper")
    metadata = _pinned_kernel_metadata(context)
    launch = metadata["oracle_launch_expectation"]
    pins = context.config["runtime_pins"]
    expected_wrapper_values = {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_phase_a_kaggle_wrapper",
        "status": "phase_a_complete_pending_local_verification",
        "safe_for_submission": False,
        "kernel_slug": launch["kernel_slug"],
        "config_sha256": context.config_sha256,
        "runner_sha256": pins["phase_a_runner_sha256"],
        "kernel_metadata_sha256": pins["phase_a_kernel_metadata_sha256"],
        "launch_expectation": launch,
        "evaluator_sha256": pins["evaluator_sha256"],
        "tests_sha256": pins["tests_sha256"],
        "fixture_builder_tests_sha256": pins["fixture_builder_tests_sha256"],
        "environment_lock_sha256": pins["environment_lock_sha256"],
        "input_manifest_sha256": input_evidence.manifest_sha256,
        "runtime_assets": {
            "denoiser_sha256": context.config["frozen_contract"]["assets"]["denoiser"]["sha256"],
            "hbt_sha256": context.config["frozen_contract"]["assets"]["hbt"]["sha256"],
        },
        "exact_code_mount_sha256": _expected_phase_a_code_mount(context),
        "finalized_phase_a_manifest": f"finalized/{PHASE_A_MANIFEST}",
        "finalized_phase_a_manifest_sha256": phase_a.envelope_sha256,
    }
    for key, expected_value in expected_wrapper_values.items():
        if wrapper.get(key) != expected_value:
            _fail(f"Phase-A Kaggle wrapper crosslink mismatch: {key}")
    shards = wrapper["shards"]
    expected_shards = [
        {"rank": index, "manifest_sha256": digest}
        for index, digest in enumerate(phase_a.shard_anchors)
    ]
    if shards != expected_shards:
        _fail("Phase-A wrapper shard anchors drift")
    mounts = _require_object(wrapper["dataset_mounts"], label="dataset_mounts")
    _require_exact_keys(mounts, {"code", "input", "runtime"}, label="dataset_mounts")
    for label, expected_dataset in launch["dataset_versions"].items():
        mount = _require_object(mounts[label], label=f"dataset_mounts.{label}")
        _require_exact_keys(mount, {"slug", "version", "path"}, label=f"dataset_mounts.{label}")
        if mount["slug"] != expected_dataset["slug"] or mount["version"] != expected_dataset["version"]:
            _fail(f"Phase-A mounted dataset version drift: {label}")
        path = PurePosixPath(str(mount["path"]))
        dataset_slug = str(expected_dataset["slug"]).split("/")[-1]
        if (
            not path.is_absolute()
            or path.parts[:3] != ("/", "kaggle", "input")
            or dataset_slug not in path.parts
        ):
            _fail(f"Phase-A mounted dataset path drift: {label}")
    _verify_phase_a_hardware(context, wrapper["hardware"])
    if _require_finite_float(wrapper["seconds"], label="wrapper.seconds") < 0.0:
        _fail("Phase-A wrapper duration is negative")

    receipt = _load_envelope_bytes(
        receipt_raw,
        expected_file_sha256=receipt_expected,
        label="Kaggle launch receipt",
    )
    receipt_keys = {
        "schema_version",
        "kind",
        "created_utc",
        "protocol_instance_id",
        "kernel",
        "dataset_versions_before_push",
        "dataset_versions_after_push",
        "local_kernel_metadata_sha256",
        "local_runner_sha256",
        "local_launcher_sha256",
        "launch_journal",
        "launch_intent",
        "raw_push_response",
        "push_response",
        "server_readback",
        "gpu_and_machine_metadata_authority",
        "push_performed_in_this_process",
        "push_response_recovered_from_raw_journal",
        "safe_for_submission",
    }
    _require_exact_keys(receipt, receipt_keys, label="Kaggle launch receipt")
    if (
        receipt["schema_version"] != 2
        or receipt["kind"] != "candidate_graph_oracle_kaggle_launch_receipt"
        or receipt["protocol_instance_id"] != EXPECTED_PROTOCOL_INSTANCE_ID
        or receipt["safe_for_submission"] is not False
        or receipt["local_kernel_metadata_sha256"] != pins["phase_a_kernel_metadata_sha256"]
        or receipt["local_runner_sha256"] != pins["phase_a_runner_sha256"]
        or receipt["local_launcher_sha256"] != pins["phase_a_launcher_sha256"]
        or receipt["gpu_and_machine_metadata_authority"]
        != "executed_phase_a_wrapper_hardware_not_normalized_get_kernel_metadata"
        or not isinstance(receipt["push_performed_in_this_process"], bool)
        or not isinstance(
            receipt["push_response_recovered_from_raw_journal"], bool
        )
        or (
            receipt["push_performed_in_this_process"]
            and receipt["push_response_recovered_from_raw_journal"]
        )
    ):
        _fail("Kaggle launch receipt header/provenance drift")
    _require_utc(receipt["created_utc"], label="launch_receipt.created_utc")
    kernel = _require_object(receipt["kernel"], label="launch_receipt.kernel")
    _require_exact_keys(kernel, {"slug", "kernel_id", "version", "url"}, label="launch_receipt.kernel")
    if (
        kernel["slug"] != launch["kernel_slug"]
        or kernel["kernel_id"] != launch["kernel_id"]
        or kernel["version"] != launch["kernel_version"]
        or not isinstance(kernel["url"], str)
        or not kernel["url"].startswith("https://www.kaggle.com/")
    ):
        _fail("Kaggle launch kernel identity/version drift")
    expected_datasets = {
        label: {**descriptor, "status": "ready"}
        for label, descriptor in launch["dataset_versions"].items()
    }
    for phase in ("before_push", "after_push"):
        datasets = _require_object(
            receipt[f"dataset_versions_{phase}"],
            label=f"launch_receipt.dataset_versions_{phase}",
        )
        _require_exact_keys(
            datasets,
            {"code", "input", "runtime"},
            label=f"launch_receipt.dataset_versions_{phase}",
        )
        if datasets != expected_datasets:
            _fail(f"Kaggle launch independent dataset readback drift: {phase}")

    journal = _require_object(receipt["launch_journal"], label="launch_journal")
    _require_exact_keys(
        journal,
        {
            "intent_file",
            "intent_sha256",
            "raw_push_response_file",
            "raw_push_response_sha256",
            "push_response_file",
            "push_response_sha256",
        },
        label="launch_journal",
    )
    if (
        journal["intent_file"] != "00_launch.intent.json"
        or journal["raw_push_response_file"] != "01_push.raw_response.json"
        or journal["push_response_file"] != "02_push.response.json"
    ):
        _fail("Kaggle launch journal filenames drift")
    _require_sha(journal["intent_sha256"], label="launch_journal.intent_sha256")
    _require_sha(
        journal["raw_push_response_sha256"],
        label="launch_journal.raw_push_response_sha256",
    )
    _require_sha(
        journal["push_response_sha256"],
        label="launch_journal.push_response_sha256",
    )

    intent = _require_object(receipt["launch_intent"], label="launch_intent")
    _require_exact_keys(
        intent,
        {
            "schema_version",
            "kind",
            "created_utc",
            "protocol_instance_id",
            "kernel",
            "dataset_versions",
            "local_kernel_metadata_sha256",
            "local_runner_sha256",
            "local_launcher_sha256",
            "reservation_receipt_sha256",
            "reservation_readback",
            "safe_for_submission",
        },
        label="launch_intent",
    )
    if (
        intent["schema_version"] != 2
        or intent["kind"] != "candidate_graph_oracle_kaggle_launch_intent"
        or intent["protocol_instance_id"] != EXPECTED_PROTOCOL_INSTANCE_ID
        or intent["kernel"]
        != {
            "slug": launch["kernel_slug"],
            "kernel_id": launch["kernel_id"],
            "reserved_version": 1,
            "intended_version": launch["kernel_version"],
        }
        or intent["dataset_versions"] != expected_datasets
        or intent["local_kernel_metadata_sha256"]
        != pins["phase_a_kernel_metadata_sha256"]
        or intent["local_runner_sha256"] != pins["phase_a_runner_sha256"]
        or intent["local_launcher_sha256"] != pins["phase_a_launcher_sha256"]
        or intent["reservation_receipt_sha256"]
        != launch["reservation_receipt_sha256"]
        or intent["safe_for_submission"] is not False
    ):
        _fail("Kaggle launch intent binding drift")
    _require_utc(intent["created_utc"], label="launch_intent.created_utc")
    if journal["intent_sha256"] != _sha256_bytes(_canonical_file_bytes(intent)):
        _fail("Kaggle launch intent journal hash mismatch")
    _verify_unversioned_kernel_readback(
        intent["reservation_readback"],
        pinned_metadata=metadata,
        expected_version=1,
        expected_source_sha256=EXPECTED_RESERVATION_RUNNER_SHA256,
        expected_dataset_sources=[],
        label="reservation_readback",
    )

    raw_response = _require_object(
        receipt["raw_push_response"], label="raw_push_response"
    )
    _require_exact_keys(
        raw_response,
        {
            "schema_version",
            "kind",
            "recorded_utc",
            "response_type",
            "public_fields",
            "object_state",
        },
        label="raw_push_response",
    )
    raw_type = _require_object(
        raw_response["response_type"], label="raw_push_response.response_type"
    )
    _require_exact_keys(
        raw_type, {"module", "qualname"}, label="raw_push_response.response_type"
    )
    raw_fields = _require_object(
        raw_response["public_fields"], label="raw_push_response.public_fields"
    )
    _require_exact_keys(
        raw_fields,
        RAW_PUSH_RESPONSE_FIELDS,
        label="raw_push_response.public_fields",
    )
    if (
        raw_response["schema_version"] != 1
        or raw_response["kind"]
        != "candidate_graph_oracle_kaggle_raw_push_response"
        or not all(isinstance(raw_type[key], str) for key in ("module", "qualname"))
    ):
        _fail("raw Kaggle push-response header drift")
    _require_utc(raw_response["recorded_utc"], label="raw_push_response.recorded_utc")
    if journal["raw_push_response_sha256"] != _sha256_bytes(
        _canonical_file_bytes(raw_response)
    ):
        _fail("raw Kaggle push-response journal hash mismatch")

    response = _require_object(receipt["push_response"], label="push_response")
    _require_exact_keys(
        response,
        {
            "schema_version",
            "kind",
            "ref",
            "kernel_id",
            "version_number",
            "url",
            "error",
            "invalid_dataset_sources",
            "invalid_competition_sources",
            "invalid_kernel_sources",
            "invalid_model_sources",
            "raw_response_file",
            "raw_response_sha256",
            "recorded_utc",
        },
        label="push_response",
    )
    if (
        response["schema_version"] != 2
        or response["kind"] != "candidate_graph_oracle_kaggle_push_response"
        or response["ref"] != launch["kernel_slug"]
        or response["kernel_id"] != launch["kernel_id"]
        or response["version_number"] != launch["kernel_version"]
        or response["url"] != kernel["url"]
        or response["error"] not in (None, "")
        or response["raw_response_file"] != "01_push.raw_response.json"
        or response["raw_response_sha256"]
        != journal["raw_push_response_sha256"]
        or any(
            response[key] != []
            for key in (
                "invalid_dataset_sources",
                "invalid_competition_sources",
                "invalid_kernel_sources",
                "invalid_model_sources",
            )
        )
    ):
        _fail("exact Kaggle push response binding drift")
    raw_ref = raw_fields["ref"]
    if raw_ref not in {
        response["ref"],
        f"/code/{response['ref']}",
    }:
        _fail("raw/validated Kaggle push response drift: ref")
    for key in (
        "kernel_id",
        "version_number",
        "url",
        "error",
        "invalid_dataset_sources",
        "invalid_competition_sources",
        "invalid_kernel_sources",
        "invalid_model_sources",
    ):
        if raw_fields[key] != response[key]:
            _fail(f"raw/validated Kaggle push response drift: {key}")
    _require_utc(response["recorded_utc"], label="push_response.recorded_utc")
    if journal["push_response_sha256"] != _sha256_bytes(
        _canonical_file_bytes(response)
    ):
        _fail("Kaggle push-response journal hash mismatch")

    normalized_sources = [
        launch["dataset_versions"][label]["slug"]
        for label in ("code", "input", "runtime")
    ]
    _verify_unversioned_kernel_readback(
        receipt["server_readback"],
        pinned_metadata=metadata,
        expected_version=launch["kernel_version"],
        expected_source_sha256=pins["phase_a_runner_sha256"],
        expected_dataset_sources=normalized_sources,
        label="server_readback",
    )
    wrapper_after, _ = _secure_absolute_file(wrapper_absolute)
    receipt_after, _ = _secure_absolute_file(receipt_absolute)
    if _sha256_bytes(wrapper_after) != wrapper_expected:
        _fail("Phase-A Kaggle wrapper changed during verification")
    if _sha256_bytes(receipt_after) != receipt_expected:
        _fail("Kaggle launch receipt changed during verification")
    return PhaseAKaggleAttestation(
        wrapper_absolute,
        wrapper_expected,
        receipt_absolute,
        receipt_expected,
        wrapper,
        receipt,
    )


def _verify_local_environment(context: ProtocolContext) -> None:
    import platform

    import cv2
    import kornia
    import scipy
    import skimage
    import torch
    from PIL import __version__ as pillow_version

    pins = context.config["runtime_pins"]
    relative = pins["environment_lock_path"]
    with AnchoredRoot.open(context.repository) as repo_root:
        raw, _ = repo_root.read_file(relative)
    if _sha256_bytes(raw) != pins["environment_lock_sha256"]:
        _fail("environment lock changed")
    lock = _require_object(
        _parse_json(raw, label="environment lock", canonical_file=False),
        label="environment lock",
    )
    if lock.get("schema_version") != 1 or lock.get("kind") != "candidate_graph_oracle_environment_lock":
        _fail("environment lock identity drift")
    expected = _require_object(
        lock.get("fixture_preparation_and_phase_b"), label="local environment lock"
    )
    actual = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "pillow": pillow_version,
            "kornia": kornia.__version__,
            "scikit_image": skimage.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
    }
    for key in ("platform", "python", "packages"):
        if expected.get(key) != actual[key]:
            _fail(f"independent verifier environment mismatch: {key}")


@dataclass(frozen=True)
class LifecycleEvidence:
    root_path: Path
    payloads: dict[str, dict[str, Any]]
    hashes: dict[str, str]
    transition_hashes: dict[str, str]
    code_config_sha256: str


def _verify_transition_receipts(
    context: ProtocolContext,
    *,
    ledger_root: AnchoredRoot,
) -> tuple[str, dict[str, str]]:
    transition_directory = "runtime_pin_transitions"
    expected_names = {
        "00_code_pins.intent.json",
        "00_code_pins.complete.json",
        "01_fixtures_pins.intent.json",
        "01_fixtures_pins.complete.json",
    }
    if ledger_root.list_names(transition_directory) != expected_names:
        _fail("runtime pin transition receipt coverage drift")
    intent_keys = {
        "schema_version",
        "kind",
        "stage",
        "stage_index",
        "protocol_instance_id",
        "frozen_contract_sha256",
        "config_relative_path",
        "previous_config_sha256",
        "intended_config_sha256",
        "pin_sha256_values",
        "created_utc",
    }
    completion_keys = {
        "schema_version",
        "kind",
        "stage",
        "stage_index",
        "protocol_instance_id",
        "frozen_contract_sha256",
        "config_relative_path",
        "previous_config_sha256",
        "final_config_sha256",
        "pin_sha256_values",
        "intent_sha256",
        "completed_utc",
    }
    pins = context.config["runtime_pins"]
    policy = context.config["runtime_pin_mutation_policy"]
    config_relative = context.config_path.relative_to(context.repository).as_posix()
    previous_final: str | None = None
    code_final = ""
    transition_hashes: dict[str, str] = {}
    for stage, stage_index, prefix, pair_key in (
        ("code", 0, "00_code_pins", "code_pin_fields"),
        ("fixtures", 1, "01_fixtures_pins", "fixture_pin_fields"),
    ):
        intent_name = f"{transition_directory}/{prefix}.intent.json"
        complete_name = f"{transition_directory}/{prefix}.complete.json"
        intent_raw, _ = ledger_root.read_file(intent_name)
        complete_raw, _ = ledger_root.read_file(complete_name)
        transition_hashes[intent_name] = _sha256_bytes(intent_raw)
        transition_hashes[complete_name] = _sha256_bytes(complete_raw)
        intent = _require_object(
            _parse_json(intent_raw, label=intent_name, canonical_file=True),
            label=intent_name,
        )
        complete = _require_object(
            _parse_json(complete_raw, label=complete_name, canonical_file=True),
            label=complete_name,
        )
        _require_exact_keys(intent, intent_keys, label=intent_name)
        _require_exact_keys(complete, completion_keys, label=complete_name)
        common = {
            "schema_version": 1,
            "stage": stage,
            "stage_index": stage_index,
            "protocol_instance_id": EXPECTED_PROTOCOL_INSTANCE_ID,
            "frozen_contract_sha256": EXPECTED_FROZEN_CONTRACT_SHA256,
            "config_relative_path": config_relative,
        }
        for key, expected in common.items():
            if intent.get(key) != expected or complete.get(key) != expected:
                _fail(f"runtime transition common field mismatch: {stage}.{key}")
        if intent["kind"] != "candidate_graph_oracle_runtime_pin_transition_intent":
            _fail("runtime transition intent kind drift")
        if complete["kind"] != "candidate_graph_oracle_runtime_pin_transition_completion":
            _fail("runtime transition completion kind drift")
        _require_utc(intent["created_utc"], label=f"{stage}.intent.created_utc")
        _require_utc(
            complete["completed_utc"], label=f"{stage}.completion.completed_utc"
        )
        pairs = policy[pair_key]
        expected_pin_map: dict[str, str] = {}
        for pair in pairs:
            sha_field = pair["sha256_field"]
            expected_pin_map[sha_field] = _require_sha(pins[sha_field], label=sha_field)
        if intent["pin_sha256_values"] != expected_pin_map or complete["pin_sha256_values"] != expected_pin_map:
            _fail(f"runtime transition pin closure mismatch: {stage}")
        previous = _require_sha(intent["previous_config_sha256"], label=f"{stage}.previous")
        intended = _require_sha(intent["intended_config_sha256"], label=f"{stage}.intended")
        if complete["previous_config_sha256"] != previous or complete["final_config_sha256"] != intended:
            _fail(f"runtime transition completion crosslink mismatch: {stage}")
        if complete["intent_sha256"] != _sha256_bytes(intent_raw):
            _fail(f"runtime transition intent hash mismatch: {stage}")
        if previous_final is not None and previous != previous_final:
            _fail("runtime transition stages do not form one config-hash chain")
        if stage == "code":
            code_final = intended
        else:
            if intended != context.config_sha256:
                _fail("fixture pin transition does not end at final config")
        previous_final = intended
    return code_final, transition_hashes


def _verify_lifecycle_prefix(
    context: ProtocolContext,
    *,
    lifecycle_ledger: str | Path,
    phase_a: PhaseAEvidence,
    states: Sequence[str],
) -> LifecycleEvidence:
    configured = (
        context.repository
        / context.config["runtime_pin_mutation_policy"]["transition_ledger_root"]
    ).absolute()
    supplied = _guard_phase_a_read_path(
        lifecycle_ledger, label="lifecycle ledger"
    )
    if supplied != configured:
        _fail("lifecycle ledger path differs from frozen config")
    with AnchoredRoot.open(supplied) as root:
        actual_top = root.list_names()
        expected_top = {f"{state}.json" for state in states} | {
            "runtime_pin_transitions"
        }
        if actual_top != expected_top:
            _fail("lifecycle ledger exact tree drift")
        code_config_sha, transition_hashes = _verify_transition_receipts(
            context, ledger_root=root
        )
        payloads: dict[str, dict[str, Any]] = {}
        hashes: dict[str, str] = {}
        predecessor: str | None = None
        for state in states:
            name = f"{state}.json"
            raw, _ = root.read_file(name)
            payload = _require_object(
                _parse_json(raw, label=name, canonical_file=True), label=name
            )
            _require_exact_keys(payload, LIFECYCLE_KEYS, label=name)
            expected = {
                "schema_version": 1,
                "kind": "candidate_graph_oracle_lifecycle",
                "protocol_instance_id": EXPECTED_PROTOCOL_INSTANCE_ID,
                "state": state,
                "frozen_contract_sha256": EXPECTED_FROZEN_CONTRACT_SHA256,
                "predecessor_sha256": predecessor,
            }
            for key, expected_value in expected.items():
                if payload.get(key) != expected_value:
                    _fail(f"lifecycle chain mismatch: {state}.{key}")
            expected_config = code_config_sha if state == "PREP" else context.config_sha256
            if payload["config_sha256_or_null"] != expected_config:
                _fail(f"lifecycle whole-config binding mismatch: {state}")
            digest = _sha256_bytes(raw)
            payloads[state] = payload
            hashes[state] = digest
            predecessor = digest
        if phase_a.payload["phase_a_lifecycle_sha256"] != hashes["PHASE_A"]:
            _fail("Phase-A envelope lifecycle binding mismatch")
    return LifecycleEvidence(
        supplied, payloads, hashes, transition_hashes, code_config_sha
    )


def verify_phase_a_lifecycle(
    context: ProtocolContext,
    *,
    lifecycle_ledger: str | Path,
    phase_a: PhaseAEvidence,
) -> LifecycleEvidence:
    """Verify the exact Phase-A-only ledger without permitting LABEL_ACCESS."""

    return _verify_lifecycle_prefix(
        context,
        lifecycle_ledger=lifecycle_ledger,
        phase_a=phase_a,
        states=LIFECYCLE_STATES[:3],
    )


def verify_lifecycle(
    context: ProtocolContext,
    *,
    lifecycle_ledger: str | Path,
    phase_a: PhaseAEvidence,
) -> LifecycleEvidence:
    return _verify_lifecycle_prefix(
        context,
        lifecycle_ledger=lifecycle_ledger,
        phase_a=phase_a,
        states=LIFECYCLE_STATES,
    )


def _independent_source_names(context: ProtocolContext) -> list[str]:
    frozen = context.config["frozen_contract"]
    selection = frozen["source_selection"]
    sealed = frozen["sealed_sets"]
    checks = (
        (selection["authoritative_manifest"], selection["authoritative_manifest_sha256"]),
        (selection["quarantine"], selection["quarantine_sha256"]),
        (sealed["audit_exclusion_ledger"], sealed["audit_exclusion_ledger_sha256"]),
    )
    loaded: dict[str, dict[str, Any]] = {}
    with AnchoredRoot.open(context.repository) as root:
        for relative, expected_sha in checks:
            raw, _ = root.read_file(relative)
            if _sha256_bytes(raw) != expected_sha:
                _fail(f"source-selection metadata hash mismatch: {relative}")
            loaded[relative] = _require_object(
                _parse_json(raw, label=relative, canonical_file=False), label=relative
            )
    manifest = loaded[selection["authoritative_manifest"]]
    splits = _require_object(manifest.get("splits"), label="denoise splits")
    train = [str(value) for value in splits.get("train", [])]
    validation = [str(value) for value in splits.get("val", [])]
    audit = [str(value) for value in splits.get("audit", [])]
    if (len(train), len(validation), len(audit)) != (4900, 700, 700):
        _fail("authoritative split cardinality drift")
    ranked = sorted(
        train,
        key=lambda name: (
            hashlib.sha256(f"assembly-v1:20260710:{name}".encode("utf-8")).digest(),
            name,
        ),
    )
    edge_development = ranked[4500:]
    if selection["split"] != "edge_development":
        _fail("oracle source split drift")
    offset = _require_exact_int(selection["offset"], label="source offset", minimum=0)
    count = _require_exact_int(selection["count"], label="source count", minimum=1)
    names = edge_development[offset : offset + count]
    if len(names) != 32 or _sha256_bytes("\n".join(names).encode("utf-8")) != EXPECTED_NAMES_SHA256:
        _fail("independent source slice/hash mismatch")
    return names


@dataclass
class LabelRecord:
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]


@dataclass
class LabelEvidence:
    root_path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    records: dict[str, LabelRecord]
    secret: bytes
    source_names: list[str]


def verify_label_fixture_after_marker(
    context: ProtocolContext,
    *,
    labels_root: str,
    input_evidence: InputEvidence,
    lifecycle: LifecycleEvidence,
) -> LabelEvidence:
    expected_sha = _require_sha(
        context.config["runtime_pins"].get("fixture_label_manifest_sha256"),
        label="fixture_label_manifest_sha256",
    )
    source_names = _independent_source_names(context)
    with AnchoredRoot.open(labels_root) as root:
        manifest_raw, _ = root.read_file(LABEL_MANIFEST)
        if _sha256_bytes(manifest_raw) != expected_sha:
            _fail("label manifest out-of-band hash mismatch")
        manifest = _require_object(
            _parse_json(manifest_raw, label=LABEL_MANIFEST, canonical_file=True),
            label=LABEL_MANIFEST,
        )
        expected_keys = {
            "schema_version",
            "created_utc",
            "kind",
            *_common_manifest_fields(context),
            "fixture_input_manifest_sha256",
            "record_count",
            "opaque_ids_sha256",
            "canonical_record_order",
            "hidden_panel_counts",
            "master_secret",
            "records",
        }
        _require_exact_keys(manifest, expected_keys, label=LABEL_MANIFEST)
        _verify_common_manifest_bindings(manifest, context)
        expected_values = {
            "schema_version": 1,
            "kind": "candidate_graph_oracle_fixture_labels",
            "fixture_input_manifest_sha256": input_evidence.manifest_sha256,
            "record_count": 64,
            "canonical_record_order": "ascending opaque_id",
            "hidden_panel_counts": {panel: 32 for panel in PANELS},
        }
        for key, expected in expected_values.items():
            if manifest.get(key) != expected:
                _fail(f"label manifest invariant drift: {key}")
        _require_utc(manifest["created_utc"], label="label.created_utc")
        secret_descriptor = _require_object(
            manifest["master_secret"], label="label.master_secret"
        )
        _require_exact_keys(
            secret_descriptor, {"path", "bytes", "sha256", "mode"}, label="label.master_secret"
        )
        if (
            secret_descriptor["path"] != MASTER_SECRET
            or secret_descriptor["bytes"] != 32
            or secret_descriptor["mode"] != "0600"
        ):
            _fail("label master-secret descriptor drift")
        secret, secret_info = root.read_file(MASTER_SECRET, expected_size=32)
        if _sha256_bytes(secret) != _require_sha(secret_descriptor["sha256"], label="master_secret.sha256"):
            _fail("label master secret hash mismatch")
        if stat.S_IMODE(secret_info.st_mode) != 0o600:
            _fail("label master secret mode mismatch")

        record_values = manifest.get("records")
        if not isinstance(record_values, list) or len(record_values) != 64:
            _fail("label manifest must contain exactly 64 records")
        records: dict[str, LabelRecord] = {}
        artifact_names: set[str] = set()
        ordered_ids: list[str] = []
        coverage: set[tuple[str, str]] = set()
        for index, record_value in enumerate(record_values):
            record = _require_object(record_value, label=f"label.records[{index}]")
            _require_exact_keys(
                record,
                {
                    "opaque_id",
                    "source_name",
                    "panel",
                    "panel_seed",
                    "target_file_sha256",
                    "artifact",
                    "arrays",
                },
                label=f"label.records[{index}]",
            )
            opaque_id = record["opaque_id"]
            source_name = record["source_name"]
            panel = record["panel"]
            if not isinstance(opaque_id, str) or not OPAQUE_ID_RE.fullmatch(opaque_id):
                _fail("label opaque id drift")
            if source_name not in source_names or panel not in PANELS:
                _fail("label source/panel drift")
            if opaque_id in records:
                _fail("duplicate label opaque id")
            coverage.add((source_name, panel))
            panel_seed = _require_exact_int(record["panel_seed"], label="label.panel_seed", minimum=0)
            if panel_seed >= 2**64:
                _fail("label panel seed outside uint64")
            _require_sha(record["target_file_sha256"], label="label.target_file_sha256")
            artifact_bytes, _, relative = _verify_artifact_descriptor(
                record["artifact"],
                root=root,
                expected_parent="records",
                suffix=".npz",
                label=f"label.records[{index}].artifact",
            )
            if PurePosixPath(relative).name != f"{opaque_id}.npz":
                _fail("label artifact name drift")
            arrays = _strict_npz(
                artifact_bytes, LABEL_ARRAY_SPECS, label=f"label record {opaque_id}"
            )
            _validate_layout(arrays["opaque_slot_permutation"], label="opaque_slot_permutation")
            _validate_layout(arrays["composed_slot_to_target"], label="composed_slot_to_target")
            descriptors = _require_object(record["arrays"], label=f"label.records[{index}].arrays")
            _require_exact_keys(descriptors, LABEL_ARRAY_SPECS, label=f"label.records[{index}].arrays")
            for key in LABEL_ARRAY_SPECS:
                _verify_array_descriptor(
                    descriptors[key],
                    arrays[key],
                    semantic=LABEL_ARRAY_SEMANTICS[key],
                    label=f"label.records[{index}].arrays.{key}",
                )
            ordered_ids.append(opaque_id)
            artifact_names.add(PurePosixPath(relative).name)
            records[opaque_id] = LabelRecord(dict(record), arrays)
        if ordered_ids != sorted(ordered_ids) or ordered_ids != sorted(input_evidence.records):
            _fail("label/input opaque-id bijection or order mismatch")
        if coverage != {(source, panel) for source in source_names for panel in PANELS}:
            _fail("label source-panel coverage mismatch")
        ids_sha = _sha256_bytes("\n".join(ordered_ids).encode("ascii"))
        if manifest["opaque_ids_sha256"] != ids_sha or ids_sha != input_evidence.manifest["opaque_ids_sha256"]:
            _fail("label opaque-id list hash mismatch")
        root.assert_exact_tree(
            top_files={LABEL_MANIFEST, MASTER_SECRET},
            directories={"records": artifact_names},
        )
    return LabelEvidence(
        Path(labels_root).expanduser().absolute(),
        manifest,
        expected_sha,
        records,
        secret,
        source_names,
    )


def verify_fixture_control(
    context: ProtocolContext,
    *,
    control_root: str | Path,
    input_evidence: InputEvidence,
    labels: LabelEvidence,
    lifecycle: LifecycleEvidence,
) -> None:
    with AnchoredRoot.open(control_root) as root:
        marker_raw, _ = root.read_file(FIXTURE_PREP_MARKER)
        lock_raw, _ = root.read_file(FIXTURE_LOCK)
        marker = _require_object(
            _parse_json(marker_raw, label=FIXTURE_PREP_MARKER, canonical_file=True),
            label=FIXTURE_PREP_MARKER,
        )
        lock = _require_object(
            _parse_json(lock_raw, label=FIXTURE_LOCK, canonical_file=True),
            label=FIXTURE_LOCK,
        )
        if _sha256_bytes(lock_raw) != context.config["runtime_pins"]["fixture_lock_sha256"]:
            _fail("fixture lock differs from final runtime pin")
        marker_keys = {
            "schema_version",
            "kind",
            "started_utc",
            *_common_manifest_fields(context),
            "builder_path",
            "builder_sha256",
            "config_sha256_before_pixels",
            "source_names_sha256",
            "source_count",
            "expected_fixture_records",
            "prep_lifecycle_sha256",
        }
        _require_exact_keys(marker, marker_keys, label=FIXTURE_PREP_MARKER)
        _verify_common_manifest_bindings(marker, context)
        _require_utc(marker["started_utc"], label="fixture_marker.started_utc")
        expected_marker = {
            "schema_version": 1,
            "kind": "candidate_graph_oracle_fixture_pixel_access_started",
            "builder_path": context.config["runtime_pins"]["fixture_builder_path"],
            "builder_sha256": context.config["runtime_pins"]["fixture_builder_sha256"],
            "config_sha256_before_pixels": lifecycle.code_config_sha256,
            "source_names_sha256": EXPECTED_NAMES_SHA256,
            "source_count": 32,
            "expected_fixture_records": 64,
            "prep_lifecycle_sha256": lifecycle.hashes["PREP"],
        }
        for key, expected in expected_marker.items():
            if marker.get(key) != expected:
                _fail(f"fixture prep marker invariant drift: {key}")
        lock_keys = {
            "schema_version",
            "kind",
            "created_utc",
            *_common_manifest_fields(context),
            "prep_marker_sha256",
            "prep_lifecycle_sha256",
            "fixture_input_manifest_sha256",
            "fixture_label_manifest_sha256",
            "record_count",
            "opaque_ids_sha256",
            "input_and_label_roots_are_distinct_siblings",
            "phase_a_may_receive_label_root",
            "phase_a_may_receive_master_secret",
        }
        _require_exact_keys(lock, lock_keys, label=FIXTURE_LOCK)
        _verify_common_manifest_bindings(lock, context)
        _require_utc(lock["created_utc"], label="fixture_lock.created_utc")
        marker_sha = _sha256_bytes(marker_raw)
        expected_lock = {
            "schema_version": 1,
            "kind": "candidate_graph_oracle_fixture_lock",
            "prep_marker_sha256": marker_sha,
            "prep_lifecycle_sha256": lifecycle.hashes["PREP"],
            "fixture_input_manifest_sha256": input_evidence.manifest_sha256,
            "fixture_label_manifest_sha256": labels.manifest_sha256,
            "record_count": 64,
            "opaque_ids_sha256": input_evidence.manifest["opaque_ids_sha256"],
            "input_and_label_roots_are_distinct_siblings": True,
            "phase_a_may_receive_label_root": False,
            "phase_a_may_receive_master_secret": False,
        }
        for key, expected in expected_lock.items():
            if lock.get(key) != expected:
                _fail(f"fixture lock invariant drift: {key}")
        secret_sha = labels.manifest["master_secret"]["sha256"]
        if _contains_exact_string(input_evidence.manifest, secret_sha) or _contains_exact_string(lock, secret_sha):
            _fail("input manifest or fixture lock binds the label-only secret hash")
        forbidden_config_hashes = {
            lifecycle.code_config_sha256,
            context.config_sha256,
        }
        for payload, label in (
            (input_evidence.manifest, "input_manifest"),
            (labels.manifest, "label_manifest"),
            (lock, "fixture_lock"),
        ):
            _forbid_whole_config_binding(
                payload,
                forbidden_hashes=forbidden_config_hashes,
                path=label,
            )
        root.assert_exact_tree(
            top_files={FIXTURE_PREP_MARKER, FIXTURE_LOCK}, directories={}
        )


def _per_source_seed(master: int, stage: str, source: str) -> int:
    digest = hashlib.sha256(f"{master}:{stage}:{source}:0".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


@dataclass(frozen=True)
class Recomposition:
    truth: np.ndarray
    clean_target: np.ndarray
    panel: str
    source_name: str
    source_index: int
    panel_seed: int


def recompose_fixture(
    context: ProtocolContext,
    *,
    opaque_id: str,
    input_record: InputRecord,
    label_record: LabelRecord,
    secret: bytes,
    source_names: Sequence[str],
) -> Recomposition:
    record = label_record.manifest
    source = str(record["source_name"])
    panel = str(record["panel"])
    if len(secret) != 32:
        _fail("fixture master-secret length drift")
    id_material = hmac.new(
        secret, f"id:{source}:{panel}".encode("utf-8"), hashlib.sha256
    ).digest()
    if not hmac.compare_digest(id_material[:16].hex(), opaque_id):
        _fail("independent opaque-id HMAC mismatch")
    shuffle_material = hmac.new(
        secret, f"shuffle:{source}:{panel}".encode("utf-8"), hashlib.sha256
    ).digest()
    shuffle_seed = int.from_bytes(shuffle_material[:8], "big", signed=False)
    permutation = (
        np.random.Generator(np.random.PCG64(shuffle_seed))
        .permutation(TILE_COUNT)
        .astype(np.int32)
    )
    if not np.array_equal(permutation, label_record.arrays["opaque_slot_permutation"]):
        _fail("independent opaque shuffle mismatch")
    master_seed = context.config["frozen_contract"]["synthetic_corruption"]["master_seed"]
    panel_seed = _per_source_seed(
        int(master_seed), f"candidate-graph-oracle-{panel}", source
    )
    if panel_seed != record["panel_seed"]:
        _fail("independent panel seed mismatch")
    clean = np.ascontiguousarray(label_record.arrays["clean_target_rgb"])
    exact = make_exact_panel(clean, panel=panel, seed=panel_seed)
    opaque_tiles = np.ascontiguousarray(np.asarray(exact.slot_tiles)[permutation])
    composed_truth = validate_permutation(
        np.asarray(exact.slot_to_target)[permutation], name="independent composed truth"
    ).astype(np.int32, copy=False)
    if not np.array_equal(opaque_tiles, input_record.arrays["slot_tiles"]):
        _fail("independent make_exact_panel/opaque input bytes mismatch")
    if not np.array_equal(composed_truth, label_record.arrays["composed_slot_to_target"]):
        _fail("independent composed truth mismatch")
    if int(input_record.arrays["qap_seed"]) != _opaque_qap_seed(opaque_id):
        _fail("independent opaque nuisance seed mismatch")
    try:
        source_index = list(source_names).index(source)
    except ValueError as error:
        raise VerificationError("label source missing from frozen slice") from error
    return Recomposition(composed_truth, clean, panel, source, source_index, panel_seed)


def _candidate_truth_metrics(
    arrays: Mapping[str, np.ndarray], truth: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    truth = validate_permutation(truth, name="truth")
    right, down = true_neighbour_slots(truth)
    if int(np.count_nonzero(right >= 0)) != 552 or int(np.count_nonzero(down >= 0)) != 552:
        _fail("truth edge cardinality drift")
    direction = arrays["candidate_direction"]
    source = arrays["candidate_source"]
    destination = arrays["candidate_destination"]
    masks = arrays["candidate_origin_mask"]
    lookup = {
        (int(d), int(first), int(second)): int(mask)
        for d, first, second, mask in zip(
            direction, source, destination, masks, strict=True
        )
    }
    if len(lookup) != len(direction):
        _fail("candidate graph keys are not unique")
    side_hits: dict[str, list[bool]] = {key: [] for key in ("right", "left", "down", "up")}
    unique_hits: list[bool] = []
    origin_hits: dict[str, list[bool]] = {key: [] for key in ORIGIN_BITS}
    true_candidate_edges: list[tuple[int, int, int]] = []
    for d, neighbours, outgoing, incoming in (
        (0, right, "right", "left"),
        (1, down, "down", "up"),
    ):
        for first in np.flatnonzero(neighbours >= 0).tolist():
            second = int(neighbours[first])
            mask = lookup.get((d, int(first), second), 0)
            hit = mask != 0
            unique_hits.append(hit)
            side_hits[outgoing].append(hit)
            side_hits[incoming].append(hit)
            for origin, bit in ORIGIN_BITS.items():
                origin_hits[origin].append(bool(mask & bit))
            if hit:
                true_candidate_edges.append((d, int(first), second))
    if len(unique_hits) != 1104:
        _fail("truth metric denominator drift")
    recall = {
        "truth_unique_edges": 1104,
        "truth_four_side_queries": 2208,
        "unique_true_edge_recall": float(np.mean(unique_hits)),
        "candidate_restricted_attainable_adjacency_fraction": float(
            np.mean(unique_hits)
        ),
        "four_side_recall": float(np.mean([hit for values in side_hits.values() for hit in values])),
        "side_recall": {
            key: float(np.mean(values)) for key, values in side_hits.items()
        },
        "origin_unique_true_edge_recall": {
            key: float(np.mean(values)) for key, values in origin_hits.items()
        },
    }
    if recall["side_recall"]["right"] != recall["side_recall"]["left"] or recall["side_recall"]["down"] != recall["side_recall"]["up"]:
        _fail("inverse-side recall identity failed")
    if recall["four_side_recall"] != recall["unique_true_edge_recall"]:
        _fail("four-side/unique recall identity failed")

    parent = np.arange(TILE_COUNT, dtype=np.int32)
    size = np.ones(TILE_COUNT, dtype=np.int32)
    minimum = np.arange(TILE_COUNT, dtype=np.int32)

    def find(node: int) -> int:
        while int(parent[node]) != node:
            parent[node] = parent[int(parent[node])]
            node = int(parent[node])
        return node

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            return
        if size[first_root] < size[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        size[first_root] += size[second_root]
        minimum[first_root] = min(int(minimum[first_root]), int(minimum[second_root]))

    for d, first, second in true_candidate_edges:
        first_target, second_target = int(truth[first]), int(truth[second])
        if d == 0:
            if second_target != first_target + 1 or first_target // GRID != second_target // GRID:
                _fail("false edge survived truth intersection")
        elif second_target != first_target + GRID:
            _fail("false edge survived truth intersection")
        union(first, second)
    groups: dict[int, list[int]] = {}
    for tile in range(TILE_COUNT):
        groups.setdefault(find(tile), []).append(tile)
    ordered_groups = sorted(groups.values(), key=lambda values: (-len(values), min(values)))
    component_sizes = [len(values) for values in ordered_groups]
    components = {
        "truth_filtered_candidate_edges": len(true_candidate_edges),
        "accepted_consistent_edges": len(true_candidate_edges),
        "connected_component_count": len(component_sizes),
        "component_sizes": component_sizes,
        "largest_connected_component": component_sizes[0],
        "non_singleton_covered_tile_fraction": float(
            sum(size for size in component_sizes if size >= 2) / TILE_COUNT
        ),
        "partition_and_relative_offsets_independently_verified": True,
    }
    origin_counts = {
        key: int(np.count_nonzero(masks.astype(np.int64) & bit))
        for key, bit in ORIGIN_BITS.items()
    }
    return recall, components, origin_counts


def _compare_json_numeric(actual: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, dict):
        value = _require_object(actual, label=label)
        _require_exact_keys(value, expected, label=label)
        for key in expected:
            _compare_json_numeric(value[key], expected[key], label=f"{label}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            _fail(f"list mismatch: {label}")
        for index, (got, wanted) in enumerate(zip(actual, expected, strict=True)):
            _compare_json_numeric(got, wanted, label=f"{label}[{index}]")
    elif isinstance(expected, float):
        _assert_close(actual, expected, label=label)
    else:
        if actual != expected:
            _fail(f"value mismatch: {label}")


def _layout_and_ssim(
    layout: np.ndarray, truth: np.ndarray, tiles: np.ndarray, clean: np.ndarray
) -> tuple[dict[str, Any], float, np.ndarray]:
    layout = _validate_layout(layout, label="scored layout")
    metrics = layout_metrics(layout, truth)
    render = _merge_tiles(tiles[layout])
    score = float(
        structural_similarity(clean, render, channel_axis=2, data_range=255)
    )
    if not math.isfinite(score):
        _fail("non-finite independent RGB SSIM")
    return metrics, score, render


def _independent_gate(panel_summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(panel_summaries) != set(PANELS):
        _fail("gate requires exactly two panels")
    guards: dict[str, dict[str, bool]] = {}
    for panel in PANELS:
        summary = panel_summaries[panel]
        recall = _require_finite_float(summary["mean_union_true_edge_recall"], label=f"{panel}.recall")
        lcc = _require_finite_float(summary["median_largest_connected_component"], label=f"{panel}.lcc")
        adjacency = _require_finite_float(summary["mean_beam_qap_adjacency_delta"], label=f"{panel}.adj")
        ssim = _require_finite_float(summary["mean_beam_qap_ssim_delta"], label=f"{panel}.ssim")
        guards[panel] = {
            "union_true_edge_recall_ge_0.65": recall >= 0.65,
            "median_lcc_ge_128": lcc >= 128.0,
            "beam_qap_adjacency_delta_nonnegative": adjacency >= 0.0,
            "beam_qap_ssim_delta_nonnegative": ssim >= 0.0,
        }
    macro_adjacency = float(
        np.mean([panel_summaries[panel]["mean_beam_qap_adjacency_delta"] for panel in PANELS])
    )
    macro_ssim = float(
        np.mean([panel_summaries[panel]["mean_beam_qap_ssim_delta"] for panel in PANELS])
    )
    all_panel = all(all(values.values()) for values in guards.values())
    major = macro_adjacency >= 0.10 or macro_ssim >= 0.02
    return {
        "panel_guards": guards,
        "all_panel_guards_passed": all_panel,
        "balanced_panel_macro_adjacency_delta": macro_adjacency,
        "balanced_panel_macro_ssim_delta": macro_ssim,
        "major_gain_adjacency_ge_0.10": macro_adjacency >= 0.10,
        "major_gain_ssim_ge_0.02": macro_ssim >= 0.02,
        "major_gain_or_passed": major,
        "continue_to_cycle_factor_synchronizer": bool(all_panel and major),
        "safe_for_submission": False,
    }


def _panel_summaries(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for panel in PANELS:
        selected = [record for record in records if record["panel"] == panel]
        if len(selected) != 32:
            _fail(f"panel record count drift: {panel}")
        recalls = np.asarray(
            [record["candidate_recall"]["unique_true_edge_recall"] for record in selected],
            dtype=np.float64,
        )
        lcc = np.asarray(
            [record["components"]["largest_connected_component"] for record in selected],
            dtype=np.float64,
        )
        adjacency = np.asarray(
            [record["paired_delta"]["combined_adjacency"] for record in selected],
            dtype=np.float64,
        )
        ssim = np.asarray(
            [record["paired_delta"]["rgb_ssim"] for record in selected],
            dtype=np.float64,
        )
        if not all(np.all(np.isfinite(value)) for value in (recalls, lcc, adjacency, ssim)):
            _fail("non-finite panel aggregate input")
        result[panel] = {
            "record_count": 32.0,
            "mean_union_true_edge_recall": float(np.mean(recalls)),
            "median_largest_connected_component": float(np.median(lcc)),
            "mean_beam_qap_adjacency_delta": float(np.mean(adjacency)),
            "mean_beam_qap_ssim_delta": float(np.mean(ssim)),
        }
    return result


def _load_npy_int32_layout(payload: bytes, *, label: str) -> np.ndarray:
    source = BytesIO(payload)
    try:
        value = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise VerificationError(f"invalid NPY layout: {label}") from error
    if source.tell() != len(payload):
        _fail(f"trailing bytes in NPY layout: {label}")
    value = np.asarray(value)
    return _validate_layout(value, label=label)


def _validate_layout_report_shape(value: Any, *, label: str) -> dict[str, Any]:
    result = _require_object(value, label=label)
    _require_exact_keys(result, {"layout_sha256", "layout", "render"}, label=label)
    _require_sha(result["layout_sha256"], label=f"{label}.layout_sha256")
    layout_value = _require_object(result["layout"], label=f"{label}.layout")
    expected_layout_keys = {
        "valid_permutation",
        "position_accuracy",
        "row_accuracy",
        "column_accuracy",
        "mean_manhattan",
        "median_manhattan",
        "q90_manhattan",
        "within_one_manhattan",
        "right_adjacency",
        "down_adjacency",
        "combined_adjacency",
        "exact_solved",
        "boundary_position_accuracy",
        "corner_position_accuracy",
        "largest_correct_component",
    }
    _require_exact_keys(layout_value, expected_layout_keys, label=f"{label}.layout")
    if layout_value["valid_permutation"] is not True:
        _fail(f"reported layout is not valid: {label}")
    for key, nested in layout_value.items():
        if key in {"valid_permutation", "exact_solved"}:
            if not isinstance(nested, bool):
                _fail(f"reported layout boolean drift: {label}.{key}")
        elif key == "largest_correct_component":
            _require_exact_int(nested, label=f"{label}.{key}", minimum=1)
        else:
            _require_finite_float(nested, label=f"{label}.{key}")
    render = _require_object(result["render"], label=f"{label}.render")
    _require_exact_keys(render, {"rgb_ssim"}, label=f"{label}.render")
    _require_finite_float(render["rgb_ssim"], label=f"{label}.render.rgb_ssim")
    return result


def _verify_target_access_marker(
    context: ProtocolContext,
    *,
    phase_b_root: AnchoredRoot,
    phase_a: PhaseAEvidence,
    lifecycle: LifecycleEvidence,
) -> tuple[dict[str, Any], str]:
    marker_raw, _ = phase_b_root.read_file(TARGET_MARKER)
    marker_sha = _sha256_bytes(marker_raw)
    marker = _require_object(
        _parse_json(marker_raw, label=TARGET_MARKER, canonical_file=True),
        label=TARGET_MARKER,
    )
    expected_keys = {
        "schema_version",
        "kind",
        "config_sha256",
        "protocol_instance_id",
        "frozen_contract_sha256",
        "phase_a_envelope_sha256",
        "script_sha256",
        "label_access_lifecycle_sha256",
        "label_paths_constructed_before_marker",
        "label_files_opened_before_marker",
    }
    _require_exact_keys(marker, expected_keys, label=TARGET_MARKER)
    expected = {
        "schema_version": 1,
        "kind": "candidate_graph_target_access_started",
        "config_sha256": context.config_sha256,
        "protocol_instance_id": EXPECTED_PROTOCOL_INSTANCE_ID,
        "frozen_contract_sha256": EXPECTED_FROZEN_CONTRACT_SHA256,
        "phase_a_envelope_sha256": phase_a.envelope_sha256,
        "script_sha256": context.config["runtime_pins"]["evaluator_sha256"],
        "label_access_lifecycle_sha256": lifecycle.hashes["LABEL_ACCESS"],
        "label_paths_constructed_before_marker": False,
        "label_files_opened_before_marker": False,
    }
    for key, expected_value in expected.items():
        if marker.get(key) != expected_value:
            _fail(f"target-access marker invariant drift: {key}")
    return marker, marker_sha


def _reload_phase_a_scoring_arrays(
    root: AnchoredRoot, record: PhaseARecord
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    descriptor = record.manifest
    graph_raw, _ = root.read_file(
        descriptor["graph_artifact"],
        expected_size=int(descriptor["graph_artifact_byte_size"]),
    )
    if _sha256_bytes(graph_raw) != descriptor["graph_artifact_sha256"]:
        _fail("Phase-A graph changed before independent Phase-B scoring")
    arrays = _strict_npz(
        graph_raw,
        GRAPH_ARRAY_SPECS,
        label=f"Phase-B reload {descriptor['opaque_id']}",
    )
    # Recheck the independent union on the exact second read.  This is not
    # redundant: it closes a TOCTOU window between Phase-A verification and
    # target-aware scoring without retaining ~1 GB of graph archives in RAM.
    rebuilt = rebuild_candidate_union(arrays)
    for key, expected in rebuilt.items():
        if not np.array_equal(arrays[key], expected):
            _fail(f"Phase-A graph changed semantically before scoring: {key}")
    render_descriptor = descriptor["renders"]["qap_w4"]
    render_raw, _ = root.read_file(render_descriptor["path"])
    if _sha256_bytes(render_raw) != render_descriptor["sha256"]:
        _fail("Phase-A w4 render changed before independent scoring")
    return arrays, _decode_png(render_raw, label=render_descriptor["path"])


def _verify_record_report(
    report_value: Any,
    *,
    index: int,
    output_root: AnchoredRoot,
    phase_a_arrays: Mapping[str, np.ndarray],
    phase_a_w4_render: np.ndarray,
    recomposed: Recomposition,
) -> tuple[dict[str, Any], str, str]:
    report = _require_object(report_value, label=f"report.records[{index}]")
    expected_keys = {
        "opaque_id",
        "source_index",
        "name",
        "panel",
        "panel_seed",
        "candidate_edge_count",
        "candidate_origin_counts",
        "candidate_recall",
        "fixture_recomposition",
        "components",
        "layouts",
        "oracle_filter_diagnostics",
        "target_assisted_translation_diagnostics",
        "paired_delta",
        "artifacts",
    }
    _require_exact_keys(report, expected_keys, label=f"report.records[{index}]")
    opaque_id = report["opaque_id"]
    if not isinstance(opaque_id, str) or not OPAQUE_ID_RE.fullmatch(opaque_id):
        _fail("report opaque id drift")
    expected_identity = {
        "source_index": recomposed.source_index,
        "name": recomposed.source_name,
        "panel": recomposed.panel,
        "panel_seed": recomposed.panel_seed,
        "candidate_edge_count": len(phase_a_arrays["candidate_direction"]),
    }
    for key, expected in expected_identity.items():
        if report.get(key) != expected:
            _fail(f"report identity/count mismatch: {opaque_id}.{key}")

    recall, components, origin_counts = _candidate_truth_metrics(
        phase_a_arrays, recomposed.truth
    )
    _compare_json_numeric(
        report["candidate_origin_counts"], origin_counts, label=f"{opaque_id}.origin_counts"
    )
    _compare_json_numeric(
        report["candidate_recall"], recall, label=f"{opaque_id}.candidate_recall"
    )
    _compare_json_numeric(
        report["components"], components, label=f"{opaque_id}.components"
    )
    expected_recomposition = {
        "opaque_id_recomputed": True,
        "opaque_permutation_recomputed": True,
        "panel_seed_recomputed": True,
        "opaque_slot_tiles_recomputed": True,
        "composed_truth_recomputed": True,
        "opaque_qap_seed_recomputed": True,
        "truth_geometry_verified": True,
    }
    if report["fixture_recomposition"] != expected_recomposition:
        _fail("report fixture recomposition receipt drift")

    artifacts = _require_object(report["artifacts"], label=f"{opaque_id}.artifacts")
    _require_exact_keys(artifacts, {"oracle_layout", "oracle_render"}, label=f"{opaque_id}.artifacts")
    layout_descriptor = _require_object(
        artifacts["oracle_layout"], label=f"{opaque_id}.oracle_layout"
    )
    render_descriptor = _require_object(
        artifacts["oracle_render"], label=f"{opaque_id}.oracle_render"
    )
    for descriptor, label in (
        (layout_descriptor, "oracle_layout"),
        (render_descriptor, "oracle_render"),
    ):
        _require_exact_keys(descriptor, {"path", "sha256"}, label=f"{opaque_id}.{label}")
    layout_relative = _valid_relative_path(
        layout_descriptor["path"], parent="artifacts", suffix=".npy"
    )
    render_relative = _valid_relative_path(
        render_descriptor["path"], parent="renders", suffix=".png"
    )
    if PurePosixPath(layout_relative).name != f"{opaque_id}__oracle_layout.npy":
        _fail("oracle layout artifact name drift")
    if PurePosixPath(render_relative).name != f"{opaque_id}__oracle.png":
        _fail("oracle render artifact name drift")
    layout_bytes, _ = output_root.read_file(layout_relative)
    render_bytes, _ = output_root.read_file(render_relative)
    if _sha256_bytes(layout_bytes) != _require_sha(layout_descriptor["sha256"], label="oracle_layout.sha256"):
        _fail("oracle layout artifact hash mismatch")
    if _sha256_bytes(render_bytes) != _require_sha(render_descriptor["sha256"], label="oracle_render.sha256"):
        _fail("oracle render artifact hash mismatch")
    oracle_layout = _load_npy_int32_layout(layout_bytes, label=layout_relative)
    decoded_oracle = _decode_png(render_bytes, label=render_relative)

    arrays = phase_a_arrays
    layout_values = _require_object(report["layouts"], label=f"{opaque_id}.layouts")
    required_layouts = {
        "softcycle_l1_k8",
        "qap_w4_b0.05_i25",
        "qap_w1_b0.05_i25",
        "oracle_filter_beam8_hungarian_qap25",
        "absolute_true_component_translation_ceiling",
    }
    _require_exact_keys(layout_values, required_layouts, label=f"{opaque_id}.layouts")
    for label in required_layouts:
        _validate_layout_report_shape(layout_values[label], label=f"{opaque_id}.layouts.{label}")

    layouts_to_recompute = {
        "softcycle_l1_k8": arrays["softcycle_layout"],
        "qap_w4_b0.05_i25": arrays["qap_w4_layout"],
        "qap_w1_b0.05_i25": arrays["qap_w1_layout"],
        "oracle_filter_beam8_hungarian_qap25": oracle_layout,
    }
    recomputed_metrics: dict[str, tuple[dict[str, Any], float, np.ndarray]] = {}
    for label, layout in layouts_to_recompute.items():
        metrics, ssim, render = _layout_and_ssim(
            layout,
            recomposed.truth,
            arrays["denoised_tiles"],
            recomposed.clean_target,
        )
        recomputed_metrics[label] = (metrics, ssim, render)
        reported = layout_values[label]
        if reported["layout_sha256"] != _array_sha256(layout):
            _fail(f"reported layout hash mismatch: {opaque_id}.{label}")
        _compare_json_numeric(
            reported["layout"], metrics, label=f"{opaque_id}.{label}.layout"
        )
        _assert_close(
            reported["render"]["rgb_ssim"],
            ssim,
            label=f"{opaque_id}.{label}.rgb_ssim",
        )
    if not np.array_equal(
        phase_a_w4_render,
        recomputed_metrics["qap_w4_b0.05_i25"][2],
    ):
        _fail("Phase-A w4 render differs during Phase-B recomputation")
    if not np.array_equal(
        decoded_oracle,
        recomputed_metrics["oracle_filter_beam8_hungarian_qap25"][2],
    ):
        _fail("Phase-B oracle render differs from frozen tiles/layout")

    oracle_diagnostics = _require_object(
        report["oracle_filter_diagnostics"], label=f"{opaque_id}.oracle_filter_diagnostics"
    )
    oracle_keys = {
        "beam_width",
        "beam_components",
        "translations_per_state",
        "placement_costs",
        "multi_tile_components",
        "beam_placed_tiles",
        "unresolved_before_hungarian",
        "hungarian_received_grid_copy",
        "pre_qap_layout_sha256",
        "qap_iterations",
        "qap_restarts",
        "qap_seed",
        "qap_objective",
        "qap_relaxed_objective",
        "qap_restart",
        "layout_sha256",
    }
    _require_exact_keys(oracle_diagnostics, oracle_keys, label=f"{opaque_id}.oracle_filter_diagnostics")
    expected_oracle_fields = {
        "beam_width": 8,
        "beam_components": 8,
        "translations_per_state": 8,
        "placement_costs": None,
        "hungarian_received_grid_copy": True,
        "qap_iterations": 25,
        "qap_restarts": 2,
        "qap_seed": _opaque_qap_seed(opaque_id),
        "layout_sha256": _array_sha256(oracle_layout),
    }
    for key, expected in expected_oracle_fields.items():
        if oracle_diagnostics.get(key) != expected:
            _fail(f"oracle packer diagnostic contract drift: {opaque_id}.{key}")
    for key in ("multi_tile_components", "beam_placed_tiles", "unresolved_before_hungarian", "qap_restart"):
        _require_exact_int(oracle_diagnostics[key], label=f"{opaque_id}.{key}", minimum=0)
    _require_sha(oracle_diagnostics["pre_qap_layout_sha256"], label="pre_qap_layout_sha256")
    _require_finite_float(oracle_diagnostics["qap_objective"], label="qap_objective")
    _require_finite_float(oracle_diagnostics["qap_relaxed_objective"], label="qap_relaxed_objective")

    translation = _require_object(
        report["target_assisted_translation_diagnostics"],
        label=f"{opaque_id}.translation_diagnostics",
    )
    translation_keys = {
        "diagnostic_only",
        "eligible_for_gate",
        "non_singleton_assisted_tiles",
        "singleton_truth_placements",
        "unresolved_before_w4_hungarian",
        "post_completion_qap",
        "accidentally_exact_after_baseline_repair",
        "layout_sha256",
    }
    _require_exact_keys(translation, translation_keys, label=f"{opaque_id}.translation_diagnostics")
    if (
        translation["diagnostic_only"] is not True
        or translation["eligible_for_gate"] is not False
        or translation["singleton_truth_placements"] != 0
        or translation["post_completion_qap"] is not False
    ):
        _fail("target-assisted translation diagnostic eligibility drift")
    _require_exact_int(translation["non_singleton_assisted_tiles"], label="translation.assisted_tiles", minimum=0)
    _require_exact_int(translation["unresolved_before_w4_hungarian"], label="translation.unresolved", minimum=0)
    if not isinstance(translation["accidentally_exact_after_baseline_repair"], bool):
        _fail("translation accidental-perfect flag drift")
    if translation["layout_sha256"] != layout_values["absolute_true_component_translation_ceiling"]["layout_sha256"]:
        _fail("translation diagnostic/layout hash mismatch")

    baseline_metrics, baseline_ssim, _ = recomputed_metrics["qap_w4_b0.05_i25"]
    oracle_metrics, oracle_ssim, _ = recomputed_metrics[
        "oracle_filter_beam8_hungarian_qap25"
    ]
    paired = {
        "combined_adjacency": float(
            oracle_metrics["combined_adjacency"]
            - baseline_metrics["combined_adjacency"]
        ),
        "rgb_ssim": float(oracle_ssim - baseline_ssim),
    }
    _compare_json_numeric(report["paired_delta"], paired, label=f"{opaque_id}.paired_delta")
    verified_report = dict(report)
    verified_report["candidate_recall"] = recall
    verified_report["components"] = components
    verified_report["paired_delta"] = paired
    return verified_report, PurePosixPath(layout_relative).name, PurePosixPath(render_relative).name


@dataclass(frozen=True)
class PhaseBVerification:
    report_sha256: str
    status: str
    continuation_gate_passed: bool


@dataclass(frozen=True)
class PhaseBRunnerAttestation:
    path: Path
    sha256: str
    payload: dict[str, Any]


def _pinned_fixture_relative_parts(
    context: ProtocolContext,
) -> tuple[PurePosixPath, PurePosixPath, PurePosixPath, PurePosixPath]:
    pins = context.config["runtime_pins"]
    input_manifest = PurePosixPath(str(pins["fixture_input_manifest_relative_path"]))
    label_manifest = PurePosixPath(str(pins["fixture_label_manifest_relative_path"]))
    fixture_lock = PurePosixPath(str(pins["fixture_lock_relative_path"]))
    prep_marker = PurePosixPath(str(pins["fixture_prep_marker_relative_path"]))
    expected = (
        (input_manifest, ("fixture_input", INPUT_MANIFEST)),
        (label_manifest, ("fixture_label", LABEL_MANIFEST)),
        (fixture_lock, ("fixture_control", FIXTURE_LOCK)),
        (prep_marker, ("fixture_control", FIXTURE_PREP_MARKER)),
    )
    for value, parts in expected:
        if value.is_absolute() or value.parts != parts:
            _fail("pinned fixture-bundle relative path drift")
    return input_manifest, label_manifest, fixture_lock, prep_marker


def _bundle_input_root_before_marker(
    context: ProtocolContext, fixture_bundle_root: str
) -> Path:
    input_manifest = PurePosixPath(
        str(context.config["runtime_pins"]["fixture_input_manifest_relative_path"])
    )
    if input_manifest.is_absolute() or input_manifest.parts != (
        "fixture_input",
        INPUT_MANIFEST,
    ):
        _fail("pinned input-fixture relative path drift")
    # Deliberately construct only the input subpath.  Do not construct, stat,
    # list, or join the label/control subpaths before TARGET_ACCESS_STARTED.
    bundle = Path(fixture_bundle_root).expanduser().absolute()
    return bundle / input_manifest.parts[0]


def _bundle_paths_after_marker(
    context: ProtocolContext, fixture_bundle_root: str
) -> tuple[Path, Path, Path]:
    _, label_manifest, fixture_lock, prep_marker = _pinned_fixture_relative_parts(
        context
    )
    if fixture_lock.parts[0] != prep_marker.parts[0]:
        _fail("fixture control paths do not share one pinned root")
    bundle = Path(fixture_bundle_root).expanduser().absolute()
    return (
        bundle,
        bundle / label_manifest.parts[0],
        bundle / fixture_lock.parts[0],
    )


def _verify_bundle_root_separation(
    bundle_root: Path,
    *,
    input_root: Path,
    label_root: Path,
    control_root: Path,
) -> None:
    expected_children = {
        input_root: "fixture_input",
        label_root: "fixture_label",
        control_root: "fixture_control",
    }
    for path, name in expected_children.items():
        if path.parent != bundle_root or path.name != name:
            _fail("fixture bundle roots are not exact siblings")
    roots = [bundle_root, input_root, label_root, control_root]
    opened: list[AnchoredRoot] = []
    try:
        for path in roots:
            opened.append(AnchoredRoot.open(path))
        identities = [
            (os.fstat(root.descriptor).st_dev, os.fstat(root.descriptor).st_ino)
            for root in opened
        ]
        if len(set(identities)) != len(identities):
            _fail("fixture bundle roots share a resolved inode")
        bundle_device = identities[0][0]
        if any(device != bundle_device for device, _ in identities[1:]):
            _fail("fixture bundle root crosses a mount/device boundary")
    finally:
        for root in opened:
            root.close()


def _post_scoring_rehash(
    context: ProtocolContext,
    *,
    input_evidence: InputEvidence,
    phase_a: PhaseAEvidence,
    labels: LabelEvidence,
    fixture_control_root: str | Path,
    lifecycle: LifecycleEvidence,
) -> None:
    config_raw, _ = _secure_absolute_file(context.config_path)
    if _sha256_bytes(config_raw) != context.config_sha256:
        _fail("protocol config changed during independent verification")
    self_raw, _ = _secure_absolute_file(Path(__file__).resolve())
    pinned_self = context.config["runtime_pins"]["result_verifier_sha256"]
    if _sha256_bytes(self_raw) != pinned_self:
        _fail("independent verifier changed during execution")
    _verify_frozen_static_bindings(context)

    with AnchoredRoot.open(input_evidence.root_path) as root:
        raw, _ = root.read_file(INPUT_MANIFEST)
        if _sha256_bytes(raw) != input_evidence.manifest_sha256:
            _fail("input manifest TOCTOU mismatch")
        for opaque_id, record in input_evidence.records.items():
            relative = record.manifest["artifact"]["path"]
            raw, _ = root.read_file(relative)
            if _sha256_bytes(raw) != record.manifest["artifact"]["sha256"]:
                _fail(f"input artifact TOCTOU mismatch: {opaque_id}")
    with AnchoredRoot.open(phase_a.root_path) as root:
        raw, _ = root.read_file(PHASE_A_MANIFEST)
        if _sha256_bytes(raw) != phase_a.envelope_sha256:
            _fail("Phase-A manifest TOCTOU mismatch")
        for opaque_id, record in phase_a.records.items():
            manifest_record = record.manifest
            raw, _ = root.read_file(manifest_record["graph_artifact"])
            if _sha256_bytes(raw) != manifest_record["graph_artifact_sha256"]:
                _fail(f"Phase-A graph TOCTOU mismatch: {opaque_id}")
            for descriptor in manifest_record["renders"].values():
                raw, _ = root.read_file(descriptor["path"])
                if _sha256_bytes(raw) != descriptor["sha256"]:
                    _fail(f"Phase-A render TOCTOU mismatch: {opaque_id}")
    _rehash_phase_a_kaggle_attestation(phase_a)
    with AnchoredRoot.open(labels.root_path) as root:
        raw, _ = root.read_file(LABEL_MANIFEST)
        if _sha256_bytes(raw) != labels.manifest_sha256:
            _fail("label manifest TOCTOU mismatch")
        secret_descriptor = labels.manifest["master_secret"]
        raw, _ = root.read_file(MASTER_SECRET)
        if _sha256_bytes(raw) != secret_descriptor["sha256"]:
            _fail("master secret TOCTOU mismatch")
        for opaque_id, record in labels.records.items():
            raw, _ = root.read_file(record.manifest["artifact"]["path"])
            if _sha256_bytes(raw) != record.manifest["artifact"]["sha256"]:
                _fail(f"label artifact TOCTOU mismatch: {opaque_id}")
    with AnchoredRoot.open(fixture_control_root) as root:
        marker_raw, _ = root.read_file(FIXTURE_PREP_MARKER)
        lock_raw, _ = root.read_file(FIXTURE_LOCK)
        if _sha256_bytes(lock_raw) != context.config["runtime_pins"]["fixture_lock_sha256"]:
            _fail("fixture lock TOCTOU mismatch")
        lock_payload = _require_object(
            _parse_json(lock_raw, label=FIXTURE_LOCK, canonical_file=True),
            label=FIXTURE_LOCK,
        )
        if _sha256_bytes(marker_raw) != lock_payload.get("prep_marker_sha256"):
            _fail("fixture prep marker TOCTOU mismatch")
    lifecycle_after = verify_lifecycle(
        context, lifecycle_ledger=lifecycle.root_path, phase_a=phase_a
    )
    if (
        lifecycle_after.hashes != lifecycle.hashes
        or lifecycle_after.transition_hashes != lifecycle.transition_hashes
    ):
        _fail("lifecycle ledger TOCTOU mismatch")


def _post_phase_a_rehash(
    context: ProtocolContext,
    *,
    input_evidence: InputEvidence,
    phase_a: PhaseAEvidence,
    allow_unpinned_verifier: bool,
) -> None:
    config_raw, _ = _secure_absolute_file(context.config_path)
    if _sha256_bytes(config_raw) != context.config_sha256:
        _fail("protocol config changed during Phase-A verification")
    self_raw, _ = _secure_absolute_file(Path(__file__).resolve())
    self_sha = _sha256_bytes(self_raw)
    pinned_self = context.config["runtime_pins"].get("result_verifier_sha256")
    if pinned_self is None:
        if not allow_unpinned_verifier:
            _fail("verifier self pin disappeared")
    elif self_sha != pinned_self:
        _fail("verifier changed during Phase-A verification")
    _verify_frozen_static_bindings(context)
    with AnchoredRoot.open(input_evidence.root_path) as root:
        raw, _ = root.read_file(INPUT_MANIFEST)
        if _sha256_bytes(raw) != input_evidence.manifest_sha256:
            _fail("input manifest changed during Phase-A verification")
        for opaque_id, record in input_evidence.records.items():
            raw, _ = root.read_file(record.manifest["artifact"]["path"])
            if _sha256_bytes(raw) != record.manifest["artifact"]["sha256"]:
                _fail(f"input artifact changed during Phase-A verification: {opaque_id}")
    with AnchoredRoot.open(phase_a.root_path) as root:
        raw, _ = root.read_file(PHASE_A_MANIFEST)
        if _sha256_bytes(raw) != phase_a.envelope_sha256:
            _fail("Phase-A manifest changed during verification")
        for opaque_id, record in phase_a.records.items():
            raw, _ = root.read_file(record.manifest["graph_artifact"])
            if _sha256_bytes(raw) != record.manifest["graph_artifact_sha256"]:
                _fail(f"Phase-A graph changed during verification: {opaque_id}")
            for descriptor in record.manifest["renders"].values():
                raw, _ = root.read_file(descriptor["path"])
                if _sha256_bytes(raw) != descriptor["sha256"]:
                    _fail(f"Phase-A render changed during verification: {opaque_id}")
    _rehash_phase_a_kaggle_attestation(phase_a)


def _rehash_phase_a_kaggle_attestation(phase_a: PhaseAEvidence) -> None:
    attestation = phase_a.kaggle_attestation
    if attestation is None:
        return
    wrapper_raw, _ = _secure_absolute_file(attestation.wrapper_path)
    receipt_raw, _ = _secure_absolute_file(attestation.launch_receipt_path)
    if _sha256_bytes(wrapper_raw) != attestation.wrapper_sha256:
        _fail("Phase-A Kaggle wrapper TOCTOU mismatch")
    if _sha256_bytes(receipt_raw) != attestation.launch_receipt_sha256:
        _fail("Kaggle launch receipt TOCTOU mismatch")


def verify_phase_b(
    context: ProtocolContext,
    *,
    phase_b_root: str | Path,
    expected_report_sha256: str,
    phase_a: PhaseAEvidence,
    input_evidence: InputEvidence,
    lifecycle: LifecycleEvidence,
    fixture_bundle_root: str,
) -> PhaseBVerification:
    _verify_local_environment(context)
    expected_report_sha = _require_sha(
        expected_report_sha256, label="phase_b_report_sha256"
    )
    with AnchoredRoot.open(phase_b_root) as output_root:
        # This marker and the external LABEL_ACCESS claim are verified before a
        # Path or directory descriptor is ever constructed for the label root.
        _, marker_sha = _verify_target_access_marker(
            context,
            phase_b_root=output_root,
            phase_a=phase_a,
            lifecycle=lifecycle,
        )

        # Only now, after LABEL_ACCESS and TARGET_ACCESS_STARTED, materialize
        # the label/control subpaths from the single opaque bundle root and the
        # immutable pinned relative-path schema.  No caller-provided label path
        # or secret path exists in this verifier's interface.
        bundle_root, labels_root, fixture_control_root = _bundle_paths_after_marker(
            context, fixture_bundle_root
        )
        _verify_bundle_root_separation(
            bundle_root,
            input_root=input_evidence.root_path,
            label_root=labels_root,
            control_root=fixture_control_root,
        )

        labels = verify_label_fixture_after_marker(
            context,
            labels_root=str(labels_root),
            input_evidence=input_evidence,
            lifecycle=lifecycle,
        )
        verify_fixture_control(
            context,
            control_root=fixture_control_root,
            input_evidence=input_evidence,
            labels=labels,
            lifecycle=lifecycle,
        )

        # Complete every independent HMAC/panel/shuffle/input/truth
        # recomposition before the first recall, component, layout, or SSIM
        # computation.  This mirrors the frozen non-interleaving contract and
        # prevents early metrics from influencing whether later labels are
        # accepted.
        recompositions = {
            opaque_id: recompose_fixture(
                context,
                opaque_id=opaque_id,
                input_record=input_evidence.records[opaque_id],
                label_record=labels.records[opaque_id],
                secret=labels.secret,
                source_names=labels.source_names,
            )
            for opaque_id in sorted(phase_a.records)
        }
        if len(recompositions) != 64:
            _fail("independent fixture recomposition coverage drift")

        report_raw, _ = output_root.read_file(REPORT_NAME)
        if _sha256_bytes(report_raw) != expected_report_sha:
            _fail("Phase-B report out-of-band SHA-256 mismatch")
        report = _load_envelope_bytes(
            report_raw,
            expected_file_sha256=expected_report_sha,
            label=REPORT_NAME,
        )
        expected_report_keys = {
            "schema_version",
            "kind",
            "status",
            "config_sha256",
            "protocol_instance_id",
            "frozen_contract_sha256",
            "phase_a_envelope_sha256",
            "target_access_marker_sha256",
            "lifecycle_sha256",
            "fixture_input_manifest_sha256",
            "fixture_label_manifest_sha256",
            "runtime_asset_sha256",
            "records",
            "panel_summaries",
            "continuation_gate",
            "integrity",
            "target_assisted_translation_contributes_to_gate",
            "qap_weight_reopened",
            "safe_for_submission",
        }
        _require_exact_keys(report, expected_report_keys, label=REPORT_NAME)
        expected_header = {
            "schema_version": 1,
            "kind": "candidate_graph_oracle_ceiling_report",
            "config_sha256": context.config_sha256,
            "protocol_instance_id": EXPECTED_PROTOCOL_INSTANCE_ID,
            "frozen_contract_sha256": EXPECTED_FROZEN_CONTRACT_SHA256,
            "phase_a_envelope_sha256": phase_a.envelope_sha256,
            "target_access_marker_sha256": marker_sha,
            "lifecycle_sha256": dict(lifecycle.hashes),
            "fixture_input_manifest_sha256": input_evidence.manifest_sha256,
            "fixture_label_manifest_sha256": labels.manifest_sha256,
            "runtime_asset_sha256": {
                key: context.config["frozen_contract"]["assets"][key]["sha256"]
                for key in ("denoiser", "hbt")
            },
            "target_assisted_translation_contributes_to_gate": False,
            "qap_weight_reopened": False,
            "safe_for_submission": False,
        }
        for key, expected in expected_header.items():
            if report.get(key) != expected:
                _fail(f"Phase-B report invariant drift: {key}")
        record_values = report.get("records")
        if not isinstance(record_values, list) or len(record_values) != 64:
            _fail("Phase-B report must contain exactly 64 records")
        ids = [value.get("opaque_id") if isinstance(value, dict) else None for value in record_values]
        if ids != sorted(phase_a.records):
            _fail("Phase-B report record order/coverage drift")
        verified_records: list[dict[str, Any]] = []
        layout_names: set[str] = set()
        render_names: set[str] = set()
        with AnchoredRoot.open(phase_a.root_path) as phase_a_root:
            for index, record_value in enumerate(record_values):
                opaque_id = str(record_value["opaque_id"])
                phase_arrays, phase_w4_render = _reload_phase_a_scoring_arrays(
                    phase_a_root, phase_a.records[opaque_id]
                )
                verified, layout_name, render_name = _verify_record_report(
                    record_value,
                    index=index,
                    output_root=output_root,
                    phase_a_arrays=phase_arrays,
                    phase_a_w4_render=phase_w4_render,
                    recomposed=recompositions[opaque_id],
                )
                verified_records.append(verified)
                if layout_name in layout_names or render_name in render_names:
                    _fail("duplicate Phase-B artifact name")
                layout_names.add(layout_name)
                render_names.add(render_name)

        summaries = _panel_summaries(verified_records)
        _compare_json_numeric(
            report["panel_summaries"], summaries, label="report.panel_summaries"
        )
        gate = _independent_gate(summaries)
        _compare_json_numeric(
            report["continuation_gate"], gate, label="report.continuation_gate"
        )
        expected_status = (
            "continue"
            if gate["continue_to_cycle_factor_synchronizer"]
            else "stop_or_pivot"
        )
        if report["status"] != expected_status:
            _fail("Phase-B report status differs from independent gate")
        expected_integrity = {
            "fixture_record_count": 64,
            "records_per_panel": 32,
            "candidate_graph_count": 64,
            "valid_baseline_permutation_count": 64,
            "valid_oracle_packer_permutation_count": 64,
            "artifact_hash_or_toctou_failures": 0,
            "opaque_id_join_errors": 0,
            "post_score_toctou_verified": True,
            "all_64_fixtures_recomposed_before_first_metric": True,
        }
        if report["integrity"] != expected_integrity:
            _fail("Phase-B report integrity block drift")

        top_files = {TARGET_MARKER, REPORT_NAME}
        output_root.assert_exact_tree(
            top_files=top_files,
            directories={"artifacts": layout_names, "renders": render_names},
        )
        # Rehash all Phase-B artifacts after metric/gate recomputation.
        report_after, _ = output_root.read_file(REPORT_NAME)
        marker_after, _ = output_root.read_file(TARGET_MARKER)
        if _sha256_bytes(report_after) != expected_report_sha:
            _fail("Phase-B report TOCTOU mismatch")
        if _sha256_bytes(marker_after) != marker_sha:
            _fail("target-access marker TOCTOU mismatch")
        for record in record_values:
            for descriptor in record["artifacts"].values():
                raw, _ = output_root.read_file(descriptor["path"])
                if _sha256_bytes(raw) != descriptor["sha256"]:
                    _fail("Phase-B record artifact TOCTOU mismatch")
        _post_scoring_rehash(
            context,
            input_evidence=input_evidence,
            phase_a=phase_a,
            labels=labels,
            fixture_control_root=fixture_control_root,
            lifecycle=lifecycle,
        )
        with AnchoredRoot.open(bundle_root) as bundle:
            if bundle.list_names() != {
                "fixture_input",
                "fixture_label",
                "fixture_control",
            }:
                _fail("fixture bundle exact top-level tree drift")
    return PhaseBVerification(
        expected_report_sha,
        expected_status,
        bool(gate["continue_to_cycle_factor_synchronizer"]),
    )


def verify_phase_b_runner_attestation(
    context: ProtocolContext,
    *,
    attestation_path: str | Path,
    expected_attestation_sha256: str,
    verification: PhaseBVerification,
    phase_a: PhaseAEvidence,
    input_evidence: InputEvidence,
    lifecycle: LifecycleEvidence,
    phase_b_root: str | Path,
    fixture_bundle_root: str | Path,
) -> PhaseBRunnerAttestation:
    """Verify the canonical stdout envelope emitted outside the evaluator tree."""

    expected_sha = _require_sha(
        expected_attestation_sha256, label="phase_b_runner_attestation_sha256"
    )
    absolute = Path(attestation_path).expanduser().absolute()
    output_root = Path(phase_b_root).expanduser().absolute()
    if absolute == output_root or output_root in absolute.parents:
        _fail("Phase-B runner attestation may not be inside evaluator output tree")
    raw, _ = _secure_absolute_file(absolute)
    payload = _load_envelope_bytes(
        raw, expected_file_sha256=expected_sha, label="Phase-B runner attestation"
    )
    expected_keys = {
        "schema_version",
        "kind",
        "status",
        "safe_for_submission",
        "process_id",
        "config_sha256",
        "runner_sha256",
        "evaluator_sha256",
        "tests_sha256",
        "environment",
        "sandbox",
        "phase_a_envelope_sha256",
        "fixture_input_manifest_sha256",
        "filesystem_bindings",
        "report_path",
        "report_sha256",
        "report_payload_sha256",
        "preflight_output_sha256",
        "preflight_output_bytes",
        "evaluator_output_sha256",
        "evaluator_output_bytes",
        "evaluator_output_tree_mutated_by_runner",
    }
    _require_exact_keys(payload, expected_keys, label="Phase-B runner attestation")
    pins = context.config["runtime_pins"]
    expected_header = {
        "schema_version": 2,
        "kind": "candidate_graph_oracle_phase_b_runner_attestation",
        "status": verification.status,
        "safe_for_submission": False,
        "config_sha256": context.config_sha256,
        "runner_sha256": pins["phase_b_runner_sha256"],
        "evaluator_sha256": pins["evaluator_sha256"],
        "tests_sha256": pins["tests_sha256"],
        "phase_a_envelope_sha256": phase_a.envelope_sha256,
        "fixture_input_manifest_sha256": input_evidence.manifest_sha256,
        "report_path": REPORT_NAME,
        "report_sha256": verification.report_sha256,
        "evaluator_output_tree_mutated_by_runner": False,
    }
    for key, expected_value in expected_header.items():
        if payload.get(key) != expected_value:
            _fail(f"Phase-B runner attestation crosslink mismatch: {key}")
    _require_exact_int(payload["process_id"], label="phase_b.process_id", minimum=1)
    for prefix in ("preflight_output", "evaluator_output"):
        _require_sha(payload[f"{prefix}_sha256"], label=f"{prefix}_sha256")
        _require_exact_int(payload[f"{prefix}_bytes"], label=f"{prefix}_bytes", minimum=1)

    report_path = output_root / REPORT_NAME
    report_raw, _ = _secure_absolute_file(report_path)
    if _sha256_bytes(report_raw) != verification.report_sha256:
        _fail("Phase-B report changed before runner-attestation verification")
    report_payload = _load_envelope_bytes(
        report_raw,
        expected_file_sha256=verification.report_sha256,
        label="Phase-B report for runner attestation",
    )
    if payload["report_payload_sha256"] != _sha256_bytes(
        _canonical_object_bytes(report_payload)
    ):
        _fail("Phase-B runner report payload hash crosslink mismatch")

    environment = _require_object(payload["environment"], label="phase_b.environment")
    _require_exact_keys(
        environment,
        {"lock_sha256", "platform", "python", "packages"},
        label="phase_b.environment",
    )
    with AnchoredRoot.open(context.repository) as root:
        lock_raw, _ = root.read_file(pins["environment_lock_path"])
    if _sha256_bytes(lock_raw) != pins["environment_lock_sha256"]:
        _fail("Phase-B environment lock changed")
    lock = _require_object(
        _parse_json(lock_raw, label="environment lock", canonical_file=False),
        label="environment lock",
    )["fixture_preparation_and_phase_b"]
    expected_environment = {
        "lock_sha256": pins["environment_lock_sha256"],
        "platform": lock["platform"],
        "python": lock["python"],
        "packages": lock["packages"],
    }
    if environment != expected_environment:
        _fail("Phase-B runner environment attestation drift")

    sandbox = _require_object(payload["sandbox"], label="phase_b.sandbox")
    _require_exact_keys(
        sandbox,
        {
            "backend",
            "profile_sha256",
            "default_deny",
            "network_policy",
            "config_readable_and_sha256_verified",
            "fresh_output_write_probe",
            "denial_probes",
        },
        label="phase_b.sandbox",
    )
    if (
        sandbox["backend"] != "/usr/bin/sandbox-exec"
        or sandbox["default_deny"] is not True
        or sandbox["network_policy"] != "deny network*"
        or sandbox["config_readable_and_sha256_verified"] is not True
        or sandbox["fresh_output_write_probe"] is not True
    ):
        _fail("Phase-B sandbox policy attestation drift")
    _require_sha(sandbox["profile_sha256"], label="sandbox.profile_sha256")
    probes = sandbox["denial_probes"]
    expected_labels = [
        "repo_puzzle_train_read",
        "repo_puzzle_train_targets_read",
        "phase_a_write",
        "network_outbound",
    ]
    if not isinstance(probes, list) or len(probes) != len(expected_labels):
        _fail("Phase-B sandbox denial-probe coverage drift")
    for index, (probe_value, expected_label) in enumerate(
        zip(probes, expected_labels, strict=True)
    ):
        probe = _require_object(probe_value, label=f"denial_probes[{index}]")
        expected_probe_keys = {"label", "denied", "errno", "errno_name"}
        if expected_label == "network_outbound":
            expected_probe_keys.add("denied_at")
        _require_exact_keys(
            probe, expected_probe_keys, label=f"denial_probes[{index}]"
        )
        probe_errno = _require_exact_int(
            probe["errno"], label=f"denial_probes[{index}].errno", minimum=1
        )
        if (
            probe["label"] != expected_label
            or probe["denied"] is not True
            or probe_errno not in {1, 13}
            or probe["errno_name"]
            != {1: "EPERM", 13: "EACCES"}[probe_errno]
        ):
            _fail(f"Phase-B sandbox denial probe failed: {expected_label}")
        if expected_label == "network_outbound" and probe["denied_at"] not in {
            "socket_create",
            "connect",
        }:
            _fail("Phase-B network denial probe stage drift")

    bindings = _require_object(
        payload["filesystem_bindings"], label="phase_b.filesystem_bindings"
    )
    expected_bindings = {
        "phase_a_root": str(phase_a.root_path.absolute()),
        "phase_a_artifact_envelope_sha256": phase_a.envelope_sha256,
        "fixture_bundle_root": str(Path(fixture_bundle_root).expanduser().absolute()),
        "fixture_input_root": str(input_evidence.root_path.absolute()),
        "fixture_input_manifest": str(
            (input_evidence.root_path / INPUT_MANIFEST).absolute()
        ),
        "fixture_input_manifest_sha256": input_evidence.manifest_sha256,
        "lifecycle_ledger_root": str(lifecycle.root_path.absolute()),
        "output_root": str(output_root),
    }
    if bindings != expected_bindings:
        _fail("Phase-B runner filesystem binding drift")
    attestation_after, _ = _secure_absolute_file(absolute)
    if _sha256_bytes(attestation_after) != expected_sha:
        _fail("Phase-B runner attestation changed during verification")
    return PhaseBRunnerAttestation(absolute, expected_sha, payload)


def _phase_a_from_args(
    context: ProtocolContext, args: argparse.Namespace
) -> tuple[InputEvidence, PhaseAEvidence]:
    if args.action == "phase-a":
        if not args.fixture_root or not args.fixture_manifest_sha256:
            _fail("phase-a verifier requires input fixture root and manifest hash")
        fixture_root = args.fixture_root
        fixture_manifest_sha256 = args.fixture_manifest_sha256
        if args.fixture_bundle_root is not None:
            _fail("phase-a verifier refuses fixture-bundle-root")
    else:
        if not args.fixture_bundle_root:
            _fail("phase-b verifier requires one opaque fixture-bundle-root")
        if args.fixture_root is not None or args.fixture_manifest_sha256 is not None:
            _fail("phase-b verifier refuses caller-provided fixture subpaths/hashes")
        fixture_root = str(
            _bundle_input_root_before_marker(context, args.fixture_bundle_root)
        )
        fixture_manifest_sha256 = context.config["runtime_pins"][
            "fixture_input_manifest_sha256"
        ]
    fixture_root = str(
        _guard_phase_a_read_path(fixture_root, label="Phase-A fixture input root")
    )
    phase_a_root = str(
        _guard_phase_a_read_path(args.phase_a_dir, label="Phase-A output root")
    )
    input_evidence = verify_input_fixture(
        context,
        fixture_root=fixture_root,
        expected_manifest_sha256=fixture_manifest_sha256,
    )
    phase_a = verify_phase_a(
        context,
        phase_a_root=phase_a_root,
        expected_envelope_sha256=args.phase_a_envelope_sha256,
        shard_anchors=args.phase_a_shard_envelope_sha256,
        input_evidence=input_evidence,
    )
    attestation_values = {
        "phase_a_wrapper": args.phase_a_wrapper,
        "phase_a_wrapper_sha256": args.phase_a_wrapper_sha256,
        "kaggle_launch_receipt": args.kaggle_launch_receipt,
        "kaggle_launch_receipt_sha256": args.kaggle_launch_receipt_sha256,
    }
    if args.allow_unpinned_verifier and not any(attestation_values.values()):
        return input_evidence, phase_a
    missing = [key for key, value in attestation_values.items() if not value]
    if missing:
        _fail(f"Phase-A Kaggle attestation arguments are incomplete: {missing}")
    phase_a.kaggle_attestation = verify_phase_a_kaggle_attestation(
        context,
        phase_a=phase_a,
        input_evidence=input_evidence,
        wrapper_path=str(args.phase_a_wrapper),
        expected_wrapper_sha256=str(args.phase_a_wrapper_sha256),
        launch_receipt_path=str(args.kaggle_launch_receipt),
        expected_launch_receipt_sha256=str(args.kaggle_launch_receipt_sha256),
    )
    return input_evidence, phase_a


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("phase-a", "phase-b"), required=True)
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs/candidate_graph_oracle_ceiling_v4.json")
    )
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--fixture-root")
    parser.add_argument("--fixture-manifest-sha256")
    parser.add_argument("--fixture-bundle-root")
    parser.add_argument("--phase-a-dir", required=True)
    parser.add_argument("--phase-a-envelope-sha256", required=True)
    parser.add_argument("--phase-a-wrapper")
    parser.add_argument("--phase-a-wrapper-sha256")
    parser.add_argument("--kaggle-launch-receipt")
    parser.add_argument("--kaggle-launch-receipt-sha256")
    parser.add_argument(
        "--phase-a-shard-envelope-sha256",
        action="append",
        required=True,
        help="Pass exactly twice, in rank-0 then rank-1 order.",
    )
    parser.add_argument("--lifecycle-ledger")
    parser.add_argument("--phase-b-dir")
    parser.add_argument("--phase-b-report-sha256")
    parser.add_argument("--phase-b-runner-attestation")
    parser.add_argument("--phase-b-runner-attestation-sha256")
    parser.add_argument(
        "--allow-unpinned-verifier",
        action="store_true",
        help="Only for pre-pin synthetic tests; forbidden for an accepted production result.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    context = _load_protocol(
        args.config,
        expected_config_sha256=args.config_sha256,
        allow_unpinned_verifier=args.allow_unpinned_verifier,
    )
    if args.action == "phase-b" and args.allow_unpinned_verifier:
        _fail("Phase-B verification may not use an unpinned verifier")
    input_evidence, phase_a = _phase_a_from_args(context, args)
    if args.action == "phase-a":
        if not args.lifecycle_ledger:
            _fail("phase-a verifier requires the exact Phase-A-only lifecycle ledger")
        forbidden = (
            args.phase_b_dir,
            args.phase_b_report_sha256,
            args.phase_b_runner_attestation,
            args.phase_b_runner_attestation_sha256,
            args.fixture_bundle_root,
        )
        if any(value is not None for value in forbidden):
            _fail("phase-a verifier refuses every lifecycle/label/Phase-B argument")
        phase_a_lifecycle = verify_phase_a_lifecycle(
            context,
            lifecycle_ledger=str(args.lifecycle_ledger),
            phase_a=phase_a,
        )
        _post_phase_a_rehash(
            context,
            input_evidence=input_evidence,
            phase_a=phase_a,
            allow_unpinned_verifier=bool(args.allow_unpinned_verifier),
        )
        phase_a_lifecycle_after = verify_phase_a_lifecycle(
            context,
            lifecycle_ledger=str(args.lifecycle_ledger),
            phase_a=phase_a,
        )
        if (
            phase_a_lifecycle_after.hashes != phase_a_lifecycle.hashes
            or phase_a_lifecycle_after.transition_hashes
            != phase_a_lifecycle.transition_hashes
            or phase_a_lifecycle_after.code_config_sha256
            != phase_a_lifecycle.code_config_sha256
        ):
            _fail("Phase-A lifecycle ledger changed during verification")
        result = {
            "schema_version": 1,
            "kind": "candidate_graph_oracle_phase_a_independent_verification",
            "status": (
                "verified_pre_pin_test_only"
                if args.allow_unpinned_verifier
                else "verified_input_only"
            ),
            "config_sha256": context.config_sha256,
            "phase_a_envelope_sha256": phase_a.envelope_sha256,
            "phase_a_shard_envelope_sha256s": list(phase_a.shard_anchors),
            "phase_a_wrapper_sha256": (
                phase_a.kaggle_attestation.wrapper_sha256
                if phase_a.kaggle_attestation is not None
                else None
            ),
            "kaggle_launch_receipt_sha256": (
                phase_a.kaggle_attestation.launch_receipt_sha256
                if phase_a.kaggle_attestation is not None
                else None
            ),
            "record_count": len(phase_a.records),
            "phase_a_lifecycle_sha256": phase_a_lifecycle.hashes["PHASE_A"],
            "lifecycle_terminal_state": "PHASE_A",
            "labels_constructed_or_opened": False,
            "pre_pin_test_only": bool(args.allow_unpinned_verifier),
            "safe_for_submission": False,
        }
    else:
        required = {
            "lifecycle_ledger": args.lifecycle_ledger,
            "phase_b_dir": args.phase_b_dir,
            "phase_b_report_sha256": args.phase_b_report_sha256,
            "phase_b_runner_attestation": args.phase_b_runner_attestation,
            "phase_b_runner_attestation_sha256": (
                args.phase_b_runner_attestation_sha256
            ),
            "fixture_bundle_root": args.fixture_bundle_root,
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            _fail(f"phase-b verifier missing arguments: {missing}")
        lifecycle = verify_lifecycle(
            context,
            lifecycle_ledger=str(args.lifecycle_ledger),
            phase_a=phase_a,
        )
        verification = verify_phase_b(
            context,
            phase_b_root=str(args.phase_b_dir),
            expected_report_sha256=str(args.phase_b_report_sha256),
            phase_a=phase_a,
            input_evidence=input_evidence,
            lifecycle=lifecycle,
            fixture_bundle_root=str(args.fixture_bundle_root),
        )
        runner_attestation = verify_phase_b_runner_attestation(
            context,
            attestation_path=str(args.phase_b_runner_attestation),
            expected_attestation_sha256=str(
                args.phase_b_runner_attestation_sha256
            ),
            verification=verification,
            phase_a=phase_a,
            input_evidence=input_evidence,
            lifecycle=lifecycle,
            phase_b_root=str(args.phase_b_dir),
            fixture_bundle_root=str(args.fixture_bundle_root),
        )
        result = {
            "schema_version": 1,
            "kind": "candidate_graph_oracle_phase_b_independent_verification",
            "status": "verified",
            "config_sha256": context.config_sha256,
            "phase_a_envelope_sha256": phase_a.envelope_sha256,
            "phase_a_wrapper_sha256": phase_a.kaggle_attestation.wrapper_sha256,
            "kaggle_launch_receipt_sha256": (
                phase_a.kaggle_attestation.launch_receipt_sha256
            ),
            "phase_b_report_sha256": verification.report_sha256,
            "phase_b_runner_attestation_sha256": runner_attestation.sha256,
            "report_status": verification.status,
            "continue_to_cycle_factor_synchronizer": verification.continuation_gate_passed,
            "target_assisted_translation_contributes_to_gate": False,
            "safe_for_submission": False,
        }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
