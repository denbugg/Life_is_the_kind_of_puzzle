"""Build and evaluate the immutable PAZZLE I11/I21 confirmation gate.

The gate deliberately has two layers:

* an immutable, self-hashed core containing the exact clean target, shuffled
  corrupted tile bytes, and exact synthetic permutation for every scene;
* a disposable score cache whose contract binds every derived array to the
  immutable scene, scorer checkpoints, and scoring code.

``freeze`` refuses to operate without a complete source-group manifest.  A
filename split is not accepted as evidence of source disjointness.  ``evaluate``
has no model-selection sweep: the raw input, corrected I11 baseline, and the
precommitted I21 blend are fixed below.  ``verify`` is CPU-only.

Source-group manifest schema::

    {
      "schema": "pazzle-source-groups-v1",
      "complete": true,
      "method": "description of source/near-duplicate verification",
      "images": {
        "img_000000.png": {"source_group": "canonical-source-id"},
        ... every target filename exactly once ...
      }
    }

The complete ``schema_version=1`` output of ``src/build_source_groups.py`` is
also accepted.  Its ``files[name].source_group`` mapping is trusted only after
``files``, ``groups``, ``stats``, and ``split`` prove exact full coverage.  The
builder's ``schema_version=2`` exclusion contract is accepted only when its
declared exclusions and deterministic selection can be reproduced exactly;
``freeze`` additionally requires the same validation IDs in ``--tuning-ranges``.

Examples::

    python src/eval_frozen_end_to_end_gate.py freeze \
      --source-groups artifacts/frozen_gate/source_groups_v1.json \
      --gate-dir E:/pazzle_work/gates/frozen_v1
    python src/eval_frozen_end_to_end_gate.py evaluate \
      --gate-dir E:/pazzle_work/gates/frozen_v1 --device cuda
    python src/eval_frozen_end_to_end_gate.py verify \
      --gate-dir E:/pazzle_work/gates/frozen_v1
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import random
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np


GRID = 24
FRAGMENT_SIZE = 20
IMAGE_SIZE = GRID * FRAGMENT_SIZE
NFRAG = GRID * GRID
NUM_DIRECTIONS = 4
DIRECTION_ORDER = ("up", "down", "left", "right")

GATE_SCHEMA = "pazzle-frozen-end-to-end-gate-v1"
SCENE_SCHEMA = "pazzle-frozen-scene-v1"
SOURCE_GROUP_SCHEMA = "pazzle-source-groups-v1"
BUILDER_SOURCE_GROUP_SCHEMAS = (1, 2)
BUILDER_V2_SELECTION_CONTRACT: dict[str, Any] = {
    "index_space": "zero-based IDs within the validation split",
    "exclusion_syntax": "comma-separated validation IDs or start:count ranges (start inclusive)",
    "eligibility_order": [
        "validation ID is at least candidate_val_min",
        "validation ID is not in excluded_val_ids",
        "source group is absent from train and known tuning validation IDs",
    ],
    "ranking": "ascending sha256(seed\\0source_group\\0name), then name; first item per source group",
}
SCORE_CACHE_SCHEMA = "pazzle-frozen-score-cache-v1"
SCORE_CACHE_INDEX_SCHEMA = "pazzle-frozen-score-cache-index-v1"

MINIMUM_GATE_SCENES = 24
DEFAULT_GATE_SEED = 20_260_806
# Validation IDs 0..99 have already participated in checkpoint selection,
# diagnostics, or protocol development.  The checked-in source manifest
# predeclares its confirmation candidates from ID 100 onward.
DEFAULT_TUNING_RANGES = "0:100"
PAIR_BATCH = 4096
BOOTSTRAP_SAMPLES = 10_000

# These are the experiment, not defaults to tune.  Verification rejects a gate
# whose manifest changes them.
FIXED_ARMS: dict[str, dict[str, Any]] = {
    "raw_input": {
        "board": "input_tile_order",
        "edge_r1": None,
        "restoration": {"method": "opencv_fast_nlm_colored", "h": 10, "h_color": 10,
                        "template_window": 7, "search_window": 21},
    },
    "i11": {
        "candidate_k_per_encoder": 64,
        "score": "candidate_ranker_raw_logits",
        "dense_conversion": "eval_seeded_qap.dense_rd",
        "solver": "corrected_buddies",
        "max_edges": 512,
        "min_margin": 0.0,
        "repair_passes": 0,
    },
    "i21": {
        "candidate_k_per_encoder": 64,
        "score": "row_z(raw)+1.25*row_z(spatial)",
        "alpha": 1.25,
        "dense_conversion": "eval_seeded_qap.dense_rd",
        "solver": "corrected_buddies",
        "max_edges": 512,
        "min_margin": 0.0,
        "repair_passes": 0,
    },
}


class FrozenGateError(RuntimeError):
    """Base class for gate contract failures."""


class SourceGroupManifestError(FrozenGateError):
    """The source identity manifest is absent, incomplete, or inconsistent."""


class IntegrityError(FrozenGateError):
    """A frozen artifact does not match its recorded digest or schema."""


class CacheContractError(FrozenGateError):
    """A derived score cache does not belong to this exact gate/model contract."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _safe_name(name: str) -> str:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise FrozenGateError(f"target filename must be a basename, got {name!r}")
    return name


def _atomic_write_bytes(path: Path, content: bytes, *, require_absent: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if require_absent and path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if require_absent and path.exists():
            raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)
    return stream.getvalue()


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Return a byte-stable compressed NPZ (NumPy's writer embeds ZIP time)."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for key, value in arrays.items():
            if not key or "/" in key or "\\" in key:
                raise ValueError(f"invalid NPZ key {key!r}")
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(np.asarray(value)), compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)
    return output.getvalue()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as stored:
            return {key: stored[key] for key in stored.files}
    except Exception as exc:  # CRC, malformed NPY, and object arrays are all integrity failures.
        raise IntegrityError(f"could not read deterministic cache {path}: {exc}") from exc


def _parse_ranges(text: str, length: int) -> list[int]:
    selected: list[int] = []
    if not text.strip():
        return selected
    for raw_item in text.split(","):
        item = raw_item.strip()
        try:
            start_text, count_text = item.split(":", 1)
            start, count = int(start_text), int(count_text)
        except Exception as exc:
            raise FrozenGateError(
                f"invalid tuning range {item!r}; expected comma-separated start:count"
            ) from exc
        if start < 0 or count < 0 or start + count > length:
            raise FrozenGateError(f"tuning range {item!r} is outside validation length {length}")
        selected.extend(range(start, start + count))
    if len(selected) != len(set(selected)):
        raise FrozenGateError("tuning ranges overlap")
    return sorted(selected)


def _validate_builder_v2_selection(
    payload: Mapping[str, Any], target_names: Sequence[str], groups: Mapping[str, str]
) -> None:
    """Reproduce every schema-v2 split decision from its declared inputs."""
    split = payload["split"]
    train_count = split["train_count"]
    val_count = split["val_count"]

    excluded = split.get("excluded_val_ids")
    if (
        not isinstance(excluded, list)
        or not excluded
        or any(type(value) is not int for value in excluded)
    ):
        raise SourceGroupManifestError(
            "build_source_groups schema_version=2 requires non-empty integer split.excluded_val_ids"
        )
    if excluded != sorted(set(excluded)):
        raise SourceGroupManifestError("split.excluded_val_ids must be sorted and unique")
    if any(value < 0 or value >= val_count for value in excluded):
        raise SourceGroupManifestError("split.excluded_val_ids must be validation-local and in bounds")
    if split.get("selection_contract") != BUILDER_V2_SELECTION_CONTRACT:
        raise SourceGroupManifestError(
            "build_source_groups schema_version=2 requires the exact selection_contract"
        )

    known_tune = split.get("known_tune_val_ids")
    if (
        not isinstance(known_tune, list)
        or len(known_tune) != 2
        or any(type(value) is not int for value in known_tune)
        or known_tune[0] != 0
        or known_tune[1] < 0
        or known_tune[1] >= val_count
    ):
        raise SourceGroupManifestError(
            "schema_version=2 split.known_tune_val_ids must be [0, inclusive_end] in bounds"
        )
    candidate_val_min = split.get("candidate_val_min")
    if type(candidate_val_min) is not int or not 0 <= candidate_val_min <= val_count:
        raise SourceGroupManifestError("schema_version=2 candidate_val_min is invalid")
    seed = split.get("selection_seed")
    if not isinstance(seed, str) or not seed:
        raise SourceGroupManifestError("schema_version=2 selection_seed must be a non-empty string")

    eligible_declared = split.get("eligible_confirmation")
    selected_declared = split.get("selected_confirmation")
    for label, declared in (
        ("eligible_confirmation", eligible_declared),
        ("selected_confirmation", selected_declared),
    ):
        if (
            not isinstance(declared, list)
            or not all(isinstance(name, str) for name in declared)
            or len(declared) != len(set(declared))
        ):
            raise SourceGroupManifestError(f"schema_version=2 split.{label} must be a unique filename list")
    if not selected_declared:
        raise SourceGroupManifestError("schema_version=2 selected_confirmation must not be empty")

    train_names = list(target_names[:train_count])
    validation_names = list(target_names[train_count:])
    validation_id = {name: index for index, name in enumerate(validation_names)}
    unknown_selected = sorted(set(selected_declared) - set(validation_names))
    if unknown_selected:
        raise SourceGroupManifestError(
            f"schema_version=2 selected_confirmation contains non-validation names: {unknown_selected[:3]}"
        )
    excluded_set = set(excluded)
    selected_exclusions = sorted(validation_id[name] for name in selected_declared if validation_id[name] in excluded_set)
    if selected_exclusions:
        raise SourceGroupManifestError(
            "schema_version=2 exclusions occur in selected_confirmation: "
            f"{selected_exclusions[:3]}"
        )
    selected_below_candidate = sorted(
        validation_id[name] for name in selected_declared if validation_id[name] < candidate_val_min
    )
    if selected_below_candidate:
        raise SourceGroupManifestError(
            "schema_version=2 selected_confirmation is outside candidate_val_min range: "
            f"{selected_below_candidate[:3]}"
        )

    tune_names = set(validation_names[known_tune[0] : known_tune[1] + 1])
    forbidden_names = set(train_names) | tune_names
    forbidden_groups = {groups[name] for name in forbidden_names}
    eligible: list[str] = []
    for val_id, name in enumerate(validation_names):
        if val_id < candidate_val_min or val_id in excluded_set:
            continue
        if groups[name] in forbidden_groups:
            continue
        eligible.append(name)
    if eligible_declared != eligible:
        raise SourceGroupManifestError(
            "schema_version=2 eligible_confirmation is inconsistent with exclusions/candidate range"
        )

    ranked = sorted(
        eligible,
        key=lambda name: (
            _sha256_bytes(f"{seed}\0{groups[name]}\0{name}".encode("utf-8")),
            name,
        ),
    )
    selected: list[str] = []
    used_groups: set[str] = set()
    for name in ranked:
        group = groups[name]
        if group in used_groups:
            continue
        selected.append(name)
        used_groups.add(group)
        if len(selected) == len(selected_declared):
            break
    if selected != selected_declared:
        raise SourceGroupManifestError(
            "schema_version=2 selected_confirmation is inconsistent with selection seed/count"
        )


def _enforce_builder_v2_freeze_contract(
    payload: Mapping[str, Any], *, validation_count: int, tuning_ranges: str, number: int, gate_seed: int
) -> None:
    if payload.get("schema_version") != 2:
        return
    split = payload["split"]
    if validation_count != split["val_count"]:
        raise SourceGroupManifestError(
            "freeze validation_count differs from schema_version=2 source manifest"
        )
    if str(gate_seed) != split["selection_seed"]:
        raise SourceGroupManifestError("freeze gate_seed differs from schema_version=2 selection_seed")
    if number != len(split["selected_confirmation"]):
        raise SourceGroupManifestError(
            "freeze scene count differs from schema_version=2 selected_confirmation count"
        )
    actual_tuning = _parse_ranges(tuning_ranges, validation_count)
    expected_tuning = sorted(
        set(range(split["candidate_val_min"])) | set(split["excluded_val_ids"])
    )
    if actual_tuning != expected_tuning:
        raise SourceGroupManifestError(
            "freeze tuning ranges do not exactly match schema_version=2 candidate prefix and exclusions"
        )


def load_source_groups(path: Path, target_names: Sequence[str]) -> tuple[dict[str, str], dict[str, Any], bytes]:
    """Load a complete explicit source identity manifest, failing closed."""
    if not path.is_file():
        raise SourceGroupManifestError(
            "a complete source-group manifest is required; no filename-only fallback is allowed: "
            f"{path}"
        )
    raw_bytes = path.read_bytes()
    try:
        payload = json.loads(raw_bytes)
    except Exception as exc:
        raise SourceGroupManifestError(f"source-group manifest is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SourceGroupManifestError("source-group manifest root must be an object")

    # Accept the documented portable schema and the checked-in manifest emitted
    # by build_source_groups.py.  The latter has no `complete=true` flag, so its
    # stronger structural proof (all files + an exact group partition + split
    # cardinalities) is mandatory; merely having a `files` key is insufficient.
    portable = payload.get("schema") == SOURCE_GROUP_SCHEMA
    builder_version = payload.get("schema_version")
    builder = (
        type(builder_version) is int
        and builder_version in BUILDER_SOURCE_GROUP_SCHEMAS
        and "files" in payload
        and "groups" in payload
    )
    if not portable and not builder:
        raise SourceGroupManifestError(
            f"source-group manifest must use schema={SOURCE_GROUP_SCHEMA!r} or "
            f"build_source_groups schema_version in {BUILDER_SOURCE_GROUP_SCHEMAS}"
        )
    if portable:
        if payload.get("complete") is not True:
            raise SourceGroupManifestError("source-group manifest must explicitly declare complete=true")
        if not isinstance(payload.get("method"), str) or not payload["method"].strip():
            raise SourceGroupManifestError("source-group manifest must describe its verification method")
        images = payload.get("images")
        if not isinstance(images, dict):
            raise SourceGroupManifestError("source-group manifest 'images' must be an object")
        normalized_schema = SOURCE_GROUP_SCHEMA
        normalized_method = payload["method"].strip()
    else:
        images = payload.get("files")
        indexed_groups = payload.get("groups")
        stats = payload.get("stats")
        split = payload.get("split")
        algorithms = payload.get("algorithms")
        if not all(isinstance(value, dict) for value in (images, indexed_groups, stats, split, algorithms)):
            raise SourceGroupManifestError(
                "build_source_groups manifest requires object fields files/groups/stats/split/algorithms"
            )
        if stats.get("files") != len(target_names):
            raise SourceGroupManifestError("build_source_groups stats.files does not prove full dataset coverage")
        train_count, val_count = split.get("train_count"), split.get("val_count")
        if (
            not isinstance(train_count, int)
            or not isinstance(val_count, int)
            or train_count + val_count != len(target_names)
        ):
            raise SourceGroupManifestError("build_source_groups split cardinalities do not cover the dataset")
        normalized_schema = f"build-source-groups-v{builder_version}"
        normalized_method = json.dumps(algorithms, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    expected = set(target_names)
    supplied = set(images)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing {len(missing)} names (first: {missing[:3]})")
        if extra:
            detail.append(f"contains {len(extra)} unknown names (first: {extra[:3]})")
        raise SourceGroupManifestError("source-group manifest is not complete for this dataset: " + "; ".join(detail))

    groups: dict[str, str] = {}
    for name in target_names:
        _safe_name(name)
        row = images[name]
        if not isinstance(row, dict):
            raise SourceGroupManifestError(f"images[{name!r}] must be an object")
        group = row.get("source_group")
        if not isinstance(group, str) or not group.strip():
            raise SourceGroupManifestError(f"images[{name!r}] needs a non-empty source_group")
        groups[name] = group.strip()
        if builder:
            file_digest = row.get("sha256")
            if (
                not isinstance(file_digest, str)
                or len(file_digest) != 64
                or any(character not in "0123456789abcdef" for character in file_digest.lower())
            ):
                raise SourceGroupManifestError(f"files[{name!r}] needs a valid sha256")

    if builder:
        # Prove that `groups` is an exact partition of `files` and agrees with
        # every files[name].source_group assignment.
        seen: set[str] = set()
        for group, members in payload["groups"].items():
            if not isinstance(group, str) or not group or not isinstance(members, list) or not members:
                raise SourceGroupManifestError("build_source_groups groups must map non-empty IDs to non-empty lists")
            for name in members:
                if name not in expected or name in seen or groups[name] != group:
                    raise SourceGroupManifestError("build_source_groups groups are not an exact consistent partition")
                seen.add(name)
        if seen != expected:
            raise SourceGroupManifestError("build_source_groups groups do not cover every target exactly once")
        if builder_version == 2:
            _validate_builder_v2_selection(payload, target_names, groups)

    normalized_payload = dict(payload)
    normalized_payload["_normalized_schema"] = normalized_schema
    normalized_payload["_normalized_method"] = normalized_method
    return groups, normalized_payload, raw_bytes


def _selection_key(seed: int, group: str, name: str) -> str:
    return _sha256_bytes(f"{seed}\0{group}\0{name}".encode("utf-8"))


def select_gate_names(
    names: Sequence[str],
    groups: Mapping[str, str],
    *,
    validation_count: int,
    tuning_ranges: str,
    number: int,
    gate_seed: int,
) -> dict[str, Any]:
    if validation_count < 1 or validation_count >= len(names):
        raise FrozenGateError("validation_count must leave non-empty training and validation splits")
    train_names = list(names[:-validation_count])
    validation_names = list(names[-validation_count:])
    tuning_indices = _parse_ranges(tuning_ranges, len(validation_names))
    tuning_names = [validation_names[index] for index in tuning_indices]
    training_groups = {groups[name] for name in train_names}
    tuning_groups = {groups[name] for name in tuning_names}
    overlap = training_groups & tuning_groups
    if overlap:
        raise SourceGroupManifestError(
            "model-training and tuning source groups overlap; cannot certify source-disjoint splits "
            f"({len(overlap)} groups, first: {sorted(overlap)[:3]})"
        )

    by_group: dict[str, list[str]] = {}
    tuning_name_set = set(tuning_names)
    for name in validation_names:
        group = groups[name]
        if name in tuning_name_set or group in training_groups or group in tuning_groups:
            continue
        by_group.setdefault(group, []).append(name)
    representatives: list[tuple[str, str, str]] = []
    for group, candidates in by_group.items():
        name = min(candidates, key=lambda candidate: _selection_key(gate_seed, group, candidate))
        representatives.append((_selection_key(gate_seed, group, name), group, name))
    representatives.sort()
    if len(representatives) < number:
        raise SourceGroupManifestError(
            f"only {len(representatives)} source-disjoint confirmation groups remain; {number} required"
        )
    selected = representatives[:number]
    gate_groups = {group for _, group, _ in selected}
    if training_groups & gate_groups or tuning_groups & gate_groups or len(gate_groups) != number:
        raise AssertionError("source-disjoint selection invariant failed")
    return {
        "train_names": train_names,
        "validation_names": validation_names,
        "tuning_names": tuning_names,
        "training_groups": sorted(training_groups),
        "tuning_groups": sorted(tuning_groups),
        "selected": [
            {"selection_key": key, "source_group": group, "name": name}
            for key, group, name in selected
        ],
    }


def _derive_seed(gate_seed: int, name: str, purpose: str) -> int:
    digest = hashlib.sha256(f"{gate_seed}\0{name}\0{purpose}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    value = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if value.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise FrozenGateError(f"target {path} has shape {value.shape}, expected {(IMAGE_SIZE, IMAGE_SIZE, 3)}")
    return np.ascontiguousarray(value)


def _to_fragments(image: np.ndarray) -> np.ndarray:
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError("image does not match fixed 24x24x20 geometry")
    return np.ascontiguousarray(
        image.reshape(GRID, FRAGMENT_SIZE, GRID, FRAGMENT_SIZE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(NFRAG, FRAGMENT_SIZE, FRAGMENT_SIZE, 3)
    )


def _assemble(tiles: np.ndarray, board: np.ndarray) -> np.ndarray:
    board = np.asarray(board, dtype=np.int64)
    _assert_permutation(board, label="board")
    ordered = tiles[board]
    return np.ascontiguousarray(
        ordered.reshape(GRID, GRID, FRAGMENT_SIZE, FRAGMENT_SIZE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
    )


def _assert_permutation(value: np.ndarray, *, label: str) -> None:
    array = np.asarray(value)
    if array.shape != (NFRAG,) or not np.issubdtype(array.dtype, np.integer):
        raise IntegrityError(f"{label} must be an integer vector of length {NFRAG}")
    if not np.array_equal(np.sort(array.astype(np.int64)), np.arange(NFRAG, dtype=np.int64)):
        raise IntegrityError(f"{label} is not a bijection over 0..{NFRAG - 1}")


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    for distribution in ("Pillow", "opencv-python", "scikit-image", "torch"):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = None
    return result


def _default_code_paths(workspace: Path) -> dict[str, Path]:
    names = (
        "distort.py",
        "solve_buddies.py",
        "eval_candidate_rank.py",
        "eval_seeded_qap.py",
        "eval_symbolic_ranker_blend.py",
        "positional_ddpm.py",
        "pipeline.py",
        "placement_metrics.py",
    )
    result = {"frozen_gate_harness": Path(__file__).resolve()}
    result.update({name[:-3]: workspace / "src" / name for name in names})
    return result


def _hash_named_files(paths: Mapping[str, Path], *, kind: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for role, raw_path in paths.items():
        path = raw_path.resolve()
        if not path.is_file():
            raise FrozenGateError(f"required {kind} {role!r} does not exist: {path}")
        result[role] = {"path": str(path), "sha256": sha256_file(path)}
    return result


def _write_integrity_files(root: Path, relative_files: Sequence[str]) -> tuple[str, str]:
    normalized = sorted(dict.fromkeys(relative_files))
    lines: list[str] = []
    for relative in normalized:
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise IntegrityError(f"artifact escapes gate directory: {relative}") from exc
        lines.append(f"{sha256_file(candidate)}  {Path(relative).as_posix()}")
    sums = ("\n".join(lines) + "\n").encode("utf-8")
    _atomic_write_bytes(root / "SHA256SUMS", sums, require_absent=True)
    root_digest = _sha256_bytes(sums)
    _atomic_write_bytes(root / "ROOT_SHA256", (root_digest + "\n").encode("ascii"), require_absent=True)
    return root_digest, _sha256_bytes(sums)


def freeze_gate(
    *,
    targets_dir: Path,
    source_groups_path: Path,
    gate_dir: Path,
    checkpoints: Mapping[str, Path],
    number: int = MINIMUM_GATE_SCENES,
    gate_seed: int = DEFAULT_GATE_SEED,
    validation_count: int = 300,
    tuning_ranges: str = DEFAULT_TUNING_RANGES,
    minimum_scenes: int = MINIMUM_GATE_SCENES,
) -> dict[str, Any]:
    """Create a byte-frozen confirmation gate and atomically publish it."""
    if number < minimum_scenes:
        raise FrozenGateError(f"gate requires at least {minimum_scenes} scenes, got {number}")
    required_checkpoint_roles = {"ranker", "affinity_primary", "affinity_secondary", "spatial"}
    if set(checkpoints) != required_checkpoint_roles:
        raise FrozenGateError(
            f"checkpoint roles must be exactly {sorted(required_checkpoint_roles)}, got {sorted(checkpoints)}"
        )
    targets_dir = targets_dir.resolve()
    if not targets_dir.is_dir():
        raise FrozenGateError(f"target directory does not exist: {targets_dir}")
    names = sorted(path.name for path in targets_dir.iterdir() if path.is_file())
    if len(names) <= validation_count:
        raise FrozenGateError("dataset is too small for the configured validation split")
    groups, source_payload, source_bytes = load_source_groups(source_groups_path.resolve(), names)
    _enforce_builder_v2_freeze_contract(
        source_payload,
        validation_count=validation_count,
        tuning_ranges=tuning_ranges,
        number=number,
        gate_seed=gate_seed,
    )
    # build_source_groups fingerprints are meaningful only for the exact target
    # corpus they were computed from.  Re-hash the complete corpus once at
    # freeze time; checking only the 24 selected files would not detect a stale
    # training-side duplicate group.
    target_file_digests: dict[str, str] = {}
    source_file_rows = source_payload.get("files")
    if isinstance(source_file_rows, dict):
        for name in names:
            actual = sha256_file(targets_dir / name)
            expected = str(source_file_rows[name]["sha256"]).lower()
            if actual != expected:
                raise SourceGroupManifestError(
                    f"target corpus differs from source-group fingerprint for {name}: {actual} != {expected}"
                )
            target_file_digests[name] = actual
    dataset_digest = _sha256_bytes(
        "".join(
            f"{name}\0{target_file_digests[name] if name in target_file_digests else sha256_file(targets_dir / name)}\n"
            for name in names
        )
        .encode("utf-8")
    )
    split = select_gate_names(
        names,
        groups,
        validation_count=validation_count,
        tuning_ranges=tuning_ranges,
        number=number,
        gate_seed=gate_seed,
    )
    declared_split = source_payload.get("split")
    if isinstance(declared_split, dict) and "selected_confirmation" in declared_split:
        declared = declared_split["selected_confirmation"]
        if not isinstance(declared, list) or not all(isinstance(name, str) for name in declared):
            raise SourceGroupManifestError("predeclared selected_confirmation must be a list of filenames")
        resolved = [row["name"] for row in split["selected"]]
        if len(declared) != number:
            raise SourceGroupManifestError(
                f"source manifest predeclares {len(declared)} confirmation scenes; requested {number}"
            )
        if resolved != declared:
            raise SourceGroupManifestError(
                "resolved confirmation selection differs from source manifest selected_confirmation; "
                "use its predeclared seed/range contract"
            )

    workspace = Path(__file__).resolve().parent.parent
    checkpoint_records = _hash_named_files(checkpoints, kind="checkpoint")
    code_records = _hash_named_files(_default_code_paths(workspace), kind="code file")
    gate_dir = gate_dir.resolve()
    if gate_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable gate directory: {gate_dir}")
    gate_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{gate_dir.name}.tmp-", dir=gate_dir.parent))
    try:
        (temporary / "scenes").mkdir()
        _atomic_write_bytes(temporary / "source_groups.input.json", source_bytes, require_absent=True)
        scene_records: list[dict[str, Any]] = []
        from distort import distort_frags

        for ordinal, selected in enumerate(split["selected"]):
            name = _safe_name(selected["name"])
            target_path = targets_dir / name
            target = _load_rgb(target_path)
            clean_fragments = _to_fragments(target)
            corruption_seed = _derive_seed(gate_seed, name, "corruption")
            permutation_seed = _derive_seed(gate_seed, name, "permutation")
            distorted = distort_frags(clean_fragments, np.random.default_rng(corruption_seed))
            if distorted.shape != clean_fragments.shape or distorted.dtype != np.uint8:
                raise FrozenGateError(f"distort_frags returned invalid data for {name}")
            permutation = np.random.default_rng(permutation_seed).permutation(NFRAG).astype(np.int16)
            _assert_permutation(permutation, label=f"{name} permutation")
            tiles = np.ascontiguousarray(distorted[permutation.astype(np.int64)])
            orientations = np.zeros(NFRAG, dtype=np.uint8)
            relative = f"scenes/{name}.npz"
            scene_path = temporary / relative
            arrays: dict[str, np.ndarray] = {
                "schema": np.asarray(SCENE_SCHEMA),
                "name": np.asarray(name),
                "tiles": tiles,
                "target": target,
                "permutation": permutation,
                "orientations_quarter_turns": orientations,
            }
            scene_bytes = deterministic_npz_bytes(arrays)
            _atomic_write_bytes(scene_path, scene_bytes, require_absent=True)
            scene_records.append(
                {
                    "ordinal": ordinal,
                    "name": name,
                    "source_group": selected["source_group"],
                    "selection_key": selected["selection_key"],
                    "file": relative,
                    "file_sha256": _sha256_bytes(scene_bytes),
                    "target_file_sha256": (
                        target_file_digests[name] if name in target_file_digests else sha256_file(target_path)
                    ),
                    "arrays_sha256": {
                        "tiles": sha256_array(tiles),
                        "target": sha256_array(target),
                        "permutation": sha256_array(permutation),
                        "orientations_quarter_turns": sha256_array(orientations),
                    },
                    "corruption_seed": corruption_seed,
                    "permutation_seed": permutation_seed,
                }
            )

        manifest: dict[str, Any] = {
            "schema": GATE_SCHEMA,
            "geometry": {
                "grid": GRID,
                "fragment_size": FRAGMENT_SIZE,
                "image_size": IMAGE_SIZE,
                "fragments": NFRAG,
                "orientation": "fixed_type1_no_rotation",
                "direction_order": list(DIRECTION_ORDER),
            },
            "gate_seed": gate_seed,
            "scene_count": number,
            "selection": {
                "algorithm": "sha256(gate_seed\\0source_group\\0filename), one filename per group",
                "validation_count": validation_count,
                "tuning_ranges": tuning_ranges,
            },
            "splits": {
                "training": {
                    "name_count": len(split["train_names"]),
                    "source_groups": split["training_groups"],
                },
                "tuning": {
                    "names": split["tuning_names"],
                    "source_groups": split["tuning_groups"],
                },
                "confirmation": {
                    "names": [row["name"] for row in scene_records],
                    "source_groups": [row["source_group"] for row in scene_records],
                },
            },
            "source_groups": {
                "schema": source_payload["_normalized_schema"],
                "method": source_payload["_normalized_method"],
                "input_path": str(source_groups_path.resolve()),
                "input_sha256": _sha256_bytes(source_bytes),
                "archived_file": "source_groups.input.json",
                "target_corpus_sha256": dataset_digest,
            },
            "corruption": {
                "function": "distort.distort_frags",
                "stored_as_actual_uint8_tile_bytes": True,
                "rng": "numpy.random.Generator(PCG64), independent corruption/permutation streams",
            },
            "arms": FIXED_ARMS,
            "primary_metric": "paired_mean_solve_ssim_i21_minus_i11",
            "checkpoints": checkpoint_records,
            "code": code_records,
            "environment": _package_versions(),
            "scenes": scene_records,
            "integrity": {
                "algorithm": "sha256",
                "sums_file": "SHA256SUMS",
                "root_file": "ROOT_SHA256",
                "note": "ROOT_SHA256 is SHA256 of exact SHA256SUMS bytes",
            },
        }
        _atomic_write_bytes(temporary / "manifest.json", _canonical_json_bytes(manifest), require_absent=True)
        artifact_files = ["manifest.json", "source_groups.input.json"] + [row["file"] for row in scene_records]
        root_digest, _ = _write_integrity_files(temporary, artifact_files)
        os.replace(temporary, gate_dir)
        return {"gate_dir": str(gate_dir), "root_sha256": root_digest, "scenes": number}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parse_sums(content: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IntegrityError("SHA256SUMS is not UTF-8") from exc
    for line in text.splitlines():
        pieces = line.split("  ", 1)
        if len(pieces) != 2 or len(pieces[0]) != 64:
            raise IntegrityError(f"malformed SHA256SUMS line: {line!r}")
        digest, relative = pieces
        if relative in result or any(character not in "0123456789abcdef" for character in digest):
            raise IntegrityError(f"invalid or duplicate SHA256SUMS entry: {line!r}")
        result[relative] = digest
    return result


def _validate_scene_arrays(scene: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    required = {
        "schema", "name", "tiles", "target", "permutation", "orientations_quarter_turns"
    }
    if set(arrays) != required:
        raise IntegrityError(f"scene {scene['name']} fields differ: {sorted(set(arrays) ^ required)}")
    if str(arrays["schema"].item()) != SCENE_SCHEMA or str(arrays["name"].item()) != scene["name"]:
        raise IntegrityError(f"scene identity/schema mismatch for {scene['name']}")
    tiles = arrays["tiles"]
    target = arrays["target"]
    permutation = arrays["permutation"]
    orientations = arrays["orientations_quarter_turns"]
    if tiles.shape != (NFRAG, FRAGMENT_SIZE, FRAGMENT_SIZE, 3) or tiles.dtype != np.uint8:
        raise IntegrityError(f"scene {scene['name']} has invalid tile bytes")
    if target.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or target.dtype != np.uint8:
        raise IntegrityError(f"scene {scene['name']} has invalid target bytes")
    _assert_permutation(permutation, label=f"{scene['name']} permutation")
    if orientations.shape != (NFRAG,) or orientations.dtype != np.uint8 or np.any(orientations != 0):
        raise IntegrityError(f"scene {scene['name']} violates fixed-orientation Type-1 contract")
    for key in ("tiles", "target", "permutation", "orientations_quarter_turns"):
        expected = scene["arrays_sha256"][key]
        actual = sha256_array(arrays[key])
        if actual != expected:
            raise IntegrityError(f"scene {scene['name']} array {key} hash mismatch")


def load_and_verify_gate(
    gate_dir: Path, *, minimum_scenes: int = MINIMUM_GATE_SCENES
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], str]:
    gate_dir = gate_dir.resolve()
    for required in ("manifest.json", "SHA256SUMS", "ROOT_SHA256"):
        if not (gate_dir / required).is_file():
            raise IntegrityError(f"gate is missing {required}: {gate_dir}")
    sums_bytes = (gate_dir / "SHA256SUMS").read_bytes()
    recorded_root = (gate_dir / "ROOT_SHA256").read_text(encoding="ascii").strip().lower()
    actual_root = _sha256_bytes(sums_bytes)
    if recorded_root != actual_root:
        raise IntegrityError("ROOT_SHA256 does not match exact SHA256SUMS bytes")
    sums = _parse_sums(sums_bytes)
    try:
        manifest = json.loads((gate_dir / "manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:
        raise IntegrityError("manifest.json is not valid JSON") from exc
    if manifest.get("schema") != GATE_SCHEMA:
        raise IntegrityError(f"gate manifest schema must be {GATE_SCHEMA!r}")
    if manifest.get("arms") != FIXED_ARMS:
        raise IntegrityError("gate arms/configuration differ from the precommitted raw/I11/I21 contract")
    geometry = manifest.get("geometry", {})
    if geometry != {
        "grid": GRID,
        "fragment_size": FRAGMENT_SIZE,
        "image_size": IMAGE_SIZE,
        "fragments": NFRAG,
        "orientation": "fixed_type1_no_rotation",
        "direction_order": list(DIRECTION_ORDER),
    }:
        raise IntegrityError("gate geometry or fixed-orientation contract differs")
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or len(scenes) < minimum_scenes or len(scenes) != manifest.get("scene_count"):
        raise IntegrityError(f"gate must contain at least {minimum_scenes} declared scenes")
    expected_files = {"manifest.json", manifest["source_groups"]["archived_file"]}
    expected_files.update(str(scene["file"]) for scene in scenes)
    if set(sums) != expected_files:
        raise IntegrityError(f"SHA256SUMS artifact set differs: {sorted(set(sums) ^ expected_files)}")
    for relative, expected in sums.items():
        path = (gate_dir / relative).resolve()
        try:
            path.relative_to(gate_dir)
        except ValueError as exc:
            raise IntegrityError(f"SHA256SUMS path escapes gate: {relative}") from exc
        if not path.is_file() or sha256_file(path) != expected:
            raise IntegrityError(f"artifact hash mismatch: {relative}")

    archived_source = gate_dir / manifest["source_groups"]["archived_file"]
    if sha256_file(archived_source) != manifest["source_groups"]["input_sha256"]:
        raise IntegrityError("archived source-group manifest hash mismatch")
    split_groups = {
        key: set(manifest["splits"][key]["source_groups"])
        for key in ("training", "tuning", "confirmation")
    }
    if (
        split_groups["training"] & split_groups["tuning"]
        or split_groups["training"] & split_groups["confirmation"]
        or split_groups["tuning"] & split_groups["confirmation"]
    ):
        raise IntegrityError("recorded train/tuning/confirmation source groups are not disjoint")
    if len(split_groups["confirmation"]) != len(scenes):
        raise IntegrityError("confirmation scenes are not one-per-source-group")
    if [scene["name"] for scene in scenes] != manifest["splits"]["confirmation"]["names"]:
        raise IntegrityError("confirmation scene list differs from split manifest")

    loaded: dict[str, dict[str, np.ndarray]] = {}
    for scene in scenes:
        relative = str(scene["file"])
        if sums[relative] != scene["file_sha256"]:
            raise IntegrityError(f"scene file digest duplicated inconsistently for {scene['name']}")
        arrays = _load_npz(gate_dir / relative)
        _validate_scene_arrays(scene, arrays)
        loaded[scene["name"]] = arrays
    return manifest, loaded, actual_root


def _verify_external_files(manifest: Mapping[str, Any]) -> None:
    for category in ("checkpoints", "code"):
        for role, record in manifest[category].items():
            path = Path(record["path"])
            if not path.is_file():
                raise IntegrityError(f"recorded {category[:-1]} {role!r} is missing: {path}")
            actual = sha256_file(path)
            if actual != record["sha256"]:
                raise IntegrityError(f"recorded {category[:-1]} {role!r} hash mismatch")


def _score_cache_contract(manifest: Mapping[str, Any], scene: Mapping[str, Any], root_digest: str) -> dict[str, Any]:
    return {
        "schema": SCORE_CACHE_SCHEMA,
        "gate_root_sha256": root_digest,
        "scene_name": scene["name"],
        "scene_file_sha256": scene["file_sha256"],
        "tiles_sha256": scene["arrays_sha256"]["tiles"],
        "candidate_k_per_encoder": FIXED_ARMS["i11"]["candidate_k_per_encoder"],
        "direction_order": list(DIRECTION_ORDER),
        "score_dtype": "float32",
        "checkpoints": {
            role: record["sha256"] for role, record in sorted(manifest["checkpoints"].items())
        },
        "code": {role: record["sha256"] for role, record in sorted(manifest["code"].items())},
    }


def _score_cache_paths(cache_dir: Path, name: str) -> tuple[Path, Path]:
    safe = _safe_name(name)
    cache = cache_dir / f"{safe}.scores.npz"
    return cache, cache.with_suffix(cache.suffix + ".sha256")


def _validate_score_arrays(
    arrays: Mapping[str, np.ndarray], expected_contract: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    required = {"contract", "candidate_ids", "candidate_valid", "raw_scores", "spatial_scores"}
    if set(arrays) != required:
        raise CacheContractError(f"score cache fields differ: {sorted(set(arrays) ^ required)}")
    try:
        contract = json.loads(str(arrays["contract"].item()))
    except Exception as exc:
        raise CacheContractError("score cache contract is invalid JSON") from exc
    if contract != expected_contract:
        raise CacheContractError("score cache belongs to a different scene/gate/checkpoint/code contract")
    candidates = arrays["candidate_ids"]
    valid = arrays["candidate_valid"]
    raw = arrays["raw_scores"]
    spatial = arrays["spatial_scores"]
    expected_width = 2 * int(FIXED_ARMS["i11"]["candidate_k_per_encoder"])
    if candidates.shape != (NFRAG, expected_width) or candidates.dtype != np.int16:
        raise CacheContractError(f"candidate_ids must be int16 ({NFRAG},{expected_width})")
    if valid.shape != candidates.shape or valid.dtype != np.bool_:
        raise CacheContractError("candidate_valid must be bool and aligned with candidate_ids")
    expected_shape = (NFRAG, NUM_DIRECTIONS, candidates.shape[1])
    if raw.shape != expected_shape or spatial.shape != expected_shape:
        raise CacheContractError(f"score tensors must have shape {expected_shape}")
    if raw.dtype != np.float32 or spatial.dtype != np.float32:
        raise CacheContractError("score tensors must be float32")
    expanded_valid = np.broadcast_to(valid[:, None, :], expected_shape)
    if not np.all(np.isfinite(raw[expanded_valid])) or not np.all(np.isfinite(spatial[expanded_valid])):
        raise CacheContractError("valid candidate scores must be finite")
    if np.any((candidates[valid] < 0) | (candidates[valid] >= NFRAG)):
        raise CacheContractError("valid candidate ID is outside 0..575")
    if not np.all(valid.any(axis=1)):
        raise CacheContractError("every anchor must retain at least one valid candidate")
    for anchor in range(NFRAG):
        row = candidates[anchor, valid[anchor]].astype(np.int64)
        if anchor in row or len(row) != len(np.unique(row)):
            raise CacheContractError(f"candidate row {anchor} contains self or duplicate valid IDs")
    return {
        "candidate_ids": candidates,
        "candidate_valid": valid,
        "raw_scores": raw,
        "spatial_scores": spatial,
    }


def write_score_cache(
    path: Path,
    *,
    contract: Mapping[str, Any],
    candidate_ids: np.ndarray,
    candidate_valid: np.ndarray,
    raw_scores: np.ndarray,
    spatial_scores: np.ndarray,
) -> str:
    arrays = {
        "contract": np.asarray(json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
        "candidate_ids": np.asarray(candidate_ids, dtype=np.int16),
        "candidate_valid": np.asarray(candidate_valid, dtype=np.bool_),
        "raw_scores": np.asarray(raw_scores, dtype=np.float32),
        "spatial_scores": np.asarray(spatial_scores, dtype=np.float32),
    }
    _validate_score_arrays(arrays, contract)
    content = deterministic_npz_bytes(arrays)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite score cache: {path}")
    _atomic_write_bytes(path, content, require_absent=True)
    digest = _sha256_bytes(content)
    try:
        _atomic_write_bytes(sidecar, f"{digest}  {path.name}\n".encode("ascii"), require_absent=True)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return digest


def load_score_cache(path: Path, expected_contract: Mapping[str, Any]) -> dict[str, np.ndarray]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise CacheContractError(f"score cache and digest sidecar must both exist: {path}")
    pieces = sidecar.read_text(encoding="ascii").strip().split("  ", 1)
    if len(pieces) != 2 or pieces[1] != path.name or sha256_file(path) != pieces[0]:
        raise CacheContractError(f"score cache digest mismatch: {path}")
    try:
        arrays = _load_npz(path)
        return _validate_score_arrays(arrays, expected_contract)
    except (IntegrityError, CacheContractError) as exc:
        raise CacheContractError(str(exc)) from exc


class _ScoringModels:
    def __init__(self, manifest: Mapping[str, Any], device_text: str) -> None:
        _verify_external_files(manifest)
        import torch
        from eval_candidate_rank import load_ranker
        from positional_ddpm import PositionalDDPM
        from train_offset_pose import load_frozen_affinity

        self.torch = torch
        self.device = torch.device(device_text)
        random.seed(int(manifest["gate_seed"]))
        np.random.seed(int(manifest["gate_seed"]) % (2**32))
        torch.manual_seed(int(manifest["gate_seed"]))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(manifest["gate_seed"]))
            torch.backends.cudnn.benchmark = False
        paths = {role: record["path"] for role, record in manifest["checkpoints"].items()}
        self.ranker, ranker_payload = load_ranker(paths["ranker"], self.device)
        self.affinity, _, _ = load_frozen_affinity(paths["affinity_primary"], self.device)
        self.affinity_secondary, _, _ = load_frozen_affinity(paths["affinity_secondary"], self.device)
        spatial_payload = torch.load(paths["spatial"], map_location="cpu", weights_only=False)
        self.spatial = PositionalDDPM(**spatial_payload["model_args"]).to(self.device)
        self.spatial.load_state_dict(spatial_payload["model"], strict=True)
        self.spatial.eval()
        for model in (self.ranker, self.affinity, self.affinity_secondary, self.spatial):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self.ranker_step = int(ranker_payload.get("step", -1))
        self.spatial_step = int(spatial_payload.get("step", -1))

    def score(self, tiles_uint8: np.ndarray) -> dict[str, np.ndarray]:
        torch = self.torch
        from eval_candidate_rank import score_full_graph
        from train_offset_pose import mine_affinity_candidates

        tiles = torch.from_numpy(np.ascontiguousarray(tiles_uint8)).permute(0, 3, 1, 2).float()
        tiles = tiles.to(self.device) / 255.0
        candidates_batch, valid_batch = mine_affinity_candidates(
            self.affinity,
            tiles.unsqueeze(0),
            candidate_k=int(FIXED_ARMS["i11"]["candidate_k_per_encoder"]),
            device=self.device,
            affinity_secondary=self.affinity_secondary,
        )
        candidates = candidates_batch[0]
        valid = valid_batch[0]
        raw_dnk = score_full_graph(
            self.ranker, tiles, candidates, valid, pair_batch=PAIR_BATCH, device=self.device
        )
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), context:
            features = self.spatial.encode_tiles(tiles.unsqueeze(0))
            full_spatial = self.spatial.directional_edge_scores(features)[0].float()
        width = int(candidates.shape[1])
        spatial = torch.empty((NFRAG, NUM_DIRECTIONS, width), dtype=torch.float32, device=self.device)
        anchors = torch.arange(NFRAG, device=self.device)[:, None]
        for direction in range(NUM_DIRECTIONS):
            spatial[:, direction] = full_spatial[direction][anchors, candidates]
        return {
            "candidate_ids": candidates.cpu().numpy().astype(np.int16),
            "candidate_valid": valid.cpu().numpy().astype(np.bool_),
            "raw_scores": raw_dnk.permute(1, 0, 2).cpu().numpy().astype(np.float32),
            "spatial_scores": spatial.cpu().numpy().astype(np.float32),
        }


def _cache_index_contract(manifest: Mapping[str, Any], root_digest: str) -> dict[str, Any]:
    return {
        "schema": SCORE_CACHE_INDEX_SCHEMA,
        "gate_root_sha256": root_digest,
        "scene_names": [scene["name"] for scene in manifest["scenes"]],
        "checkpoints": {
            role: record["sha256"] for role, record in sorted(manifest["checkpoints"].items())
        },
        "code": {role: record["sha256"] for role, record in sorted(manifest["code"].items())},
        "candidate_k_per_encoder": FIXED_ARMS["i11"]["candidate_k_per_encoder"],
    }


def _ensure_cache_index(cache_dir: Path, expected: Mapping[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "CACHE_INDEX.json"
    content = _canonical_json_bytes(expected)
    if path.exists():
        if path.read_bytes() != content:
            raise CacheContractError(f"score cache directory belongs to another contract: {cache_dir}")
    else:
        _atomic_write_bytes(path, content, require_absent=True)


def _neighbor_targets_numpy(permutation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _assert_permutation(permutation, label="permutation")
    inverse = np.argsort(permutation.astype(np.int64))
    targets = np.full((NFRAG, NUM_DIRECTIONS), -1, dtype=np.int64)
    exists = np.zeros_like(targets, dtype=np.bool_)
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for tile in range(NFRAG):
        cell = int(permutation[tile])
        row, column = divmod(cell, GRID)
        for direction, (dr, dc) in enumerate(offsets):
            rr, cc = row + dr, column + dc
            if 0 <= rr < GRID and 0 <= cc < GRID:
                exists[tile, direction] = True
                targets[tile, direction] = int(inverse[rr * GRID + cc])
    return targets, exists


def edge_r1(
    candidates: np.ndarray,
    valid: np.ndarray,
    scores: np.ndarray,
    permutation: np.ndarray,
) -> float:
    expected = (NFRAG, NUM_DIRECTIONS, candidates.shape[1])
    if scores.shape != expected or valid.shape != candidates.shape:
        raise ValueError("candidate/valid/score shapes differ")
    expanded = np.broadcast_to(valid[:, None, :], scores.shape)
    masked = np.where(expanded, scores, -np.inf)
    if not np.all(np.isfinite(masked).any(axis=2)):
        raise IntegrityError("a candidate direction row is entirely invalid")
    top_slots = np.argmax(masked, axis=2)
    top = candidates[np.arange(NFRAG)[:, None], top_slots]
    targets, exists = _neighbor_targets_numpy(permutation)
    return float(np.mean(top[exists] == targets[exists]))


def _board_metrics(
    *,
    tiles: np.ndarray,
    target: np.ndarray,
    permutation: np.ndarray,
    board: np.ndarray,
    restorer: Callable[[np.ndarray], np.ndarray],
) -> dict[str, float]:
    from placement_metrics import neighbour_accuracy, placement_accuracy
    from skimage.metrics import structural_similarity

    _assert_permutation(board, label="predicted board")
    truth_board = np.argsort(permutation.astype(np.int64))
    assembled = _assemble(tiles, board)
    restored = np.asarray(restorer(assembled))
    if restored.shape != target.shape or restored.dtype != np.uint8:
        raise IntegrityError("fixed restorer returned invalid output")
    return {
        "placement": float(placement_accuracy(board, truth_board)[0]),
        "neighbour": float(neighbour_accuracy(board, truth_board)[0]),
        "solve_ssim": float(structural_similarity(target, assembled, channel_axis=2, data_range=255)),
        "final_ssim": float(structural_similarity(target, restored, channel_axis=2, data_range=255)),
    }


def _solve_scores(candidates: np.ndarray, scores: np.ndarray, *, arm: str) -> np.ndarray:
    import torch
    from eval_seeded_qap import dense_rd
    from solve_buddies import solve_buddies_from_scores

    config = FIXED_ARMS[arm]
    candidate_tensor = torch.from_numpy(candidates.astype(np.int64, copy=False)).long()
    score_tensor = torch.from_numpy(scores).permute(1, 0, 2).contiguous()
    right, down = dense_rd(candidate_tensor, score_tensor)
    board, _ = solve_buddies_from_scores(
        right.numpy(),
        down.numpy(),
        max_edges=int(config["max_edges"]),
        min_margin=float(config["min_margin"]),
        repair_passes=int(config["repair_passes"]),
    )
    _assert_permutation(board, label=f"{arm} board")
    return board.astype(np.int64, copy=False)


def _summarize_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in ("edge_r1", "placement", "neighbour", "solve_ssim", "final_ssim"):
        values = [float(row[metric]) for row in rows if row[metric] is not None]
        result[metric] = (
            {"mean": float(np.mean(values)), "median": float(np.median(values))}
            if values
            else None
        )
    return result


def _paired_summary(
    per_scene: Sequence[Mapping[str, Any]],
    *,
    left: str,
    right: str,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {}
    for metric in ("edge_r1", "placement", "neighbour", "solve_ssim", "final_ssim"):
        differences = np.asarray(
            [scene["arms"][left][metric] - scene["arms"][right][metric] for scene in per_scene],
            dtype=np.float64,
        )
        count = len(differences)
        sampled = differences[rng.integers(0, count, size=(BOOTSTRAP_SAMPLES, count))].mean(axis=1)
        result[metric] = {
            "mean_delta": float(differences.mean()),
            "median_delta": float(np.median(differences)),
            "bootstrap_95_ci": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
            "wins": int(np.sum(differences > 0)),
            "ties": int(np.sum(differences == 0)),
            "losses": int(np.sum(differences < 0)),
        }
    return result


def _fixed_nlm(image: np.ndarray) -> np.ndarray:
    import cv2
    from pipeline import nlm_restore

    cv2.setNumThreads(1)
    return np.asarray(nlm_restore(image, h=10), dtype=np.uint8)


def evaluate_gate(
    *,
    gate_dir: Path,
    score_cache_dir: Path,
    report_path: Path,
    device: str,
) -> dict[str, Any]:
    manifest, scene_arrays, root_digest = load_and_verify_gate(gate_dir)
    # Scoring, dense conversion, solving, restoration, and metric code are all
    # part of the frozen experiment.  A complete cache is not permission to run
    # it through changed code or silently substituted checkpoints.
    _verify_external_files(manifest)
    _ensure_cache_index(score_cache_dir, _cache_index_contract(manifest, root_digest))
    missing: list[Mapping[str, Any]] = []
    caches: dict[str, dict[str, np.ndarray]] = {}
    for scene in manifest["scenes"]:
        cache_path, sidecar = _score_cache_paths(score_cache_dir, scene["name"])
        contract = _score_cache_contract(manifest, scene, root_digest)
        if cache_path.exists() or sidecar.exists():
            caches[scene["name"]] = load_score_cache(cache_path, contract)
        else:
            missing.append(scene)
    models: _ScoringModels | None = None
    if missing:
        models = _ScoringModels(manifest, device)
        for scene in missing:
            name = scene["name"]
            computed = models.score(scene_arrays[name]["tiles"])
            cache_path, _ = _score_cache_paths(score_cache_dir, name)
            contract = _score_cache_contract(manifest, scene, root_digest)
            write_score_cache(cache_path, contract=contract, **computed)
            caches[name] = load_score_cache(cache_path, contract)

    from eval_symbolic_ranker_blend import _standardize_rows

    per_scene: list[dict[str, Any]] = []
    for scene in manifest["scenes"]:
        name = scene["name"]
        frozen = scene_arrays[name]
        cache = caches[name]
        candidates = cache["candidate_ids"].astype(np.int64)
        valid = cache["candidate_valid"]
        raw = cache["raw_scores"]
        spatial = cache["spatial_scores"]
        valid_scores = np.broadcast_to(valid[:, None, :], raw.shape).copy()
        raw_masked = raw.copy()
        raw_masked[~valid_scores] = -np.inf
        raw_z = _standardize_rows(raw, valid_scores)
        spatial_z = _standardize_rows(spatial, valid_scores)
        blend = raw_z + float(FIXED_ARMS["i21"]["alpha"]) * spatial_z
        blend[~valid_scores] = -np.inf
        permutation = frozen["permutation"]

        raw_board = np.arange(NFRAG, dtype=np.int64)
        i11_board = _solve_scores(candidates, raw_masked, arm="i11")
        i21_board = _solve_scores(candidates, blend, arm="i21")
        arms: dict[str, dict[str, Any]] = {
            "raw_input": {
                "edge_r1": None,
                **_board_metrics(
                    tiles=frozen["tiles"], target=frozen["target"], permutation=permutation,
                    board=raw_board, restorer=_fixed_nlm,
                ),
            },
            "i11": {
                "edge_r1": edge_r1(candidates, valid, raw_masked, permutation),
                **_board_metrics(
                    tiles=frozen["tiles"], target=frozen["target"], permutation=permutation,
                    board=i11_board, restorer=_fixed_nlm,
                ),
            },
            "i21": {
                "edge_r1": edge_r1(candidates, valid, blend, permutation),
                **_board_metrics(
                    tiles=frozen["tiles"], target=frozen["target"], permutation=permutation,
                    board=i21_board, restorer=_fixed_nlm,
                ),
            },
        }
        per_scene.append({"name": name, "source_group": scene["source_group"], "arms": arms})
        print(json.dumps({"scene": name, "arms": arms}, sort_keys=True), flush=True)

    by_arm = {
        arm: _summarize_arm([scene["arms"][arm] for scene in per_scene])
        for arm in ("raw_input", "i11", "i21")
    }
    paired = _paired_summary(
        per_scene, left="i21", right="i11", seed=int(manifest["gate_seed"]) + 91_117
    )
    report: dict[str, Any] = {
        "schema": "pazzle-frozen-end-to-end-report-v1",
        "gate_root_sha256": root_digest,
        "scene_count": len(per_scene),
        "arms": FIXED_ARMS,
        "aggregate": by_arm,
        "paired_i21_minus_i11": paired,
        "primary": {
            "metric": "paired_mean_solve_ssim_i21_minus_i11",
            "value": paired["solve_ssim"]["mean_delta"],
        },
        "per_scene": per_scene,
        "score_cache": {
            "schema": SCORE_CACHE_SCHEMA,
            "pair_batch": PAIR_BATCH,
        },
    }
    content = _canonical_json_bytes(report)
    report_path = report_path.resolve()
    if report_path.exists():
        if report_path.read_bytes() != content:
            raise IntegrityError(f"existing report differs; refusing overwrite: {report_path}")
        sidecar = report_path.with_suffix(report_path.suffix + ".sha256")
        expected_sidecar = f"{_sha256_bytes(content)}  {report_path.name}\n".encode("ascii")
        if not sidecar.is_file() or sidecar.read_bytes() != expected_sidecar:
            raise IntegrityError(f"existing report digest sidecar is missing or differs: {sidecar}")
    else:
        _atomic_write_bytes(report_path, content, require_absent=True)
        _atomic_write_bytes(
            report_path.with_suffix(report_path.suffix + ".sha256"),
            f"{_sha256_bytes(content)}  {report_path.name}\n".encode("ascii"),
            require_absent=True,
        )
    return report


def verify_score_cache_directory(
    manifest: Mapping[str, Any], root_digest: str, score_cache_dir: Path, *, require_complete: bool
) -> dict[str, Any]:
    expected_index = _canonical_json_bytes(_cache_index_contract(manifest, root_digest))
    index_path = score_cache_dir / "CACHE_INDEX.json"
    if not index_path.is_file() or index_path.read_bytes() != expected_index:
        raise CacheContractError("score cache index is missing or belongs to another gate contract")
    verified = 0
    missing: list[str] = []
    for scene in manifest["scenes"]:
        cache, sidecar = _score_cache_paths(score_cache_dir, scene["name"])
        if not cache.exists() and not sidecar.exists():
            missing.append(scene["name"])
            continue
        load_score_cache(cache, _score_cache_contract(manifest, scene, root_digest))
        verified += 1
    if require_complete and missing:
        raise CacheContractError(f"score cache is incomplete; missing {len(missing)} scenes")
    return {"verified": verified, "missing": missing}


def _default_paths() -> dict[str, Path]:
    workspace = Path(__file__).resolve().parent.parent
    data_root = Path(os.environ.get("PAZZLE_DATA", "E:/pazzle_data"))
    work_root = Path(os.environ.get("PAZZLE_WORK", "E:/pazzle_work"))
    return {
        "workspace": workspace,
        "targets": data_root / "train" / "targets",
        "gate": work_root / "gates" / "frozen_v1",
        "ranker": workspace / "artifacts" / "candidate_rank" / "rank_v2w64_best.pt",
        "affinity_primary": workspace / "artifacts" / "macro_affinity" / "affinity_r1_1200_best.pt",
        "affinity_secondary": workspace / "artifacts" / "macro_affinity" / "affinity_r3_1000_best.pt",
        "spatial": work_root / "positional_ddpm" / "positional_ddpm_train_latest.pt",
    }


def _build_parser() -> argparse.ArgumentParser:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="create the immutable scene-byte gate")
    freeze.add_argument("--source-groups", type=Path, required=True,
                        help=f"complete {SOURCE_GROUP_SCHEMA} JSON; missing/partial manifests are rejected")
    freeze.add_argument("--targets-dir", type=Path, default=defaults["targets"])
    freeze.add_argument("--gate-dir", type=Path, default=defaults["gate"])
    freeze.add_argument("--n", type=int, default=MINIMUM_GATE_SCENES)
    freeze.add_argument("--gate-seed", type=int, default=DEFAULT_GATE_SEED)
    freeze.add_argument("--tuning-ranges", default=DEFAULT_TUNING_RANGES,
                        help="validation-relative start:count ranges already touched by selection/reports")
    freeze.add_argument("--ranker", type=Path, default=defaults["ranker"])
    freeze.add_argument("--affinity-primary", type=Path, default=defaults["affinity_primary"])
    freeze.add_argument("--affinity-secondary", type=Path, default=defaults["affinity_secondary"])
    freeze.add_argument("--spatial", type=Path, default=defaults["spatial"])

    evaluate = subparsers.add_parser("evaluate", help="evaluate fixed raw/I11/I21 arms")
    evaluate.add_argument("--gate-dir", type=Path, default=defaults["gate"])
    evaluate.add_argument("--score-cache-dir", type=Path, default=None)
    evaluate.add_argument("--report", type=Path, default=None)
    evaluate.add_argument("--device", default="cuda" if _torch_cuda_available() else "cpu")

    verify = subparsers.add_parser("verify", help="verify immutable bytes and optional derived caches")
    verify.add_argument("--gate-dir", type=Path, default=defaults["gate"])
    verify.add_argument("--score-cache-dir", type=Path, default=None)
    verify.add_argument("--require-externals", action="store_true",
                        help="also require current checkpoint/code paths to match their frozen hashes")
    return parser


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "freeze":
        result = freeze_gate(
            targets_dir=args.targets_dir,
            source_groups_path=args.source_groups,
            gate_dir=args.gate_dir,
            checkpoints={
                "ranker": args.ranker,
                "affinity_primary": args.affinity_primary,
                "affinity_secondary": args.affinity_secondary,
                "spatial": args.spatial,
            },
            number=args.n,
            gate_seed=args.gate_seed,
            validation_count=300,
            tuning_ranges=args.tuning_ranges,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "evaluate":
        cache_dir = args.score_cache_dir or args.gate_dir.with_name(args.gate_dir.name + "_score_cache")
        report = args.report or args.gate_dir.with_name(args.gate_dir.name + "_report.json")
        result = evaluate_gate(
            gate_dir=args.gate_dir, score_cache_dir=cache_dir, report_path=report, device=args.device
        )
        print(json.dumps({"report": str(report.resolve()), "primary": result["primary"]}, indent=2))
        return
    manifest, _, root_digest = load_and_verify_gate(args.gate_dir)
    if args.require_externals:
        _verify_external_files(manifest)
    result: dict[str, Any] = {
        "gate_dir": str(args.gate_dir.resolve()),
        "root_sha256": root_digest,
        "scenes": len(manifest["scenes"]),
        "core": "verified",
    }
    if args.score_cache_dir is not None:
        result["score_cache"] = verify_score_cache_directory(
            manifest, root_digest, args.score_cache_dir.resolve(), require_complete=True
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
