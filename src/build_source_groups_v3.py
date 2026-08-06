"""Build the v3-protocol source manifest for the clean E11 artifact attempt v4.

This versioned builder intentionally leaves :mod:`build_source_groups`
unchanged.  It reuses that module's image fingerprints and deterministic DSU
group identity, while replacing its lossy four-chunk pHash candidate index
with five disjoint chunks of 13, 13, 13, 13, and 12 bits.  By the pigeonhole
principle, every pair of 64-bit hashes at Hamming distance at most four shares
at least one complete chunk and is therefore examined.

All experiment controls are constants.  The CLI exposes only the target path
and publishes only the canonical create-once E: manifest.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import build_source_groups as legacy
import eval_frozen_end_to_end_gate as frozen


SCHEMA_VERSION = 2
TRAIN_COUNT = 6700
VALIDATION_COUNT = 300
KNOWN_TUNE_VAL_MAX = 99
CANDIDATE_VAL_MIN = 100
SELECTION_SEED = "20260808"
SELECTION_COUNT = 48
PHASH_THRESHOLD = 4
DHASH_THRESHOLD = 6
MEAN_RGB_THRESHOLD = 36.0
PHASH_CHUNK_SIZES = (13, 13, 13, 13, 12)
PHASH_CHUNK_OFFSETS = (0, 13, 26, 39, 52)

GATE_V1_VALIDATION_IDS = (
    119,
    122,
    136,
    138,
    142,
    157,
    158,
    164,
    170,
    172,
    203,
    208,
    215,
    218,
    219,
    228,
    229,
    248,
    252,
    253,
    256,
    263,
    279,
    295,
)
GATE_V2_VALIDATION_IDS = (
    100,
    102,
    105,
    121,
    174,
    206,
    207,
    211,
    212,
    221,
    223,
    224,
    227,
    238,
    249,
    251,
    268,
    272,
    275,
    282,
    283,
    287,
    290,
    299,
)
PRIOR_GATE_VALIDATION_IDS = tuple(sorted(GATE_V1_VALIDATION_IDS + GATE_V2_VALIDATION_IDS))

CANONICAL_OUTPUT = Path("E:/pazzle_work/rank96_e11_v4/source_groups_v4.json")
DEFAULT_TARGETS = Path(os.environ.get("PAZZLE_DATA", "E:/pazzle_data")) / "train" / "targets"

ALGORITHMS_CONTRACT: dict[str, Any] = {
    "contract": "pazzle-source-groups-v3-fixed-five-chunk-index-v1",
    "exact": "sha256(file_bytes)",
    "phash": "cv2.dct(gray32) low8 median threshold",
    "dhash": "gray9x8 horizontal difference",
    "phash_threshold": 4,
    "dhash_threshold": 6,
    "mean_rgb_threshold": 36.0,
    "candidate_index": {
        "method": "five_disjoint_contiguous_phash_chunks",
        "bit_order": "least-significant-bit first",
        "chunk_sizes": [13, 13, 13, 13, 12],
        "chunk_offsets": [0, 13, 26, 39, 52],
        "guarantee": "every 64-bit pair with Hamming distance <= 4 shares a complete chunk",
    },
}

BUILDER_CONTRACT: dict[str, Any] = {
    "schema": "pazzle-source-groups-e11-v3-builder-v1",
    "legacy_fingerprint_module": "build_source_groups.py",
    "fixed_algorithms": ALGORITHMS_CONTRACT,
    "fixed_split": {
        "train_count": 6700,
        "val_count": 300,
        "known_tune_val_ids": [0, 99],
        "candidate_val_min": 100,
        "selection_seed": "20260808",
        "selection_count": 48,
        "excluded_prior_validation_ids": list(PRIOR_GATE_VALIDATION_IDS),
        "prior_group_rule": "exclude current-v3 groups containing any prior gate scene name",
    },
}

BASE_SELECTION_CONTRACT = copy.deepcopy(frozen.BUILDER_V2_SELECTION_CONTRACT)
V3_SELECTION_CONTRACT: dict[str, Any] = {
    "schema": "pazzle-source-groups-e11-v3-selection-v1",
    "base": BASE_SELECTION_CONTRACT,
    "additional_forbidden_groups": "current-v3 groups containing any prior gate scene name",
    "ranking": "ascending sha256(seed\\0source_group\\0name), then name; first item per source group",
}

_LEGACY_BUILD_LOCK = threading.RLock()


@dataclass(frozen=True)
class SelectionV3:
    eligible: tuple[str, ...]
    selected: tuple[str, ...]
    prior_scene_names: tuple[str, ...]
    prior_source_groups_v3: tuple[str, ...]


def candidate_pairs_five_chunks(values: Sequence[int]) -> Iterable[tuple[int, int]]:
    """Yield a superset of every pair at 64-bit Hamming distance <= 4."""

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, raw_value in enumerate(values):
        value = int(raw_value)
        if value < 0 or value >= 1 << 64:
            raise ValueError(f"pHash must be an unsigned 64-bit integer, got {value}")
        for chunk, (offset, size) in enumerate(zip(PHASH_CHUNK_OFFSETS, PHASH_CHUNK_SIZES)):
            mask = (1 << size) - 1
            buckets[(chunk, (value >> offset) & mask)].append(index)
    seen: set[tuple[int, int]] = set()
    for members in buckets.values():
        for offset, left in enumerate(members):
            for right in members[offset + 1 :]:
                pair = (left, right) if left < right else (right, left)
                if pair not in seen:
                    seen.add(pair)
                    yield pair


def build_groups_v3(
    items: Sequence[legacy.Fingerprint],
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, int]]:
    """Reuse the legacy grouping bytecode with only its candidate index replaced.

    The legacy module stays byte-for-byte untouched.  The narrow temporary
    injection is serialized and restored in ``finally`` so exceptions cannot
    leak the E11-specific index into legacy callers.
    """

    values = list(items)
    with _LEGACY_BUILD_LOCK:
        original_candidate_pairs = legacy._candidate_pairs
        legacy._candidate_pairs = candidate_pairs_five_chunks
        try:
            return legacy.build_groups(
                values,
                phash_threshold=PHASH_THRESHOLD,
                dhash_threshold=DHASH_THRESHOLD,
                mean_rgb_threshold=MEAN_RGB_THRESHOLD,
            )
        finally:
            legacy._candidate_pairs = original_candidate_pairs


def _validate_name_space(names: Sequence[str], group_for_name: Mapping[str, str]) -> list[str]:
    ordered = list(names)
    if ordered != sorted(ordered) or len(ordered) != TRAIN_COUNT + VALIDATION_COUNT:
        raise ValueError(
            f"E11 v3 requires exactly {TRAIN_COUNT + VALIDATION_COUNT} sorted target names"
        )
    if len(set(ordered)) != len(ordered) or set(group_for_name) != set(ordered):
        raise ValueError("E11 v3 source-group mapping must cover every target exactly once")
    if any(not isinstance(group_for_name[name], str) or not group_for_name[name] for name in ordered):
        raise ValueError("E11 v3 source-group IDs must be non-empty strings")
    return ordered


def select_confirmation_v3(
    names: Sequence[str], group_for_name: Mapping[str, str]
) -> SelectionV3:
    """Apply the fixed split, excluding prior identities under current v3 groups."""

    ordered = _validate_name_space(names, group_for_name)
    train_names = ordered[:TRAIN_COUNT]
    validation_names = ordered[TRAIN_COUNT:]
    prior_names = tuple(validation_names[index] for index in PRIOR_GATE_VALIDATION_IDS)
    prior_groups = {group_for_name[name] for name in prior_names}
    forbidden_groups = {group_for_name[name] for name in train_names}
    forbidden_groups.update(
        group_for_name[name] for name in validation_names[:CANDIDATE_VAL_MIN]
    )
    forbidden_groups.update(prior_groups)
    excluded_ids = set(PRIOR_GATE_VALIDATION_IDS)
    eligible = tuple(
        name
        for validation_id, name in enumerate(validation_names)
        if validation_id >= CANDIDATE_VAL_MIN
        and validation_id not in excluded_ids
        and group_for_name[name] not in forbidden_groups
    )
    ranked = sorted(
        eligible,
        key=lambda name: (
            hashlib.sha256(
                f"{SELECTION_SEED}\0{group_for_name[name]}\0{name}".encode("utf-8")
            ).hexdigest(),
            name,
        ),
    )
    selected: list[str] = []
    used_groups: set[str] = set()
    for name in ranked:
        group = group_for_name[name]
        if group in used_groups:
            continue
        selected.append(name)
        used_groups.add(group)
        if len(selected) == SELECTION_COUNT:
            break
    if len(selected) != SELECTION_COUNT:
        raise RuntimeError(
            f"only {len(selected)} independently eligible v3 source groups; "
            f"{SELECTION_COUNT} required"
        )
    return SelectionV3(
        eligible=eligible,
        selected=tuple(selected),
        prior_scene_names=prior_names,
        prior_source_groups_v3=tuple(sorted(prior_groups)),
    )


def validate_manifest_v3(
    payload: Mapping[str, Any], target_names: Sequence[str], group_for_name: Mapping[str, str]
) -> SelectionV3:
    """Fail closed unless a schema-v2 manifest is the exact E11 v3 contract."""

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("E11 source manifest must use schema_version=2")
    if payload.get("algorithms") != ALGORITHMS_CONTRACT:
        raise ValueError("E11 source manifest algorithms differ from the fixed v3 contract")
    if payload.get("builder_contract") != BUILDER_CONTRACT:
        raise ValueError("E11 source manifest builder_contract differs")
    split = payload.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("E11 source manifest split is malformed")
    exact_fields = {
        "train_count": TRAIN_COUNT,
        "val_count": VALIDATION_COUNT,
        "known_tune_val_ids": [0, KNOWN_TUNE_VAL_MAX],
        "candidate_val_min": CANDIDATE_VAL_MIN,
        "selection_seed": SELECTION_SEED,
        "excluded_val_ids": list(PRIOR_GATE_VALIDATION_IDS),
        "selection_contract": BASE_SELECTION_CONTRACT,
        "v3_selection_contract": V3_SELECTION_CONTRACT,
    }
    for key, expected in exact_fields.items():
        if split.get(key) != expected:
            raise ValueError(f"E11 source manifest split.{key} differs from the fixed contract")
    selection = select_confirmation_v3(target_names, group_for_name)
    expected_derived = {
        "prior_scene_names": list(selection.prior_scene_names),
        "prior_source_groups_v3": list(selection.prior_source_groups_v3),
        "eligible_confirmation": list(selection.eligible),
        "selected_confirmation": list(selection.selected),
    }
    for key, expected in expected_derived.items():
        if split.get(key) != expected:
            raise ValueError(f"E11 source manifest split.{key} is not independently reproducible")
    return selection


def build_manifest_v3(targets_dir: Path) -> dict[str, Any]:
    root = targets_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"target directory does not exist: {root}")
    paths = sorted(
        path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".png"
    )
    if len(paths) != TRAIN_COUNT + VALIDATION_COUNT:
        raise ValueError(
            f"E11 v3 requires exactly {TRAIN_COUNT + VALIDATION_COUNT} PNG targets, "
            f"found {len(paths)}"
        )
    items = [legacy.fingerprint(path) for path in paths]
    names = [item.name for item in items]
    group_for_name, groups, stats = build_groups_v3(items)
    selection = select_confirmation_v3(names, group_for_name)
    split = {
        "train_count": TRAIN_COUNT,
        "val_count": VALIDATION_COUNT,
        "known_tune_val_ids": [0, KNOWN_TUNE_VAL_MAX],
        "candidate_val_min": CANDIDATE_VAL_MIN,
        "selection_seed": SELECTION_SEED,
        "excluded_val_ids": list(PRIOR_GATE_VALIDATION_IDS),
        "selection_contract": BASE_SELECTION_CONTRACT,
        "v3_selection_contract": V3_SELECTION_CONTRACT,
        "prior_scene_names": list(selection.prior_scene_names),
        "prior_source_groups_v3": list(selection.prior_source_groups_v3),
        "eligible_confirmation": list(selection.eligible),
        "selected_confirmation": list(selection.selected),
    }
    file_rows = {
        item.name: {
            "sha256": item.sha256,
            "phash64": f"{item.phash:016x}",
            "dhash64": f"{item.dhash:016x}",
            "mean_rgb": [round(value, 6) for value in item.mean_rgb],
            "source_group": group_for_name[item.name],
        }
        for item in items
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "builder_contract": BUILDER_CONTRACT,
        "targets_root": str(root),
        "algorithms": ALGORITHMS_CONTRACT,
        "split": split,
        "stats": {"files": len(items), **stats},
        "groups": groups,
        "files": file_rows,
    }
    validate_manifest_v3(payload, names, group_for_name)
    return payload


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _publish_create_once(path: Path, content: bytes) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise FileExistsError(f"refusing to overwrite different E11 source manifest: {path}")
        return "already_identical"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.stage-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            if os.name == "nt":
                os.rename(temporary, path)
            else:
                os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != content:
                raise FileExistsError(
                    f"concurrent E11 source manifest differs; refusing overwrite: {path}"
                )
            return "already_identical"
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


def write_canonical_manifest(payload: Mapping[str, Any]) -> tuple[str, str]:
    content = _canonical_bytes(payload)
    status = _publish_create_once(CANONICAL_OUTPUT, content)
    return hashlib.sha256(content).hexdigest(), status


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-dir", type=Path, default=DEFAULT_TARGETS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = build_manifest_v3(args.targets_dir)
    digest, status = write_canonical_manifest(manifest)
    print(
        json.dumps(
            {
                "output": str(CANONICAL_OUTPUT.resolve()),
                "sha256": digest,
                "status": status,
                "selected": manifest["split"]["selected_confirmation"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
