"""Build deterministic source-content groups for an immutable validation gate.

The PAZZLE train targets are numbered, while model selection has repeatedly used
the lexicographic validation tail.  A confirmation gate must therefore avoid
both known tuning indices and near-duplicate source photographs that also occur
in the model-training prefix.  This tool groups exact duplicates and a
conservative set of perceptual near-duplicates, then deterministically selects
one image per eligible group.

This is a split-integrity tool, not a training component.  It never reads dirty
inputs and never changes image pixels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


SCHEMA_VERSION = 1
EXCLUSION_SCHEMA_VERSION = 2
EXCLUSION_SYNTAX = "comma-separated validation IDs or start:count ranges (start inclusive)"


class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.size[a] < self.size[b] or (self.size[a] == self.size[b] and a > b):
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


@dataclass(frozen=True)
class Fingerprint:
    name: str
    sha256: str
    phash: int
    dhash: int
    mean_rgb: tuple[float, float, float]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bool(bit))
    return value


def _perceptual_hash(rgb: np.ndarray) -> int:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    flat = dct.reshape(-1)
    threshold = float(np.median(flat[1:]))
    return _bits_to_int(flat > threshold)


def _difference_hash(rgb: np.ndarray) -> int:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return _bits_to_int(small[:, 1:] > small[:, :-1])


def fingerprint(path: Path) -> Fingerprint:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if rgb.shape != (480, 480, 3):
        raise ValueError(f"{path.name}: expected RGB 480x480, got {rgb.shape}")
    mean = tuple(float(v) for v in rgb.reshape(-1, 3).mean(axis=0))
    return Fingerprint(
        name=path.name,
        sha256=_file_sha256(path),
        phash=_perceptual_hash(rgb),
        dhash=_difference_hash(rgb),
        mean_rgb=mean,
    )


def _candidate_pairs(values: list[int]) -> Iterable[tuple[int, int]]:
    """Yield pairs that share at least one 16-bit pHash chunk.

    Any two 64-bit hashes within Hamming distance four must share at least one
    complete 16-bit chunk, so this is lossless for the default pHash gate while
    avoiding a dense 7,000^2 comparison.
    """
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        for chunk in range(4):
            buckets[(chunk, (value >> (16 * chunk)) & 0xFFFF)].append(index)
    seen: set[tuple[int, int]] = set()
    for members in buckets.values():
        for offset, left in enumerate(members):
            for right in members[offset + 1 :]:
                pair = (left, right) if left < right else (right, left)
                if pair not in seen:
                    seen.add(pair)
                    yield pair


def build_groups(
    items: list[Fingerprint],
    phash_threshold: int = 4,
    dhash_threshold: int = 6,
    mean_rgb_threshold: float = 36.0,
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, int]]:
    dsu = _DSU(len(items))
    exact: dict[str, int] = {}
    exact_unions = 0
    for index, item in enumerate(items):
        previous = exact.get(item.sha256)
        if previous is None:
            exact[item.sha256] = index
        else:
            dsu.union(previous, index)
            exact_unions += 1

    perceptual_unions = 0
    for left, right in _candidate_pairs([item.phash for item in items]):
        a, b = items[left], items[right]
        if (a.phash ^ b.phash).bit_count() > phash_threshold:
            continue
        if (a.dhash ^ b.dhash).bit_count() > dhash_threshold:
            continue
        mean_distance = max(abs(x - y) for x, y in zip(a.mean_rgb, b.mean_rgb))
        if mean_distance > mean_rgb_threshold:
            continue
        before = dsu.find(left) == dsu.find(right)
        dsu.union(left, right)
        perceptual_unions += int(not before)

    by_root: dict[int, list[str]] = defaultdict(list)
    for index, item in enumerate(items):
        by_root[dsu.find(index)].append(item.name)

    group_for_name: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    for names in sorted((sorted(names) for names in by_root.values()), key=lambda row: row[0]):
        payload = "\0".join(names).encode("utf-8")
        group_id = f"g_{hashlib.sha256(payload).hexdigest()[:16]}"
        groups[group_id] = names
        for name in names:
            group_for_name[name] = group_id
    stats = {
        "exact_unions": exact_unions,
        "perceptual_unions": perceptual_unions,
        "groups": len(groups),
        "non_singleton_groups": sum(len(names) > 1 for names in groups.values()),
        "largest_group": max((len(names) for names in groups.values()), default=0),
    }
    return group_for_name, groups, stats


def parse_excluded_val_indices(spec: str, *, val_count: int) -> list[int]:
    """Parse validation-local IDs from ``N`` and ``start:count`` tokens.

    For example, ``"3,10:2"`` excludes validation IDs 3, 10, and 11.  Ranges
    use a start plus a positive count rather than an inclusive end.  Duplicate,
    negative, malformed, and out-of-bounds IDs are rejected so an exclusion
    contract cannot be silently broader or narrower than intended.
    """
    if val_count < 0:
        raise ValueError("val_count must be non-negative")
    if not spec.strip():
        return []

    result: list[int] = []
    seen: set[int] = set()
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError(f"empty exclusion token in {spec!r}")
        if token.count(":") > 1:
            raise ValueError(f"malformed exclusion token: {token!r}")
        if ":" in token:
            raw_start, raw_count = (part.strip() for part in token.split(":"))
            try:
                start = int(raw_start)
                count = int(raw_count)
            except ValueError as exc:
                raise ValueError(f"malformed exclusion range: {token!r}") from exc
            if count <= 0:
                raise ValueError(f"exclusion range count must be positive: {token!r}")
            values = range(start, start + count)
        else:
            try:
                values = (int(token),)
            except ValueError as exc:
                raise ValueError(f"malformed exclusion index: {token!r}") from exc

        for val_id in values:
            if val_id < 0 or val_id >= val_count:
                raise ValueError(
                    f"excluded validation ID {val_id} is outside [0, {val_count})"
                )
            if val_id in seen:
                raise ValueError(f"duplicate excluded validation ID: {val_id}")
            seen.add(val_id)
            result.append(val_id)
    return sorted(result)


def _validated_excluded_val_ids(values: Iterable[int], *, val_count: int) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"excluded validation IDs must be integers, got {value!r}")
        if value < 0 or value >= val_count:
            raise ValueError(f"excluded validation ID {value} is outside [0, {val_count})")
        if value in seen:
            raise ValueError(f"duplicate excluded validation ID: {value}")
        seen.add(value)
        result.append(value)
    return sorted(result)


def select_confirmation(
    names: list[str],
    group_for_name: dict[str, str],
    groups: dict[str, list[str]],
    *,
    train_count: int,
    val_count: int,
    tune_val_max: int,
    candidate_val_min: int,
    count: int,
    seed: str,
    excluded_val_ids: Iterable[int] = (),
) -> tuple[list[str], list[str]]:
    if train_count + val_count != len(names):
        raise ValueError("train_count + val_count must equal the number of targets")
    excluded = set(_validated_excluded_val_ids(excluded_val_ids, val_count=val_count))
    train_names = set(names[:train_count])
    val_names = names[train_count:]
    tune_names = set(val_names[: tune_val_max + 1])
    forbidden_names = train_names | tune_names
    forbidden_groups = {group_for_name[name] for name in forbidden_names}

    eligible: list[str] = []
    for val_id, name in enumerate(val_names):
        if val_id < candidate_val_min:
            continue
        if val_id in excluded:
            continue
        group_id = group_for_name[name]
        if group_id in forbidden_groups:
            continue
        if any(member in forbidden_names for member in groups[group_id]):
            continue
        eligible.append(name)

    ranked = sorted(
        eligible,
        key=lambda name: (
            hashlib.sha256(f"{seed}\0{group_for_name[name]}\0{name}".encode("utf-8")).hexdigest(),
            name,
        ),
    )
    selected: list[str] = []
    used_groups: set[str] = set()
    for name in ranked:
        group_id = group_for_name[name]
        if group_id in used_groups:
            continue
        selected.append(name)
        used_groups.add(group_id)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"only {len(selected)} eligible source groups for requested {count}")
    return eligible, selected


def build_manifest(args: argparse.Namespace) -> dict:
    root = Path(args.targets).resolve()
    paths = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".png")
    if not paths:
        raise FileNotFoundError(f"no PNG targets in {root}")
    items = [fingerprint(path) for path in paths]
    names = [item.name for item in items]
    group_for_name, groups, stats = build_groups(
        items,
        phash_threshold=args.phash_threshold,
        dhash_threshold=args.dhash_threshold,
        mean_rgb_threshold=args.mean_rgb_threshold,
    )
    exclusion_spec = getattr(args, "exclude_val_indices", "")
    excluded_val_ids = parse_excluded_val_indices(exclusion_spec, val_count=args.val_count)
    eligible, selected = select_confirmation(
        names,
        group_for_name,
        groups,
        train_count=args.train_count,
        val_count=args.val_count,
        tune_val_max=args.tune_val_max,
        candidate_val_min=args.candidate_val_min,
        count=args.select_count,
        seed=args.seed,
        excluded_val_ids=excluded_val_ids,
    )
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
    split = {
        "train_count": args.train_count,
        "val_count": args.val_count,
        "known_tune_val_ids": [0, args.tune_val_max],
        "candidate_val_min": args.candidate_val_min,
        "selection_seed": args.seed,
        "eligible_confirmation": eligible,
        "selected_confirmation": selected,
    }
    if excluded_val_ids:
        split.update(
            {
                "excluded_val_ids": excluded_val_ids,
                "selection_contract": {
                    "index_space": "zero-based IDs within the validation split",
                    "exclusion_syntax": EXCLUSION_SYNTAX,
                    "eligibility_order": [
                        "validation ID is at least candidate_val_min",
                        "validation ID is not in excluded_val_ids",
                        "source group is absent from train and known tuning validation IDs",
                    ],
                    "ranking": "ascending sha256(seed\\0source_group\\0name), then name; first item per source group",
                },
            }
        )

    return {
        "schema_version": EXCLUSION_SCHEMA_VERSION if excluded_val_ids else SCHEMA_VERSION,
        "targets_root": str(root),
        "algorithms": {
            "exact": "sha256(file_bytes)",
            "phash": "cv2.dct(gray32) low8 median threshold",
            "dhash": "gray9x8 horizontal difference",
            "phash_threshold": args.phash_threshold,
            "dhash_threshold": args.dhash_threshold,
            "mean_rgb_threshold": args.mean_rgb_threshold,
        },
        "split": split,
        "stats": {"files": len(items), **stats},
        "groups": groups,
        "files": file_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-count", type=int, default=6700)
    parser.add_argument("--val-count", type=int, default=300)
    parser.add_argument("--tune-val-max", type=int, default=55)
    parser.add_argument("--candidate-val-min", type=int, default=100)
    parser.add_argument(
        "--exclude-val-indices",
        default="",
        help=f"exclude validation-local IDs before selection; {EXCLUSION_SYNTAX}",
    )
    parser.add_argument("--select-count", type=int, default=24)
    parser.add_argument("--seed", default="20260806")
    parser.add_argument("--phash-threshold", type=int, default=4)
    parser.add_argument("--dhash-threshold", type=int, default=6)
    parser.add_argument("--mean-rgb-threshold", type=float, default=36.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {output}")
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    output.write_text(encoded, encoding="utf-8")
    print(json.dumps({"output": str(output), "stats": manifest["stats"], "selected": manifest["split"]["selected_confirmation"]}, indent=2))


if __name__ == "__main__":
    main()
