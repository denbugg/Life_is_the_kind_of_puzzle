"""Metadata-only source inventory for a future four-emitter real protocol.

The audit is intentionally independent of pixels, labels, model checkpoints and
predictions.  It recursively collects explicit source-panel rosters from prior
config/report JSON artifacts and can prove when organizer-train has no unused
source left for a genuinely source-disjoint development panel.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiijc_puzzle.protocol import DECLARED_SOURCE_PANEL_KEYS, sha256_file


@dataclass(frozen=True)
class RosterArtifact:
    """One JSON artifact that explicitly declares prior source membership."""

    path: str
    sha256: str
    declared_count: int
    row_count: int
    union_count: int
    union_digest: str


def names_digest(names: Sequence[str]) -> str:
    """Hash one ordered newline-delimited filename sequence."""

    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def singular_source_filenames(value: Any, *, parent_key: str = "") -> set[str]:
    """Collect row-wise ``source_filename`` values from nested metadata."""

    names: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            names.update(singular_source_filenames(child, parent_key=str(key)))
    elif isinstance(value, list):
        for child in value:
            names.update(singular_source_filenames(child, parent_key=parent_key))
    elif (
        parent_key == "source_filename"
        and isinstance(value, str)
        and value.startswith("img_")
        and value.endswith(".png")
    ):
        names.add(value)
    return names


def declared_source_filenames(value: Any, *, parent_key: str = "") -> set[str]:
    """Collect every explicit ``*_filenames`` source roster conservatively.

    A bare generic ``filenames`` key is deliberately ignored so a copied
    organizer manifest or submission inventory is not mistaken for an opened
    source panel.  Named role rosters such as ``fit_filenames``,
    ``terminal_filenames`` and ``lineage_exposed_filenames`` are included.
    """

    names: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            names.update(declared_source_filenames(child, parent_key=str(key)))
    elif isinstance(value, list):
        is_roster = (
            parent_key in DECLARED_SOURCE_PANEL_KEYS
            or (parent_key.endswith("_filenames") and parent_key != "filenames")
        )
        if is_roster:
            if not all(isinstance(item, str) for item in value):
                raise ValueError(f"declared roster {parent_key!r} must contain strings")
            names.update(
                item
                for item in value
                if item.startswith("img_") and item.endswith(".png")
            )
        else:
            for child in value:
                names.update(declared_source_filenames(child, parent_key=parent_key))
    return names


def source_union_from_payload(value: Any) -> tuple[set[str], set[str], set[str]]:
    """Return declared-panel, row-wise, and combined source sets."""

    declared = declared_source_filenames(value)
    singular = singular_source_filenames(value)
    return declared, singular, declared | singular


def recursive_json_roster_inventory(
    project_root: Path,
    *,
    reserved: Sequence[Path] = (),
) -> tuple[tuple[RosterArtifact, ...], tuple[str, ...], str]:
    """Scan config/output JSON recursively under the established semantic rule."""

    root = project_root.resolve()
    excluded = {path.resolve() for path in reserved}
    paths = (
        *sorted((root / "configs").rglob("*.json")),
        *sorted((root / "outputs").rglob("*.json")),
    )
    names: set[str] = set()
    artifacts: list[RosterArtifact] = []
    for path in paths:
        if path.resolve() in excluded:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        declared, rows, union = source_union_from_payload(payload)
        if not union:
            continue
        ordered = tuple(sorted(union))
        names.update(ordered)
        artifacts.append(
            RosterArtifact(
                path=str(path.resolve().relative_to(root)),
                sha256=sha256_file(path),
                declared_count=len(declared),
                row_count=len(rows),
                union_count=len(ordered),
                union_digest=names_digest(ordered),
            )
        )
    inventory_digest = hashlib.sha256(
        "\n".join(
            f"{item.path}\0{item.sha256}\0{item.union_digest}" for item in artifacts
        ).encode()
    ).hexdigest()
    return tuple(artifacts), tuple(sorted(names)), inventory_digest


def deterministic_fresh_roster(
    train_names: Sequence[str],
    excluded_names: Sequence[str],
    *,
    count: int,
    namespace: str,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return one deterministic roster and the complete eligible inventory."""

    if isinstance(count, bool) or count <= 0:
        raise ValueError("count must be positive")
    if not namespace:
        raise ValueError("namespace must be non-empty")
    train = tuple(train_names)
    if len(train) != len(set(train)):
        raise ValueError("organizer-train roster contains duplicates")
    excluded = set(excluded_names)
    eligible = tuple(name for name in train if name not in excluded)
    prefix = f"{namespace}\0{seed}\0".encode()
    ranked = tuple(
        sorted(
            eligible,
            key=lambda name: (hashlib.sha256(prefix + name.encode()).digest(), name),
        )
    )
    return ranked[:count] if len(ranked) >= count else (), eligible


def minimal_inventory_blocker(
    train_names: Sequence[str],
    prior_excluded_train: Sequence[str],
    final_scored_sources: Sequence[str],
) -> dict[str, Any]:
    """Prove exhaustion from an earlier signed complement plus its later use."""

    train = set(train_names)
    prior = set(prior_excluded_train) & train
    final = set(final_scored_sources) & train
    overlap = prior & final
    covered = prior | final
    eligible = tuple(sorted(train - covered))
    return {
        "organizer_train_count": len(train),
        "previously_excluded_train_count": len(prior),
        "final_complement_scored_source_count": len(final),
        "prior_final_overlap_count": len(overlap),
        "covered_train_count": len(covered),
        "remaining_fresh_train_count": len(eligible),
        "remaining_fresh_train_filenames": list(eligible),
        "complete_inventory_blocker": len(covered) == len(train) and not eligible,
    }


__all__ = [
    "RosterArtifact",
    "declared_source_filenames",
    "deterministic_fresh_roster",
    "minimal_inventory_blocker",
    "names_digest",
    "recursive_json_roster_inventory",
    "singular_source_filenames",
    "source_union_from_payload",
]
