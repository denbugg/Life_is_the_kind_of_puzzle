"""Conservatively select between Union and direct rank-delta layouts.

The rank-delta arm changes only the order in which the Union hard edges enter
the translation-component builder.  This selector therefore compares the two
builders using evidence produced before component packing or cyclic placement:

1. prefer the arm with more redundant constraints that close consistently;
2. if those counts tie, prefer the arm with the larger rigid component;
3. on an exact evidence tie, retain the established Union arm.

No image reference, target layout, learned selector, score weight or threshold
is involved.  The strict fallback on ties is intentional: rank-delta must
provide positive component-geometry evidence before it may replace Union.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

ArmName = Literal["union_v2", "rank_delta_transfer"]


@dataclass(frozen=True)
class ComponentConsistencyEvidence:
    """Two target-free diagnostics from one translation-component build."""

    consistent_redundant_constraints: int
    largest_component: int
    tile_count: int

    def __post_init__(self) -> None:
        for name, value in (
            ("consistent_redundant_constraints", self.consistent_redundant_constraints),
            ("largest_component", self.largest_component),
            ("tile_count", self.tile_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.consistent_redundant_constraints < 0:
            raise ValueError("consistent_redundant_constraints must be non-negative")
        if self.tile_count < 1:
            raise ValueError("tile_count must be positive")
        if not 1 <= self.largest_component <= self.tile_count:
            raise ValueError("largest_component must be in [1, tile_count]")

    @property
    def lexicographic_key(self) -> tuple[int, int]:
        """Return the fixed evidence order used by the selector."""

        return (self.consistent_redundant_constraints, self.largest_component)

    def as_dict(self) -> dict[str, int]:
        """Return JSON-compatible evidence."""

        return asdict(self)


@dataclass(frozen=True)
class DirectRankDeltaComponentSelection:
    """Auditable outcome of the conservative two-arm selector."""

    union_v2: ComponentConsistencyEvidence
    rank_delta_transfer: ComponentConsistencyEvidence
    selected_arm: ArmName
    reason: Literal[
        "more_consistent_redundant_constraints",
        "consistent_tie_larger_component",
        "union_conservative_fallback",
    ]

    @property
    def treatment_selected(self) -> bool:
        return self.selected_arm == "rank_delta_transfer"

    def report(self) -> dict[str, Any]:
        """Return the frozen rule and evidence without any target information."""

        return {
            "schema": "aiijc-direct-rank-delta-component-selector-v1",
            "rule": (
                "lexicographically maximize "
                "(consistent_redundant_constraints, largest_component); "
                "exact ties retain union_v2"
            ),
            "selected_arm": self.selected_arm,
            "reason": self.reason,
            "union_v2": self.union_v2.as_dict(),
            "rank_delta_transfer": self.rank_delta_transfer.as_dict(),
        }


def select_direct_rank_delta_component_arm(
    union_v2: ComponentConsistencyEvidence,
    rank_delta_transfer: ComponentConsistencyEvidence,
) -> DirectRankDeltaComponentSelection:
    """Choose one whole-layout arm from target-free component geometry.

    The two evidence records must describe the same board.  Exact ties select
    ``union_v2``; this makes the treatment a conservative opt-in rather than a
    default replacement.
    """

    if union_v2.tile_count != rank_delta_transfer.tile_count:
        raise ValueError("both arms must describe the same tile_count")

    baseline_key = union_v2.lexicographic_key
    treatment_key = rank_delta_transfer.lexicographic_key
    if treatment_key > baseline_key:
        reason = (
            "more_consistent_redundant_constraints"
            if treatment_key[0] > baseline_key[0]
            else "consistent_tie_larger_component"
        )
        return DirectRankDeltaComponentSelection(
            union_v2=union_v2,
            rank_delta_transfer=rank_delta_transfer,
            selected_arm="rank_delta_transfer",
            reason=reason,
        )
    return DirectRankDeltaComponentSelection(
        union_v2=union_v2,
        rank_delta_transfer=rank_delta_transfer,
        selected_arm="union_v2",
        reason="union_conservative_fallback",
    )


__all__ = [
    "ArmName",
    "ComponentConsistencyEvidence",
    "DirectRankDeltaComponentSelection",
    "select_direct_rank_delta_component_arm",
]
