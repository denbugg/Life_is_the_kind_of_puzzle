"""Sparse order-2 Growing Consensus proposals for directional puzzle edges.

This module deliberately implements only the smallest inference step that is
missing from the existing soft-cycle and component solvers: an edge that is not
present in the candidate graph may be proposed when multiple distinct
three-edge squares imply it.  It does not hard-lock edges or assemble a layout;
those decisions require a separately frozen precision/coverage gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, order=True)
class DirectedConsensusEdge:
    """A canonical right/down directed adjacency."""

    first: int
    second: int
    dx: int
    dy: int

    def __post_init__(self) -> None:
        if (self.dx, self.dy) not in ((1, 0), (0, 1)):
            raise ValueError("consensus edges must point right or down")
        if self.first == self.second:
            raise ValueError("a tile cannot be adjacent to itself")


@dataclass(frozen=True, order=True)
class SquareWitness:
    """Tiles in top-left, top-right, bottom-left, bottom-right order."""

    top_left: int
    top_right: int
    bottom_left: int
    bottom_right: int

    def __post_init__(self) -> None:
        if len(
            {
                self.top_left,
                self.top_right,
                self.bottom_left,
                self.bottom_right,
            }
        ) != 4:
            raise ValueError("a 2x2 witness must contain four distinct tiles")


@dataclass(frozen=True)
class ConsensusEdgeProposal:
    """A missing edge and the distinct incomplete squares supporting it."""

    edge: DirectedConsensusEdge
    witnesses: tuple[SquareWitness, ...]

    @property
    def support(self) -> int:
        return len(self.witnesses)


@dataclass(frozen=True)
class Order2ConsensusResult:
    """Complete input loops and score-independent missing-edge proposals."""

    complete_loops: tuple[SquareWitness, ...]
    proposals: tuple[ConsensusEdgeProposal, ...]
    input_edge_count: int


def _normalize_candidates(
    candidates: Sequence[Iterable[int]], *, tile_count: int, name: str
) -> tuple[frozenset[int], ...]:
    if tile_count <= 0:
        raise ValueError("tile_count must be positive")
    if len(candidates) != tile_count:
        raise ValueError(f"{name} must contain {tile_count} rows")
    normalized: list[frozenset[int]] = []
    for first, row in enumerate(candidates):
        values = frozenset(int(second) for second in row)
        if first in values:
            raise ValueError(f"{name}[{first}] contains a self edge")
        if any(second < 0 or second >= tile_count for second in values):
            raise ValueError(f"{name}[{first}] contains an out-of-range tile")
        normalized.append(values)
    return tuple(normalized)


def _incoming(
    outgoing: tuple[frozenset[int], ...], *, tile_count: int
) -> tuple[frozenset[int], ...]:
    values: list[set[int]] = [set() for _ in range(tile_count)]
    for first, row in enumerate(outgoing):
        for second in row:
            values[second].add(first)
    return tuple(frozenset(row) for row in values)


def discover_order2_consensus(
    right_candidates: Sequence[Iterable[int]],
    down_candidates: Sequence[Iterable[int]],
    *,
    tile_count: int,
    min_support: int = 2,
) -> Order2ConsensusResult:
    """Find complete 2x2 loops and missing edges supported by incomplete loops.

    Each witness contains exactly three input edges and one absent edge.  The
    absent edge is emitted only when at least ``min_support`` distinct squares
    imply the same oriented adjacency.  Candidate score magnitude is never used.

    With at most ``k`` candidates per side, runtime is ``O(tile_count * k**3)``
    and storage is proportional to the number of discovered loop witnesses.
    """
    if min_support < 2:
        raise ValueError("min_support must be at least two")
    right = _normalize_candidates(
        right_candidates, tile_count=tile_count, name="right_candidates"
    )
    down = _normalize_candidates(
        down_candidates, tile_count=tile_count, name="down_candidates"
    )
    right_in = _incoming(right, tile_count=tile_count)
    down_in = _incoming(down, tile_count=tile_count)

    complete: set[SquareWitness] = set()
    support: dict[DirectedConsensusEdge, set[SquareWitness]] = {}

    def record(
        first: int,
        second: int,
        dx: int,
        dy: int,
        top_left: int,
        top_right: int,
        bottom_left: int,
        bottom_right: int,
    ) -> None:
        if len({top_left, top_right, bottom_left, bottom_right}) != 4:
            return
        edge = DirectedConsensusEdge(first, second, dx, dy)
        witness = SquareWitness(
            top_left=top_left,
            top_right=top_right,
            bottom_left=bottom_left,
            bottom_right=bottom_right,
        )
        support.setdefault(edge, set()).add(witness)

    for top_left in range(tile_count):
        # Full loops and squares missing their bottom or right edge.  The top
        # and left edges provide the common corner for these enumerations.
        for top_right in right[top_left]:
            for bottom_left in down[top_left]:
                if top_right == bottom_left:
                    continue
                for bottom_right in down[top_right]:
                    if bottom_right in right[bottom_left]:
                        if len(
                            {top_left, top_right, bottom_left, bottom_right}
                        ) == 4:
                            complete.add(
                                SquareWitness(
                                    top_left,
                                    top_right,
                                    bottom_left,
                                    bottom_right,
                                )
                            )
                    else:
                        record(
                            bottom_left,
                            bottom_right,
                            1,
                            0,
                            top_left,
                            top_right,
                            bottom_left,
                            bottom_right,
                        )
                for bottom_right in right[bottom_left]:
                    if bottom_right not in down[top_right]:
                        record(
                            top_right,
                            bottom_right,
                            0,
                            1,
                            top_left,
                            top_right,
                            bottom_left,
                            bottom_right,
                        )

        # Squares missing their top edge.  The left, bottom and right edges
        # identify the otherwise absent top-right tile through down-incoming.
        for bottom_left in down[top_left]:
            for bottom_right in right[bottom_left]:
                for top_right in down_in[bottom_right]:
                    if top_right not in right[top_left]:
                        record(
                            top_left,
                            top_right,
                            1,
                            0,
                            top_left,
                            top_right,
                            bottom_left,
                            bottom_right,
                        )

        # Squares missing their left edge, symmetrically using right-incoming.
        for top_right in right[top_left]:
            for bottom_right in down[top_right]:
                for bottom_left in right_in[bottom_right]:
                    if bottom_left not in down[top_left]:
                        record(
                            top_left,
                            bottom_left,
                            0,
                            1,
                            top_left,
                            top_right,
                            bottom_left,
                            bottom_right,
                        )

    proposals = [
        ConsensusEdgeProposal(edge=edge, witnesses=tuple(sorted(witnesses)))
        for edge, witnesses in support.items()
        if len(witnesses) >= min_support
    ]
    proposals.sort(
        key=lambda proposal: (
            -proposal.support,
            proposal.edge.dy,
            proposal.edge.dx,
            proposal.edge.first,
            proposal.edge.second,
            proposal.witnesses,
        )
    )
    return Order2ConsensusResult(
        complete_loops=tuple(sorted(complete)),
        proposals=tuple(proposals),
        input_edge_count=sum(len(row) for row in right)
        + sum(len(row) for row in down),
    )
