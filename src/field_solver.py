"""Absolute-field puzzle solver with optional rigid seam islands.

This module deliberately does *assembly only*.  It consumes one or more coarse
predictions of the clean photograph and returns a bijection between the 576
observed fragments and the 24x24 cells.  Pixel restoration belongs downstream.

The factorisation follows the measurements in M428/M429/M449/M456/M471:

* a 96x96 description (4x4 values per puzzle cell) contains enough absolute
  information for a Hungarian assignment;
* the best seam edges are useful as rigid relative constraints, but growing a
  component with less reliable edges welds correct islands together;
* a seam objective is safe for selecting among a small posterior sample, not
  for freely optimising over the enormous permutation space.

The main primitive is :func:`solve_field`.  With no components it is exactly a
Hungarian assignment.  With components, a bounded beam enumerates legal board
translations of the rigid islands and Hungarian assigns every remaining tile.
Every returned layout is therefore a full permutation by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from field_diffusion import GRID, N, cell_desc, frag_desc


Component = Mapping[int, tuple[int, int]]


def _standardize_rows(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Standardise each cell/fragment descriptor independently.

    The degradation applies an independent affine photometric transform to each
    fragment.  Descriptor standardisation removes most of that nuisance when
    the predicted field is already accurate (M429).  It is intentionally an
    explicit mode: on weak field predictions it amplifies noise in flat tiles.
    """
    x = np.asarray(x, np.float64)
    mu = x.mean(1, keepdims=True)
    sd = x.std(1, keepdims=True)
    return (x - mu) / np.maximum(sd, eps)


def descriptor_cost(field: np.ndarray, frags: np.ndarray,
                    mode: str = "raw") -> np.ndarray:
    """Return the ``(cell, tile)`` content cost matrix.

    ``raw`` is the robust operating point for imperfect predictions.  ``zscore``
    removes per-fragment brightness/contrast and is the higher oracle ceiling.
    ``blend:X`` forms ``(1-X)*raw + X*zscore`` after robust scale matching,
    which is useful for a validation sweep without changing the solver.
    """
    a = cell_desc(np.asarray(field, np.float64))
    b = frag_desc(np.asarray(frags, np.float64))

    def sqdist(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        c = ((u * u).sum(1)[:, None] + (v * v).sum(1)[None, :]
             - 2.0 * u @ v.T)
        return np.maximum(c, 0.0)

    raw = sqdist(a, b)
    if mode == "raw":
        return raw
    z = sqdist(_standardize_rows(a), _standardize_rows(b))
    if mode == "zscore":
        return z
    if mode.startswith("blend:"):
        w = float(mode.split(":", 1)[1])
        if not 0.0 <= w <= 1.0:
            raise ValueError("blend weight must be in [0, 1]")
        # The raw and standardised distances have unrelated units.  A robust
        # global scale makes the blend weight meaningful while preserving all
        # rankings within either matrix.
        rp = float(np.percentile(raw, 50))
        zp = float(np.percentile(z, 50))
        z_scaled = z * (rp / max(zp, 1e-12))
        return (1.0 - w) * raw + w * z_scaled
    raise ValueError(f"unknown descriptor mode: {mode}")


def hungarian_layout(cost: np.ndarray) -> np.ndarray:
    """Minimum-cost full cell->tile bijection."""
    cost = np.asarray(cost, np.float64)
    if cost.shape != (N, N):
        raise ValueError(f"expected {(N, N)} cost matrix, got {cost.shape}")
    rows, cols = linear_sum_assignment(cost)
    layout = np.empty(N, np.int64)
    layout[rows] = cols
    return layout


def _normalise_component(component: Component) -> dict[int, tuple[int, int]]:
    if not component:
        raise ValueError("empty component")
    out = {int(t): (int(p[0]), int(p[1])) for t, p in component.items()}
    if len(out) != len(component):
        raise ValueError("duplicate tile in component")
    coords = list(out.values())
    if len(set(coords)) != len(coords):
        raise ValueError("two component tiles occupy the same relative cell")
    y0 = min(y for y, _ in coords)
    x0 = min(x for _, x in coords)
    out = {t: (y - y0, x - x0) for t, (y, x) in out.items()}
    if max(y for y, _ in out.values()) >= GRID or max(
            x for _, x in out.values()) >= GRID:
        raise ValueError("component does not fit the board")
    return out


def validate_components(components: Iterable[Component]) -> list[dict[int,
                                                                       tuple[int, int]]]:
    """Normalise components and reject cross-component tile duplication."""
    clean = []
    used: set[int] = set()
    for comp in components:
        c = _normalise_component(comp)
        bad = set(c) & used
        if bad:
            raise ValueError(f"tiles occur in more than one component: {sorted(bad)[:5]}")
        if min(c) < 0 or max(c) >= N:
            raise ValueError("component tile index outside the puzzle")
        used.update(c)
        clean.append(c)
    return clean


@dataclass(frozen=True)
class _Placement:
    cells: tuple[int, ...]
    tiles: tuple[int, ...]
    cost: float
    offset: tuple[int, int]


@dataclass
class _BeamState:
    occupied: np.ndarray
    placements: list[_Placement]
    cost: float


def _component_placements(component: Component, cost: np.ndarray,
                          topk: int) -> list[_Placement]:
    tiles = tuple(component)
    coords = tuple(component.values())
    h = max(y for y, _ in coords) + 1
    w = max(x for _, x in coords) + 1
    out = []
    for y0 in range(GRID - h + 1):
        for x0 in range(GRID - w + 1):
            cells = tuple((y0 + y) * GRID + x0 + x for y, x in coords)
            value = float(sum(cost[cell, tile]
                              for cell, tile in zip(cells, tiles)))
            out.append(_Placement(cells, tiles, value, (y0, x0)))
    out.sort(key=lambda p: p.cost)
    return out[:max(1, topk)]


def _complete_state(state: _BeamState, cost: np.ndarray,
                    component_tiles: set[int]) -> tuple[np.ndarray, float]:
    layout = np.full(N, -1, np.int64)
    for p in state.placements:
        layout[np.asarray(p.cells, np.int64)] = np.asarray(p.tiles, np.int64)
    free_cells = np.flatnonzero(layout < 0)
    free_tiles = np.asarray(sorted(set(range(N)) - component_tiles), np.int64)
    if len(free_cells) != len(free_tiles):
        raise RuntimeError("component placement does not leave a square assignment")
    tail = 0.0
    if len(free_cells):
        rows, cols = linear_sum_assignment(cost[np.ix_(free_cells, free_tiles)])
        layout[free_cells[rows]] = free_tiles[cols]
        tail = float(cost[free_cells[rows], free_tiles[cols]].sum())
    if len(np.unique(layout)) != N or np.any(layout < 0):
        raise RuntimeError("solver did not produce a bijection")
    return layout, state.cost + tail


def constrained_layout(cost: np.ndarray, components: Sequence[Component],
                       beam: int = 64, offsets: int = 24) -> tuple[np.ndarray,
                                                                   float]:
    """Solve a field assignment while preserving rigid relative components.

    Only the top ``offsets`` translations under the absolute field are expanded
    for each island.  The bounded beam resolves overlap between islands.  At the
    end, one Hungarian solve assigns all tiles outside the islands.  Components
    are ordered by decreasing size and then by translation margin, so the most
    informative rigid evidence consumes board space first.
    """
    cost = np.asarray(cost, np.float64)
    if cost.shape != (N, N):
        raise ValueError(f"expected {(N, N)} cost matrix, got {cost.shape}")
    comps = validate_components(components)
    if not comps:
        layout = hungarian_layout(cost)
        return layout, float(cost[np.arange(N), layout].sum())
    if beam < 1 or offsets < 1:
        raise ValueError("beam and offsets must be positive")

    options = [_component_placements(c, cost, offsets) for c in comps]
    margins = [(o[1].cost - o[0].cost) / max(len(c), 1)
               if len(o) > 1 else np.inf
               for c, o in zip(comps, options)]
    order = sorted(range(len(comps)),
                   key=lambda i: (-len(comps[i]), -margins[i]))
    options = [options[i] for i in order]

    states = [_BeamState(np.zeros(N, np.bool_), [], 0.0)]
    for choices in options:
        nxt: list[_BeamState] = []
        for state in states:
            for p in choices:
                ci = np.asarray(p.cells, np.int64)
                if state.occupied[ci].any():
                    continue
                occ = state.occupied.copy()
                occ[ci] = True
                nxt.append(_BeamState(occ, state.placements + [p],
                                      state.cost + p.cost))
        if not nxt:
            raise RuntimeError(
                "no non-overlapping component placement survived; increase "
                "offsets/beam or inspect the component geometry")
        # A per-tile average prevents a large early component from setting an
        # arbitrary scale for later beam pruning; all states at this depth have
        # placed the same components, so the ordering is otherwise unchanged.
        nxt.sort(key=lambda s: s.cost)
        states = nxt[:beam]

    component_tiles = {t for c in comps for t in c}
    completed = [_complete_state(s, cost, component_tiles) for s in states]
    return min(completed, key=lambda z: z[1])


def solve_field(field: np.ndarray, frags: np.ndarray, *, mode: str = "raw",
                components: Sequence[Component] = (), beam: int = 64,
                offsets: int = 24) -> tuple[np.ndarray, float]:
    """Coarse photograph -> complete puzzle layout and assignment cost."""
    cost = descriptor_cost(field, frags, mode)
    return constrained_layout(cost, components, beam, offsets)


def components_from_score_tail(right_cost: np.ndarray, down_cost: np.ndarray,
                               keep: int = 127) -> list[dict[int,
                                                             tuple[int, int]]]:
    """Build conflict-safe rigid islands from only the global score tail.

    M449 measured the best 127 directed pairs per board at roughly 98.5%
    precision.  This function intentionally takes the *board-wide* tail rather
    than one best edge from every fragment: weak/flat fragments contribute no
    edge, while a textured fragment may contribute two.  ``keep`` is therefore
    a precision operating point, not a target component size.
    """
    if keep <= 0:
        return []
    mats = [np.asarray(right_cost, np.float64),
            np.asarray(down_cost, np.float64)]
    if any(m.shape != (N, N) for m in mats):
        raise ValueError(f"score matrices must both have shape {(N, N)}")
    # Convert costs to weights and remove self-pairs.  Selecting a small tail
    # with argpartition avoids sorting all ~663k directed candidates.
    flat = []
    for direction, m in enumerate(mats):
        w = -m.copy()
        np.fill_diagonal(w, -np.inf)
        flat.append((direction, w.reshape(-1)))
    joined = np.concatenate([x[1] for x in flat])
    k = min(int(keep), int(np.isfinite(joined).sum()))
    ids = np.argpartition(joined, -k)[-k:]
    ids = ids[np.argsort(joined[ids])[::-1]]
    anchors, directions, targets, weights = [], [], [], []
    span = N * N
    for q in ids:
        d, rem = divmod(int(q), span)
        i, j = divmod(rem, N)
        anchors.append(i)
        directions.append(3 if d == 0 else 1)  # right / down convention
        targets.append(j)
        weights.append(float(joined[q]))
    # Imported lazily so descriptor-only evaluation does not pull the legacy
    # model stack into memory.
    from solve_buddies import build_directed_components
    return build_directed_components(anchors, directions, targets, weights,
                                     max_edges=k)


def seam_energy(layout: np.ndarray, right_cost: np.ndarray,
                down_cost: np.ndarray) -> float:
    """Mean realised seam cost of a complete layout (lower is better)."""
    a = np.asarray(layout, np.int64).reshape(GRID, GRID)
    h = np.asarray(right_cost, np.float64)[a[:, :-1], a[:, 1:]]
    v = np.asarray(down_cost, np.float64)[a[:-1, :], a[1:, :]]
    return float((h.sum() + v.sum()) / (h.size + v.size))


def select_layout(layouts: Sequence[np.ndarray], assignment_costs: Sequence[float],
                  right_cost: np.ndarray | None = None,
                  down_cost: np.ndarray | None = None,
                  seam_weight: float = 0.25) -> int:
    """Select one posterior hypothesis without opening a global search.

    Assignment costs are robustly standardised across the sampled hypotheses.
    If seam matrices are supplied, their realised energies provide a weak,
    independent tie-break.  The function only ranks the finite sample it was
    handed; it never optimises the seam objective over arbitrary permutations.
    """
    if len(layouts) == 0 or len(layouts) != len(assignment_costs):
        raise ValueError("layouts and costs must be non-empty and aligned")
    vals = np.asarray(assignment_costs, np.float64)

    def rz(x: np.ndarray) -> np.ndarray:
        med = np.median(x)
        scale = np.median(np.abs(x - med)) * 1.4826
        return (x - med) / max(float(scale), 1e-12)

    score = rz(vals)
    if (right_cost is None) != (down_cost is None):
        raise ValueError("right_cost and down_cost must be supplied together")
    if right_cost is not None:
        se = np.asarray([seam_energy(x, right_cost, down_cost)
                         for x in layouts])
        score = score + float(seam_weight) * rz(se)
    return int(np.argmin(score))
