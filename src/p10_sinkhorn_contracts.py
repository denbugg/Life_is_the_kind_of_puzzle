"""ORBIT-24 P10 G0a: synthetic Sinkhorn/permutation contracts.

This module intentionally accepts no puzzle images, targets, graph cache, or learned
checkpoint.  It validates the exact differentiable assignment primitive required by
P10 before any FIT-source pipeline is constructed.

P10 scope: 24x24 fixed-orientation permutation; 576 observed tiles and 576 slots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

GRID = 24
N_TILES = GRID * GRID
SINKHORN_ITERS = 20


def log_sinkhorn(logits: torch.Tensor, iterations: int = SINKHORN_ITERS) -> torch.Tensor:
    """Return a non-negative doubly-stochastic matrix from tile-to-slot logits.

    The operation is performed entirely in log-space.  It supports a leading batch
    dimension but P10 G0a invokes it on one matrix to keep every assertion explicit.
    """
    if logits.ndim < 2 or logits.shape[-1] != logits.shape[-2]:
        raise ValueError(f"expected square [..., N, N] logits, received {tuple(logits.shape)}")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")

    z = logits
    for _ in range(iterations):
        z = z - torch.logsumexp(z, dim=-1, keepdim=True)
        z = z - torch.logsumexp(z, dim=-2, keepdim=True)
    p = torch.exp(z)
    if not torch.isfinite(p).all():
        raise FloatingPointError("non-finite Sinkhorn matrix")
    return p


def decode_linear_assignment(logits: torch.Tensor) -> np.ndarray:
    """Deterministically maximize tile-to-slot logits and return tile->slot."""
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError(f"expected [N,N] logits, received {tuple(logits.shape)}")
    values = logits.detach().float().cpu().numpy()
    rows, cols = linear_sum_assignment(-values)
    n = values.shape[0]
    if not np.array_equal(rows, np.arange(n, dtype=rows.dtype)):
        raise RuntimeError("linear assignment did not return every tile row exactly once")
    tile_to_slot = np.empty(n, dtype=np.int64)
    tile_to_slot[rows] = cols
    if np.unique(tile_to_slot).size != n:
        raise RuntimeError("discrete decoder violated slot bijection")
    return tile_to_slot


def _assert_close(name: str, value: float, threshold: float) -> None:
    if value > threshold:
        raise AssertionError(f"{name}={value:.8g} exceeds threshold={threshold:.8g}")


def _matrix_errors(p: torch.Tensor) -> tuple[float, float]:
    n = p.shape[-1]
    ones = torch.ones(n, device=p.device, dtype=p.dtype)
    row_error = float((p.sum(dim=-1) - ones).abs().max().detach().cpu())
    col_error = float((p.sum(dim=-2) - ones).abs().max().detach().cpu())
    return row_error, col_error


def run_contracts(device: torch.device, seed: int) -> dict[str, object]:
    """Run fixed synthetic checks and return only reproducible scalar evidence."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Contract 1: the 24x24 zero-noise identity must survive 20 log-domain passes
    # and decode to the exact identity permutation.
    identity_logits = torch.full((N_TILES, N_TILES), -12.0, device=device)
    identity_logits.fill_diagonal_(12.0)
    identity_soft = log_sinkhorn(identity_logits, SINKHORN_ITERS)
    identity_pred = decode_linear_assignment(identity_logits)
    identity_expected = np.arange(N_TILES, dtype=np.int64)
    identity_accuracy = float(np.mean(identity_pred == identity_expected))
    id_row_error, id_col_error = _matrix_errors(identity_soft)
    _assert_close("identity_row_error", id_row_error, 1e-5)
    _assert_close("identity_col_error", id_col_error, 1e-5)
    if identity_accuracy != 1.0:
        raise AssertionError(f"zero-noise identity decode accuracy={identity_accuracy}")

    # Contract 2: tile-row equivariance.  A new row t represents canonical tile
    # perm[t], so it must continue to select canonical slot perm[t].
    perm = torch.randperm(N_TILES, device=device)
    permuted_logits = identity_logits.index_select(0, perm)
    permuted_soft = log_sinkhorn(permuted_logits, SINKHORN_ITERS)
    equivariance_error = float((permuted_soft - identity_soft.index_select(0, perm)).abs().max().detach().cpu())
    permuted_pred = decode_linear_assignment(permuted_logits)
    permuted_expected = perm.detach().cpu().numpy().astype(np.int64, copy=False)
    permutation_accuracy = float(np.mean(permuted_pred == permuted_expected))
    if permutation_accuracy != 1.0:
        raise AssertionError(f"row-permutation decode accuracy={permutation_accuracy}")
    _assert_close("row_permutation_equivariance_error", equivariance_error, 1e-6)

    # Contract 3: finite gradients through a non-symmetric 576x576 assignment.
    # Use a deterministic low-amplitude perturbation rather than random targets.
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1)
    grad_logits = (0.01 * torch.randn((N_TILES, N_TILES), generator=generator, device=device)).requires_grad_(True)
    grad_soft = log_sinkhorn(grad_logits, SINKHORN_ITERS)
    weights = torch.linspace(-1.0, 1.0, N_TILES, device=device).unsqueeze(0)
    loss = (grad_soft * weights).square().mean()
    loss.backward()
    if grad_logits.grad is None or not torch.isfinite(grad_logits.grad).all():
        raise AssertionError("Sinkhorn gradient is missing or non-finite")
    grad_abs_max = float(grad_logits.grad.abs().max().detach().cpu())
    grad_row_error, grad_col_error = _matrix_errors(grad_soft)
    _assert_close("gradient_row_error", grad_row_error, 5e-4)
    _assert_close("gradient_col_error", grad_col_error, 5e-4)

    # Contract 4: inference always returns a 576-way valid bijection even when
    # logits are tied by construction.  The solver's deterministic tie handling
    # may choose any permutation, but must choose each slot exactly once.
    tied_logits = torch.zeros((N_TILES, N_TILES), device=device)
    tied_pred = decode_linear_assignment(tied_logits)
    tied_is_bijection = bool(np.unique(tied_pred).size == N_TILES)
    if not tied_is_bijection:
        raise AssertionError("tied-logit decoder is not a bijection")

    return {
        "experiment": "P10_sinkhorn_refiner",
        "gate": "G0a_synthetic_permutation_and_sinkhorn_contracts",
        "status": "PASS",
        "grid": GRID,
        "tiles": N_TILES,
        "sinkhorn_iterations": SINKHORN_ITERS,
        "device": str(device),
        "seed": seed,
        "identity_accuracy": identity_accuracy,
        "identity_row_error": id_row_error,
        "identity_col_error": id_col_error,
        "row_permutation_accuracy": permutation_accuracy,
        "row_permutation_equivariance_error": equivariance_error,
        "gradient_abs_max": grad_abs_max,
        "gradient_row_error": grad_row_error,
        "gradient_col_error": grad_col_error,
        "tied_decode_is_bijection": tied_is_bijection,
        "targets_opened": False,
        "cal_target_opened": False,
        "dev_targets_opened": False,
        "test_accessed": False,
        "p8_labels_imported": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    report = run_contracts(device, args.seed)
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
