"""ORBIT-24 P10 G0b: one-FIT source canonical-layout contract.

The harness is intentionally non-training. It proves that the planned P10 model
can consume the frozen P9 canonical rank96 graph, a canonical buddies board, and
corrupted FIT tiles while producing a valid 576-way Sinkhorn/linear assignment.
Only one explicitly named FIT target may be read to reconstruct P9's approved
corruption state. CAL, DEV, test, P8 labels, and P8 scores are forbidden.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

THIS = Path(__file__).resolve().parent
if str(THIS) not in sys.path:
    sys.path.insert(0, str(THIS))

from p10_sinkhorn_contracts import (  # noqa: E402
    GRID,
    N_TILES,
    SINKHORN_ITERS,
    decode_linear_assignment,
    log_sinkhorn,
)
from p9_rank96_loop_g1 import (  # noqa: E402
    SEED as P9_SEED,
    canonical_dense_rd,
    distort_frags,
    to_frags,
)
from solve_buddies import solve_buddies_from_scores  # noqa: E402

TILE = 20


def coordinate_fourier(slot_ids: torch.Tensor) -> torch.Tensor:
    """Fourier coordinate features for flattened 24x24 canonical slots."""
    if slot_ids.ndim != 1:
        raise ValueError(f"expected [N] slot IDs, received {tuple(slot_ids.shape)}")
    slot_ids = slot_ids.long()
    row = torch.div(slot_ids, GRID, rounding_mode="floor").float() / float(GRID - 1)
    col = torch.remainder(slot_ids, GRID).float() / float(GRID - 1)
    xy = torch.stack((row, col), dim=-1)
    freq = (2.0 ** torch.arange(4, device=slot_ids.device, dtype=torch.float32)) * torch.pi
    phase = xy.unsqueeze(-1) * freq
    return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1).flatten(start_dim=1)


class LayoutConditionedSinkhornRefiner(nn.Module):
    """Small P10 architecture used only to exercise the declared data path at G0b."""

    def __init__(self, width: int = 64, heads: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.tile_encoder = nn.Sequential(
            nn.Conv2d(3, width // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 2, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.token_projection = nn.Sequential(
            nn.LayerNorm(width + 16),
            nn.Linear(width + 16, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=layers)
        self.slot_projection = nn.Sequential(
            nn.Linear(16, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.slot_bias = nn.Parameter(torch.zeros(N_TILES, width))

    def forward(self, tiles: torch.Tensor, initial_tile_to_slot: torch.Tensor) -> torch.Tensor:
        if tiles.shape != (N_TILES, 3, TILE, TILE):
            raise ValueError(f"expected tiles {(N_TILES, 3, TILE, TILE)}, got {tuple(tiles.shape)}")
        if initial_tile_to_slot.shape != (N_TILES,):
            raise ValueError("expected one initial canonical slot per input tile")
        if torch.unique(initial_tile_to_slot).numel() != N_TILES:
            raise ValueError("initial canonical layout is not bijective")
        tile_features = self.tile_encoder(tiles)
        observed_position = coordinate_fourier(initial_tile_to_slot)
        token = self.token_projection(torch.cat((tile_features, observed_position), dim=-1))
        token = self.context(token.unsqueeze(0)).squeeze(0)
        all_slots = torch.arange(N_TILES, device=tiles.device)
        slot_features = self.slot_projection(coordinate_fourier(all_slots)) + self.slot_bias
        return token @ slot_features.T / float(token.shape[-1]) ** 0.5


def load_clean_rgb(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    expected = (GRID * TILE, GRID * TILE, 3)
    if image.shape != expected:
        raise ValueError(f"expected FIT target image {expected}, got {image.shape}")
    return image


def validate_cache(cache_path: Path, source: str) -> dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as z:
        required = {"anchors", "directions", "members", "baseline", "candidates", "scores", "permutation", "source"}
        missing = required.difference(z.files)
        if missing:
            raise RuntimeError(f"P9 cache lacks keys: {sorted(missing)}")
        cache = {key: z[key].copy() for key in required}
    cache_source = str(cache["source"].item())
    if cache_source != source:
        raise RuntimeError(f"cache source mismatch: {cache_source!r} != {source!r}")
    if cache["candidates"].shape[0] != N_TILES or cache["scores"].shape[:2] != (4, N_TILES):
        raise RuntimeError(f"unexpected P9 graph shapes {cache['candidates'].shape}, {cache['scores'].shape}")
    permutation = cache["permutation"].astype(np.int64, copy=False)
    if permutation.shape != (N_TILES,) or np.unique(permutation).size != N_TILES:
        raise RuntimeError("P9 permutation is not a 576-way bijection")
    return cache


def tile_to_slot_from_board(board: np.ndarray) -> np.ndarray:
    flat = np.asarray(board, dtype=np.int64).reshape(-1)
    if flat.shape != (N_TILES,) or np.unique(flat).size != N_TILES:
        raise RuntimeError("canonical buddies solver returned a non-bijective board")
    tile_to_slot = np.empty(N_TILES, dtype=np.int64)
    tile_to_slot[flat] = np.arange(N_TILES, dtype=np.int64)
    return tile_to_slot


def run_g0b(source: str, source_index: int, cache_path: Path, targets: Path, device: torch.device, seed: int) -> dict[str, object]:
    cache = validate_cache(cache_path, source)

    # This is the sole target read. The source is assumed to be FIT-only and its
    # frozen P9 cache supplies the fixed corruption permutation/state.
    clean = load_clean_rgb(targets / source)
    fragments = distort_frags(to_frags(clean), np.random.default_rng(P9_SEED * 1009 + source_index))
    tiles_np = fragments[cache["permutation"]]
    if tiles_np.shape != (N_TILES, TILE, TILE, 3):
        raise RuntimeError(f"unexpected reconstructed tile shape {tiles_np.shape}")

    # The initial spatial hypothesis must be reconstructed only from frozen P9
    # candidates/scores. No mining, ranker evaluation, P8 signal, or target label
    # is used after the approved FIT tile reconstruction above.
    right, down = canonical_dense_rd(cache["candidates"], cache["scores"])
    board, objective = solve_buddies_from_scores(right, down, max_edges=96)
    initial_tile_to_slot_np = tile_to_slot_from_board(board)
    rank96_placement_accuracy = float(np.mean(initial_tile_to_slot_np == cache["permutation"]))

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LayoutConditionedSinkhornRefiner().to(device).eval()
    tiles = torch.from_numpy(tiles_np).permute(0, 3, 1, 2).contiguous().float().div_(255.0).to(device)
    initial_slots = torch.from_numpy(initial_tile_to_slot_np).to(device)
    with torch.no_grad():
        logits = model(tiles, initial_slots)
        soft = log_sinkhorn(logits, SINKHORN_ITERS)
    decoded_tile_to_slot = decode_linear_assignment(logits)

    row_error = float((soft.sum(dim=-1) - 1.0).abs().max().cpu())
    col_error = float((soft.sum(dim=-2) - 1.0).abs().max().cpu())
    if row_error > 1e-3 or col_error > 1e-3:
        raise AssertionError(f"Sinkhorn contract failed: row={row_error}, col={col_error}")
    if np.unique(decoded_tile_to_slot).size != N_TILES:
        raise AssertionError("P10 discrete decoder did not produce a bijection")

    return {
        "experiment": "P10_sinkhorn_refiner",
        "gate": "G0b_one_FIT_canonical_layout_contract",
        "status": "PASS",
        "source": source,
        "source_index": source_index,
        "cache": str(cache_path),
        "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "device": str(device),
        "seed": seed,
        "tiles_shape": list(tiles_np.shape),
        "p9_permutation_is_bijection": True,
        "initial_rank96_board_is_bijection": True,
        "initial_rank96_placement_accuracy": rank96_placement_accuracy,
        "initial_rank96_objective": float(objective),
        "logits_shape": list(logits.shape),
        "sinkhorn_iterations": SINKHORN_ITERS,
        "sinkhorn_row_error": row_error,
        "sinkhorn_col_error": col_error,
        "p10_discrete_decode_is_bijection": True,
        "fit_target_opened": True,
        "cal_target_opened": False,
        "dev_targets_opened": False,
        "test_accessed": False,
        "p8_labels_imported": False,
        "p8_scores_imported": False,
        "rank96_mining_invoked": False,
        "rank96_ranker_invoked": False,
        "model_trained": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="img_003194.png")
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    report = run_g0b(args.source, args.source_index, args.cache, args.targets, torch.device(args.device), args.seed)
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
