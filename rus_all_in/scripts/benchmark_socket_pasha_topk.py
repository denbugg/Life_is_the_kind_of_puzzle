#!/usr/bin/env python3
"""Dirty-only runtime benchmark for bounded Pasha-on-Socket top-32 reranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image

from aiijc_puzzle.legacy_upgrade import layout_digest
from aiijc_puzzle.pasha883_pairwise import load_pasha883_pairwise
from aiijc_puzzle.protocol import IMAGE_SIZE, TILE_COUNT, sha256_file, split_tiles
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments
from aiijc_puzzle.socket_pasha_topk import (
    DEFAULT_TOP_K,
    decode_socket_with_pasha_topk_priority,
)
from aiijc_puzzle.socket_sorter_production import (
    choose_deterministic_device,
    load_socket_checkpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOCKET_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/socket_matcher.pt"
)
DEFAULT_PASHA_CHECKPOINT = PROJECT_ROOT / "artifacts/prior-pasha883/pair_best.pt"
GRID = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="one explicit dirty RGB 480x480 PNG; no target or label is read",
    )
    parser.add_argument("--socket-checkpoint", type=Path, default=DEFAULT_SOCKET_CHECKPOINT)
    parser.add_argument("--pasha-checkpoint", type=Path, default=DEFAULT_PASHA_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected RGB {IMAGE_SIZE}x{IMAGE_SIZE} PNG: {path}")
        return np.asarray(image, dtype=np.uint8)


def _strict_permutation(layout: np.ndarray) -> bool:
    value = np.asarray(layout)
    return bool(
        value.shape == (TILE_COUNT,)
        and np.array_equal(np.sort(value), np.arange(TILE_COUNT))
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    device = choose_deterministic_device(args.device)
    image = _load_rgb(args.input)
    tiles = split_tiles(image)
    socket = load_socket_checkpoint(args.socket_checkpoint, device=device)
    pasha = load_pasha883_pairwise(args.pasha_checkpoint, device=device)

    tensor = torch.from_numpy(tiles.astype(np.float32)).permute(0, 3, 1, 2).div_(255.0)
    matcher_started = perf_counter()
    socket_output = socket.model(tensor.unsqueeze(0).to(device), grid=GRID)
    right = socket_output.right_log_assignment[0].float().cpu().numpy()
    down = socket_output.down_log_assignment[0].float().cpu().numpy()
    matcher_seconds = perf_counter() - matcher_started
    decoder_config = SocketDecoderConfig(
        component_edge_budget_per_axis=144,
        swap_edge_budget_per_axis=144,
        max_swap_steps=24,
    )

    control_started = perf_counter()
    control = decode_socket_assignments(
        right,
        down,
        grid=GRID,
        config=decoder_config,
    )
    control_seconds = perf_counter() - control_started
    reranked = decode_socket_with_pasha_topk_priority(
        pasha.model,
        tiles,
        right,
        down,
        device=device,
        grid=GRID,
        top_k=DEFAULT_TOP_K,
        batch_size=args.batch_size,
        config=decoder_config,
    )
    if reranked.rerank.pair_evaluations != 2 * TILE_COUNT * DEFAULT_TOP_K:
        raise RuntimeError("bounded pair-evaluation invariant failed")
    if not _strict_permutation(control.layout) or not _strict_permutation(
        reranked.decoder.layout
    ):
        raise RuntimeError("decoder did not return a strict permutation")

    report = {
        "experiment": "socket-pasha-topk32-dirty-only-runtime-v1",
        "status": "benchmark-only-no-quality-or-promotion-claim",
        "input": {
            "path": str(args.input.resolve()),
            "sha256": sha256_file(args.input),
        },
        "socket_checkpoint": {
            "path": str(args.socket_checkpoint.resolve()),
            "sha256": socket.sha256,
            "architecture": socket.contract["architecture"],
            "dimension": socket.contract["dimension"],
        },
        "pasha_checkpoint": {
            "path": str(args.pasha_checkpoint.resolve()),
            "sha256": sha256_file(args.pasha_checkpoint),
            "architecture": "checkpoint-exact PairwiseNet C64 global-average-pooling",
            "step": pasha.step,
        },
        "protocol": {
            "dirty_pixels_only": True,
            "targets_or_reference_layouts_opened": False,
            "competition_test_opened": False,
            "socket_candidate_source": "partial-OT real block; top 32 per outgoing row",
            "pasha_vertical_contract": "transpose each 20x20 tile before 20x40 concat",
            "fusion": "fixed 50/50 within-top-32 row-rank percentiles",
            "self_masked": True,
            "unscored_pairs_masked": True,
            "decoder_use": (
                "fusion reprioritises only hard component edges; Socket OT matching, "
                "dustbins, border unary and QAP objective remain unchanged"
            ),
        },
        "rerank": reranked.rerank.report(),
        "runtime_seconds": {
            "socket_matcher": matcher_seconds,
            "control_decoder144": control_seconds,
            "pasha_topk32": reranked.rerank.pasha_seconds,
            "reranked_decoder144": reranked.decoder_seconds,
            "reranked_total_after_socket": (
                reranked.rerank.pasha_seconds + reranked.decoder_seconds
            ),
        },
        "outputs": {
            "control_layout_sha256": layout_digest(control.layout),
            "reranked_layout_sha256": layout_digest(reranked.decoder.layout),
            "control_strict_permutation": True,
            "reranked_strict_permutation": True,
        },
        "verdict": (
            "Runtime evidence only. The matched source-exposed full-score gate lost global "
            "adjacency, so this bounded approximation is not quality-evaluated or promoted."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
