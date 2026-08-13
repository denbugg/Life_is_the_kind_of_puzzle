"""Paired R4/R5 comparison on exactly the same frozen rank96 layouts.

For every source-disjoint DEV board, rank96 infers one assignment from input
only.  Raw pixels, R4 tiled MatchDenoiser pixels, and R5 full-layout RestoreNet
pixels are compared against the clean target only after that assignment is
fixed.  There is no test access or submission writing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from config import CKPT_DIR, TRAIN_INP, TRAIN_TGT
from match_preprocess import apply_match_denoiser_np, load_match_denoiser
from models import RestoreNet
import infer_rank96 as rank96

DEFAULT_WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R5_restore_unet")
DEFAULT_SPLIT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json")
DEFAULT_R5 = DEFAULT_WORK / "r5_capacity_fp32.pt"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def lower_95(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return float(arr[0]) if len(arr) else float("nan")
    return float(arr.mean() - 1.96 * arr.std(ddof=1) / math.sqrt(len(arr)))


def build_config(device: str, pair_batch: int, work: Path) -> tuple[Any, dict[str, Path]]:
    paths = rank96._default_checkpoints()
    config = rank96.InferenceConfig(
        input_dir=Path(TRAIN_INP),
        output_dir=work / "rank96_unused_outputs",
        output_zip=None,
        ranker_checkpoint=paths["ranker"],
        affinity_primary_checkpoint=paths["affinity_primary"],
        affinity_secondary_checkpoint=paths["affinity_secondary"],
        device=device,
        pair_batch=pair_batch,
        expected_count=700,
    )
    return config, paths


def load_r5(path: Path, device: str, base: int, depth: int) -> RestoreNet:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if isinstance(payload, dict):
        state = payload.get("model") or payload.get("model_state_dict") or payload.get("state_dict") or payload
    else:
        state = payload
    model = RestoreNet(base=base, depth=depth).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def restore_r5_layout(layout: np.ndarray, model: RestoreNet, device: str) -> np.ndarray:
    if layout.dtype != np.uint8 or layout.ndim != 3 or layout.shape[-1] != 3:
        raise ValueError(f"expected uint8 HWC layout, got {layout.shape} {layout.dtype}")
    height, width = layout.shape[:2]
    if height % 8 or width % 8:
        raise ValueError(f"RestoreNet depth-4 requires dimensions divisible by 8, got {layout.shape}")
    with torch.no_grad():
        source = torch.from_numpy(layout).to(device=device, dtype=torch.float32)
        source = source.permute(2, 0, 1).unsqueeze(0).div_(255.0)
        restored = model(source).clamp_(0.0, 1.0)
        output = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.rint(output * 255.0).clip(0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--partition", choices=("cal", "dev"), default="dev")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pair-batch", type=int, default=128)
    parser.add_argument("--r5-checkpoint", type=Path, default=DEFAULT_R5)
    parser.add_argument("--r5-base", type=int, default=32)
    parser.add_argument("--r5-depth", type=int, default=4)
    parser.add_argument("--denoise-tag", default="matchden")
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    if args.n < 2:
        parser.error("--n must be at least two for lower-confidence comparisons")
    if not args.r5_checkpoint.is_file():
        raise FileNotFoundError(f"R5 checkpoint unavailable: {args.r5_checkpoint}")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    names = list(split["splits"][args.partition][: args.n])
    if len(names) != args.n:
        raise RuntimeError("requested split size unavailable")

    args.work.mkdir(parents=True, exist_ok=True)
    config, champion_paths = build_config(args.device, args.pair_batch, args.work)
    resolved_device = rank96.resolve_device(args.device)
    models = rank96.load_models(config, resolved_device)
    r5 = load_r5(args.r5_checkpoint, str(resolved_device), args.r5_base, args.r5_depth)
    denoiser, denoiser_meta = load_match_denoiser(args.denoise_tag, device=args.device)
    if denoiser is None:
        raise FileNotFoundError("frozen MatchDenoiser checkpoint unavailable")
    denoiser_path = Path(CKPT_DIR) / f"{args.denoise_tag}_best.pt"

    rows: list[dict[str, Any]] = []
    r4_deltas: list[float] = []
    r5_deltas: list[float] = []
    r5_minus_r4: list[float] = []
    for ordinal, name in enumerate(names, 1):
        dirty_image = rank96.load_rgb_strict(Path(TRAIN_INP) / name)
        target = rank96.load_rgb_strict(Path(TRAIN_TGT) / name)
        inferred = rank96.infer_one(dirty_image, models, pair_batch=args.pair_batch)
        dirty_tiles = rank96.split_upright_tiles(dirty_image)
        raw_layout = rank96.assemble_upright_tiles(dirty_tiles, inferred.board)
        r4_tiles = apply_match_denoiser_np(dirty_tiles, denoiser, device=args.device)
        r4_layout = rank96.assemble_upright_tiles(r4_tiles, inferred.board)
        r5_layout = restore_r5_layout(raw_layout, r5, str(resolved_device))
        raw_ssim = float(ssim(raw_layout, target, channel_axis=2, data_range=255))
        r4_ssim = float(ssim(r4_layout, target, channel_axis=2, data_range=255))
        r5_ssim = float(ssim(r5_layout, target, channel_axis=2, data_range=255))
        row = {
            "name": name,
            "rank96_objective": float(inferred.objective),
            "raw_layout_ssim": raw_ssim,
            "r4_layout_ssim": r4_ssim,
            "r5_layout_ssim": r5_ssim,
            "r4_delta": r4_ssim - raw_ssim,
            "r5_delta": r5_ssim - raw_ssim,
            "r5_minus_r4": r5_ssim - r4_ssim,
            "board_sha256": rank96.sha256_array(inferred.board.astype(np.int16)),
            "candidate_ids_sha256": inferred.candidate_ids_sha256,
            "raw_scores_sha256": inferred.raw_scores_sha256,
        }
        rows.append(row)
        r4_deltas.append(row["r4_delta"])
        r5_deltas.append(row["r5_delta"])
        r5_minus_r4.append(row["r5_minus_r4"])
        print(json.dumps({"ordinal": ordinal, **row}), flush=True)

    summary = {
        "raw_layout_ssim_mean": float(np.mean([row["raw_layout_ssim"] for row in rows])),
        "r4_layout_ssim_mean": float(np.mean([row["r4_layout_ssim"] for row in rows])),
        "r5_layout_ssim_mean": float(np.mean([row["r5_layout_ssim"] for row in rows])),
        "r4_delta_mean": float(np.mean(r4_deltas)),
        "r5_delta_mean": float(np.mean(r5_deltas)),
        "r4_lower_95_delta": lower_95(r4_deltas),
        "r5_lower_95_delta": lower_95(r5_deltas),
        "r5_minus_r4_mean": float(np.mean(r5_minus_r4)),
        "r5_minus_r4_min": float(np.min(r5_minus_r4)),
        "r5_minus_r4_lower_95": lower_95(r5_minus_r4),
    }
    r5_pass = bool(summary["r5_delta_mean"] > 0 and summary["r5_lower_95_delta"] > 0)
    r5_beats_r4 = bool(summary["r5_minus_r4_mean"] > 0 and summary["r5_minus_r4_lower_95"] > 0)
    report = {
        "experiment": "R5_vs_R4_paired_on_frozen_rank96_layout",
        "scope": "source-disjoint held-out paired comparison: exactly one input-only rank96 assignment per board; R4 and R5 modify pixels only; clean target is post-hoc SSIM only; no test access",
        "split": str(args.split),
        "split_sha256": file_sha256(args.split),
        "partition": args.partition,
        "names": names,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "r5_checkpoint": {"path": str(args.r5_checkpoint), "sha256": file_sha256(args.r5_checkpoint)},
        "r4_checkpoint": {"path": str(denoiser_path), "sha256": file_sha256(denoiser_path), "metadata": denoiser_meta},
        "rank96_checkpoints": {key: {"path": str(value), "sha256": file_sha256(value)} for key, value in champion_paths.items()},
        "rows": rows,
        "summary": summary,
        "gate": {
            "r5_post_layout_restoration_pass": r5_pass,
            "r5_strictly_beats_r4_pass": r5_beats_r4,
            "condition": "R5 delta mean/lower-95 > 0; replacement additionally requires paired R5-minus-R4 mean/lower-95 > 0",
            "decision": "retain_R5_as_stronger_restorer" if r5_beats_r4 else ("retain_R5_as_additional_restorer" if r5_pass else "reject_R5_for_rank96_composition"),
        },
    }
    destination = args.report or args.work / f"r5_vs_r4_rank96_{args.partition}{args.n}.json"
    destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"summary": summary, "gate": report["gate"], "report": str(destination)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
