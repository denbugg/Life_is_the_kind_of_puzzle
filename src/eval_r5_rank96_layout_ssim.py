"""R5 phase-2: RestoreNet SSIM on frozen rank96 layouts.

Rank96 placement is inferred exclusively from the corrupted training mosaics by
unchanged champion components.  The clean target is read only after placement
for post-hoc SSIM.  This script changes pixels only; it cannot change tile
assignment.  Test data and submission writing are deliberately absent.
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

from config import TRAIN_INP, TRAIN_TGT
from models import RestoreNet
import infer_rank96 as rank96

DEFAULT_WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R5_restore_unet")
DEFAULT_SPLIT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json")
DEFAULT_CKPT = DEFAULT_WORK / "r5_capacity_fp32.pt"


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
    # R5 saves an explicit payload, while fallback branches preserve compatibility
    # with a plain state_dict checkpoint if the harness is upgraded later.
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


def restore_tiles(tiles: np.ndarray, model: RestoreNet, device: str, batch: int) -> np.ndarray:
    if tiles.dtype != np.uint8 or tiles.ndim != 4 or tiles.shape[-1] != 3:
        raise ValueError(f"expected uint8 tiles NHWC, got {tiles.shape} {tiles.dtype}")
    result: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tiles), batch):
            chunk = torch.from_numpy(tiles[start : start + batch]).to(device=device, dtype=torch.float32)
            chunk = chunk.permute(0, 3, 1, 2).div_(255.0)
            restored = model(chunk).clamp_(0.0, 1.0)
            restored = restored.permute(0, 2, 3, 1).cpu().numpy()
            result.append(np.rint(restored * 255.0).clip(0, 255).astype(np.uint8))
    return np.concatenate(result, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--partition", choices=("cal", "dev"), default="dev")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pair-batch", type=int, default=128)
    parser.add_argument("--tile-batch", type=int, default=64)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    if args.n < 2:
        parser.error("--n must be at least two for the registered lower-confidence bound")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"R5 checkpoint unavailable: {args.checkpoint}")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    names = list(split["splits"][args.partition][: args.n])
    if len(names) != args.n:
        raise RuntimeError("requested split size unavailable")

    args.work.mkdir(parents=True, exist_ok=True)
    config, champion_paths = build_config(args.device, args.pair_batch, args.work)
    models = rank96.load_models(config)
    r5 = load_r5(args.checkpoint, args.device, args.base, args.depth)
    rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    for ordinal, name in enumerate(names, 1):
        dirty_image = rank96.load_rgb_strict(Path(TRAIN_INP) / name)
        target = rank96.load_rgb_strict(Path(TRAIN_TGT) / name)
        inferred = rank96.infer_one(dirty_image, models, pair_batch=args.pair_batch)
        dirty_tiles = rank96.split_upright_tiles(dirty_image)
        restored_tiles = restore_tiles(dirty_tiles, r5, args.device, args.tile_batch)
        raw_layout = rank96.assemble_upright_tiles(dirty_tiles, inferred.board)
        restored_layout = rank96.assemble_upright_tiles(restored_tiles, inferred.board)
        raw_ssim = float(ssim(raw_layout, target, channel_axis=2, data_range=255))
        restored_ssim = float(ssim(restored_layout, target, channel_axis=2, data_range=255))
        row = {
            "name": name,
            "rank96_objective": float(inferred.objective),
            "raw_layout_ssim": raw_ssim,
            "restored_layout_ssim": restored_ssim,
            "delta": restored_ssim - raw_ssim,
            "board_sha256": rank96.sha256_array(inferred.board.astype(np.int16)),
            "candidate_ids_sha256": inferred.candidate_ids_sha256,
            "raw_scores_sha256": inferred.raw_scores_sha256,
        }
        rows.append(row)
        deltas.append(row["delta"])
        print(json.dumps({"ordinal": ordinal, **row}), flush=True)

    summary = {
        "raw_layout_ssim_mean": float(np.mean([row["raw_layout_ssim"] for row in rows])),
        "restored_layout_ssim_mean": float(np.mean([row["restored_layout_ssim"] for row in rows])),
        "mean_delta": float(np.mean(deltas)),
        "min_delta": float(np.min(deltas)),
        "lower_95_delta": lower_95(deltas),
    }
    passed = bool(summary["mean_delta"] > 0 and summary["lower_95_delta"] > 0)
    report = {
        "experiment": "R5_RestoreNet_on_frozen_rank96_layout",
        "scope": "source-disjoint held-out evaluation; rank96 layout uses input only; RestoreNet changes pixels only; target used only for SSIM; no test access",
        "split": str(args.split),
        "split_sha256": file_sha256(args.split),
        "partition": args.partition,
        "names": names,
        "args": vars(args) | {"split": str(args.split), "checkpoint": str(args.checkpoint), "work": str(args.work)},
        "r5_checkpoint": {"path": str(args.checkpoint), "sha256": file_sha256(args.checkpoint)},
        "rank96_checkpoints": {key: {"path": str(value), "sha256": file_sha256(value)} for key, value in champion_paths.items()},
        "rows": rows,
        "summary": summary,
        "gate": {
            "condition": "mean restoration SSIM delta>0 and lower_95_delta>0 on unchanged frozen rank96 layouts",
            "passed": passed,
            "decision": "advance_R5_to_R4_comparison" if passed else "reject_R5_for_rank96_composition",
        },
    }
    destination = args.report or args.work / f"r5_rank96_layout_{args.partition}{args.n}.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "gate": report["gate"], "report": str(destination)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
