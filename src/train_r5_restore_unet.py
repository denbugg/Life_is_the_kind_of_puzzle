"""R5 capacity gate: source-disjoint, corruption-matched MS-SSIM U-Net restorer.

The model only changes RGB pixels after a layout.  It never scores seams or
predicts permutations.  Training consumes clean FIT targets and applies the
already audited independent-tile distortion generator; evaluation uses fixed
corruptions and clean targets only for post-hoc SSIM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from skimage.metrics import structural_similarity as sk_ssim

from config import CKPT_DIR, TRAIN_TGT
from distort import distort_frags
from imgio import from_frags, load, to_frags
from match_preprocess import apply_match_denoiser_np, load_match_denoiser
from models import RestoreNet, restore_loss, count_params

DEFAULT_SPLIT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json")
DEFAULT_WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R5_restore_unet")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def corrupt(clean: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return from_frags(distort_frags(to_frags(clean), rng))


def crop_pair(clean: np.ndarray, dirty: np.ndarray, rng: np.random.Generator, crop: int) -> tuple[np.ndarray, np.ndarray]:
    if crop % 20:
        raise ValueError("crop must preserve whole 20px tiles")
    width = clean.shape[0]
    grid = width // 20
    cells = crop // 20
    top = int(rng.integers(0, grid - cells + 1)) * 20
    left = int(rng.integers(0, grid - cells + 1)) * 20
    return clean[top:top + crop, left:left + crop], dirty[top:top + crop, left:left + crop]


def tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(device)


def image(t: torch.Tensor) -> np.ndarray:
    value = t.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    return np.rint(value * 255.0).clip(0, 255).astype(np.uint8)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(sk_ssim(a, b, channel_axis=2, data_range=255))


def evaluate(model: torch.nn.Module, names: list[str], seeds: list[int], crop: int, device: torch.device, matchden: Any) -> tuple[list[dict[str, Any]], dict[str, float]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for name, value in zip(names, seeds):
            clean = load(Path(TRAIN_TGT) / name)
            dirty = corrupt(clean, value)
            if crop < 480:
                clean, dirty = crop_pair(clean, dirty, np.random.default_rng(value + 991), crop)
            pred = image(model(tensor(dirty, device)))
            match = from_frags(apply_match_denoiser_np(to_frags(dirty), matchden, device=str(device)))
            rows.append({
                "name": name,
                "seed": value,
                "dirty_ssim": ssim(dirty, clean),
                "matchden_ssim": ssim(match, clean),
                "r5_ssim": ssim(pred, clean),
            })
    for row in rows:
        row["r5_vs_dirty"] = row["r5_ssim"] - row["dirty_ssim"]
        row["r5_vs_matchden"] = row["r5_ssim"] - row["matchden_ssim"]
    result = {key: float(np.mean([row[key] for row in rows])) for key in ("dirty_ssim", "matchden_ssim", "r5_ssim", "r5_vs_dirty", "r5_vs_matchden")}
    return rows, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--fit-n", type=int, default=2)
    parser.add_argument("--eval-n", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--crop", type=int, default=240)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=2501)
    parser.add_argument("--denoise-tag", default="matchden")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    if args.fit_n < 2 or args.eval_n < 2:
        parser.error("R5 capacity gate requires at least two FIT examples")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("R5 is restricted to the local CUDA GPU")
    if args.crop % 20 or args.crop < 160:
        parser.error("crop must be a whole-tile multiple and at least 160")
    seed_everything(args.seed)
    device = torch.device(args.device)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    fit = list(split["splits"]["fit"][:args.fit_n])
    evaluation_names = fit[:args.eval_n]
    if len(fit) < args.fit_n:
        raise RuntimeError("not enough FIT names")
    model = RestoreNet(base=args.base, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    denoiser, meta = load_match_denoiser(args.denoise_tag, device=args.device)
    if denoiser is None:
        raise FileNotFoundError("frozen MatchDenoiser checkpoint not available")
    denoiser_path = Path(CKPT_DIR) / f"{args.denoise_tag}_best.pt"
    if not denoiser_path.is_file():
        raise FileNotFoundError(denoiser_path)

    fixed_seeds = [args.seed + 40_000 + index for index in range(args.eval_n)]
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, Any]] = []
    torch.backends.cudnn.benchmark = True
    for step in range(1, args.steps + 1):
        name = fit[(step - 1) % len(fit)]
        clean_full = load(Path(TRAIN_TGT) / name)
        dirty_full = corrupt(clean_full, args.seed * 100_000 + step)
        clean_crop, dirty_crop = crop_pair(clean_full, dirty_full, rng, args.crop)
        target = tensor(clean_crop, device)
        source = tensor(dirty_crop, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(source)
            loss = restore_loss(output, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % max(1, args.steps // 8) == 0 or step == args.steps:
            rows, summary = evaluate(model, evaluation_names, fixed_seeds, args.crop, device, denoiser)
            record = {"step": step, "loss": float(loss.detach()), **summary}
            history.append(record)
            print(json.dumps(record), flush=True)
    rows, summary = evaluate(model, evaluation_names, fixed_seeds, args.crop, device, denoiser)
    capacity_pass = bool(summary["r5_vs_matchden"] > 0 and summary["r5_vs_dirty"] > 0)
    report = {
        "experiment": "R5_source_disjoint_MS_SSIM_UNet_capacity",
        "scope": "FIT-only two-scene capacity control; target only for post-hoc SSIM; no test access; no layout changes",
        "split": str(args.split),
        "split_sha256": sha256(args.split),
        "fit_names": fit,
        "fixed_eval_names": evaluation_names,
        "args": vars(args) | {"split": str(args.split), "work": str(args.work), "report": str(args.report) if args.report else None, "checkpoint": str(args.checkpoint) if args.checkpoint else None},
        "model_params": int(count_params(model)),
        "matchden_checkpoint": {"path": str(denoiser_path), "sha256": sha256(denoiser_path), "metadata_type": type(meta).__name__},
        "history": history,
        "rows": rows,
        "summary": summary,
        "gate": {"condition": "mean R5 SSIM greater than frozen MatchDenoiser and dirty input on fixed FIT corruptions", "passed": capacity_pass, "decision": "advance_R5_to_source_disjoint_DEV" if capacity_pass else "reject_R5_before_global_training"},
    }
    destination = args.report or args.work / "r5_capacity_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    checkpoint = args.checkpoint or args.work / "r5_capacity.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "report": report}, checkpoint)
    print(json.dumps({"summary": summary, "gate": report["gate"], "report": str(destination)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
