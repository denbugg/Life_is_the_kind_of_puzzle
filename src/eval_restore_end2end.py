"""End-to-end gate: shuffled board -> restore -> score -> solve -> SSIM.

Everything upstream is measured on tiles held in TRUE grid order, which is a
diagnostic convenience, not the deployment condition.  This runs the actual
pipeline on the SHUFFLED input exactly as test data arrives, so its numbers are
directly comparable to the platform score (current S1 benchmark 0.23748526).

Reports, per arm: placement accuracy, raw-layout SSIM and SSIM after the
canonical NLM h=10, against the same held-out boards.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as sk_ssim

from config import CACHE_DIR, CKPT_DIR, GRID as G, NFRAG as N, TRAIN_INP, TRAIN_TGT, VAL_COUNT
from restore_tile import TileRestorer, ridge_cost, to_frags
from mgc import mgc_cost
from solve_loop import solve as solve_loop
from torus_origin import fix_origin


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"unreadable: {path}")
    return np.ascontiguousarray(img[:, :, ::-1])


def assemble(frags: np.ndarray, order: np.ndarray) -> np.ndarray:
    f = frags[np.asarray(order)]
    return f.reshape(G, G, f.shape[1], f.shape[2], 3).transpose(0, 2, 1, 3, 4).reshape(
        G * f.shape[1], G * f.shape[2], 3)


def nlm(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(sk_ssim(a, b, channel_axis=2, data_range=255))


@torch.no_grad()
def restore_tiles(model, tiles: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(tiles).permute(0, 3, 1, 2).to(device)
    with torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
        out = torch.cat([model(x[i:i + 288]) for i in range(0, len(x), 288)])
    return out.float().permute(0, 2, 3, 1).clamp(0, 255).cpu().numpy()


def run_arm(tiles_for_scoring: np.ndarray, tiles_for_pixels: np.ndarray,
            target: np.ndarray, inv: np.ndarray, w: float, cols: int,
            repair_passes: int, metric: str = "mgc") -> dict[str, float]:
    """Score -> solve -> restore the absolute origin -> assemble.

    On clean tiles this exact chain reaches place_acc 0.9965.  Two rules it
    encodes, both measured: MGC beats the ridge cost by a wide margin as a
    compatibility measure (bb_prec 0.994 vs 0.796 on clean tiles), and the
    origin cut must be scored with the SAME measure that produced the layout
    (a ridge cut on an MGC board recovered 0.6655 instead of 0.9965).
    """
    if metric == "mgc":
        ch, cv_ = mgc_cost(tiles_for_scoring, "h"), mgc_cost(tiles_for_scoring, "v")
    else:
        ch = ridge_cost(tiles_for_scoring, w, cols, "h")
        cv_ = ridge_cost(tiles_for_scoring, w, cols, "v")
    place, _ = solve_loop(ch, cv_)
    place = fix_origin(place, tiles_for_scoring, w, cols, metric)
    px = np.clip(tiles_for_pixels, 0, 255).astype(np.uint8)
    raw = assemble(px, place)
    return {"place_acc": float(np.mean(place == inv)),
            "ssim_raw": ssim(raw, target),
            "ssim_nlm": ssim(nlm(raw), target)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=Path(CACHE_DIR) / "restore_labels.npz")
    ap.add_argument("--ckpt", type=Path, default=Path(CKPT_DIR) / "tile_restorer_seam.pt")
    ap.add_argument("--boards", type=int, default=12)
    ap.add_argument("--ridge-w", type=float, default=0.03)
    ap.add_argument("--ridge-cols", type=int, default=3)
    ap.add_argument("--repair-passes", type=int, default=0)
    ap.add_argument("--metric", choices=("mgc", "ridge"), default="mgc")
    args = ap.parse_args()

    blob = np.load(args.labels, allow_pickle=True)
    names, inv_all = blob["names"], blob["inv"]
    n_val = min(VAL_COUNT, len(names) // 4)
    va = slice(len(names) - n_val, len(names))
    names, inv_all = names[va], inv_all[va]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = TileRestorer(ck["args"]["ch"], ck["args"]["blocks"],
                         ck["args"].get("residual", False)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"restorer: step {ck['step']}  gate bb_prec {ck['bb_prec']:.4f}", flush=True)

    rows: dict[str, list[dict[str, float]]] = {}
    started = time.perf_counter()
    for k in range(args.boards):
        nm = str(names[k])
        shuffled = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)
        target = load_rgb(Path(TRAIN_TGT) / nm)
        inv = inv_all[k].astype(np.int64)
        restored = restore_tiles(model, shuffled, device)
        for tag, score_src, pixel_src in (
                ("baseline_raw", shuffled, shuffled),
                ("restored_score_rawpx", restored, shuffled),
                ("restored_both", restored, restored)):
            rows.setdefault(tag, []).append(
                run_arm(score_src, pixel_src, target, inv, args.ridge_w, args.ridge_cols,
                        args.repair_passes, args.metric))
        print(f"  {k+1}/{args.boards} {nm}", flush=True)

    print(f"\n{'arm':24s} {'place_acc':>10} {'ssim_raw':>10} {'ssim_nlm':>10}")
    for tag, vals in rows.items():
        g = lambda key: float(np.mean([v[key] for v in vals]))
        print(f"{tag:24s} {g('place_acc'):10.4f} {g('ssim_raw'):10.4f} {g('ssim_nlm'):10.4f}")
    print(f"\nplatform S1 benchmark (rank96+R5+NLM): 0.23748526")
    print(f"elapsed {(time.perf_counter()-started)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
