"""Test-set inference: restore -> MGC -> solve_loop -> fix_origin -> NLM -> ZIP.

This is the deployment form of the chain verified at place_acc 0.9965 on clean
tiles.  Nothing here reads a target; the only inputs are the 700 test images and
the frozen restorer checkpoint.

Assembly quality depends entirely on the restorer: the chain needs MGC R@1
around 0.47 before placement rises above chance, so run --gate first on
held-out train boards and do not build a submission below that.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from config import CKPT_DIR, GRID as G, NFRAG as N, TEST_DIR
from mgc import mgc_cost
from restore_tile import TileRestorer, to_frags
from solve_loop import solve as solve_loop
from torus_origin import fix_origin


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.shape != (G * 20, G * 20, 3):
        raise RuntimeError(f"bad input: {path}")
    return np.ascontiguousarray(img[:, :, ::-1])


def assemble(tiles: np.ndarray, board: np.ndarray) -> np.ndarray:
    f = np.clip(tiles[np.asarray(board)], 0, 255).astype(np.uint8)
    return f.reshape(G, G, 20, 20, 3).transpose(0, 2, 1, 3, 4).reshape(480, 480, 3)


@torch.no_grad()
def restore(model, tiles: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(tiles).permute(0, 3, 1, 2).to(device)
    with torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
        out = torch.cat([model(x[i:i + 288]) for i in range(0, len(x), 288)])
    return out.float().permute(0, 2, 3, 1).clamp(0, 255).cpu().numpy()


def solve_board(scored: np.ndarray) -> np.ndarray:
    board, _ = solve_loop(mgc_cost(scored, "h"), mgc_cost(scored, "v"))
    return fix_origin(board, scored, metric="mgc")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=Path(CKPT_DIR) / "rest_mgc_big.pt")
    ap.add_argument("--input-dir", type=Path, default=Path(TEST_DIR))
    ap.add_argument("--out-dir", type=Path, default=Path(r"E:\pazzle_work\submissions\mgc_chain\png"))
    ap.add_argument("--out-zip", type=Path, default=Path(r"E:\pazzle_work\submissions\mgc_chain\submission_mgc_chain.zip"))
    ap.add_argument("--expected", type=int, default=700)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = TileRestorer(ck["args"]["ch"], ck["args"]["blocks"],
                         ck["args"].get("residual", False)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"restorer {args.ckpt.name}: step {ck['step']} gate {ck['bb_prec']:.4f}", flush=True)

    names = sorted(p.name for p in args.input_dir.glob("*.png"))
    if args.expected and len(names) != args.expected:
        raise RuntimeError(f"expected {args.expected} inputs, found {len(names)}")
    todo = names[: args.limit] if args.limit else names
    args.out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    for i, nm in enumerate(todo, 1):
        tiles = to_frags(load_rgb(args.input_dir / nm)).astype(np.float32)
        board = solve_board(restore(model, tiles, device))
        # assemble from RAW pixels: the restorer is tuned for matching, and its
        # output is measurably worse than the input as an image (M23).
        final = cv2.fastNlMeansDenoisingColored(assemble(tiles, board), None, 10, 10, 7, 21)
        Image.fromarray(final, mode="RGB").save(args.out_dir / nm, format="PNG")
        if i % 25 == 0:
            rate = i / (time.perf_counter() - started)
            print(f"  {i}/{len(todo)}  {rate:.2f} img/s  eta {(len(todo)-i)/rate/60:.1f} min", flush=True)

    if args.limit:
        print("partial run: no ZIP written", flush=True)
        return
    tmp = args.out_zip.with_suffix(".tmp")
    args.out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for nm in names:
            info = zipfile.ZipInfo(nm, date_time=(2026, 8, 18, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            z.writestr(info, (args.out_dir / nm).read_bytes(),
                       compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    os.replace(tmp, args.out_zip)
    digest = hashlib.sha256(args.out_zip.read_bytes()).hexdigest()
    print(f"zip {args.out_zip}  {args.out_zip.stat().st_size/1e6:.1f} MB  sha256 {digest}", flush=True)


if __name__ == "__main__":
    main()
