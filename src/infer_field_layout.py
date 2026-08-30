"""Assemble the 24x24 puzzle with a bag-conditioned absolute field.

No restoration is performed here.  Output PNGs contain the original corrupted
20x20 fragments, moved once into the predicted cells.  This keeps placement and
restoration experimentally separate, as required by the project decision.

For each board the algorithm is:

1. draw ``samples`` coarse 96x96 photograph hypotheses conditioned on the
   unordered bag of 576 fragments;
2. optionally extract only the globally strongest seam edges as rigid islands;
3. solve a complete cell<->fragment bijection for every hypothesis;
4. select among that finite set, using realised seam energy only as a weak
   tie-break (never as a free global objective).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from config import CACHE_DIR, CKPT_DIR, GRID, TEST_DIR, TRAIN_INP, TRAIN_TGT
from field_diffusion import FieldDiffusion, Schedule, sample
from field_solver import (components_from_score_tail, select_layout,
                          solve_field)
from imgio import load as load_rgb
from seam_cost import costs_from_models
from seam_embed import SeamEmbed

FS = 20
N = GRID * GRID


def to_frags(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img.reshape(GRID, FS, GRID, FS, 3).transpose(
        0, 2, 1, 3, 4).reshape(N, FS, FS, 3))


def assemble(frags: np.ndarray, layout: np.ndarray) -> np.ndarray:
    a = frags[np.asarray(layout, np.int64)].reshape(
        GRID, GRID, FS, FS, 3).transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(a.reshape(GRID * FS, GRID * FS, 3))


def _path(root: Path, name: str) -> Path:
    p = Path(name)
    return p if p.is_absolute() else root / p


def load_field_model(path: Path, device: str):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    model = FieldDiffusion(a["d"], a["layers"], a["heads"], a["base"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, a


def load_matchers(names: list[str], device: str) -> list[SeamEmbed]:
    out = []
    for name in names:
        ck = torch.load(_path(Path(CKPT_DIR), name), map_location=device,
                        weights_only=False)
        a = ck["args"]
        model = SeamEmbed(a["ch"], a["blocks"], a["dim"], a["strip"],
                          a.get("head", "global"),
                          predict=any(k.startswith("pred.")
                                      for k in ck["model"])).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        out.append(model)
    return out


@torch.no_grad()
def predict_fields(model: FieldDiffusion, args: dict, tiles: np.ndarray,
                   count: int, steps: int, snap_from: float, seed: int,
                   device: str) -> list[np.ndarray]:
    x = torch.from_numpy(tiles.astype(np.float32))[None].to(device)
    mode = args.get("mode", "diffusion")
    if mode == "regress":
        ctx = model.bag(x)
        pred = model.net(torch.zeros(1, 3, 96, 96, device=device),
                         torch.zeros(1, dtype=torch.long, device=device), ctx)
        return [((pred[0].permute(1, 2, 0).float().cpu().numpy() + 1.0)
                 * 127.5).clip(0, 255)]

    sched = Schedule(int(args.get("steps", 1000)), device)
    fields = []
    for k in range(count):
        gen = torch.Generator(device=device).manual_seed(seed + k)
        pred = sample(model, x, sched, steps=steps,
                      frags=tiles[None] if snap_from <= 1.0 else None,
                      snap_from=snap_from, device=device, generator=gen)
        fields.append(((pred[0].permute(1, 2, 0).float().cpu().numpy() + 1.0)
                       * 127.5).clip(0, 255))
    return fields


def adjacency(layout: np.ndarray) -> float:
    a = np.asarray(layout).reshape(GRID, GRID)
    good = (a[:, 1:] == a[:, :-1] + 1).sum()
    good += (a[1:] == a[:-1] + GRID).sum()
    return float(good / (2 * GRID * (GRID - 1)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="field_diff.pt")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--snap-from", type=float, default=1.01,
                    help="late diffusion fraction for exact bag projection; "
                         "above 1 disables it")
    ap.add_argument("--mode", default="raw",
                    help="raw, zscore, or blend:X descriptor assignment")
    ap.add_argument("--matchers", nargs="*", default=[])
    ap.add_argument("--edge-keep", type=int, default=0,
                    help="globally strongest seam edges locked as islands; "
                         "127 is the measured ~98.5%% precision point")
    ap.add_argument("--beam", type=int, default=64)
    ap.add_argument("--offsets", type=int, default=96)
    ap.add_argument("--seam-weight", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--validate", type=int, default=8,
                    help="held-out boards; 0 runs test inference")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--test-dir", default=TEST_DIR)
    ap.add_argument("--out", default="field_layout_output")
    a = ap.parse_args()
    if a.edge_keep and not a.matchers:
        ap.error("--edge-keep requires --matchers")

    dev = a.device
    model, model_args = load_field_model(_path(Path(CKPT_DIR), a.checkpoint), dev)
    matchers = load_matchers(a.matchers, dev)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {k: getattr(a, k) for k in
              ("checkpoint", "samples", "sample_steps", "snap_from", "mode",
               "matchers", "edge_keep", "beam", "offsets", "seam_weight")}
    print(json.dumps(report), flush=True)

    if a.validate:
        z = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
        names = [str(x) for x in z["names"][-300:]][:a.validate]
        inv = z["inv"][-300:][:a.validate]
        rows = []
        source = [(name, Path(TRAIN_INP) / name, inv[k])
                  for k, name in enumerate(names)]
    else:
        paths = sorted(Path(a.test_dir).glob("*.png"))
        if a.limit:
            paths = paths[:a.limit]
        source = [(p.name, p, None) for p in paths]
        rows = None

    for board_no, (name, path, permutation) in enumerate(source):
        tiles = to_frags(load_rgb(str(path))).astype(np.float32)
        # Validation stores fragments in recovered true-cell order solely so
        # layout==arange has a meaning. The model is permutation invariant and
        # receives no position associated with this storage order.
        if permutation is not None:
            tiles = tiles[np.asarray(permutation, np.int64)]
        right = down = None
        components = []
        if matchers:
            right, down = costs_from_models(matchers, tiles)
            if a.edge_keep:
                components = components_from_score_tail(right, down, a.edge_keep)
        fields = predict_fields(model, model_args, tiles, a.samples,
                                a.sample_steps, a.snap_from,
                                a.seed + 1009 * board_no, dev)
        layouts, costs = [], []
        for field in fields:
            layout, value = solve_field(field, tiles, mode=a.mode,
                                        components=components, beam=a.beam,
                                        offsets=a.offsets)
            layouts.append(layout)
            costs.append(value)
        pick = select_layout(layouts, costs, right, down, a.seam_weight)
        layout = layouts[pick]
        image = assemble(tiles, layout).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(out / name), image[:, :, ::-1])
        np.save(out / f"{Path(name).stem}_layout.npy", layout)
        msg = (f"[{board_no + 1}/{len(source)}] {name} sample={pick} "
               f"components={len(components)}")
        if rows is not None:
            target = load_rgb(str(Path(TRAIN_TGT) / name))
            place = float((layout == np.arange(N)).mean())
            adj = adjacency(layout)
            score = float(ssim(image, target, channel_axis=2, data_range=255))
            rows.append((place, adj, score))
            msg += f" place={place:.4f} adjacency={adj:.4f} SSIM={score:.4f}"
        print(msg, flush=True)

    if rows:
        mean = np.mean(rows, axis=0)
        print(f"MEAN place={mean[0]:.4f} adjacency={mean[1]:.4f} "
              f"solve_SSIM={mean[2]:.4f}")


if __name__ == "__main__":
    main()

