"""Evaluate and submit restorer + border ranker + global solver."""
import json
import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import torch
import torch.nn.functional as F

from global_solver_candidate import solve_layout
from train_real_tile_restorer import FragmentRestorer
from train_restored_border_ranker import BorderRanker, descriptors, seam_features
from evaluate_real_noisy_student_assemblies import PositionPrior

GRID, TILE, N = 24, 20, 576
DATA_ROOT = Path(os.getenv("DATA_ROOT", "data/real/train"))
TEST_DIR = Path(os.getenv("TEST_DIR", "data/real/test"))
MAP_FILE = Path(os.getenv("MAP_FILE", "real_tile_maps.npz"))
RESTORER_CKPT = Path(os.getenv("RESTORER_CKPT", "outputs_real_restorer/real_fragment_restorer_best.pt"))
RANKER_CKPT = Path(os.getenv("RANKER_CKPT", "outputs_restored_border_ranker/border_ranker_best.pt"))
POS_CKPT = Path(os.getenv("POS_CKPT", "/home/kva/pazzle_assembly_diffusion_v3/outputs/position_prior_diffusion_v3_epoch4.pt"))
OUT_DIR = Path(os.getenv("OUT_DIR", "outputs_border_pipeline"))
MODE = os.getenv("MODE", "validation")
VAL_COUNT = int(os.getenv("VAL_COUNT", "20")); MAX_TEST = int(os.getenv("MAX_TEST", "0"))
TOPK = int(os.getenv("TOPK", "48")); SCORE_BATCH = int(os.getenv("SCORE_BATCH", "8192"))
SEED = int(os.getenv("SEED", "20260818"))


def split(path):
    x = np.asarray(Image.open(path).convert("RGB"), np.uint8)
    return x.reshape(GRID, TILE, GRID, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(N, TILE, TILE, 3)


def assemble(tiles, layout):
    return tiles[layout].reshape(GRID, GRID, TILE, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(480, 480, 3)


@torch.inference_mode()
def restore(model, tiles, device):
    x = torch.from_numpy(np.ascontiguousarray(tiles.transpose(0, 3, 1, 2))).float().div_(255)
    chunks = [model(x[i:i + 256].to(device)).cpu() for i in range(0, N, 256)]
    return torch.cat(chunks).permute(0, 2, 3, 1).mul(255).round().clamp(0, 255).byte().numpy()


@torch.inference_mode()
def ranker_matrix(model, tiles, direction, device):
    tensor = torch.from_numpy(np.ascontiguousarray(tiles.transpose(0, 3, 1, 2))).float().div_(255).to(device)
    anchor_desc, candidate_desc = descriptors(tensor, direction)
    distance = torch.cdist(anchor_desc, candidate_desc).square().div_(anchor_desc.shape[1])
    distance.fill_diagonal_(float("inf"))
    candidate_ids = distance.topk(TOPK, largest=False, dim=1).indices
    anchor_ids = torch.arange(N, device=device)[:, None].expand(-1, TOPK)
    flat_a, flat_b = anchor_ids.flatten(), candidate_ids.flatten()
    scores = []
    for start in range(0, len(flat_a), SCORE_BATCH):
        a = tensor[flat_a[start:start + SCORE_BATCH]]; b = tensor[flat_b[start:start + SCORE_BATCH]]
        scores.append(model(seam_features(a, b, direction)).float())
    scores = torch.cat(scores).reshape(N, TOPK)
    scores = F.log_softmax(scores, dim=1)
    matrix = torch.full((N, N), -30.0, device=device)
    matrix.scatter_(1, candidate_ids, scores)
    # Reciprocal-best bonus discourages one-way attractive but inconsistent edges.
    best_out = matrix.argmax(1); best_in = matrix.argmax(0)
    ids = torch.arange(N, device=device); mutual = best_in[best_out] == ids
    matrix[ids[mutual], best_out[mutual]] += 0.5
    matrix.fill_diagonal_(-30.0)
    return matrix.cpu().numpy()


@torch.inference_mode()
def position_matrix(model, tiles, device):
    x = torch.from_numpy(np.ascontiguousarray(tiles.transpose(0, 3, 1, 2))).float().div_(127.5).sub_(1).to(device)
    row_logits, col_logits = model(x); row = F.log_softmax(row_logits, 1); col = F.log_softmax(col_logits, 1)
    rows, cols = np.divmod(np.arange(N), GRID)
    return (row[:, rows] + col[:, cols]).cpu().numpy()


def adjacency(layout, true_layout):
    target_of = np.empty(N, np.int32); target_of[true_layout] = np.arange(N)
    x = target_of[layout].reshape(GRID, GRID)
    right = (x[:, 1:] == x[:, :-1] + 1) & (x[:, 1:] // GRID == x[:, :-1] // GRID)
    down = x[1:] == x[:-1] + GRID
    return float((right.sum() + down.sum()) / (right.size + down.size))


def load_models(device):
    rest_ck = torch.load(RESTORER_CKPT, map_location="cpu", weights_only=False)
    rank_ck = torch.load(RANKER_CKPT, map_location="cpu", weights_only=False)
    restorer = FragmentRestorer().to(device).eval(); restorer.load_state_dict(rest_ck["model"])
    ranker = BorderRanker().to(device).eval(); ranker.load_state_dict(rank_ck["model"])
    position = PositionPrior().to(device).eval(); position.load_state_dict(torch.load(POS_CKPT, map_location="cpu", weights_only=False)["model"])
    return restorer, ranker, position, rest_ck, rank_ck


def solve_one(path, models, device, seed):
    restorer, ranker, position = models[:3]
    restored = restore(restorer, split(path), device)
    right = ranker_matrix(ranker, restored, 0, device); down = ranker_matrix(ranker, restored, 1, device)
    pos = position_matrix(position, restored, device)
    layout = solve_layout(right, down, pos, seed)
    return restored, layout


def validation(models, device):
    z = np.load(MAP_FILE); stems, maps = z["stems"], z["maps"]
    order = np.arange(len(stems)); np.random.default_rng(SEED).shuffle(order); val = order[-max(100, len(order) // 10):]
    chosen = val[np.linspace(0, len(val) - 1, min(VAL_COUNT, len(val)), dtype=int)]
    rows = []
    for k, j in enumerate(chosen):
        stem = str(stems[j]); restored, layout = solve_one(DATA_ROOT / "inputs" / f"{stem}.png", models, device, SEED + k * 100)
        target = np.asarray(Image.open(DATA_ROOT / "targets" / f"{stem}.png").convert("RGB"), np.uint8)
        score = structural_similarity(target, assemble(restored, layout), channel_axis=2, data_range=255)
        item = {"stem": stem, "ssim": float(score), "adjacency": adjacency(layout, maps[j]),
                "tile_exact": float(np.mean(layout == maps[j]))}
        rows.append(item); print(json.dumps(item), flush=True)
    report = {"count": len(rows), "mean_ssim": float(np.mean([x["ssim"] for x in rows])),
              "mean_adjacency": float(np.mean([x["adjacency"] for x in rows])),
              "mean_tile_exact": float(np.mean([x["tile_exact"] for x in rows])), "images": rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True); (OUT_DIR / "validation_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def submission(models, device):
    files = sorted(TEST_DIR.glob("*.png"));
    if MAX_TEST: files = files[:MAX_TEST]
    image_dir = OUT_DIR / "submission_images"; image_dir.mkdir(parents=True, exist_ok=True)
    for k, path in enumerate(files):
        output = image_dir / path.name
        if not output.exists():
            restored, layout = solve_one(path, models, device, SEED + 10000 + k * 100)
            Image.fromarray(assemble(restored, layout)).save(output, optimize=False)
        if (k + 1) % 10 == 0: print(json.dumps({"done": k + 1, "total": len(files)}), flush=True)
    archive_path = OUT_DIR / "submission_border_pipeline.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files: archive.write(image_dir / path.name, arcname=path.name)
    print(json.dumps({"submission": str(archive_path), "files": len(files)}), flush=True)


def main():
    device = torch.device("cuda"); loaded = load_models(device)
    print(json.dumps({"device": torch.cuda.get_device_name(0), "mode": MODE, "topk": TOPK,
                      "restorer_epoch": loaded[3].get("epoch"), "ranker_epoch": loaded[4].get("epoch")}), flush=True)
    if MODE in ("validation", "both"): validation(loaded, device)
    if MODE in ("submission", "both"): submission(loaded, device)


if __name__ == "__main__": main()
