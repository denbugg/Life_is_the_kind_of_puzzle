"""Build paired clean/challenge-noisy fused V27+V28 score matrices.

Both branches consume the exact same tile bytes.  This avoids the invalid
clean-V27/noisy-V28 blend that would otherwise leak an easier representation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path("/home/kva/pazzle_global_autoresearch_v32_noise")
V25_ROOT = Path("/home/kva/pazzle_v18_v22_v23_fusion_v25")
V27_ROOT = Path("/home/kva/pazzle_set_transformer_v27")
V28_ROOT = Path("/home/kva/pazzle_multimodal_boundary_v28")
V30_ROOT = Path("/home/kva/pazzle_edge_unary_lns_v30")
sys.path[:0] = [str(ROOT), str(V30_ROOT), str(V28_ROOT), str(V27_ROOT), str(V25_ROOT)]

import synthetic_noise
import train_solver_v30 as v30
import train_multimodal_v28 as v28

SCENES = tuple(range(6700, 6728)) + tuple(range(6957, 6989))
CACHE = ROOT / "noisy_score_cache"
CONTRACT = "brightness[-30,30]_contrast[.70,1.30]_sigma[40,55]_blur3_jpeg[35,50]"


def seed_for(scene: int, replica: int) -> int:
    digest = hashlib.sha256(f"v32:{scene}:{replica}:{CONTRACT}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF


@torch.inference_mode()
def score_tiles(raw: torch.Tensor, models: dict[str, object], device: torch.device) -> list[np.ndarray]:
    # V25/V27 branch from exactly these bytes.
    x = raw.unsqueeze(0)
    _base22, refined22 = v28.v25.v22.refine(
        models["model22"], models["model18"], models["winner"], x)
    score_sets = []
    for model in (models["small"], models["xl"]):
        embeddings = model(raw)
        score_sets.append([
            v28.v25.row_z((embeddings["right"] @ embeddings["left"].t()).float().cpu().numpy()),
            v28.v25.row_z((embeddings["bottom"] @ embeddings["top"].t()).float().cpu().numpy()),
        ])
    seam = []
    for source_side, target_side in (("right", "left"), ("bottom", "top")):
        source = models["small"].side_features(raw, source_side).flatten(1)
        target = models["small"].side_features(raw, target_side).flatten(1)
        source = torch.nn.functional.normalize(source - source.mean(1, keepdim=True), dim=1)
        target = torch.nn.functional.normalize(target - target.mean(1, keepdim=True), dim=1)
        seam.append(v28.v25.row_z((source @ target.t()).float().cpu().numpy()))
    old = ([v28.v25.row_z(value) for value in refined22], [
        v28.v25.row_z(.25 * score_sets[0][d] + .75 * score_sets[1][d] + .50 * seam[d])
        for d in range(2)
    ])
    base = v28.v27.rerank_scene(models["reranker"], old, 1.35, device)

    # V28 multimodal branch, recomputed after corruption (no clean modality leak).
    views = v28.modality_views(raw, models["denoiser"], models["contour_module"],
                               models["contour_net"], models["threshold"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        embeddings = models["multimodal"](views)
    scale = float(models["multimodal"].scale())
    extra = [
        v28.v25.row_z((scale * embeddings["right"] @ embeddings["left"].t()).float().cpu().numpy()),
        v28.v25.row_z((scale * embeddings["bottom"] @ embeddings["top"].t()).float().cpu().numpy()),
    ]
    return [v30.row_z(.30 * base[d] + .70 * extra[d]).astype(np.float32) for d in range(2)]


def load_models(device: torch.device) -> dict[str, object]:
    model18, model22, small, xl, _state18, _state22 = v28.v25.load_models(device)
    winner = v28.v25.load_winner(device)
    reranker_state = torch.load(v28.V27_CKPT, map_location=device, weights_only=True)
    reranker = v28.v27.SetReranker().to(device)
    reranker.load_state_dict(reranker_state["model"]); reranker.eval()
    denoiser, _ = v28.tile_denoiser.load_denoiser(v28.DENOISER_CKPT, device)
    contour_module, _, _, contour_net, threshold, _ = v28.v25.load_winner(device)
    state = torch.load(v28.OUT / "multimodal_best.pt", map_location="cpu", weights_only=True)
    multimodal = v28.MultimodalBoundary(v28.v23.ModelConfig(**state["model_config"])).to(device)
    multimodal.load_state_dict(state["model"], strict=True); multimodal.eval()
    return dict(model18=model18, model22=model22, small=small, xl=xl, winner=winner,
                reranker=reranker, denoiser=denoiser, contour_module=contour_module,
                contour_net=contour_net, threshold=threshold, multimodal=multimodal)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=len(SCENES))
    parser.add_argument("--replicas", type=int, default=2)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    models = load_models(device)
    targets = v28.v25.RAW_INPUTS.parent / "targets"
    CACHE.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for ordinal, scene in enumerate(SCENES[args.start:args.stop], args.start + 1):
        image = v28.v25.v10.load_rgb(targets / f"img_{scene:06d}.png")
        clean = np.ascontiguousarray(v28.v25.v10.image_to_tiles(image))
        views = [("clean", -1, clean, None)]
        for replica in range(args.replicas):
            seed = seed_for(scene, replica)
            noisy, _draws = synthetic_noise.corrupt_tiles(clean, seed)
            views.append((f"noise_{replica}", replica, noisy, seed))
        for view, replica, tiles, seed in views:
            path = CACHE / f"scene_{scene:06d}_{view}.npz"
            if not path.exists():
                raw = torch.from_numpy(tiles).permute(0, 3, 1, 2).float().div_(255).to(device)
                scores = score_tiles(raw, models, device)
                np.savez_compressed(path, scores=np.asarray(scores, np.float16), tiles=tiles,
                                    scene=scene, replica=replica, seed=-1 if seed is None else seed,
                                    contract=np.asarray(CONTRACT))
            print(json.dumps({"event": "score_cache", "scene": scene, "view": view,
                              "index": ordinal, "of": args.stop - args.start,
                              "cached": path.exists(), "seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
