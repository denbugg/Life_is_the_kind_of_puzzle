"""Cache frozen V28 multimodal matrices for V31 train/validation scenes."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

V28_ROOT = Path("/home/kva/pazzle_multimodal_boundary_v28")
sys.path.insert(0, str(V28_ROOT))
import train_multimodal_v28 as v28

SCENES = tuple(range(6700, 6728)) + tuple(range(6957, 6989))


def main():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    maps = np.load(v28.v25.MAP_FILE)["inv"]
    denoiser, _ = v28.tile_denoiser.load_denoiser(v28.DENOISER_CKPT, device)
    contour_module, _, _, contour_net, threshold, _ = v28.v25.load_winner(device)
    state = torch.load(v28.OUT / "multimodal_best.pt", map_location="cpu", weights_only=True)
    model = v28.MultimodalBoundary(v28.v23.ModelConfig(**state["model_config"])).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    v28.CACHE.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for index, scene in enumerate(SCENES, 1):
        path = v28.CACHE / f"scene_{scene:06d}.npz"
        cached = path.exists()
        if not cached:
            scores = v28.score_scene(model, scene, denoiser, contour_module,
                                     contour_net, threshold, maps, device)
            np.savez_compressed(path, scores=np.asarray(scores, np.float16))
        print(json.dumps({"event": "cache", "scene": scene, "index": index,
                          "of": len(SCENES), "cached": cached,
                          "seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()

