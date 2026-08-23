from __future__ import annotations

import torch

from canvas_data import CanvasDataset
from eval_r2l_affinity_union import DEFAULT_AFFINITY_A, DEFAULT_AFFINITY_B, DEFAULT_R2L, _load_r2, _union_candidates
from imgio import train_val_split
from train_offset_pose import load_frozen_affinity, mine_affinity_candidates


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_names = train_val_split()
    sample = CanvasDataset(val_names[:1], real_prob=0.0, seed=240815)[0]
    tiles = sample["tiles"].to(device)
    affinity, _, _ = load_frozen_affinity(DEFAULT_AFFINITY_A, device)
    affinity2, _, _ = load_frozen_affinity(DEFAULT_AFFINITY_B, device)
    r2 = _load_r2(DEFAULT_R2L, device)
    base, base_valid = mine_affinity_candidates(affinity, tiles.unsqueeze(0), candidate_k=64, device=device, affinity_secondary=affinity2)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else torch.no_grad():
        score = r2(tiles.unsqueeze(0)).float()
    candidates, valid = _union_candidates(base[0], base_valid[0], score[0], 8)
    print({
        "tiles": tuple(tiles.shape),
        "base": tuple(base.shape),
        "base_valid": tuple(base_valid.shape),
        "r2": tuple(score.shape),
        "union": tuple(candidates.shape),
        "valid": tuple(valid.shape),
        "r2_min": float(score.min()),
        "r2_max": float(score.max()),
    }, flush=True)


if __name__ == "__main__":
    main()
