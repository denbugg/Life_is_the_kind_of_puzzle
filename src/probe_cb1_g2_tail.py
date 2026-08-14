from __future__ import annotations

import numpy as np
import torch

from eval_cb1_g2_cal_graph import CACHE, CHECKPOINT, INPUT, BoundaryBuddyNet, directional_l1_order, score_candidates
from infer_rank96 import load_rgb_strict, split_upright_tiles

with np.load(CACHE, allow_pickle=False) as cache:
    base = np.asarray(cache["candidate_ids"], dtype=np.int64).copy()
tiles = split_upright_tiles(load_rgb_strict(INPUT))
device = torch.device("cuda")
state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model = BoundaryBuddyNet().to(device)
model.load_state_dict(state["state_dict"], strict=True)
model.eval()
with torch.no_grad():
    for anchor in range(432, 576):
        print({"anchor": anchor}, flush=True)
        frozen = [int(x) for x in base[anchor]]
        for direction in range(4):
            l1 = [int(x) for x in directional_l1_order(tiles, anchor, direction)[:128]]
            candidates = list(dict.fromkeys(frozen + l1))
            scores = score_candidates(model, tiles, anchor, candidates, direction, device, 4096)
            if not np.isfinite(scores).all():
                raise FloatingPointError((anchor, direction))
print({"tail_probe": "complete"}, flush=True)
