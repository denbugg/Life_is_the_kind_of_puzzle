"""P8 FIT-only P3 hardlist contract probe; no CAL/DEV/test access or assembly."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import torch

import infer_rank96 as rank96
import p3_g1_cdcs_capacity as p3
import p8_context_candidate_graph as p8
from eval_candidate_rank import score_full_graph
from train_eval_cb1_g1_capacity import distort_frags, load_rgb, to_frags
from train_offset_pose import mine_affinity_candidates

WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P8_context_candidate_graph\hardlist_contract_probe")
c = SimpleNamespace(
    targets=p8.FIT_TARGETS, split=p8.SPLIT, p7=p8.P7_CKPT, work=WORK,
    device="cuda", seed=20260820, train_sources=128, eval_sources=32,
    steps=4000, batch_queries=12, eval_queries=256,
)
train, _ = p8.sources(c)
name = train[0]
device = torch.device("cuda")
print({"stage": "start", "source": name}, flush=True)
models = rank96.load_models(p3.config(), device)
clean = load_rgb(c.targets / name)
frags = distort_frags(to_frags(clean), np.random.default_rng(c.seed * 1009))
perm = np.random.default_rng(c.seed * 2029).permutation(p8.N).astype(np.int32)
tiles = frags[perm]
ten = torch.from_numpy(tiles).permute(0, 3, 1, 2).contiguous().float().to(device)
with torch.no_grad():
    candidates, valid = mine_affinity_candidates(
        models.affinity_primary, ten.unsqueeze(0), candidate_k=64, device=device,
        affinity_secondary=models.affinity_secondary,
    )
    scores = score_full_graph(
        models.ranker, ten, candidates[0], valid[0], pair_batch=4096, device=device,
    )
cn = candidates[0].cpu().numpy()
vv = valid[0].cpu().numpy()
sc = scores.cpu().numpy()
anchors, dirs, members = p3.hardlists(cn, vv, sc, perm)
inv = np.empty(p8.N, dtype=np.int32)
inv[perm] = np.arange(p8.N, dtype=np.int32)
issues: list[dict[str, object]] = []
for q, (anchor, direction, row) in enumerate(zip(anchors, dirs, members)):
    truth = int(inv[p8.neighbor(int(perm[int(anchor)]), int(direction))])
    for member in row:
        if not np.any(cn[int(anchor)] == member) and int(member) != truth:
            issues.append({
                "q": int(q), "anchor": int(anchor), "direction": int(direction),
                "truth": truth, "member": int(member), "row": row.astype(int).tolist(),
                "candidate_anchor_slice": cn[int(anchor)].astype(int).tolist(),
                "candidate_direction_anchor_slice": (
                    cn[int(direction), int(anchor)].astype(int).tolist() if cn.ndim == 3 else None
                ),
            })
            if len(issues) >= 5:
                break
    if len(issues) >= 5:
        break
out = {
    "candidate_shape": list(cn.shape), "valid_shape": list(vv.shape),
    "scores_shape": list(sc.shape), "members_shape": list(members.shape),
    "issue_count_capped": len(issues), "issues": issues,
}
print(json.dumps(out, indent=2), flush=True)
