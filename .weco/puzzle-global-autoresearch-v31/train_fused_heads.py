"""Train larger coordinate/border GNN directly on frozen fused V28 matrices."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/kva/pazzle_global_autoresearch_v31")
V30_ROOT = Path("/home/kva/pazzle_edge_unary_lns_v30")
sys.path.insert(0, str(V30_ROOT))
import train_solver_v30 as v30

TRAIN = tuple(range(6700, 6728)) + tuple(range(6957, 6981))
VALID = tuple(range(6981, 6989))
WIDTH = 160
STEPS = 16
UPDATES = 1200
SEED = 311826


def fused(scene, reranker, device):
    base = v30.load_v27(scene, reranker, device)
    with np.load(v30.V28_ROOT / "score_cache" / f"scene_{scene:06d}.npz") as data:
        extra = data["scores"].astype(np.float32)
    return [v30.row_z(.30 * base[d] + .70 * extra[d]) for d in range(2)]


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda")
    state27 = torch.load(v30.V27_ROOT / "outputs/set_reranker_best.pt",
                         map_location=device, weights_only=True)
    reranker = v30.v27.SetReranker().to(device)
    reranker.load_state_dict(state27["model"])
    reranker.eval()
    bundles = {scene: fused(scene, reranker, device) for scene in TRAIN + VALID}
    prepared = {scene: v30.graph_inputs(matrices) for scene, matrices in bundles.items()}
    model = v30.DirectionalCoordinateGNN(width=WIDTH, steps=STEPS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-3)
    row_t, col_t, border_t = v30.targets(device)
    rng = np.random.default_rng(SEED)
    best_score = -1.0
    best = None
    history = []
    started = time.perf_counter()
    for step in range(1, UPDATES + 1):
        scene = int(rng.choice(TRAIN))
        node, nbr, weights = prepared[scene]
        row, col, border = model(torch.from_numpy(node).to(device),
                                 torch.from_numpy(nbr).to(device),
                                 torch.from_numpy(weights).to(device))
        loss = F.cross_entropy(row, row_t, label_smoothing=.03)
        loss += F.cross_entropy(col, col_t, label_smoothing=.03)
        loss += .25 * F.binary_cross_entropy_with_logits(
            border, border_t, pos_weight=torch.full((4,), 12.0, device=device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if step % 100 == 0:
            metrics = v30.head_metrics(model, {s: bundles[s] for s in VALID}, device)
            score = metrics["row"] + metrics["column"] + .20 * metrics["border_f1"]
            record = {"step": step, "loss": float(loss.detach()), "score": score,
                      "seconds": time.perf_counter() - started, **metrics}
            history.append(record)
            print(json.dumps(record), flush=True)
            if score > best_score:
                best_score = score
                best = copy.deepcopy(model.state_dict())
    parameters = sum(value.numel() for value in model.parameters())
    payload = {"heads": best, "width": WIDTH, "steps": STEPS,
               "parameters": parameters, "score": best_score,
               "history": history, "train_scenes": TRAIN, "valid_scenes": VALID}
    output = ROOT / "outputs/fused_heads_v31.pt"
    torch.save(payload, output)
    (ROOT / "outputs/fused_heads_v31.json").write_text(json.dumps({
        key: value for key, value in payload.items() if key != "heads"}, indent=2))
    print(json.dumps({"event": "complete", "path": str(output),
                      "parameters": parameters, "score": best_score}), flush=True)


if __name__ == "__main__":
    main()

