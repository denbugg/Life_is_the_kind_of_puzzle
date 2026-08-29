"""Learn a listwise reranker over the V22/V23 top-32 candidate union."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("/home/kva/pazzle_union_reranker_v26")
V25_ROOT = Path("/home/kva/pazzle_v18_v22_v23_fusion_v25")
sys.path.insert(0, str(V25_ROOT))
import evaluate_fusion_v25 as v25

CACHE = ROOT / "cache"
OUT = ROOT / "outputs"
TRAIN_SCENES = range(6700, 6716)
VALID_SCENES = range(6716, 6720)
HOLDOUT_SCENES = range(6957, 6973)
GRID = 24
TOP_K = 32
MAX_CANDIDATES = TOP_K * 2
SEED = 260826


class Reranker(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(.10),
            nn.Linear(128, 128), nn.GELU(), nn.Dropout(.10), nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def sources_targets(direction: int):
    grid = np.arange(GRID * GRID).reshape(GRID, GRID)
    if direction == 0:
        return grid[:, :-1].reshape(-1), grid[:, 1:].reshape(-1)
    return grid[:-1].reshape(-1), grid[1:].reshape(-1)


def ranks_desc(values):
    order = np.argsort(-values, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(values.shape[0])[:, None], order] = np.arange(values.shape[1])[None, :]
    return ranks


def candidate_union(a, b, source):
    ca = np.argpartition(-a[source], TOP_K - 1)[:TOP_K]
    cb = np.argpartition(-b[source], TOP_K - 1)[:TOP_K]
    return np.unique(np.concatenate((ca, cb)))


def context(a, b):
    base = v25.row_z(.55 * a + .45 * b)
    ra, rb = ranks_desc(a), ranks_desc(b)
    # Column ranks express whether a target also selects this source strongly.
    cra, crb = ranks_desc(a.T).T, ranks_desc(b.T).T
    return base, ra, rb, cra, crb


def features(a, b, direction, source, candidates, ctx):
    base, ra, rb, cra, crb = ctx
    sa, sb, sf = a[source, candidates], b[source, candidates], base[source, candidates]
    r1, r2 = ra[source, candidates], rb[source, candidates]
    c1, c2 = cra[source, candidates], crb[source, candidates]
    scale = np.log1p(a.shape[0])
    values = np.column_stack([
        sa, sb, sf, sa - sb, sa * sb,
        np.log1p(r1) / scale, np.log1p(r2) / scale,
        np.log1p(c1) / scale, np.log1p(c2) / scale,
        np.abs(r1 - r2) / a.shape[0],
        r1 < 1, r2 < 1, r1 < 5, r2 < 5, r1 < 16, r2 < 16,
        np.full(len(candidates), float(direction)),
        sf - np.max(base[source]),
    ]).astype(np.float32)
    return values


def cache_scene(scene, models, winner, device, maps):
    path = CACHE / f"scene_{scene:06d}.npz"
    if not path.exists():
        model18, model22, small, xl = models
        scores = v25.score_scene(scene, model18, model22, winner, small, xl, device, maps)
        np.savez_compressed(path, v22=np.asarray(scores[1], np.float16), v23=np.asarray(scores[2], np.float16))
    with np.load(path) as data:
        return data["v22"].astype(np.float32), data["v23"].astype(np.float32)


def build_groups(scene_scores):
    rows, masks, labels = [], [], []
    for a2, b2 in scene_scores:
        for direction in range(2):
            a, b = a2[direction], b2[direction]
            ctx = context(a, b)
            sources, targets = sources_targets(direction)
            for source, target in zip(sources, targets):
                candidates = candidate_union(a, b, source)
                positions = np.flatnonzero(candidates == target)
                if not len(positions):
                    continue
                row = np.zeros((MAX_CANDIDATES, 18), np.float32)
                mask = np.zeros(MAX_CANDIDATES, bool)
                row[:len(candidates)] = features(a, b, direction, source, candidates, ctx)
                mask[:len(candidates)] = True
                rows.append(row); masks.append(mask); labels.append(int(positions[0]))
    return (torch.from_numpy(np.stack(rows)), torch.from_numpy(np.stack(masks)),
            torch.tensor(labels, dtype=torch.long))


@torch.inference_mode()
def rerank_scene(model, scores, beta, device):
    output = []
    a2, b2 = scores
    for direction in range(2):
        a, b = a2[direction], b2[direction]
        ctx = context(a, b); final = ctx[0].copy()
        sources, _ = sources_targets(direction)
        for source in sources:
            candidates = candidate_union(a, b, source)
            x = torch.from_numpy(features(a, b, direction, source, candidates, ctx)).to(device)
            pred = model(x).float().cpu().numpy()
            pred = (pred - pred.mean()) / (pred.std() + 1e-6)
            final[source, candidates] += beta * pred
        output.append(final)
    return output


def aggregate(rows):
    return {key: float(np.mean([x[key] for x in rows])) for key in rows[0]}


def evaluate(model, scene_scores, beta, device):
    return aggregate([v25.metrics(rerank_scene(model, scores, beta, device)) for scores in scene_scores])


def train_model(train_groups, valid_groups, device):
    torch.manual_seed(SEED)
    model = Reranker(train_groups[0].shape[-1]).to(device)
    loader = DataLoader(TensorDataset(*train_groups), batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    best = None; best_loss = float("inf"); history = []
    vx, vm, vy = [x.to(device) for x in valid_groups]
    for epoch in range(1, 26):
        model.train(); losses = []
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logits = model(x).masked_fill(~mask, -1e4)
            loss = nn.functional.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            losses.append(float(loss))
        model.eval()
        with torch.inference_mode():
            valid_loss = float(nn.functional.cross_entropy(model(vx).masked_fill(~vm, -1e4), vy))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_loss": valid_loss})
        print(json.dumps({"event": "epoch", **history[-1]}), flush=True)
        if valid_loss < best_loss:
            best_loss, best = valid_loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return model.eval(), history, best_loss


def main():
    CACHE.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda"); torch.backends.cuda.matmul.allow_tf32 = True
    loaded = v25.load_models(device)
    models, state18, state22 = loaded[:4], loaded[4], loaded[5]
    winner = v25.load_winner(device); maps = np.load(v25.MAP_FILE)["inv"]
    all_scenes = list(TRAIN_SCENES) + list(VALID_SCENES) + list(HOLDOUT_SCENES)
    score_by_scene = {}; started = time.perf_counter()
    for index, scene in enumerate(all_scenes, 1):
        score_by_scene[scene] = cache_scene(scene, models, winner, device, maps)
        print(json.dumps({"event": "cache", "scene": scene, "of": len(all_scenes),
                          "seconds": time.perf_counter() - started}), flush=True)
    train_scores = [score_by_scene[x] for x in TRAIN_SCENES]
    valid_scores = [score_by_scene[x] for x in VALID_SCENES]
    holdout_scores = [score_by_scene[x] for x in HOLDOUT_SCENES]
    train_groups = build_groups(train_scores); valid_groups = build_groups(valid_scores)
    model, history, best_loss = train_model(train_groups, valid_groups, device)
    # This fixed, predeclared grid produced the accepted V26 result. The separate
    # beta-grid-3 report records the wider follow-up ablation without changing it.
    betas = (0., .05, .10, .15, .20, .30, .40, .55, .70, 1.)
    trials = []
    for beta in betas:
        value = evaluate(model, valid_scores, beta, device)
        trials.append({"beta": beta, **value, "objective": v25.objective(value)})
        print(json.dumps({"event": "beta", **trials[-1]}), flush=True)
    selected = max(trials, key=lambda x: x["objective"])
    baseline = evaluate(model, holdout_scores, 0., device)
    result = evaluate(model, holdout_scores, selected["beta"], device)
    checkpoint = {"model": model.state_dict(), "features": 18, "beta": selected["beta"], "seed": SEED}
    torch.save(checkpoint, OUT / "reranker_best.pt")
    report = {
        "schema": "puzzle-union-reranker-v26", "v18_step": state18["step"], "v22_step": state22["step"],
        "train_scenes": [min(TRAIN_SCENES), max(TRAIN_SCENES)],
        "validation_scenes": [min(VALID_SCENES), max(VALID_SCENES)],
        "holdout_scenes": [min(HOLDOUT_SCENES), max(HOLDOUT_SCENES)],
        "train_groups": len(train_groups[2]), "validation_groups": len(valid_groups[2]),
        "parameters": sum(p.numel() for p in model.parameters()), "best_validation_loss": best_loss,
        "selected": selected, "holdout_v25_baseline": baseline, "holdout_v26": result,
        "trials": trials, "history": history, "seconds": time.perf_counter() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "complete", "report": report}), flush=True)


if __name__ == "__main__":
    main()
