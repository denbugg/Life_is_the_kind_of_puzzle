"""Group-CV nonlinear candidate scorer trained with within-scene RankNet loss."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import solver_v31 as s

TRAIN = tuple(range(6700, 6728)) + tuple(range(6957, 6981))
VALID = tuple(range(6981, 6989))
CACHE = s.ROOT / "critic_cache"


class BoardCritic(nn.Module):
    def __init__(self, features, width=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, width), nn.GELU(), nn.LayerNorm(width),
                                 nn.Dropout(.08), nn.Linear(width, width // 2), nn.GELU(),
                                 nn.Linear(width // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_scene(scene):
    with np.load(CACHE / f"scene_{scene:06d}.npz", allow_pickle=False) as data:
        return data["features"].astype(np.float32), data["labels"].astype(np.float32), data["names"].astype(str)


def pairs(data, scenes):
    left, right, sign, weight = [], [], [], []
    for scene in scenes:
        features, labels, names = data[scene]
        keep = np.flatnonzero(names != "baseline_marker")
        for offset, i in enumerate(keep):
            for j in keep[offset + 1:]:
                delta = float(labels[i] - labels[j])
                if abs(delta) < 1e-9:
                    continue
                left.append(features[i]); right.append(features[j])
                sign.append(1.0 if delta > 0 else -1.0)
                weight.append(np.sqrt(abs(delta)) + .02)
    return tuple(torch.from_numpy(np.asarray(value, np.float32)) for value in (left, right, sign, weight))


def fit(data, scenes, normalizer, seed, device, steps=500):
    torch.manual_seed(seed)
    left, right, sign, weight = (value.to(device) for value in pairs(data, scenes))
    mean, std = (torch.from_numpy(value).to(device) for value in normalizer)
    model = BoardCritic(left.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    for step in range(steps):
        index = torch.randint(len(left), (min(2048, len(left)),), generator=generator, device=device)
        a = model((left[index] - mean) / std)
        b = model((right[index] - mean) / std)
        loss = (F.softplus(-sign[index] * (a - b)) * weight[index]).sum() / weight[index].sum()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
    return model.eval()


@torch.inference_mode()
def scores(models, features, normalizer, device):
    mean, std = (torch.from_numpy(value).to(device) for value in normalizer)
    x = (torch.from_numpy(features).to(device) - mean) / std
    return torch.stack([model(x) for model in models]).mean(0).cpu().numpy()


def evaluate(data, scenes, models, normalizer, device):
    selected, baseline, oracle, rows = [], [], [], []
    for scene in scenes:
        features, labels, names = data[scene]
        candidate = np.flatnonzero(names != "baseline_marker")
        prediction = scores(models, features, normalizer, device)
        picked = int(candidate[np.argmax(prediction[candidate])])
        base = int(np.flatnonzero(names == "baseline_marker")[0])
        best = int(candidate[np.argmax(labels[candidate])])
        selected.append(labels[picked]); baseline.append(labels[base]); oracle.append(labels[best])
        rows.append({"scene": scene, "selected": str(names[picked]),
                     "selected_adjacency": float(labels[picked]),
                     "baseline_adjacency": float(labels[base]),
                     "oracle_adjacency": float(labels[best])})
    return {"selected": float(np.mean(selected)), "baseline": float(np.mean(baseline)),
            "oracle": float(np.mean(oracle)), "rows": rows}


def main():
    device = torch.device("cuda")
    data = {scene: load_scene(scene) for scene in TRAIN + VALID}
    train_features = np.concatenate([data[scene][0][data[scene][2] != "baseline_marker"] for scene in TRAIN])
    normalizer = (train_features.mean(0).astype(np.float32),
                  (train_features.std(0) + 1e-5).astype(np.float32))
    folds = tuple(tuple(TRAIN[index::4]) for index in range(4))
    oof_rows = []
    for fold, heldout in enumerate(folds):
        fit_scenes = tuple(scene for scene in TRAIN if scene not in heldout)
        models = [fit(data, fit_scenes, normalizer, 41000 + fold * 10 + seed, device)
                  for seed in range(2)]
        oof_rows.append(evaluate(data, heldout, models, normalizer, device))
        print(json.dumps({"event": "fold", "fold": fold, **oof_rows[-1]}), flush=True)
    ensemble = [fit(data, TRAIN, normalizer, 42000 + seed, device) for seed in range(3)]
    validation = evaluate(data, VALID, ensemble, normalizer, device)
    torch.save({"models": [model.state_dict() for model in ensemble],
                "mean": normalizer[0], "std": normalizer[1],
                "features": train_features.shape[1]}, s.OUT / "nonlinear_board_critic_v31.pt")
    report = {"oof_selected": float(np.mean([row["selected"] for row in oof_rows])),
              "oof_baseline": float(np.mean([row["baseline"] for row in oof_rows])),
              "validation": validation, "parameters": sum(p.numel() for p in ensemble[0].parameters())}
    (s.OUT / "nonlinear_board_critic_v31.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"event": "complete", **report}), flush=True)


if __name__ == "__main__":
    main()

