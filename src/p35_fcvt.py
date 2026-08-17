"""P35 FCVT-24: source-invariant coordinate regression; no targets or P8 artifacts."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p29_dpcg import N, load_tiles
from p32_dscp import dino, features

GRID = 24
SLOTS = torch.stack(
    torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="ij"), -1
).reshape(N, 2).float() / float(GRID - 1)


def seed(value: int = 20260817) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def projection(coords: np.ndarray) -> np.ndarray:
    if coords.shape != (N, 2) or not np.isfinite(coords).all():
        raise RuntimeError("invalid coordinates")
    slots = SLOTS.numpy()
    costs = ((coords[:, None, :] - slots[None, :, :]) ** 2).sum(-1)
    rows, cols = linear_sum_assignment(costs)
    out = np.empty(N, np.int64)
    out[rows] = cols
    if np.unique(out).size != N:
        raise RuntimeError("non-bijective projection")
    return out


class CoordSet(nn.Module):
    """DeepSets coordinate regressor; it has no source, filename, or tile-index input."""
    def __init__(self, dim: int = 384, width: int = 512) -> None:
        super().__init__()
        self.tile = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(width * 2), nn.Linear(width * 2, width), nn.GELU(),
            nn.Linear(width, 2), nn.Sigmoid(),
        )

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        token = self.tile(tiles)
        context = token.mean(1, keepdim=True).expand_as(token)
        return self.head(torch.cat((token, context), dim=-1))


def locked_sources(manifest: Path):
    data = json.loads(manifest.read_text())
    train = list(data["train_sources"])
    held = list(data["held_sources"])
    if len(train) != 128 or len(held) != 32 or len(set(train + held)) != 160:
        raise RuntimeError("pinned split mismatch")
    return train[:96], train[96:], held


def cached_label(label_dir: Path, source: str):
    with np.load(label_dir / (Path(source).stem + ".npz"), allow_pickle=False) as data:
        slots = data["target_tile_to_slot"].astype(np.int64)
        cached_source = str(data["source"])
    if cached_source != source or slots.shape != (N,) or np.unique(slots).size != N:
        raise RuntimeError("invalid cached label")
    coords = np.stack((slots // GRID, slots % GRID), axis=-1).astype(np.float32) / float(GRID - 1)
    return torch.from_numpy(coords), slots


def frozen_features(backbone, inputs: Path, sources, device, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for index, source in enumerate(sources, 1):
        path = cache_dir / (Path(source).stem + ".pt")
        if path.exists():
            feature = torch.load(path, map_location="cpu", weights_only=True)
        else:
            with torch.no_grad():
                feature = features(backbone, load_tiles(inputs, source), device).cpu()
            if feature.shape != (N, 384) or not torch.isfinite(feature).all():
                raise RuntimeError("invalid frozen DINO feature")
            torch.save(feature, path)
        result[source] = feature.float()
        if index % 4 == 0:
            print(json.dumps({"stage": "features", "done": index, "total": len(sources)}), flush=True)
    return result


def gate_g0(args):
    seed()
    permutation = torch.randperm(N)
    expected = permutation.numpy()
    recovered = projection(SLOTS[permutation].numpy())
    ok = bool(np.array_equal(expected, recovered))
    return {
        "experiment": "P35_FCVT24", "gate": "G0", "synthetic_exact": ok,
        "invalid": 0, "targets_opened": False, "p8_imported": False,
        "passes_G0": ok,
    }


def gate_g1(args):
    seed()
    device = torch.device("cuda")
    backbone = dino(device)
    model = CoordSet().to(device).eval()
    rows = []
    with torch.no_grad():
        for source in args.sources:
            feature = features(backbone, load_tiles(args.inputs, source), device).to(device).unsqueeze(0)
            permutation = torch.randperm(N, device=device)
            reference = model(feature)[0]
            shuffled = model(feature[:, permutation])[0]
            inverse = torch.empty_like(permutation)
            inverse[permutation] = torch.arange(N, device=device)
            error = float((reference - shuffled[inverse]).abs().max().cpu())
            rows.append({
                "source": source, "equivariance_max_abs": error,
                "finite": bool(torch.isfinite(reference).all()),
            })
    maximum = max(row["equivariance_max_abs"] for row in rows)
    invalid = sum(not row["finite"] for row in rows)
    return {
        "experiment": "P35_FCVT24", "gate": "G1", "sources": len(rows),
        "max_equivariance_error": maximum, "invalid": invalid,
        "targets_opened": False, "p8_imported": False,
        "passes_G1": bool(maximum < 1e-5 and invalid == 0), "rows": rows,
    }


def fit_and_evaluate(model, feature_map, label_map, device, epochs: int = 18):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    names = list(feature_map)
    model.train()
    for epoch in range(epochs):
        random.Random(20260817 + epoch).shuffle(names)
        losses = []
        for source in names:
            feature = feature_map[source].unsqueeze(0).to(device)
            label = label_map[source].unsqueeze(0).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(feature)
            loss = F.smooth_l1_loss(prediction, label, beta=0.08)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(json.dumps({"stage": "train", "epoch": epoch + 1, "epochs": epochs,
                          "loss": sum(losses) / len(losses)}), flush=True)
    rows = []
    model.eval()
    with torch.no_grad():
        for source in names:
            prediction = model(feature_map[source].unsqueeze(0).to(device))[0].cpu().numpy()
            label = label_map[source].numpy()
            truth_slots = (label[:, 0] * (GRID - 1)).round().astype(np.int64) * GRID
            truth_slots += (label[:, 1] * (GRID - 1)).round().astype(np.int64)
            assigned = projection(prediction)
            rows.append({
                "source": source,
                "mae_slots": float(np.abs((prediction - label) * (GRID - 1)).mean()),
                "exact_placement": float(np.mean(assigned == truth_slots)),
                "valid": True,
            })
    return rows


def gate_g2(args):
    seed()
    train_sources, _, _ = locked_sources(args.manifest)
    if len(args.sources) != 96 or set(args.sources) != set(train_sources):
        raise RuntimeError("G2 must use exactly the locked 96 FIT-train sources")
    device = torch.device("cuda")
    backbone = dino(device)
    feature_map = frozen_features(backbone, args.inputs, args.sources, device, args.feature_cache)
    label_map = {source: cached_label(args.labels, source)[0] for source in args.sources}
    model = CoordSet().to(device)
    started = time.perf_counter()
    rows = fit_and_evaluate(model, feature_map, label_map, device)
    seconds = time.perf_counter() - started
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "sources": list(args.sources)}, args.model_out)
    mae = float(np.mean([row["mae_slots"] for row in rows]))
    exact = float(np.mean([row["exact_placement"] for row in rows]))
    invalid = sum(not row["valid"] for row in rows)
    return {
        "experiment": "P35_FCVT24", "gate": "G2", "sources": len(rows),
        "mae_slots": mae, "exact_placement": exact, "invalid": invalid,
        "train_seconds": seconds, "targets_opened": False, "p8_imported": False,
        "selection_opened": False, "held_opened": False,
        "passes_G2": bool(mae < 6.0 and invalid == 0 and seconds < 900), "rows": rows,
    }


def gate_g3(args):
    seed()
    _, selection_sources, _ = locked_sources(args.manifest)
    if len(args.sources) != 32 or set(args.sources) != set(selection_sources):
        raise RuntimeError("G3 must use exactly the locked FIT-selection sources")
    device = torch.device("cuda")
    backbone = dino(device)
    feature_map = frozen_features(backbone, args.inputs, args.sources, device, args.feature_cache)
    label_map = {source: cached_label(args.labels, source)[0] for source in args.sources}
    model = CoordSet().to(device)
    checkpoint = torch.load(args.model_out, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    rows = fit_and_evaluate(model, feature_map, label_map, device, epochs=0)
    mae = float(np.mean([row["mae_slots"] for row in rows]))
    exact = float(np.mean([row["exact_placement"] for row in rows]))
    invalid = sum(not row["valid"] for row in rows)
    return {
        "experiment": "P35_FCVT24", "gate": "G3", "sources": len(rows),
        "mae_slots": mae, "exact_placement": exact, "invalid": invalid,
        "targets_opened": False, "p8_imported": False, "held_opened": False,
        "passes_G3": bool(exact >= 0.03189887152777778 and invalid == 0), "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("g0", "g1", "g2", "g3"), required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--model-out", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sources", nargs="*", default=[])
    args = parser.parse_args()
    report = {"g0": gate_g0, "g1": gate_g1, "g2": gate_g2, "g3": gate_g3}[args.mode](args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report[f"passes_{args.mode.upper()}"]:
        raise RuntimeError(f"P35 {args.mode.upper()} rejected")


if __name__ == "__main__":
    main()
