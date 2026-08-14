"""Protocol-matched frozen R2L DirectionalSiamese CAL evaluation for R7-G1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from canvas_data import CanvasDataset
from siamese_directional import DirectionalSiamese
from train_siamese_directional import evaluate

SPLIT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json")
R2L = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R2L_siamese\best.pt")
OUT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R7_full_contrastive_retriever\g1_capacity\r2l_matched_cal_report.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal-examples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260843)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(args.seed)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))["splits"]
    fit = set(split["fit"])
    cal_names = list(split["cal"][:args.cal_examples])
    if set(cal_names) & fit:
        raise RuntimeError("FIT/CAL overlap in pinned manifest")
    dataset = CanvasDataset(cal_names, real_prob=0.0, seed=args.seed)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    checkpoint = torch.load(R2L, map_location=device, weights_only=False)
    model = DirectionalSiamese(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    budgets = (128, 256, 384)
    metrics = evaluate(model, loader, device=device, maximum_images=args.cal_examples, budgets=budgets)
    report = {
        "experiment": "R7_G1_frozen_R2L_DirectionalSiamese_matched_CAL",
        "sources": {"split": str(SPLIT), "cal_count": len(cal_names), "fit_cal_overlap": 0, "real_prob": 0.0, "seed": args.seed},
        "frozen_model": {"path": str(R2L), "model_kwargs": checkpoint["model_kwargs"], "checkpoint_metrics": checkpoint.get("metrics")},
        "protocol": {"budgets": list(budgets), "fixed_orientation": True, "input": "synthetically_corrupted_permuted_tile_bag"},
        "metrics": metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
