"""Protocol-matched frozen R2L CAL evaluation for R7-G1 adjudication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from canvas_data import CanvasDataset
from direct_pose import DirectPoseNet
from train_direct_pose import evaluate
from train_offset_pose import load_frozen_affinity, make_loader

SPLIT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json")
R2L = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R2L_siamese\best.pt")
AFF1 = r"C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed\artifacts\macro_affinity\affinity_r1_1200_best.pt"
AFF2 = r"C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed\artifacts\macro_affinity\affinity_r3_1000_best.pt"
OUT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R7_full_contrastive_retriever\g1_capacity\r2l_matched_cal_report.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal-examples", type=int, default=32)
    ap.add_argument("--candidate-k", type=int, default=128)
    ap.add_argument("--pair-batch", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=20260843)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
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
    loader = make_loader(dataset, batch_size=1, workers=0, shuffle=False, device=device)
    payload = torch.load(R2L, map_location=device, weights_only=False)
    model = DirectPoseNet(**payload["model_kwargs"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    affinity1, provenance1, _ = load_frozen_affinity(AFF1, device)
    affinity2, provenance2, _ = load_frozen_affinity(AFF2, device)
    metrics = evaluate(
        model,
        affinity1,
        loader,
        candidate_k=args.candidate_k,
        max_images=args.cal_examples,
        pair_batch=args.pair_batch,
        device=device,
        affinity_secondary=affinity2,
        direct_threshold=0.5,
        non_direct_weight=1.0,
        direction_weight=1.0,
    )
    report = {
        "experiment": "R7_G1_frozen_R2L_matched_CAL",
        "sources": {"split": str(SPLIT), "cal_count": len(cal_names), "fit_cal_overlap": 0, "real_prob": 0.0, "seed": args.seed},
        "frozen_models": {"r2l": str(R2L), "affinity_primary": dict(provenance1), "affinity_secondary": dict(provenance2)},
        "protocol": {"candidate_k": args.candidate_k, "pair_batch": args.pair_batch, "fixed_orientation": True},
        "metrics": metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
