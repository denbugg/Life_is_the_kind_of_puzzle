"""Fresh-corruption full graph gate for spatial-head + candidate-ranker fusion."""
from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from candidate_rank import neighbor_targets
from canvas_data import CanvasDataset
from config import NFRAG, SEED, WORK_ROOT
from eval_candidate_rank import load_ranker, score_full_graph
from eval_seeded_qap import dense_rd
from eval_symbolic_ranker_blend import _standardize_rows
from imgio import train_val_split
from placement_metrics import neighbour_accuracy, placement_accuracy
from positional_ddpm import PositionalDDPM
from solve_buddies import solve_buddies_from_scores
from train_offset_pose import load_frozen_affinity, mine_affinity_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=SEED + 7331)
    parser.add_argument("--alphas", default="0,0.1,0.25,0.5,0.75,1,1.25")
    parser.add_argument("--budgets", default="128,256,384,512")
    parser.add_argument("--pair-batch", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ranker", default="artifacts/candidate_rank/rank_v2w64_best.pt")
    parser.add_argument(
        "--spatial",
        default=str(Path(WORK_ROOT) / "positional_ddpm" / "positional_ddpm_train_latest.pt"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(WORK_ROOT) / "gates" / "fresh_spatial_ranker_blend.json",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    ranker, rank_payload = load_ranker(args.ranker, device)
    recorded = rank_payload.get("candidate_graph", {})
    encoders = list(recorded.get("encoders", ())) if isinstance(recorded, Mapping) else []
    training_args = rank_payload.get("args", {}) if isinstance(rank_payload.get("args"), Mapping) else {}
    if not encoders:
        raise RuntimeError("ranker checkpoint has no recorded affinity graph")
    affinity, _, _ = load_frozen_affinity(str(encoders[0]["path"]), device)
    affinity_secondary = None
    if len(encoders) > 1:
        affinity_secondary, _, _ = load_frozen_affinity(str(encoders[1]["path"]), device)
    candidate_k = int(training_args.get("candidate_k", 64))

    spatial_payload = torch.load(args.spatial, map_location="cpu", weights_only=False)
    spatial_model = PositionalDDPM(**spatial_payload["model_args"]).to(device)
    spatial_model.load_state_dict(spatial_payload["model"], strict=True)
    spatial_model.eval()

    _, validation_names = train_val_split()
    dataset = CanvasDataset(validation_names[:args.n], real_prob=0.0, seed=args.seed)
    alphas = [float(item) for item in args.alphas.split(",")]
    budgets = [int(item) for item in args.budgets.split(",")]
    rows: dict[str, list[dict[str, float]]] = {
        f"{alpha}:{budget}": [] for alpha in alphas for budget in budgets
    }
    for image_index in range(args.n):
        sample = dataset[image_index]
        tiles = sample["tiles"].to(device)
        permutation = sample["perm"].numpy().astype(np.int64)
        candidates_batched, valid_batched = mine_affinity_candidates(
            affinity,
            tiles.unsqueeze(0),
            candidate_k=candidate_k,
            device=device,
            affinity_secondary=affinity_secondary,
        )
        candidates_t = candidates_batched[0]
        valid_t = valid_batched[0]
        raw_dnk = score_full_graph(
            ranker,
            tiles,
            candidates_t,
            valid_t,
            pair_batch=args.pair_batch,
            device=device,
        )
        candidates = candidates_t.cpu().numpy().astype(np.int64)
        candidate_valid = valid_t.cpu().numpy()
        valid = np.broadcast_to(candidate_valid[:, None, :], (NFRAG, 4, candidate_valid.shape[1])).copy()
        raw = raw_dnk.permute(1, 0, 2).cpu().numpy().astype(np.float32)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            features = spatial_model.encode_tiles(tiles.unsqueeze(0))
            full_spatial = spatial_model.directional_edge_scores(features)[0].float().cpu().numpy()
        spatial = np.empty_like(raw)
        anchors = np.arange(NFRAG)[:, None]
        for direction in range(4):
            spatial[:, direction] = full_spatial[direction][anchors, candidates]
        raw_z = _standardize_rows(raw, valid)
        spatial_z = _standardize_rows(spatial, valid)
        truth, exists = neighbor_targets(torch.from_numpy(permutation)[None])
        truth = truth[0].numpy()
        exists = exists[0].numpy()
        target_board = np.argsort(permutation)

        for alpha in alphas:
            blend = raw_z + alpha * spatial_z
            blend[~valid] = -np.inf
            top = candidates[np.arange(NFRAG)[:, None], np.argmax(blend, axis=2)]
            r1 = float((top == truth)[exists].mean())
            right, down = dense_rd(
                torch.from_numpy(candidates).long(),
                torch.from_numpy(blend).permute(1, 0, 2).contiguous(),
            )
            for budget in budgets:
                board, _ = solve_buddies_from_scores(
                    right.numpy(), down.numpy(), max_edges=budget, min_margin=0.0, repair_passes=0
                )
                rows[f"{alpha}:{budget}"].append(
                    {
                        "edge_r1": r1,
                        "placement": placement_accuracy(board, target_board)[0],
                        "neighbour": neighbour_accuracy(board, target_board)[0],
                    }
                )
        print(json.dumps({"image": image_index + 1, "of": args.n}), flush=True)

    summary = {
        key: {
            metric: float(np.mean([row[metric] for row in values]))
            for metric in ("edge_r1", "placement", "neighbour")
        }
        for key, values in rows.items()
    }
    best_key = max(summary, key=lambda key: summary[key]["neighbour"])
    baseline_key = max(
        (key for key in summary if key.startswith("0.0:")),
        key=lambda key: summary[key]["neighbour"],
    )
    report = {
        "experiment": "fresh_full_graph_spatial_ranker_blend",
        "images": args.n,
        "seed": args.seed,
        "ranker_step": int(rank_payload.get("step", -1)),
        "spatial_step": int(spatial_payload.get("step", -1)),
        "baseline_key": baseline_key,
        "baseline": summary[baseline_key],
        "best_key": best_key,
        "best": summary[best_key],
        "delta": {
            metric: summary[best_key][metric] - summary[baseline_key][metric]
            for metric in ("edge_r1", "placement", "neighbour")
        },
        "summary": summary,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
