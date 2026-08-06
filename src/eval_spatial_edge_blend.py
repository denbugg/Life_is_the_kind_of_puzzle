"""Held-out fusion of the spatial directional head and the strong seam ranker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from candidate_rank import neighbor_targets
from config import NFRAG, SEED, WORK_ROOT
from eval_seeded_qap import dense_rd
from eval_symbolic_ranker_blend import _parse_groups, _recreate_group, _standardize_rows
from imgio import train_val_split
from placement_metrics import neighbour_accuracy, placement_accuracy
from positional_ddpm import PositionalDDPM
from solve_buddies import solve_buddies_from_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", default="0:2,10:12")
    parser.add_argument("--alphas", default="0,0.03,0.06,0.1,0.2,0.35,0.5,0.75,1")
    parser.add_argument("--max-edges", type=int, default=128)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(WORK_ROOT) / "positional_ddpm" / "positional_ddpm_train_latest.pt",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "full_graph_cache",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(WORK_ROOT) / "gates" / "spatial_edge_ranker_blend.json",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PositionalDDPM(**payload["model_args"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    _, validation_names = train_val_split()
    samples: dict[int, dict[str, torch.Tensor]] = {}
    for start, count in _parse_groups(args.groups):
        samples.update(_recreate_group(validation_names, start, count, seed=args.seed))
    alphas = [float(value) for value in args.alphas.split(",")]
    rows: dict[str, list[dict[str, float]]] = {str(alpha): [] for alpha in alphas}
    standalone: list[dict[str, float]] = []

    for image_id, sample in samples.items():
        stored = np.load(args.cache_dir / f"image_{image_id:04d}_k64.npz", allow_pickle=False)
        permutation = stored["permutation"].astype(np.int64)
        if not np.array_equal(sample["perm"].numpy(), permutation):
            raise RuntimeError(f"could not reproduce cached synthetic bag {image_id}")
        candidates = stored["candidate_ids"].astype(np.int64)
        raw = stored["candidate_scores"].reshape(NFRAG, 4, -1).astype(np.float32)
        valid = np.isfinite(raw)
        tiles = sample["tiles"].unsqueeze(0).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            features = model.encode_tiles(tiles)
            full_spatial = model.directional_edge_scores(features)[0].float().cpu().numpy()
        spatial = np.empty_like(raw)
        anchors = np.arange(NFRAG)[:, None]
        for direction in range(4):
            spatial[:, direction] = full_spatial[direction][anchors, candidates]
        raw_z = _standardize_rows(raw, valid)
        spatial_z = _standardize_rows(spatial, valid)
        truth, exists = neighbor_targets(torch.from_numpy(permutation)[None].long())
        truth = truth[0].numpy()
        exists = exists[0].numpy()
        target_board = np.argsort(permutation)

        # Spatial head alone on the exact same candidate graph.
        spatial_masked = spatial_z.copy()
        spatial_masked[~valid] = -np.inf
        spatial_top = candidates[np.arange(NFRAG)[:, None], np.argmax(spatial_masked, axis=2)]
        spatial_r1 = float((spatial_top == truth)[exists].mean())
        spatial_right, spatial_down = dense_rd(
            torch.from_numpy(candidates).long(),
            torch.from_numpy(spatial_masked).permute(1, 0, 2).contiguous(),
        )
        spatial_board, _ = solve_buddies_from_scores(
            spatial_right.numpy(), spatial_down.numpy(), max_edges=args.max_edges, repair_passes=0
        )
        standalone.append(
            {
                "image": float(image_id),
                "edge_r1": spatial_r1,
                "placement": placement_accuracy(spatial_board, target_board)[0],
                "neighbour": neighbour_accuracy(spatial_board, target_board)[0],
            }
        )

        for alpha in alphas:
            blend = raw_z + alpha * spatial_z
            blend[~valid] = -np.inf
            top = candidates[np.arange(NFRAG)[:, None], np.argmax(blend, axis=2)]
            edge_r1 = float((top == truth)[exists].mean())
            right, down = dense_rd(
                torch.from_numpy(candidates).long(),
                torch.from_numpy(blend).permute(1, 0, 2).contiguous(),
            )
            board, _ = solve_buddies_from_scores(
                right.numpy(), down.numpy(), max_edges=args.max_edges, min_margin=0.0, repair_passes=0
            )
            metric = {
                "image": float(image_id),
                "edge_r1": edge_r1,
                "placement": placement_accuracy(board, target_board)[0],
                "neighbour": neighbour_accuracy(board, target_board)[0],
            }
            rows[str(alpha)].append(metric)
        print(json.dumps({"image": image_id, "spatial": standalone[-1]}), flush=True)

    def summarize(values: list[dict[str, float]]) -> dict[str, float]:
        return {
            key: float(np.mean([row[key] for row in values]))
            for key in ("edge_r1", "placement", "neighbour")
        }

    summary = {alpha: summarize(values) for alpha, values in rows.items()}
    best_alpha = max(summary, key=lambda alpha: summary[alpha]["neighbour"])
    report = {
        "experiment": "spatial_directional_plus_candidate_ranker",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload.get("step", -1)),
        "images": len(samples),
        "groups": args.groups,
        "max_edges": args.max_edges,
        "spatial_standalone": summarize(standalone),
        "ranker_baseline": summary[str(alphas[0])],
        "best_alpha": best_alpha,
        "best": summary[best_alpha],
        "delta_vs_ranker": {
            key: summary[best_alpha][key] - summary[str(alphas[0])][key]
            for key in ("edge_r1", "placement", "neighbour")
        },
        "summary": summary,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
