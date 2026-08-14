"""P2 CAL-only direct CB1 directional score fusion.

All alpha-specific layouts are frozen before the sole allowed CAL target read.
CB1 rank confidence is added once per oriented R/D relation; rank96 local scores
and the buddies decoder otherwise remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

import infer_rank96 as rank96
from eval_candidate_rank import score_full_graph
from eval_seeded_qap import dense_rd
from solve_buddies import solve_buddies_from_scores
from train_eval_cb1_g1_capacity import sha256_file
from train_offset_pose import mine_affinity_candidates

INPUT = Path(r"E:\pazzle_data\train\inputs\img_000051.png")
TARGET = Path(r"E:\pazzle_data\train\targets\img_000051.png")
G2_LISTS = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P1_CB1_boundary_buddies\g2_cal_graph\cb1_g2_lists.npz")
SOURCE_ROOT = Path(r"C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed")
WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P2_CB1_directional_score_fusion\g0_g1_cal")
ALPHAS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40)
NFRAG = 576


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def make_config() -> rank96.InferenceConfig:
    return rank96.InferenceConfig(
        input_dir=INPUT.parent, output_dir=WORK / "unused_rank96_outputs", output_zip=None,
        ranker_checkpoint=SOURCE_ROOT / "artifacts" / "candidate_rank" / "rank_v2w64_best.pt",
        affinity_primary_checkpoint=SOURCE_ROOT / "artifacts" / "macro_affinity" / "affinity_r1_1200_best.pt",
        affinity_secondary_checkpoint=SOURCE_ROOT / "artifacts" / "macro_affinity" / "affinity_r3_1000_best.pt",
        device="cuda",
    )


def confidence_boost(cb1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if cb1.shape != (NFRAG, 4, 32):
        raise ValueError(cb1.shape)
    boost_r = np.zeros((NFRAG, NFRAG), dtype=np.float32)
    boost_d = np.zeros((NFRAG, NFRAG), dtype=np.float32)
    for anchor in range(NFRAG):
        for direction in range(4):
            for rank, candidate in enumerate(cb1[anchor, direction]):
                candidate = int(candidate)
                if candidate < 0 or candidate >= NFRAG or candidate == anchor:
                    raise ValueError((anchor, direction, rank, candidate))
                q = float(32 - rank) / 32.0
                if direction == 0:  # right
                    boost_r[anchor, candidate] = max(boost_r[anchor, candidate], q)
                elif direction == 1:  # down
                    boost_d[anchor, candidate] = max(boost_d[anchor, candidate], q)
                elif direction == 2:  # left: candidate -> anchor right
                    boost_r[candidate, anchor] = max(boost_r[candidate, anchor], q)
                elif direction == 3:  # up: candidate -> anchor down
                    boost_d[candidate, anchor] = max(boost_d[candidate, anchor], q)
    np.fill_diagonal(boost_r, 0.0)
    np.fill_diagonal(boost_d, 0.0)
    return boost_r, boost_d


def raw_ssim(target: np.ndarray, output: np.ndarray) -> float:
    return float(structural_similarity(target, output, channel_axis=2, data_range=255))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--report", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    cfg = parse_args()
    if cfg.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("P2 requires local CUDA")
    for path in (INPUT, TARGET, G2_LISTS):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(G2_LISTS, allow_pickle=False) as payload:
        cb1 = np.asarray(payload["cb1_candidates"], dtype=np.int64)
    boost_r, boost_d = confidence_boost(cb1)
    cfg.work.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    rank96._set_deterministic_runtime(20260814, device)
    models = rank96.load_models(make_config(), device)
    image = rank96.load_rgb_strict(INPUT)
    tiles = rank96.split_upright_tiles(image)
    tensor = torch.from_numpy(tiles).permute(0, 3, 1, 2).contiguous().float().to(device)
    with torch.no_grad():
        candidates, valid = mine_affinity_candidates(
            models.affinity_primary, tensor.unsqueeze(0), candidate_k=64, device=device,
            affinity_secondary=models.affinity_secondary,
        )
        candidates = candidates[0]
        valid = valid[0]
        raw = score_full_graph(models.ranker, tensor, candidates, valid, pair_batch=4096, device=device)
        right0, down0 = dense_rd(candidates, raw)
    r0 = right0.detach().cpu().numpy().astype(np.float32)
    d0 = down0.detach().cpu().numpy().astype(np.float32)
    layouts: dict[float, np.ndarray] = {}
    artifacts: dict[str, dict[str, object]] = {}
    # Target remains unread while every graph and layout is materialized.
    for alpha in ALPHAS:
        right = r0 + np.float32(alpha) * boost_r
        down = d0 + np.float32(alpha) * boost_d
        board, objective = solve_buddies_from_scores(right, down, max_edges=96, min_margin=0.0, repair_passes=0)
        board = rank96._assert_board(board)
        output = rank96.assemble_upright_tiles(tiles, board)
        artifact = cfg.work / f"alpha_{alpha:.2f}_immutable.npz"
        np.savez_compressed(artifact, alpha=np.float32(alpha), right=right, down=down, board=board, output=output)
        layouts[alpha] = output
        artifacts[f"{alpha:.2f}"] = {
            "artifact": str(artifact), "artifact_sha256": sha256_file(artifact), "board_sha256": sha256_array(board), "objective": float(objective),
        }
    target = rank96.load_rgb_strict(TARGET)
    ssim = {f"{alpha:.2f}": raw_ssim(target, layouts[alpha]) for alpha in ALPHAS}
    maximum = max(ssim.values())
    selected = min(alpha for alpha in ALPHAS if ssim[f"{alpha:.2f}"] == maximum)
    report = {
        "experiment": "P2_CB1_directional_score_fusion", "gate": "G0_G1_CAL_only",
        "input": str(INPUT), "input_sha256": sha256_file(INPUT), "target": str(TARGET), "target_sha256": sha256_file(TARGET),
        "g2_lists": str(G2_LISTS), "g2_lists_sha256": sha256_file(G2_LISTS),
        "alphas": list(ALPHAS), "ssim": ssim, "selected_alpha": selected,
        "boost_r_sha256": sha256_array(boost_r), "boost_d_sha256": sha256_array(boost_d), "baseline_r_sha256": sha256_array(r0), "baseline_d_sha256": sha256_array(d0),
        "artifacts": artifacts, "target_images_opened": [TARGET.name], "DEV_targets_opened": False, "restorer_used": False, "test_accessed": False,
        "passes_G1": bool(selected > 0.0 and ssim[f"{selected:.2f}"] >= ssim["0.00"]),
    }
    report["decision"] = "advance_to_P2_G2_DEV" if report["passes_G1"] else "reject_P2_before_DEV"
    destination = cfg.report or cfg.work / "p2_g0_g1_report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
