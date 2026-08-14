"""P1/CB1 G4: CAL-only ranker-rescored candidate expansion selection.

CB1 candidate identities are supplied by frozen G2 artifacts. For each capacity,
the unchanged CandidateSeamRanker rescoring and buddies solver produce an
immutable raw layout before the sole CAL target is read to select capacity.
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

INPUT = Path(r"E:\pazzle_data\train\inputs\img_000051.png")
TARGET = Path(r"E:\pazzle_data\train\targets\img_000051.png")
G2_LISTS = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P1_CB1_boundary_buddies\g2_cal_graph\cb1_g2_lists.npz")
SOURCE_ROOT = Path(r"C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed")
WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P1_CB1_boundary_buddies\g4_cal_rescored")
CAPACITIES = (0, 16, 32, 48)
NFRAG = 576
WIDTH = 128


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def make_config() -> rank96.InferenceConfig:
    return rank96.InferenceConfig(
        input_dir=INPUT.parent,
        output_dir=WORK / "unused_rank96_outputs",
        output_zip=None,
        ranker_checkpoint=SOURCE_ROOT / "artifacts" / "candidate_rank" / "rank_v2w64_best.pt",
        affinity_primary_checkpoint=SOURCE_ROOT / "artifacts" / "macro_affinity" / "affinity_r1_1200_best.pt",
        affinity_secondary_checkpoint=SOURCE_ROOT / "artifacts" / "macro_affinity" / "affinity_r3_1000_best.pt",
        device="cuda",
    )


def valid_unique_rows(ids: np.ndarray) -> np.ndarray:
    valid = np.zeros(ids.shape, dtype=bool)
    for anchor in range(NFRAG):
        seen: set[int] = set()
        for index, candidate in enumerate(ids[anchor]):
            value = int(candidate)
            if 0 <= value < NFRAG and value != anchor and value not in seen:
                valid[anchor, index] = True
                seen.add(value)
    return valid


def ranked_novelties(base_ids: np.ndarray, base_valid: np.ndarray, cb1_ids: np.ndarray) -> list[list[int]]:
    if cb1_ids.shape != (NFRAG, 4, 32):
        raise ValueError(cb1_ids.shape)
    output: list[list[int]] = []
    for anchor in range(NFRAG):
        known = {int(x) for x, ok in zip(base_ids[anchor], base_valid[anchor]) if bool(ok)}
        confidence: dict[int, float] = {}
        for direction in range(4):
            for rank, candidate in enumerate(cb1_ids[anchor, direction]):
                candidate = int(candidate)
                if candidate == anchor or candidate in known:
                    continue
                score = float(32 - rank) / 32.0
                confidence[candidate] = max(confidence.get(candidate, -1.0), score)
        output.append([candidate for candidate, _ in sorted(confidence.items(), key=lambda pair: (-pair[1], pair[0]))])
    return output


def expanded_storage(base_ids: np.ndarray, base_valid: np.ndarray, novelties: list[list[int]], capacity: int) -> tuple[np.ndarray, np.ndarray]:
    if capacity not in CAPACITIES:
        raise ValueError(capacity)
    ids = np.zeros((NFRAG, WIDTH), dtype=np.int64)
    valid = np.zeros((NFRAG, WIDTH), dtype=bool)
    for anchor in range(NFRAG):
        frozen = [int(x) for x, ok in zip(base_ids[anchor], base_valid[anchor]) if bool(ok)]
        chosen = frozen + novelties[anchor][:capacity]
        chosen = list(dict.fromkeys(chosen))
        if len(chosen) > WIDTH:
            chosen = chosen[:WIDTH]
        if anchor in chosen or not chosen:
            raise RuntimeError(f"invalid expanded row {anchor}")
        ids[anchor, :len(chosen)] = np.asarray(chosen, dtype=np.int64)
        valid[anchor, :len(chosen)] = True
    return ids, valid


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
        raise RuntimeError("G4 requires local CUDA")
    for path in (INPUT, TARGET, G2_LISTS):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(G2_LISTS, allow_pickle=False) as data:
        base_ids = np.asarray(data["union_candidates"], dtype=np.int64)
        cb1_ids = np.asarray(data["cb1_candidates"], dtype=np.int64)
    if base_ids.shape[0] != NFRAG:
        raise ValueError(base_ids.shape)
    base_valid = valid_unique_rows(base_ids)
    novelties = ranked_novelties(base_ids, base_valid, cb1_ids)
    cfg.work.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    rank96._set_deterministic_runtime(20260814, device)
    models = rank96.load_models(make_config(), device)
    image = rank96.load_rgb_strict(INPUT)
    tiles = rank96.split_upright_tiles(image)
    tiles_tensor = torch.from_numpy(tiles).permute(0, 3, 1, 2).contiguous().float().to(device)
    layouts: dict[int, np.ndarray] = {}
    provenance: dict[int, dict[str, object]] = {}
    # Target is intentionally unopened throughout graph scoring and layout construction.
    with torch.no_grad():
        for capacity in CAPACITIES:
            candidate_ids, valid = expanded_storage(base_ids, base_valid, novelties, capacity)
            candidate_tensor = torch.from_numpy(candidate_ids).to(device)
            valid_tensor = torch.from_numpy(valid).to(device)
            raw = score_full_graph(models.ranker, tiles_tensor, candidate_tensor, valid_tensor, pair_batch=4096, device=device)
            right, down = dense_rd(candidate_tensor, raw)
            board, objective = solve_buddies_from_scores(
                right.detach().cpu().numpy().astype(np.float32),
                down.detach().cpu().numpy().astype(np.float32),
                max_edges=96, min_margin=0.0, repair_passes=0,
            )
            board = rank96._assert_board(board)
            layout = rank96.assemble_upright_tiles(tiles, board)
            artifact = cfg.work / f"cal_capacity_{capacity:02d}_immutable.npz"
            np.savez_compressed(artifact, candidate_ids=candidate_ids, valid=valid, board=board, output=layout)
            layouts[capacity] = layout
            provenance[capacity] = {
                "artifact": str(artifact), "artifact_sha256": sha256_file(artifact), "candidate_ids_sha256": sha256_array(candidate_ids),
                "valid_sha256": sha256_array(valid), "board_sha256": sha256_array(board), "objective": float(objective),
            }
    # The only permitted target read in G4 occurs after all four immutable artifacts exist.
    target = rank96.load_rgb_strict(TARGET)
    scores = {capacity: raw_ssim(target, layouts[capacity]) for capacity in CAPACITIES}
    maximum = max(scores.values())
    selected_capacity = min(capacity for capacity in CAPACITIES if scores[capacity] == maximum)
    report = {
        "experiment": "P1_CB1_boundary_buddies", "gate": "G4_CAL_ranker_rescored_capacity",
        "input": str(INPUT), "input_sha256": sha256_file(INPUT), "target": str(TARGET), "target_sha256": sha256_file(TARGET),
        "g2_lists": str(G2_LISTS), "g2_lists_sha256": sha256_file(G2_LISTS),
        "ranker_checkpoint": str(make_config().ranker_checkpoint), "ranker_checkpoint_sha256": sha256_file(make_config().ranker_checkpoint),
        "capacities": list(CAPACITIES), "ssim": scores, "selected_capacity": selected_capacity, "artifacts": provenance,
        "target_images_opened": [TARGET.name], "DEV_targets_opened": False, "layouts_assembled": True, "restorer_used": False, "test_accessed": False,
        "decision": "freeze_capacity_for_CB1_G5_DEV" if selected_capacity in CAPACITIES else "reject_CB1",
    }
    destination = cfg.report or cfg.work / "cb1_g4_report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
