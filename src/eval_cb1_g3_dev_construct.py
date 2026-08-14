"""P1/CB1 G3: target-safe eight-board DEV candidate construction.

Uses raw DEV inputs plus frozen rank96 affinity encoders and CB1 only. It writes
candidate provenance, never reads targets/permutations/labels or assembles a
layout. Coverage and layout quality remain sealed for later gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import infer_rank96 as rank96
from eval_cb1_g2_cal_graph import directional_l1_order, score_candidates
from train_eval_cb1_g1_capacity import BoundaryBuddyNet, sha256_file
from train_offset_pose import mine_affinity_candidates

INPUTS = Path(r"E:\pazzle_data\train\inputs")
SOURCE_ROOT = Path(r"C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed")
CB1_CHECKPOINT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P1_CB1_boundary_buddies\full_fit\cb1_full_fit.pt")
WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P1_CB1_boundary_buddies\g3_dev_construct")
DEV_IDS = (8, 14, 20, 33, 48, 57, 64, 81)
NFRAG = 576


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def make_rank96_config() -> rank96.InferenceConfig:
    return rank96.InferenceConfig(
        input_dir=INPUTS,
        output_dir=WORK / "unused_rank96_outputs",
        output_zip=None,
        ranker_checkpoint=SOURCE_ROOT / "artifacts" / "candidate_rank" / "rank_v2w64_best.pt",
        affinity_primary_checkpoint=SOURCE_ROOT / "artifacts" / "macro_affinity" / "affinity_r1_1200_best.pt",
        affinity_secondary_checkpoint=SOURCE_ROOT / "artifacts" / "macro_affinity" / "affinity_r3_1000_best.pt",
        device="cuda",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--report", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    cfg = parse_args()
    if cfg.device != "cuda":
        raise ValueError("G3 is frozen to the local RTX 2070 CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if not CB1_CHECKPOINT.is_file():
        raise FileNotFoundError(CB1_CHECKPOINT)
    inputs = [INPUTS / f"img_{identifier:06d}.png" for identifier in DEV_IDS]
    if not all(path.is_file() for path in inputs):
        missing = [str(path) for path in inputs if not path.is_file()]
        raise FileNotFoundError(missing)
    device = torch.device("cuda")
    rank96._set_deterministic_runtime(20260814, device)
    rank_models = rank96.load_models(make_rank96_config(), device)
    cb1_state = torch.load(CB1_CHECKPOINT, map_location=device, weights_only=False)
    cb1 = BoundaryBuddyNet().to(device)
    cb1.load_state_dict(cb1_state["state_dict"], strict=True)
    cb1.eval()
    cfg.work.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for ordinal, path in enumerate(inputs, start=1):
            image = rank96.load_rgb_strict(path)
            tiles = rank96.split_upright_tiles(image)
            tensor = torch.from_numpy(tiles).permute(0, 3, 1, 2).contiguous().float().to(device).unsqueeze(0)
            frozen, valid = mine_affinity_candidates(
                rank_models.affinity_primary, tensor, candidate_k=64, device=device,
                affinity_secondary=rank_models.affinity_secondary,
            )
            frozen_ids = frozen[0].detach().cpu().numpy().astype(np.int64, copy=False)
            frozen_valid = valid[0].detach().cpu().numpy().astype(bool, copy=False)
            if frozen_ids.shape != (NFRAG, 128) or frozen_valid.shape != (NFRAG, 128):
                raise RuntimeError(f"unexpected frozen candidate shapes for {path.name}")
            cb1_ids = np.full((NFRAG, 4, 32), -1, dtype=np.int64)
            cb1_scores = np.full((NFRAG, 4, 32), np.nan, dtype=np.float32)
            for anchor in range(NFRAG):
                frozen_list = [int(candidate) for candidate, is_valid in zip(frozen_ids[anchor], frozen_valid[anchor]) if bool(is_valid)]
                for direction in range(4):
                    l1 = [int(candidate) for candidate in directional_l1_order(tiles, anchor, direction)[:128]]
                    pool = list(dict.fromkeys(frozen_list + l1))
                    if anchor in pool or len(pool) < 32:
                        raise RuntimeError(f"invalid target-free pool {path.name} anchor={anchor} direction={direction}")
                    scores = score_candidates(cb1, tiles, anchor, pool, direction, device, 4096)
                    selected = np.argsort(-scores, kind="stable")[:32]
                    cb1_ids[anchor, direction] = np.asarray([pool[int(index)] for index in selected], dtype=np.int64)
                    cb1_scores[anchor, direction] = scores[selected].astype(np.float32)
            if np.any(cb1_ids < 0) or np.any(cb1_ids == np.arange(NFRAG)[:, None, None]) or not np.isfinite(cb1_scores).all():
                raise RuntimeError(f"invalid CB1 artifact for {path.name}")
            output = cfg.work / f"{path.stem}_cb1_g3.npz"
            np.savez_compressed(output, frozen_candidate_ids=frozen_ids, frozen_valid=frozen_valid, cb1_candidate_ids=cb1_ids, cb1_scores=cb1_scores)
            rows.append({
                "ordinal": ordinal, "input": str(path), "input_sha256": sha256_file(path),
                "artifact": str(output), "artifact_sha256": sha256_file(output),
                "frozen_ids_sha256": sha256_array(frozen_ids), "frozen_valid_sha256": sha256_array(frozen_valid),
                "cb1_ids_sha256": sha256_array(cb1_ids), "cb1_scores_sha256": sha256_array(cb1_scores),
                "shape": [NFRAG, 4, 32],
            })
            print(json.dumps({"ordinal": ordinal, "input": path.name, "artifact": output.name}), flush=True)
    report = {
        "experiment": "P1_CB1_boundary_buddies", "gate": "G3_target_safe_pinned_DEV_candidate_construction",
        "dev_inputs": [path.name for path in inputs], "rows": rows,
        "cb1_checkpoint": str(CB1_CHECKPOINT), "cb1_checkpoint_sha256": sha256_file(CB1_CHECKPOINT),
        "rank96_checkpoints": {"ranker": str(make_rank96_config().ranker_checkpoint), "affinity_primary": str(make_rank96_config().affinity_primary_checkpoint), "affinity_secondary": str(make_rank96_config().affinity_secondary_checkpoint)},
        "target_images_opened": False, "permutations_opened": False, "labels_opened": False, "layouts_assembled": False, "restorer_used": False, "test_accessed": False,
        "passes_G3": len(rows) == len(DEV_IDS), "decision": "advance_to_CB1_G4_immutable_layouts" if len(rows) == len(DEV_IDS) else "reject_CB1_before_layouts",
    }
    destination = cfg.report or cfg.work / "cb1_g3_report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
