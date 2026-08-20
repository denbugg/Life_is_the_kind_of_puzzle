"""Locked sparse restored-ranker critic used by E20."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(HERE)]

import kaggle_e14_solver as e14
from e20_common import BONUS_WEIGHT, Z_CLIP, candidate_union
from train_restored_border_ranker import seam_features


def robust_row_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float32)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return np.clip((values - median) / max(float(mad), 1e-6),
                   -Z_CLIP, Z_CLIP).astype(np.float32)


@torch.inference_mode()
def verified_scores(model, restored: np.ndarray, good: np.ndarray,
                    e14_scores: np.ndarray, direction: int,
                    device: torch.device, batch_size: int = 4096):
    """Add the locked sparse bonus, evaluating the ranker only on the union."""
    restored = np.asarray(restored, np.uint8)
    good = np.asarray(good, np.bool_)
    if restored.shape != (e14.N, e14.TILE, e14.TILE, 3):
        raise ValueError(f"invalid restored shape: {restored.shape}")
    if good.shape != (e14.N,):
        raise ValueError(f"invalid good-mask shape: {good.shape}")
    unions, e14_ids, descriptor_ids = candidate_union(
        e14_scores, restored, direction
    )
    anchors = np.concatenate([
        np.full(len(ids), row, np.int32) for row, ids in enumerate(unions)
    ])
    candidates = np.concatenate(unions).astype(np.int32, copy=False)
    tensor = torch.from_numpy(
        np.ascontiguousarray(restored.transpose(0, 3, 1, 2))
    ).float().div_(255.0).to(device)
    ranker_values = []
    for start in range(0, len(anchors), batch_size):
        anchor_ids = torch.from_numpy(anchors[start:start + batch_size]).to(device)
        candidate_ids = torch.from_numpy(candidates[start:start + batch_size]).to(device)
        features = seam_features(
            tensor[anchor_ids], tensor[candidate_ids], direction
        )
        ranker_values.append(model(features).float().cpu().numpy())
    ranker_values = np.concatenate(ranker_values)

    output = np.asarray(e14_scores, np.float32).copy()
    cursor = 0
    active_pairs = 0
    z_min, z_max = np.inf, -np.inf
    for row, ids in enumerate(unions):
        count = len(ids)
        z = robust_row_z(ranker_values[cursor:cursor + count])
        enabled = good[row] & good[ids]
        output[row, ids] += BONUS_WEIGHT * enabled.astype(np.float32) * z
        active_pairs += int(enabled.sum())
        z_min = min(z_min, float(z.min()))
        z_max = max(z_max, float(z.max()))
        cursor += count
    np.fill_diagonal(output, -1e4)
    stats = {
        "candidate_pairs": int(len(anchors)),
        "active_good_pairs": active_pairs,
        "mean_union_size": float(np.mean([len(ids) for ids in unions])),
        "min_union_size": int(min(map(len, unions))),
        "max_union_size": int(max(map(len, unions))),
        "ranker_z_min": z_min,
        "ranker_z_max": z_max,
        "e14_topk": int(e14_ids.shape[1]),
        "descriptor_topk": int(descriptor_ids.shape[1]),
    }
    return output, stats
