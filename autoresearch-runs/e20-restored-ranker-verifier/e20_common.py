"""Shared, target-free score construction for the locked E20 verifier."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]

import kaggle_e14_solver as e14
from train_restored_border_ranker import descriptors

EXPECTED_CACHE_SHA256 = "74db2b62e9d5eafffae33117c7771512d823b0dcaa0095ef5807adb8e86a25df"
EXPECTED_RESTORER_SHA256 = "6fcc7de2cf8063b4f2f45d4b96b8999d5eb9c29a071ff2c0031d2703c70d6695"
EXPECTED_RANKER_SHA256 = "8eb7b7e106c0333b9a099f88894eac7b1081555643d3828e479aaf4e56137be1"
TOP_K = 32
BONUS_WEIGHT = 0.25
Z_CLIP = 4.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(cache_path: Path, sidecar_path: Path):
    cache_hash = sha256(cache_path)
    sidecar_hash = sha256(sidecar_path)
    if cache_hash != EXPECTED_CACHE_SHA256:
        raise ValueError(f"cache hash mismatch: {cache_hash}")
    data = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    sidecar = np.load(sidecar_path, mmap_mode="r", allow_pickle=False)
    provenance = json.loads(str(sidecar["provenance_json"]))
    if provenance["source_cache_sha256"] != cache_hash:
        raise ValueError("sidecar cache provenance mismatch")
    if provenance["checkpoint_sha256"] != EXPECTED_RESTORER_SHA256:
        raise ValueError("sidecar restorer provenance mismatch")
    if provenance["residual_multiplier"] != 0.5:
        raise ValueError("sidecar restorer architecture mismatch")
    if not np.array_equal(data["stems"], sidecar["stems"]):
        raise ValueError("sidecar stem order mismatch")
    return data, sidecar, provenance, cache_hash, sidecar_hash


def topk_high(matrix: np.ndarray, k: int = TOP_K) -> np.ndarray:
    scores = np.asarray(matrix).copy()
    np.fill_diagonal(scores, -np.inf)
    ids = np.argpartition(scores, -k, axis=1)[:, -k:]
    row = np.arange(len(scores))[:, None]
    order = np.argsort(-scores[row, ids], axis=1, kind="stable")
    return ids[row, order].astype(np.int32)


@torch.inference_mode()
def descriptor_topk(restored: np.ndarray, direction: int,
                    k: int = TOP_K) -> np.ndarray:
    tensor = torch.from_numpy(
        np.ascontiguousarray(restored.transpose(0, 3, 1, 2))
    ).float().div_(255.0)
    anchor, candidate = descriptors(tensor, direction)
    distance = torch.cdist(anchor, candidate).square().div_(anchor.shape[1])
    distance.fill_diagonal_(float("inf"))
    return distance.topk(k, largest=False, dim=1).indices.numpy().astype(np.int32)


def candidate_union(e14_scores: np.ndarray, restored: np.ndarray,
                    direction: int) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    e14_ids = topk_high(e14_scores)
    descriptor_ids = descriptor_topk(restored, direction)
    unions = []
    for row in range(e14.N):
        ids = np.unique(np.concatenate((e14_ids[row], descriptor_ids[row])))
        ids = ids[ids != row].astype(np.int32, copy=False)
        unions.append(ids)
    return unions, e14_ids, descriptor_ids


def truth_neighbours(truth: np.ndarray, direction: int):
    board = np.asarray(truth, np.int32).reshape(e14.GRID, e14.GRID)
    if direction == 0:
        return board[:, :-1].reshape(-1), board[:, 1:].reshape(-1)
    return board[:-1].reshape(-1), board[1:].reshape(-1)


def coverage_counts(candidates: list[np.ndarray], truth: np.ndarray,
                    direction: int) -> tuple[int, int]:
    anchors, neighbours = truth_neighbours(truth, direction)
    hits = sum(int(neighbour in candidates[int(anchor)])
               for anchor, neighbour in zip(anchors, neighbours))
    return hits, len(anchors)
