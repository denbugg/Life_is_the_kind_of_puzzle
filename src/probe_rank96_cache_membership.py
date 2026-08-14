from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import infer_rank96 as rank96
from eval_r10a_frozen_layout import capture_canonical_scores

INPUT = Path(r"E:\pazzle_data\train\inputs\img_000051.png")
CACHE = Path(r"E:\pazzle_work\edge_confidence\full_graph_cache\image_0051_k64.npz")


def candidates_from_scores(right: np.ndarray, down: np.ndarray, width: int = 128) -> np.ndarray:
    n = right.shape[0]
    if right.shape != (n, n) or down.shape != (n, n):
        raise ValueError((right.shape, down.shape))
    out = np.empty((n, width), dtype=np.int64)
    for anchor in range(n):
        scores = np.maximum.reduce((right[anchor], down[anchor], right[:, anchor], down[:, anchor]))
        scores[anchor] = -np.inf
        out[anchor] = np.argsort(-scores, kind="stable")[:width]
    return out


device = torch.device("cuda")
rank96._set_deterministic_runtime(20260814, device)
source_root = Path(r"C:\\Users\\pasha\\Documents\\GitHub\\pazzle_will_be_killed")
config = rank96.InferenceConfig(
    input_dir=Path(r"E:\\pazzle_data\\train\\inputs"), output_dir=Path(r"E:\\pazzle_work\\pazzle_fixed_orientation_20260813\\P1_CB1_boundary_buddies\\probe_unused"), output_zip=None,
    ranker_checkpoint=source_root / "artifacts" / "candidate_rank" / "rank_v2w64_best.pt",
    affinity_primary_checkpoint=source_root / "artifacts" / "macro_affinity" / "affinity_r1_1200_best.pt",
    affinity_secondary_checkpoint=source_root / "artifacts" / "macro_affinity" / "affinity_r3_1000_best.pt",
    device="cuda",
)
models = rank96.load_models(config, device)
image = rank96.load_rgb_strict(INPUT)
_, right, down = capture_canonical_scores(image, models, 4096)
derived = candidates_from_scores(right, down)
with np.load(CACHE, allow_pickle=False) as cache:
    frozen = np.asarray(cache["candidate_ids"], dtype=np.int64)
intersections = np.array([len(set(map(int, frozen[a])) & set(map(int, derived[a]))) for a in range(576)])
print({"mean_intersection": float(intersections.mean()), "mean_jaccard": float(np.mean(intersections / np.array([len(set(map(int, frozen[a])) | set(map(int, derived[a]))) for a in range(576)]))), "exact_rows": int(np.sum(intersections == 128))})
