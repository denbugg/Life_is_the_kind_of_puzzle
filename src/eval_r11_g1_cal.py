"""R11-G1: one-board CAL calibration for rank-normalized loop consensus.

This evaluator captures unchanged frozen rank96 R/D matrices for exactly the
pre-registered CAL board. It makes every layout selection without a target, then
opens that one CAL target solely to choose the smallest lambda on the fixed grid
that reaches maximal raw-layout SSIM. No DEV input or target is touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as sk_ssim

import infer_rank96 as rank96
from eval_r11_rank_loop_consensus import candidate_scores, generate_layouts, select_layout

INPUTS = Path(r"E:\pazzle_data\train\inputs")
TARGETS = Path(r"E:\pazzle_data\train\targets")
SPLIT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json")
WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R11_rank_loop_consensus\g1_cal")
CAL_NAME = "img_000051.png"
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def write_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path)


def capture_scores(image: np.ndarray, models: rank96.LoadedModels, pair_batch: int) -> Tuple[Any, np.ndarray, np.ndarray]:
    captured: Dict[str, np.ndarray] = {}
    original = rank96.solve_dense_tiles

    def spy(tiles: np.ndarray, right: np.ndarray, down: np.ndarray, solver=None, restorer=None):
        captured["right"] = np.asarray(right, dtype=np.float32).copy()
        captured["down"] = np.asarray(down, dtype=np.float32).copy()
        return original(tiles, right, down, solver=solver, restorer=restorer)

    rank96.solve_dense_tiles = spy
    try:
        inferred = rank96.infer_one(image, models, pair_batch=pair_batch)
    finally:
        rank96.solve_dense_tiles = original
    if set(captured) != {"right", "down"}:
        raise RuntimeError("failed to capture canonical rank96 R/D matrices")
    return inferred, captured["right"], captured["down"]


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, default=INPUTS)
    p.add_argument("--targets", type=Path, default=TARGETS)
    p.add_argument("--split", type=Path, default=SPLIT)
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--pair-batch", type=int, default=4096)
    return p.parse_args()


def main() -> None:
    cfg = args()
    input_path, target_path = cfg.inputs / CAL_NAME, cfg.targets / CAL_NAME
    if not input_path.is_file() or not target_path.is_file() or not cfg.split.is_file():
        raise FileNotFoundError("R11-G1 requires the frozen CAL input/target and pinned split manifest")
    manifest = json.loads(cfg.split.read_text(encoding="utf-8"))
    if CAL_NAME not in json.dumps(manifest, sort_keys=True):
        raise RuntimeError("pre-registered CAL source is absent from pinned split manifest")

    device = torch.device(cfg.device)
    rank96._set_deterministic_runtime(20260814, device)
    checkpoints = rank96._default_checkpoints()
    frozen_config = rank96.InferenceConfig(
        input_dir=cfg.inputs,
        output_dir=cfg.work,
        output_zip=None,
        ranker_checkpoint=checkpoints["ranker"],
        affinity_primary_checkpoint=checkpoints["affinity_primary"],
        affinity_secondary_checkpoint=checkpoints["affinity_secondary"],
        device=cfg.device,
        seed=20260814,
        pair_batch=cfg.pair_batch,
        expected_count=0,
    )
    models = rank96.load_models(frozen_config, device)
    image = rank96.load_rgb_strict(input_path)
    tiles = rank96.split_upright_tiles(image)
    inferred, right, down = capture_scores(image, models, cfg.pair_batch)
    layouts = generate_layouts(right, down)
    edge, loop = candidate_scores(layouts, right, down)

    # No target was read until score capture, layouts, and target-independent objectives were complete.
    target = rank96.load_rgb_strict(target_path)
    raw_images = [rank96.assemble_upright_tiles(tiles, layout) for layout in layouts]
    canonical_ssim = float(sk_ssim(target, raw_images[0], channel_axis=2, data_range=255))
    lambda_rows = []
    for lam in LAMBDAS:
        selected_index, objective = select_layout(layouts, edge, loop, lam)
        selected_ssim = float(sk_ssim(target, raw_images[selected_index], channel_axis=2, data_range=255))
        lambda_rows.append({
            "lambda": lam,
            "selected_index": selected_index,
            "selected_raw_ssim": selected_ssim,
            "selected_edge_score": float(edge[selected_index]),
            "selected_loop_score": float(loop[selected_index]),
            "selected_objective": float(objective[selected_index]),
        })
    best_ssim = max(row["selected_raw_ssim"] for row in lambda_rows)
    chosen = next(row for row in lambda_rows if row["selected_raw_ssim"] == best_ssim)
    selected_index = int(chosen["selected_index"])

    cfg.work.mkdir(parents=True, exist_ok=True)
    write_rgb(cfg.work / "img_000051_canonical_raw.png", raw_images[0])
    write_rgb(cfg.work / "img_000051_r11_selected_raw.png", raw_images[selected_index])
    report = {
        "experiment": "R11_rank_loop_consensus",
        "gate": "G1_single_CAL_lambda_calibration",
        "cal_name": CAL_NAME,
        "selected_lambda": float(chosen["lambda"]),
        "selected_layout_index": selected_index,
        "canonical_layout_index": 0,
        "canonical_raw_ssim": canonical_ssim,
        "selected_raw_ssim": float(chosen["selected_raw_ssim"]),
        "selected_minus_canonical_raw_ssim": float(chosen["selected_raw_ssim"] - canonical_ssim),
        "lambda_grid": lambda_rows,
        "all_layouts": [{"index": int(i), "edge_score": float(edge[i]), "loop_score": float(loop[i])} for i in range(len(layouts))],
        "provenance": {
            "layout_count": 32,
            "max_edges": 96,
            "random_seed_base": 20260814,
            "temperature": 0.03,
            "order_jitter": 0.25,
            "input_sha256": sha256_file(input_path),
            "target_sha256": sha256_file(target_path),
            "split_manifest_sha256": sha256_file(cfg.split),
            "right_scores_sha256": sha256_array(right),
            "down_scores_sha256": sha256_array(down),
            "canonical_solver_objective": float(inferred.objective),
            "fixed_orientation": True,
            "DEV_accessed": False,
        },
        "targets_opened": [CAL_NAME],
        "metric": "skimage.metrics.structural_similarity(channel_axis=2,data_range=255)",
    }
    report["passes_G1"] = bool(report["selected_raw_ssim"] >= report["canonical_raw_ssim"])
    report["decision"] = "advance_to_R11_G2_DEV" if report["passes_G1"] else "reject_R11_before_DEV"
    report_path = cfg.report or cfg.work / "r11_g1_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
