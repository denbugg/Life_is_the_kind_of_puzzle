"""R11-G2: paired raw-layout SSIM on eight pre-registered unseen DEV boards.

This is blocked unless R11-G1 passed. It imports G1's fixed lambda, captures the
unchanged rank96 R/D matrices for all eight inputs, and chooses every R11 layout
without target access. DEV targets are opened only after all choices exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as sk_ssim

import infer_rank96 as rank96
from eval_r11_g1_cal import capture_scores, sha256_array, sha256_file
from eval_r11_rank_loop_consensus import candidate_scores, generate_layouts, select_layout

INPUTS = Path(r"E:\pazzle_data\train\inputs")
TARGETS = Path(r"E:\pazzle_data\train\targets")
G1_REPORT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R11_rank_loop_consensus\g1_cal\r11_g1_report.json")
WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R11_rank_loop_consensus\g2_dev")
DEV_NAMES = (
    "img_000008.png", "img_000014.png", "img_000020.png", "img_000033.png",
    "img_000048.png", "img_000057.png", "img_000064.png", "img_000081.png",
)


def write_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path)


def lower_95(values: np.ndarray) -> float:
    if len(values) < 2:
        return float(values.mean())
    return float(values.mean() - 1.96 * values.std(ddof=1) / math.sqrt(len(values)))


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, default=INPUTS)
    p.add_argument("--targets", type=Path, default=TARGETS)
    p.add_argument("--g1-report", type=Path, default=G1_REPORT)
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--pair-batch", type=int, default=4096)
    return p.parse_args()


def main() -> None:
    cfg = args()
    g1: Dict[str, Any] = json.loads(cfg.g1_report.read_text(encoding="utf-8"))
    if not bool(g1.get("passes_G1")) or g1.get("decision") != "advance_to_R11_G2_DEV":
        raise RuntimeError("R11-G2 blocked: frozen G1 report did not pass")
    lam = float(g1["selected_lambda"])
    if lam not in (0.0, 0.25, 0.5, 1.0, 2.0):
        raise RuntimeError("R11-G2 blocked: selected lambda is not in the pre-registered grid")
    for name in DEV_NAMES:
        if not (cfg.inputs / name).is_file() or not (cfg.targets / name).is_file():
            raise FileNotFoundError(f"missing R11-G2 pinned DEV board: {name}")

    cfg.work.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    rank96._set_deterministic_runtime(20260814, device)
    checkpoints = rank96._default_checkpoints()
    frozen_config = rank96.InferenceConfig(
        input_dir=cfg.inputs, output_dir=cfg.work, output_zip=None,
        ranker_checkpoint=checkpoints["ranker"],
        affinity_primary_checkpoint=checkpoints["affinity_primary"],
        affinity_secondary_checkpoint=checkpoints["affinity_secondary"],
        device=cfg.device, seed=20260814, pair_batch=cfg.pair_batch, expected_count=0,
    )
    models = rank96.load_models(frozen_config, device)

    rows: List[Dict[str, Any]] = []
    # Target-free score capture and selection for every board completes before the target loop below.
    for name in DEV_NAMES:
        image = rank96.load_rgb_strict(cfg.inputs / name)
        tiles = rank96.split_upright_tiles(image)
        inferred, right, down = capture_scores(image, models, cfg.pair_batch)
        layouts = generate_layouts(right, down)
        edge, loop = candidate_scores(layouts, right, down)
        selected_index, objective = select_layout(layouts, edge, loop, lam)
        canonical = rank96.assemble_upright_tiles(tiles, layouts[0])
        selected = rank96.assemble_upright_tiles(tiles, layouts[selected_index])
        stem = Path(name).stem
        canonical_path = cfg.work / f"{stem}_canonical_raw.png"
        selected_path = cfg.work / f"{stem}_r11_raw.png"
        write_rgb(canonical_path, canonical)
        write_rgb(selected_path, selected)
        rows.append({
            "name": name,
            "selected_layout_index": int(selected_index),
            "canonical_layout_index": 0,
            "canonical_path": canonical_path.name,
            "r11_path": selected_path.name,
            "selected_edge_score": float(edge[selected_index]),
            "selected_loop_score": float(loop[selected_index]),
            "selected_objective": float(objective[selected_index]),
            "canonical_edge_score": float(edge[0]),
            "canonical_loop_score": float(loop[0]),
            "canonical_objective": float(objective[0]),
            "score_delta_vs_canonical": float(objective[selected_index] - objective[0]),
            "input_sha256": sha256_file(cfg.inputs / name),
            "right_scores_sha256": sha256_array(right),
            "down_scores_sha256": sha256_array(down),
            "canonical_solver_objective": float(inferred.objective),
        })

    # Targets may only be consumed after selections above are immutable.
    deltas: List[float] = []
    for row in rows:
        name = str(row["name"])
        target = rank96.load_rgb_strict(cfg.targets / name)
        canonical = rank96.load_rgb_strict(cfg.work / str(row["canonical_path"]))
        selected = rank96.load_rgb_strict(cfg.work / str(row["r11_path"]))
        base = float(sk_ssim(target, canonical, channel_axis=2, data_range=255))
        value = float(sk_ssim(target, selected, channel_axis=2, data_range=255))
        row["target_sha256"] = sha256_file(cfg.targets / name)
        row["canonical_raw_ssim"] = base
        row["r11_raw_ssim"] = value
        row["paired_delta"] = value - base
        deltas.append(value - base)

    values = np.asarray(deltas, dtype=np.float64)
    report = {
        "experiment": "R11_rank_loop_consensus",
        "gate": "G2_paired_raw_layout_DEV_SSIM",
        "frozen_g1_report": str(cfg.g1_report),
        "frozen_selected_lambda": lam,
        "dev_names": list(DEV_NAMES),
        "rows": rows,
        "paired_mean_delta": float(values.mean()),
        "paired_lower_95_delta": lower_95(values),
        "metric": "skimage.metrics.structural_similarity(channel_axis=2,data_range=255)",
        "provenance": {
            "layout_count": 32, "max_edges": 96, "random_seed_base": 20260814,
            "temperature": 0.03, "order_jitter": 0.25, "fixed_orientation": True,
            "all_target_independent_selections_completed_before_targets": True,
        },
        "targets_opened": list(DEV_NAMES),
    }
    report["passes_G2"] = bool(report["paired_mean_delta"] > 0.0 and report["paired_lower_95_delta"] > 0.0)
    report["decision"] = "advance_to_R11_G3_R5NLM" if report["passes_G2"] else "reject_R11_before_R5NLM_test_submission"
    report_path = cfg.report or cfg.work / "r11_g2_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
