"""Build a resumable confidence-gated source-aware test submission."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import evaluate_submit_border_pipeline as pipeline
from global_solver_candidate import solve_layout
from source_aware_ablation import assemble, build_gallery, hungarian_layout, set_desc, standardize_gallery


def source_candidate(
    restored: np.ndarray,
    files: np.ndarray,
    gallery_raw: np.ndarray,
    topk: int,
) -> tuple[np.ndarray, dict]:
    gallery, query = standardize_gallery(gallery_raw, set_desc(restored))
    distances = ((gallery - query) ** 2).mean(1)
    ids = np.argpartition(distances, topk)[:topk]
    ids = ids[np.argsort(distances[ids])]
    candidates = []
    for candidate_id in ids:
        source = np.asarray(Image.open(files[candidate_id]).convert("RGB"), np.uint8)
        layout, cost, assignment_margin = hungarian_layout(restored, source)
        candidates.append((cost, int(candidate_id), layout, assignment_margin))
    candidates.sort(key=lambda item: item[0])
    cost, source_id, layout, assignment_margin = candidates[0]
    retrieval_margin = float((candidates[1][0] - cost) / (abs(cost) + 1e-8))
    return layout, {
        "source": files[source_id].name,
        "assignment_cost": float(cost),
        "assignment_margin": float(assignment_margin),
        "retrieval_margin": retrieval_margin,
        "retrieval_distance": float(distances[source_id]),
    }


def validate_archive(path: Path, expected: int) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        bad = archive.testzip()
        root_png = all("/" not in name and name.lower().endswith(".png") for name in names)
        unique = len(set(names)) == len(names)
        shapes_ok = True
        for name in names:
            with archive.open(name) as handle:
                image = Image.open(handle)
                if image.size != (480, 480) or image.mode != "RGB":
                    shapes_ok = False
                    break
    return {
        "count": len(names), "expected": expected, "unique": unique,
        "root_png": root_png, "testzip": bad, "shapes_ok": shapes_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--gallery-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.10403286346600248)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--pixel-mode", choices=("raw", "guarded"), default="raw")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "images"
    image_dir.mkdir(exist_ok=True)
    progress_path = args.output / "progress.jsonl"
    files, gallery_raw = build_gallery(args.targets, args.gallery_cache)
    device = torch.device("cuda")
    loaded = pipeline.load_models(device)
    restorer, ranker, position = loaded[:3]
    test_files = sorted(args.test.glob("*.png"))

    for index, path in enumerate(test_files):
        output = image_dir / path.name
        if output.exists():
            continue
        raw_tiles = pipeline.split(path)
        restored = pipeline.restore(restorer, raw_tiles, device)
        source_layout, info = source_candidate(restored, files, gallery_raw, args.topk)
        if info["retrieval_margin"] >= args.threshold:
            layout = source_layout
            route = "source"
        else:
            right = pipeline.ranker_matrix(ranker, restored, 0, device)
            down = pipeline.ranker_matrix(ranker, restored, 1, device)
            pos = pipeline.position_matrix(position, restored, device)
            layout = solve_layout(right, down, pos, 20260819 + index * 100)
            route = "baseline"
        output_tiles = raw_tiles
        reverted_tiles = 576
        if args.pixel_mode == "guarded":
            raw_f = raw_tiles.astype(np.float32)
            restored_f = restored.astype(np.float32)
            raw_std = raw_f.std((1, 2, 3)); restored_std = restored_f.std((1, 2, 3))
            raw_mean = raw_f.mean((1, 2, 3)); restored_mean = restored_f.mean((1, 2, 3))
            restored_rgb = restored_f.mean((1, 2))
            restored_sat = restored_rgb.max(1) - restored_rgb.min(1)
            bad = (
                (restored_std < np.maximum(10.0, 0.72 * raw_std))
                | (np.abs(restored_mean - raw_mean) > 24.0)
                | ((restored_sat < 10.0) & (restored_std < 25.0) & (raw_std >= 10.0))
            )
            output_tiles = restored.copy()
            output_tiles[bad] = raw_tiles[bad]
            reverted_tiles = int(bad.sum())
        Image.fromarray(assemble(output_tiles, np.asarray(layout, np.int32))).save(output, optimize=False)
        record = {"index": index, "file": path.name, "route": route,
                  "pixel_mode": args.pixel_mode, "reverted_tiles": reverted_tiles, **info}
        with progress_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps({"done": index + 1, "total": len(test_files), **record}), flush=True)

    archive_path = args.output / "submission_source_aware_hybrid.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in test_files:
            archive.write(image_dir / path.name, arcname=path.name)
    validation = validate_archive(archive_path, len(test_files))
    records = [json.loads(line) for line in progress_path.read_text().splitlines() if line.strip()]
    report = {
        "threshold": args.threshold,
        "topk": args.topk,
        "pixel_mode": args.pixel_mode,
        "source_count": sum(item["route"] == "source" for item in records),
        "baseline_count": sum(item["route"] == "baseline" for item in records),
        "archive": str(archive_path),
        "validation": validation,
    }
    (args.output / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not (
        validation["count"] == validation["expected"]
        and validation["unique"] and validation["root_png"]
        and validation["testzip"] is None and validation["shapes_ok"]
    ):
        raise RuntimeError("submission archive validation failed")


if __name__ == "__main__":
    main()
