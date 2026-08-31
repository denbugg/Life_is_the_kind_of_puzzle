#!/usr/bin/env python3
"""Evaluate a frozen restoration model on genuinely inferred bijective layouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image

from aiijc_puzzle.compliant_atlas_decoder import (
    audit_raw_permutation,
    population_position_scores,
    solve_buddies_with_position,
)
from aiijc_puzzle.legacy_upgrade import directional_scores, layout_digest, solve_buddies
from aiijc_puzzle.low_frequency_prior import FrozenLowFrequencyPrior
from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    EXPERIMENT_SUBSET_SEED,
    IMAGE_SIZE,
    assemble_tiles,
    compute_protocol_digest,
    contest_ssim,
    select_manifest_records,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.restoration_r6 import (
    TileAwareDualNAFNet,
    nlm_color,
    restore_image,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_ATLAS = PROJECT_ROOT / "artifacts/low-frequency-prior/train5600-v1.npz"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "outputs/restoration-r6/compliant-r6-medium-train256-step2000-h10.pt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/restoration-r6/compliant-layout-transfer-calibration12.json"
)
LAYOUTS = (
    "bilateral_buddies96",
    "bilateral_buddies96_atlas_w0p03",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--inputs", type=Path, default=Path("data/raw/train/inputs"))
    parser.add_argument("--targets", type=Path, default=Path("data/raw/train/targets"))
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("calibration", "holdout"), default="calibration")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--nlm-h", type=int, default=10)
    parser.add_argument("--max-nlm-passes", type=int, default=10)
    parser.add_argument(
        "--skip-r6",
        action="store_true",
        help="evaluate only the geometry-preserving iterative NLM tail",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--allow-holdout", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def load_rgb_verified(path: Path, expected_hash: str) -> np.ndarray:
    if sha256_file(path) != expected_hash:
        raise ValueError(f"content hash mismatch: {path}")
    with Image.open(path) as image:
        image.load()
        if (
            image.format != "PNG"
            or image.mode != "RGB"
            or image.size
            != (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        ):
            raise ValueError(f"expected strict RGB 480x480 PNG: {path}")
        return np.asarray(image, dtype=np.uint8)


def array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def names_digest(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(str(record["filename"]) for record in records).encode()
    ).hexdigest()


def nlm_tail_name(passes: int) -> str:
    if passes < 1:
        raise ValueError("NLM pass count must be positive")
    if passes == 1:
        return "nlm"
    if passes == 2:
        return "nlm_twice"
    return f"nlm_{passes}x"


def tail_names(max_nlm_passes: int, include_r6: bool) -> tuple[str, ...]:
    names = ("raw",) + tuple(nlm_tail_name(passes) for passes in range(1, max_nlm_passes + 1))
    if include_r6:
        names += ("r6", "r6_then_nlm", "r6_then_nlm_twice")
    return names


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_model(
    path: Path, device: torch.device, nlm_h: int
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    configuration = checkpoint.get("model_configuration")
    expected_keys = {"architecture", "base", "depth", "blocks"}
    if not isinstance(configuration, dict) or set(configuration) != expected_keys:
        raise ValueError("checkpoint model configuration is missing or malformed")
    if configuration["architecture"] != "dual_naf":
        raise ValueError("only the frozen dual_naf checkpoint is supported")
    training = checkpoint.get("training_configuration")
    if not isinstance(training, dict) or training.get("nlm_h") != nlm_h:
        raise ValueError("checkpoint conditioning NLM strength differs from evaluation")
    model = TileAwareDualNAFNet(
        base=int(configuration["base"]),
        depth=int(configuration["depth"]),
        blocks=int(configuration["blocks"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


def infer_layouts(
    image: np.ndarray,
    atlas: FrozenLowFrequencyPrior,
) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, object]]]:
    """Build only exact tile permutations; this function has no target argument."""

    tiles = split_tiles(image)
    right, down = directional_scores(tiles, views=("bilateral",))["bilateral"]
    position = population_position_scores(tiles, atlas.generic_tile_template)
    results = {
        "bilateral_buddies96": solve_buddies(right, down, max_edges=96),
        "bilateral_buddies96_atlas_w0p03": solve_buddies_with_position(
            right, down, position, position_weight=0.03, max_edges=96
        ),
    }
    if tuple(results) != LAYOUTS:
        raise RuntimeError("layout roster drifted")
    frozen = {}
    for name, result in results.items():
        raw = assemble_tiles(tiles[result.layout])
        audit = audit_raw_permutation(
            image, raw, result.layout, restoration_applied_after_audit=True
        )
        if not audit.passed:
            raise RuntimeError(f"permutation audit failed for {name}")
        frozen[name] = (result.layout.copy(), raw, audit.as_dict())
    return frozen


def bootstrap_interval(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(EXPERIMENT_SUBSET_SEED)
    indices = rng.integers(0, len(values), size=(20_000, len(values)))
    return tuple(float(value) for value in np.quantile(values[indices].mean(1), (0.025, 0.975)))


def aggregate(rows: list[dict[str, Any]], tails: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for layout in LAYOUTS:
        baseline = np.asarray([row["variants"][layout]["ssim"]["nlm"] for row in rows])
        for tail in tails:
            scores = np.asarray([row["variants"][layout]["ssim"][tail] for row in rows])
            difference = scores - baseline
            result[f"{layout}__{tail}"] = {
                "mean_ssim": float(scores.mean()),
                "mean_gain_vs_same_layout_nlm": float(difference.mean()),
                "gain_ci95": list(bootstrap_interval(difference)),
                "wins_vs_same_layout_nlm": int(np.sum(difference > 0)),
                "count": len(scores),
            }
    return result


def main() -> None:
    args = parse_args()
    if not args.run:
        raise SystemExit("refusing evaluation without --run")
    if args.split == "holdout" and not args.allow_holdout:
        raise ValueError("holdout requires explicit --allow-holdout after a frozen gate")
    if args.limit < 1 or args.offset < 0 or args.nlm_h < 1 or not 1 <= args.max_nlm_passes <= 100:
        raise ValueError(
            "limit/nlm-h/max-nlm-passes must be positive, offset non-negative, "
            "and max-nlm-passes at most 100"
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("manifest digest mismatch")
    selected = select_manifest_records(
        manifest,
        args.split,
        limit=args.offset + args.limit,
        seed=EXPERIMENT_SUBSET_SEED,
        namespace=EXPERIMENT_SUBSET_NAMESPACE,
    )
    records = [dict(record) for record in selected[args.offset :]]
    atlas = FrozenLowFrequencyPrior.load(args.atlas)
    if atlas.metadata.get("target_contract") != "manifest train split only":
        raise ValueError("atlas target provenance is not train-only")
    device_name = args.device
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    device = torch.device(device_name)
    model: torch.nn.Module | None = None
    checkpoint: dict[str, Any] | None = None
    if not args.skip_r6:
        model, checkpoint = load_model(args.checkpoint, device, args.nlm_h)
    tails = tail_names(args.max_nlm_passes, include_r6=model is not None)

    rows: list[dict[str, Any]] = []
    started = perf_counter()
    for index, record in enumerate(records, start=1):
        name = str(record["filename"])
        dirty = load_rgb_verified(args.inputs / name, str(record["input_sha256"]))
        layouts = infer_layouts(dirty, atlas)

        # Every layout, raw assembly, restoration output and hash is frozen
        # before the paired validation target is decoded.
        frozen: dict[str, dict[str, Any]] = {}
        for layout_name, (layout, raw, audit) in layouts.items():
            predictions = {"raw": raw}
            iterative = raw
            for passes in range(1, args.max_nlm_passes + 1):
                iterative = nlm_color(iterative, args.nlm_h)
                predictions[nlm_tail_name(passes)] = iterative
            if model is not None:
                restored = restore_image(model, raw, device, nlm_h=args.nlm_h)
                restored_nlm = nlm_color(restored, args.nlm_h)
                predictions.update(
                    {
                        "r6": restored,
                        "r6_then_nlm": restored_nlm,
                        "r6_then_nlm_twice": nlm_color(restored_nlm, args.nlm_h),
                    }
                )
            if tuple(predictions) != tails:
                raise RuntimeError("restoration tail roster drifted")
            frozen[layout_name] = {
                "layout": layout,
                "audit": audit,
                "predictions": predictions,
                "hashes": {key: array_digest(value) for key, value in predictions.items()},
            }

        target = load_rgb_verified(args.targets / name, str(record["target_sha256"]))
        variants = {}
        for layout_name, inference in frozen.items():
            variants[layout_name] = {
                "tile_at_position": inference["layout"].tolist(),
                "layout_sha256": layout_digest(inference["layout"]),
                "permutation_audit": inference["audit"],
                "prediction_sha256": inference["hashes"],
                "ssim": {
                    tail: contest_ssim(target, prediction)
                    for tail, prediction in inference["predictions"].items()
                },
            }
        rows.append(
            {
                "filename": name,
                "input_sha256": record["input_sha256"],
                "target_sha256": record["target_sha256"],
                "predictions_frozen_before_target_decode": True,
                "variants": variants,
            }
        )
        best = max(
            (score, f"{layout_name}__{tail}")
            for layout_name, metrics in variants.items()
            for tail, score in metrics["ssim"].items()
        )
        print(json.dumps({"done": index, "total": len(records), "best": best}), flush=True)

    summary = aggregate(rows, tails)
    champion = max(summary, key=lambda name: summary[name]["mean_ssim"])
    report = {
        "schema": "aiijc-compliant-restoration-transfer-v2",
        "status": "completed",
        "split": args.split,
        "count": len(records),
        "offset": args.offset,
        "protocol_digest": manifest["protocol_digest"],
        "selection_namespace": EXPERIMENT_SUBSET_NAMESPACE,
        "selection_seed": EXPERIMENT_SUBSET_SEED,
        "selection_digest": names_digest(records),
        "inference_target_access": False,
        "predictions_frozen_before_target_decode": True,
        "compliance": {
            "all_576_input_tiles_used_exactly_once": True,
            "raw_assembly_pixel_preserving": True,
            "restoration_after_layout_only": True,
            "spatial_warp_or_tile_substitution": False,
            "all_permutation_audits_passed": all(
                variant["permutation_audit"]["passed"]
                for row in rows
                for variant in row["variants"].values()
            ),
        },
        "configuration": {
            "layouts": list(LAYOUTS),
            "tails": list(tails),
            "atlas": str(args.atlas.resolve()),
            "atlas_sha256": sha256_file(args.atlas),
            "checkpoint": None if checkpoint is None else str(args.checkpoint.resolve()),
            "checkpoint_sha256": (None if checkpoint is None else sha256_file(args.checkpoint)),
            "checkpoint_model_configuration": (
                None if checkpoint is None else checkpoint["model_configuration"]
            ),
            "checkpoint_training_configuration": (
                None if checkpoint is None else checkpoint["training_configuration"]
            ),
            "nlm_h": args.nlm_h,
            "max_nlm_passes": args.max_nlm_passes,
            "r6_evaluated": model is not None,
            "device": str(device),
        },
        "champion": champion,
        "champion_summary": summary[champion],
        "summary": summary,
        "runtime_seconds": perf_counter() - started,
        "source_sha256": {
            str(Path(__file__).relative_to(PROJECT_ROOT)): sha256_file(Path(__file__)),
            "src/aiijc_puzzle/restoration_r6.py": sha256_file(
                PROJECT_ROOT / "src/aiijc_puzzle/restoration_r6.py"
            ),
            "src/aiijc_puzzle/compliant_atlas_decoder.py": sha256_file(
                PROJECT_ROOT / "src/aiijc_puzzle/compliant_atlas_decoder.py"
            ),
        },
        "per_board": rows,
    }
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "champion": champion,
                "summary": summary[champion],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
