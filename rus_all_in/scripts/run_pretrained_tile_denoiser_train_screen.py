#!/usr/bin/env python3
"""Train-only development screen for legal tile-wise pretrained DRUNet tails."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image

from aiijc_puzzle.compliant_atlas_decoder import audit_raw_permutation
from aiijc_puzzle.legacy_upgrade import directional_scores, layout_digest, solve_buddies
from aiijc_puzzle.pixel_tails import apply_nlm_color
from aiijc_puzzle.postassembly_harmonizer import (
    LuminanceGainConfig,
    SeamGraphConfig,
    apply_luminance_gains,
    apply_rgb_offsets,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
)
from aiijc_puzzle.pretrained_tile_denoiser import (
    blend_uint8_fraction,
    load_drunet_color,
    render_drunet_tiles,
)
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
INPUTS = PROJECT_ROOT / "data/raw/train/inputs"
TARGETS = PROJECT_ROOT / "data/raw/train/targets"
ASSET_ROOT = PROJECT_ROOT / "artifacts/pretrained-denoisers/kair-fc1732f"
CHECKPOINT = ASSET_ROOT / "drunet_color.pth"
OUTPUT = (
    PROJECT_ROOT
    / "outputs/pretrained-tile-denoiser/train-development-offset512-count16/report.json"
)
EXPECTED_MANIFEST_SHA256 = "4781e370e092ad272c63e6d5165b25951aaf93fae5fde74c75d534a9e8efc9da"
EXPECTED_PROTOCOL_DIGEST = "2a9e3b74f7defa8c00846a05eb598fd263fd16c2787c70e77d3b7a4b585bfbf4"
EXPECTED_ASSET_SHA256 = {
    "LICENSE": "448e69b705d64f21bf8cb86562301e0edd99ac79026064ddd75af8242b067be5",
    "drunet_color.pth": "479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4",
    "models/basicblock.py": "48406db8867394ac5ae233ebeec7711ac10acfc3a6bbf0072c33aa77d659b6fd",
    "models/network_unet.py": "8043b6350f1589d5f08892e3be0b4d12c5a502058014285107b7360696d12bf5",
}
KAIR_COMMIT = "fc1732f4a4514e42ce15e5b3a1e18c828af47a1e"
TRAIN_OFFSET = 512
SELECTION_COUNT = 8
VERIFICATION_COUNT = 8
TRAIN_COUNT = SELECTION_COUNT + VERIFICATION_COUNT
EDGE_BUDGET = 96
DIRECT_SIGMAS = (10, 20, 30, 40)
POST_H28_SIGMAS = (5, 10, 20)
BLEND_ALPHAS = (Fraction(1, 8), Fraction(1, 4))
BASELINES = ("nlm_h20", "nlm_h28")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=144)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_rgb_verified(path: Path, expected_hash: str) -> np.ndarray:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError(f"content hash mismatch: {path}")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or image.size != (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise ValueError(f"expected strict RGB 480x480 PNG: {path}")
        return np.asarray(image, dtype=np.uint8)


def select_device(requested: str) -> torch.device:
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(requested)


def names_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(str(record["filename"]) for record in records).encode("utf-8")
    ).hexdigest()


def load_train_records() -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("validation manifest file changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("validation manifest self-digest is invalid")
    if manifest.get("protocol_digest") != EXPECTED_PROTOCOL_DIGEST:
        raise ValueError("validation protocol changed")
    records = tuple(
        select_manifest_records(
            manifest,
            "train",
            limit=TRAIN_OFFSET + TRAIN_COUNT,
            seed=EXPERIMENT_SUBSET_SEED,
            namespace=EXPERIMENT_SUBSET_NAMESPACE,
        )[TRAIN_OFFSET:]
    )
    if len(records) != TRAIN_COUNT:
        raise RuntimeError("train development roster count drifted")
    return manifest, records


def verify_assets() -> dict[str, str]:
    observed = {
        relative: sha256_file(ASSET_ROOT / relative) for relative in EXPECTED_ASSET_SHA256
    }
    if observed != EXPECTED_ASSET_SHA256:
        raise ValueError(f"official KAIR asset hashes changed: {observed}")
    license_text = (ASSET_ROOT / "LICENSE").read_text(encoding="utf-8")
    if "MIT License" not in license_text or "Copyright (c) 2019 Kai Zhang" not in license_text:
        raise ValueError("KAIR license text changed")
    return observed


def apply_rgb_luma(ordered_tiles: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    offsets, rgb_diagnostics = seam_graph_rgb_offsets(ordered_tiles, SeamGraphConfig())
    rgb_tiles = apply_rgb_offsets(ordered_tiles, offsets)
    gains, luma_diagnostics = seam_graph_luminance_gains(rgb_tiles, LuminanceGainConfig())
    harmonized_tiles = apply_luminance_gains(rgb_tiles, gains)
    return harmonized_tiles, {
        "rgb_seam_offsets": rgb_diagnostics,
        "bounded_luminance_gains": luma_diagnostics,
    }


def alpha_tag(alpha: Fraction) -> str:
    return f"{alpha.numerator}of{alpha.denominator}"


def arm_names() -> tuple[str, ...]:
    names = list(BASELINES)
    for sigma in DIRECT_SIGMAS:
        direct = f"drunet_sigma{sigma}"
        names.extend((direct, f"{direct}_then_nlm20", f"{direct}_then_nlm28"))
        for alpha in BLEND_ALPHAS:
            tag = alpha_tag(alpha)
            names.extend((f"blend_{tag}_h20_{direct}", f"blend_{tag}_h28_{direct}"))
    for sigma in POST_H28_SIGMAS:
        post = f"h28_then_drunet_sigma{sigma}"
        names.append(post)
        for alpha in BLEND_ALPHAS:
            names.append(f"blend_{alpha_tag(alpha)}_h28_{post}")
    return tuple(names)


def render_arms(
    harmonized_tiles: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    *,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    harmonized = assemble_tiles(harmonized_tiles)
    predictions: dict[str, np.ndarray] = {
        "nlm_h20": apply_nlm_color(harmonized, h=20).image,
        "nlm_h28": apply_nlm_color(harmonized, h=28).image,
    }
    diagnostics: dict[str, Any] = {}
    for sigma in DIRECT_SIGMAS:
        direct_name = f"drunet_sigma{sigma}"
        restored_tiles, render_diagnostics = render_drunet_tiles(
            model,
            harmonized_tiles,
            sigma_255=sigma,
            device=device,
            batch_size=batch_size,
        )
        direct = assemble_tiles(restored_tiles)
        predictions[direct_name] = direct
        predictions[f"{direct_name}_then_nlm20"] = apply_nlm_color(direct, h=20).image
        predictions[f"{direct_name}_then_nlm28"] = apply_nlm_color(direct, h=28).image
        diagnostics[direct_name] = render_diagnostics.as_dict()
        for alpha in BLEND_ALPHAS:
            tag = alpha_tag(alpha)
            predictions[f"blend_{tag}_h20_{direct_name}"] = blend_uint8_fraction(
                predictions["nlm_h20"], direct, alpha
            )
            predictions[f"blend_{tag}_h28_{direct_name}"] = blend_uint8_fraction(
                predictions["nlm_h28"], direct, alpha
            )

    h28_tiles = split_tiles(predictions["nlm_h28"])
    for sigma in POST_H28_SIGMAS:
        post_name = f"h28_then_drunet_sigma{sigma}"
        restored_tiles, render_diagnostics = render_drunet_tiles(
            model,
            h28_tiles,
            sigma_255=sigma,
            device=device,
            batch_size=batch_size,
        )
        post = assemble_tiles(restored_tiles)
        predictions[post_name] = post
        diagnostics[post_name] = render_diagnostics.as_dict()
        for alpha in BLEND_ALPHAS:
            predictions[f"blend_{alpha_tag(alpha)}_h28_{post_name}"] = (
                blend_uint8_fraction(predictions["nlm_h28"], post, alpha)
            )
    roster = arm_names()
    if set(predictions) != set(roster):
        raise RuntimeError(f"development arm roster drifted: {set(predictions) ^ set(roster)}")
    return {name: predictions[name] for name in roster}, diagnostics


def summary_for(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    baseline: str,
) -> dict[str, Any]:
    candidate = np.asarray([float(row["ssim"][arm]) for row in rows])
    control = np.asarray([float(row["ssim"][baseline]) for row in rows])
    difference = candidate - control
    return {
        "mean_ssim": float(candidate.mean()),
        "baseline_mean_ssim": float(control.mean()),
        "mean_delta": float(difference.mean()),
        "median_delta": float(np.median(difference)),
        "wins_ties_losses": [
            int(np.sum(difference > 0)),
            int(np.sum(difference == 0)),
            int(np.sum(difference < 0)),
        ],
        "board_deltas": difference.tolist(),
    }


def main() -> None:
    args = parse_args()
    if not args.run:
        raise SystemExit("Refusing to run without --run")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite train-only report: {OUTPUT}")
    assets = verify_assets()
    manifest, records = load_train_records()
    device = select_device(args.device)
    model = load_drunet_color(CHECKPOINT, device)
    roster = arm_names()
    source_paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/aiijc_puzzle/pretrained_tile_denoiser.py",
        PROJECT_ROOT / "src/aiijc_puzzle/legacy_upgrade.py",
        PROJECT_ROOT / "src/aiijc_puzzle/postassembly_harmonizer.py",
        PROJECT_ROOT / "src/aiijc_puzzle/pixel_tails.py",
        PROJECT_ROOT / "src/aiijc_puzzle/protocol.py",
    )
    started = perf_counter()
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        filename = str(record["filename"])
        dirty = load_rgb_verified(INPUTS / filename, str(record["input_sha256"]))
        input_tiles = split_tiles(dirty)
        right, down = directional_scores(input_tiles, views=("bilateral",))["bilateral"]
        solved = solve_buddies(right, down, max_edges=EDGE_BUDGET)
        ordered = np.ascontiguousarray(input_tiles[solved.layout])
        raw = assemble_tiles(ordered)
        audit = audit_raw_permutation(
            dirty, raw, solved.layout, restoration_applied_after_audit=True
        )
        if not audit.passed:
            raise RuntimeError(f"raw permutation audit failed for {filename}")
        harmonized_tiles, harmonizer = apply_rgb_luma(ordered)
        predictions, render_diagnostics = render_arms(
            harmonized_tiles,
            model,
            device,
            batch_size=args.batch_size,
        )
        pixel_hashes = {
            name: hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
            for name, image in predictions.items()
        }
        if len(set(pixel_hashes.values())) != len(pixel_hashes):
            raise RuntimeError(f"development predictions were not all distinct: {filename}")

        # Train target decoding happens only after this board's full fixed arm roster exists.
        target = load_rgb_verified(TARGETS / filename, str(record["target_sha256"]))
        scores = {
            name: contest_ssim(target, prediction)
            for name, prediction in predictions.items()
        }
        rows.append(
            {
                "filename": filename,
                "panel_role": "selection" if index <= SELECTION_COUNT else "verification",
                "input_sha256": record["input_sha256"],
                "target_sha256": record["target_sha256"],
                "layout_sha256": layout_digest(solved.layout),
                "raw_permutation_audit": audit.as_dict(),
                "solver": solved.solver,
                "objective": float(solved.objective),
                "harmonizer": harmonizer,
                "pixel_sha256": pixel_hashes,
                "render_diagnostics": render_diagnostics,
                "ssim": scores,
            }
        )
        print(
            json.dumps(
                {
                    "done": index,
                    "total": len(records),
                    "filename": filename,
                    "h20": scores["nlm_h20"],
                    "h28": scores["nlm_h28"],
                    "best": max(scores, key=scores.get),
                    "best_ssim": max(scores.values()),
                }
            ),
            flush=True,
        )

    selection = rows[:SELECTION_COUNT]
    verification = rows[SELECTION_COUNT:]
    candidates = tuple(name for name in roster if name not in BASELINES)
    selection_means = {
        name: float(np.mean([row["ssim"][name] for row in selection])) for name in roster
    }
    selected = max(candidates, key=lambda name: selection_means[name])
    comparisons = {
        role: {
            baseline: summary_for(panel, selected, baseline) for baseline in BASELINES
        }
        for role, panel in (("selection", selection), ("verification", verification))
    }
    selection_material = all(
        comparisons["selection"][baseline]["mean_delta"] >= 0.001
        and comparisons["selection"][baseline]["wins_ties_losses"][0] >= 6
        for baseline in BASELINES
    )
    verification_positive = all(
        comparisons["verification"][baseline]["mean_delta"] > 0
        and comparisons["verification"][baseline]["wins_ties_losses"][0] >= 5
        for baseline in BASELINES
    )
    report = {
        "schema": "aiijc-pretrained-tile-denoiser-train-development-v1",
        "status": "train_only_development_complete",
        "split": "train",
        "calibration_targets_accessed": False,
        "holdout_targets_accessed": False,
        "competition_test_accessed": False,
        "offset": TRAIN_OFFSET,
        "count": TRAIN_COUNT,
        "selection_count": SELECTION_COUNT,
        "verification_count": VERIFICATION_COUNT,
        "selection_digest": names_digest(records),
        "manifest_sha256": sha256_file(MANIFEST),
        "protocol_digest": manifest["protocol_digest"],
        "device": str(device),
        "batch_size": args.batch_size,
        "architecture": "official KAIR colour DRUNet, discriminative Gaussian denoiser",
        "architecture_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "official_repository": "https://github.com/cszn/KAIR",
        "official_commit": KAIR_COMMIT,
        "checkpoint_url": (
            "https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth"
        ),
        "license": "MIT",
        "assets_sha256": assets,
        "local_source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in source_paths
        },
        "geometry_contract": {
            "strict_layout": "bilateral buddies96",
            "tile_size": 20,
            "tile_identity_and_order_unchanged_after_layout": True,
            "reflection_padding_same_tile_only": True,
            "crop_back_exact_20x20": True,
            "resizing_or_warping": False,
            "cross_tile_network_context": False,
            "external_reference_or_template_pixels": False,
        },
        "arm_names": list(roster),
        "selection_means": selection_means,
        "selected_on_first_8_train_only": selected,
        "selected_comparisons": comparisons,
        "train_signal_gate": {
            "selection_delta_ge_0_001_and_wins_ge_6_of_8_each_baseline": (
                selection_material
            ),
            "verification_delta_positive_and_wins_ge_5_of_8_each_baseline": (
                verification_positive
            ),
            "materially_positive": selection_material and verification_positive,
        },
        "calibration_preregistration_authorized": selection_material and verification_positive,
        "runtime_seconds": perf_counter() - started,
        "rows": rows,
    }
    atomic_json(OUTPUT, report)
    print(
        json.dumps(
            {
                "report": str(OUTPUT),
                "report_sha256": sha256_file(OUTPUT),
                "selected": selected,
                "comparisons": comparisons,
                "train_signal_gate": report["train_signal_gate"],
                "calibration_preregistration_authorized": report[
                    "calibration_preregistration_authorized"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
