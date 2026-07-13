#!/usr/bin/env python3
"""Development-only audit of conservative TileNAF/block5x5 output hybrids.

This script deliberately uses only the frozen protocol's development sources.
It never constructs or opens frozen-gate target paths.  The aim is to determine
whether the 5x5 fine-tune's ordered-image gain can be retained while keeping the
current TileNAF prediction at tile borders.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity

from puzzle_denoise_v2.block5x5 import canonical_name_hash, load_protocol, sha256_file
from puzzle_denoise_v2.degradation import SyntheticTileDegrader
from puzzle_denoise_v2.metrics import tile_metrics
from puzzle_denoise_v2.model import TileNAFNet
from puzzle_denoise_v2.tiles import merge_tiles_numpy
from puzzle_denoise_v2.training import (
    choose_device,
    load_manifest,
    make_fixed_validation_plan,
    render_fixed_validation,
    resolved_device_fingerprint,
    runtime_versions,
)


CURRENT_SHA256 = "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_model(path: Path, device: torch.device, *, candidate: bool) -> tuple[TileNAFNet, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if candidate:
        if checkpoint.get("schema_version") != 1 or checkpoint.get("kind") != "tile_naf_block5x5_finetune":
            raise ValueError(f"candidate checkpoint schema mismatch: {path}")
    elif checkpoint.get("model_name") != "tile-naf":
        raise ValueError("current checkpoint is not TileNAF")
    state = checkpoint.get("ema_state")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no EMA state: {path}")
    model = TileNAFNet()
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def predict(model: torch.nn.Module, tiles: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, len(tiles), batch_size):
        batch = (
            torch.from_numpy(np.ascontiguousarray(tiles[start : start + batch_size].transpose(0, 3, 1, 2)))
            .float()
            .div_(255.0)
            .to(device)
        )
        restored = model(batch)
        parts.append(
            restored.detach()
            .cpu()
            .mul(255.0)
            .round()
            .clamp(0, 255)
            .byte()
            .permute(0, 2, 3, 1)
            .numpy()
        )
    return np.concatenate(parts)


def spatial_mask(border: int) -> np.ndarray:
    if border < 0 or border >= 10:
        raise ValueError("border must be in [0, 9]")
    mask = np.ones((1, 20, 20, 1), dtype=np.float32)
    if border:
        mask[:, :border] = 0.0
        mask[:, -border:] = 0.0
        mask[:, :, :border] = 0.0
        mask[:, :, -border:] = 0.0
    return mask


def blur_residual(residual: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return residual
    output = np.empty_like(residual, dtype=np.float32)
    for index, tile in enumerate(residual):
        output[index] = cv2.GaussianBlur(
            tile,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
    return output


def source_metrics(prediction: np.ndarray, clean: np.ndarray) -> dict[str, float]:
    result = tile_metrics(prediction, clean)
    result["ordered_image_ssim"] = float(
        structural_similarity(
            merge_tiles_numpy(clean),
            merge_tiles_numpy(prediction),
            channel_axis=2,
            data_range=255,
        )
    )
    return result


def evaluate(prediction: np.ndarray, clean: np.ndarray, names: list[str]) -> tuple[dict[str, float], list[dict]]:
    records: list[dict] = []
    for index, name in enumerate(names):
        start = index * 576
        records.append(
            {
                "source": name,
                **source_metrics(prediction[start : start + 576], clean[start : start + 576]),
            }
        )
    macro = {
        key: float(np.mean([float(record[key]) for record in records]))
        for key in records[0]
        if key != "source"
    }
    return macro, records


def config_grid() -> list[dict]:
    """Small predeclared family; keep this bounded to avoid dev-set search abuse."""
    configs: list[dict] = []
    for variant in ("moderate", "strong"):
        for alpha in (0.125, 0.25, 0.5, 0.75):
            configs.append({"variant": variant, "alpha": alpha, "border": 0, "blur_sigma": 0.0})
        for border in (1, 2, 3, 4):
            for alpha in (0.25, 0.5, 0.75, 1.0):
                configs.append(
                    {"variant": variant, "alpha": alpha, "border": border, "blur_sigma": 0.0}
                )
        for sigma in (1.0, 2.0):
            for border in (0, 3):
                for alpha in (0.25, 0.5, 0.75):
                    configs.append(
                        {"variant": variant, "alpha": alpha, "border": border, "blur_sigma": sigma}
                    )
    for index, config in enumerate(configs):
        config["id"] = f"hybrid_{index:03d}"
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--manifest", default="configs/denoise_splits_seed20260710.json")
    parser.add_argument("--protocol", default="configs/denoise_block5x5_v1.json")
    parser.add_argument("--selection", default="runs/denoise_v2/kaggle_block5x5_readback/output_v2/block5x5_selection.json")
    parser.add_argument("--current-checkpoint", default="runs/denoise_v2/release/selected_tilenaf_synth_50k.pt")
    parser.add_argument("--moderate-checkpoint", default="runs/denoise_v2/kaggle_block5x5_readback/output_v2/block5x5_moderate.pt")
    parser.add_argument("--strong-checkpoint", default="runs/denoise_v2/kaggle_block5x5_readback/output_v2/block5x5_strong.pt")
    parser.add_argument("--output", default="runs/denoise_v2/block5x5_hybrid_development_v1.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()

    started = time.perf_counter()
    torch.set_num_threads(args.torch_threads)
    cv2.setNumThreads(args.torch_threads)
    protocol_path = Path(args.protocol)
    protocol = load_protocol(protocol_path)
    selection_path = Path(args.selection)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("decision") != "stop_no_development_signal":
        raise ValueError("hybrid audit expects the original no-promotion decision")

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    if sha256_file(manifest_path) != protocol["inputs"]["manifest_sha256"]:
        raise ValueError("manifest hash mismatch")
    names = list(protocol["source_partitions"]["development"]["names"])
    if canonical_name_hash(names) != protocol["source_partitions"]["development"]["names_sha256"]:
        raise ValueError("development source hash mismatch")
    if not set(names) <= set(manifest["splits"]["val"]):
        raise ValueError("development sources are outside whole-source validation")
    if set(names) & set(manifest["splits"]["train"]):
        raise ValueError("development sources overlap training")

    current_path = Path(args.current_checkpoint)
    if sha256_file(current_path) != CURRENT_SHA256:
        raise ValueError("current checkpoint SHA256 mismatch")
    candidate_paths = {
        "moderate": Path(args.moderate_checkpoint),
        "strong": Path(args.strong_checkpoint),
    }
    selection_by_variant = {record["variant"]: record for record in selection["candidates"]}
    for variant, path in candidate_paths.items():
        if sha256_file(path) != selection_by_variant[variant]["sha256"]:
            raise ValueError(f"{variant} checkpoint SHA256 mismatch")

    device = choose_device(args.device)
    current_model, _ = load_model(current_path, device, candidate=False)
    candidate_models = {
        variant: load_model(path, device, candidate=True)[0]
        for variant, path in candidate_paths.items()
    }

    target_dir = Path(args.data_root) / "train" / "targets"
    degrader = SyntheticTileDegrader()
    plan = make_fixed_validation_plan(
        target_dir,
        names,
        576,
        int(protocol["panel_seeds"]["development"]),
        degrader,
    )
    corruptions = {
        "primary_kornia": render_fixed_validation(plan, degrader, args.batch_size, codec="kornia"),
        "independent_libjpeg": render_fixed_validation(
            plan, degrader, min(args.batch_size, 512), codec="pillow"
        ),
    }

    grid = config_grid()
    panel_state: dict[str, dict] = {}
    for panel_name, corrupt in corruptions.items():
        current = predict(current_model, corrupt, device, args.batch_size)
        candidates = {
            variant: predict(model, corrupt, device, args.batch_size)
            for variant, model in candidate_models.items()
        }
        baseline_macro, baseline_per_source = evaluate(current, plan.clean, names)
        residual_cache: dict[tuple[str, float], np.ndarray] = {}
        for variant, prediction in candidates.items():
            residual = prediction.astype(np.float32) - current.astype(np.float32)
            for sigma in (0.0, 1.0, 2.0):
                residual_cache[(variant, sigma)] = blur_residual(residual, sigma)
        records: list[dict] = []
        current_float = current.astype(np.float32)
        for config in grid:
            residual = residual_cache[(config["variant"], config["blur_sigma"])]
            prediction = np.clip(
                np.rint(
                    current_float
                    + float(config["alpha"])
                    * spatial_mask(int(config["border"]))
                    * residual
                ),
                0,
                255,
            ).astype(np.uint8)
            macro, per_source = evaluate(prediction, plan.clean, names)
            deltas = {key: macro[key] - baseline_macro[key] for key in macro}
            records.append({"config": config, "macro": macro, "deltas": deltas, "per_source": per_source})
        panel_state[panel_name] = {
            "baseline_macro": baseline_macro,
            "baseline_per_source": baseline_per_source,
            "records": records,
        }
        print(json.dumps({"event": "hybrid_panel_complete", "panel": panel_name}), flush=True)

    combined: list[dict] = []
    for index, config in enumerate(grid):
        panels = {
            panel_name: panel_state[panel_name]["records"][index]
            for panel_name in panel_state
        }
        balanced_macro = {
            key: float(np.mean([record["macro"][key] for record in panels.values()]))
            for key in next(iter(panels.values()))["macro"]
        }
        balanced_baseline = {
            key: float(
                np.mean([panel_state[name]["baseline_macro"][key] for name in panel_state])
            )
            for key in balanced_macro
        }
        deltas = {key: balanced_macro[key] - balanced_baseline[key] for key in balanced_macro}
        checks = {
            "ordered_image_ssim_delta_at_least_0_002": deltas["ordered_image_ssim"] >= 0.002,
            "tile_ssim_delta_nonnegative": deltas["tile_ssim"] >= 0.0,
            "boundary_mae_growth_at_most_0_2pct": balanced_macro["boundary_mae"]
            <= balanced_baseline["boundary_mae"] * 1.002,
            "gradient_mae_growth_at_most_0_2pct": balanced_macro["gradient_mae"]
            <= balanced_baseline["gradient_mae"] * 1.002,
            "both_panel_ordered_image_ssim_positive": all(
                record["deltas"]["ordered_image_ssim"] > 0.0 for record in panels.values()
            ),
        }
        combined.append(
            {
                "config": config,
                "panels": panels,
                "balanced_macro": balanced_macro,
                "balanced_baseline": balanced_baseline,
                "deltas": deltas,
                "checks": checks,
                "eligible": all(checks.values()),
            }
        )

    eligible = [record for record in combined if record["eligible"]]
    eligible.sort(
        key=lambda record: (
            -float(record["deltas"]["ordered_image_ssim"]),
            -float(record["deltas"]["tile_ssim"]),
            str(record["config"]["id"]),
        )
    )
    selected = eligible[0] if eligible else None
    payload = {
        "schema_version": 1,
        "kind": "denoise_block5x5_hybrid_development_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "eligible_hybrid_found" if selected else "stop_no_hybrid_signal",
        "frozen_gate_pixels_accessed": False,
        "frozen_gate_paths_constructed": False,
        "development_source_names": names,
        "development_source_names_sha256": canonical_name_hash(names),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "selection": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "checkpoint_sha256": {
            "current": sha256_file(current_path),
            **{variant: sha256_file(path) for variant, path in candidate_paths.items()},
        },
        "grid_sha256": hashlib.sha256(
            json.dumps(grid, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "grid_count": len(grid),
        "selected": selected,
        "candidates": combined,
        "runtime_versions": runtime_versions(),
        "device": resolved_device_fingerprint(device),
        "seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "event": "hybrid_audit_complete",
                "decision": payload["decision"],
                "selected": None if selected is None else selected["config"],
                "output": str(output),
                "output_sha256": sha256_file(output),
                "seconds": payload["seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
