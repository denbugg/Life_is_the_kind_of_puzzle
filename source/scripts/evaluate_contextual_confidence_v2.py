#!/usr/bin/env python3
"""Two-phase development evaluation for solver-gated contextual refinement.

``freeze-development`` builds frozen QAP layouts, analytic inputs, the fixed v1
neural output, target-blind confidence maps, and hashes for every precommitted
candidate and rolled-confidence placebo.  It computes no target metric and does
not store clean pixels.

``score-development`` verifies all hashes and regenerates every frozen render
before reading targets.  It scores only the allocated development sources and
selects at most one candidate under the precommitted rule.  It never reads the
reserved assembly-hypothesis one-shot gate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for root in (SRC_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from evaluate_postassembly_actual_layout import _predict_qap_w4  # noqa: E402
from puzzle_assembly.compatibility import build_classical_score_bank, fuse_ranked_scores  # noqa: E402
from puzzle_assembly.contextual_confidence import (  # noqa: E402
    CONFIDENCE_MAP_NAMES,
    apply_confidence_to_fixed_candidate,
    solver_layout_confidence,
)
from puzzle_assembly.contextual_refiner import (  # noqa: E402
    ContextualResidualNAF,
    build_context_features,
)
from puzzle_assembly.geometry import validate_permutation  # noqa: E402
from puzzle_assembly.learned import load_embedding_checkpoint  # noqa: E402
from puzzle_assembly.panels import make_exact_panel  # noqa: E402
from puzzle_assembly.postassembly_harmonizer import (  # noqa: E402
    SeamGraphConfig,
    apply_rgb_offsets,
    blend_tiles_uint8,
    image_quality_metrics,
    paired_bootstrap_ci,
    seam_graph_rgb_offsets,
)
from puzzle_assembly.protocol import per_source_seed  # noqa: E402
from puzzle_denoise_v2.inference import load_restorer  # noqa: E402
from puzzle_denoise_v2.tiles import merge_tiles_numpy, split_tiles_numpy  # noqa: E402


PANELS = ("primary_kornia", "independent_libjpeg")
GRID = 24
TILE = 20


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze-development", "score-development"))
    parser.add_argument(
        "--config", default="configs/postassembly_contextual_refiner_v2_development.json"
    )
    parser.add_argument("--phase-a-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--classical-chunk-size", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_hash(path: Path, expected: str, role: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{role} hash mismatch: expected {expected}, got {actual}: {path}"
        )


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise RuntimeError(f"invalid RGB image {path}: {values.shape}")
    return values


def _candidate_id(map_name: str, threshold: float, strength: float) -> str:
    return f"{map_name}__t{round(100 * threshold):03d}__s{round(100 * strength):03d}"


def _candidate_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for map_name in config["target_blind_confidence"]["map_names"]:
        for threshold in config["candidate_set"]["thresholds"]:
            for strength in config["candidate_set"]["strengths"]:
                result.append(
                    {
                        "id": _candidate_id(map_name, threshold, strength),
                        "map_name": map_name,
                        "threshold": float(threshold),
                        "strength": float(strength),
                    }
                )
    if len(result) != config["candidate_set"]["candidate_count"]:
        raise RuntimeError("candidate count drift")
    return result


def _protocol(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    config = _load_json(config_path)
    if config.get("kind") != "postassembly_contextual_refiner_v2_confidence_development_protocol":
        raise RuntimeError("unexpected v2 protocol kind")
    if config.get("status") != "precommitted_before_development_phase_a":
        raise RuntimeError("v2 protocol was not precommitted")
    allocation_record = config["allocation"]
    allocation_path = REPO_ROOT / allocation_record["path"]
    _require_hash(allocation_path, allocation_record["sha256"], "v2 allocation")
    allocation = _load_json(allocation_path)
    if allocation.get("status") != "frozen_before_development_phase_a":
        raise RuntimeError("allocation is not frozen")
    names = [str(value) for value in allocation["development"]["source_names"]]
    if len(names) != 32 or _json_sha256(names) != allocation["development"]["source_names_sha256"]:
        raise RuntimeError("development source allocation drift")
    for role, record in config["fixed_assets"].items():
        _require_hash(REPO_ROOT / record["path"], record["sha256"], role)
    runner = config["runner"]
    _require_hash(REPO_ROOT / runner["path"], runner["sha256"], "v2 runner")
    confidence = config["target_blind_confidence"]
    _require_hash(
        REPO_ROOT / confidence["code_path"], confidence["code_sha256"], "confidence code"
    )
    predictor = config["frozen_layout_predictor"]
    _require_hash(
        REPO_ROOT / predictor["implementation_path"],
        predictor["implementation_sha256"],
        "frozen QAP implementation",
    )
    if tuple(confidence["map_names"]) != CONFIDENCE_MAP_NAMES:
        raise RuntimeError("confidence map order drift")
    if tuple(config["synthetic_panels"]["names"]) != PANELS:
        raise RuntimeError("panel order drift")
    _candidate_specs(config)
    return config, allocation, names


def _load_contextual_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> ContextualResidualNAF:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_record = checkpoint["model"]
    model = ContextualResidualNAF(
        width=int(model_record["width"]),
        blocks=int(model_record["blocks"]),
        downsample=int(model_record["downsample"]),
        base_limit_rgb=float(model_record["base_residual_limit_rgb_255"]) / 255.0,
        seam_limit_rgb=float(model_record["seam_residual_limit_rgb_255"]) / 255.0,
    )
    model.load_state_dict(checkpoint["ema_state"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _tensor_image(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1)))
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
        .unsqueeze(0)
    )


@torch.no_grad()
def _fixed_candidate(
    model: ContextualResidualNAF,
    *,
    preanalytic: np.ndarray,
    analytic: np.ndarray,
    seam_confidence: float,
    device: torch.device,
) -> np.ndarray:
    base = _tensor_image(analytic, device)
    before = _tensor_image(preanalytic, device)
    seam_grid = torch.full(
        (1, 1, GRID, GRID), float(seam_confidence), device=device
    )
    layout_ones = torch.ones_like(seam_grid)
    features, gate, seam = build_context_features(
        base, before, seam_grid, layout_ones
    )
    prediction = model(base, features, gate, seam)[0]
    return (
        prediction.detach()
        .float()
        .cpu()
        .mul(255.0)
        .round()
        .clamp(0, 255)
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )


def _render_candidates(
    analytic: np.ndarray,
    fixed_candidate: np.ndarray,
    maps: dict[str, np.ndarray],
    specs: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    candidates = {}
    placebos = {}
    for spec in specs:
        confidence = maps[spec["map_name"]]
        candidates[spec["id"]] = apply_confidence_to_fixed_candidate(
            analytic,
            fixed_candidate,
            confidence,
            threshold=spec["threshold"],
            strength=spec["strength"],
        )
        placebos[spec["id"]] = apply_confidence_to_fixed_candidate(
            analytic,
            fixed_candidate,
            np.roll(confidence, shift=(5, 7), axis=(0, 1)),
            threshold=spec["threshold"],
            strength=spec["strength"],
        )
    return candidates, placebos


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _freeze_development(args: argparse.Namespace, config_path: Path) -> None:
    config, _, names = _protocol(config_path)
    phase_root = (REPO_ROOT / args.phase_a_dir).resolve()
    if phase_root.exists():
        raise FileExistsError(f"refusing to overwrite Phase A: {phase_root}")
    (phase_root / "artifacts").mkdir(parents=True)
    _write_json(
        phase_root / "PHASE_A_STARTED.json",
        {
            "kind": "contextual_confidence_v2_development_phase_a_started",
            "created_utc": _utc_now(),
            "config_sha256": _sha256(config_path),
            "source_count": len(names),
            "target_metrics_computed": False,
            "gate_pixels_opened": False,
        },
    )
    torch.set_num_threads(args.torch_threads)
    assets = config["fixed_assets"]
    selected_model, device, selected_metadata = load_restorer(
        REPO_ROOT / assets["selected_tilenaf"]["path"],
        device=args.device,
        state=assets["selected_tilenaf"]["state"],
    )
    seam_model, seam_device, seam_metadata = load_restorer(
        REPO_ROOT / assets["production_seam_tilenaf"]["path"],
        device=str(device),
        state=assets["production_seam_tilenaf"]["state"],
    )
    if seam_device != device:
        raise RuntimeError("restorer device mismatch")
    embedding, embedding_metadata = load_embedding_checkpoint(
        REPO_ROOT / assets["hbt"]["path"], device=device
    )
    contextual = _load_contextual_model(
        REPO_ROOT / assets["contextual_checkpoint"]["path"], device=device
    )
    specs = _candidate_specs(config)
    records = []
    started = time.perf_counter()
    for source_index, source in enumerate(names):
        clean = _read_rgb(REPO_ROOT / "puzzle/train/targets" / source)
        for panel_index, panel in enumerate(PANELS):
            record_started = time.perf_counter()
            seed = per_source_seed(
                int(config["synthetic_panels"]["master_seed"]),
                f"contextual-refiner-v2-{panel}",
                source,
                0,
            )
            exact = make_exact_panel(clean, panel=panel, seed=seed)
            arrays, layout_diagnostics = _predict_qap_w4(
                exact.slot_tiles,
                source=source,
                selected_model=selected_model,
                seam_model=seam_model,
                embedding_model=embedding,
                device=device,
                config=config,
                denoise_batch_size=args.denoise_batch_size,
                classical_chunk_size=args.classical_chunk_size,
            )
            layout = validate_permutation(
                arrays["position_to_slot"], name="position_to_slot"
            )
            # Rebuild the exact frozen score used by QAP.  This is input-only
            # and supplies confidence; it does not change the frozen layout.
            bank = build_classical_score_bank(
                arrays["selected_slot_tiles"],
                prefix="denoised",
                chunk_size=args.classical_chunk_size,
            )
            c1_names = [
                key
                for key in sorted(bank)
                if key.startswith("denoised_") and not key.endswith("_c2")
            ]
            c1 = fuse_ranked_scores(bank, names=c1_names, name="denoised_C1")
            from puzzle_assembly.learned import learned_compatibility

            hbt, _ = learned_compatibility(
                embedding,
                arrays["selected_slot_tiles"],
                device=device,
                name="denoised_hbt_l1",
            )
            w4 = fuse_ranked_scores(
                {c1.name: c1, hbt.name: hbt},
                names=[c1.name, hbt.name],
                weights={hbt.name: 4.0},
                name="denoised_C1_HBTw4_rank_fusion",
            )
            confidence = solver_layout_confidence(
                w4,
                layout,
                top_k=int(config["target_blind_confidence"]["top_k"]),
                scale_quantile=float(
                    config["target_blind_confidence"]["scale_quantile"]
                ),
            )
            ordered_selected = arrays["selected_slot_tiles"][layout]
            ordered_seam = arrays["seam_slot_tiles"][layout]
            preanalytic_tiles = blend_tiles_uint8(
                ordered_selected, ordered_seam, auxiliary_weight=0.5
            )
            offsets, seam_diagnostics = seam_graph_rgb_offsets(
                preanalytic_tiles, SeamGraphConfig()
            )
            analytic_tiles = apply_rgb_offsets(preanalytic_tiles, offsets)
            preanalytic = merge_tiles_numpy(preanalytic_tiles)
            analytic = merge_tiles_numpy(analytic_tiles)
            fixed = _fixed_candidate(
                contextual,
                preanalytic=preanalytic,
                analytic=analytic,
                seam_confidence=float(seam_diagnostics["confidence_mean"]),
                device=device,
            )
            candidates, placebos = _render_candidates(
                analytic, fixed, confidence.maps, specs
            )
            key = f"{source.removesuffix('.png')}__{panel}"
            artifact = phase_root / "artifacts" / f"{key}.npz"
            artifact_arrays = {
                "raw_slot_tiles": arrays["raw_slot_tiles"],
                "selected_slot_tiles": arrays["selected_slot_tiles"],
                "seam_slot_tiles": arrays["seam_slot_tiles"],
                "position_to_slot": layout.astype(np.int32),
                "preanalytic": preanalytic,
                "analytic_identity": analytic,
                "fixed_candidate": fixed,
                "w4_right": w4.right.astype(np.float32),
                "w4_down": w4.down.astype(np.float32),
            }
            for map_name, values in confidence.maps.items():
                artifact_arrays[f"confidence__{map_name}"] = values
            _atomic_npz(artifact, artifact_arrays)
            records.append(
                {
                    "source": source,
                    "source_index": source_index,
                    "panel": panel,
                    "panel_index": panel_index,
                    "panel_seed": seed,
                    "artifact": str(artifact.relative_to(phase_root)),
                    "artifact_sha256": _sha256(artifact),
                    "array_sha256": {
                        name: _array_sha256(value)
                        for name, value in artifact_arrays.items()
                    },
                    "layout_sha256": _array_sha256(layout.astype(np.int32)),
                    "layout_diagnostics": layout_diagnostics,
                    "seam_diagnostics": seam_diagnostics,
                    "confidence_diagnostics": confidence.diagnostics,
                    "candidate_render_sha256": {
                        spec["id"]: _array_sha256(candidates[spec["id"]])
                        for spec in specs
                    },
                    "placebo_render_sha256": {
                        spec["id"]: _array_sha256(placebos[spec["id"]])
                        for spec in specs
                    },
                    "seconds": float(time.perf_counter() - record_started),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "contextual_confidence_v2_phase_a",
                        "completed": len(records),
                        "total": len(names) * len(PANELS),
                        "source": source,
                        "panel": panel,
                        "seconds": records[-1]["seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    manifest = {
        "schema_version": 1,
        "kind": "contextual_confidence_v2_development_phase_a_manifest",
        "created_utc": _utc_now(),
        "config": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": _sha256(config_path),
        "allocation_sha256": config["allocation"]["sha256"],
        "script_sha256": _sha256(Path(__file__).resolve()),
        "confidence_code_sha256": config["target_blind_confidence"]["code_sha256"],
        "source_count": len(names),
        "record_count": len(records),
        "source_names": names,
        "source_names_sha256": _json_sha256(names),
        "candidate_specs": specs,
        "candidate_specs_sha256": _json_sha256(specs),
        "target_metrics_computed": False,
        "target_pixels_stored": False,
        "gate_pixels_opened": False,
        "all_layouts_valid": True,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "seconds": float(time.perf_counter() - started),
        },
        "model_metadata": {
            "selected_tilenaf": selected_metadata,
            "production_seam_tilenaf": seam_metadata,
            "hbt": embedding_metadata,
            "contextual_checkpoint_sha256": assets["contextual_checkpoint"]["sha256"],
        },
        "records": records,
    }
    _write_json(phase_root / "manifest.json", manifest)
    _write_json(
        phase_root / "PHASE_A_COMPLETE.json",
        {
            "kind": "contextual_confidence_v2_development_phase_a_complete",
            "created_utc": _utc_now(),
            "manifest_sha256": _sha256(phase_root / "manifest.json"),
            "record_count": len(records),
            "target_metrics_computed": False,
            "gate_pixels_opened": False,
        },
    )


def _texture_gradient_mae(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = prediction.astype(np.float32) / 255.0
    true = target.astype(np.float32) / 255.0
    pred_x, true_x = pred[:, 1:] - pred[:, :-1], true[:, 1:] - true[:, :-1]
    pred_y, true_y = pred[1:] - pred[:-1], true[1:] - true[:-1]
    magnitude_x = np.sqrt(np.mean(true_x * true_x, axis=2))
    magnitude_y = np.sqrt(np.mean(true_y * true_y, axis=2))
    mask_x = magnitude_x >= float(np.quantile(magnitude_x, 0.75))
    mask_y = magnitude_y >= float(np.quantile(magnitude_y, 0.75))
    for boundary in range(TILE, 480, TILE):
        mask_x[:, boundary - 1] = False
        mask_y[boundary - 1, :] = False
    error_x = np.abs(pred_x - true_x).mean(axis=2)[mask_x]
    error_y = np.abs(pred_y - true_y).mean(axis=2)[mask_y]
    return float((error_x.sum() + error_y.sum()) / max(1, len(error_x) + len(error_y)))


_CV2_DATA = getattr(cv2, "data", None)
_CASCADE_CANDIDATES = [
    Path(getattr(_CV2_DATA, "haarcascades", ""))
    / "haarcascade_frontalface_default.xml",
    Path(sys.prefix)
    / "share/opencv5/haarcascades/haarcascade_frontalface_default.xml",
    Path(sys.prefix)
    / "share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    Path("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"),
]
_CASCADE_PATH = next((path for path in _CASCADE_CANDIDATES if path.is_file()), None)
_CASCADE = (
    cv2.CascadeClassifier(str(_CASCADE_PATH)) if _CASCADE_PATH is not None else None
)


def _face_boxes(target: np.ndarray) -> np.ndarray:
    if _CASCADE is None:
        return np.empty((0, 4), dtype=np.int32)
    return _CASCADE.detectMultiScale(
        cv2.cvtColor(target, cv2.COLOR_RGB2GRAY),
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(24, 24),
    )


def _quality(prediction: np.ndarray, target: np.ndarray, boxes: np.ndarray) -> dict[str, float]:
    result = image_quality_metrics(
        split_tiles_numpy(prediction), split_tiles_numpy(target)
    )
    result["texture_gradient_mae"] = _texture_gradient_mae(prediction, target)
    face_values = []
    for x, y, width, height in boxes:
        true_crop = target[y : y + height, x : x + width]
        pred_crop = prediction[y : y + height, x : x + width]
        if min(true_crop.shape[:2]) >= 7:
            face_values.append(
                float(
                    structural_similarity(
                        true_crop, pred_crop, channel_axis=2, data_range=255
                    )
                )
            )
    result["face_roi_count"] = float(len(face_values))
    result["face_roi_ssim_sum"] = float(sum(face_values))
    return result


def _summarize_candidate(
    records: list[dict[str, Any]], candidate_id: str, *, bootstrap_seed: int
) -> dict[str, Any]:
    panel_result = {}
    for panel_index, panel in enumerate(PANELS):
        subset = [record for record in records if record["panel"] == panel]
        base = [record["metrics"]["analytic_identity"] for record in subset]
        candidate = [record["metrics"][candidate_id] for record in subset]
        placebo = [record["metrics"][f"placebo__{candidate_id}"] for record in subset]
        deltas = np.asarray([c["ssim"] - b["ssim"] for c, b in zip(candidate, base, strict=True)])
        low, high = paired_bootstrap_ci(
            deltas, seed=bootstrap_seed + panel_index, resamples=20000
        )
        face_count = sum(item["face_roi_count"] for item in base)
        panel_result[panel] = {
            "source_count": len(subset),
            "mean_ssim_delta": float(deltas.mean()),
            "paired_bootstrap_95_ci": [low, high],
            "wins_ties_losses": [
                int(np.sum(deltas > 1e-12)),
                int(np.sum(np.abs(deltas) <= 1e-12)),
                int(np.sum(deltas < -1e-12)),
            ],
            "regressions_below_minus_0_005": int(np.sum(deltas < -0.005)),
            "mean_boundary_band_mae_delta": float(
                np.mean([c["boundary_band_mae"] for c in candidate])
                - np.mean([b["boundary_band_mae"] for b in base])
            ),
            "mean_target_referenced_seam_error_delta": float(
                np.mean([c["target_referenced_seam_error"] for c in candidate])
                - np.mean([b["target_referenced_seam_error"] for b in base])
            ),
            "texture_gradient_mae_ratio": float(
                np.mean([c["texture_gradient_mae"] for c in candidate])
                / max(np.mean([b["texture_gradient_mae"] for b in base]), 1e-12)
            ),
            "face_roi_count": int(face_count),
            "face_roi_mean_ssim_delta": (
                float(
                    (
                        sum(c["face_roi_ssim_sum"] for c in candidate)
                        - sum(b["face_roi_ssim_sum"] for b in base)
                    )
                    / face_count
                )
                if face_count
                else None
            ),
            "mean_ssim_advantage_over_rolled_confidence_placebo": float(
                np.mean([c["ssim"] for c in candidate])
                - np.mean([p["ssim"] for p in placebo])
            ),
        }
    return panel_result


def _candidate_eligible(summary: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    checks = {}
    for panel in PANELS:
        value = summary[panel]
        panel_checks = {
            "positive_mean": value["mean_ssim_delta"] > 0.0,
            "positive_ci_lower": value["paired_bootstrap_95_ci"][0] > 0.0,
            "boundary_nonregression": value["mean_boundary_band_mae_delta"] <= 0.0,
            "seam_nonregression": value["mean_target_referenced_seam_error_delta"] <= 0.0,
            "texture_ratio": value["texture_gradient_mae_ratio"] <= 1.005,
            "no_large_regression": value["regressions_below_minus_0_005"] == 0,
            "placebo_advantage": value[
                "mean_ssim_advantage_over_rolled_confidence_placebo"
            ]
            >= 0.00025,
        }
        if value["face_roi_count"] >= 8:
            panel_checks["face_nonregression"] = value["face_roi_mean_ssim_delta"] >= -0.0005
        checks[panel] = panel_checks
    return bool(all(all(v.values()) for v in checks.values())), checks


def _contained_artifact(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    artifact_root = (root / "artifacts").resolve()
    if path.parent != artifact_root or path.suffix != ".npz":
        raise RuntimeError(f"artifact path escaped Phase A: {relative}")
    return path


def _score_development(args: argparse.Namespace, config_path: Path) -> None:
    if not args.output_dir:
        raise ValueError("--output-dir is required for score-development")
    config, _, names = _protocol(config_path)
    phase_root = (REPO_ROOT / args.phase_a_dir).resolve()
    output_root = (REPO_ROOT / args.output_dir).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite score output: {output_root}")
    output_root.mkdir(parents=True)
    manifest_path = phase_root / "manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("kind") != "contextual_confidence_v2_development_phase_a_manifest":
        raise RuntimeError("unexpected Phase A manifest")
    if manifest["config_sha256"] != _sha256(config_path):
        raise RuntimeError("Phase A config hash drift")
    if manifest["script_sha256"] != _sha256(Path(__file__).resolve()):
        raise RuntimeError("Phase A/scorer script drift")
    if manifest["source_names"] != names or manifest["target_metrics_computed"] is not False:
        raise RuntimeError("Phase A source or target-metric contract drift")
    specs = _candidate_specs(config)
    if manifest["candidate_specs"] != specs:
        raise RuntimeError("Phase A candidate set drift")
    scored_records = []
    for index, record in enumerate(manifest["records"], start=1):
        artifact = _contained_artifact(phase_root, record["artifact"])
        _require_hash(artifact, record["artifact_sha256"], "Phase A artifact")
        with np.load(artifact, allow_pickle=False) as payload:
            arrays = {name: np.asarray(payload[name]) for name in payload.files}
        for name, expected in record["array_sha256"].items():
            if _array_sha256(arrays[name]) != expected:
                raise RuntimeError(f"array hash mismatch: {artifact}: {name}")
        maps = {
            name: arrays[f"confidence__{name}"]
            for name in config["target_blind_confidence"]["map_names"]
        }
        candidates, placebos = _render_candidates(
            arrays["analytic_identity"], arrays["fixed_candidate"], maps, specs
        )
        for spec in specs:
            if _array_sha256(candidates[spec["id"]]) != record["candidate_render_sha256"][spec["id"]]:
                raise RuntimeError(f"candidate render hash mismatch: {record['source']}: {spec['id']}")
            if _array_sha256(placebos[spec["id"]]) != record["placebo_render_sha256"][spec["id"]]:
                raise RuntimeError(f"placebo render hash mismatch: {record['source']}: {spec['id']}")
        target = _read_rgb(REPO_ROOT / "puzzle/train/targets" / record["source"])
        boxes = _face_boxes(target)
        metrics = {
            "analytic_identity": _quality(arrays["analytic_identity"], target, boxes),
            "fixed_ungated_v1_candidate": _quality(arrays["fixed_candidate"], target, boxes),
        }
        for spec in specs:
            metrics[spec["id"]] = _quality(candidates[spec["id"]], target, boxes)
            metrics[f"placebo__{spec['id']}"] = _quality(placebos[spec["id"]], target, boxes)
        scored_records.append(
            {
                "source": record["source"],
                "panel": record["panel"],
                "layout_sha256": record["layout_sha256"],
                "metrics": metrics,
            }
        )
        print(
            json.dumps(
                {"event": "contextual_confidence_v2_scored", "completed": index, "total": len(manifest["records"])},
                sort_keys=True,
            ),
            flush=True,
        )
    summaries = {}
    checks = {}
    eligible = []
    for candidate_index, spec in enumerate(specs):
        summary = _summarize_candidate(
            scored_records,
            spec["id"],
            bootstrap_seed=int(config["development_selection_rule"]["paired_bootstrap_seed"])
            + 10 * candidate_index,
        )
        summaries[spec["id"]] = summary
        is_eligible, candidate_checks = _candidate_eligible(summary)
        checks[spec["id"]] = candidate_checks
        if is_eligible:
            eligible.append(spec)
    map_order = {
        name: index
        for index, name in enumerate(config["target_blind_confidence"]["map_names"])
    }
    eligible.sort(
        key=lambda spec: (
            -min(summaries[spec["id"]][panel]["mean_ssim_delta"] for panel in PANELS),
            spec["strength"],
            -spec["threshold"],
            map_order[spec["map_name"]],
        )
    )
    selected = eligible[0] if eligible else None
    report = {
        "schema_version": 1,
        "kind": "contextual_confidence_v2_development_score_report",
        "created_utc": _utc_now(),
        "status": "eligible_candidate_selected_freeze_before_gate" if selected else "no_eligible_candidate_stop_gate_sealed",
        "config_sha256": _sha256(config_path),
        "phase_a_manifest_sha256": _sha256(manifest_path),
        "source_count": len(names),
        "record_count": len(scored_records),
        "whole_source_panels": list(PANELS),
        "gate_pixels_opened": False,
        "submission_promotion_allowed": False,
        "candidate_specs": specs,
        "candidate_summaries": summaries,
        "candidate_checks": checks,
        "eligible_candidate_ids": [spec["id"] for spec in eligible],
        "selected_candidate": selected,
        "records": scored_records,
    }
    _write_json(output_root / "report.json", report)
    _write_json(
        output_root / "RESULT.json",
        {
            "status": report["status"],
            "report": str((output_root / "report.json").relative_to(REPO_ROOT)),
            "report_sha256": _sha256(output_root / "report.json"),
            "phase_a_manifest_sha256": report["phase_a_manifest_sha256"],
            "selected_candidate": selected,
            "gate_pixels_opened": False,
        },
    )


def main() -> None:
    args = _parse_args()
    config_path = (REPO_ROOT / args.config).resolve()
    if args.action == "freeze-development":
        _freeze_development(args, config_path)
    else:
        _score_development(args, config_path)


if __name__ == "__main__":
    main()
