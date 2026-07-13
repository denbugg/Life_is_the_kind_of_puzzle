from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image


STARTED = time.time()


def _nvidia_compute_capabilities() -> list[tuple[int, int]]:
    try:
        probe = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
    except FileNotFoundError:
        return []
    result = []
    for line in probe.stdout.splitlines():
        parts = line.strip().split(".")
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            result.append((int(parts[0]), int(parts[1])))
    return result


BOOTSTRAP_CAPABILITIES = _nvidia_compute_capabilities()
BOOTSTRAP_TORCH_INSTALLED = False
if BOOTSTRAP_CAPABILITIES and min(BOOTSTRAP_CAPABILITIES) < (7, 0):
    # Kaggle's current Torch 2.10 image omits sm_60.  The already validated
    # project fallback is the cu124 Torch 2.6 wheel, whose arch list includes
    # sm_60.  Install before the first torch import so no stale module remains.
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--upgrade",
        "torch==2.6.0",
        "--index-url",
        "https://download.pytorch.org/whl/cu124",
    ]
    print(
        json.dumps(
            {
                "event": "p100_torch_bootstrap",
                "capabilities": BOOTSTRAP_CAPABILITIES,
                "command": command,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    subprocess.run(command, check=True)
    BOOTSTRAP_TORCH_INSTALLED = True


JOB_ROOT = Path(__file__).resolve().parent


def _find_bundle_root() -> Path:
    marker = Path("configs/postassembly_contextual_refiner_v1.json")
    candidates = [JOB_ROOT / "bundle" / marker]
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        candidates.extend(input_root.rglob(marker.name))
    valid = [
        path.parents[1]
        for path in candidates
        if path.is_file()
        and (path.parents[1] / "src/puzzle_assembly/contextual_refiner.py").is_file()
    ]
    unique = sorted(set(valid))
    if len(unique) != 1:
        raise RuntimeError(f"expected one contextual-refiner code bundle, got {unique}")
    return unique[0]


BUNDLE_ROOT = _find_bundle_root()
sys.path.insert(0, str(BUNDLE_ROOT / "src"))

import cv2
import scipy
from skimage.metrics import structural_similarity
import torch
from torch import nn

from puzzle_assembly.contextual_refiner import (
    ContextualResidualNAF,
    build_context_features,
    model_parameter_count,
)
from puzzle_assembly.contextual_refiner_training import ContextualRefinerLoss
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.postassembly_harmonizer import (
    SeamGraphConfig,
    apply_rgb_offsets,
    blend_tiles_uint8,
    image_quality_metrics,
    ordered_from_slots,
    paired_bootstrap_ci,
    seam_graph_rgb_offsets,
)
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_denoise_v2.inference import restore_tiles_uint8
from puzzle_denoise_v2.tiles import merge_tiles_numpy, split_tiles_numpy
from puzzle_denoise_v2.training import build_model


CONFIG_SHA256 = "7de1a724a128a104467bf47ab1b075062bd2f689ba724c2a34897c1b28317c8e"
GATE_MANIFEST_SHA256 = "a288a773022ff068705642e9227cde7ba8d17abd4d84476d168359c55c137117"
PANELS = ("primary_kornia", "independent_libjpeg")
SEED = 20260712
TRAIN_SOURCES = 512
VAL_SOURCES = 32
STEPS = 2500
GLOBAL_BATCH = 2
EVAL_INTERVAL = 500


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


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise RuntimeError(f"invalid target image {path}: {values.shape}")
    return values


def _require_hash(path: Path, expected: str, role: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{role} hash mismatch: expected {expected}, got {actual}: {path}"
        )


def _discover_one(root: Path, pattern: str, role: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {role}, got {matches}")
    return matches[0]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _hardware_probe() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    devices = []
    for index in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{index}")
        value = torch.randn(256, 256, device=device) @ torch.randn(
            256, 256, device=device
        )
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "tensor_op_mean": float(value.mean().cpu()),
            }
        )
    nvidia_smi = subprocess.run(
        ["nvidia-smi"], capture_output=True, check=False, text=True
    )
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": devices,
        "nvidia_smi": nvidia_smi.stdout,
    }


def _load_tilenaf(checkpoint_path: Path, expected_sha256: str, device: torch.device):
    _require_hash(checkpoint_path, expected_sha256, checkpoint_path.name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(str(checkpoint["model_name"]))
    model.load_state_dict(checkpoint["ema_state"])
    return model.to(device).eval()


def _render_record(
    *,
    source: str,
    panel: str,
    target_dir: Path,
    selected_model: nn.Module,
    seam_model: nn.Module,
    device: torch.device,
    include_controls: bool,
) -> dict[str, np.ndarray | float | str | int]:
    clean = _read_rgb(target_dir / source)
    seed = per_source_seed(SEED, f"contextual-refiner-{panel}", source, 0)
    exact = make_exact_panel(clean, panel=panel, seed=seed)
    raw = ordered_from_slots(exact.slot_tiles, exact.slot_to_target)
    selected = restore_tiles_uint8(selected_model, raw, device, batch_size=576)
    seam = restore_tiles_uint8(seam_model, raw, device, batch_size=576)
    preanalytic = blend_tiles_uint8(selected, seam, auxiliary_weight=0.5)
    offsets, diagnostics = seam_graph_rgb_offsets(preanalytic, SeamGraphConfig())
    harmonized = apply_rgb_offsets(preanalytic, offsets)
    record = {
        "source": source,
        "panel": panel,
        "seed": seed,
        "preanalytic": merge_tiles_numpy(preanalytic),
        "harmonized": merge_tiles_numpy(harmonized),
        "target": clean,
        "seam_confidence": float(diagnostics["confidence_mean"]),
    }
    if include_controls:
        placebo_seed = per_source_seed(
            SEED, f"contextual-refiner-placebo-{panel}", source, 0
        )
        placebo_offsets, _ = seam_graph_rgb_offsets(
            preanalytic, SeamGraphConfig(), placebo_seed=placebo_seed
        )
        record["raw"] = merge_tiles_numpy(raw)
        record["placebo"] = merge_tiles_numpy(
            apply_rgb_offsets(preanalytic, placebo_offsets)
        )
    return record


def _save_record(path: Path, record: dict) -> None:
    arrays = {
        key: record[key] for key in ("preanalytic", "harmonized", "target")
    }
    for key in ("raw", "placebo"):
        if key in record:
            arrays[key] = record[key]
    arrays["seam_confidence"] = np.asarray(
        record["seam_confidence"], dtype=np.float32
    )
    np.savez(path, **arrays)


def _build_cache(
    *,
    cache_root: Path,
    train_names: list[str],
    val_names: list[str],
    target_dir: Path,
    selected_model: nn.Module,
    seam_model: nn.Module,
    device: torch.device,
) -> tuple[list[Path], list[dict]]:
    train_dir = cache_root / "train"
    val_dir = cache_root / "validation"
    train_dir.mkdir(parents=True)
    val_dir.mkdir()
    train_paths = []
    validation = []
    work = [("train", source, "primary_kornia") for source in train_names]
    work.extend(
        ("validation", source, panel) for source in val_names for panel in PANELS
    )
    for index, (split, source, panel) in enumerate(work, start=1):
        record = _render_record(
            source=source,
            panel=panel,
            target_dir=target_dir,
            selected_model=selected_model,
            seam_model=seam_model,
            device=device,
            include_controls=split == "validation",
        )
        name = f"{Path(source).stem}__{panel}.npz"
        path = (train_dir if split == "train" else val_dir) / name
        _save_record(path, record)
        if split == "train":
            train_paths.append(path)
        else:
            validation.append(
                {
                    "path": path,
                    "source": source,
                    "panel": panel,
                    "seam_confidence": record["seam_confidence"],
                }
            )
        if index % 16 == 0 or index == len(work):
            print(
                json.dumps(
                    {
                        "event": "cache_progress",
                        "completed": index,
                        "total": len(work),
                        "split": split,
                        "source": source,
                        "panel": panel,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return train_paths, validation


def _batch_from_paths(paths: list[Path], device: torch.device):
    values = [np.load(path, allow_pickle=False) for path in paths]
    try:
        arrays = {}
        for key in ("preanalytic", "harmonized", "target"):
            arrays[key] = torch.from_numpy(
                np.stack([np.asarray(value[key]) for value in values])
                .transpose(0, 3, 1, 2)
                .copy()
            ).to(device=device, dtype=torch.float32).div_(255.0)
        confidence = torch.tensor(
            [float(value["seam_confidence"]) for value in values],
            device=device,
            dtype=torch.float32,
        )[:, None, None, None].expand(-1, 1, 24, 24).contiguous()
    finally:
        for value in values:
            value.close()
    return arrays, confidence


def _augment(
    arrays: dict[str, torch.Tensor],
    confidence: torch.Tensor,
    rng: np.random.Generator,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    output = {key: value.clone() for key, value in arrays.items()}
    conf = confidence.clone()
    for index in range(len(conf)):
        k = int(rng.integers(0, 4))
        if k:
            for key in output:
                output[key][index] = torch.rot90(output[key][index], k, dims=(1, 2))
            conf[index] = torch.rot90(conf[index], k, dims=(1, 2))
        if bool(rng.integers(0, 2)):
            for key in output:
                output[key][index] = output[key][index].flip(-1)
            conf[index] = conf[index].flip(-1)
    return output, conf


def _update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for target, source in zip(ema.parameters(), model.parameters(), strict=True):
            target.mul_(decay).add_(source, alpha=1.0 - decay)
        for target, source in zip(ema.buffers(), model.buffers(), strict=True):
            target.copy_(source)


def _texture_gradient_mae(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = prediction.astype(np.float32) / 255.0
    true = target.astype(np.float32) / 255.0
    pred_x = pred[:, 1:] - pred[:, :-1]
    true_x = true[:, 1:] - true[:, :-1]
    pred_y = pred[1:] - pred[:-1]
    true_y = true[1:] - true[:-1]
    magnitude_x = np.sqrt(np.mean(true_x * true_x, axis=2))
    magnitude_y = np.sqrt(np.mean(true_y * true_y, axis=2))
    threshold_x = float(np.quantile(magnitude_x, 0.75))
    threshold_y = float(np.quantile(magnitude_y, 0.75))
    mask_x = magnitude_x >= threshold_x
    mask_y = magnitude_y >= threshold_y
    # Remove gradients that cross a predicted-layout seam.
    for boundary in range(20, 480, 20):
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
_FACE_CASCADE = (
    cv2.CascadeClassifier(str(_CASCADE_PATH)) if _CASCADE_PATH is not None else None
)


def _face_boxes(target: np.ndarray) -> np.ndarray:
    if _FACE_CASCADE is None:
        return np.empty((0, 4), dtype=np.int32)
    return _FACE_CASCADE.detectMultiScale(
        cv2.cvtColor(target, cv2.COLOR_RGB2GRAY),
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(24, 24),
    )


def _face_roi_ssim(
    prediction: np.ndarray,
    target: np.ndarray,
    boxes: np.ndarray,
) -> list[float]:
    values = []
    for x, y, width, height in boxes:
        true_crop = target[y : y + height, x : x + width]
        pred_crop = prediction[y : y + height, x : x + width]
        if min(true_crop.shape[:2]) >= 7:
            values.append(
                float(
                    structural_similarity(
                        true_crop, pred_crop, channel_axis=2, data_range=255
                    )
                )
            )
    return values


def _quality(
    prediction: np.ndarray,
    target: np.ndarray,
    face_boxes: np.ndarray,
) -> dict[str, float]:
    metrics = image_quality_metrics(
        split_tiles_numpy(prediction), split_tiles_numpy(target)
    )
    metrics["texture_gradient_mae"] = _texture_gradient_mae(prediction, target)
    face = _face_roi_ssim(prediction, target, face_boxes)
    metrics["face_roi_count"] = float(len(face))
    metrics["face_roi_ssim_sum"] = float(sum(face))
    return metrics


def _tensor_image(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1)))
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
        .unsqueeze(0)
    )


@torch.no_grad()
def _predict_arms(
    model: nn.Module,
    *,
    preanalytic: np.ndarray,
    harmonized: np.ndarray,
    seam_confidence: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    base = _tensor_image(harmonized, device)
    before = _tensor_image(preanalytic, device)
    confidence = torch.full(
        (1, 1, 24, 24), float(seam_confidence), device=device
    )
    layout = torch.ones_like(confidence)
    features, gate, seam = build_context_features(base, before, confidence, layout)
    prediction = model(base, features, gate, seam)
    shuffled = features.clone()
    # Preserve the candidate RGB itself while breaking both input-derived
    # correction context fields (analytic delta and 5x5 consensus).
    shuffled[:, 3:9] = torch.roll(
        shuffled[:, 3:9], shifts=(5 * 20, 7 * 20), dims=(2, 3)
    )
    context_placebo = model(base, shuffled, gate, seam)
    zero = model(base, features, torch.zeros_like(gate), seam)

    def uint8(value: torch.Tensor) -> np.ndarray:
        return (
            value[0]
            .detach()
            .float()
            .cpu()
            .mul(255.0)
            .round()
            .clamp(0, 255)
            .byte()
            .permute(1, 2, 0)
            .numpy()
        )

    return {
        "candidate": uint8(prediction),
        "context_placebo": uint8(context_placebo),
        "zero_confidence": uint8(zero),
    }


def _summarize(records: list[dict]) -> tuple[dict, dict]:
    panel_macro = {}
    comparisons = {}
    for panel_index, panel in enumerate(PANELS):
        subset = [record for record in records if record["panel"] == panel]
        arms = sorted(subset[0]["metrics"])
        panel_macro[panel] = {}
        for arm in arms:
            panel_macro[panel][arm] = {
                key: float(np.mean([record["metrics"][arm][key] for record in subset]))
                for key in subset[0]["metrics"][arm]
            }
        deltas = np.asarray(
            [
                record["metrics"]["candidate"]["ssim"]
                - record["metrics"]["analytic_identity"]["ssim"]
                for record in subset
            ]
        )
        low, high = paired_bootstrap_ci(
            deltas, seed=SEED + panel_index, resamples=20000
        )
        baseline_texture = panel_macro[panel]["analytic_identity"][
            "texture_gradient_mae"
        ]
        candidate_texture = panel_macro[panel]["candidate"]["texture_gradient_mae"]
        face_count = panel_macro[panel]["analytic_identity"]["face_roi_count"] * len(
            subset
        )
        candidate_face_sum = sum(
            record["metrics"]["candidate"]["face_roi_ssim_sum"] for record in subset
        )
        baseline_face_sum = sum(
            record["metrics"]["analytic_identity"]["face_roi_ssim_sum"]
            for record in subset
        )
        comparisons[panel] = {
            "source_count": len(subset),
            "mean_ssim_delta": float(deltas.mean()),
            "paired_bootstrap_95_ci": [low, high],
            "wins_ties_losses": [
                int(np.sum(deltas > 1e-12)),
                int(np.sum(np.abs(deltas) <= 1e-12)),
                int(np.sum(deltas < -1e-12)),
            ],
            "large_regressions_below_minus_0_01": int(np.sum(deltas < -0.01)),
            "mean_boundary_band_mae_delta": float(
                panel_macro[panel]["candidate"]["boundary_band_mae"]
                - panel_macro[panel]["analytic_identity"]["boundary_band_mae"]
            ),
            "mean_target_referenced_seam_error_delta": float(
                panel_macro[panel]["candidate"]["target_referenced_seam_error"]
                - panel_macro[panel]["analytic_identity"][
                    "target_referenced_seam_error"
                ]
            ),
            "texture_gradient_mae_ratio": float(
                candidate_texture / max(baseline_texture, 1e-12)
            ),
            "candidate_advantage_over_context_placebo": float(
                panel_macro[panel]["candidate"]["ssim"]
                - panel_macro[panel]["context_placebo"]["ssim"]
            ),
            "face_roi_count": int(round(face_count)),
            "face_roi_mean_ssim_delta": (
                float((candidate_face_sum - baseline_face_sum) / face_count)
                if face_count > 0
                else None
            ),
            "zero_confidence_byte_identity_all": bool(
                all(record["zero_confidence_byte_identity"] for record in subset)
            ),
        }
    return panel_macro, comparisons


@torch.no_grad()
def _evaluate_correct_layout(
    model: nn.Module,
    validation: list[dict],
    device: torch.device,
) -> dict:
    records = []
    for item in validation:
        with np.load(item["path"], allow_pickle=False) as payload:
            raw = np.asarray(payload["raw"])
            preanalytic = np.asarray(payload["preanalytic"])
            harmonized = np.asarray(payload["harmonized"])
            analytic_placebo = np.asarray(payload["placebo"])
            target = np.asarray(payload["target"])
            seam_confidence = float(payload["seam_confidence"])
        predicted = _predict_arms(
            model,
            preanalytic=preanalytic,
            harmonized=harmonized,
            seam_confidence=seam_confidence,
            device=device,
        )
        arms = {
            "raw": raw,
            "current_preanalytic": preanalytic,
            "analytic_identity": harmonized,
            "analytic_topology_placebo": analytic_placebo,
            **predicted,
        }
        faces = _face_boxes(target)
        records.append(
            {
                "source": item["source"],
                "panel": item["panel"],
                "metrics": {
                    name: _quality(value, target, faces) for name, value in arms.items()
                },
                "zero_confidence_byte_identity": bool(
                    np.array_equal(predicted["zero_confidence"], harmonized)
                ),
            }
        )
    panel_macro, comparisons = _summarize(records)
    return {
        "kind": "correct_layout_development_gate",
        "records": records,
        "panel_macro": panel_macro,
        "comparisons": comparisons,
    }


@torch.no_grad()
def _evaluate_actual_layout(
    model: nn.Module,
    *,
    gate_root: Path,
    gate_manifest: dict,
    target_dir: Path,
    device: torch.device,
) -> dict:
    records = []
    for item in gate_manifest["records"]:
        path = gate_root / item["artifact"]
        _require_hash(path, item["artifact_sha256"], "frozen gate artifact")
        with np.load(path, allow_pickle=False) as payload:
            arrays = {key: np.asarray(payload[key]) for key in ("raw", "preanalytic", "harmonized", "placebo")}
        for key, value in arrays.items():
            if _array_sha256(value) != item["array_sha256"][key]:
                raise RuntimeError(f"frozen gate array hash mismatch: {path}: {key}")
        preanalytic = merge_tiles_numpy(arrays["preanalytic"])
        harmonized = merge_tiles_numpy(arrays["harmonized"])
        predicted = _predict_arms(
            model,
            preanalytic=preanalytic,
            harmonized=harmonized,
            seam_confidence=float(item["target_blind_seam_confidence_mean"]),
            device=device,
        )
        target = _read_rgb(target_dir / item["source"])
        arms = {
            "raw": merge_tiles_numpy(arrays["raw"]),
            "current_preanalytic": preanalytic,
            "analytic_identity": harmonized,
            "analytic_topology_placebo": merge_tiles_numpy(arrays["placebo"]),
            **predicted,
        }
        faces = _face_boxes(target)
        records.append(
            {
                "source": item["source"],
                "panel": item["panel"],
                "layout_sha256": item["layout_sha256"],
                "layout_changed": item["layout_changed"],
                "metrics": {
                    name: _quality(value, target, faces) for name, value in arms.items()
                },
                "zero_confidence_byte_identity": bool(
                    np.array_equal(predicted["zero_confidence"], harmonized)
                ),
            }
        )
    panel_macro, comparisons = _summarize(records)
    return {
        "kind": "frozen_actual_qap_layout_one_shot_gate",
        "records": records,
        "panel_macro": panel_macro,
        "comparisons": comparisons,
        "all_layouts_unchanged": bool(
            all(record["layout_changed"] is False for record in records)
        ),
    }


def _gate(correct: dict, actual: dict) -> dict:
    per_panel = {}
    for panel in PANELS:
        corr = correct["comparisons"][panel]
        real = actual["comparisons"][panel]
        checks = {
            "correct_mean_ssim_delta_at_least_0_005": corr["mean_ssim_delta"] >= 0.005,
            "correct_bootstrap_lower_above_zero": corr["paired_bootstrap_95_ci"][0] > 0.0,
            "correct_boundary_mae_nonregression": corr["mean_boundary_band_mae_delta"] <= 0.0,
            "correct_texture_ratio_at_most_1_01": corr["texture_gradient_mae_ratio"] <= 1.01,
            "correct_no_large_regression": corr["large_regressions_below_minus_0_01"] == 0,
            "actual_mean_ssim_delta_at_least_0_002": real["mean_ssim_delta"] >= 0.002,
            "actual_bootstrap_lower_above_zero": real["paired_bootstrap_95_ci"][0] > 0.0,
            "actual_seam_error_nonregression": real["mean_target_referenced_seam_error_delta"] <= 0.0,
            "actual_texture_ratio_at_most_1_01": real["texture_gradient_mae_ratio"] <= 1.01,
            "actual_no_large_regression": real["large_regressions_below_minus_0_01"] == 0,
            "candidate_beats_context_placebo_by_0_001": real[
                "candidate_advantage_over_context_placebo"
            ] >= 0.001,
            "zero_confidence_exact_identity": real["zero_confidence_byte_identity_all"],
        }
        if real["face_roi_count"] >= 8:
            checks["actual_face_roi_nonregression"] = (
                real["face_roi_mean_ssim_delta"] >= -0.001
            )
        per_panel[panel] = {"checks": checks, "passed": bool(all(checks.values()))}
    return {
        "per_panel": per_panel,
        "all_layouts_unchanged": actual["all_layouts_unchanged"],
        "passed": bool(
            actual["all_layouts_unchanged"]
            and all(value["passed"] for value in per_panel.values())
        ),
        "continuation_to_10000_allowed": bool(
            actual["all_layouts_unchanged"]
            and all(value["passed"] for value in per_panel.values())
        ),
        "submission_promotion_allowed": False,
    }


def main() -> None:
    _seed_everything(SEED)
    probe = _hardware_probe()
    print(json.dumps({"event": "hardware_probe", **probe}, sort_keys=True), flush=True)
    input_root = Path("/kaggle/input")
    working = Path("/kaggle/working")
    config_path = BUNDLE_ROOT / "configs/postassembly_contextual_refiner_v1.json"
    _require_hash(config_path, CONFIG_SHA256, "contextual refiner config")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for role in ("manifest", "quarantine", "audit_exclusion"):
        record = config["authoritative_inputs"][role]
        _require_hash(BUNDLE_ROOT / record["path"], record["sha256"], role)
    for role, key in (("model", "model_path"), ("loss", "loss_path")):
        record = config["code"]
        _require_hash(
            BUNDLE_ROOT / record[key],
            record[f"{role}_sha256_at_design_freeze"],
            role,
        )

    target_dir = _discover_one(input_root, "train/targets", "target directory")
    selected_checkpoint = _discover_one(
        input_root, "selected_tilenaf_synth_50k.pt", "selected TileNAF"
    )
    seam_checkpoint = _discover_one(
        input_root, "seam_denoiser_gpu.pt", "production seam TileNAF"
    )
    gate_manifest_path = _discover_one(
        input_root, "gate_manifest.json", "frozen QAP gate manifest"
    )
    _require_hash(gate_manifest_path, GATE_MANIFEST_SHA256, "gate manifest")
    gate_manifest = json.loads(gate_manifest_path.read_text(encoding="utf-8"))
    if gate_manifest.get("target_pixels_included") is not False or gate_manifest.get("record_count") != 64:
        raise RuntimeError("frozen gate dataset contract mismatch")
    gate_root = gate_manifest_path.parent

    manifest_path = BUNDLE_ROOT / config["authoritative_inputs"]["manifest"]["path"]
    quarantine_path = BUNDLE_ROOT / config["authoritative_inputs"]["quarantine"]["path"]
    audit_path = BUNDLE_ROOT / config["authoritative_inputs"]["audit_exclusion"]["path"]
    train_names = source_names_for_split(
        "edge_train",
        manifest_path=manifest_path,
        quarantine_path=quarantine_path,
        audit_exclusion_path=audit_path,
    )[:TRAIN_SOURCES]
    val_names = source_names_for_split(
        "assembly_cal",
        manifest_path=manifest_path,
        quarantine_path=quarantine_path,
        audit_exclusion_path=audit_path,
    )[:VAL_SOURCES]
    actual_names = sorted({record["source"] for record in gate_manifest["records"]})
    if len(actual_names) != 32:
        raise RuntimeError("actual gate must contain exactly 32 whole sources")
    if set(train_names) & (set(val_names) | set(actual_names)) or set(val_names) & set(actual_names):
        raise RuntimeError("whole-source split overlap")

    render_device = torch.device("cuda:0")
    selected_model = _load_tilenaf(
        selected_checkpoint,
        config["authoritative_inputs"]["selected_tilenaf"]["sha256"],
        render_device,
    )
    seam_model = _load_tilenaf(
        seam_checkpoint,
        config["authoritative_inputs"]["production_seam_tilenaf"]["sha256"],
        render_device,
    )
    cache_root = Path("/kaggle/temp/contextual_refiner_cache")
    train_paths, validation = _build_cache(
        cache_root=cache_root,
        train_names=train_names,
        val_names=val_names,
        target_dir=target_dir,
        selected_model=selected_model,
        seam_model=seam_model,
        device=render_device,
    )
    del selected_model, seam_model
    torch.cuda.empty_cache()

    model = ContextualResidualNAF(width=48, blocks=12).to(render_device)
    if model_parameter_count(model) != config["model"]["parameter_count_expected"]:
        raise RuntimeError("model parameter-count drift")
    ema = copy.deepcopy(model).eval()
    train_model: nn.Module = model
    if torch.cuda.device_count() >= 2:
        train_model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    loss_fn = ContextualRefinerLoss().to(render_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    def lr_factor(step: int) -> float:
        if step < 100:
            return max(step, 1) / 100.0
        progress = (step - 100) / max(1, STEPS - 100)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(train_paths))
    pointer = 0
    history = []
    best_score = -math.inf
    best_step = 0
    best_path = working / "contextual_refiner_smoke_best.pt"

    for step in range(1, STEPS + 1):
        if pointer + GLOBAL_BATCH > len(order):
            order = rng.permutation(len(train_paths))
            pointer = 0
        indices = order[pointer : pointer + GLOBAL_BATCH]
        pointer += GLOBAL_BATCH
        arrays, confidence = _batch_from_paths(
            [train_paths[int(index)] for index in indices], render_device
        )
        arrays, confidence = _augment(arrays, confidence, rng)
        layout = torch.ones_like(confidence)
        features, gate, seam = build_context_features(
            arrays["harmonized"], arrays["preanalytic"], confidence, layout
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            prediction = train_model(arrays["harmonized"], features, gate, seam)
        loss, terms = loss_fn(
            prediction.float(), arrays["target"], arrays["harmonized"], seam
        )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        _update_ema(ema, model, decay=0.999)

        if step % 50 == 0:
            print(
                json.dumps(
                    {
                        "event": "train",
                        "step": step,
                        "loss": float(loss.detach()),
                        "lr": optimizer.param_groups[0]["lr"],
                        **{key: float(value) for key, value in terms.items()},
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if step % EVAL_INTERVAL == 0:
            correct = _evaluate_correct_layout(ema, validation, render_device)
            score = float(
                np.mean(
                    [
                        correct["comparisons"][panel]["mean_ssim_delta"]
                        for panel in PANELS
                    ]
                )
            )
            record = {
                "step": step,
                "correct_layout_mean_panel_ssim_delta": score,
                "comparisons": correct["comparisons"],
            }
            history.append(record)
            print(json.dumps({"event": "validation", **record}, sort_keys=True), flush=True)
            if score > best_score:
                best_score = score
                best_step = step
                torch.save(
                    {
                        "schema_version": 1,
                        "kind": "bounded_contextual_refiner_smoke",
                        "model_state": model.state_dict(),
                        "ema_state": ema.state_dict(),
                        "step": step,
                        "config_sha256": CONFIG_SHA256,
                        "train_names": train_names,
                        "val_names": val_names,
                        "history": history,
                        "model": config["model"],
                    },
                    best_path,
                )

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    ema.load_state_dict(checkpoint["ema_state"])
    ema.to(render_device).eval()
    correct_final = _evaluate_correct_layout(ema, validation, render_device)
    # This is the only checkpoint scored against the frozen actual-QAP neural gate.
    actual_final = _evaluate_actual_layout(
        ema,
        gate_root=gate_root,
        gate_manifest=gate_manifest,
        target_dir=target_dir,
        device=render_device,
    )
    gate = _gate(correct_final, actual_final)
    report = {
        "schema_version": 1,
        "kind": "bounded_contextual_refiner_smoke_report",
        "created_utc": _utc_now(),
        "status": "smoke_gate_passed" if gate["passed"] else "smoke_gate_failed_stop_or_pivot",
        "submission_promotion_allowed": False,
        "config_sha256": CONFIG_SHA256,
        "gate_manifest_sha256": GATE_MANIFEST_SHA256,
        "best_step": best_step,
        "best_correct_layout_selection_score": best_score,
        "checkpoint_sha256": _sha256(best_path),
        "source_protocol": {
            "train_count": len(train_names),
            "correct_layout_validation_count": len(val_names),
            "actual_qap_gate_count": len(actual_names),
            "whole_source_disjoint": True,
            "actual_qap_scored_checkpoint_count": 1,
        },
        "runtime": {
            "seconds": time.time() - STARTED,
            "hardware": probe,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "opencv": cv2.__version__,
            "data_parallel": isinstance(train_model, nn.DataParallel),
            "bootstrap_capabilities": BOOTSTRAP_CAPABILITIES,
            "bootstrap_torch_installed": BOOTSTRAP_TORCH_INSTALLED,
        },
        "history": history,
        "correct_layout": correct_final,
        "frozen_actual_qap": actual_final,
        "gate": gate,
    }
    report_path = working / "contextual_refiner_smoke_report.json"
    _write_json(report_path, report)
    result = {
        "status": report["status"],
        "best_step": best_step,
        "checkpoint": str(best_path),
        "checkpoint_sha256": _sha256(best_path),
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
        "gate": gate,
    }
    _write_json(working / "RESULT.json", result)
    shutil.rmtree(cache_root)
    print(json.dumps({"event": "complete", **result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
