"""Exact uint8 evaluation metrics for isolated tiles."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

from .tiles import merge_tiles_numpy


def _as_uint8_hwc(tiles: np.ndarray) -> np.ndarray:
    array = np.asarray(tiles)
    if array.ndim != 4:
        raise ValueError(f"expected NHWC or NCHW tiles, got {array.shape}")
    if array.shape[1:] == (3, 20, 20):
        array = array.transpose(0, 2, 3, 1)
    if array.shape[1:] != (20, 20, 3):
        raise ValueError(f"expected 20x20 RGB tiles, got {array.shape}")
    if array.dtype != np.uint8:
        scale = 255.0 if float(array.max(initial=0)) <= 1.0 else 1.0
        array = np.clip(np.rint(array * scale), 0, 255).astype(np.uint8)
    return array


def tile_metrics(prediction: np.ndarray, target: np.ndarray, boundary_band: int = 3) -> dict[str, float]:
    prediction = _as_uint8_hwc(prediction)
    target = _as_uint8_hwc(target)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")

    pred_float = prediction.astype(np.float64)
    target_float = target.astype(np.float64)
    error = pred_float - target_float
    mse = float(np.mean(np.square(error)))
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10((255.0**2) / mse))
    ssims = [
        structural_similarity(true, pred, channel_axis=2, data_range=255, win_size=7)
        for pred, true in zip(prediction, target, strict=True)
    ]

    mask = np.zeros((20, 20), dtype=bool)
    mask[:boundary_band] = True
    mask[-boundary_band:] = True
    mask[:, :boundary_band] = True
    mask[:, -boundary_band:] = True
    absolute = np.abs(error)
    grad_pred_x = pred_float[:, :, 1:] - pred_float[:, :, :-1]
    grad_true_x = target_float[:, :, 1:] - target_float[:, :, :-1]
    grad_pred_y = pred_float[:, 1:] - pred_float[:, :-1]
    grad_true_y = target_float[:, 1:] - target_float[:, :-1]

    return {
        "tile_ssim": float(np.mean(ssims)),
        "psnr": psnr,
        "mae": float(absolute.mean()),
        "boundary_mae": float(absolute[:, mask, :].mean()),
        "interior_mae": float(absolute[:, ~mask, :].mean()),
        "gradient_mae": float(
            0.5 * (np.abs(grad_pred_x - grad_true_x).mean() + np.abs(grad_pred_y - grad_true_y).mean())
        ),
        "signed_bias_r": float(error[..., 0].mean()),
        "signed_bias_g": float(error[..., 1].mean()),
        "signed_bias_b": float(error[..., 2].mean()),
    }


def ordered_image_ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    """Official RGB SSIM after merging each consecutive set of 576 tiles."""
    prediction = _as_uint8_hwc(prediction)
    target = _as_uint8_hwc(target)
    if prediction.shape != target.shape or len(prediction) % 576:
        raise ValueError("ordered_image_ssim requires matching complete 576-tile images")
    scores = []
    for start in range(0, len(prediction), 576):
        pred_image = merge_tiles_numpy(prediction[start : start + 576])
        true_image = merge_tiles_numpy(target[start : start + 576])
        scores.append(structural_similarity(true_image, pred_image, channel_axis=2, data_range=255))
    return float(np.mean(scores))
