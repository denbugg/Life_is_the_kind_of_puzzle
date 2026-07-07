"""Synthetic per-fragment distortion matching the challenge degradation.

Order (reverse-engineered from real pairs): per fragment, independently:
  affine(contrast around gray-mean + additive brightness) -> add Gaussian noise
  -> 3x3 Gaussian blur -> JPEG re-encode.
The heavy injected noise (sigma 40-55) is smoothed by blur+JPEG to ~14 std
correlated noise in the final image, matching measurements on real inputs.
"""
import numpy as np
import cv2
from config import FS, BRIGHT, CONTRAST, NOISE_SIGMA, JPEG_Q

_GRAY = np.array([0.299, 0.587, 0.114], np.float32)


def _blur3(x):
    """Separable 3x3 Gaussian ([.25,.5,.25]) with reflect pad. x:(N,H,W,3) float."""
    k0, k1, k2 = 0.25, 0.5, 0.25
    xp = np.pad(x, ((0, 0), (1, 1), (0, 0), (0, 0)), mode="reflect")
    x = k0 * xp[:, :-2] + k1 * xp[:, 1:-1] + k2 * xp[:, 2:]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (0, 0)), mode="reflect")
    x = k0 * xp[:, :, :-2] + k1 * xp[:, :, 1:-1] + k2 * xp[:, :, 2:]
    return x


def distort_frags(frags, rng=None):
    """frags: (N,20,20,3) uint8 -> distorted (N,20,20,3) uint8, per-fragment independent."""
    if rng is None:
        rng = np.random
    N = frags.shape[0]
    x = frags.astype(np.float32)

    # --- affine: contrast around per-fragment gray mean, then +brightness ---
    a = rng.uniform(CONTRAST[0], CONTRAST[1], size=(N, 1, 1, 1)).astype(np.float32)
    b = rng.uniform(-BRIGHT, BRIGHT, size=(N, 1, 1, 1)).astype(np.float32)
    pivot = (x * _GRAY).sum(-1, keepdims=True).mean(axis=(1, 2), keepdims=True)
    x = a * (x - pivot) + pivot + b

    # --- additive Gaussian noise, per-fragment sigma ---
    sig = rng.uniform(NOISE_SIGMA[0], NOISE_SIGMA[1], size=(N, 1, 1, 1)).astype(np.float32)
    x = x + rng.standard_normal(x.shape).astype(np.float32) * sig
    x = np.clip(x, 0, 255)

    # --- 3x3 Gaussian blur (per fragment, reflect border) ---
    x = _blur3(x)
    x = np.clip(x, 0, 255).astype(np.uint8)

    # --- per-fragment JPEG re-encode ---
    q = rng.integers(JPEG_Q[0], JPEG_Q[1] + 1, size=N)
    out = np.empty_like(x)
    for i in range(N):
        bgr = np.ascontiguousarray(x[i][..., ::-1])  # RGB->BGR for standard JPEG luma
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(q[i])])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)  # BGR
            out[i] = dec[..., ::-1]                    # back to RGB
        else:
            out[i] = x[i]
    return out


def distort_image(img, rng=None):
    """Distort every 20x20 fragment of a 480x480 image, keeping positions (correct order)."""
    from imgio import to_frags, from_frags
    return from_frags(distort_frags(to_frags(img), rng))
