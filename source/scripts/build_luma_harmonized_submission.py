#!/usr/bin/env python3
"""Build the promoted RGB-harmonized submission plus bounded luminance gain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import build_harmonized_submission as base
from puzzle_assembly.postassembly_harmonizer import (
    LuminanceGainConfig,
    apply_luminance_gains,
    seam_graph_luminance_gains,
)
from puzzle_denoise_v2.tiles import merge_tiles_numpy, split_tiles_numpy


GAIN_CONFIG = LuminanceGainConfig(
    extrapolation_band=3,
    confidence_scale=0.08,
    confidence_floor=0.05,
    ridge=0.5,
    huber_delta=0.025,
    irls_steps=4,
    max_fractional_gain=0.04,
    luminance_floor=12.0,
    luminance_ceiling=243.0,
)
CALIBRATION_REPORT_SHA256 = "9593b8809d2b0e6a0f928e7b0cf41e47f4406e5dd3d8151ea6b75a521c65bbee"
CONFIRMATION_REPORT_SHA256 = "aeac52ebdde35581a974ac863ccb9f7af22f4c4153b667ccb81f68c884b158c1"


def main() -> None:
    original_renderer = base.render_harmonized_tiles

    def render_with_gain(
        selected_slot_tiles: np.ndarray,
        seam_slot_tiles: np.ndarray,
        layout: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        rgb_image, diagnostics = original_renderer(
            selected_slot_tiles, seam_slot_tiles, layout
        )
        rgb_tiles = split_tiles_numpy(rgb_image)
        gains, gain_diagnostics = seam_graph_luminance_gains(rgb_tiles, GAIN_CONFIG)
        corrected = merge_tiles_numpy(apply_luminance_gains(rgb_tiles, gains))
        return corrected, {
            **diagnostics,
            "bounded_luminance_gain": gain_diagnostics,
            "gain_sha256": hashlib.sha256(
                np.asarray(gains, dtype=np.float32).tobytes()
            ).hexdigest(),
        }

    base.render_harmonized_tiles = render_with_gain
    args = base.parse_args()
    payload = base.build_submission(args)
    payload["kind"] = "luma_harmonized_frozen_qap_submission_report"
    payload["method"]["bounded_luminance_gain"] = {
        "input": "seam_graph_rgb_on_frozen_qap_w4",
        "extrapolation_band": GAIN_CONFIG.extrapolation_band,
        "confidence_scale": GAIN_CONFIG.confidence_scale,
        "confidence_floor": GAIN_CONFIG.confidence_floor,
        "ridge": GAIN_CONFIG.ridge,
        "huber_delta": GAIN_CONFIG.huber_delta,
        "irls_steps": GAIN_CONFIG.irls_steps,
        "max_fractional_gain": GAIN_CONFIG.max_fractional_gain,
        "luminance_floor": GAIN_CONFIG.luminance_floor,
        "luminance_ceiling": GAIN_CONFIG.luminance_ceiling,
        "calibration_report_sha256": CALIBRATION_REPORT_SHA256,
        "confirmation_report_sha256": CONFIRMATION_REPORT_SHA256,
        "confirmation_passed": True,
    }
    payload["method_sha256"] = base.canonical_sha256(payload["method"])
    payload["promotion_evidence"] = {
        "calibration_report_sha256": CALIBRATION_REPORT_SHA256,
        "confirmation_report_sha256": CONFIRMATION_REPORT_SHA256,
        "source_disjoint_confirmation_passed": True,
    }
    Path(args.report).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "luma_harmonized_submission_complete",
                "output": str(args.output),
                "sha256": payload["archive"]["sha256"],
                "count": payload["count"],
                "method_sha256": payload["method_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
