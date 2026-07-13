#!/usr/bin/env python3
"""Re-render one preselected fixed layout family with a different denoiser."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from puzzle_assembly.metrics import predicted_image_metrics
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8
from puzzle_denoise_v2.tiles import split_tiles_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-report", required=True)
    parser.add_argument("--layout-variant", required=True)
    parser.add_argument("--renderer-denoiser", required=True)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise ValueError(f"unexpected image shape for {path}: {values.shape}")
    return values


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {output}")
    layout_path = Path(args.layout_report)
    layout_report = json.loads(layout_path.read_text(encoding="utf-8"))
    if not layout_report.get("anti_leakage", {}).get(
        "target_opened_after_layouts_frozen", False
    ):
        raise SystemExit("layout report lacks the input-only anti-leakage invariant")
    names = layout_report["source_names"]
    source_by_name = {source["source"]: source for source in layout_report["sources"]}
    if set(names) != set(source_by_name):
        raise SystemExit("layout report source mismatch")
    renderer_path = Path(args.renderer_denoiser)
    renderer, device, renderer_metadata = load_restorer(
        renderer_path, device=args.device, state="ema"
    )
    sources = []
    for index, name in enumerate(names):
        frozen = source_by_name[name]["variants"].get(args.layout_variant)
        if frozen is None:
            raise SystemExit(f"missing layout variant {args.layout_variant!r} for {name}")
        layout = np.asarray(frozen["position_to_slot"], dtype=np.int32)
        if layout.shape != (576,) or not np.array_equal(np.sort(layout), np.arange(576)):
            raise SystemExit(f"invalid frozen layout for {name}")
        input_image = _read_rgb(Path(args.data_root) / "train" / "inputs" / name)
        render_tiles = restore_tiles_uint8(
            renderer,
            split_tiles_numpy(input_image),
            device,
            batch_size=args.batch_size,
        )
        # Target access occurs only after the fixed layout and render pixels exist.
        target = _read_rgb(Path(args.data_root) / "train" / "targets" / name)
        metrics = predicted_image_metrics(layout, render_tiles, target)
        sources.append({"source": name, "metrics": metrics})
        print(
            json.dumps(
                {
                    "event": "fixed_layout_rerender_source",
                    "index": index + 1,
                    "count": len(names),
                    "source": name,
                    "ssim": metrics["predicted_layout_ssim"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    macro = {
        metric: float(np.mean([source["metrics"][metric] for source in sources]))
        for metric in ("predicted_layout_ssim", "psnr", "mae")
    }
    payload = {
        "schema_version": 1,
        "kind": "fixed_input_only_layout_renderer_evaluation",
        "anti_leakage": {
            "layout_report_is_frozen": True,
            "fixed_variant_selected_globally": True,
            "target_opened_after_layout_and_render_frozen": True,
        },
        "layout_report": str(layout_path),
        "layout_report_sha256": _sha256(layout_path),
        "layout_variant": args.layout_variant,
        "renderer_denoiser": str(renderer_path),
        "renderer_denoiser_sha256": _sha256(renderer_path),
        "renderer_metadata": renderer_metadata,
        "source_names": names,
        "sources": sources,
        "macro": macro,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "fixed_layout_rerender_complete", "macro": macro}))


if __name__ == "__main__":
    main()
