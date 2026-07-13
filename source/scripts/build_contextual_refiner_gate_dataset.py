#!/usr/bin/env python3
"""Build the target-free reduced frozen-QAP gate dataset for Kaggle.

The source frozen render bundle is preserved.  This script copies only the four
pixel arms needed by the contextual-refiner evaluator and records byte hashes;
it never reads or packages clean targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = {
    "raw": "raw_on_frozen_qap_w4",
    "preanalytic": "fixed_alpha_0_5_on_frozen_qap_w4",
    "harmonized": "seam_graph_rgb_on_frozen_qap_w4",
    "placebo": "shuffled_neighbor_placebo_on_frozen_qap_w4",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    render_root = args.render_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    manifest_path = render_root / "render_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for record in manifest["records"]:
        source_path = render_root / record["render_artifact"]
        if _sha256(source_path) != record["render_artifact_sha256"]:
            raise RuntimeError(f"source render hash mismatch: {source_path}")
        with np.load(source_path, allow_pickle=False) as payload:
            arrays = {short: np.asarray(payload[long]) for short, long in ARMS.items()}
        for short, long in ARMS.items():
            expected = record["arm_sha256"][long]
            if _array_sha256(arrays[short]) != expected:
                raise RuntimeError(f"arm hash mismatch: {source_path}: {long}")
        # Keep files flat: Kaggle's default dataset uploader skips nested
        # directories unless it turns them into opaque zip/tar members.
        output_path = output_root / source_path.name
        np.savez_compressed(output_path, **arrays)
        records.append(
            {
                "source": record["source"],
                "panel": record["panel"],
                "artifact": str(output_path.relative_to(output_root)),
                "artifact_sha256": _sha256(output_path),
                "array_sha256": {
                    short: _array_sha256(array) for short, array in arrays.items()
                },
                "source_render_artifact_sha256": record["render_artifact_sha256"],
                "layout_sha256": record["layout_sha256"],
                "layout_changed": record["layout_changed"],
                "target_blind_seam_confidence_mean": record["diagnostics"][
                    "seam_graph_rgb_on_frozen_qap_w4"
                ]["confidence_mean"],
            }
        )
    if len(records) != 64:
        raise RuntimeError(f"expected 64 frozen records, got {len(records)}")
    output_manifest = {
        "schema_version": 1,
        "kind": "contextual_refiner_target_free_frozen_qap_gate",
        "target_pixels_included": False,
        "source_render_manifest_sha256": _sha256(manifest_path),
        "target_derived_metrics_included": False,
        "record_count": len(records),
        "records": records,
    }
    gate_manifest = output_root / "gate_manifest.json"
    gate_manifest.write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "record_count": len(records),
                "gate_manifest_sha256": _sha256(gate_manifest),
                "target_pixels_included": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
