#!/usr/bin/env python3
"""Export layout-only HBT/QAP real16 evidence from an audited solver report.

The source report also contains post-hoc target metrics.  This one-time exporter
whitelists only source names and frozen ``position_to_slot`` arrays, verifies
the source report's anti-leakage contract, and emits no metric or target field.
The resulting manifest is the only solver artifact consumed by the critic
diagnostic before predictions are frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from puzzle_assembly.layout_energy_hybrid import (  # noqa: E402
    sha256_array,
    sha256_file,
    validate_layout,
)


EXPECTED_SOURCE_REPORT_SHA256 = (
    "cc1b694b1501ba9b02e5618ad838e155ae40af7990bbbf4542b281fc21adec60"
)
LAYOUT_VARIANTS = {
    "hbt": "softcycle_l1_k8",
    "qap": "qap_softcycle_l1_k8",
}


def _input_shape(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return tuple(int(value) for value in array.shape)


def extract_layout_only_manifest(
    source_report: dict[str, Any],
    *,
    source_report_name: str,
    source_report_sha256: str,
    data_root: Path,
    expected_source_count: int = 16,
) -> dict[str, Any]:
    if source_report.get("schema_version") != 1:
        raise ValueError("unsupported source report schema")
    if source_report.get("kind") != "real_input_only_assembly_target_only_score":
        raise ValueError("wrong source report kind")
    if source_report.get("split") != "assembly_cal":
        raise ValueError("frozen layouts must come from assembly_cal")
    anti_leakage = source_report.get("anti_leakage")
    if anti_leakage != {
        "predictor_accepts_target": False,
        "target_opened_after_layouts_frozen": True,
        "pseudo_mapping_used": False,
    }:
        raise ValueError("source report anti-leakage contract is not exact")
    qap = source_report.get("qap")
    expected_qap = {
        "boundary_weight": 0.05,
        "initial_weight": 0.75,
        "iterations": 25,
        "noise_scale": 1.0,
        "noisy_components": 3,
        "refine_swaps": 8,
        "restarts": 2,
        "score": "l1w4",
        "seeds": ["softcycle_l1_k8"],
    }
    if qap != expected_qap:
        raise ValueError("source report is not the promoted boundary-QAP contract")
    names = source_report.get("source_names")
    sources = source_report.get("sources")
    if not isinstance(names, list) or not isinstance(sources, list):
        raise ValueError("source report names/sources are missing")
    if len(names) != expected_source_count or len(set(names)) != len(names):
        raise ValueError("unexpected source count or duplicate source name")
    if [record.get("source") for record in sources] != names:
        raise ValueError("source report order disagrees with source_names")

    exported_sources: list[dict[str, Any]] = []
    for name, record in zip(names, sources, strict=True):
        input_path = data_root / "train" / "inputs" / name
        if not input_path.is_file():
            raise FileNotFoundError(f"missing raw input: {input_path}")
        if _input_shape(input_path) != (480, 480, 3):
            raise ValueError(f"unexpected raw input shape: {input_path}")
        variants = record.get("variants")
        if not isinstance(variants, dict):
            raise ValueError(f"variants missing for {name}")
        layouts: dict[str, list[int]] = {}
        layout_hashes: dict[str, str] = {}
        for label, variant_prefix in LAYOUT_VARIANTS.items():
            raw_key = f"{variant_prefix}__raw_render"
            denoised_key = f"{variant_prefix}__denoised_render"
            raw_record = variants.get(raw_key)
            denoised_record = variants.get(denoised_key)
            if not isinstance(raw_record, dict) or not isinstance(denoised_record, dict):
                raise ValueError(f"required {label} variants missing for {name}")
            raw_layout = validate_layout(
                raw_record.get("position_to_slot"), count=576
            )
            denoised_layout = validate_layout(
                denoised_record.get("position_to_slot"), count=576
            )
            if not np.array_equal(raw_layout, denoised_layout):
                raise ValueError(f"raw/denoised {label} layouts disagree for {name}")
            layouts[label] = raw_layout.tolist()
            layout_hashes[label] = sha256_array(raw_layout)
        exported_sources.append(
            {
                "source": name,
                "raw_input": f"train/inputs/{name}",
                "raw_input_sha256": sha256_file(input_path),
                "layouts": layouts,
                "layout_sha256": layout_hashes,
            }
        )
    return {
        "schema_version": 1,
        "kind": "frozen_real16_hbt_qap_layout_only_manifest",
        "safe_for_submission": False,
        "source_report": source_report_name,
        "source_report_sha256": source_report_sha256,
        "source_report_contract": {
            "split": "assembly_cal",
            "target_opened_after_layouts_frozen": True,
            "variant_prefixes": LAYOUT_VARIANTS,
            "qap": expected_qap,
        },
        "export_contract": {
            "allowed_source_fields": [
                "source",
                "variants.*.position_to_slot",
            ],
            "target_paths_accessed": False,
            "target_metrics_exported": False,
            "raw_inputs_read_only_for_shape_and_sha256": True,
        },
        "source_names": names,
        "sources": exported_sources,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--expected-source-report-sha256",
        default=EXPECTED_SOURCE_REPORT_SHA256,
    )
    parser.add_argument("--expected-source-count", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_report)
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {output_path}")
    actual_sha256 = sha256_file(source_path)
    if actual_sha256 != args.expected_source_report_sha256:
        raise SystemExit(
            "source report sha256 mismatch: "
            f"expected {args.expected_source_report_sha256}, got {actual_sha256}"
        )
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    manifest = extract_layout_only_manifest(
        payload,
        source_report_name=source_path.name,
        source_report_sha256=actual_sha256,
        data_root=Path(args.data_root),
        expected_source_count=args.expected_source_count,
    )
    _atomic_json(output_path, manifest)
    print(
        json.dumps(
            {
                "event": "frozen_layout_manifest_written",
                "output": str(output_path),
                "sha256": sha256_file(output_path),
                "source_count": len(manifest["source_names"]),
                "target_metrics_exported": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
