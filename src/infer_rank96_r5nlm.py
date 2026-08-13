"""Offline S1 runner: frozen rank96 layout followed by R5 and canonical NLM.

The only change from the canonical rank96 output arm is pixel restoration:
raw assembled board -> frozen FP32 R5 RestoreNet -> infer_rank96.fixed_nlm.
Candidate mining, scoring, assignment, orientation and all test filename rules
remain fixed.  This program never reads targets or creates a platform upload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from config import TEST_DIR
from models import RestoreNet
import infer_rank96 as rank96

DEFAULT_WORK = Path(r"E:\pazzle_work\submissions\rank96_r5nlm_s1")
DEFAULT_R5 = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R5_restore_unet\r5_capacity_fp32.pt")
MANIFEST_SCHEMA = "pazzle-rank96-r5nlm-s1-manifest-v1"
REPORT_SCHEMA = "pazzle-rank96-r5nlm-s1-report-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def atomic_rgb_png(path: Path, image: np.ndarray) -> None:
    value = rank96._validate_rgb_array(image, label="S1 R5-NLM output")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.stem + ".", suffix=".png", dir=path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        Image.fromarray(value, mode="RGB").save(temp_path, format="PNG", optimize=False)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def deterministic_zip(output_dir: Path, names: list[str], destination: Path) -> str:
    temp = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in names:
            path = output_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"missing output for deterministic ZIP: {path}")
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 14, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    os.replace(temp, destination)
    return file_sha256(destination)


def load_r5(path: Path, device: str, base: int, depth: int) -> RestoreNet:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if isinstance(payload, dict):
        state = payload.get("model") or payload.get("model_state_dict") or payload.get("state_dict") or payload
    else:
        state = payload
    model = RestoreNet(base=base, depth=depth).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def restore_r5(layout: np.ndarray, model: RestoreNet, device: str) -> np.ndarray:
    value = rank96._validate_rgb_array(layout, label="raw rank96 assembled layout")
    height, width = value.shape[:2]
    if height % 8 or width % 8:
        raise RuntimeError(f"R5 depth-4 input must be divisible by 8, got {value.shape}")
    with torch.no_grad():
        source = torch.from_numpy(value).to(device=device, dtype=torch.float32)
        source = source.permute(2, 0, 1).unsqueeze(0).div_(255.0)
        result = model(source).clamp_(0.0, 1.0)
        result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return rank96._validate_rgb_array(np.rint(result * 255.0).clip(0, 255).astype(np.uint8), label="R5 output")


def discover_names(input_dir: Path, expected_count: int) -> list[str]:
    names = sorted(path.name for path in input_dir.glob("*.png") if path.is_file())
    if expected_count and len(names) != expected_count:
        raise RuntimeError(f"expected exactly {expected_count} .png inputs, found {len(names)} in {input_dir}")
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate filename discovered")
    return names


def build_config(args: argparse.Namespace) -> tuple[rank96.InferenceConfig, dict[str, Path]]:
    paths = rank96._default_checkpoints()
    config = rank96.InferenceConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_zip=args.output_zip,
        ranker_checkpoint=paths["ranker"],
        affinity_primary_checkpoint=paths["affinity_primary"],
        affinity_secondary_checkpoint=paths["affinity_secondary"],
        device=args.device,
        pair_batch=args.pair_batch,
        expected_count=args.expected_count,
    )
    return config, paths


def contract(args: argparse.Namespace, checkpoint_paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "pipeline": "fixed_rank96_raw_layout__R5_FP32__canonical_fixed_nlm",
        "orientation": "fixed_upright_tiles_no_rotation",
        "rank96_contract": rank96.RANK96_CONTRACT,
        "r5": {
            "checkpoint": str(args.r5_checkpoint.resolve()),
            "sha256": file_sha256(args.r5_checkpoint),
            "base": args.r5_base,
            "depth": args.r5_depth,
            "dtype": "float32",
            "input": "assembled_480x480_rgb_raw_rank96_layout",
        },
        "rank96_checkpoints": {key: {"path": str(value.resolve()), "sha256": file_sha256(value)} for key, value in checkpoint_paths.items()},
        "expected_count": args.expected_count,
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "seed": args.seed,
        "pair_batch": args.pair_batch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline S1: frozen rank96 plus R5-to-canonical-NLM restoration.")
    parser.add_argument("--input-dir", type=Path, default=Path(TEST_DIR))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_WORK / "png")
    parser.add_argument("--output-zip", type=Path, default=DEFAULT_WORK / "submission_rank96_r5nlm_s1.zip")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--r5-checkpoint", type=Path, default=DEFAULT_R5)
    parser.add_argument("--r5-base", type=int, default=32)
    parser.add_argument("--r5-depth", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pair-batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--expected-count", type=int, default=700)
    parser.add_argument("--limit", type=int, default=0, help="process at most N names; zero means all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runtime-seconds", type=float, default=0.0)
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"input directory missing: {args.input_dir}")
    if not args.r5_checkpoint.is_file():
        raise FileNotFoundError(f"R5 checkpoint missing: {args.r5_checkpoint}")
    if args.expected_count and args.limit and args.limit < args.expected_count:
        # A bounded smoke is explicitly not a submission and cannot receive a ZIP.
        args.output_zip = None
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config, checkpoint_paths = build_config(args)
    manifest_path = args.manifest or args.output_dir / "s1_manifest.json"
    report_path = args.report or args.output_dir / "s1_report.json"
    immutable_contract = contract(args, checkpoint_paths)
    contract_digest = hashlib.sha256(canonical_json(immutable_contract)).hexdigest()
    names = discover_names(args.input_dir, args.expected_count)
    selected_names = names[: args.limit] if args.limit else names
    args.output_dir.mkdir(parents=True, exist_ok=True)

    existing: dict[str, Any] = {}
    if args.resume and manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("contract_digest") != contract_digest:
            raise RuntimeError("refusing resume: immutable S1 contract differs from existing manifest")
        existing = dict(prior.get("completed", {}))
    elif not args.resume and any(args.output_dir.glob("*.png")):
        raise RuntimeError("output directory already contains PNGs; use a new directory or --resume with matching manifest")

    resolved_device = rank96.resolve_device(args.device)
    models = rank96.load_models(config, resolved_device)
    r5 = load_r5(args.r5_checkpoint, str(resolved_device), args.r5_base, args.r5_depth)
    started = time.perf_counter()
    completed: dict[str, Any] = dict(existing)
    new_count = 0
    skipped_count = 0
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "in_progress",
        "contract": immutable_contract,
        "contract_digest": contract_digest,
        "input_count": len(names),
        "selected_count": len(selected_names),
        "completed": completed,
    }
    atomic_json(manifest_path, manifest)

    for ordinal, name in enumerate(selected_names, 1):
        input_path = args.input_dir / name
        output_path = args.output_dir / name
        input_hash = file_sha256(input_path)
        prior = completed.get(name)
        if prior and output_path.is_file() and prior.get("input_sha256") == input_hash and prior.get("output_sha256") == file_sha256(output_path):
            skipped_count += 1
            continue
        image_started = time.perf_counter()
        dirty_image = rank96.load_rgb_strict(input_path)
        inferred = rank96.infer_one(dirty_image, models, pair_batch=args.pair_batch)
        dirty_tiles = rank96.split_upright_tiles(dirty_image)
        raw_layout = rank96.assemble_upright_tiles(dirty_tiles, inferred.board)
        r5_layout = restore_r5(raw_layout, r5, str(resolved_device))
        final_layout = rank96.fixed_nlm(r5_layout)
        atomic_rgb_png(output_path, final_layout)
        output_hash = file_sha256(output_path)
        record = {
            "input_sha256": input_hash,
            "output_sha256": output_hash,
            "raw_layout_sha256": array_sha256(raw_layout),
            "r5_layout_sha256": array_sha256(r5_layout),
            "final_layout_sha256": array_sha256(final_layout),
            "board_sha256": rank96.sha256_array(inferred.board.astype(np.int16)),
            "candidate_ids_sha256": inferred.candidate_ids_sha256,
            "raw_scores_sha256": inferred.raw_scores_sha256,
            "rank96_objective": float(inferred.objective),
            "bytes": output_path.stat().st_size,
            "seconds": round(time.perf_counter() - image_started, 3),
        }
        completed[name] = record
        new_count += 1
        manifest["completed"] = completed
        atomic_json(manifest_path, manifest)
        print(json.dumps({"ordinal": ordinal, "name": name, "seconds": record["seconds"], "board_sha256": record["board_sha256"]}), flush=True)
        if args.max_runtime_seconds and time.perf_counter() - started >= args.max_runtime_seconds:
            manifest["status"] = "incomplete_budget"
            atomic_json(manifest_path, manifest)
            report = {"schema": REPORT_SCHEMA, "status": manifest["status"], "contract_digest": contract_digest, "completed_count": len(completed), "new_count": new_count, "skipped_count": skipped_count, "elapsed_seconds": time.perf_counter() - started, "manifest": str(manifest_path)}
            atomic_json(report_path, report)
            raise SystemExit(rank96.INCOMPLETE_EXIT_CODE)

    manifest["status"] = "completed" if len(selected_names) == len(names) else "smoke_completed"
    atomic_json(manifest_path, manifest)
    zip_sha256 = None
    if len(selected_names) == len(names) and args.output_zip is not None:
        zip_sha256 = deterministic_zip(args.output_dir, names, args.output_zip)
    report = {
        "schema": REPORT_SCHEMA,
        "status": manifest["status"],
        "contract_digest": contract_digest,
        "input_count": len(names),
        "completed_count": len(completed),
        "new_count": new_count,
        "skipped_count": skipped_count,
        "elapsed_seconds": time.perf_counter() - started,
        "manifest": str(manifest_path),
        "output_dir": str(args.output_dir),
        "output_zip": str(args.output_zip) if zip_sha256 else None,
        "output_zip_sha256": zip_sha256,
    }
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
