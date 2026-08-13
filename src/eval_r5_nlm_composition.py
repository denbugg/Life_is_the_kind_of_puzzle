"""Source-disjoint R5/NLM composition gate on shared frozen rank96 layouts.

Each board is inferred exactly once from its corrupted input.  All variants are
pixel-only transformations of that same 480x480 board; target access occurs only
when calculating post-hoc SSIM.  Test images and submission writing are absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from config import TRAIN_INP, TRAIN_TGT
from models import RestoreNet
import infer_rank96 as rank96

DEFAULT_WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R5_restore_unet")
DEFAULT_SPLIT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json")
DEFAULT_R5 = DEFAULT_WORK / "r5_capacity_fp32.pt"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def lower_95(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return float(arr[0]) if len(arr) else float("nan")
    return float(arr.mean() - 1.96 * arr.std(ddof=1) / math.sqrt(len(arr)))


def build_config(device: str, pair_batch: int, work: Path) -> tuple[Any, dict[str, Path]]:
    paths = rank96._default_checkpoints()
    config = rank96.InferenceConfig(
        input_dir=Path(TRAIN_INP),
        output_dir=work / "rank96_unused_outputs",
        output_zip=None,
        ranker_checkpoint=paths["ranker"],
        affinity_primary_checkpoint=paths["affinity_primary"],
        affinity_secondary_checkpoint=paths["affinity_secondary"],
        device=device,
        pair_batch=pair_batch,
        expected_count=700,
    )
    return config, paths


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
    if layout.dtype != np.uint8 or layout.ndim != 3 or layout.shape[-1] != 3:
        raise ValueError(f"expected uint8 HWC layout, got {layout.shape} {layout.dtype}")
    height, width = layout.shape[:2]
    if height % 8 or width % 8:
        raise ValueError(f"RestoreNet depth-4 requires dimensions divisible by 8, got {layout.shape}")
    with torch.no_grad():
        source = torch.from_numpy(layout).to(device=device, dtype=torch.float32)
        source = source.permute(2, 0, 1).unsqueeze(0).div_(255.0)
        output = model(source).clamp_(0.0, 1.0)
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.rint(output * 255.0).clip(0, 255).astype(np.uint8)


def canonical_nlm(layout: np.ndarray, h: float, h_color: float, template: int, search: int) -> np.ndarray:
    """OpenCV colored fast-NLM under canonical rank96 contract parameters.

    imgio/rank96 arrays are RGB.  Explicit BGR conversion maintains the OpenCV
    API color contract and returns the repository-standard RGB representation.
    """
    if layout.dtype != np.uint8 or layout.ndim != 3 or layout.shape[-1] != 3:
        raise ValueError(f"expected uint8 HWC RGB layout, got {layout.shape} {layout.dtype}")
    bgr = cv2.cvtColor(layout, cv2.COLOR_RGB2BGR)
    filtered = cv2.fastNlMeansDenoisingColored(bgr, None, h, h_color, template, search)
    return cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)


def paired_summary(rows: list[dict[str, Any]], key: str, baseline: str) -> dict[str, float]:
    differences = [float(row[key] - row[baseline]) for row in rows]
    return {
        "mean": float(np.mean(differences)),
        "min": float(np.min(differences)),
        "lower_95": lower_95(differences),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--partition", choices=("cal", "dev"), default="dev")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pair-batch", type=int, default=128)
    parser.add_argument("--r5-checkpoint", type=Path, default=DEFAULT_R5)
    parser.add_argument("--r5-base", type=int, default=32)
    parser.add_argument("--r5-depth", type=int, default=4)
    parser.add_argument("--nlm-h", type=float, default=10.0)
    parser.add_argument("--nlm-h-color", type=float, default=10.0)
    parser.add_argument("--nlm-template", type=int, default=7)
    parser.add_argument("--nlm-search", type=int, default=21)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    if args.n < 2:
        parser.error("--n must be at least two for lower-confidence composition tests")
    if not args.r5_checkpoint.is_file():
        raise FileNotFoundError(f"R5 checkpoint unavailable: {args.r5_checkpoint}")

    split = json.loads(args.split.read_text(encoding="utf-8"))
    names = list(split["splits"][args.partition][: args.n])
    if len(names) != args.n:
        raise RuntimeError("requested split size unavailable")
    args.work.mkdir(parents=True, exist_ok=True)

    config, rank96_paths = build_config(args.device, args.pair_batch, args.work)
    resolved_device = rank96.resolve_device(args.device)
    models = rank96.load_models(config, resolved_device)
    r5 = load_r5(args.r5_checkpoint, str(resolved_device), args.r5_base, args.r5_depth)

    rows: list[dict[str, Any]] = []
    for ordinal, name in enumerate(names, 1):
        dirty_image = rank96.load_rgb_strict(Path(TRAIN_INP) / name)
        target = rank96.load_rgb_strict(Path(TRAIN_TGT) / name)
        inferred = rank96.infer_one(dirty_image, models, pair_batch=args.pair_batch)
        dirty_tiles = rank96.split_upright_tiles(dirty_image)
        raw = rank96.assemble_upright_tiles(dirty_tiles, inferred.board)
        nlm = canonical_nlm(raw, args.nlm_h, args.nlm_h_color, args.nlm_template, args.nlm_search)
        r5_raw = restore_r5(raw, r5, str(resolved_device))
        r5_then_nlm = canonical_nlm(r5_raw, args.nlm_h, args.nlm_h_color, args.nlm_template, args.nlm_search)
        nlm_then_r5 = restore_r5(nlm, r5, str(resolved_device))
        variants = {
            "raw_ssim": raw,
            "nlm_ssim": nlm,
            "r5_ssim": r5_raw,
            "r5_then_nlm_ssim": r5_then_nlm,
            "nlm_then_r5_ssim": nlm_then_r5,
        }
        scores = {key: float(ssim(image, target, channel_axis=2, data_range=255)) for key, image in variants.items()}
        row = {
            "name": name,
            "rank96_objective": float(inferred.objective),
            **scores,
            "nlm_delta": scores["nlm_ssim"] - scores["raw_ssim"],
            "r5_delta": scores["r5_ssim"] - scores["raw_ssim"],
            "r5_then_nlm_delta": scores["r5_then_nlm_ssim"] - scores["raw_ssim"],
            "nlm_then_r5_delta": scores["nlm_then_r5_ssim"] - scores["raw_ssim"],
            "board_sha256": rank96.sha256_array(inferred.board.astype(np.int16)),
            "candidate_ids_sha256": inferred.candidate_ids_sha256,
            "raw_scores_sha256": inferred.raw_scores_sha256,
        }
        rows.append(row)
        print(json.dumps({"ordinal": ordinal, **row}), flush=True)

    metric_keys = ("raw_ssim", "nlm_ssim", "r5_ssim", "r5_then_nlm_ssim", "nlm_then_r5_ssim")
    summary: dict[str, Any] = {key: float(np.mean([row[key] for row in rows])) for key in metric_keys}
    primary_variants = ("r5_ssim", "r5_then_nlm_ssim", "nlm_then_r5_ssim")
    primary = {key: paired_summary(rows, key, "nlm_ssim") for key in primary_variants}
    raw_deltas = {key: paired_summary(rows, key, "raw_ssim") for key in metric_keys if key != "raw_ssim"}
    strict_winners = [key for key in primary_variants if primary[key]["mean"] > 0 and primary[key]["lower_95"] > 0]
    champion = max(strict_winners, key=lambda key: primary[key]["mean"]) if strict_winners else "nlm_ssim"
    report = {
        "experiment": "R5_NLM_composition_on_shared_frozen_rank96_layout",
        "scope": "source-disjoint held-out composition gate; exactly one input-only frozen rank96 layout per board; all variants pixel-only; target used only for post-hoc SSIM; no test access or submission writing",
        "split": str(args.split),
        "split_sha256": file_sha256(args.split),
        "partition": args.partition,
        "names": names,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "r5_checkpoint": {"path": str(args.r5_checkpoint), "sha256": file_sha256(args.r5_checkpoint)},
        "rank96_checkpoints": {key: {"path": str(value), "sha256": file_sha256(value)} for key, value in rank96_paths.items()},
        "opencv": {"version": cv2.__version__, "operator": "fastNlMeansDenoisingColored", "input_conversion": "RGB->BGR->RGB"},
        "rows": rows,
        "summary_mean_ssim": summary,
        "raw_delta": raw_deltas,
        "paired_vs_canonical_nlm": primary,
        "gate": {
            "condition": "retain an R5-containing variant only when paired mean and lower-95 SSIM versus canonical NLM are both >0 on unchanged shared layouts",
            "strict_winners": strict_winners,
            "champion": champion,
            "passed": bool(strict_winners),
            "decision": "advance_champion_to_submission_candidate" if strict_winners else "retain_canonical_nlm",
        },
    }
    destination = args.report or args.work / f"r5_nlm_composition_{args.partition}{args.n}.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary_mean_ssim": summary, "paired_vs_canonical_nlm": primary, "gate": report["gate"], "report": str(destination)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
