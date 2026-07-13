#!/usr/bin/env python3
"""Target-blind actual-input diagnostic for frozen dual-LambdaRank -> QAP.

Phase A reads only real train inputs, freezes layouts and rendered PNGs, and
hashes every artifact.  Phase B writes a durable access marker before it
constructs any target path, verifies Phase A byte identity, and only then
computes official RGB SSIM.  This is development evidence and can never be
safe for submission.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import lightgbm as lgb
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
for value in (REPO_ROOT / "src", SCRIPT_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from puzzle_assembly.compatibility import CompatibilityMatrices, rank_normalize
from puzzle_assembly.components import soft_cycle_component_solver
from puzzle_assembly.geometry import TILE_COUNT, validate_permutation
from puzzle_assembly.learned import load_embedding_checkpoint
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_assembly.qap import directional_qap
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8
from puzzle_denoise_v2.tiles import merge_tiles_numpy, split_tiles_numpy
from train_binary_edge_verifier import (
    build_scores,
    feature_names as legacy_feature_names,
    read_rgb,
)
from train_evaluate_dual_lambdarank import (
    ORIGIN_BITS,
    corrected_features,
    layout_edges,
    percentile_by_group,
    seven_origin_graph,
)


EXPECTED_SHA256 = {
    "outgoing": "e7c53b80c100c3705e465a45e70bcee0eee2f72d40c44a6992b479eb19be1963",
    "incoming": "267abde975348b194eb1b2eb9c61f57c524abcbc518e34c0cc1874bbcb1dd9c1",
    "retrieval_report": "ff9db95ea02aa8529415eae09ba7dead325f93d6f37f99bb476753bca6294ee3",
    "denoiser": "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734",
    "embedding": "c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787",
    "manifest": "de4fd2e596efa0d157d2d4480eed5fb84812d358138a1db53c1706bfb580e345",
    "quarantine": "38dfd12f60579d77999c0cdb4a648fb4ff0343a8fb5e4c421f0b29f8b7bd6215",
    "dual_producer": "bab1185a05948a4a2619e9d94cc01ac579f347b87bb51f41fb7d0afef529f50f",
    "feature_producer": "ef3686ebc015d6647ddcc8878d3ac4b9cafb558ab4408667206da282bdaebab9",
}
EXPECTED_FEATURE_COUNT = 28
EXPECTED_SEED = 20260713
SPLIT = "edge_development"
SOURCE_OFFSET = 308
SOURCE_COUNT = 8
EXPECTED_SOURCE_NAMES_SHA256 = "5a24743a542ff854588d94daceed29c520990f4ca47826bc3978545bf3931bdc"
DUAL_MISSING_NEUTRAL_COST = 0.5
DUAL_FUSION_WEIGHT = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--denoiser", required=True)
    parser.add_argument("--embedding-checkpoint", required=True)
    parser.add_argument("--outgoing-model", required=True)
    parser.add_argument("--incoming-model", required=True)
    parser.add_argument("--retrieval-report", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--quarantine", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=EXPECTED_SEED)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def png_bytes(values: np.ndarray) -> bytes:
    if values.shape != (480, 480, 3) or values.dtype != np.uint8:
        raise RuntimeError("render must be uint8 RGB 480x480")
    buffer = BytesIO()
    Image.fromarray(values, mode="RGB").save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()


def validate_frozen_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed != EXPECTED_SEED:
        raise RuntimeError("seed drift")
    paths = {
        "outgoing": Path(args.outgoing_model),
        "incoming": Path(args.incoming_model),
        "retrieval_report": Path(args.retrieval_report),
        "denoiser": Path(args.denoiser),
        "embedding": Path(args.embedding_checkpoint),
        "manifest": Path(args.manifest),
        "quarantine": Path(args.quarantine),
        "dual_producer": SCRIPT_ROOT / "train_evaluate_dual_lambdarank.py",
        "feature_producer": SCRIPT_ROOT / "train_binary_edge_verifier.py",
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"frozen artifact drift: {actual}")
    report = json.loads(paths["retrieval_report"].read_text(encoding="utf-8"))
    expected_features = legacy_feature_names() + [
        "origin_softcycle", "origin_qap_w4", "origin_qap_w1"
    ]
    if report.get("feature_names") != expected_features or len(expected_features) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError("frozen feature order drift")
    if report.get("gate", {}).get("open_external_assembly_gate") is not False:
        raise RuntimeError("original retrieval gate must remain closed")
    if report.get("safe_for_submission") is not False:
        raise RuntimeError("unexpected original promotion state")
    return {"sha256": actual, "feature_names": expected_features}


def run_qap(score: CompatibilityMatrices, *, initial: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    result = directional_qap(
        score,
        initial=validate_permutation(initial).copy(),
        iterations=25,
        restarts=2,
        seed=seed,
        boundary_weight=0.05,
        initial_weight=0.75,
        noisy_components=3,
        noise_scale=1.0,
        refine_swaps=8,
        refine_weak_cells=32,
    )
    return validate_permutation(result.position_to_slot), {
        "objective": float(result.objective), "restart": int(result.restart)
    }


def prepare_input_only_record(
    name: str,
    *,
    args: argparse.Namespace,
    restorer: Any,
    embedding: Any,
    device: Any,
) -> dict[str, Any]:
    input_path = Path(args.data_root) / "train/inputs" / name
    input_rgb = read_rgb(input_path)
    raw_tiles = split_tiles_numpy(input_rgb)
    denoised_tiles = restore_tiles_uint8(
        restorer, raw_tiles, device, batch_size=args.denoise_batch_size
    )
    scores = build_scores(raw_tiles, denoised_tiles, embedding, device=device)
    soft = soft_cycle_component_solver(
        scores["hbt"], top_k=8, keep_per_tile=1,
        proposal_keep_fraction=0.5, loop_weight=1.0, reciprocal_weight=0.35,
    )
    initial = validate_permutation(soft.position_to_slot)
    qap_seed = per_source_seed(args.seed, "dual-lambdarank-qap-actual", name, 0)
    layouts = {"softcycle": initial.copy()}
    for score_name in ("w4", "w1"):
        layout, _ = run_qap(scores[score_name], initial=initial, seed=qap_seed)
        layouts[f"qap_{score_name}"] = layout
    graph = seven_origin_graph(scores, layouts)
    _, features = corrected_features(scores, graph)
    return {
        "name": name,
        "input_rgb": input_rgb,
        "raw_tiles": raw_tiles,
        "denoised_tiles": denoised_tiles,
        "scores": scores,
        "initial": initial,
        "baseline_layout": layouts["qap_w4"],
        "graph": graph,
        "features": features,
        "qap_seed": int(qap_seed),
        "input_file_sha256": sha256_file(input_path),
    }


def dual_rank_cost(
    record: dict[str, Any], outgoing: lgb.Booster, incoming: lgb.Booster
) -> CompatibilityMatrices:
    graph = record["graph"]
    features = np.asarray(record["features"], dtype=np.float32)
    if features.shape[1] != EXPECTED_FEATURE_COUNT:
        raise RuntimeError("dual feature width drift")
    if outgoing.num_feature() != EXPECTED_FEATURE_COUNT or incoming.num_feature() != EXPECTED_FEATURE_COUNT:
        raise RuntimeError("LightGBM feature width drift")
    out = percentile_by_group(
        np.asarray(outgoing.predict(features)), graph.direction, graph.source
    )
    inc = percentile_by_group(
        np.asarray(incoming.predict(features)), graph.direction, graph.destination
    )
    combined = 0.5 * (out + inc)
    matrices = []
    for direction in (0, 1):
        matrix = np.full(
            (TILE_COUNT, TILE_COUNT), DUAL_MISSING_NEUTRAL_COST, dtype=np.float32
        )
        indices = np.flatnonzero(graph.direction == direction)
        matrix[graph.source[indices], graph.destination[indices]] = 1.0 - combined[indices]
        np.fill_diagonal(matrix, np.inf)
        matrices.append(matrix)
    return CompatibilityMatrices("frozen_dual_rank_cost", matrices[0], matrices[1])


def fuse_with_neutral_dual(
    c1: CompatibilityMatrices,
    hbt: CompatibilityMatrices,
    dual: CompatibilityMatrices,
) -> CompatibilityMatrices:
    def one(first: np.ndarray, second: np.ndarray, learned: np.ndarray) -> np.ndarray:
        output = (
            rank_normalize(first)
            + 4.0 * rank_normalize(second)
            + DUAL_FUSION_WEIGHT * np.asarray(learned, dtype=np.float32)
        ) / (5.0 + DUAL_FUSION_WEIGHT)
        np.fill_diagonal(output, np.inf)
        return output.astype(np.float32)

    return CompatibilityMatrices(
        "C1_HBTw4_dualw1_neutral_missing",
        one(c1.right, hbt.right, dual.right),
        one(c1.down, hbt.down, dual.down),
    )


def assert_qap_w4_origin_matches_baseline(record: dict[str, Any]) -> None:
    graph = record["graph"]
    actual = {
        (int(graph.direction[index]), int(graph.source[index]), int(graph.destination[index]))
        for index in np.flatnonzero((graph.origin_mask & ORIGIN_BITS["qap_w4"]) != 0)
    }
    expected = set()
    for direction in (0, 1):
        expected.update(
            (direction, int(source), int(destination))
            for source, destination in layout_edges(record["baseline_layout"], direction).tolist()
        )
    if actual != expected:
        raise RuntimeError("qap_w4 graph origin differs from frozen baseline layout")


def phase_a(
    args: argparse.Namespace,
    *,
    names: list[str],
    frozen: dict[str, Any],
    phase_root: Path,
) -> tuple[Path, dict[str, Any]]:
    if phase_root.exists():
        raise RuntimeError("Phase-A directory already exists")
    phase_root.mkdir(parents=True)
    artifacts = phase_root / "artifacts"
    artifacts.mkdir()
    restorer, device, denoiser_metadata = load_restorer(
        args.denoiser, device=args.device, state="ema"
    )
    embedding, embedding_metadata = load_embedding_checkpoint(
        args.embedding_checkpoint, device=device
    )
    for model in (restorer, embedding):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    outgoing = lgb.Booster(model_file=str(args.outgoing_model))
    incoming = lgb.Booster(model_file=str(args.incoming_model))
    records = []
    for index, name in enumerate(names, 1):
        record = prepare_input_only_record(
            name, args=args, restorer=restorer, embedding=embedding, device=device
        )
        assert_qap_w4_origin_matches_baseline(record)
        dual = dual_rank_cost(record, outgoing, incoming)
        candidate_score = fuse_with_neutral_dual(
            record["scores"]["c1"], record["scores"]["hbt"], dual
        )
        candidate_layout, candidate_solver = run_qap(
            candidate_score, initial=record["initial"], seed=record["qap_seed"]
        )
        baseline_layout = validate_permutation(record["baseline_layout"])
        baseline_png = png_bytes(
            merge_tiles_numpy(record["denoised_tiles"][baseline_layout])
        )
        candidate_png = png_bytes(
            merge_tiles_numpy(record["denoised_tiles"][candidate_layout])
        )
        paths = {
            "baseline_layout": artifacts / f"{name}.baseline.npy",
            "candidate_layout": artifacts / f"{name}.candidate.npy",
            "baseline_png": artifacts / f"{name}.baseline.png",
            "candidate_png": artifacts / f"{name}.candidate.png",
        }
        np.save(paths["baseline_layout"], baseline_layout.astype(np.int32))
        np.save(paths["candidate_layout"], candidate_layout.astype(np.int32))
        paths["baseline_png"].write_bytes(baseline_png)
        paths["candidate_png"].write_bytes(candidate_png)
        records.append({
            "name": name,
            "input_file_sha256": record["input_file_sha256"],
            "input_pixel_sha256": array_sha256(record["input_rgb"]),
            "raw_tiles_sha256": array_sha256(record["raw_tiles"]),
            "denoised_tiles_sha256": array_sha256(record["denoised_tiles"]),
            "candidate_graph_sha256": hashlib.sha256(
                b"".join(
                    np.ascontiguousarray(getattr(record["graph"], field)).tobytes()
                    for field in ("direction", "source", "destination", "origin_mask")
                )
            ).hexdigest(),
            "features_sha256": array_sha256(record["features"]),
            "qap_seed": record["qap_seed"],
            "baseline_layout_value_sha256": array_sha256(baseline_layout.astype(np.int32)),
            "candidate_layout_value_sha256": array_sha256(candidate_layout.astype(np.int32)),
            "candidate_solver": candidate_solver,
            "artifacts": {
                key: {"path": str(path.relative_to(phase_root)), "sha256": sha256_file(path)}
                for key, path in paths.items()
            },
        })
        print(json.dumps({"phase": "A", "done": index, "total": len(names)}), flush=True)
    manifest = {
        "schema_version": 1,
        "kind": "dual_lambdarank_qap_input_only_phase_a",
        "target_paths_constructed": False,
        "target_files_opened": False,
        "split": f"{SPLIT}[{SOURCE_OFFSET}:{SOURCE_OFFSET + SOURCE_COUNT}]",
        "source_names": names,
        "frozen_artifacts": frozen,
        "mapping": {
            "dual_score": "mean outgoing/incoming within-group percentiles",
            "candidate_cost": "1-dual_score",
            "missing_cost": DUAL_MISSING_NEUTRAL_COST,
            "fusion": "(rank(C1)+4*rank(HBT)+dual_cost)/6",
            "weight_sweep": False,
        },
        "qap": {
            "same_softcycle_initial": True,
            "same_seed": True,
            "same_budget": True,
            "iterations": 25,
            "restarts": 2,
            "boundary_weight": 0.05,
        },
        "records": records,
        "model_metadata": {
            "denoiser": denoiser_metadata, "embedding": embedding_metadata
        },
    }
    manifest_path = phase_root / "PHASE_A_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path, manifest


def official_ssim(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        structural_similarity(target, prediction, channel_axis=2, data_range=255)
    )


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    delta = np.asarray(
        [record["candidate_ssim"] - record["baseline_ssim"] for record in records],
        dtype=np.float64,
    )
    return {
        "source_count": len(records),
        "mean_baseline_ssim": float(np.mean([r["baseline_ssim"] for r in records])),
        "mean_candidate_ssim": float(np.mean([r["candidate_ssim"] for r in records])),
        "mean_ssim_delta": float(delta.mean()),
        "wins_ties_losses": [
            int(np.count_nonzero(delta > 0)),
            int(np.count_nonzero(delta == 0)),
            int(np.count_nonzero(delta < 0)),
        ],
        "worst_ssim_delta": float(delta.min()),
    }


def gate(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "source_count_eq_8": summary["source_count"] == 8,
        "mean_ssim_delta_ge_0_003": summary["mean_ssim_delta"] >= 0.003,
        "wins_ge_6": summary["wins_ties_losses"][0] >= 6,
        "no_ssim_delta_below_minus_0_01": summary["worst_ssim_delta"] >= -0.01,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "pass_behavior": "development signal only; design one fresh confirmation",
        "fail_behavior": "retire frozen dual-LambdaRank-to-QAP transfer",
        "safe_for_submission": False,
    }


def phase_b(
    args: argparse.Namespace,
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    manifest_sha256 = sha256_file(manifest_path)
    marker = manifest_path.parent / "TARGET_ACCESS_MARKER.json"
    if marker.exists():
        raise RuntimeError("target access marker already exists")
    marker_payload = {
        "kind": "dual_lambdarank_qap_target_access_event",
        "phase_a_manifest_sha256": manifest_sha256,
        "target_access_started": True,
        "marker_precedes_first_target_path_construction": True,
    }
    marker.write_text(json.dumps(marker_payload, indent=2, sort_keys=True) + "\n")
    marker_sha256 = sha256_file(marker)
    records = []
    phase_root = manifest_path.parent
    for record in manifest["records"]:
        decoded = {}
        for key, descriptor in record["artifacts"].items():
            path = phase_root / descriptor["path"]
            if sha256_file(path) != descriptor["sha256"]:
                raise RuntimeError("Phase-A artifact changed before target scoring")
            if key.endswith("_png"):
                with Image.open(path) as image:
                    decoded[key] = np.asarray(image.convert("RGB"), dtype=np.uint8)
        target_path = Path(args.data_root) / "train/targets" / record["name"]
        target = read_rgb(target_path)
        records.append({
            "name": record["name"],
            "baseline_ssim": official_ssim(target, decoded["baseline_png"]),
            "candidate_ssim": official_ssim(target, decoded["candidate_png"]),
            "phase_a_record": record,
            "target_file_sha256": sha256_file(target_path),
        })
    if sha256_file(manifest_path) != manifest_sha256 or sha256_file(marker) != marker_sha256:
        raise RuntimeError("frozen envelope changed during target scoring")
    summary = summarize(records)
    gate_result = gate(summary)
    payload = {
        "schema_version": 1,
        "kind": "frozen_dual_lambdarank_qap_actual_input_development_diagnostic",
        "status": "pass_development_signal_only" if gate_result["passed"] else "stop_no_assembly_signal",
        "phase_a_manifest": str(manifest_path),
        "phase_a_manifest_sha256": manifest_sha256,
        "target_access_marker": str(marker),
        "target_access_marker_sha256": marker_sha256,
        "target_access_count": len(records),
        "summary": summary,
        "gate": gate_result,
        "records": records,
        "safe_for_submission": False,
        "sealed_paths_opened": False,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    phase_root = output.parent / f"{output.stem}_phase_a"
    frozen = validate_frozen_artifacts(args)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    names = source_names_for_split(
        SPLIT, manifest_path=args.manifest, quarantine_path=args.quarantine
    )[SOURCE_OFFSET : SOURCE_OFFSET + SOURCE_COUNT]
    names_sha256 = hashlib.sha256("\n".join(names).encode()).hexdigest()
    if len(names) != SOURCE_COUNT or names_sha256 != EXPECTED_SOURCE_NAMES_SHA256:
        raise RuntimeError("frozen development source slice drift")
    started = time.time()
    manifest_path, manifest = phase_a(
        args, names=names, frozen=frozen, phase_root=phase_root
    )
    payload = phase_b(
        args, manifest_path=manifest_path, manifest=manifest, output=output
    )
    payload["seconds"] = time.time() - started
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "summary": payload["summary"], "gate": payload["gate"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
