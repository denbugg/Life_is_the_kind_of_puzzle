#!/usr/bin/env python3
"""Isolate rank/origin signal on the full C1/HBT candidate union."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from puzzle_assembly.compatibility import CompatibilityMatrices
from puzzle_assembly.geometry import TILE_COUNT
from puzzle_assembly.learned import load_embedding_checkpoint
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_denoise_v2.inference import load_restorer
from train_binary_edge_verifier import (
    CandidateGraph,
    PreparedSource,
    binary_metrics,
    candidate_features,
    candidate_labels,
    component_metrics,
    feature_names,
    precision_frontier,
    prepare_source,
)


PANELS = ("primary_kornia", "independent_libjpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument(
        "--denoiser", default="runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
    )
    parser.add_argument(
        "--embedding-checkpoint",
        default="runs/assembly_v1/kaggle/edge2vec_gradient_gpu/hbt_d320_denoised_rgb_sobel.pt",
    )
    parser.add_argument("--manifest", default="configs/denoise_splits_seed20260710.json")
    parser.add_argument(
        "--quarantine", default="configs/denoise_validation_quarantine_v1.json"
    )
    parser.add_argument("--fit-sources", type=int, default=24)
    parser.add_argument("--calibration-sources", type=int, default=8)
    parser.add_argument("--negative-per-record", type=int, default=8000)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--denoise-batch-size", type=int, default=256)
    parser.add_argument(
        "--fixture-root",
        default="runs/assembly_v1/candidate_graph_oracle_fixtures_v4_6c0fe4e8524ce39d830d9a5bee118d8b",
    )
    parser.add_argument(
        "--graph-root",
        default="runs/assembly_v1/kaggle/candidate_graph_oracle_v4_phase_a_readback/candidate_graph_oracle_v4_phase_a/finalized",
    )
    parser.add_argument("--max-v4-records", type=int)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_training(
    prepared: PreparedSource,
    *,
    negative_per_record: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.flatnonzero(prepared.labels > 0.5)
    negative = np.flatnonzero(prepared.labels < 0.5)
    # Keep the tightest rank-consensus negatives plus a random tail so the
    # classifier cannot solve the task from one candidate-cutoff artifact.
    hard_count = min(len(negative), negative_per_record * 3 // 4)
    hard = negative[
        np.argsort(prepared.features[negative, 16], kind="stable")[:hard_count]
    ]
    remaining = np.setdiff1d(negative, hard, assume_unique=False)
    random_count = min(len(remaining), negative_per_record - len(hard))
    random_negative = (
        rng.choice(remaining, size=random_count, replace=False)
        if random_count
        else np.empty(0, dtype=np.int64)
    )
    selected = np.concatenate([positive, hard, random_negative])
    rng.shuffle(selected)
    return prepared.features[selected], prepared.labels[selected].astype(np.uint8)


def load_v4(
    fixture_root: Path, graph_root: Path, *, max_records: int | None = None
) -> list[tuple[dict, PreparedSource]]:
    manifest = json.loads(
        (fixture_root / "fixture_label/fixture_label_manifest.json").read_text()
    )
    output = []
    records = manifest["records"]
    if max_records is not None:
        records = records[:max_records]
    for meta in records:
        opaque_id = str(meta["opaque_id"])
        graph_path = graph_root / "artifacts" / f"{opaque_id}.graph.npz"
        label_path = fixture_root / "fixture_label/records" / f"{opaque_id}.npz"
        input_path = fixture_root / "fixture_input/records" / f"{opaque_id}.npz"
        with np.load(graph_path, allow_pickle=False) as graph_values, np.load(
            label_path, allow_pickle=False
        ) as label_values, np.load(input_path, allow_pickle=False) as input_values:
            graph = CandidateGraph(
                direction=np.asarray(graph_values["candidate_direction"]),
                source=np.asarray(graph_values["candidate_source"], dtype=np.int32),
                destination=np.asarray(
                    graph_values["candidate_destination"], dtype=np.int32
                ),
                origin_mask=np.asarray(graph_values["candidate_origin_mask"]),
            )
            scores = {
                name: CompatibilityMatrices(
                    name,
                    np.asarray(graph_values[f"{name}_right"]),
                    np.asarray(graph_values[f"{name}_down"]),
                )
                for name in ("c1", "hbt", "w1", "w4")
            }
            truth = np.asarray(label_values["composed_slot_to_target"])
            features = candidate_features(scores, graph)
            labels = candidate_labels(graph, truth)
            prepared = PreparedSource(
                name=str(meta["source_name"]),
                panel=str(meta["panel"]),
                seed=int(meta["panel_seed"]),
                raw_tiles=np.asarray(input_values["slot_tiles"]),
                denoised_tiles=np.asarray(graph_values["denoised_tiles"]),
                truth=truth,
                scores=scores,
                graph=graph,
                features=features,
                labels=labels,
            )
        output.append((meta, prepared))
    return output


def panel_summary(records: list[dict], panel: str) -> dict[str, float]:
    selected = [record for record in records if record["panel"] == panel]
    return {
        "records": len(selected),
        "average_precision": float(
            np.mean([record["binary"]["average_precision"] for record in selected])
        ),
        "roc_auc": float(
            np.mean([record["binary"]["roc_auc"] for record in selected])
        ),
        "accepted_precision": float(
            np.mean([record["components"]["accepted_precision"] for record in selected])
        ),
        "largest_component": float(
            np.mean([record["components"]["largest_component"] for record in selected])
        ),
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    report_path = output_root / "report.json"
    model_path = output_root / "full_union_tabular.joblib"
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise SystemExit("output root is not empty; pass --overwrite")
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rng = np.random.default_rng(args.seed)
    restorer, device, denoiser_metadata = load_restorer(
        args.denoiser, device=args.device
    )
    embedding, embedding_metadata = load_embedding_checkpoint(
        args.embedding_checkpoint, device=device
    )
    restorer.eval()
    embedding.eval()
    for frozen in (restorer, embedding):
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
    source_names = source_names_for_split(
        "edge_development",
        manifest_path=args.manifest,
        quarantine_path=args.quarantine,
    )
    fit_names = source_names[: args.fit_sources]
    calibration_names = source_names[
        args.fit_sources : args.fit_sources + args.calibration_sources
    ]
    if len(fit_names) != args.fit_sources or len(calibration_names) != args.calibration_sources:
        raise RuntimeError("requested source slices are unavailable")
    if set(fit_names) & set(calibration_names):
        raise RuntimeError("fit/calibration source overlap")

    fit_x, fit_y = [], []
    for source_index, name in enumerate(fit_names):
        for panel in PANELS:
            seed = per_source_seed(args.seed, f"full-union-tabular-{panel}", name, 0)
            prepared = prepare_source(
                name,
                panel,
                seed,
                args=args,
                restorer=restorer,
                embedding_model=embedding,
                device=device,
            )
            x, y = sample_training(
                prepared,
                negative_per_record=args.negative_per_record,
                rng=rng,
            )
            fit_x.append(x)
            fit_y.append(y)
        print(
            json.dumps(
                {"stage": "fit_features", "done": source_index + 1, "total": len(fit_names)}
            ),
            flush=True,
        )
    fit_x_array = np.concatenate(fit_x)
    fit_y_array = np.concatenate(fit_y)
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=args.max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=64,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=args.seed,
    )
    model.fit(fit_x_array, fit_y_array)

    calibration_records = []
    calibration_y, calibration_probability = [], []
    for source_index, name in enumerate(calibration_names):
        for panel in PANELS:
            seed = per_source_seed(args.seed, f"full-union-tabular-{panel}", name, 0)
            prepared = prepare_source(
                name,
                panel,
                seed,
                args=args,
                restorer=restorer,
                embedding_model=embedding,
                device=device,
            )
            probability = model.predict_proba(prepared.features)[:, 1]
            calibration_records.append(
                {
                    "name": name,
                    "panel": panel,
                    "binary": binary_metrics(prepared.labels, probability),
                }
            )
            calibration_y.append(prepared.labels)
            calibration_probability.append(probability)
        print(
            json.dumps(
                {
                    "stage": "calibration",
                    "done": source_index + 1,
                    "total": len(calibration_names),
                }
            ),
            flush=True,
        )
    cal_y = np.concatenate(calibration_y)
    cal_p = np.concatenate(calibration_probability)
    frontiers = {
        str(target): precision_frontier(cal_y, cal_p, target_precision=target)
        for target in (0.75, 0.80, 0.85, 0.90)
    }
    frozen = frontiers["0.85"]
    if not frozen["target_achieved"]:
        # Use the 0.80 frontier only if it actually exists; otherwise the branch
        # is already scientifically dead and v4 components will stay empty.
        frozen = frontiers["0.8"]
    threshold = float(frozen["threshold"])

    v4_records = []
    all_v4_y, all_v4_p = [], []
    v4_sources = set()
    for record_index, (meta, prepared) in enumerate(
        load_v4(
            Path(args.fixture_root),
            Path(args.graph_root),
            max_records=args.max_v4_records,
        ),
        1,
    ):
        if prepared.name in set(fit_names) | set(calibration_names):
            raise RuntimeError("development/v4 whole-source overlap")
        v4_sources.add(prepared.name)
        probability = model.predict_proba(prepared.features)[:, 1]
        components = component_metrics(prepared, probability, threshold)
        v4_records.append(
            {
                "name": prepared.name,
                "panel": prepared.panel,
                "binary": binary_metrics(prepared.labels, probability),
                "candidate_recall": float(prepared.labels.sum() / 1104.0),
                "components": components,
            }
        )
        all_v4_y.append(prepared.labels)
        all_v4_p.append(probability)
        print(
            json.dumps(
                {
                    "stage": "v4",
                    "done": record_index,
                    "total": args.max_v4_records or 64,
                }
            ),
            flush=True,
        )
    v4_y = np.concatenate(all_v4_y)
    v4_p = np.concatenate(all_v4_p)
    report = {
        "schema_version": 1,
        "kind": "full_union_tabular_edge_verifier",
        "args": vars(args),
        "feature_names": feature_names(),
        "split": {
            "fit_names": fit_names,
            "calibration_names": calibration_names,
            "v4_names": sorted(v4_sources),
            "pairwise_source_disjoint": True,
        },
        "fit": {
            "examples": len(fit_y_array),
            "positives": int(fit_y_array.sum()),
        },
        "calibration": {
            "binary": binary_metrics(cal_y, cal_p),
            "frontiers": frontiers,
            "frozen_threshold": threshold,
            "records": calibration_records,
        },
        "v4": {
            "binary": binary_metrics(v4_y, v4_p),
            "mean_accepted_precision": float(
                np.mean([r["components"]["accepted_precision"] for r in v4_records])
            ),
            "mean_largest_component": float(
                np.mean([r["components"]["largest_component"] for r in v4_records])
            ),
            "panels": {panel: panel_summary(v4_records, panel) for panel in PANELS},
            "records": v4_records,
        },
        "denoiser_metadata": denoiser_metadata,
        "embedding_metadata": embedding_metadata,
        "seconds": time.time() - started,
    }
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names(),
            "threshold": threshold,
            "fit_names": fit_names,
            "calibration_names": calibration_names,
        },
        model_path,
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "report_sha256": sha256(report_path),
                "model": str(model_path),
                "model_sha256": sha256(model_path),
                "calibration": report["calibration"]["binary"],
                "frozen_threshold": threshold,
                "v4": {
                    key: report["v4"][key]
                    for key in (
                        "binary",
                        "mean_accepted_precision",
                        "mean_largest_component",
                        "panels",
                    )
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
