#!/usr/bin/env python3
"""Leakage-audited input-complexity router over frozen QAP candidates.

Phase ``fit`` consumes only the exact-panel section of an upgrade report,
recreates the corresponding corrupted inputs, and fits a deliberately tiny
decision stump with leave-one-source-out validation.  It writes and hashes the
frozen rule before any calibration targets are inspected.  Phase ``score``
then applies that rule to raw calibration inputs and reads the already-frozen
candidate metrics only for a paired diagnostic.  A failed cross-validation
gate collapses to the best fixed exact candidate rather than routing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.protocol import per_source_seed
from puzzle_denoise_v2.tiles import split_tiles_numpy


FEATURE_NAMES = (
    "gradient_mean",
    "gradient_median",
    "gradient_q90",
    "tile_luma_mean_std",
    "tile_luma_std_mean",
    "tile_rgb_mean_std",
    "saturation_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fit", "score"), required=True)
    parser.add_argument("--upgrade-report", required=True)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--router", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--baseline-label", default="qap_w4_b0.05_i25")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise ValueError(f"unexpected image shape for {path}: {values.shape}")
    return values


def _features(tiles: np.ndarray) -> dict[str, float]:
    values = np.asarray(tiles, dtype=np.float32) / 255.0
    if values.shape != (576, 20, 20, 3):
        raise ValueError(f"tiles must be 576x20x20x3, got {values.shape}")
    horizontal = np.abs(values[:, :, 1:] - values[:, :, :-1]).mean(
        axis=(1, 2, 3)
    )
    vertical = np.abs(values[:, 1:] - values[:, :-1]).mean(axis=(1, 2, 3))
    gradient = 0.5 * (horizontal + vertical)
    luma = (
        0.2126 * values[..., 0]
        + 0.7152 * values[..., 1]
        + 0.0722 * values[..., 2]
    )
    tile_luma_mean = luma.mean(axis=(1, 2))
    tile_luma_std = luma.std(axis=(1, 2))
    tile_rgb_mean = values.mean(axis=(1, 2))
    maximum = values.max(axis=3)
    minimum = values.min(axis=3)
    saturation = np.zeros_like(maximum)
    np.divide(
        maximum - minimum,
        maximum,
        out=saturation,
        where=maximum > 1e-8,
    )
    result = {
        "gradient_mean": float(gradient.mean()),
        "gradient_median": float(np.median(gradient)),
        "gradient_q90": float(np.quantile(gradient, 0.9)),
        "tile_luma_mean_std": float(tile_luma_mean.std()),
        "tile_luma_std_mean": float(tile_luma_std.mean()),
        "tile_rgb_mean_std": float(tile_rgb_mean.std(axis=0).mean()),
        "saturation_mean": float(saturation.mean()),
    }
    if set(result) != set(FEATURE_NAMES) or not all(np.isfinite(list(result.values()))):
        raise RuntimeError("invalid complexity features")
    return result


def _score_table(records: list[dict[str, Any]], labels: list[str]) -> dict[tuple[str, str], dict[str, float]]:
    table: dict[tuple[str, str], dict[str, float]] = {}
    for record in records:
        label = str(record["label"])
        if label not in labels:
            continue
        key = (str(record["source"]), str(record["panel"]))
        table.setdefault(key, {})[label] = float(record["predicted_layout_ssim"])
    if not table or any(set(values) != set(labels) for values in table.values()):
        raise RuntimeError("exact candidate score table is incomplete")
    return table


def _best_fixed(keys: list[tuple[str, str]], table: dict[tuple[str, str], dict[str, float]], labels: list[str]) -> tuple[str, float]:
    candidates = [
        (float(np.mean([table[key][label] for key in keys])), label)
        for label in labels
    ]
    score, label = max(candidates, key=lambda item: (item[0], item[1]))
    return label, score


def _fit_stump(
    keys: list[tuple[str, str]],
    table: dict[tuple[str, str], dict[str, float]],
    features: dict[tuple[str, str], dict[str, float]],
    labels: list[str],
) -> dict[str, Any]:
    best: tuple[float, tuple[str, float, str, str]] | None = None
    for feature in FEATURE_NAMES:
        values = sorted({features[key][feature] for key in keys})
        if len(values) < 2:
            continue
        thresholds = [0.5 * (a + b) for a, b in zip(values[:-1], values[1:])]
        for threshold in thresholds:
            for low in labels:
                for high in labels:
                    predicted = [
                        low if features[key][feature] <= threshold else high
                        for key in keys
                    ]
                    score = float(
                        np.mean(
                            [table[key][label] for key, label in zip(keys, predicted)]
                        )
                    )
                    rule_key = (feature, float(threshold), low, high)
                    if best is None or (score, rule_key) > best:
                        best = (score, rule_key)
    if best is None:
        fixed, score = _best_fixed(keys, table, labels)
        return {"kind": "fixed", "label": fixed, "train_mean_ssim": score}
    score, (feature, threshold, low, high) = best
    return {
        "kind": "stump",
        "feature": feature,
        "threshold": threshold,
        "low_label": low,
        "high_label": high,
        "train_mean_ssim": score,
    }


def _predict(rule: dict[str, Any], feature: dict[str, float]) -> str:
    if rule["kind"] == "fixed":
        return str(rule["label"])
    return str(
        rule["low_label"]
        if feature[str(rule["feature"])] <= float(rule["threshold"])
        else rule["high_label"]
    )


def _bootstrap(values: np.ndarray, resamples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def fit(args: argparse.Namespace) -> None:
    report_path = Path(args.upgrade_report)
    router_path = Path(args.router)
    if router_path.exists() and not args.overwrite:
        raise SystemExit("router exists; pass --overwrite")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    exact = report["exact"]
    labels = [str(value) for value in exact["selection"]["selected_for_real"]]
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise RuntimeError("router requires at least two unique frozen candidates")
    table = _score_table(exact["records"], labels)
    data_root = Path(args.data_root)
    seed = int(report["args"]["seed"])
    feature_table: dict[tuple[str, str], dict[str, float]] = {}
    for source, panel in sorted(table):
        clean = _read_rgb(data_root / "train" / "targets" / source)
        panel_seed = per_source_seed(seed, f"upgrade-matrix-{panel}", source, 0)
        exact_panel = make_exact_panel(clean, panel=panel, seed=panel_seed)
        feature_table[(source, panel)] = _features(exact_panel.slot_tiles)

    sources = sorted({source for source, _ in table})
    fold_records: list[dict[str, Any]] = []
    routed_values: list[float] = []
    fixed_values: list[float] = []
    routed_panels: list[str] = []
    for held_out in sources:
        train_keys = [key for key in sorted(table) if key[0] != held_out]
        test_keys = [key for key in sorted(table) if key[0] == held_out]
        fixed_label, _ = _best_fixed(train_keys, table, labels)
        rule = _fit_stump(train_keys, table, feature_table, labels)
        for key in test_keys:
            routed_label = _predict(rule, feature_table[key])
            routed = table[key][routed_label]
            fixed = table[key][fixed_label]
            routed_values.append(routed)
            fixed_values.append(fixed)
            routed_panels.append(key[1])
            fold_records.append(
                {
                    "held_out_source": held_out,
                    "panel": key[1],
                    "rule": rule,
                    "fixed_label": fixed_label,
                    "routed_label": routed_label,
                    "routed_ssim": routed,
                    "fixed_ssim": fixed,
                    "delta": routed - fixed,
                }
            )
    delta = np.asarray(routed_values) - np.asarray(fixed_values)
    panel_delta = {
        panel: float(
            np.mean([value for value, name in zip(delta, routed_panels) if name == panel])
        )
        for panel in sorted(set(routed_panels))
    }
    checks = {
        "nested_loso_mean_delta_at_least_0_001": float(delta.mean()) >= 0.001,
        "nested_loso_wins_at_least_10_of_16": int(np.sum(delta > 0)) >= 10,
        "no_panel_regression": min(panel_delta.values()) >= 0.0,
    }
    routed = bool(all(checks.values()))
    all_keys = sorted(table)
    final_rule = _fit_stump(all_keys, table, feature_table, labels)
    fixed_label, fixed_exact = _best_fixed(all_keys, table, labels)
    if not routed:
        final_rule = {"kind": "fixed", "label": fixed_label, "train_mean_ssim": fixed_exact}
    payload = {
        "schema_version": 1,
        "kind": "input_complexity_qap_router",
        "status": "router_gate_passed" if routed else "router_gate_failed_fixed_fallback",
        "created_before_real_scoring": True,
        "upgrade_report": str(report_path),
        "upgrade_report_sha256": _sha256(report_path),
        "exact_scope": {
            "source_names": exact["source_names"],
            "panels": exact["panels"],
            "candidate_labels": labels,
            "feature_names": FEATURE_NAMES,
        },
        "nested_leave_one_source_out": {
            "records": fold_records,
            "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "wins": int(np.sum(delta > 0)),
            "ties": int(np.sum(delta == 0)),
            "losses": int(np.sum(delta < 0)),
            "mean_delta_by_panel": panel_delta,
            "checks": checks,
        },
        "fixed_exact_best": {"label": fixed_label, "mean_ssim": fixed_exact},
        "frozen_rule": final_rule,
        "frozen_rule_sha256": _canonical_sha256(final_rule),
        "feature_table": [
            {"source": key[0], "panel": key[1], **feature_table[key]}
            for key in sorted(feature_table)
        ],
        "anti_leakage": {
            "fit_reads_report_section": "exact only",
            "real16_metrics_used_for_fit": False,
            "features_are_input_only": True,
            "whole_source_loso": True,
        },
    }
    router_path.parent.mkdir(parents=True, exist_ok=True)
    router_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "router_fitted", "status": payload["status"], "router": str(router_path), "sha256": _sha256(router_path), "rule": final_rule}, sort_keys=True))


def score(args: argparse.Namespace) -> None:
    if not args.output:
        raise SystemExit("--output is required in score phase")
    report_path = Path(args.upgrade_report)
    router_path = Path(args.router)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite")
    router = json.loads(router_path.read_text(encoding="utf-8"))
    if router.get("created_before_real_scoring") is not True:
        raise RuntimeError("router is not marked frozen before real scoring")
    if router["upgrade_report_sha256"] != _sha256(report_path):
        raise RuntimeError("upgrade report hash differs from router fit input")
    rule = router["frozen_rule"]
    if router["frozen_rule_sha256"] != _canonical_sha256(rule):
        raise RuntimeError("frozen router rule hash mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = report["real16"]["records"]
    labels = list(router["exact_scope"]["candidate_labels"])
    metric_table: dict[str, dict[str, float]] = {}
    for record in records:
        if record["label"] in labels:
            metric_table.setdefault(record["source"], {})[record["label"]] = float(
                record["predicted_layout_ssim"]
            )
    sources = list(report["real16"]["source_names"])
    if any(set(metric_table.get(source, {})) != set(labels) for source in sources):
        raise RuntimeError("real frozen candidate metrics are incomplete")
    data_root = Path(args.data_root)
    selections: list[dict[str, Any]] = []
    for source in sources:
        raw = _read_rgb(data_root / "train" / "inputs" / source)
        feature = _features(split_tiles_numpy(raw))
        label = _predict(rule, feature)
        selections.append(
            {
                "source": source,
                "features": feature,
                "selected_label": label,
                "selected_ssim": metric_table[source][label],
                "baseline_ssim": metric_table[source][args.baseline_label],
            }
        )
    selected = np.asarray([record["selected_ssim"] for record in selections])
    baseline = np.asarray([record["baseline_ssim"] for record in selections])
    delta = selected - baseline
    fixed_label = str(router["fixed_exact_best"]["label"])
    fixed = np.asarray([metric_table[source][fixed_label] for source in sources])
    versus_fixed = selected - fixed
    result = {
        "schema_version": 1,
        "kind": "input_complexity_qap_router_real16_score",
        "status": (
            "real_gate_passed"
            if float(delta.mean()) >= 0.005
            and int(np.sum(delta > 0)) >= 10
            and _bootstrap(delta, args.bootstrap_resamples, args.seed)[0] > 0
            else "real_gate_failed"
        ),
        "router": str(router_path),
        "router_sha256": _sha256(router_path),
        "upgrade_report": str(report_path),
        "upgrade_report_sha256": _sha256(report_path),
        "frozen_rule": rule,
        "selections": selections,
        "paired_vs_promoted_baseline": {
            "mean_ssim": float(selected.mean()),
            "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "wins": int(np.sum(delta > 0)),
            "ties": int(np.sum(delta == 0)),
            "losses": int(np.sum(delta < 0)),
            "bootstrap_95_ci": _bootstrap(delta, args.bootstrap_resamples, args.seed),
        },
        "paired_vs_fixed_exact_best": {
            "fixed_label": fixed_label,
            "mean_delta": float(versus_fixed.mean()),
            "wins": int(np.sum(versus_fixed > 0)),
            "losses": int(np.sum(versus_fixed < 0)),
            "bootstrap_95_ci": _bootstrap(versus_fixed, args.bootstrap_resamples, args.seed + 1),
        },
        "anti_leakage": {
            "router_frozen_before_this_phase": True,
            "features_are_raw_input_only": True,
            "candidate_layouts_were_frozen_before_calibration_targets": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "router_scored", "status": result["status"], "output": str(output), "sha256": _sha256(output), "paired": result["paired_vs_promoted_baseline"]}, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.bootstrap_resamples <= 0:
        raise SystemExit("bootstrap-resamples must be positive")
    if args.phase == "fit":
        fit(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
