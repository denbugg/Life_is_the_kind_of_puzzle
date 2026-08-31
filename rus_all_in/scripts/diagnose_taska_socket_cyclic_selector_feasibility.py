#!/usr/bin/env python3
"""Opened-local feasibility for a conservative Socket cyclic selector.

No new panel or reference is opened.  The script re-derives inference-visible
features on the already frozen local32 cases and joins them to the labels that
already exist in the step-147 report.  Its source-LOO result is diagnostic,
not a model or threshold selection run.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.socket_sorter_production import (
    choose_deterministic_device,
    load_socket_checkpoint,
)
from aiijc_puzzle.taska_layout_portfolio import total_taska_adjacent_seam_cost
from aiijc_puzzle.taska_pair_pipeline import PAIR_DENOMINATOR
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES

try:
    from scripts import run_taska_focal_current_finetune as finetune
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import run_taska_focal_current_finetune as finetune


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSFER_ROOT = PROJECT_ROOT / "outputs/taska-socket-cyclic-origin-transfer/local32-v1"
TRANSFER_REPORT = TRANSFER_ROOT / "report.json"
TRANSFER_ARCHIVE = TRANSFER_ROOT / "frozen-target-free-eval.npz"
TRANSFER_METADATA = TRANSFER_ROOT / "frozen-target-free-eval.json"
ARM_ROOT = PROJECT_ROOT / "outputs/taska-six-arm-learned-selector/fixed-v1/local32"
ARM_ARCHIVE = ARM_ROOT / "frozen-target-free-arms.npz"
STACKER_ROOT = PROJECT_ROOT / "outputs/taska-focal-feature-stacker/train96-v1/local32"
STACKER_ARCHIVE = STACKER_ROOT / "frozen-target-free-eval.npz"
STACKER_METADATA = STACKER_ROOT / "frozen-target-free-eval.json"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/socket_matcher.pt"
)
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/taska-socket-cyclic-origin-transfer/local32-selector-feasibility-v1"
)
GRID = 24
COUNT = GRID * GRID
CONTROL = "confirmed_six_arm_fusion"
CANDIDATE = "socket_cyclic_border5_origin"
PAIR_FLOOR = -2
MINIMUM_GAIN = 1e-9
MODEL_FEATURES = (
    "socket_total_gain",
    "socket_row_margin",
    "socket_column_margin",
    "raw_taska_cost_delta_per_pair",
    "toroidal_roll_l1",
    "nonzero_axis_count",
    "six_arm_full_roll_support_fraction",
    "six_arm_positive_gain_support_fraction",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        rendered = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": sha256_file(resolved)}


def _json_rows(path: Path) -> list[Mapping[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8")).get("rows")
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError(f"expected 32 rows in {path}")
    return rows


def _strict(value: Any) -> np.ndarray:
    layout = np.asarray(value, dtype=np.int32)
    if layout.shape != (COUNT,) or not np.array_equal(
        np.sort(layout), np.arange(COUNT, dtype=np.int32)
    ):
        raise ValueError("expected strict 576-tile layout")
    return np.ascontiguousarray(layout)


def _conditional_dustbin_probability(values: np.ndarray) -> np.ndarray:
    probability = np.exp(
        np.clip(values + math.log(float(COUNT + GRID)), -60.0, 0.0)
    )
    return np.clip(probability, 1e-6, 1.0 - 1e-6)


def _horizontal_score(
    layout: np.ndarray, right: np.ndarray, *, border_weight: float = 5.0
) -> float:
    board = layout.reshape(GRID, GRID)
    real = float(right[board[:, :-1], board[:, 1:]].sum(dtype=np.float64))
    left_p = _conditional_dustbin_probability(right[COUNT, :COUNT])
    right_p = _conditional_dustbin_probability(right[:COUNT, COUNT])
    columns = np.tile(np.arange(GRID), GRID)
    flat = board.reshape(-1)
    border = np.where(columns == 0, np.log(left_p[flat]), np.log1p(-left_p[flat]))
    border += np.where(
        columns == GRID - 1,
        np.log(right_p[flat]),
        np.log1p(-right_p[flat]),
    )
    return real + border_weight * float(border.sum(dtype=np.float64))


def _vertical_score(
    layout: np.ndarray, down: np.ndarray, *, border_weight: float = 5.0
) -> float:
    board = layout.reshape(GRID, GRID)
    real = float(down[board[:-1], board[1:]].sum(dtype=np.float64))
    top_p = _conditional_dustbin_probability(down[COUNT, :COUNT])
    bottom_p = _conditional_dustbin_probability(down[:COUNT, COUNT])
    rows = np.repeat(np.arange(GRID), GRID)
    flat = board.reshape(-1)
    border = np.where(rows == 0, np.log(top_p[flat]), np.log1p(-top_p[flat]))
    border += np.where(
        rows == GRID - 1,
        np.log(bottom_p[flat]),
        np.log1p(-bottom_p[flat]),
    )
    return real + border_weight * float(border.sum(dtype=np.float64))


def _best_index(scores: np.ndarray) -> int:
    best = 0
    for index in range(1, len(scores)):
        if scores[index] > scores[best] + MINIMUM_GAIN:
            best = index
    return best


def _axis_features(
    layout: np.ndarray, right: np.ndarray, down: np.ndarray
) -> dict[str, Any]:
    board = layout.reshape(GRID, GRID)
    column_scores = np.asarray(
        [
            _horizontal_score(
                np.roll(board, shift=(0, roll), axis=(0, 1)).reshape(-1), right
            )
            for roll in range(GRID)
        ],
        dtype=np.float64,
    )
    row_scores = np.asarray(
        [
            _vertical_score(
                np.roll(board, shift=(roll, 0), axis=(0, 1)).reshape(-1), down
            )
            for roll in range(GRID)
        ],
        dtype=np.float64,
    )
    row_roll = _best_index(row_scores)
    column_roll = _best_index(column_scores)

    def margin(scores: np.ndarray, selected: int) -> float:
        competitors = np.delete(scores, selected)
        return float(scores[selected] - competitors.max())

    row_gain = float(row_scores[row_roll] - row_scores[0])
    column_gain = float(column_scores[column_roll] - column_scores[0])
    return {
        "row_roll": row_roll,
        "column_roll": column_roll,
        "row_gain": row_gain,
        "column_gain": column_gain,
        "total_gain": row_gain + column_gain,
        "row_margin": margin(row_scores, row_roll),
        "column_margin": margin(column_scores, column_roll),
        "row_scores": row_scores,
        "column_scores": column_scores,
    }


def _raw_axis_costs(
    layout: np.ndarray, right: np.ndarray, down: np.ndarray
) -> tuple[float, float]:
    board = layout.reshape(GRID, GRID)
    horizontal = float(right[board[:, :-1], board[:, 1:]].sum(dtype=np.float64))
    vertical = float(down[board[:-1], board[1:]].sum(dtype=np.float64))
    return horizontal, vertical


def _feature_vector(
    control: np.ndarray,
    candidate: np.ndarray,
    arms: Mapping[str, np.ndarray],
    right: np.ndarray,
    down: np.ndarray,
    raw_right: np.ndarray,
    raw_down: np.ndarray,
) -> tuple[dict[str, float], dict[str, Any]]:
    control_axis = _axis_features(control, right, down)
    row_roll = int(control_axis["row_roll"])
    column_roll = int(control_axis["column_roll"])
    expected = np.roll(
        control.reshape(GRID, GRID),
        shift=(row_roll, column_roll),
        axis=(0, 1),
    ).reshape(-1)
    if not np.array_equal(candidate, expected):
        raise RuntimeError("axis decomposition disagrees with frozen step-147 roll")

    arm_axes = {name: _axis_features(layout, right, down) for name, layout in arms.items()}
    full_support = sum(
        detail["row_roll"] == row_roll and detail["column_roll"] == column_roll
        for detail in arm_axes.values()
    )
    row_support = sum(detail["row_roll"] == row_roll for detail in arm_axes.values())
    column_support = sum(
        detail["column_roll"] == column_roll for detail in arm_axes.values()
    )
    gains_for_control_roll = [
        float(
            detail["row_scores"][row_roll]
            + detail["column_scores"][column_roll]
            - detail["row_scores"][0]
            - detail["column_scores"][0]
        )
        for detail in arm_axes.values()
    ]
    positive_support = sum(gain > MINIMUM_GAIN for gain in gains_for_control_roll)

    raw_control_h, raw_control_v = _raw_axis_costs(
        control, raw_right, raw_down
    )
    raw_candidate_h, raw_candidate_v = _raw_axis_costs(
        candidate, raw_right, raw_down
    )
    raw_delta = total_taska_adjacent_seam_cost(
        candidate, raw_right, raw_down, grid=GRID
    ) - total_taska_adjacent_seam_cost(control, raw_right, raw_down, grid=GRID)
    if not np.isclose(
        raw_delta,
        (raw_candidate_h - raw_control_h) + (raw_candidate_v - raw_control_v),
    ):
        raise RuntimeError("raw TASKA axis cost decomposition failed")

    row_distance = min(row_roll, GRID - row_roll)
    column_distance = min(column_roll, GRID - column_roll)
    features = {
        "socket_total_gain": float(control_axis["total_gain"]),
        "socket_row_gain": float(control_axis["row_gain"]),
        "socket_column_gain": float(control_axis["column_gain"]),
        "socket_row_margin": float(control_axis["row_margin"]),
        "socket_column_margin": float(control_axis["column_margin"]),
        "raw_taska_cost_delta_per_pair": float(raw_delta / PAIR_DENOMINATOR),
        "raw_taska_horizontal_cost_delta_per_pair": float(
            (raw_candidate_h - raw_control_h) / (GRID * (GRID - 1))
        ),
        "raw_taska_vertical_cost_delta_per_pair": float(
            (raw_candidate_v - raw_control_v) / (GRID * (GRID - 1))
        ),
        "toroidal_roll_l1": float(row_distance + column_distance),
        "nonzero_axis_count": float((row_roll != 0) + (column_roll != 0)),
        "six_arm_full_roll_support_fraction": float(
            full_support / len(FUSION_ARM_NAMES)
        ),
        "six_arm_row_roll_support_fraction": float(
            row_support / len(FUSION_ARM_NAMES)
        ),
        "six_arm_column_roll_support_fraction": float(
            column_support / len(FUSION_ARM_NAMES)
        ),
        "six_arm_positive_gain_support_fraction": float(
            positive_support / len(FUSION_ARM_NAMES)
        ),
        "six_arm_mean_gain_for_control_roll": float(np.mean(gains_for_control_roll)),
    }
    diagnostics = {
        "row_roll": row_roll,
        "column_roll": column_roll,
        "six_arm_full_roll_support": full_support,
        "six_arm_row_roll_support": row_support,
        "six_arm_column_roll_support": column_support,
        "six_arm_positive_gain_support": positive_support,
        "six_arm_gains_for_control_roll": gains_for_control_roll,
    }
    return features, diagnostics


def _choice_summary(rows: Sequence[Mapping[str, Any]], chosen: np.ndarray) -> dict[str, Any]:
    exact = np.asarray([row["exact_delta"] for row in rows], dtype=np.float64)
    pairs = np.asarray([row["pair_delta"] for row in rows], dtype=np.float64)
    selected = np.asarray(chosen, dtype=bool)
    return {
        "selected_board_count": int(selected.sum()),
        "mean_exact_delta_per_all_boards": float(exact[selected].sum() / len(rows)),
        "mean_pair_delta_per_all_boards": float(pairs[selected].sum() / len(rows)),
        "selected_exact_delta_sum": float(exact[selected].sum()),
        "selected_pair_delta_sum": float(pairs[selected].sum()),
    }


def _oracle(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    exact_only = np.asarray([row["exact_delta"] > 0 for row in rows])
    safe_exact = np.asarray([row["safe_exact_positive"] for row in rows])
    return {
        "exact_positive_oracle": _choice_summary(rows, exact_only),
        "hard_pair_safe_exact_oracle": _choice_summary(rows, safe_exact),
        "positive_counts": {
            "exact_positive": int(exact_only.sum()),
            "hard_pair_safe_exact": int(safe_exact.sum()),
        },
    }


def _source_loo(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    changed = [row for row in rows if row["changed"]]
    features = np.asarray(
        [[row["features"][name] for name in MODEL_FEATURES] for row in changed],
        dtype=np.float64,
    )
    labels = np.asarray([row["safe_exact_positive"] for row in changed], dtype=np.int32)
    groups = np.asarray([row["source_filename"] for row in changed])
    if len(np.unique(groups)) != len(groups):
        raise RuntimeError("opened local selector diagnostic expected one draw per source")
    probabilities = np.empty(len(changed), dtype=np.float64)
    splitter = LeaveOneGroupOut()
    for train, validation in splitter.split(features, labels, groups):
        if len(np.unique(labels[train])) != 2:
            raise RuntimeError("source-LOO fold lost one hard-utility class")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                solver="liblinear",
                random_state=0,
                max_iter=1000,
            ),
        )
        model.fit(features[train], labels[train])
        probabilities[validation] = model.predict_proba(features[validation])[:, 1]
    chosen_changed = probabilities >= 0.5
    changed_choice = _choice_summary(changed, chosen_changed)
    chosen_all = np.zeros(len(rows), dtype=bool)
    changed_index = 0
    for index, row in enumerate(rows):
        if row["changed"]:
            chosen_all[index] = chosen_changed[changed_index]
            changed_index += 1
    selected_labels = labels[chosen_changed]
    precision = float(selected_labels.mean()) if len(selected_labels) else 0.0
    recall = float(selected_labels.sum() / labels.sum())
    univariate = {}
    for column, name in enumerate(MODEL_FEATURES):
        auc = float(roc_auc_score(labels, features[:, column]))
        univariate[name] = {
            "signed_auc": auc,
            "best_orientation_auc": max(auc, 1.0 - auc),
        }
    return {
        "protocol": {
            "scope": "17 changed local32 boards only; unchanged rolls are no-ops",
            "splitter": "LeaveOneGroupOut(source_filename)",
            "model": (
                "StandardScaler -> LogisticRegression(C=1, "
                "class_weight=balanced, liblinear, random_state=0)"
            ),
            "fixed_probability_threshold": 0.5,
            "model_features": list(MODEL_FEATURES),
            "exploratory_only": True,
        },
        "changed_board_count": len(changed),
        "positive_count": int(labels.sum()),
        "negative_count": int((labels == 0).sum()),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "selected_positive_precision": precision,
        "positive_recall": recall,
        "selected_on_changed_denominator": changed_choice,
        "selected_on_all32_denominator": _choice_summary(rows, chosen_all),
        "univariate_auc": univariate,
        "rows": [
            {
                "prefix": row["prefix"],
                "source_filename": row["source_filename"],
                "label": int(label),
                "probability": float(probability),
                "selected": bool(selected),
            }
            for row, label, probability, selected in zip(
                changed, labels, probabilities, chosen_changed, strict=True
            )
        ],
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    started = perf_counter()
    transfer = json.loads(TRANSFER_REPORT.read_text(encoding="utf-8"))
    scored_by_prefix = {
        str(row["prefix"]): row for row in transfer["local32"]["rows"]
    }
    transfer_rows = _json_rows(TRANSFER_METADATA)
    stacker_rows = _json_rows(STACKER_METADATA)
    for first, second in zip(transfer_rows, stacker_rows, strict=True):
        identity = ("prefix", "source_filename", "draw_index", "dirty_sha256")
        if any(first.get(field) != second.get(field) for field in identity):
            raise RuntimeError("transfer and stacker local32 rows do not align")

    device = choose_deterministic_device(args.device)
    checkpoint = load_socket_checkpoint(args.checkpoint.resolve(), device=device)
    source_config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(source_config)
    cache = finetune.CleanTileCache(args.targets.resolve(), maximum_boards=2)
    output_rows: list[dict[str, Any]] = []

    with (
        np.load(TRANSFER_ARCHIVE, allow_pickle=False) as transferred,
        np.load(ARM_ARCHIVE, allow_pickle=False) as arm_archive,
        np.load(STACKER_ARCHIVE, allow_pickle=False) as stacker,
    ):
        for index, row in enumerate(transfer_rows, start=1):
            prefix = str(row["prefix"])
            source = str(row["source_filename"])
            draw = int(row["draw_index"])
            dirty = finetune._dirty_case(cache, lookup[source], source, draw)
            if finetune._dirty_sha256(dirty.dirty_tiles) != row["dirty_sha256"]:
                raise RuntimeError("diagnostic recreated different dirty bytes")
            tensor = torch.from_numpy(dirty.dirty_tiles.astype(np.float32)).permute(
                0, 3, 1, 2
            ) / 255.0
            socket = checkpoint.model(tensor.unsqueeze(0).to(device), grid=GRID)
            right = socket.right_log_assignment[0].float().cpu().numpy()
            down = socket.down_log_assignment[0].float().cpu().numpy()
            control = _strict(transferred[f"{prefix}__{CONTROL}_layout"])
            candidate = _strict(transferred[f"{prefix}__{CANDIDATE}_layout"])
            arms = {
                name: _strict(arm_archive[f"{prefix}__{name}_layout"])
                for name in FUSION_ARM_NAMES
            }
            features, diagnostics = _feature_vector(
                control,
                candidate,
                arms,
                right,
                down,
                np.asarray(stacker[f"{prefix}__cost_right"], dtype=np.float64),
                np.asarray(stacker[f"{prefix}__cost_down"], dtype=np.float64),
            )
            scored = scored_by_prefix[prefix]
            control_metrics = scored["metrics"][CONTROL]
            candidate_metrics = scored["metrics"][CANDIDATE]
            exact_delta = int(
                candidate_metrics["exact_tiles"] - control_metrics["exact_tiles"]
            )
            pair_delta = int(
                candidate_metrics["satisfied_adjacent_pairs"]
                - control_metrics["satisfied_adjacent_pairs"]
            )
            output_rows.append(
                {
                    "prefix": prefix,
                    "source_filename": source,
                    "checkpoint_lineage_overlap": bool(
                        row["checkpoint_lineage_overlap"]
                    ),
                    "changed": not np.array_equal(control, candidate),
                    "features": features,
                    "diagnostics": diagnostics,
                    "exact_delta": exact_delta,
                    "pair_delta": pair_delta,
                    "hard_pair_safe": pair_delta >= PAIR_FLOOR,
                    "safe_exact_positive": exact_delta > 0 and pair_delta >= PAIR_FLOOR,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "socket_selector_feature",
                        "case": index,
                        "changed": output_rows[-1]["changed"],
                        "safe_exact_positive": output_rows[-1][
                            "safe_exact_positive"
                        ],
                    }
                ),
                flush=True,
            )

    disjoint = [row for row in output_rows if not row["checkpoint_lineage_overlap"]]
    report = {
        "schema": "aiijc-taska-socket-cyclic-selector-opened-local-feasibility-v1",
        "status": "opened-local-diagnostic-only",
        "contains_target_assisted_labels": True,
        "new_exact_reference_or_panel_opened": False,
        "terminal_or_fresh_accessed": False,
        "competition_test_accessed": False,
        "hard_utility": {
            "positive": "candidate exact_delta > 0 and per-board pair_delta >= -2",
            "otherwise": "reject candidate",
            "pair_floor": PAIR_FLOOR,
        },
        "all32_oracle": _oracle(output_rows),
        "socket_lineage_disjoint26_oracle": _oracle(disjoint),
        "source_grouped_oof": _source_loo(output_rows),
        "rows": output_rows,
        "runtime_seconds": perf_counter() - started,
        "artifacts": {
            "step147_report": _record(TRANSFER_REPORT),
            "step147_archive": _record(TRANSFER_ARCHIVE),
            "step147_metadata": _record(TRANSFER_METADATA),
            "six_arm_archive": _record(ARM_ARCHIVE),
            "raw_taska_cost_archive": _record(STACKER_ARCHIVE),
            "raw_taska_cost_metadata": _record(STACKER_METADATA),
            "socket_checkpoint": _record(args.checkpoint),
            "script": _record(Path(__file__)),
        },
    }
    _write_json(args.output_dir / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "all32_oracle": report["all32_oracle"],
                "disjoint26_oracle": report[
                    "socket_lineage_disjoint26_oracle"
                ],
                "source_grouped_oof": report["source_grouped_oof"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
