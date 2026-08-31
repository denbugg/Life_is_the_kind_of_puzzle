#!/usr/bin/env python3
"""Fit local32 query confidence and gate it once on frozen confirm24.

This runner cannot decode layouts and has no competition-test path.  It fits a
68-parameter logistic calibrator only on the already-opened v1 local32, then
opens the preregistered source-disjoint confirm24 once.  Passing its discovery
gate authorizes a separate decoder40 run; it is not a promotion decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from aiijc_puzzle.component_anchor_diagnostic import rebuild_decoder_components
from aiijc_puzzle.component_relation_confidence import (
    FEATURE_NAMES,
    LogisticConfidenceCalibrator,
    QueryConfidenceFeatures,
    aggregate_confidence_observations,
    build_query_confidence_features,
    confidence_query_observations,
    fit_confidence_calibrator,
)
from aiijc_puzzle.component_relation_reranker import (
    ComponentRelationReranker,
    aggregate_relation_observations,
    build_component_relation_candidates,
    component_relation_targets,
    extract_frozen_socket_context,
    relation_query_observations,
)
from aiijc_puzzle.component_shift_head import component_descriptors_from_decoder
from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    compute_protocol_digest,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.socket_sorter_production import (
    LoadedSocketCheckpoint,
    choose_deterministic_device,
    load_socket_checkpoint,
)
from aiijc_puzzle.synthetic_socket_evaluation import make_exact_synthetic_case

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREGISTRATION = (
    PROJECT_ROOT / "configs/component_relation_confidence_preregistered_v1_1.json"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
GRID = 24
TILE_COUNT = GRID * GRID
EXPECTED_RELATION_PARAMETERS = 131_665
EXPECTED_CALIBRATOR_PARAMETERS = 68
CONFIDENCE_CAPS = (32, 144)


@dataclass(frozen=True)
class PreparedCase:
    case_id: str
    source_filename: str
    dirty_tiles: np.ndarray
    input_tile_to_position: np.ndarray


@dataclass(frozen=True)
class FrozenCaseOutput:
    logits: torch.Tensor
    candidates: tuple[Any, ...]
    labels: tuple[Any, ...]
    oracle_relations: frozenset[tuple[int, str, int, int, int]]
    profiles: tuple[Any, ...]
    components: tuple[Any, ...]
    socket_output: Any
    runtime_seconds: dict[str, float]


class CleanTileCache:
    """Small verified target cache used only for an explicitly opened roster."""

    def __init__(self, targets: Path, *, maximum_boards: int = 32) -> None:
        if maximum_boards <= 0:
            raise ValueError("maximum_boards must be positive")
        self.targets = targets
        self.maximum_boards = maximum_boards
        self.values: OrderedDict[str, np.ndarray] = OrderedDict()

    def load(self, record: Mapping[str, Any]) -> np.ndarray:
        filename = str(record["filename"])
        if filename in self.values:
            value = self.values.pop(filename)
            self.values[filename] = value
            return value
        path = self.targets / filename
        expected = record.get("target_sha256")
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise ValueError(f"manifest target hash mismatch: {filename}")
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(f"expected RGB 480x480 target: {path}")
            value = split_tiles(np.asarray(image, dtype=np.uint8)).copy()
        self.values[filename] = value
        while len(self.values) > self.maximum_boards:
            self.values.popitem(last=False)
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    return parser.parse_args()


def filename_digest(names: Sequence[str], *, sort_names: bool = False) -> str:
    values = sorted(names) if sort_names else list(names)
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def collect_filename_lists(value: Any, *, parent_key: str = "") -> set[str]:
    """Fail-closed recursive collection of arbitrary ``*_filenames`` lists."""

    names: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key.endswith("_filenames"):
                if not isinstance(child, (list, tuple)) or not all(
                    isinstance(item, str) and item for item in child
                ):
                    raise ValueError(f"{key} must be a list of non-empty filenames")
                if len(set(child)) != len(child):
                    raise ValueError(f"{key} contains duplicate filenames")
                names.update(Path(item).name for item in child)
            names.update(collect_filename_lists(child, parent_key=key))
    elif isinstance(value, (list, tuple)) and not parent_key.endswith("_filenames"):
        for child in value:
            names.update(collect_filename_lists(child, parent_key=parent_key))
    return names


def load_preregistration(path: Path) -> tuple[dict[str, Any], str]:
    digest_path = path.with_name(f"{path.name}.sha256")
    expected = digest_path.read_text(encoding="utf-8").split()[0]
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError("v1.1 preregistration hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("registered_before_confirm24_target_access"):
        raise ValueError("v1.1 was not preregistered before confirm access")
    return value, observed


def manifest_record_lookup(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if manifest.get("protocol_digest") != compute_protocol_digest(dict(manifest)):
        raise ValueError("validation manifest protocol digest is invalid")
    splits = manifest.get("splits")
    train = splits.get("train") if isinstance(splits, Mapping) else None
    if not isinstance(train, list):
        raise ValueError("validation manifest has no train split")
    lookup = {str(record["filename"]): record for record in train}
    if len(lookup) != len(train):
        raise ValueError("manifest train filenames must be unique")
    return lookup


def prepare_case(
    cache: CleanTileCache,
    record: Mapping[str, Any],
    *,
    draw_index: int,
    seed: int,
) -> PreparedCase:
    clean = cache.load(record)
    dirty, reference = make_exact_synthetic_case(
        clean,
        source_filename=str(record["filename"]),
        draw_index=draw_index,
        seed=seed,
    )
    tile_to_position = np.empty(TILE_COUNT, dtype=np.int32)
    tile_to_position[reference.tile_at_position] = np.arange(TILE_COUNT, dtype=np.int32)
    return PreparedCase(
        case_id=dirty.case_id,
        source_filename=dirty.source_filename,
        dirty_tiles=dirty.tiles,
        input_tile_to_position=tile_to_position,
    )


def _tile_tensor(tiles: np.ndarray, *, device: torch.device) -> torch.Tensor:
    value = np.asarray(tiles)
    if value.shape != (TILE_COUNT, 20, 20, 3) or value.dtype != np.uint8:
        raise ValueError("dirty tiles violate the exact original-tile contract")
    return (
        torch.from_numpy(np.ascontiguousarray(value))
        .permute(0, 3, 1, 2)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
        .unsqueeze(0)
    )


def frozen_case_forward(
    case: PreparedCase,
    *,
    socket: LoadedSocketCheckpoint,
    head: ComponentRelationReranker,
    device: torch.device,
    attach_exact_labels: bool = True,
) -> FrozenCaseOutput:
    runtime: dict[str, float] = {}
    started = perf_counter()
    with torch.no_grad():
        tokens, socket_output = extract_frozen_socket_context(
            socket.model,
            _tile_tensor(case.dirty_tiles, device=device),
            grid=GRID,
        )
    runtime["frozen_socket_d64"] = perf_counter() - started
    started = perf_counter()
    component_build = rebuild_decoder_components(
        socket_output.right_log_assignment,
        socket_output.down_log_assignment,
        grid=GRID,
        edge_budget_per_axis=144,
    )
    components = component_descriptors_from_decoder(component_build, grid=GRID)
    candidates = build_component_relation_candidates(
        components,
        socket_output,
        grid=GRID,
        proposal_topk=8,
        max_candidates_per_query=64,
    )
    runtime["components_and_target_blind_candidates"] = perf_counter() - started
    started = perf_counter()
    with torch.no_grad():
        logits = head(tokens[0], components, candidates)
    runtime["frozen_relation_head"] = perf_counter() - started
    labels: tuple[Any, ...] = ()
    oracle: frozenset[tuple[int, str, int, int, int]] = frozenset()
    profiles: tuple[Any, ...] = ()
    if attach_exact_labels:
        # Labels enter only after candidates, features, and logits are frozen.
        started = perf_counter()
        labels, oracle, profiles = component_relation_targets(
            candidates,
            components,
            case.input_tile_to_position,
            grid=GRID,
        )
        runtime["exact_labels_after_freeze"] = perf_counter() - started
    return FrozenCaseOutput(
        logits=logits,
        candidates=candidates,
        labels=labels,
        oracle_relations=oracle,
        profiles=profiles,
        components=components,
        socket_output=socket_output,
        runtime_seconds=runtime,
    )


def _ranking_digest(
    logits: torch.Tensor,
    candidates: tuple[Any, ...],
) -> str:
    learned = logits.detach().float().cpu().numpy()
    grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        grouped[candidate.query_key].append(index)
    canonical: list[str] = []
    for query in sorted(grouped):
        ordered = sorted(
            grouped[query],
            key=lambda index: (-float(learned[index]), candidates[index].relation_key),
        )
        canonical.append(
            repr((query, tuple(candidates[index].relation_key for index in ordered)))
        )
    return hashlib.sha256("\n".join(canonical).encode()).hexdigest()


def evaluate_confirm_gate(
    relation_metrics: Mapping[str, Any],
    confidence_metrics: Mapping[str, Any],
    *,
    ranking_unchanged: bool,
    gate_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Two-tier preregistered discovery gate; pass is decoder eligibility only."""

    learned = relation_metrics["learned"]
    raw_relation = relation_metrics["raw_socket_component_baseline"]
    calibrated = confidence_metrics["calibrated"]["high_confidence"]
    raw_confidence = confidence_metrics["raw_socket_component_baseline"][
        "high_confidence"
    ]
    r1_gain = float(learned["r1"]) - float(raw_relation["r1"])
    r5_gain = float(learned["r5"]) - float(raw_relation["r5"])
    top32_correct_gain = float(calibrated["top32"]["correct_per_board"]) - float(
        raw_confidence["top32"]["correct_per_board"]
    )
    top32_precision_gain = float(calibrated["top32"]["precision"]) - float(
        raw_confidence["top32"]["precision"]
    )
    top144_correct_gain = float(
        calibrated["top144"]["correct_per_board"]
    ) - float(raw_confidence["top144"]["correct_per_board"])
    either = gate_contract["top32_either"]
    checks = {
        "learned_pair_translation_r1_signal_retained": {
            "observed_gain": r1_gain,
            "required_gain": gate_contract[
                "minimum_learned_pair_translation_r1_gain_over_raw"
            ],
            "pass": r1_gain
            >= gate_contract["minimum_learned_pair_translation_r1_gain_over_raw"],
        },
        "learned_pair_translation_r5_signal_retained": {
            "observed_gain": r5_gain,
            "required_gain": gate_contract[
                "minimum_learned_pair_translation_r5_gain_over_raw"
            ],
            "pass": r5_gain
            >= gate_contract["minimum_learned_pair_translation_r5_gain_over_raw"],
        },
        "candidate_ranking_bitwise_unchanged": {
            "observed": ranking_unchanged,
            "required": True,
            "pass": ranking_unchanged,
        },
        "top32_correct_gain": {
            "observed_gain": top32_correct_gain,
            "required_gain": either[
                "minimum_correct_attachments_per_board_gain_over_raw"
            ],
            "pass": top32_correct_gain
            >= either["minimum_correct_attachments_per_board_gain_over_raw"],
        },
        "top32_matched_precision_gain": {
            "observed_gain": top32_precision_gain,
            "required_gain": either["minimum_matched_precision_gain_over_raw"],
            "pass": top32_precision_gain
            >= either["minimum_matched_precision_gain_over_raw"],
        },
        "top144_correct_non_regression": {
            "observed_gain": top144_correct_gain,
            "required_gain": gate_contract[
                "minimum_top144_correct_attachments_per_board_gain_over_raw"
            ],
            "pass": top144_correct_gain
            >= gate_contract[
                "minimum_top144_correct_attachments_per_board_gain_over_raw"
            ],
        },
    }
    top32_pass = bool(checks["top32_correct_gain"]["pass"]) or bool(
        checks["top32_matched_precision_gain"]["pass"]
    )
    required = (
        "learned_pair_translation_r1_signal_retained",
        "learned_pair_translation_r5_signal_retained",
        "candidate_ranking_bitwise_unchanged",
        "top144_correct_non_regression",
    )
    passed = top32_pass and all(bool(checks[name]["pass"]) for name in required)
    return {
        "status": "pass-decoder40-eligible" if passed else "stop",
        "pass": passed,
        "decoder40_authorized": passed,
        "promotion_authorized": False,
        "top32_either_pass": top32_pass,
        "checks": checks,
    }


def _mean_runtime(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {
        key: float(np.mean([float(row.get(key, 0.0)) for row in rows])) for key in keys
    }


def _calibration_fit_metrics(
    calibrator: LogisticConfidenceCalibrator,
    rows: Sequence[QueryConfidenceFeatures],
    labels: Sequence[bool],
) -> dict[str, Any]:
    target = np.asarray(labels, dtype=np.int64)
    probabilities = calibrator.predict_probabilities([row.values for row in rows])
    return {
        "rows": len(rows),
        "positive_rows": int(target.sum()),
        "negative_rows": int((1 - target).sum()),
        "positive_fraction": float(target.mean()),
        "roc_auc_resubstitution_diagnostic": float(roc_auc_score(target, probabilities)),
        "brier_resubstitution_diagnostic": float(brier_score_loss(target, probabilities)),
        "log_loss_resubstitution_diagnostic": float(log_loss(target, probabilities)),
    }


def main() -> None:
    args = parse_args()
    random.seed(20260910)
    np.random.seed(20260910)
    torch.manual_seed(20260910)
    device = choose_deterministic_device(args.device)
    print(
        json.dumps(
            {
                "event": "start",
                "pid": os.getpid(),
                "device": str(device),
                "stage": "local32-fit-then-confirm24",
            }
        ),
        flush=True,
    )
    prereg, prereg_hash = load_preregistration(args.preregistration)

    frozen = prereg["frozen_inputs"]
    relation_checkpoint = PROJECT_ROOT / frozen["relation_checkpoint"]
    relation_report_path = PROJECT_ROOT / frozen["relation_report"]
    if sha256_file(relation_checkpoint) != frozen["relation_checkpoint_sha256"]:
        raise ValueError("frozen v1 relation checkpoint hash mismatch")
    if sha256_file(relation_report_path) != frozen["relation_report_sha256"]:
        raise ValueError("frozen v1 relation report hash mismatch")
    relation_payload = torch.load(relation_checkpoint, map_location="cpu", weights_only=True)
    relation_report = json.loads(relation_report_path.read_text(encoding="utf-8"))
    if not isinstance(relation_payload, Mapping):
        raise ValueError("frozen relation checkpoint must be a mapping")
    exposed = collect_filename_lists(relation_payload)

    local_contract = prereg["calibration_fit"]
    local_names = list(relation_report["selection"]["local_eval_filenames"])
    if (
        len(local_names) != local_contract["count"]
        or filename_digest(local_names) != local_contract["digest"]
    ):
        raise ValueError("local32 roster differs from the frozen v1 artifact")
    split = prereg["reserved64_split"]
    confirm_names = list(split["confirm24"]["filenames"])
    decoder_names = list(split["decoder40"]["filenames"])
    if (
        filename_digest(confirm_names) != split["confirm24"]["digest"]
        or filename_digest(decoder_names) != split["decoder40"]["digest"]
        or filename_digest(confirm_names + decoder_names, sort_names=True)
        != split["parent_digest"]
    ):
        raise ValueError("reserved64 split digest mismatch")
    if set(confirm_names) & set(decoder_names) or (
        set(confirm_names) | set(decoder_names)
    ) & exposed:
        raise ValueError("reserved rosters overlap frozen v1 lineage")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    lookup = manifest_record_lookup(manifest)
    missing = (set(local_names) | set(confirm_names) | set(decoder_names)) - set(lookup)
    if missing:
        raise ValueError(f"preregistered train sources missing from manifest: {sorted(missing)}")

    socket_path = PROJECT_ROOT / frozen["socket_checkpoint"]
    socket = load_socket_checkpoint(socket_path, device=device)
    if str(socket.sha256) != relation_payload["socket_checkpoint"]["sha256"]:
        raise ValueError("Socket lineage differs from the v1 relation artifact")
    relation_contract = relation_payload["contract"]
    head = ComponentRelationReranker(
        int(relation_contract["tile_dimension"]),
        grid=int(relation_contract["grid"]),
        hidden_dimension=int(relation_contract["hidden_dimension"]),
    ).to(device)
    head.load_state_dict(relation_payload["state_dict"], strict=True)
    head.eval()
    if sum(parameter.numel() for parameter in head.parameters()) != EXPECTED_RELATION_PARAMETERS:
        raise ValueError("frozen v1 relation head parameter contract changed")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    calibrator_path = output_dir / "component_relation_confidence.json"
    report_path = output_dir / "report.json"
    if calibrator_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite an existing confidence artifact")

    cache = CleanTileCache(args.targets)
    fit_rows: list[QueryConfidenceFeatures] = []
    fit_labels: list[bool] = []
    fit_runtime: list[dict[str, float]] = []
    for index, filename in enumerate(local_names):
        case = prepare_case(
            cache,
            lookup[filename],
            draw_index=int(local_contract["draw_index"]),
            seed=int(local_contract["synthetic_seed"]),
        )
        output = frozen_case_forward(case, socket=socket, head=head, device=device)
        rows = build_query_confidence_features(
            output.logits,
            output.candidates,
            output.components,
            board_id=case.case_id,
            grid=GRID,
        )
        fit_rows.extend(rows)
        fit_labels.extend(
            bool(output.labels[row.learned_top_candidate].positive) for row in rows
        )
        fit_runtime.append(output.runtime_seconds)
        print(
            json.dumps(
                {"event": "fit-feature", "done": index + 1, "total": len(local_names)}
            ),
            flush=True,
        )
    calibrator = fit_confidence_calibrator(
        fit_rows,
        fit_labels,
        regularization_c=float(local_contract["regularization_c"]),
        maximum_iterations=int(local_contract["maximum_iterations"]),
        random_seed=int(local_contract["random_seed"]),
    )
    if calibrator.parameter_count != EXPECTED_CALIBRATOR_PARAMETERS or tuple(
        calibrator.feature_names
    ) != FEATURE_NAMES:
        raise RuntimeError("tiny confidence calibrator contract changed")

    # confirm24 target access starts here, after the roster/gate/calibrator are frozen.
    confirm_contract = prereg["confirm24"]
    confirm_relation_observations: list[dict[str, Any]] = []
    confirm_confidence_observations: list[dict[str, Any]] = []
    confirm_runtime: list[dict[str, float]] = []
    ranking_before: list[str] = []
    ranking_after: list[str] = []
    confirm_cases: list[dict[str, Any]] = []
    for index, filename in enumerate(confirm_names):
        case = prepare_case(
            cache,
            lookup[filename],
            draw_index=int(confirm_contract["draw_index"]),
            seed=int(confirm_contract["synthetic_seed"]),
        )
        output = frozen_case_forward(case, socket=socket, head=head, device=device)
        rows = build_query_confidence_features(
            output.logits,
            output.candidates,
            output.components,
            board_id=case.case_id,
            grid=GRID,
        )
        before = _ranking_digest(output.logits, output.candidates)
        _ = calibrator.predict_probabilities([row.values for row in rows])
        after = _ranking_digest(output.logits, output.candidates)
        ranking_before.append(before)
        ranking_after.append(after)
        confirm_relation_observations.extend(
            relation_query_observations(
                output.logits,
                output.candidates,
                output.labels,
                output.oracle_relations,
                output.profiles,
                board_id=case.case_id,
            )
        )
        confirm_confidence_observations.extend(
            confidence_query_observations(
                rows,
                calibrator,
                output.candidates,
                output.labels,
                output.oracle_relations,
                output.profiles,
            )
        )
        confirm_runtime.append(output.runtime_seconds)
        confirm_cases.append(
            {
                "case_id": case.case_id,
                "source_filename": case.source_filename,
                "component_count": len(output.components),
                "candidate_count": len(output.candidates),
                "query_count": len(rows),
            }
        )
        print(
            json.dumps(
                {"event": "confirm24", "done": index + 1, "total": len(confirm_names)}
            ),
            flush=True,
        )

    relation_metrics = aggregate_relation_observations(
        confirm_relation_observations,
        high_confidence_caps=CONFIDENCE_CAPS,
    )
    confidence_metrics = aggregate_confidence_observations(
        confirm_confidence_observations,
        caps=CONFIDENCE_CAPS,
    )
    ranking_unchanged = ranking_before == ranking_after
    gate = evaluate_confirm_gate(
        relation_metrics,
        confidence_metrics,
        ranking_unchanged=ranking_unchanged,
        gate_contract=confirm_contract["decoder_eligibility_gate"],
    )
    calibrator_artifact = {
        **calibrator.as_dict(),
        "contract": {
            "purpose": "cross-query probability that frozen learned top1 is correct",
            "within_query_candidate_ranking_changed": False,
            "target_free_at_inference": True,
            "fit_sources": "frozen v1 local32 only",
        },
        "frozen_relation_checkpoint_sha256": frozen["relation_checkpoint_sha256"],
        "selection": {
            "fit_filenames": local_names,
            "fit_digest": local_contract["digest"],
        },
    }
    calibrator_path.write_text(
        json.dumps(calibrator_artifact, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    report = {
        "experiment": prereg["experiment"],
        "status": gate["status"],
        "competition_test_opened": False,
        "decoder40_opened": False,
        "promotion_authorized": False,
        "preregistration": {
            "path": str(args.preregistration.resolve()),
            "sha256": prereg_hash,
            "policy_revision": prereg["policy_revision"],
        },
        "architecture": {
            "calibrator": "standardized L2 logistic regression",
            "feature_count": len(FEATURE_NAMES),
            "parameters": calibrator.parameter_count,
            "target_free_at_inference": True,
            "candidate_ranking_changed": False,
        },
        "frozen_inputs": frozen,
        "selection": {
            "fit_filenames": local_names,
            "fit_digest": filename_digest(local_names),
            "confirm_filenames": confirm_names,
            "confirm_digest": filename_digest(confirm_names),
            "decoder_reserved_filenames": decoder_names,
            "decoder_reserved_digest": filename_digest(decoder_names),
        },
        "fit_metrics": _calibration_fit_metrics(calibrator, fit_rows, fit_labels),
        "confirm_relation_metrics": relation_metrics,
        "confirm_confidence_metrics": confidence_metrics,
        "ranking_invariance": {
            "unchanged": ranking_unchanged,
            "before_digest": filename_digest(ranking_before),
            "after_digest": filename_digest(ranking_after),
        },
        "gate": gate,
        "runtime_seconds": {
            "mean_fit_board": _mean_runtime(fit_runtime),
            "mean_confirm_board": _mean_runtime(confirm_runtime),
        },
        "confirm_cases": confirm_cases,
        "artifacts": {
            "calibrator": str(calibrator_path),
            "calibrator_sha256": sha256_file(calibrator_path),
            "report": str(report_path),
        },
    }
    if not all(
        math.isfinite(value)
        for value in (
            report["fit_metrics"]["roc_auc_resubstitution_diagnostic"],
            report["fit_metrics"]["brier_resubstitution_diagnostic"],
        )
    ):
        raise RuntimeError("calibration diagnostics are non-finite")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "status": gate["status"],
                "decoder40_authorized": gate["decoder40_authorized"],
                "report": str(report_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
