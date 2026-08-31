#!/usr/bin/env python3
"""Run the preregistered learned Union-v2 hard-edge priority pilot.

``selection`` freezes a source/draw roster and every upstream artifact hash
without opening organizer targets.  ``run`` first materialises target-free
340-D Union hard-edge boards for the complete fit and evaluation panels.  Fit
labels are attached only after both feature caches are durable.  Evaluation
layouts and priorities are likewise frozen before exact references are
recreated and scored.

The treatment changes only decoder priority on the existing Union-v2 hard
projection.  Denoised pixels are matcher-only evidence and both decoded arms
remain strict permutations of the original upright dirty tiles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.component_relation_reranker import extract_frozen_socket_context
from aiijc_puzzle.direct_hard_edge_priority import prepare_direct_hard_edge_board
from aiijc_puzzle.direct_hard_edge_production import (
    FROZEN_DIRECT_HARD_EDGE_SHA256,
    LoadedDirectHardEdgeCheckpoint,
    load_direct_hard_edge_checkpoint,
)
from aiijc_puzzle.fullres_fusion_union_priority import (
    FusionUnionPriorityConfig,
    build_fullres_fusion_union_priority,
)
from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import compute_protocol_digest, select_manifest_records, sha256_file
from aiijc_puzzle.raw_twin_union_production import (
    CYCLIC_BORDER_WEIGHT,
    FROZEN_TWIN_SHA256,
    FROZEN_UNION_CHECKPOINT_SHA256,
    LoadedFullResolutionTwinCheckpoint,
    LoadedRawTwinUnionCheckpoint,
    infer_raw_twin_union_assignments,
    load_fullres_twin_checkpoint,
    load_raw_twin_union_checkpoint,
)
from aiijc_puzzle.raw_twin_union_reranker import (
    prepare_raw_twin_union_board,
    restricted_partial_ot,
)
from aiijc_puzzle.socket_confidence_calibration import (
    HardEdgeFeatures,
    exact_edge_labels,
    extract_hard_edge_features,
)
from aiijc_puzzle.socket_decoder import (
    SocketDecoderConfig,
    decode_socket_assignments,
    hard_partial_axis_matching,
)
from aiijc_puzzle.socket_sorter_production import (
    DECODER_EDGE_BUDGET,
    DECODER_SWAP_STEPS,
    LoadedSocketCheckpoint,
)
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)
from aiijc_puzzle.union_hard_edge_priority import (
    FEATURE_NAMES,
    UnionHardEdgeBoard,
    UnionHardEdgePriority,
    prepare_union_hard_edge_board,
    union_hard_edge_listwise_loss,
    union_hard_edge_priority_matrices,
)

try:
    from scripts.run_component_relation_reranker import CleanTileCache, prepare_case
    from scripts.run_direct_hard_edge_priority import (
        _collect_actual_roster_filenames as _direct_actual_roster_filenames,
    )
    from scripts.run_direct_hard_edge_priority import (
        _collect_filename_lists as _direct_declared_filename_lists,
    )
    from scripts.run_direct_hard_edge_priority import (
        _load_json_or_checkpoint as _load_exclusion_artifact,
    )
    from scripts.run_fullres_fusion_union_priority_opened40 import (
        TWIN_CHECKPOINT,
        UNION_CHECKPOINT,
        UNION_CONFIG,
        UNION_SELECTION,
    )
    from scripts.run_fullres_relation_fusion import (
        PROJECT_ROOT,
        prepare_fusion_board,
    )
    from scripts.run_fullres_relation_fusion import (
        _load_config as _load_fusion_preregistration,
    )
    from scripts.run_fullres_relation_fusion import (
        _load_models as _load_fusion_dependencies,
    )
    from scripts.run_fullres_relation_fusion_decoder_d2 import (
        DEFAULT_CONFIG as DEFAULT_D2_CONFIG,
    )
    from scripts.run_fullres_relation_fusion_decoder_d2 import (
        DEFAULT_MANIFEST,
        DEFAULT_TARGETS,
        _load_fusion,
        load_d2_config,
        validate_frozen_inputs,
    )
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    from run_component_relation_reranker import CleanTileCache, prepare_case
    from run_direct_hard_edge_priority import (
        _collect_actual_roster_filenames as _direct_actual_roster_filenames,
    )
    from run_direct_hard_edge_priority import (
        _collect_filename_lists as _direct_declared_filename_lists,
    )
    from run_direct_hard_edge_priority import (
        _load_json_or_checkpoint as _load_exclusion_artifact,
    )
    from run_fullres_fusion_union_priority_opened40 import (
        TWIN_CHECKPOINT,
        UNION_CHECKPOINT,
        UNION_CONFIG,
        UNION_SELECTION,
    )
    from run_fullres_relation_fusion import (
        PROJECT_ROOT,
        prepare_fusion_board,
    )
    from run_fullres_relation_fusion import (
        _load_config as _load_fusion_preregistration,
    )
    from run_fullres_relation_fusion import (
        _load_models as _load_fusion_dependencies,
    )
    from run_fullres_relation_fusion_decoder_d2 import (
        DEFAULT_CONFIG as DEFAULT_D2_CONFIG,
    )
    from run_fullres_relation_fusion_decoder_d2 import (
        DEFAULT_MANIFEST,
        DEFAULT_TARGETS,
        _load_fusion,
        load_d2_config,
        validate_frozen_inputs,
    )


CONFIG_SCHEMA = "aiijc-union-hard-priority-pilot-v1"
ROSTER_AUDIT_SCHEMA = "aiijc-union-hard-priority-roster-audit-v1"
EXPERIMENT = "union-hard-edge-priority-pilot-v1"
SELECTION_NAMESPACE = "aiijc-union-hard-priority-pilot-v1-fit64x2-eval16x2"
SELECTION_SEED = 1_506_880_951
SYNTHETIC_SEED = 1_267_233_517
ORGANIZER_TRAIN_COUNT = 5_600
ELIGIBLE_TRAIN_COUNT = 2_536
EXCLUDED_TRAIN_COUNT = 3_064
EXCLUDED_TRAIN_DIGEST = "96560e08a123d9d53ffc981388e67bd9c7a8943fa9a59f77347a84b2c31922b6"
GLOBAL_EXCLUSION_COUNT = 3_176
GLOBAL_EXCLUSION_DIGEST = "d1c1f455ebbbb80384c6a4d1afe7cb53e1fc21e158531fd94eb6e38dc2d8785d"
ACTIVE_GLOBAL_REGISTRY_INDICES = (7, 0, 3, 4, 5, 6, 8, 9, 10, 11)
LINEAGE_ONLY_REGISTRY_INDICES = (1, 2)
FIT_SOURCE_ORDER_DIGEST = "2cafca0d2d231857afb626dc84335dc38740a9fdd3c1f6427376fa1f5a3c78fc"
EVAL_SOURCE_ORDER_DIGEST = "13f7fe84262f9c4d0aee7ce80dfdc1edeec3ce7f1b5082f06ae5c6aceda6fa5f"
FIT_CASES_DIGEST = "4c123e3a00cfe726d022c2d597f3b1feb65c79cf967c6f936178dd837de4c907"
EVAL_CASES_DIGEST = "6b872422b32562b845deb514fd478cd21379384cccdcf07a9e4db257e71b6e04"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/union_hard_edge_priority_pilot_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/union-hard-edge-priority/pilot-v1"
DIRECT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/direct-hard-edge-priority/v1-fit256-s600-d1-32-cpu"
    / "direct_hard_edge_priority.pt"
)
GRID = 24
COUNT = GRID * GRID
HARD_EDGES_PER_AXIS = GRID * (GRID - 1)
HARD_EDGE_COUNT = 2 * HARD_EDGES_PER_AXIS
FIT_SOURCE_COUNT = 64
EVAL_SOURCE_COUNT = 16
DRAWS = (0, 1)
FIT_CASE_COUNT = FIT_SOURCE_COUNT * len(DRAWS)
EVAL_CASE_COUNT = EVAL_SOURCE_COUNT * len(DRAWS)
TRAINING_STEPS = 400
HIDDEN_DIMENSION = 64
RESIDUAL_LIMIT = 2.0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PAIRWISE_WEIGHT = 0.75
RESIDUAL_WEIGHT = 1e-3
GRADIENT_CLIP = 1.0
FUSION_QUERY_CAP = 32
FUSION_CANDIDATE_RANK_CAP = 5
FUSION_BOOST_SCALE = 1.0
INFERENCE_BATCH = 576
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 287_368_430

DEFAULT_FROZEN_INPUTS = {
    "d2_config": DEFAULT_D2_CONFIG,
    "twin_checkpoint": TWIN_CHECKPOINT,
    "union_checkpoint": UNION_CHECKPOINT,
    "union_config": UNION_CONFIG,
    "union_selection": UNION_SELECTION,
    "direct_checkpoint": DIRECT_CHECKPOINT,
}

RUNTIME_SOURCE_PATHS = {
    "pilot_runner": Path(__file__).resolve(),
    "union_hard_edge_priority": PROJECT_ROOT / "src/aiijc_puzzle/union_hard_edge_priority.py",
    "fullres_relation_fusion_adapter": PROJECT_ROOT / "scripts/run_fullres_relation_fusion.py",
    "fullres_fusion_union_priority": PROJECT_ROOT
    / "src/aiijc_puzzle/fullres_fusion_union_priority.py",
    "raw_twin_union_production": PROJECT_ROOT / "src/aiijc_puzzle/raw_twin_union_production.py",
    "raw_twin_union_features": PROJECT_ROOT / "src/aiijc_puzzle/raw_twin_union_reranker.py",
    "direct_hard_edge_features": PROJECT_ROOT / "src/aiijc_puzzle/direct_hard_edge_priority.py",
}


@dataclass(frozen=True)
class RunPaths:
    cache_dir: Path
    fit_labels: Path
    fit_labels_metadata: Path
    checkpoint: Path
    frozen_eval: Path
    frozen_eval_metadata: Path
    report: Path


@dataclass(frozen=True)
class FrozenModels:
    socket: LoadedSocketCheckpoint
    relation: torch.nn.Module
    denoiser: torch.nn.Module
    fusion: torch.nn.Module
    twin: LoadedFullResolutionTwinCheckpoint
    union: LoadedRawTwinUnionCheckpoint
    direct: LoadedDirectHardEdgeCheckpoint
    d2_config: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TargetFreeCase:
    """Synthetic matcher view that cannot expose an exact tile-position map."""

    case_id: str
    source_filename: str
    dirty_tiles: np.ndarray


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("selection", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument("--benchmark-one-case", action="store_true")
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="exclusively materialise --config from a target-blind roster audit",
    )
    parser.add_argument(
        "--roster-audit",
        type=Path,
        help="canonical target-blind roster audit used only by --write-config",
    )
    parser.add_argument("--inference-batch", type=int, default=INFERENCE_BATCH)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args(argv)


def _select_device(name: str, *, allow_nondeterministic_mps: bool) -> torch.device:
    if name == "mps":
        if not allow_nondeterministic_mps:
            raise ValueError("MPS requires --allow-nondeterministic-mps")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        torch.use_deterministic_algorithms(False)
        return torch.device("mps")
    if name != "cpu":
        raise ValueError("device must be cpu or mps")
    if allow_nondeterministic_mps:
        raise ValueError("--allow-nondeterministic-mps requires MPS")
    torch.use_deterministic_algorithms(True)
    return torch.device("cpu")


def _names_digest(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _cases_digest(names: Sequence[str], draws: Sequence[int]) -> str:
    values = [f"{name}\0{int(draw)}" for name in names for draw in draws]
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _effective_namespace(manifest_digest: str, registry_digest: str) -> str:
    return (
        f"{SELECTION_NAMESPACE}\0manifest={manifest_digest}"
        f"\0registry={registry_digest}\0excluded={EXCLUDED_TRAIN_DIGEST}"
    )


def _canonical_json_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dirty_sha256(tiles: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(tiles).tobytes()).hexdigest()


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _resolve_path(value: Any, *, name: str) -> Path:
    raw = value.get("path") if isinstance(value, Mapping) else value
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"frozen input {name} path is malformed")
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"frozen input {name} does not exist: {path}")
    return path


def _dotted_value(payload: Mapping[str, Any], field: str) -> Any:
    value: Any = payload
    for part in field.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"exclusion registry field is absent: {field}")
        value = value[part]
    return value


def _explicit_registry_names(payload: Mapping[str, Any], fields: Any) -> set[str]:
    if (
        not isinstance(fields, list)
        or not fields
        or not all(isinstance(field, str) and field for field in fields)
    ):
        raise ValueError("exclusion registry fields are malformed")
    names: set[str] = set()
    for field in fields:
        values = _dotted_value(payload, field)
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value.endswith(".png") for value in values
        ):
            raise ValueError(f"exclusion registry field is not an explicit PNG list: {field}")
        names.update(Path(value).name for value in values)
    return names


def _direct_transitive_names(
    payload: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> set[str]:
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Direct exclusion artifact has no selection mapping")
    exclusion = selection.get("exclusion")
    nested = exclusion.get("registry") if isinstance(exclusion, Mapping) else None
    if not isinstance(nested, list) or nested != evidence.get("nested_registry"):
        raise ValueError("Direct nested exclusion registry differs from roster audit")
    names: set[str] = set()
    for index, row in enumerate(nested):
        if not isinstance(row, Mapping):
            raise ValueError(f"Direct nested registry row {index} is malformed")
        path = _resolve_path(row, name=f"direct_nested_registry_{index}")
        if row.get("sha256") != sha256_file(path):
            raise ValueError(f"Direct nested registry row {index} SHA-256 mismatch")
        child = _load_exclusion_artifact(path)
        role = str(row.get("role", ""))
        found = (
            _direct_actual_roster_filenames(child)
            if role.startswith("actual-panel-roster-exclusion")
            else _direct_declared_filename_lists(child)
        )
        if len(found) != int(row.get("filename_count", -1)):
            raise ValueError(f"Direct nested registry row {index} count mismatch")
        expected_digest = row.get("filename_digest")
        if expected_digest is not None and _names_digest(tuple(sorted(found))) != expected_digest:
            raise ValueError(f"Direct nested registry row {index} digest mismatch")
        names.update(found)
    additional_fields = evidence.get("additional_fields")
    expected_additional = [
        "selection.fit_source_filenames",
        "selection.d1_source_filenames",
    ]
    if additional_fields != expected_additional:
        raise ValueError("Direct exclusion additional fields changed")
    names.update(_explicit_registry_names(payload, additional_fields))
    if len(names) != int(evidence.get("resolved_filename_count", -1)) or _names_digest(
        tuple(sorted(names))
    ) != evidence.get("resolved_filename_digest"):
        raise ValueError("Direct transitive exclusion resolution mismatch")
    return names


def _recompute_global_exclusion(
    registry: list[Mapping[str, Any]],
    exclusion: Mapping[str, Any],
) -> set[str]:
    membership = exclusion.get("membership_recipe")
    if (
        not isinstance(membership, Mapping)
        or membership.get("active_registry_indices") != list(ACTIVE_GLOBAL_REGISTRY_INDICES)
        or membership.get("lineage_pin_only_registry_indices")
        != list(LINEAGE_ONLY_REGISTRY_INDICES)
    ):
        raise ValueError("roster audit active registry membership recipe changed")
    resolutions = exclusion.get("registry_resolution")
    if not isinstance(resolutions, list):
        raise ValueError("roster audit registry resolution evidence is absent")
    resolution_by_index: dict[int, Mapping[str, Any]] = {}
    for row in resolutions:
        if not isinstance(row, Mapping) or not isinstance(row.get("registry_index"), int):
            raise ValueError("roster audit registry resolution row is malformed")
        index = int(row["registry_index"])
        if index in resolution_by_index:
            raise ValueError("roster audit registry resolution indices repeat")
        resolution_by_index[index] = row
    direct_evidence = exclusion.get("direct_transitive_resolution")
    if 7 in ACTIVE_GLOBAL_REGISTRY_INDICES and (
        not isinstance(direct_evidence, Mapping) or direct_evidence.get("registry_index") != 7
    ):
        raise ValueError("Direct transitive exclusion evidence is absent")

    result: set[str] = set()
    for index in ACTIVE_GLOBAL_REGISTRY_INDICES:
        if not 0 <= index < len(registry):
            raise ValueError("active exclusion registry index is out of range")
        registry_row = registry[index]
        artifact_path = _resolve_path(registry_row, name=f"roster_registry_{index}")
        payload = _load_exclusion_artifact(artifact_path)
        found = (
            _direct_transitive_names(payload, direct_evidence)
            if index == 7
            else _explicit_registry_names(payload, registry_row.get("fields"))
        )
        resolution = resolution_by_index.get(index)
        if (
            not isinstance(resolution, Mapping)
            or resolution.get("active_for_membership") is not True
        ):
            raise ValueError(f"active registry resolution {index} is absent")
        if len(found) != int(resolution.get("resolved_filename_count", -1)) or _names_digest(
            tuple(sorted(found))
        ) != resolution.get("resolved_filename_digest"):
            raise ValueError(f"active registry resolution {index} mismatch")
        result.update(found)
    for index in LINEAGE_ONLY_REGISTRY_INDICES:
        resolution = resolution_by_index.get(index)
        if (
            not isinstance(resolution, Mapping)
            or resolution.get("active_for_membership") is not False
        ):
            raise ValueError(f"lineage-only registry resolution {index} changed")
    return result


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise ValueError(f"pilot config does not exist: {path}")
    digest_path = path.with_name(f"{path.name}.sha256")
    if not digest_path.is_file():
        raise ValueError("pilot config has no immutable SHA-256 sidecar")
    tokens = digest_path.read_text(encoding="utf-8").split()
    if not tokens or tokens[0] != sha256_file(path):
        raise ValueError("pilot config SHA-256 sidecar mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("experiment") != EXPERIMENT:
        raise ValueError("unexpected Union hard-edge pilot experiment")
    if payload.get("registered_before_target_access") is not True:
        raise ValueError("pilot config lacks pre-target registration")
    training = payload.get("training")
    if not isinstance(training, Mapping) or int(training.get("steps", -1)) != TRAINING_STEPS:
        raise ValueError(f"pilot training must contain exactly {TRAINING_STEPS} steps")
    if training.get("hyperparameter_sweep", False) is not False:
        raise ValueError("pilot forbids a hyperparameter sweep")
    pinned_training = {
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "pairwise_weight": PAIRWISE_WEIGHT,
        "residual_weight": RESIDUAL_WEIGHT,
        "gradient_clip": GRADIENT_CLIP,
    }
    if any(training.get(name) != value for name, value in pinned_training.items()):
        raise ValueError("pilot training hyperparameters changed")
    model = payload.get("model")
    pinned_model = {
        "architecture": "union-hard-edge-deepsets-bounded-residual-v1",
        "feature_dimension": len(FEATURE_NAMES),
        "hidden_dimension": HIDDEN_DIMENSION,
        "residual_limit": RESIDUAL_LIMIT,
        "hard_edge_count": HARD_EDGE_COUNT,
        "edge_budget_per_axis": DECODER_EDGE_BUDGET,
        "zero_initialised_residual": True,
    }
    if (
        not isinstance(model, Mapping)
        or any(model.get(name) != value for name, value in pinned_model.items())
        or tuple(model.get("feature_names", ())) != FEATURE_NAMES
    ):
        raise ValueError("pilot model contract changed")
    fullres_priority = payload.get("fullres_priority")
    pinned_fullres = {
        "query_cap": FUSION_QUERY_CAP,
        "candidate_rank_cap": FUSION_CANDIDATE_RANK_CAP,
        "boost_scale": FUSION_BOOST_SCALE,
    }
    if not isinstance(fullres_priority, Mapping) or any(
        fullres_priority.get(name) != value for name, value in pinned_fullres.items()
    ):
        raise ValueError("pilot full-resolution priority config changed")
    runtime = payload.get("runtime")
    pinned_runtime = {
        "pilot_device": "mps",
        "mps_requires_explicit_nondeterminism": True,
        "cpu_benchmark_allowed": True,
        "inference_batch": INFERENCE_BATCH,
        "feature_cache_dtype": "float32",
    }
    if not isinstance(runtime, Mapping) or any(
        runtime.get(name) != value for name, value in pinned_runtime.items()
    ):
        raise ValueError("pilot runtime contract changed")
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping) or not all(
        protocol.get(name) is True
        for name in (
            "commitment_written_before_target_access",
            "eval_predictions_frozen_before_reference",
            "single_eval_no_tuning",
            "calibration_holdout_test_forbidden",
        )
    ):
        raise ValueError("pilot protocol gates are incomplete")
    audit = payload.get("roster_audit")
    if not isinstance(audit, Mapping) or set(audit) != {"path", "sha256"}:
        raise ValueError("roster_audit needs explicit path and sha256")
    audit_path = _resolve_path(audit, name="roster_audit")
    if sha256_file(audit_path) != audit.get("sha256"):
        raise ValueError("roster audit changed after config generation")
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit_payload.get("schema") != ROSTER_AUDIT_SCHEMA:
        raise ValueError("roster audit schema changed after config generation")
    return payload, sha256_file(path)


def write_preregistered_config(args: argparse.Namespace) -> Path:
    """Materialise the immutable pilot config without opening organizer targets."""

    if args.mode != "selection" or not args.write_config:
        raise ValueError("config generation is valid only in selection mode")
    if args.benchmark_one_case:
        raise ValueError("config generation cannot benchmark a case")
    if args.roster_audit is None:
        raise ValueError("--write-config requires --roster-audit")
    config_path = args.config.resolve()
    digest_path = config_path.with_name(f"{config_path.name}.sha256")
    if config_path.exists() or digest_path.exists():
        raise FileExistsError("refusing to overwrite a preregistered pilot config")
    audit_path = args.roster_audit.resolve()
    if not audit_path.is_file():
        raise ValueError(f"roster audit does not exist: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema") != ROSTER_AUDIT_SCHEMA:
        raise ValueError("unexpected Union hard-edge roster audit schema")
    if (
        audit.get("created_before_target_access") is not True
        or audit.get("target_images_accessed") is not False
    ):
        raise ValueError("roster audit is not target-blind and pre-access")
    audit_selection = audit.get("selection")
    if not isinstance(audit_selection, Mapping):
        raise ValueError("roster audit has no explicit selection mapping")
    exclusion = audit.get("exclusion")
    if not isinstance(exclusion, Mapping):
        raise ValueError("roster audit has no explicit exclusion evidence")
    registry = exclusion.get("registry")
    if not isinstance(registry, list) or not registry:
        raise ValueError("roster audit exclusion registry is absent")
    for index, row in enumerate(registry):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256", "fields"}
            or not isinstance(row.get("path"), str)
        ):
            raise ValueError(f"roster audit registry row {index} is malformed")
        artifact_path = _resolve_path(row, name=f"roster_registry_{index}")
        if row.get("sha256") != sha256_file(artifact_path):
            raise ValueError(f"roster audit registry row {index} SHA-256 mismatch")
    registry_digest = _canonical_json_digest(registry)
    if (
        audit_selection.get("registry_digest") != registry_digest
        or exclusion.get("registry_digest") != registry_digest
    ):
        raise ValueError("roster audit registry digest mismatch")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("manifest protocol digest is invalid")
    audit_manifest = audit.get("manifest")
    if not isinstance(audit_manifest, Mapping) or (
        audit_manifest.get("protocol_digest") != manifest["protocol_digest"]
        or audit_manifest.get("sha256") != sha256_file(args.manifest)
    ):
        raise ValueError("roster audit does not bind the supplied manifest")

    train_rows = manifest.get("splits", {}).get("train")
    if not isinstance(train_rows, list) or len(train_rows) != ORGANIZER_TRAIN_COUNT:
        raise ValueError("manifest organizer-train roster changed")
    train_names = tuple(str(row["filename"]) for row in train_rows)
    if len(set(train_names)) != ORGANIZER_TRAIN_COUNT:
        raise ValueError("manifest organizer-train filenames are not unique")

    def explicit_names(field: str, expected: int, digest: str) -> tuple[str, ...]:
        raw = exclusion.get(field)
        if not isinstance(raw, list) or not all(
            isinstance(name, str) and Path(name).name == name and name.endswith(".png")
            for name in raw
        ):
            raise ValueError(f"roster audit exclusion {field} is malformed")
        names = tuple(raw)
        if len(names) != expected or len(set(names)) != expected:
            raise ValueError(f"roster audit exclusion {field} count changed")
        if _names_digest(tuple(sorted(names))) != digest:
            raise ValueError(f"roster audit exclusion {field} digest mismatch")
        return names

    excluded_train = explicit_names(
        "excluded_train_filenames",
        EXCLUDED_TRAIN_COUNT,
        EXCLUDED_TRAIN_DIGEST,
    )
    global_exclusion = explicit_names(
        "global_exclusion_filenames",
        GLOBAL_EXCLUSION_COUNT,
        GLOBAL_EXCLUSION_DIGEST,
    )
    train_set = set(train_names)
    excluded_train_set = set(excluded_train)
    global_exclusion_set = set(global_exclusion)
    if _recompute_global_exclusion(registry, exclusion) != global_exclusion_set:
        raise ValueError("explicit global exclusion differs from active registry resolution")
    if excluded_train_set != train_set & global_exclusion_set:
        raise ValueError("excluded-train roster is not the global-exclusion train intersection")
    if len(train_set - excluded_train_set) != ELIGIBLE_TRAIN_COUNT:
        raise ValueError("eligible organizer-train count changed")

    selection_fields = (
        "namespace",
        "effective_namespace",
        "registry_digest",
        "selection_seed",
        "synthetic_seed",
        "bootstrap_seed",
        "draw_indices",
        "organizer_train_count",
        "eligible_count",
        "excluded_train_count",
        "excluded_train_digest",
        "global_exclusion_count",
        "global_exclusion_digest",
        "fit_source_count",
        "eval_source_count",
        "fit_source_filenames",
        "fit_source_order_digest",
        "fit_source_set_digest",
        "fit_cases_digest",
        "eval_source_filenames",
        "eval_source_order_digest",
        "eval_source_set_digest",
        "eval_cases_digest",
        "overlaps",
    )
    missing = [name for name in selection_fields if name not in audit_selection]
    if missing:
        raise ValueError(f"roster audit selection is missing explicit fields: {missing}")
    selection = {name: audit_selection[name] for name in selection_fields}
    selection["manifest_split"] = "train"
    expected_namespace = _effective_namespace(
        str(manifest["protocol_digest"]),
        registry_digest,
    )
    if selection["effective_namespace"] != expected_namespace:
        raise ValueError("roster audit effective namespace recipe changed")
    recipes = audit.get("recipes")
    namespace_recipe = recipes.get("effective_namespace") if isinstance(recipes, Mapping) else None
    if (
        not isinstance(namespace_recipe, Mapping)
        or namespace_recipe.get("global_exclusion_digest_in_namespace") is not False
        or namespace_recipe.get("exact_value") != expected_namespace
    ):
        raise ValueError("global exclusion digest must remain outside the ranking namespace")
    selection["global_exclusion_digest_in_namespace"] = False
    if (
        audit_selection.get("fit_case_count") != FIT_CASE_COUNT
        or audit_selection.get("eval_case_count") != EVAL_CASE_COUNT
    ):
        raise ValueError("roster audit case counts changed")
    ranked = select_manifest_records(
        manifest,
        "train",
        limit=ORGANIZER_TRAIN_COUNT,
        seed=SELECTION_SEED,
        namespace=str(selection["effective_namespace"]),
    )
    eligible = tuple(
        str(record["filename"])
        for record in ranked
        if str(record["filename"]) not in excluded_train_set
    )
    recomputed_fit = eligible[:FIT_SOURCE_COUNT]
    recomputed_eval = eligible[FIT_SOURCE_COUNT : FIT_SOURCE_COUNT + EVAL_SOURCE_COUNT]
    if (
        tuple(selection["fit_source_filenames"]) != recomputed_fit
        or tuple(selection["eval_source_filenames"]) != recomputed_eval
    ):
        raise ValueError("roster audit exact80 differs from deterministic fresh selection")
    if (set(recomputed_fit) | set(recomputed_eval)) & global_exclusion_set:
        raise RuntimeError("deterministic fresh selection overlaps global exclusion")
    computed_overlaps = {"fit_eval": 0, "global_exclusion": 0}
    if selection["overlaps"] != computed_overlaps:
        raise ValueError("roster audit overlaps differ from recomputed all-zero overlaps")
    selection["overlaps"] = computed_overlaps
    frozen_inputs = {
        name: {"path": _project_path(path), "sha256": sha256_file(path)}
        for name, path in DEFAULT_FROZEN_INPUTS.items()
    }
    payload: dict[str, Any] = {
        "schema": CONFIG_SCHEMA,
        "experiment": EXPERIMENT,
        "registered_before_target_access": True,
        "protocol": {
            "commitment_written_before_target_access": True,
            "eval_predictions_frozen_before_reference": True,
            "single_eval_no_tuning": True,
            "calibration_holdout_test_forbidden": True,
        },
        "roster_audit": {
            "path": _project_path(audit_path),
            "sha256": sha256_file(audit_path),
        },
        "selection": selection,
        "frozen_inputs": frozen_inputs,
        "model": {
            "architecture": "union-hard-edge-deepsets-bounded-residual-v1",
            "feature_dimension": len(FEATURE_NAMES),
            "feature_names": list(FEATURE_NAMES),
            "hidden_dimension": HIDDEN_DIMENSION,
            "residual_limit": RESIDUAL_LIMIT,
            "hard_edge_count": HARD_EDGE_COUNT,
            "edge_budget_per_axis": DECODER_EDGE_BUDGET,
            "zero_initialised_residual": True,
        },
        "fullres_priority": {
            "query_cap": FUSION_QUERY_CAP,
            "candidate_rank_cap": FUSION_CANDIDATE_RANK_CAP,
            "boost_scale": FUSION_BOOST_SCALE,
        },
        "training": {
            "steps": TRAINING_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "pairwise_weight": PAIRWISE_WEIGHT,
            "residual_weight": RESIDUAL_WEIGHT,
            "gradient_clip": GRADIENT_CLIP,
            "hyperparameter_sweep": False,
        },
        "runtime": {
            "pilot_device": "mps",
            "mps_requires_explicit_nondeterminism": True,
            "cpu_benchmark_allowed": True,
            "inference_batch": INFERENCE_BATCH,
            "feature_cache_dtype": "float32",
        },
        "legality": {
            "organizer_train_only": True,
            "restored_pixels_matcher_only": True,
            "strict_original_tile_layouts": True,
            "competition_test_forbidden": True,
        },
    }
    _manifest_records(payload, manifest)
    _frozen_input_records(payload)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_exclusive(config_path, payload)
    digest = sha256_file(config_path)
    _write_text_exclusive(digest_path, f"{digest}  {config_path.name}\n")
    return config_path


def _manifest_records(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], tuple[str, ...], tuple[str, ...]]:
    if manifest.get("protocol_digest") != compute_protocol_digest(dict(manifest)):
        raise ValueError("manifest protocol digest is invalid")
    selection = config.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("pilot config has no selection mapping")
    split = str(selection.get("manifest_split", "train"))
    if split != "train":
        raise ValueError("pilot may use organizer train only")
    pinned = {
        "namespace": SELECTION_NAMESPACE,
        "selection_seed": SELECTION_SEED,
        "synthetic_seed": SYNTHETIC_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "organizer_train_count": ORGANIZER_TRAIN_COUNT,
        "eligible_count": ELIGIBLE_TRAIN_COUNT,
        "excluded_train_count": EXCLUDED_TRAIN_COUNT,
        "excluded_train_digest": EXCLUDED_TRAIN_DIGEST,
        "global_exclusion_count": GLOBAL_EXCLUSION_COUNT,
        "global_exclusion_digest": GLOBAL_EXCLUSION_DIGEST,
        "fit_source_count": FIT_SOURCE_COUNT,
        "eval_source_count": EVAL_SOURCE_COUNT,
        "fit_cases_digest": FIT_CASES_DIGEST,
        "eval_cases_digest": EVAL_CASES_DIGEST,
    }
    for name, expected_value in pinned.items():
        if selection.get(name) != expected_value:
            raise ValueError(f"selection pinned field {name} changed")
    registry_digest = selection.get("registry_digest")
    if not isinstance(registry_digest, str) or len(registry_digest) != 64:
        raise ValueError("selection registry_digest must be a full SHA-256")
    effective_namespace = selection.get("effective_namespace")
    if effective_namespace != _effective_namespace(
        str(manifest["protocol_digest"]),
        registry_digest,
    ):
        raise ValueError("selection effective_namespace recipe changed")
    if selection.get("global_exclusion_digest_in_namespace") is not False:
        raise ValueError("global exclusion digest must remain outside the ranking namespace")
    overlaps = selection.get("overlaps")
    if (
        not isinstance(overlaps, Mapping)
        or not overlaps
        or any(value != 0 for value in overlaps.values())
    ):
        raise ValueError("selection overlaps must be an explicit all-zero mapping")
    if selection.get("draw_indices") != list(DRAWS):
        raise ValueError(f"selection draw_indices must be exactly {list(DRAWS)}")
    rows = manifest.get("splits", {}).get(split)
    if not isinstance(rows, list):
        raise ValueError("manifest train split is absent")
    if len(rows) != ORGANIZER_TRAIN_COUNT:
        raise ValueError("manifest organizer-train count changed")
    lookup = {str(row["filename"]): row for row in rows}

    def roster(
        name: str,
        expected: int,
        *,
        order_digest_name: str,
        pinned_order_digest: str,
        set_digest_name: str,
    ) -> tuple[str, ...]:
        raw = selection.get(name)
        if not isinstance(raw, list) or not all(
            isinstance(value, str) and Path(value).name == value and value.endswith(".png")
            for value in raw
        ):
            raise ValueError(f"selection {name} is malformed")
        values = tuple(raw)
        if len(values) != expected or len(set(values)) != expected:
            raise ValueError(f"selection {name} must contain {expected} unique sources")
        observed_order_digest = _names_digest(values)
        if (
            selection.get(order_digest_name) != pinned_order_digest
            or observed_order_digest != pinned_order_digest
        ):
            raise ValueError(f"selection {name} order digest mismatch")
        observed_set_digest = _names_digest(tuple(sorted(values)))
        if selection.get(set_digest_name) != observed_set_digest:
            raise ValueError(f"selection {name} set digest mismatch")
        if set(values) - set(lookup):
            raise ValueError(f"selection {name} contains a source absent from train")
        return values

    fit = roster(
        "fit_source_filenames",
        FIT_SOURCE_COUNT,
        order_digest_name="fit_source_order_digest",
        pinned_order_digest=FIT_SOURCE_ORDER_DIGEST,
        set_digest_name="fit_source_set_digest",
    )
    evaluation = roster(
        "eval_source_filenames",
        EVAL_SOURCE_COUNT,
        order_digest_name="eval_source_order_digest",
        pinned_order_digest=EVAL_SOURCE_ORDER_DIGEST,
        set_digest_name="eval_source_set_digest",
    )
    if set(fit) & set(evaluation):
        raise ValueError("fit and eval source rosters overlap")
    if _cases_digest(fit, DRAWS) != FIT_CASES_DIGEST:
        raise ValueError("fit source/draw case digest mismatch")
    if _cases_digest(evaluation, DRAWS) != EVAL_CASES_DIGEST:
        raise ValueError("eval source/draw case digest mismatch")
    return lookup, fit, evaluation


def _frozen_input_records(config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    configured = config.get("frozen_inputs", {})
    if not isinstance(configured, Mapping):
        raise ValueError("frozen_inputs must be a mapping")
    result: dict[str, dict[str, str]] = {}
    for name in DEFAULT_FROZEN_INPUTS:
        requested = configured.get(name)
        if not isinstance(requested, Mapping) or set(requested) != {"path", "sha256"}:
            raise ValueError(f"frozen input {name} needs explicit path and sha256")
        path = _resolve_path(requested, name=name)
        observed = sha256_file(path)
        if requested.get("sha256") != observed:
            raise ValueError(f"frozen input {name} SHA-256 mismatch")
        result[name] = {"path": _project_path(path), "sha256": observed}
    return result


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _write_text_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(value)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def create_selection_commitment(args: argparse.Namespace) -> Path:
    if args.benchmark_one_case:
        raise ValueError("--benchmark-one-case is valid only in run mode")
    config, config_sha = _load_config(args.config)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    _, fit, evaluation = _manifest_records(config, manifest)
    frozen = _frozen_input_records(config)
    runtime_sources = {
        name: {"path": _project_path(source), "sha256": sha256_file(source)}
        for name, source in RUNTIME_SOURCE_PATHS.items()
    }
    selection = config["selection"]
    path = args.output_dir.resolve() / "selection-commitment.json"
    _write_json_exclusive(
        path,
        {
            "schema": CONFIG_SCHEMA,
            "artifact": "selection-commitment",
            "experiment": EXPERIMENT,
            "created_before_target_access": True,
            "config": {"path": _project_path(args.config), "sha256": config_sha},
            "roster_audit": dict(config["roster_audit"]),
            "manifest": {
                "path": _project_path(args.manifest),
                "sha256": sha256_file(args.manifest),
                "protocol_digest": manifest["protocol_digest"],
                "split": "train",
            },
            "namespace": selection["namespace"],
            "effective_namespace": selection["effective_namespace"],
            "global_exclusion_digest_in_namespace": False,
            "registry_digest": selection["registry_digest"],
            "selection_seed": int(selection["selection_seed"]),
            "synthetic_seed": int(selection["synthetic_seed"]),
            "bootstrap_seed": int(selection["bootstrap_seed"]),
            "draw_indices": list(DRAWS),
            "organizer_train_count": int(selection["organizer_train_count"]),
            "eligible_count": int(selection["eligible_count"]),
            "excluded_train_count": int(selection["excluded_train_count"]),
            "excluded_train_digest": selection["excluded_train_digest"],
            "global_exclusion_count": int(selection["global_exclusion_count"]),
            "global_exclusion_digest": selection["global_exclusion_digest"],
            "overlaps": dict(selection["overlaps"]),
            "fit": {
                "source_filenames": list(fit),
                "source_order_digest": _names_digest(fit),
                "source_set_digest": _names_digest(tuple(sorted(fit))),
                "cases_digest": selection["fit_cases_digest"],
                "draws": list(DRAWS),
                "case_count": FIT_CASE_COUNT,
            },
            "eval": {
                "source_filenames": list(evaluation),
                "source_order_digest": _names_digest(evaluation),
                "source_set_digest": _names_digest(tuple(sorted(evaluation))),
                "cases_digest": selection["eval_cases_digest"],
                "draws": list(DRAWS),
                "case_count": EVAL_CASE_COUNT,
            },
            "frozen_inputs": frozen,
            "runtime_sources": runtime_sources,
            "training": {
                "steps": TRAINING_STEPS,
                "single_fixed_arm": True,
                "hyperparameter_sweep": False,
            },
            "legality": {
                "organizer_train_only": True,
                "restored_pixels_matcher_only": True,
                "strict_original_tile_layouts": True,
                "targets_or_labels_in_feature_builder": False,
            },
        },
    )
    return path


def _load_commitment(
    output_dir: Path,
    config_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    path = output_dir.resolve() / "selection-commitment.json"
    if not path.is_file():
        raise RuntimeError("run mode requires a prior selection commitment")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("artifact") != (
        "selection-commitment"
    ):
        raise ValueError("selection commitment schema changed")
    if payload.get("created_before_target_access") is not True:
        raise ValueError("selection commitment timing contract is absent")
    if payload["config"]["sha256"] != sha256_file(config_path):
        raise ValueError("config changed after selection commitment")
    if payload["manifest"]["sha256"] != sha256_file(manifest_path):
        raise ValueError("manifest changed after selection commitment")
    pinned = {
        "namespace": SELECTION_NAMESPACE,
        "selection_seed": SELECTION_SEED,
        "synthetic_seed": SYNTHETIC_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "organizer_train_count": ORGANIZER_TRAIN_COUNT,
        "eligible_count": ELIGIBLE_TRAIN_COUNT,
        "excluded_train_count": EXCLUDED_TRAIN_COUNT,
        "excluded_train_digest": EXCLUDED_TRAIN_DIGEST,
        "global_exclusion_count": GLOBAL_EXCLUSION_COUNT,
        "global_exclusion_digest": GLOBAL_EXCLUSION_DIGEST,
    }
    if any(payload.get(name) != value for name, value in pinned.items()):
        raise ValueError("selection commitment pinned fields changed")
    registry_digest = payload.get("registry_digest")
    if not isinstance(registry_digest, str) or payload.get(
        "effective_namespace"
    ) != _effective_namespace(
        str(payload["manifest"]["protocol_digest"]),
        registry_digest,
    ):
        raise ValueError("selection commitment effective namespace changed")
    if payload.get("global_exclusion_digest_in_namespace") is not False:
        raise ValueError("selection commitment namespace policy changed")
    for name, record in payload["frozen_inputs"].items():
        path_value = _resolve_path(record, name=name)
        if sha256_file(path_value) != record["sha256"]:
            raise ValueError(f"frozen input {name} changed after selection")
    runtime_sources = payload.get("runtime_sources")
    if not isinstance(runtime_sources, Mapping) or set(runtime_sources) != set(
        RUNTIME_SOURCE_PATHS
    ):
        raise ValueError("selection commitment runtime-source roster changed")
    for name, record in runtime_sources.items():
        path_value = _resolve_path(record, name=f"runtime_source_{name}")
        expected_path = RUNTIME_SOURCE_PATHS[name].resolve()
        if path_value != expected_path or sha256_file(path_value) != record.get("sha256"):
            raise ValueError(f"runtime source {name} changed after selection")
    return payload


def _prepare_run_paths(output_dir: Path) -> RunPaths:
    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = RunPaths(
        cache_dir=root / "target-free-cache",
        fit_labels=root / "fit-labels.npy",
        fit_labels_metadata=root / "fit-labels.json",
        checkpoint=root / "union-hard-edge-priority.pt",
        frozen_eval=root / "frozen-target-free-eval.npz",
        frozen_eval_metadata=root / "frozen-target-free-eval.json",
        report=root / "report.json",
    )
    if any(path.exists() for path in paths.__dict__.values()):
        raise FileExistsError("refusing to overwrite a Union hard-edge pilot run")
    return paths


def _target_free_case(case: Any) -> TargetFreeCase:
    return TargetFreeCase(
        case_id=str(case.case_id),
        source_filename=str(case.source_filename),
        dirty_tiles=np.asarray(case.dirty_tiles),
    )


def _strict_layout(value: Any) -> np.ndarray:
    layout = np.ascontiguousarray(value, dtype=np.int32)
    if layout.shape != (COUNT,) or not np.array_equal(np.sort(layout), np.arange(COUNT)):
        raise ValueError("layout is not a strict original-tile permutation")
    return layout


def _decode_layout(
    right: np.ndarray,
    down: np.ndarray,
    *,
    component_edge_priority: Mapping[str, Any] | None,
) -> np.ndarray:
    decoder = decode_socket_assignments(
        right,
        down,
        grid=GRID,
        config=SocketDecoderConfig(
            component_edge_budget_per_axis=DECODER_EDGE_BUDGET,
            swap_edge_budget_per_axis=DECODER_EDGE_BUDGET,
            max_swap_steps=DECODER_SWAP_STEPS,
        ),
        component_edge_priority=component_edge_priority,
    )
    cyclic = select_global_cyclic_translation(
        decoder.layout,
        right,
        down,
        grid=GRID,
        config=CyclicTranslationConfig(border_weight=CYCLIC_BORDER_WEIGHT),
    )
    return _strict_layout(cyclic.layout)


def _tile_tensor(tiles: np.ndarray, *, device: torch.device) -> torch.Tensor:
    value = np.asarray(tiles)
    if value.shape != (COUNT, 20, 20, 3) or value.dtype != np.uint8:
        raise ValueError("dirty tiles violate the 576x20x20 RGB contract")
    return (
        torch.from_numpy(np.ascontiguousarray(value))
        .permute(0, 3, 1, 2)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
        .unsqueeze(0)
    )


def _hard_identities(assignment: Any, *, axis: str) -> tuple[tuple[int, int], ...]:
    matching = hard_partial_axis_matching(assignment, grid=GRID, axis=axis)
    return tuple(sorted((edge.source, edge.target) for edge in matching.edges))


def _load_models_from_commitment(
    commitment: Mapping[str, Any],
    *,
    device: torch.device,
) -> FrozenModels:
    frozen = commitment["frozen_inputs"]

    def path(name: str) -> Path:
        return _resolve_path(frozen[name], name=name)

    d2_path = path("d2_config")
    d2_config, d2_sha = load_d2_config(d2_path)
    validate_frozen_inputs(d2_config)
    d1_path = PROJECT_ROOT / str(d2_config["frozen_inputs"]["fusion_preregistration"])
    d1_config, d1_sha = _load_fusion_preregistration(d1_path)
    socket, relation, denoiser, fusion_dependencies = _load_fusion_dependencies(
        d1_config,
        device=device,
    )
    fusion = _load_fusion(d2_config, device=device)
    twin = load_fullres_twin_checkpoint(path("twin_checkpoint"), device=device)
    union = load_raw_twin_union_checkpoint(
        path("union_checkpoint"),
        config_path=path("union_config"),
        selection_path=path("union_selection"),
        device=device,
    )
    direct = load_direct_hard_edge_checkpoint(path("direct_checkpoint"), device=device)
    if twin.sha256 != FROZEN_TWIN_SHA256:
        raise ValueError("Twin checkpoint identity changed")
    if union.sha256 != FROZEN_UNION_CHECKPOINT_SHA256:
        raise ValueError("Union-v2 checkpoint identity changed")
    if direct.sha256 != FROZEN_DIRECT_HARD_EDGE_SHA256:
        raise ValueError("Direct hard-edge checkpoint identity changed")
    return FrozenModels(
        socket=socket,
        relation=relation,
        denoiser=denoiser,
        fusion=fusion,
        twin=twin,
        union=union,
        direct=direct,
        d2_config=d2_config,
        metadata={
            "d2_config_sha256": d2_sha,
            "d1_fusion_config_sha256": d1_sha,
            "fusion_dependencies": fusion_dependencies,
            "twin_checkpoint_sha256": twin.sha256,
            "union_checkpoint_sha256": union.sha256,
            "direct_checkpoint_sha256": direct.sha256,
        },
    )


@torch.inference_mode()
def _prepare_target_free_board(
    case: TargetFreeCase,
    models: FrozenModels,
    *,
    device: torch.device,
    inference_batch: int,
    assert_production_parity: bool = False,
) -> tuple[UnionHardEdgeBoard, np.ndarray, np.ndarray, dict[str, Any]]:
    started = perf_counter()
    candidate_contract = models.d2_config["candidate_and_decoder"]
    fusion_board = prepare_fusion_board(
        case,
        socket=models.socket,
        relation=models.relation,
        denoiser=models.denoiser,
        device=device,
        inference_batch=inference_batch,
        raw_topk=int(candidate_contract["raw_proposal_topk_per_exposed_member"]),
        raw_cap=int(candidate_contract["raw_candidate_cap_per_query"]),
        union_cap=int(candidate_contract["union_candidate_cap_per_query"]),
        attach_exact_labels=False,
    )
    if fusion_board.union_labels or fusion_board.oracle_relations or fusion_board.profiles:
        raise RuntimeError("exact labels entered target-free fusion preparation")
    fusion_features = torch.from_numpy(fusion_board.features).to(device)
    relation_scores = torch.from_numpy(fusion_board.frozen_relation_scores).to(device)
    fusion_output = models.fusion(fusion_features, relation_scores)

    tensor = _tile_tensor(case.dirty_tiles, device=device)
    tokens, socket_output = extract_frozen_socket_context(
        models.socket.model,
        tensor,
        grid=GRID,
    )
    twin_output = models.twin.model(tensor)
    union_board = prepare_raw_twin_union_board(
        tokens[0],
        socket_output,
        twin_output,
        grid=GRID,
        topk=int(models.union.contract["raw_topk"]),
    )
    union_output = models.union.model(union_board)
    manual_right, manual_down = restricted_partial_ot(
        union_board,
        union_output.scores,
        socket_output,
        iterations=int(models.socket.contract["sinkhorn_iterations"]),
    )
    right = np.ascontiguousarray(manual_right[0].float().cpu().numpy(), dtype=np.float32)
    down = np.ascontiguousarray(manual_down[0].float().cpu().numpy(), dtype=np.float32)
    parity_report: dict[str, Any] = {"production_parity_checked": False}
    if assert_production_parity:
        inference = infer_raw_twin_union_assignments(
            case.dirty_tiles,
            models.socket,
            models.twin,
            models.union,
            device=device,
        )
        for name, reconstructed, frozen in (
            ("right", right, inference.learned_right_log_assignment),
            ("down", down, inference.learned_down_log_assignment),
        ):
            if _hard_identities(reconstructed, axis=name) != _hard_identities(
                frozen,
                axis=name,
            ):
                raise RuntimeError("reconstructed Union board changed the frozen hard projection")
        parity_report = {
            "production_parity_checked": True,
            "production_inference": inference.report(),
        }

    direct_features = extract_hard_edge_features(
        right_log_assignment=socket_output.right_log_assignment[0],
        down_log_assignment=socket_output.down_log_assignment[0],
        right_raw=socket_output.right_raw[0],
        down_raw=socket_output.down_raw[0],
        grid=GRID,
    )
    direct_board = prepare_direct_hard_edge_board(
        tokens[0],
        direct_features,
        socket_output,
        grid=GRID,
        provisional_edge_budget_per_axis=int(
            models.direct.contract["provisional_raw_edge_budget_per_axis"]
        ),
    )
    direct_scores = models.direct.model(
        direct_board.values,
        direct_board.raw_priority,
        direct_board.axis,
    )
    fusion_priority = build_fullres_fusion_union_priority(
        right,
        down,
        fusion_board.union_candidates,
        fusion_output.scores,
        fusion_output.confidence_logits,
        grid=GRID,
        config=FusionUnionPriorityConfig(
            query_cap=FUSION_QUERY_CAP,
            candidate_rank_cap=FUSION_CANDIDATE_RANK_CAP,
            boost_scale=FUSION_BOOST_SCALE,
        ),
    )
    board = prepare_union_hard_edge_board(
        tokens[0],
        union_board,
        union_output.scores,
        socket_output,
        right,
        down,
        grid=GRID,
        edge_budget_per_axis=DECODER_EDGE_BUDGET,
        provisional_edge_budget_per_axis=int(
            models.direct.contract["provisional_raw_edge_budget_per_axis"]
        ),
        direct_board=direct_board,
        direct_scores=direct_scores,
        fullres_priority=fusion_priority.component_edge_priority,
    )
    return (
        board,
        right,
        down,
        {
            "case_total": perf_counter() - started,
            "fusion_board": fusion_board.runtime_seconds,
            "union_inference": parity_report,
            "fullres_priority": fusion_priority.report(),
            "direct_matches_per_axis": list(board.direct_matches_per_axis),
            "fullres_supported_per_axis": list(board.fullres_supported_per_axis),
        },
    )


def _case_specs(
    commitment: Mapping[str, Any],
    lookup: Mapping[str, Mapping[str, Any]],
    *,
    role: str,
) -> list[tuple[Mapping[str, Any], str, int]]:
    section = commitment[role]
    source_names = tuple(section["source_filenames"])
    draws = tuple(int(draw) for draw in section["draws"])
    if draws != DRAWS or _cases_digest(source_names, draws) != section["cases_digest"]:
        raise RuntimeError(f"{role} source/draw digest differs from commitment")
    result = [(lookup[name], name, int(draw)) for name in source_names for draw in draws]
    if len(result) != int(section["case_count"]):
        raise RuntimeError(f"{role} case expansion differs from commitment")
    return result


def _cache_paths(cache_dir: Path, role: str) -> dict[str, Path]:
    return {
        name: cache_dir / f"{role}-{name}.npy"
        for name in (
            "values",
            "base",
            "scale",
            "axis",
            "source",
            "target",
            "right-assignment",
            "down-assignment",
            "direct-matches",
            "fullres-supported",
        )
    }


def _build_target_free_cache(
    paths: RunPaths,
    specs_by_role: Mapping[str, list[tuple[Mapping[str, Any], str, int]]],
    commitment: Mapping[str, Any],
    models: FrozenModels,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> Path:
    paths.cache_dir.mkdir(parents=False, exist_ok=False)
    rows_by_role: dict[str, list[dict[str, Any]]] = {}
    target_cache = CleanTileCache(args.targets)
    for role in ("fit", "eval"):
        specs = specs_by_role[role]
        cache_paths = _cache_paths(paths.cache_dir, role)
        shape = (len(specs), HARD_EDGE_COUNT)
        arrays = {
            "values": np.lib.format.open_memmap(
                cache_paths["values"],
                mode="w+",
                dtype=np.float32,
                shape=(*shape, len(FEATURE_NAMES)),
            ),
            "base": np.lib.format.open_memmap(
                cache_paths["base"], mode="w+", dtype=np.float32, shape=shape
            ),
            "scale": np.lib.format.open_memmap(
                cache_paths["scale"], mode="w+", dtype=np.float32, shape=shape
            ),
            "axis": np.lib.format.open_memmap(
                cache_paths["axis"], mode="w+", dtype=np.int8, shape=shape
            ),
            "source": np.lib.format.open_memmap(
                cache_paths["source"], mode="w+", dtype=np.int16, shape=shape
            ),
            "target": np.lib.format.open_memmap(
                cache_paths["target"], mode="w+", dtype=np.int16, shape=shape
            ),
            "right-assignment": np.lib.format.open_memmap(
                cache_paths["right-assignment"],
                mode="w+",
                dtype=np.float32,
                shape=(len(specs), COUNT + 1, COUNT + 1),
            ),
            "down-assignment": np.lib.format.open_memmap(
                cache_paths["down-assignment"],
                mode="w+",
                dtype=np.float32,
                shape=(len(specs), COUNT + 1, COUNT + 1),
            ),
            "direct-matches": np.lib.format.open_memmap(
                cache_paths["direct-matches"],
                mode="w+",
                dtype=np.int16,
                shape=(len(specs), 2),
            ),
            "fullres-supported": np.lib.format.open_memmap(
                cache_paths["fullres-supported"],
                mode="w+",
                dtype=np.int16,
                shape=(len(specs), 2),
            ),
        }
        rows: list[dict[str, Any]] = []
        for index, (record, source_name, draw) in enumerate(specs):
            case = prepare_case(
                target_cache,
                record,
                draw_index=draw,
                seed=int(commitment["synthetic_seed"]),
            )
            matcher_case = _target_free_case(case)
            board, right, down, diagnostics = _prepare_target_free_board(
                matcher_case,
                models,
                device=device,
                inference_batch=args.inference_batch,
                assert_production_parity=False,
            )
            float_values = board.values.float().cpu().numpy()
            stored_values = np.asarray(float_values, dtype=np.float32)
            if not np.isfinite(stored_values).all():
                raise RuntimeError(f"non-finite features for {role} case {case.case_id}")
            arrays["values"][index] = stored_values
            arrays["base"][index] = board.base_priority.float().cpu().numpy()
            arrays["scale"][index] = board.priority_scale.float().cpu().numpy()
            arrays["axis"][index] = board.axis.cpu().numpy()
            arrays["source"][index] = board.source
            arrays["target"][index] = board.target
            arrays["right-assignment"][index] = right
            arrays["down-assignment"][index] = down
            arrays["direct-matches"][index] = board.direct_matches_per_axis
            arrays["fullres-supported"][index] = board.fullres_supported_per_axis
            rows.append(
                {
                    "index": index,
                    "case_id": case.case_id,
                    "source_filename": source_name,
                    "draw_index": draw,
                    "dirty_sha256": _dirty_sha256(case.dirty_tiles),
                    "runtime": diagnostics,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "cache_target_free",
                        "role": role,
                        "done": index + 1,
                        "total": len(specs),
                    }
                ),
                flush=True,
            )
        for array in arrays.values():
            array.flush()
        del arrays
        rows_by_role[role] = rows
    metadata_path = paths.cache_dir / "metadata.json"
    files = {
        role: {
            name: {
                "path": _project_path(path),
                "sha256": sha256_file(path),
            }
            for name, path in _cache_paths(paths.cache_dir, role).items()
        }
        for role in ("fit", "eval")
    }
    _write_json_exclusive(
        metadata_path,
        {
            "schema": "aiijc-union-hard-edge-target-free-cache-v1",
            "contains_exact_references_or_labels": False,
            "contains_pixels": False,
            "feature_names": list(FEATURE_NAMES),
            "feature_dimension": len(FEATURE_NAMES),
            "hard_edge_count": HARD_EDGE_COUNT,
            "values_storage_dtype": "float32",
            "files": files,
            "cases": rows_by_role,
        },
    )
    return metadata_path


def _cached_board(
    cache_dir: Path,
    role: str,
    index: int,
    *,
    device: torch.device,
) -> UnionHardEdgeBoard:
    paths = _cache_paths(cache_dir, role)
    arrays = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
    return UnionHardEdgeBoard(
        values=torch.from_numpy(np.asarray(arrays["values"][index], dtype=np.float32).copy()).to(
            device
        ),
        base_priority=torch.from_numpy(
            np.asarray(arrays["base"][index], dtype=np.float32).copy()
        ).to(device),
        priority_scale=torch.from_numpy(
            np.asarray(arrays["scale"][index], dtype=np.float32).copy()
        ).to(device),
        axis=torch.from_numpy(np.asarray(arrays["axis"][index], dtype=np.int64).copy()).to(device),
        source=np.asarray(arrays["source"][index], dtype=np.int32),
        target=np.asarray(arrays["target"][index], dtype=np.int32),
        grid=GRID,
        edge_budget_per_axis=DECODER_EDGE_BUDGET,
        direct_matches_per_axis=tuple(int(value) for value in arrays["direct-matches"][index]),
        fullres_supported_per_axis=tuple(
            int(value) for value in arrays["fullres-supported"][index]
        ),
    )


def _fixed_budget_label_correct(
    scores: Any,
    labels: Any,
    axis: Any,
) -> int:
    priority = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=bool)
    axes = np.asarray(axis, dtype=np.int8)
    if (
        priority.shape != (HARD_EDGE_COUNT,)
        or truth.shape != priority.shape
        or axes.shape != priority.shape
        or not np.isfinite(priority).all()
    ):
        raise ValueError("fixed-budget fit vectors violate the hard-edge contract")
    total = 0
    for axis_index in (0, 1):
        indices = np.flatnonzero(axes == axis_index)
        if len(indices) != HARD_EDGES_PER_AXIS:
            raise ValueError("fixed-budget fit axis cardinality changed")
        order = np.argsort(-priority[indices], kind="stable")[:DECODER_EDGE_BUDGET]
        total += int(np.count_nonzero(truth[indices[order]]))
    return total


def _attach_fit_labels(
    paths: RunPaths,
    fit_specs: list[tuple[Mapping[str, Any], str, int]],
    commitment: Mapping[str, Any],
    args: argparse.Namespace,
    cache_metadata_path: Path,
) -> None:
    metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
    rows = metadata["cases"]["fit"]
    source = np.load(_cache_paths(paths.cache_dir, "fit")["source"], mmap_mode="r")
    target = np.load(_cache_paths(paths.cache_dir, "fit")["target"], mmap_mode="r")
    axis = np.load(_cache_paths(paths.cache_dir, "fit")["axis"], mmap_mode="r")
    labels = np.lib.format.open_memmap(
        paths.fit_labels,
        mode="w+",
        dtype=bool,
        shape=(len(fit_specs), HARD_EDGE_COUNT),
    )
    target_cache = CleanTileCache(args.targets)
    positive_per_case: list[tuple[int, int]] = []
    for index, ((record, source_name, draw), frozen) in enumerate(
        zip(fit_specs, rows, strict=True)
    ):
        case = prepare_case(
            target_cache,
            record,
            draw_index=draw,
            seed=int(commitment["synthetic_seed"]),
        )
        if (
            case.case_id != frozen["case_id"]
            or source_name != frozen["source_filename"]
            or _dirty_sha256(case.dirty_tiles) != frozen["dirty_sha256"]
        ):
            raise RuntimeError("fit-label phase recreated a different synthetic case")
        proxy = HardEdgeFeatures(
            values=np.zeros((HARD_EDGE_COUNT, 20), dtype=np.float32),
            source=np.asarray(source[index], dtype=np.int32),
            target=np.asarray(target[index], dtype=np.int32),
            axis=np.asarray(axis[index], dtype=np.int8),
        )
        reference = _strict_layout(np.argsort(case.input_tile_to_position))
        case_labels = exact_edge_labels(proxy, reference, grid=GRID)
        counts = tuple(
            int(np.count_nonzero(case_labels[np.asarray(axis[index]) == axis_index]))
            for axis_index in (0, 1)
        )
        if any(count <= 0 or count >= HARD_EDGES_PER_AXIS for count in counts):
            raise RuntimeError(
                f"fit case {case.case_id} lacks positive/negative hard edges on an axis"
            )
        positive_per_case.append(counts)
        labels[index] = case_labels
    labels.flush()
    del labels
    _write_json_exclusive(
        paths.fit_labels_metadata,
        {
            "schema": "aiijc-union-hard-edge-fit-labels-v1",
            "created_after_complete_target_free_fit_and_eval_cache": True,
            "target_free_cache_metadata_sha256": sha256_file(cache_metadata_path),
            "labels_path": _project_path(paths.fit_labels),
            "labels_sha256": sha256_file(paths.fit_labels),
            "case_count": len(fit_specs),
            "exact_references_persisted": False,
            "positive_count_preflight": {
                "every_case_has_positive_and_negative_edges_per_axis": True,
                "minimum_per_axis": [
                    min(value[axis_index] for value in positive_per_case) for axis_index in (0, 1)
                ],
                "mean_per_axis": [
                    float(np.mean([value[axis_index] for value in positive_per_case]))
                    for axis_index in (0, 1)
                ],
                "maximum_per_axis": [
                    max(value[axis_index] for value in positive_per_case) for axis_index in (0, 1)
                ],
            },
        },
    )


def _train(
    paths: RunPaths,
    commitment: Mapping[str, Any],
    models: FrozenModels,
    *,
    device: torch.device,
    log_every: int,
) -> tuple[UnionHardEdgePriority, list[dict[str, Any]], float, dict[str, Any]]:
    labels = np.load(paths.fit_labels, mmap_mode="r")
    model = UnionHardEdgePriority(
        hidden_dimension=HIDDEN_DIMENSION,
        residual_limit=RESIDUAL_LIMIT,
    ).to(device)
    first = _cached_board(paths.cache_dir, "fit", 0, device=device)
    with torch.inference_mode():
        if not torch.equal(model(first).scores, first.base_priority):
            raise RuntimeError("zero-init head does not reproduce Union priorities")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    generator = np.random.default_rng(int(commitment["synthetic_seed"]) + 90_001)
    order = generator.permutation(FIT_CASE_COUNT)
    cursor = 0
    history: list[dict[str, Any]] = []
    started = perf_counter()
    model.train()
    for step in range(TRAINING_STEPS):
        if cursor == len(order):
            order = generator.permutation(FIT_CASE_COUNT)
            cursor = 0
        case_index = int(order[cursor])
        cursor += 1
        board = _cached_board(paths.cache_dir, "fit", case_index, device=device)
        truth = torch.from_numpy(np.asarray(labels[case_index], dtype=bool)).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(board)
        loss, diagnostics = union_hard_edge_listwise_loss(
            output,
            board,
            truth,
            pairwise_weight=PAIRWISE_WEIGHT,
            residual_weight=RESIDUAL_WEIGHT,
        )
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        if not bool(torch.isfinite(gradient).item()):
            raise RuntimeError("non-finite Union hard-edge gradient")
        optimizer.step()
        row = {
            "step": step + 1,
            "case_index": case_index,
            **diagnostics,
            "gradient_norm": float(gradient.detach().cpu()),
        }
        history.append(row)
        if (step + 1) % log_every == 0 or step == 0:
            print(json.dumps({"event": "train", **row}), flush=True)
    runtime = perf_counter() - started
    model.eval()
    fit_fixed_rows: list[dict[str, int]] = []
    with torch.inference_mode():
        for case_index in range(FIT_CASE_COUNT):
            board = _cached_board(paths.cache_dir, "fit", case_index, device=device)
            output = model(board)
            truth = np.asarray(labels[case_index], dtype=bool)
            axes = board.axis.cpu().numpy()
            baseline_correct = _fixed_budget_label_correct(
                board.base_priority.float().cpu().numpy(),
                truth,
                axes,
            )
            learned_correct = _fixed_budget_label_correct(
                output.scores.float().cpu().numpy(),
                truth,
                axes,
            )
            fit_fixed_rows.append(
                {
                    "baseline_correct": baseline_correct,
                    "learned_correct": learned_correct,
                    "delta": learned_correct - baseline_correct,
                }
            )
    fit_fixed_diagnostic = {
        "schema": "aiijc-union-hard-edge-fit-fixed-top288-diagnostic-v1",
        "case_count": FIT_CASE_COUNT,
        "baseline_mean_correct": float(
            np.mean([row["baseline_correct"] for row in fit_fixed_rows])
        ),
        "learned_mean_correct": float(np.mean([row["learned_correct"] for row in fit_fixed_rows])),
        "mean_delta": float(np.mean([row["delta"] for row in fit_fixed_rows])),
        "improved_cases": int(sum(row["delta"] > 0 for row in fit_fixed_rows)),
        "tied_cases": int(sum(row["delta"] == 0 for row in fit_fixed_rows)),
        "worsened_cases": int(sum(row["delta"] < 0 for row in fit_fixed_rows)),
        "used_for_hyperparameter_or_arm_selection": False,
    }
    torch.save(
        {
            "schema": "aiijc-union-hard-edge-priority-checkpoint-v1",
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "contract": {
                "architecture": "union-hard-edge-deepsets-bounded-residual-v1",
                "feature_names": list(FEATURE_NAMES),
                "feature_dimension": len(FEATURE_NAMES),
                "hidden_dimension": model.hidden_dimension,
                "residual_limit": model.residual_limit,
                "hard_edge_count": HARD_EDGE_COUNT,
                "edge_budget_per_axis": DECODER_EDGE_BUDGET,
                "training_steps": TRAINING_STEPS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "pairwise_weight": PAIRWISE_WEIGHT,
                "residual_weight": RESIDUAL_WEIGHT,
                "gradient_clip": GRADIENT_CLIP,
                "feature_cache_dtype": "float32",
                "fullres_priority": {
                    "query_cap": FUSION_QUERY_CAP,
                    "candidate_rank_cap": FUSION_CANDIDATE_RANK_CAP,
                    "boost_scale": FUSION_BOOST_SCALE,
                },
                "pixel_prediction": False,
            },
            "selection_commitment_sha256": sha256_file(
                paths.checkpoint.parent / "selection-commitment.json"
            ),
            "fit_labels_sha256": sha256_file(paths.fit_labels),
            "frozen_lineage": models.metadata,
            "fit_fixed_top288_diagnostic": fit_fixed_diagnostic,
            "history": history,
        },
        paths.checkpoint,
    )
    return model, history, runtime, fit_fixed_diagnostic


def _freeze_eval(
    paths: RunPaths,
    model: UnionHardEdgePriority,
    eval_specs: list[tuple[Mapping[str, Any], str, int]],
    cache_metadata_path: Path,
    *,
    device: torch.device,
) -> tuple[str, str]:
    cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
    cache_rows = cache_metadata["cases"]["eval"]
    right_cache = np.load(_cache_paths(paths.cache_dir, "eval")["right-assignment"], mmap_mode="r")
    down_cache = np.load(_cache_paths(paths.cache_dir, "eval")["down-assignment"], mmap_mode="r")
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    strict_count = 0
    with torch.inference_mode():
        for index, ((_, source_name, draw), cached) in enumerate(
            zip(eval_specs, cache_rows, strict=True)
        ):
            board = _cached_board(paths.cache_dir, "eval", index, device=device)
            output = model(board)
            priorities = union_hard_edge_priority_matrices(board, output.scores)
            right = np.asarray(right_cache[index], dtype=np.float32)
            down = np.asarray(down_cache[index], dtype=np.float32)
            axis_np = board.axis.cpu().numpy()
            for axis_index, axis_name, assignment in (
                (0, "right", right),
                (1, "down", down),
            ):
                selected = axis_np == axis_index
                board_identities = tuple(
                    sorted(
                        zip(
                            board.source[selected].tolist(),
                            board.target[selected].tolist(),
                            strict=True,
                        )
                    )
                )
                if board_identities != _hard_identities(assignment, axis=axis_name):
                    raise RuntimeError(f"eval cache hard-edge identities drifted for {axis_name}")
            baseline = _decode_layout(right, down, component_edge_priority=None)
            treatment = _decode_layout(
                right,
                down,
                component_edge_priority=priorities,
            )
            strict_count += int(
                np.array_equal(np.sort(baseline), np.arange(COUNT))
                and np.array_equal(np.sort(treatment), np.arange(COUNT))
            )
            prefix = f"case_{index:04d}"
            arrays[f"{prefix}__union_v2_layout"] = baseline
            arrays[f"{prefix}__learned_priority_layout"] = treatment
            learned_np = output.scores.float().cpu().numpy()
            base_np = board.base_priority.float().cpu().numpy()
            for axis_index in (0, 1):
                selected = axis_np == axis_index
                arrays[f"{prefix}__axis_{axis_index}_source"] = board.source[selected]
                arrays[f"{prefix}__axis_{axis_index}_target"] = board.target[selected]
                arrays[f"{prefix}__axis_{axis_index}_baseline_priority"] = base_np[selected]
                arrays[f"{prefix}__axis_{axis_index}_learned_priority"] = learned_np[selected]
            rows.append(
                {
                    "prefix": prefix,
                    "case_id": cached["case_id"],
                    "source_filename": source_name,
                    "draw_index": draw,
                    "dirty_sha256": cached["dirty_sha256"],
                }
            )
    if strict_count != len(eval_specs):
        raise RuntimeError("eval layout freeze produced a non-strict tile permutation")
    np.savez_compressed(paths.frozen_eval, **arrays)
    _write_json_exclusive(
        paths.frozen_eval_metadata,
        {
            "schema": "aiijc-union-hard-edge-frozen-target-free-eval-v1",
            "checkpoint_sha256": sha256_file(paths.checkpoint),
            "target_free_cache_metadata_sha256": sha256_file(cache_metadata_path),
            "contains_exact_references_or_labels": False,
            "contains_strict_original_tile_layouts": True,
            "contains_frozen_hard_priorities": True,
            "strict_layout_count_per_arm": strict_count,
            "rows": rows,
        },
    )
    return sha256_file(paths.frozen_eval), sha256_file(paths.frozen_eval_metadata)


def _edge_truth(
    source: np.ndarray,
    target: np.ndarray,
    *,
    axis: int,
    reference: np.ndarray,
) -> np.ndarray:
    position = np.empty(COUNT, dtype=np.int32)
    position[reference] = np.arange(COUNT, dtype=np.int32)
    source_position = position[source]
    target_position = position[target]
    if axis == 0:
        return (target_position == source_position + 1) & (source_position % GRID != GRID - 1)
    if axis == 1:
        return target_position == source_position + GRID
    raise ValueError("axis must be 0 or 1")


def _fixed_top288_correct(
    archive: Mapping[str, Any],
    prefix: str,
    reference: np.ndarray,
    *,
    arm: str,
) -> int:
    if arm not in {"baseline", "learned"}:
        raise ValueError("arm must be baseline or learned")
    total = 0
    for axis in (0, 1):
        source = np.asarray(archive[f"{prefix}__axis_{axis}_source"], dtype=np.int32)
        target = np.asarray(archive[f"{prefix}__axis_{axis}_target"], dtype=np.int32)
        priority = np.asarray(archive[f"{prefix}__axis_{axis}_{arm}_priority"], dtype=np.float64)
        if (
            source.shape != (HARD_EDGES_PER_AXIS,)
            or target.shape != source.shape
            or priority.shape != source.shape
            or not np.isfinite(priority).all()
        ):
            raise ValueError("frozen arrays violate the fixed-top288 contract")
        order = np.argsort(-priority, kind="stable")[:DECODER_EDGE_BUDGET]
        total += int(
            np.count_nonzero(_edge_truth(source, target, axis=axis, reference=reference)[order])
        )
    return total


def source_clustered_delta_ci(
    values: Sequence[float],
    sources: Sequence[str],
    *,
    seed: int,
    resamples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    if len(values) != len(sources) or len(values) == 0:
        raise ValueError("values and sources must be aligned and non-empty")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, source in zip(values, sources, strict=True):
        if not math.isfinite(float(value)):
            raise ValueError("clustered delta values must be finite")
        grouped[str(source)].append(float(value))
    if any(len(group) != len(DRAWS) for group in grouped.values()):
        raise ValueError("every source cluster must contain exactly both registered draws")
    source_means = np.asarray([np.mean(grouped[name]) for name in sorted(grouped)])
    generator = np.random.default_rng(seed)
    distribution = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 2048):
        stop = min(start + 2048, resamples)
        indices = generator.integers(0, len(source_means), size=(stop - start, len(source_means)))
        distribution[start:stop] = source_means[indices].mean(axis=1)
    return {
        "mean": float(np.mean(values)),
        "source_cluster_mean": float(source_means.mean()),
        "ci95_lower": float(np.quantile(distribution, 0.025)),
        "ci95_upper": float(np.quantile(distribution, 0.975)),
        "source_count": len(source_means),
        "case_count": len(values),
        "wins": int(np.count_nonzero(source_means > 0)),
        "ties": int(np.count_nonzero(source_means == 0)),
        "losses": int(np.count_nonzero(source_means < 0)),
    }


def _validate_pre_score_hashes(
    paths: RunPaths,
    expected_hashes: Mapping[str, str],
) -> None:
    frozen_paths = {
        "target_free_cache_metadata": paths.cache_dir / "metadata.json",
        "checkpoint": paths.checkpoint,
        "frozen_eval": paths.frozen_eval,
        "frozen_eval_metadata": paths.frozen_eval_metadata,
    }
    if set(expected_hashes) != set(frozen_paths):
        raise ValueError("pre-score frozen hash roster changed")
    for name, path in frozen_paths.items():
        if sha256_file(path) != expected_hashes[name]:
            raise RuntimeError(f"{name} changed before exact-reference scoring")


def _score_frozen_eval(
    paths: RunPaths,
    eval_specs: list[tuple[Mapping[str, Any], str, int]],
    commitment: Mapping[str, Any],
    args: argparse.Namespace,
    expected_hashes: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_pre_score_hashes(paths, expected_hashes)
    frozen_metadata = json.loads(paths.frozen_eval_metadata.read_text(encoding="utf-8"))
    rows = frozen_metadata["rows"]
    scored: list[dict[str, Any]] = []
    target_cache = CleanTileCache(args.targets)
    with np.load(paths.frozen_eval) as archive:
        for (record, source_name, draw), frozen in zip(eval_specs, rows, strict=True):
            case = prepare_case(
                target_cache,
                record,
                draw_index=draw,
                seed=int(commitment["synthetic_seed"]),
            )
            if (
                case.case_id != frozen["case_id"]
                or source_name != frozen["source_filename"]
                or _dirty_sha256(case.dirty_tiles) != frozen["dirty_sha256"]
            ):
                raise RuntimeError("eval scoring recreated a different synthetic case")
            reference = _strict_layout(np.argsort(case.input_tile_to_position))
            prefix = frozen["prefix"]
            baseline_layout = _strict_layout(archive[f"{prefix}__union_v2_layout"])
            learned_layout = _strict_layout(archive[f"{prefix}__learned_priority_layout"])
            baseline = evaluate_layout(
                baseline_layout, reference, reference_is_exact=True
            ).as_dict()
            learned = evaluate_layout(learned_layout, reference, reference_is_exact=True).as_dict()
            scored.append(
                {
                    "source_filename": source_name,
                    "draw_index": draw,
                    "case_id": case.case_id,
                    "union_v2": {
                        "exact_tiles": int(baseline["correct_tile_count"]),
                        "adjacency": float(baseline["adjacency"]),
                        "fixed_top288_correct": _fixed_top288_correct(
                            archive, prefix, reference, arm="baseline"
                        ),
                    },
                    "learned_priority": {
                        "exact_tiles": int(learned["correct_tile_count"]),
                        "adjacency": float(learned["adjacency"]),
                        "fixed_top288_correct": _fixed_top288_correct(
                            archive, prefix, reference, arm="learned"
                        ),
                    },
                }
            )

    metrics = ("fixed_top288_correct", "adjacency", "exact_tiles")

    def deltas(metric: str) -> list[float]:
        return [
            float(row["learned_priority"][metric]) - float(row["union_v2"][metric])
            for row in scored
        ]

    sources = [str(row["source_filename"]) for row in scored]
    summary = {
        "arms": {
            arm: {
                metric: float(np.mean([float(row[arm][metric]) for row in scored]))
                for metric in metrics
            }
            for arm in ("union_v2", "learned_priority")
        },
        "deltas": {
            metric: source_clustered_delta_ci(
                deltas(metric),
                sources,
                seed=BOOTSTRAP_SEED + index,
            )
            for index, metric in enumerate(metrics)
        },
    }
    return scored, summary


def _benchmark_one_case(
    args: argparse.Namespace,
    commitment: Mapping[str, Any],
    lookup: Mapping[str, Mapping[str, Any]],
    models: FrozenModels,
    *,
    device: torch.device,
) -> None:
    path = args.output_dir.resolve() / "benchmark-one-case.json"
    if path.exists():
        raise FileExistsError("refusing to overwrite a one-case benchmark")
    record, source_name, draw = _case_specs(commitment, lookup, role="fit")[0]
    cache = CleanTileCache(args.targets)
    case = prepare_case(
        cache,
        record,
        draw_index=draw,
        seed=int(commitment["synthetic_seed"]),
    )
    matcher_case = _target_free_case(case)
    started = perf_counter()
    board, _, _, diagnostics = _prepare_target_free_board(
        matcher_case,
        models,
        device=device,
        inference_batch=args.inference_batch,
        assert_production_parity=True,
    )
    _write_json_exclusive(
        path,
        {
            "schema": "aiijc-union-hard-edge-priority-one-case-benchmark-v1",
            "source_filename": source_name,
            "draw_index": draw,
            "device": str(device),
            "nondeterministic_mps_explicitly_allowed": bool(args.allow_nondeterministic_mps),
            "feature_shape": list(board.values.shape),
            "target_or_exact_label_attached": False,
            "elapsed_seconds": perf_counter() - started,
            "diagnostics": diagnostics,
        },
    )


def run(args: argparse.Namespace) -> None:
    if args.inference_batch != INFERENCE_BATCH:
        raise ValueError(f"pilot inference-batch must remain {INFERENCE_BATCH}")
    if args.log_every <= 0:
        raise ValueError("log-every must be positive")
    if not args.benchmark_one_case and args.device != "mps":
        raise ValueError("the preregistered full pilot requires MPS")
    config, _ = _load_config(args.config)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    lookup, fit_names, eval_names = _manifest_records(config, manifest)
    commitment = _load_commitment(args.output_dir, args.config, args.manifest)
    if (
        tuple(commitment["fit"]["source_filenames"]) != fit_names
        or tuple(commitment["eval"]["source_filenames"]) != eval_names
    ):
        raise ValueError("config roster differs from selection commitment")
    device = _select_device(
        args.device,
        allow_nondeterministic_mps=args.allow_nondeterministic_mps,
    )
    random.seed(int(commitment["synthetic_seed"]))
    np.random.seed(int(commitment["synthetic_seed"]))
    torch.manual_seed(int(commitment["synthetic_seed"]))
    models = _load_models_from_commitment(commitment, device=device)
    if args.benchmark_one_case:
        _benchmark_one_case(args, commitment, lookup, models, device=device)
        return

    paths = _prepare_run_paths(args.output_dir)
    specs_by_role = {role: _case_specs(commitment, lookup, role=role) for role in ("fit", "eval")}
    started = perf_counter()
    cache_metadata_path = _build_target_free_cache(
        paths,
        specs_by_role,
        commitment,
        models,
        args,
        device=device,
    )
    print(
        json.dumps(
            {
                "event": "all_target_free_fit_and_eval_features_frozen",
                "metadata_sha256": sha256_file(cache_metadata_path),
            }
        ),
        flush=True,
    )
    cache_metadata_sha = sha256_file(cache_metadata_path)
    _attach_fit_labels(
        paths,
        specs_by_role["fit"],
        commitment,
        args,
        cache_metadata_path,
    )
    model, history, training_seconds, fit_fixed_diagnostic = _train(
        paths,
        commitment,
        models,
        device=device,
        log_every=args.log_every,
    )
    prediction_sha, prediction_metadata_sha = _freeze_eval(
        paths,
        model,
        specs_by_role["eval"],
        cache_metadata_path,
        device=device,
    )
    checkpoint_sha = sha256_file(paths.checkpoint)
    print(
        json.dumps(
            {
                "event": "checkpoint_priorities_and_layouts_frozen_before_eval_scoring",
                "checkpoint_sha256": sha256_file(paths.checkpoint),
                "predictions_sha256": prediction_sha,
            }
        ),
        flush=True,
    )
    scored, metrics = _score_frozen_eval(
        paths,
        specs_by_role["eval"],
        commitment,
        args,
        {
            "target_free_cache_metadata": cache_metadata_sha,
            "checkpoint": checkpoint_sha,
            "frozen_eval": prediction_sha,
            "frozen_eval_metadata": prediction_metadata_sha,
        },
    )
    fixed_delta = float(metrics["deltas"]["fixed_top288_correct"]["mean"])
    adjacency_delta = float(metrics["deltas"]["adjacency"]["mean"])
    exact_delta = float(metrics["deltas"]["exact_tiles"]["mean"])
    gate = {
        "fixed_top288_gain_at_least_half_edge": fixed_delta >= 0.5,
        "adjacency_not_materially_worse": adjacency_delta >= -0.0005,
        "exact_nonnegative": exact_delta >= 0.0,
        "all_layouts_strict": len(scored) == EVAL_CASE_COUNT,
    }
    gate["passed"] = all(gate.values())
    _write_json_exclusive(
        paths.report,
        {
            "schema": "aiijc-union-hard-edge-priority-pilot-report-v1",
            "status": "local-gate-pass" if gate["passed"] else "local-gate-fail",
            "device": {
                "value": str(device),
                "nondeterministic_mps_explicitly_allowed": bool(args.allow_nondeterministic_mps),
                "determinism_claimed": device.type != "mps",
            },
            "selection_commitment": {
                "path": _project_path(args.output_dir / "selection-commitment.json"),
                "sha256": sha256_file(args.output_dir / "selection-commitment.json"),
            },
            "frozen_lineage": models.metadata,
            "target_free_cache": {
                "metadata_path": _project_path(cache_metadata_path),
                "metadata_sha256": sha256_file(cache_metadata_path),
                "fit_and_eval_complete_before_fit_labels": True,
            },
            "checkpoint": {
                "path": _project_path(paths.checkpoint),
                "sha256": sha256_file(paths.checkpoint),
                "steps": TRAINING_STEPS,
                "training_seconds": training_seconds,
                "final_20_loss": float(np.mean([row["loss"] for row in history[-20:]])),
            },
            "fit_fixed_top288_diagnostic": fit_fixed_diagnostic,
            "frozen_eval": {
                "path": _project_path(paths.frozen_eval),
                "sha256": prediction_sha,
                "metadata_path": _project_path(paths.frozen_eval_metadata),
                "metadata_sha256": prediction_metadata_sha,
                "checkpoint_priorities_and_both_layouts_frozen_before_references": True,
                "contains_exact_references_or_labels": False,
            },
            "metrics": metrics,
            "gate": gate,
            "rows": scored,
            "runtime_seconds": perf_counter() - started,
            "legality": {
                "organizer_train_only": True,
                "restored_pixels_matcher_only": True,
                "restored_pixels_emitted": False,
                "new_hard_edges_introduced": False,
                "original_upright_tile_permutations_only": True,
                "hyperparameter_or_arm_sweep": False,
            },
        },
    )


def main() -> None:
    args = parse_args()
    if args.mode == "selection":
        if args.write_config:
            path = write_preregistered_config(args)
            print(
                json.dumps(
                    {
                        "event": "preregistered_config_written",
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "target_accessed": False,
                    }
                ),
                flush=True,
            )
            return
        if args.roster_audit is not None:
            raise ValueError("--roster-audit requires --write-config")
        path = create_selection_commitment(args)
        print(json.dumps({"event": "selection_frozen", "path": str(path)}), flush=True)
    else:
        if args.write_config or args.roster_audit is not None:
            raise ValueError("config generation flags are valid only in selection mode")
        run(args)


if __name__ == "__main__":
    main()
