"""SHA-locked non-default production adapter for frozen raw/twin Union-v2.

The public prediction API accepts only one dirty RGB board and explicitly
loaded frozen models.  It builds target-free Socket d64 and full-resolution
twin evidence, reranks the immutable raw32/twin32/raw-hard union, applies the
frozen restricted partial-OT projection, and decodes with decoder144 plus
cyclic-border5.  Every output is assembled from each original upright tile
exactly once.

Union-v2 remains opt-in.  When neither learned artifact is supplied the adapter
delegates to :func:`predict_socket_sorter` exactly.  A partially configured,
corrupt, or incompatible learned arm fails closed instead of silently falling
back.  No target, manifest, filename, restored tile, absolute tile index, or
competition-test-specific input is accepted by this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import torch

from aiijc_puzzle.component_relation_reranker import extract_frozen_socket_context
from aiijc_puzzle.fullres_twin_side_matcher import FullResolutionTwinSideMatcher
from aiijc_puzzle.protocol import (
    GRID_SIZE,
    IMAGE_SIZE,
    RGB_CHANNELS,
    TILE_COUNT,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.raw_twin_union_reranker import (
    FEATURE_NAMES,
    RawTwinUnionReranker,
    candidate_score_matrices,
    prepare_raw_twin_union_board,
    restricted_partial_ot,
)
from aiijc_puzzle.socket_decoder import (
    SocketDecoderConfig,
    decode_socket_assignments,
    hard_partial_axis_matching,
)
from aiijc_puzzle.socket_matcher import SocketOutput
from aiijc_puzzle.socket_sorter_production import (
    DECODER_EDGE_BUDGET,
    DECODER_SWAP_STEPS,
    IDENTITY_PIXEL_TAIL,
    LoadedSocketCheckpoint,
    SocketSorterPrediction,
    assemble_audited_original_tiles,
    predict_socket_sorter,
)
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)
from aiijc_puzzle.union_fragment_synchronizer import (
    UnionCandidateSnapshot,
    freeze_union_candidate_snapshot,
)

UNION_ADAPTER_SCHEMA = "aiijc-raw-twin-union-production-adapter-v1"
UNION_CHECKPOINT_SCHEMA = "raw-twin-union-reranker-v2"
FROZEN_UNION_CHECKPOINT_SHA256 = (
    "a5f882ab3c827e4e3779be3372c62d2a8fb9cd95d3558fd30cc566a9c3137f79"
)
FROZEN_UNION_CONFIG_SHA256 = (
    "6741e92e832a630f1b83bde6edc8a341a348f52daa82313c40a8f32c7c1173d4"
)
FROZEN_UNION_SELECTION_SHA256 = (
    "71ae4f5095489613857fcd25c541fe496da0d6861f6ff604850147dd04b91cd2"
)
FROZEN_SOCKET_SHA256 = (
    "0e9df49a503c65aac7f1468e9acd6a074a5e658ae8b61f8954be086272c49670"
)
FROZEN_TWIN_SHA256 = (
    "c5b44901e8da459e3c48b6e7af7153c5d7eed26f1c1b52c8712c4fa0dc4ea8ae"
)
FROZEN_UNION_CONTRACT: dict[str, Any] = {
    "architecture": UNION_CHECKPOINT_SCHEMA,
    "feature_dimension": 280,
    "hidden_dimension": 64,
    "residual_limit": 2.0,
    "raw_topk": 32,
    "twin_topk": 32,
    "outside_union_fill": -10000.0,
    "pixel_prediction": False,
}
FROZEN_TWIN_CONTRACT: dict[str, Any] = {
    "architecture": "fullres-ordered-twin-side-matcher-v1",
    "dimension": 48,
    "field_blocks": 4,
    "sequence_blocks": 2,
    "raw_skip_gain": 0.35,
    "parameter_count": 61_970,
    "feature_resolution": [20, 20],
    "spatial_downsampling": False,
    "ordered_side_positions": 20,
    "pixel_prediction_head": False,
    "matcher_only": True,
}
FROZEN_UNION_PARAMETER_COUNT = 54_449
CYCLIC_BORDER_WEIGHT = 5.0


def _array_sha256(value: np.ndarray, *, dtype: str | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _names_digest(names: Sequence[str], *, sort_names: bool = False) -> str:
    values = sorted(names) if sort_names else list(names)
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest") from error
    return value


def _require_regular_file(path: Path) -> Path:
    """Reject symlinks in every existing path component and require a file."""

    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    current = Path(absolute.parts[0])
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError as error:
            raise ValueError(f"artifact path does not exist: {absolute}") from error
        if stat.S_ISLNK(mode):
            raise ValueError(f"symlink artifact paths are forbidden: {current}")
    if not stat.S_ISREG(os.lstat(absolute).st_mode):
        raise ValueError(f"expected a regular artifact file: {absolute}")
    return absolute


def _load_sha_locked_json(path: Path, *, expected_sha256: str, name: str) -> dict[str, Any]:
    expected = _validate_sha256(expected_sha256, name=f"expected_{name}_sha256")
    regular = _require_regular_file(path)
    if sha256_file(regular) != expected:
        raise ValueError(f"{name} SHA-256 mismatch")
    payload = json.loads(regular.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} payload must be a mapping")
    return payload


def _validated_roster(value: Any, *, name: str, expected_count: int) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str)
        and Path(item).name == item
        and item.endswith(".png")
        for item in value
    ):
        raise ValueError(f"{name} roster is malformed")
    result = tuple(value)
    if len(result) != expected_count or len(result) != len(set(result)):
        raise ValueError(f"{name} roster count is invalid")
    return result


@dataclass(frozen=True)
class FrozenFitEvaluationLineage:
    """Content-addressed organizer-train fit/evaluation rosters."""

    fit_count: int
    fit_order_digest: str
    evaluation_count: int
    evaluation_order_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "fit_count": self.fit_count,
            "fit_order_digest": self.fit_order_digest,
            "evaluation_count": self.evaluation_count,
            "evaluation_order_digest": self.evaluation_order_digest,
        }


@dataclass(frozen=True)
class LoadedFullResolutionTwinCheckpoint:
    """Strict frozen full-resolution matcher and source lineage."""

    path: Path
    sha256: str
    model: FullResolutionTwinSideMatcher
    contract: dict[str, Any]
    lineage: FrozenFitEvaluationLineage


def _checkpoint_rosters(
    selection: Any,
    *,
    fit_key: str,
    fit_digest_key: str,
    evaluation_key: str,
    evaluation_digest_key: str,
    fit_count: int,
    evaluation_count: int,
    sort_fit_digest: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...], FrozenFitEvaluationLineage]:
    if not isinstance(selection, Mapping):
        raise ValueError("checkpoint has no selection mapping")
    fit = _validated_roster(selection.get(fit_key), name="fit", expected_count=fit_count)
    evaluation = _validated_roster(
        selection.get(evaluation_key),
        name="evaluation",
        expected_count=evaluation_count,
    )
    if set(fit) & set(evaluation):
        raise ValueError("checkpoint fit and evaluation rosters overlap")
    fit_digest = _names_digest(fit, sort_names=sort_fit_digest)
    evaluation_digest = _names_digest(evaluation)
    if selection.get(fit_digest_key) != fit_digest:
        raise ValueError("checkpoint fit roster digest is invalid")
    if selection.get(evaluation_digest_key) != evaluation_digest:
        raise ValueError("checkpoint evaluation roster digest is invalid")
    return (
        fit,
        evaluation,
        FrozenFitEvaluationLineage(
            fit_count=len(fit),
            fit_order_digest=_names_digest(fit),
            evaluation_count=len(evaluation),
            evaluation_order_digest=evaluation_digest,
        ),
    )


def load_fullres_twin_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    expected_sha256: str = FROZEN_TWIN_SHA256,
) -> LoadedFullResolutionTwinCheckpoint:
    """Load only the exact frozen ordered-side Twin checkpoint."""

    expected = _validate_sha256(expected_sha256, name="expected_twin_sha256")
    path = _require_regular_file(checkpoint_path)
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError("full-resolution Twin checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Twin checkpoint payload must be a mapping")
    contract = payload.get("contract")
    if not isinstance(contract, Mapping) or dict(contract) != FROZEN_TWIN_CONTRACT:
        raise ValueError("full-resolution Twin architecture contract changed")
    _, _, lineage = _checkpoint_rosters(
        payload.get("selection"),
        fit_key="train_filenames",
        fit_digest_key="train_digest",
        evaluation_key="evaluation_filenames",
        evaluation_digest_key="evaluation_digest",
        fit_count=256,
        evaluation_count=24,
    )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Twin checkpoint has no state_dict mapping")
    model = FullResolutionTwinSideMatcher(
        dimension=int(contract["dimension"]),
        field_blocks=int(contract["field_blocks"]),
        sequence_blocks=int(contract["sequence_blocks"]),
        raw_skip_gain=float(contract["raw_skip_gain"]),
    ).to(device)
    model.load_state_dict(dict(state_dict), strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != int(
        contract["parameter_count"]
    ):
        raise ValueError("full-resolution Twin parameter-count contract changed")
    if any(not bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters()):
        raise ValueError("full-resolution Twin checkpoint contains non-finite parameters")
    model.eval().requires_grad_(False)
    return LoadedFullResolutionTwinCheckpoint(path, observed, model, dict(contract), lineage)


@dataclass(frozen=True)
class LoadedRawTwinUnionCheckpoint:
    """Frozen Union-v2 head plus exact config, selection and model lineage."""

    path: Path
    sha256: str
    model: RawTwinUnionReranker
    contract: dict[str, Any]
    config_path: Path
    config_sha256: str
    selection_path: Path
    selection_sha256: str
    socket_checkpoint_sha256: str
    twin_checkpoint_sha256: str
    lineage: FrozenFitEvaluationLineage


def _validate_union_config(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "aiijc-raw-twin-union-reranker-v2-preregistered":
        raise ValueError("unsupported Union-v2 preregistration schema")
    frozen = payload.get("frozen_inputs")
    candidates = payload.get("candidate_roster")
    model = payload.get("model")
    inference = payload.get("inference")
    legality = payload.get("legality")
    sections = (frozen, candidates, model, inference, legality)
    if not all(isinstance(value, Mapping) for value in sections):
        raise ValueError("Union-v2 preregistration contract is incomplete")
    assert isinstance(frozen, Mapping)
    assert isinstance(candidates, Mapping)
    assert isinstance(model, Mapping)
    assert isinstance(inference, Mapping)
    assert isinstance(legality, Mapping)
    if frozen.get("socket_d64", {}).get("sha256") != FROZEN_SOCKET_SHA256:
        raise ValueError("Union-v2 config Socket lineage changed")
    if frozen.get("fullres_twin", {}).get("sha256") != FROZEN_TWIN_SHA256:
        raise ValueError("Union-v2 config Twin lineage changed")
    if (
        candidates.get("raw_candidates_per_query") != 32
        or candidates.get("twin_candidates_per_query") != 32
        or candidates.get("frozen_raw_hard_projection_edges_per_axis") != 552
        or candidates.get("maximum_candidates_per_query") != 65
        or candidates.get("raw_candidate_roster_immutable") is not True
    ):
        raise ValueError("Union-v2 candidate roster contract changed")
    if (
        model.get("hidden_dimension") != 64
        or model.get("residual_limit") != 2.0
        or model.get("input_index_embedding") is not False
        or model.get("pixel_prediction") is not False
    ):
        raise ValueError("Union-v2 model contract changed")
    if inference.get("fixed_hard_edge_budget_per_axis") != DECODER_EDGE_BUDGET:
        raise ValueError("Union-v2 decoder budget changed")
    required_legality = {
        "organizer_train_only": True,
        "no_target_at_inference": True,
        "strict_original_upright_tile_identities_if_decoder_runs": True,
        "no_restored_or_generated_output_pixels": True,
        "no_holdout_or_competition_test": True,
    }
    if any(legality.get(key) is not value for key, value in required_legality.items()):
        raise ValueError("Union-v2 legality contract changed")


def load_raw_twin_union_checkpoint(
    checkpoint_path: Path,
    *,
    config_path: Path,
    selection_path: Path,
    device: torch.device,
    expected_checkpoint_sha256: str = FROZEN_UNION_CHECKPOINT_SHA256,
    expected_config_sha256: str = FROZEN_UNION_CONFIG_SHA256,
    expected_selection_sha256: str = FROZEN_UNION_SELECTION_SHA256,
) -> LoadedRawTwinUnionCheckpoint:
    """Load frozen Union-v2 only after all content-addressed lineage checks."""

    expected_checkpoint = _validate_sha256(
        expected_checkpoint_sha256,
        name="expected_union_checkpoint_sha256",
    )
    config = _load_sha_locked_json(
        config_path,
        expected_sha256=expected_config_sha256,
        name="Union-v2 config",
    )
    _validate_union_config(config)
    selection = _load_sha_locked_json(
        selection_path,
        expected_sha256=expected_selection_sha256,
        name="Union-v2 selection",
    )
    if (
        selection.get("schema")
        != "aiijc-raw-twin-union-reranker-selection-commitment-v2"
        or selection.get("status") != "frozen-before-selected-target-access"
        or selection.get("preregistration_sha256") != expected_config_sha256
        or selection.get("socket_checkpoint_sha256") != FROZEN_SOCKET_SHA256
        or selection.get("twin_checkpoint_sha256") != FROZEN_TWIN_SHA256
        or selection.get("holdout_and_competition_test_opened") is not False
    ):
        raise ValueError("Union-v2 frozen selection lineage changed")
    fit = _validated_roster(selection.get("fit_filenames"), name="fit", expected_count=256)
    evaluation = _validated_roster(
        selection.get("evaluation_filenames"),
        name="evaluation",
        expected_count=24,
    )
    if set(fit) & set(evaluation) or selection.get("fit_evaluation_overlap") != []:
        raise ValueError("Union-v2 fit and evaluation rosters overlap")
    if selection.get("fit_order_digest") != _names_digest(fit):
        raise ValueError("Union-v2 selection fit digest is invalid")
    if selection.get("evaluation_order_digest") != _names_digest(evaluation):
        raise ValueError("Union-v2 selection evaluation digest is invalid")

    path = _require_regular_file(checkpoint_path)
    observed = sha256_file(path)
    if observed != expected_checkpoint:
        raise ValueError("raw/twin Union-v2 checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Union-v2 checkpoint payload must be a mapping")
    contract = payload.get("contract")
    if not isinstance(contract, Mapping) or dict(contract) != FROZEN_UNION_CONTRACT:
        raise ValueError("raw/twin Union-v2 architecture contract changed")
    checkpoint_fit, checkpoint_evaluation, lineage = _checkpoint_rosters(
        payload.get("selection"),
        fit_key="train_filenames",
        fit_digest_key="train_digest",
        evaluation_key="evaluation_filenames",
        evaluation_digest_key="evaluation_digest",
        fit_count=256,
        evaluation_count=24,
        sort_fit_digest=True,
    )
    if checkpoint_fit != fit or checkpoint_evaluation != evaluation:
        raise ValueError("Union-v2 checkpoint and frozen selection rosters differ")
    history = payload.get("training_history")
    if not isinstance(history, list) or len(history) != 400:
        raise ValueError("Union-v2 checkpoint training-history contract changed")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Union-v2 checkpoint has no state_dict mapping")
    model = RawTwinUnionReranker(
        int(contract["feature_dimension"]),
        hidden_dimension=int(contract["hidden_dimension"]),
        residual_limit=float(contract["residual_limit"]),
    ).to(device)
    model.load_state_dict(dict(state_dict), strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != FROZEN_UNION_PARAMETER_COUNT:
        raise ValueError("raw/twin Union-v2 parameter-count contract changed")
    if any(not bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters()):
        raise ValueError("raw/twin Union-v2 checkpoint contains non-finite parameters")
    model.eval().requires_grad_(False)
    return LoadedRawTwinUnionCheckpoint(
        path=path,
        sha256=observed,
        model=model,
        contract=dict(contract),
        config_path=_require_regular_file(config_path),
        config_sha256=expected_config_sha256,
        selection_path=_require_regular_file(selection_path),
        selection_sha256=expected_selection_sha256,
        socket_checkpoint_sha256=FROZEN_SOCKET_SHA256,
        twin_checkpoint_sha256=FROZEN_TWIN_SHA256,
        lineage=lineage,
    )


def _module_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("production model has no parameters") from error


def _compatible_device(actual: torch.device, requested: torch.device) -> bool:
    """Treat an unspecified accelerator index as its canonical default index."""

    if actual.type != requested.type:
        return False
    if requested.index is None:
        return actual.index in {None, 0}
    return actual.index == requested.index


def _validate_adapter_lineage(
    socket: LoadedSocketCheckpoint,
    twin: LoadedFullResolutionTwinCheckpoint,
    union: LoadedRawTwinUnionCheckpoint,
    *,
    device: torch.device,
) -> None:
    if socket.sha256 != FROZEN_SOCKET_SHA256 or socket.sha256 != union.socket_checkpoint_sha256:
        raise ValueError("Union-v2 was not trained against this Socket checkpoint")
    if twin.sha256 != FROZEN_TWIN_SHA256 or twin.sha256 != union.twin_checkpoint_sha256:
        raise ValueError("Union-v2 was not trained against this Twin checkpoint")
    models = (socket.model, twin.model, union.model)
    if any(not _compatible_device(_module_device(model), device) for model in models):
        raise ValueError("Socket, Twin, Union-v2 and requested inference device differ")
    if any(model.training for model in models):
        raise ValueError("production inference requires every model in eval mode")
    if (
        socket.contract.get("dimension") != 64
        or twin.contract.get("dimension") != 48
        or union.model.feature_dimension != len(FEATURE_NAMES)
    ):
        raise ValueError("frozen Socket/Twin/Union feature dimensions differ")


@dataclass(frozen=True)
class RawTwinUnionInference:
    """Target-free learned assignments plus frozen Socket evidence."""

    socket_output: SocketOutput
    candidate_snapshot: UnionCandidateSnapshot
    learned_right_log_assignment: np.ndarray
    learned_down_log_assignment: np.ndarray
    residual: np.ndarray
    candidate_count: int
    candidates_per_axis: tuple[int, int]
    candidate_row_minimum: int
    candidate_row_maximum: int
    matcher_seconds: float
    twin_seconds: float
    reranker_seconds: float

    def report(self) -> dict[str, Any]:
        return {
            "candidate_snapshot": {
                "count": self.candidate_snapshot.count,
                "sha256": self.candidate_snapshot.sha256,
                "contains_targets_or_absolute_slots": False,
            },
            "candidate_count": self.candidate_count,
            "candidates_per_axis": {
                "right": self.candidates_per_axis[0],
                "down": self.candidates_per_axis[1],
            },
            "candidate_row_minimum": self.candidate_row_minimum,
            "candidate_row_maximum": self.candidate_row_maximum,
            "residual_sha256": _array_sha256(self.residual, dtype="<f4"),
            "learned_right_assignment_sha256": _array_sha256(
                self.learned_right_log_assignment,
                dtype="<f4",
            ),
            "learned_down_assignment_sha256": _array_sha256(
                self.learned_down_log_assignment,
                dtype="<f4",
            ),
            "hard_projection_edges_per_axis": GRID_SIZE * (GRID_SIZE - 1),
            "hard_projection_inside_immutable_union": True,
            "runtime_seconds": {
                "socket_d64": self.matcher_seconds,
                "fullres_twin": self.twin_seconds,
                "union_reranker_and_restricted_ot": self.reranker_seconds,
            },
        }


def _assert_projection_inside_union(
    assignment: np.ndarray,
    rows: tuple[np.ndarray, ...],
    *,
    axis: str,
) -> None:
    matching = hard_partial_axis_matching(assignment, grid=GRID_SIZE, axis=axis)
    expected = GRID_SIZE * (GRID_SIZE - 1)
    if len(matching.edges) != expected:
        raise RuntimeError("Union-v2 hard projection cardinality changed")
    for edge in matching.edges:
        if int(edge.target) not in rows[int(edge.source)]:
            raise RuntimeError("Union-v2 hard projection escaped the immutable candidate union")


@torch.inference_mode()
def infer_raw_twin_union_assignments(
    dirty_tiles: np.ndarray,
    socket: LoadedSocketCheckpoint,
    twin: LoadedFullResolutionTwinCheckpoint,
    union: LoadedRawTwinUnionCheckpoint,
    *,
    device: torch.device,
) -> RawTwinUnionInference:
    """Convert one board's original dirty tiles to frozen Union-v2 assignments."""

    _validate_adapter_lineage(socket, twin, union, device=device)
    tiles = np.asarray(dirty_tiles)
    expected = (TILE_COUNT, 20, 20, RGB_CHANNELS)
    if tiles.shape != expected or tiles.dtype != np.uint8:
        raise ValueError(f"dirty_tiles must be uint8 RGB with shape {expected}")
    tensor = (
        torch.from_numpy(np.ascontiguousarray(tiles))
        .permute(0, 3, 1, 2)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
        .unsqueeze(0)
    )
    socket_started = perf_counter()
    tokens, socket_output = extract_frozen_socket_context(
        socket.model,
        tensor,
        grid=GRID_SIZE,
    )
    matcher_seconds = perf_counter() - socket_started
    twin_started = perf_counter()
    twin_output = twin.model(tensor)
    twin_seconds = perf_counter() - twin_started
    reranker_started = perf_counter()
    board = prepare_raw_twin_union_board(
        tokens[0],
        socket_output,
        twin_output,
        grid=GRID_SIZE,
        topk=int(union.contract["raw_topk"]),
    )
    if board.values.shape[1] != int(union.contract["feature_dimension"]):
        raise RuntimeError("runtime Union-v2 feature contract changed")
    output = union.model(board)
    candidate_snapshot = freeze_union_candidate_snapshot(
        board.axis,
        board.source,
        board.target,
        output.scores,
        grid=GRID_SIZE,
    )
    right_dense, down_dense = candidate_score_matrices(
        board,
        output.scores,
        fill_value=float(union.contract["outside_union_fill"]),
    )
    for axis, dense in enumerate((right_dense[0], down_dense[0])):
        for source, candidates in enumerate(board.rows[axis]):
            outside = np.ones(TILE_COUNT, dtype=bool)
            outside[candidates] = False
            outside[source] = False
            outside_tensor = torch.from_numpy(outside).to(device=dense.device)
            if not bool(
                torch.all(
                    dense[source, outside_tensor]
                    == float(union.contract["outside_union_fill"])
                ).item()
            ):
                raise RuntimeError("outside-union learned score is not forbidden")
    right, down = restricted_partial_ot(
        board,
        output.scores,
        socket_output,
        iterations=int(socket.contract["sinkhorn_iterations"]),
    )
    right_array = np.ascontiguousarray(right[0].float().cpu().numpy(), dtype=np.float32)
    down_array = np.ascontiguousarray(down[0].float().cpu().numpy(), dtype=np.float32)
    _assert_projection_inside_union(right_array, board.rows[0], axis="right")
    _assert_projection_inside_union(down_array, board.rows[1], axis="down")
    per_axis = tuple(int(torch.count_nonzero(board.axis == axis).item()) for axis in (0, 1))
    row_counts = [len(row) for axis_rows in board.rows for row in axis_rows]
    return RawTwinUnionInference(
        socket_output=socket_output,
        candidate_snapshot=candidate_snapshot,
        learned_right_log_assignment=right_array,
        learned_down_log_assignment=down_array,
        residual=np.ascontiguousarray(output.residual.float().cpu().numpy(), dtype=np.float32),
        candidate_count=len(board.values),
        candidates_per_axis=(per_axis[0], per_axis[1]),
        candidate_row_minimum=min(row_counts),
        candidate_row_maximum=max(row_counts),
        matcher_seconds=matcher_seconds,
        twin_seconds=twin_seconds,
        reranker_seconds=perf_counter() - reranker_started,
    )


def _decode_prediction(
    image: np.ndarray,
    right_log_assignment: Any,
    down_log_assignment: Any,
    *,
    matcher_seconds: float,
    started: float,
) -> SocketSorterPrediction:
    decoder = decode_socket_assignments(
        right_log_assignment,
        down_log_assignment,
        grid=GRID_SIZE,
        config=SocketDecoderConfig(
            component_edge_budget_per_axis=DECODER_EDGE_BUDGET,
            swap_edge_budget_per_axis=DECODER_EDGE_BUDGET,
            max_swap_steps=DECODER_SWAP_STEPS,
        ),
    )
    cyclic = select_global_cyclic_translation(
        decoder.layout,
        right_log_assignment,
        down_log_assignment,
        grid=GRID_SIZE,
        config=CyclicTranslationConfig(border_weight=CYCLIC_BORDER_WEIGHT),
    )
    raw, audit = assemble_audited_original_tiles(image, cyclic.layout)
    final = IDENTITY_PIXEL_TAIL.apply(raw)
    if not np.array_equal(final, raw):
        raise RuntimeError("identity Union-v2 adapter altered original-tile assembly")
    return SocketSorterPrediction(
        layout=np.ascontiguousarray(cyclic.layout, dtype=np.int32),
        raw=raw,
        output=final,
        audit=audit,
        decoder_report=decoder.report(),
        cyclic_report=cyclic.report(),
        matcher_seconds=matcher_seconds,
        total_seconds=perf_counter() - started,
    )


@dataclass(frozen=True)
class RawTwinUnionProductionPrediction:
    """Unchanged baseline and optional learned Union-v2 board variants."""

    selected_variant: Literal["baseline", "raw-twin-union-v2"]
    fallback_reason: str | None
    selected: SocketSorterPrediction
    baseline: SocketSorterPrediction
    learned: SocketSorterPrediction | None
    union_checkpoint_sha256: str | None
    twin_checkpoint_sha256: str | None
    inference_report: dict[str, Any] | None

    def report(self) -> dict[str, Any]:
        arms: dict[str, Any] = {
            "baseline": {
                "layout_sha256": _array_sha256(self.baseline.layout, dtype="<i4"),
                "decoder": self.baseline.decoder_report,
                "cyclic": self.baseline.cyclic_report,
                "permutation_audit": self.baseline.audit.as_dict(),
            }
        }
        if self.learned is not None:
            arms["raw-twin-union-v2"] = {
                "layout_sha256": _array_sha256(self.learned.layout, dtype="<i4"),
                "decoder": self.learned.decoder_report,
                "cyclic": self.learned.cyclic_report,
                "permutation_audit": self.learned.audit.as_dict(),
            }
        return {
            "schema": UNION_ADAPTER_SCHEMA,
            "selected_variant": self.selected_variant,
            "fallback_reason": self.fallback_reason,
            "union_checkpoint_sha256": self.union_checkpoint_sha256,
            "twin_checkpoint_sha256": self.twin_checkpoint_sha256,
            "inference": self.inference_report,
            "arms": arms,
            "policy": {
                "default_without_union_artifacts": "baseline",
                "targets_manifest_or_filenames_accepted": False,
                "restored_or_generated_pixels_used": False,
                "candidate_union": "raw32+twin32+frozen-raw-hard-projection",
                "restricted_partial_ot": True,
                "decoder_edge_budget_per_axis": DECODER_EDGE_BUDGET,
                "cyclic_border_weight": CYCLIC_BORDER_WEIGHT,
                "all_original_upright_tiles_used_exactly_once": True,
            },
        }


@torch.inference_mode()
def predict_raw_twin_union_variants(
    input_image: np.ndarray,
    socket: LoadedSocketCheckpoint,
    *,
    device: torch.device,
    twin: LoadedFullResolutionTwinCheckpoint | None = None,
    union: LoadedRawTwinUnionCheckpoint | None = None,
) -> RawTwinUnionProductionPrediction:
    """Run explicit Union-v2 or delegate bit-for-bit to the existing baseline.

    Both ``twin`` and ``union`` must be supplied to activate the learned arm.
    Supplying only one is a configuration error and never triggers fallback.
    """

    image = np.asarray(input_image)
    expected = (IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS)
    if image.shape != expected or image.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB input {expected}, got {image.dtype} {image.shape}")
    if twin is None and union is None:
        baseline = predict_socket_sorter(
            image,
            socket,
            device=device,
            cyclic_border5=True,
            pixel_tail=IDENTITY_PIXEL_TAIL,
        )
        return RawTwinUnionProductionPrediction(
            selected_variant="baseline",
            fallback_reason="union-artifacts-not-configured",
            selected=baseline,
            baseline=baseline,
            learned=None,
            union_checkpoint_sha256=None,
            twin_checkpoint_sha256=None,
            inference_report=None,
        )
    if twin is None or union is None:
        raise ValueError("Union-v2 activation requires both Twin and reranker checkpoints")

    started = perf_counter()
    inference = infer_raw_twin_union_assignments(
        split_tiles(image),
        socket,
        twin,
        union,
        device=device,
    )
    baseline = _decode_prediction(
        image,
        inference.socket_output.right_log_assignment,
        inference.socket_output.down_log_assignment,
        matcher_seconds=inference.matcher_seconds,
        started=started,
    )
    learned = _decode_prediction(
        image,
        inference.learned_right_log_assignment,
        inference.learned_down_log_assignment,
        matcher_seconds=inference.matcher_seconds,
        started=started,
    )
    return RawTwinUnionProductionPrediction(
        selected_variant="raw-twin-union-v2",
        fallback_reason=None,
        selected=learned,
        baseline=baseline,
        learned=learned,
        union_checkpoint_sha256=union.sha256,
        twin_checkpoint_sha256=twin.sha256,
        inference_report=inference.report(),
    )


__all__ = [
    "CYCLIC_BORDER_WEIGHT",
    "FROZEN_SOCKET_SHA256",
    "FROZEN_TWIN_CONTRACT",
    "FROZEN_TWIN_SHA256",
    "FROZEN_UNION_CHECKPOINT_SHA256",
    "FROZEN_UNION_CONFIG_SHA256",
    "FROZEN_UNION_CONTRACT",
    "FROZEN_UNION_SELECTION_SHA256",
    "FrozenFitEvaluationLineage",
    "LoadedFullResolutionTwinCheckpoint",
    "LoadedRawTwinUnionCheckpoint",
    "RawTwinUnionInference",
    "RawTwinUnionProductionPrediction",
    "UNION_ADAPTER_SCHEMA",
    "infer_raw_twin_union_assignments",
    "load_fullres_twin_checkpoint",
    "load_raw_twin_union_checkpoint",
    "predict_raw_twin_union_variants",
]
