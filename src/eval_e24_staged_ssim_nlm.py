"""Exact mechanics for the sealed E24 staged image gate.

This module contains the parts of the staged CRS-v1 decision that are already
literal in ``E24_CONTEXT_RELATION_SELECTOR.md``:

* byte-stable commits for the OOF relation ordering, margins, cap, DSU
  outcomes, and ordered decoded components;
* the one fixed candidate packer and the exact RR96 comparator;
* upright corrupted-tile assembly and the unchanged NLM10 restoration;
* paired SSIM/neighbour aggregation and the five inclusive hard gates.

This module still has no real-scene, archive, or clean-target loader.  The only
real metric capability lives in the companion runner's authenticated, narrow
subprocess.  It returns one canonical metric row and cannot export pixels or a
permutation.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence


_DEFAULT_PYCACHE = Path("E:/pazzle_work/posegraph_e24_selector/pycache")
if sys.pycache_prefix is None or Path(sys.pycache_prefix).drive.upper() != "E:":
    sys.pycache_prefix = str(_DEFAULT_PYCACHE)

import numpy as np

import eval_e24_context_relation_selector as e24_eval


class E24StagedContractError(RuntimeError):
    """The frozen staged evaluator or an immutable artifact drifted."""


SCHEMA_VERSION = 1
CALIBRATION_IDS = tuple(range(10, 18))
OOF_FOLDS: Mapping[int, tuple[int, int]] = MappingProxyType(
    {0: (10, 14), 1: (11, 15), 2: (12, 16), 3: (13, 17)}
)
NUM_TILES = 576
GRID = 24
TILE_SIZE = 20
IMAGE_SIZE = 480
CANDIDATE_WIDTH = 128
NUM_DIRECTIONS = 4

DECODE_ARTIFACT_SCHEMA = "pazzle-e24-crs-v1-staged-decode-artifact-v1"
DECODE_COMMIT_SCHEMA = "pazzle-e24-crs-v1-staged-decode-commit-v1"
BOARD_ARTIFACT_SCHEMA = "pazzle-e24-crs-v1-staged-board-artifact-v1"
BOARD_COMMIT_SCHEMA = "pazzle-e24-crs-v1-staged-board-commit-v1"
PREMETRIC_SEAL_SCHEMA = "pazzle-e24-crs-v1-staged-premetric-seal-v1"
STAGED_REPORT_SCHEMA = "pazzle-e24-crs-v1-staged-ssim-nlm-report-v1"
METRIC_REQUEST_SCHEMA = "pazzle-e24-crs-v1-staged-metric-request-v1"
METRIC_RESPONSE_SCHEMA = "pazzle-e24-crs-v1-staged-metric-response-v1"

PINNED_E12_REPORT_SHA256 = (
    "16ceecfea99e006a1126b17d7d58fb5d188ec694c6a5097310dfe021bd2f901a"
)
PINNED_CALIBRATION_REPORT_SHA256 = (
    "3b76d6bed59df13eb98af049c3a756151b4485c2e50b1da88ec50fb7a1dfe305"
)
PINNED_SCENE_PROVENANCE_DIGEST = (
    "00cd2fdd9189d6453e7c1b215e4ee067b843bc51cdcd0122fa66fdc076779c98"
)
PINNED_RR96_MEAN_SOLVE_SSIM = 0.094607964147414
PINNED_RR96_MEAN_FINAL_SSIM = 0.15930445310452002
RR96_MEAN_ABSOLUTE_TOLERANCE = 1.0e-12

END_TO_END_GATES: Mapping[str, float | int] = MappingProxyType(
    {
        "solve_ssim_delta_mean_min": 0.003,
        "final_ssim_delta_mean_min": 0.002,
        "final_wins_min": 5,
        "worst_final_delta_min": -0.020,
        "neighbour_delta_mean_min": 0.005,
    }
)

STAGED_PROTOCOL: Mapping[str, Any] = MappingProxyType(
    {
        "schema": "pazzle-e24-crs-v1-staged-ssim-nlm-protocol-v1",
        "calibration_ids": CALIBRATION_IDS,
        "folds": {str(fold): ids for fold, ids in OOF_FOLDS.items()},
        "orientation_degrees": (0,),
        "reflection": False,
        "candidate_packer": {
            "call": "solve_components_from_scores",
            "repair_passes": 0,
            "restarts": 1,
            "seed": 1234,
        },
        "baseline": {
            "name": "exact_rr96",
            "call": "solve_buddies_from_scores",
            "max_edges": 96,
            "min_margin": 0.0,
            "repair_passes": 0,
        },
        "dense_scores": "eval_seeded_qap.dense_rd_cpu_float32",
        "assembly": "imgio.assemble_original_corrupted_upright_tiles",
        "restoration": {
            "name": "champion_nlm10",
            "h": 10,
            "h_color": 10,
            "template_window": 7,
            "search_window": 21,
        },
        "ssim": {
            "implementation": "skimage.metrics.structural_similarity",
            "channel_axis": 2,
            "data_range": 255,
        },
        "gates": dict(END_TO_END_GATES),
        "clean_target_broker": {
            "mode": "authenticated_isolated_subprocess_per_scene",
            "permutation": "ledger_pinned_raw_npz_exact_permutation.npy_member_only",
            "target": "literal_E_drive_target_path_from_exact_generation3_validation_name",
            "directory_enumeration": False,
            "output": "one_canonical_metric_row_no_arrays",
            "required_barrier": "caller_supplied_exact_decode_board_barrier_sha256",
        },
        "rr96_metric_source": {
            "choice": "reuse_exact_pinned_E12_RR_record_no_second_NLM_call",
            "e12_report_sha256": PINNED_E12_REPORT_SHA256,
            "calibration_report_sha256": PINNED_CALIBRATION_REPORT_SHA256,
            "scene_provenance_digest": PINNED_SCENE_PROVENANCE_DIGEST,
            "verify_committed_board_solved_restored_hashes": True,
        },
        "e25_opened": False,
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise E24StagedContractError("value is not canonical finite JSON") from exc
    return (encoded + "\n").encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


PROTOCOL_SHA256 = sha256_bytes(canonical_json_bytes(dict(STAGED_PROTOCOL)))

METRIC_BROKER_CONTRACT: Mapping[str, Any] = MappingProxyType(
    {
        "schema": "pazzle-e24-crs-v1-trusted-metric-broker-contract-v1",
        "calibration_ids": CALIBRATION_IDS,
        "process_scope": "one_scene_per_fresh_subprocess",
        "authority": "authenticate_authority_generation3_then_exact_barrier_sha",
        "request_order": "10_through_17_sha256_chained",
        "raw_archive": {
            "authority": "generation3_ledger_label_free_projection",
            "allowed_member": "permutation.npy",
            "member_count_read": 1,
            "export": False,
        },
        "target_lineage": {
            "validation_names": tuple(f"img_0067{image:02d}.png" for image in CALIBRATION_IDS),
            "path": "E:/pazzle_data/train/targets/{validation_name}",
            "reader": "PIL.Image.open_exact_path_convert_RGB",
            "directory_enumeration": False,
            "expected_target_sha_source": "pinned_E12_scene_provenance",
            "export": False,
        },
        "rr96": dict(STAGED_PROTOCOL["rr96_metric_source"]),
        "candidate": "measure_committed_solved_and_NLM10_canvases_once",
        "response": "canonical_JSON_metric_row_only_create_once",
        "forbidden": (
            "RawScene",
            "directory_scan",
            "unrestricted_npz",
            "additional_archive_member",
            "eval_clean_score_oracle_import",
            "eval_e14_cc192_discovery_import",
            "canvas_data_import",
            "train_val_split",
            "list_train",
            "pixel_or_permutation_output",
            "E25",
            "test",
        ),
    }
)
METRIC_BROKER_CONTRACT_SHA256 = sha256_bytes(
    canonical_json_bytes(dict(METRIC_BROKER_CONTRACT))
)
METRIC_CHAIN_GENESIS_SHA256 = sha256_bytes(
    canonical_json_bytes(
        {"schema": "pazzle-e24-crs-v1-metric-chain-genesis-v1", "next_image": 10}
    )
)


def fixed_nlm10(image: np.ndarray) -> np.ndarray:
    """Literal NLM10 tail without importing a broad historical evaluator."""

    import cv2

    value = _strict_rgb(image, size=IMAGE_SIZE, label="NLM10 input")
    cv2.setNumThreads(1)
    restored = cv2.fastNlMeansDenoisingColored(
        np.ascontiguousarray(value), None, 10, 10, 7, 21
    )
    return _strict_rgb(restored, size=IMAGE_SIZE, label="NLM10 output")


def _lower_sha(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise E24StagedContractError(f"{label} must be a lowercase SHA256")
    return value


def _require_e24_path(path: str | os.PathLike[str], *, label: str) -> Path:
    try:
        return e24_eval._require_e24_storage_path(path, label=label)
    except e24_eval.E24EvaluatorContractError as exc:
        raise E24StagedContractError(str(exc)) from exc


def load_canonical_json(
    path: str | os.PathLike[str], *, label: str
) -> dict[str, Any]:
    source = _require_e24_path(path, label=label)
    try:
        raw = source.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except Exception as exc:
        raise E24StagedContractError(f"{label} is unreadable") from exc
    if type(value) is not dict or raw != canonical_json_bytes(value):
        raise E24StagedContractError(f"{label} must be canonical JSON")
    return value


def commit_canonical_create_once(
    path: str | os.PathLike[str], payload: Mapping[str, Any]
) -> str:
    destination = _require_e24_path(path, label="canonical staged artifact")
    body = canonical_json_bytes(dict(payload))
    try:
        e24_eval._atomic_write_create(destination, body)
    except e24_eval.E24EvaluatorContractError as exc:
        raise E24StagedContractError(str(exc)) from exc
    return sha256_bytes(body)


def commit_canonical_or_verify(
    path: str | os.PathLike[str], payload: Mapping[str, Any]
) -> str:
    destination = _require_e24_path(path, label="canonical staged artifact")
    body = canonical_json_bytes(dict(payload))
    try:
        e24_eval._atomic_write_create_or_verify(destination, body)
    except e24_eval.E24EvaluatorContractError as exc:
        raise E24StagedContractError(str(exc)) from exc
    return sha256_bytes(body)


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream, np.ascontiguousarray(value), version=(1, 0), allow_pickle=False
    )
    return stream.getvalue()


def _canonical_npz_bytes(members: Sequence[tuple[str, np.ndarray]]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in members:
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(value))
    return stream.getvalue()


def _commit_npz_or_verify(path: Path, payload: bytes) -> str:
    destination = _require_e24_path(path, label="canonical staged NPZ")
    try:
        e24_eval._atomic_write_create_or_verify(destination, payload)
    except e24_eval.E24EvaluatorContractError as exc:
        raise E24StagedContractError(str(exc)) from exc
    return sha256_bytes(payload)


_REASON_TO_CODE = MappingProxyType(
    {"tree": 0, "cycle": 1, "conflict": 2, "contact": 3, "collision": 4, "span": 5}
)
_CODE_TO_REASON = tuple(
    key for key, _value in sorted(_REASON_TO_CODE.items(), key=lambda item: item[1])
)

_DECODE_NPZ_NAMES = (
    "schema",
    "schema_version",
    "protocol_sha256",
    "image",
    "fold",
    "base_component_count",
    "component_offsets",
    "component_tiles",
    "component_positions",
    "selected_hypothesis_ids",
    "selected_relation_ids",
    "selected_relations",
    "selected_scores",
    "selected_support",
    "attempted_count",
    "outcome_reason_codes",
)


@dataclass(frozen=True)
class FrozenDecode:
    image: int
    fold: int
    base_component_count: int
    component_offsets: np.ndarray
    component_tiles: np.ndarray
    component_positions: np.ndarray
    selected_hypothesis_ids: np.ndarray
    selected_relation_ids: np.ndarray
    selected_relations: np.ndarray
    selected_scores: np.ndarray
    selected_support: np.ndarray
    attempted_count: int
    outcome_reason_codes: np.ndarray

    @property
    def components(self) -> tuple[dict[int, tuple[int, int]], ...]:
        output: list[dict[int, tuple[int, int]]] = []
        for start, stop in zip(self.component_offsets[:-1], self.component_offsets[1:]):
            component: dict[int, tuple[int, int]] = {}
            for index in range(int(start), int(stop)):
                component[int(self.component_tiles[index])] = (
                    int(self.component_positions[index, 0]),
                    int(self.component_positions[index, 1]),
                )
            output.append(component)
        return tuple(output)


def _fold_for_image(image: int) -> int:
    if type(image) is not int or image not in CALIBRATION_IDS:
        raise E24StagedContractError("image must be one of the exact E24 IDs 10..17")
    return next(fold for fold, ids in OOF_FOLDS.items() if image in ids)


def _readonly(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _validate_frozen_decode(value: FrozenDecode) -> FrozenDecode:
    if type(value) is not FrozenDecode:
        raise E24StagedContractError("decode artifact has the wrong exact type")
    if value.fold != _fold_for_image(value.image):
        raise E24StagedContractError("decode artifact belongs to the wrong OOF fold")
    if type(value.base_component_count) is not int or not (
        1 <= value.base_component_count <= NUM_TILES
    ):
        raise E24StagedContractError("base component count is invalid")

    offsets = np.asarray(value.component_offsets)
    tiles = np.asarray(value.component_tiles)
    positions = np.asarray(value.component_positions)
    if offsets.dtype != np.int64 or offsets.ndim != 1 or offsets.size < 2:
        raise E24StagedContractError("component offsets must be one-dimensional int64")
    if tiles.dtype != np.int64 or tiles.shape != (NUM_TILES,):
        raise E24StagedContractError("component tiles must be int64[576]")
    if positions.dtype != np.int16 or positions.shape != (NUM_TILES, 2):
        raise E24StagedContractError("component positions must be int16[576,2]")
    if not (
        offsets.flags.c_contiguous
        and tiles.flags.c_contiguous
        and positions.flags.c_contiguous
    ):
        raise E24StagedContractError("component arrays must be C-contiguous")
    if (
        int(offsets[0]) != 0
        or int(offsets[-1]) != NUM_TILES
        or bool(np.any(np.diff(offsets) <= 0))
        or not np.array_equal(np.sort(tiles), np.arange(NUM_TILES, dtype=np.int64))
    ):
        raise E24StagedContractError("components do not partition all 576 tiles")
    for start, stop in zip(offsets[:-1], offsets[1:]):
        local = positions[int(start) : int(stop)].astype(np.int64, copy=False)
        local_tiles = tiles[int(start) : int(stop)]
        if (
            int(local[:, 0].min()) != 0
            or int(local[:, 1].min()) != 0
            or int(local[:, 0].max()) >= GRID
            or int(local[:, 1].max()) >= GRID
            or len({(int(row), int(col)) for row, col in local}) != len(local)
            or bool(np.any(np.diff(local_tiles) <= 0))
        ):
            raise E24StagedContractError(
                "decoded component geometry/order is not legal/normalized"
            )

    hypothesis_ids = np.asarray(value.selected_hypothesis_ids)
    relation_ids = np.asarray(value.selected_relation_ids)
    relations = np.asarray(value.selected_relations)
    scores = np.asarray(value.selected_scores)
    support = np.asarray(value.selected_support)
    selected_count = len(hypothesis_ids)
    if (
        hypothesis_ids.dtype != np.int64
        or relation_ids.dtype != np.int64
        or relations.dtype != np.int64
        or scores.dtype != np.float64
        or support.dtype != np.int64
        or hypothesis_ids.shape != (selected_count,)
        or relation_ids.shape != (selected_count,)
        or relations.shape != (selected_count, 4)
        or scores.shape != (selected_count, 3)
        or support.shape != (selected_count,)
        or not all(
            item.flags.c_contiguous
            for item in (hypothesis_ids, relation_ids, relations, scores, support)
        )
    ):
        raise E24StagedContractError("selected-relation arrays have drifted")
    if not bool(np.isfinite(scores).all()) or bool(np.any(scores[:, 2] <= 0.0)):
        raise E24StagedContractError("selected scores/margins must be finite and positive")
    if selected_count and not np.array_equal(scores[:, 0] - scores[:, 1], scores[:, 2]):
        raise E24StagedContractError("stored margins are not exact score-minus-NONE values")
    if bool(np.any(relations[:, 0] >= relations[:, 1])) or bool(np.any(support < 1)):
        raise E24StagedContractError("selected relation canonicality/support drifted")
    observed_order = sorted(
        range(selected_count),
        key=lambda index: (
            -float(scores[index, 2]),
            int(relations[index, 0]),
            int(relations[index, 1]),
            int(relations[index, 2]),
            int(relations[index, 3]),
        ),
    )
    if observed_order != list(range(selected_count)):
        raise E24StagedContractError("selected relation ordering drifted")
    attempted = value.attempted_count
    expected_attempted = min(selected_count, 2 * (value.base_component_count - 1))
    reasons = np.asarray(value.outcome_reason_codes)
    if type(attempted) is not int or attempted != expected_attempted:
        raise E24StagedContractError("attempt prefix/cap drifted")
    if (
        reasons.dtype != np.uint8
        or reasons.shape != (attempted,)
        or not reasons.flags.c_contiguous
        or bool(np.any(reasons >= len(_CODE_TO_REASON)))
    ):
        raise E24StagedContractError("DSU outcome reason array drifted")

    return FrozenDecode(
        image=value.image,
        fold=value.fold,
        base_component_count=value.base_component_count,
        component_offsets=_readonly(offsets, np.dtype(np.int64)),
        component_tiles=_readonly(tiles, np.dtype(np.int64)),
        component_positions=_readonly(positions, np.dtype(np.int16)),
        selected_hypothesis_ids=_readonly(hypothesis_ids, np.dtype(np.int64)),
        selected_relation_ids=_readonly(relation_ids, np.dtype(np.int64)),
        selected_relations=_readonly(relations, np.dtype(np.int64)),
        selected_scores=_readonly(scores, np.dtype(np.float64)),
        selected_support=_readonly(support, np.dtype(np.int64)),
        attempted_count=attempted,
        outcome_reason_codes=_readonly(reasons, np.dtype(np.uint8)),
    )


def freeze_decode_result(
    *, image: int, base_component_count: int, decoded: object
) -> FrozenDecode:
    """Detach the exact relation ordering/DSU state without labels or images."""

    fold = _fold_for_image(image)
    required_attributes = (
        "selected",
        "attempted",
        "outcomes",
        "components",
        "attempt_cap",
    )
    if any(not hasattr(decoded, name) for name in required_attributes):
        raise E24StagedContractError("decoder result interface drifted")
    selected = tuple(decoded.selected)
    attempted = tuple(decoded.attempted)
    outcomes = tuple(decoded.outcomes)
    components = tuple(decoded.components)
    if tuple(selected[: len(attempted)]) != attempted or int(decoded.attempt_cap) != len(
        attempted
    ):
        raise E24StagedContractError("decoder attempt prefix is not exact")
    if len(outcomes) != len(attempted):
        raise E24StagedContractError("decoder outcome count differs from attempts")

    component_offsets = [0]
    component_tiles: list[int] = []
    component_positions: list[tuple[int, int]] = []
    for component in components:
        if type(component) is not dict or not component:
            raise E24StagedContractError("decoded component must be a nonempty dict")
        for tile, position in component.items():
            if (
                type(tile) is not int
                or type(position) is not tuple
                or len(position) != 2
                or any(type(coordinate) is not int for coordinate in position)
            ):
                raise E24StagedContractError("decoded component entry type drifted")
            component_tiles.append(tile)
            component_positions.append(position)
        component_offsets.append(len(component_tiles))

    def selected_field(name: str, dtype: np.dtype[Any]) -> np.ndarray:
        return np.ascontiguousarray([getattr(item, name) for item in selected], dtype=dtype)

    relation_values = np.ascontiguousarray(
        [[item.u, item.v, item.dr, item.dc] for item in selected], dtype=np.int64
    ).reshape(len(selected), 4)
    score_values = np.ascontiguousarray(
        [[item.score, item.none_score, item.margin] for item in selected],
        dtype=np.float64,
    ).reshape(len(selected), 3)
    reason_values: list[int] = []
    for index, outcome in enumerate(outcomes):
        if outcome.selection != attempted[index] or outcome.reason not in _REASON_TO_CODE:
            raise E24StagedContractError("DSU outcome/selection binding drifted")
        expected_accepted = outcome.reason in {"tree", "cycle"}
        if (
            type(outcome.accepted) is not bool
            or outcome.accepted is not expected_accepted
            or outcome.tree_merge is not (outcome.reason == "tree")
            or outcome.cycle is not (outcome.reason == "cycle")
        ):
            raise E24StagedContractError("DSU outcome algebra drifted")
        reason_values.append(_REASON_TO_CODE[outcome.reason])

    return _validate_frozen_decode(
        FrozenDecode(
            image=image,
            fold=fold,
            base_component_count=int(base_component_count),
            component_offsets=np.ascontiguousarray(component_offsets, dtype=np.int64),
            component_tiles=np.ascontiguousarray(component_tiles, dtype=np.int64),
            component_positions=np.ascontiguousarray(
                component_positions, dtype=np.int16
            ).reshape(NUM_TILES, 2),
            selected_hypothesis_ids=selected_field("hypothesis_id", np.dtype(np.int64)),
            selected_relation_ids=selected_field("relation_id", np.dtype(np.int64)),
            selected_relations=relation_values,
            selected_scores=score_values,
            selected_support=selected_field("support", np.dtype(np.int64)),
            attempted_count=len(attempted),
            outcome_reason_codes=np.ascontiguousarray(reason_values, dtype=np.uint8),
        )
    )


def _decode_npz_members(value: FrozenDecode) -> tuple[tuple[str, np.ndarray], ...]:
    frozen = _validate_frozen_decode(value)
    return (
        ("schema", np.asarray(DECODE_ARTIFACT_SCHEMA)),
        ("schema_version", np.asarray(SCHEMA_VERSION, dtype=np.int16)),
        ("protocol_sha256", np.asarray(PROTOCOL_SHA256)),
        ("image", np.asarray(frozen.image, dtype=np.int16)),
        ("fold", np.asarray(frozen.fold, dtype=np.int8)),
        ("base_component_count", np.asarray(frozen.base_component_count, dtype=np.int16)),
        ("component_offsets", frozen.component_offsets),
        ("component_tiles", frozen.component_tiles),
        ("component_positions", frozen.component_positions),
        ("selected_hypothesis_ids", frozen.selected_hypothesis_ids),
        ("selected_relation_ids", frozen.selected_relation_ids),
        ("selected_relations", frozen.selected_relations),
        ("selected_scores", frozen.selected_scores),
        ("selected_support", frozen.selected_support),
        ("attempted_count", np.asarray(frozen.attempted_count, dtype=np.int64)),
        ("outcome_reason_codes", frozen.outcome_reason_codes),
    )


def decode_npz_bytes(value: FrozenDecode) -> bytes:
    return _canonical_npz_bytes(_decode_npz_members(value))


_DECODE_PROVENANCE_KEYS = frozenset(
    {
        "ledger_sha256",
        "run_contract_sha256",
        "premetric_seal_sha256",
        "structural_report_sha256",
        "orchestration_receipt_sha256",
        "fold_commit_sha256",
        "model_sha256",
        "prediction_sha256",
        "feature_sha256",
        "input_manifest_sha256",
        "source_scene_contract_sha256",
        "right_sha256",
        "down_sha256",
    }
)


def _normalize_provenance(value: object, *, keys: frozenset[str]) -> dict[str, str]:
    if type(value) is not dict or set(value) != keys:
        raise E24StagedContractError("staged provenance field set drifted")
    return {key: _lower_sha(value[key], label=key) for key in sorted(keys)}


def commit_decode(
    *,
    artifact_path: str | os.PathLike[str],
    commit_path: str | os.PathLike[str],
    value: FrozenDecode,
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    frozen = _validate_frozen_decode(value)
    artifact = _require_e24_path(artifact_path, label="decode artifact")
    commit = _require_e24_path(commit_path, label="decode commit")
    if artifact == commit or artifact.suffix.lower() != ".npz":
        raise E24StagedContractError("decode artifact/commit paths are invalid")
    body = decode_npz_bytes(frozen)
    artifact_sha = _commit_npz_or_verify(artifact, body)
    normalized = _normalize_provenance(dict(provenance), keys=_DECODE_PROVENANCE_KEYS)
    reasons = [_CODE_TO_REASON[int(code)] for code in frozen.outcome_reason_codes]
    payload: dict[str, Any] = {
        "schema": DECODE_COMMIT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete_label_free",
        "protocol_sha256": PROTOCOL_SHA256,
        "image": frozen.image,
        "fold": frozen.fold,
        "base_component_count": frozen.base_component_count,
        "final_component_count": len(frozen.components),
        "selected_relations": len(frozen.selected_hypothesis_ids),
        "attempted_relations": frozen.attempted_count,
        "tree_merges": sum(reason == "tree" for reason in reasons),
        "cycle_acceptances": sum(reason == "cycle" for reason in reasons),
        "rejection_counts": {
            reason: reasons.count(reason)
            for reason in ("collision", "conflict", "contact", "span")
            if reasons.count(reason)
        },
        "provenance": normalized,
        "artifact": {
            "path": str(artifact),
            "bytes": len(body),
            "sha256": artifact_sha,
        },
        "components_sha256": sha256_bytes(
            _canonical_npz_bytes(
                (
                    ("component_offsets", frozen.component_offsets),
                    ("component_tiles", frozen.component_tiles),
                    ("component_positions", frozen.component_positions),
                )
            )
        ),
        "margins_order_sha256": sha256_bytes(
            _canonical_npz_bytes(
                (
                    ("selected_relations", frozen.selected_relations),
                    ("selected_scores", frozen.selected_scores),
                    ("attempted_count", np.asarray(frozen.attempted_count, dtype=np.int64)),
                    ("outcome_reason_codes", frozen.outcome_reason_codes),
                )
            )
        ),
        "labels_targets_or_metrics_opened": False,
        "e25_opened": False,
    }
    commit_canonical_create_once(commit, payload)
    return payload


def load_decode(
    commit_path: str | os.PathLike[str],
    *,
    expected_image: int | None = None,
    expected_provenance: Mapping[str, str] | None = None,
) -> tuple[FrozenDecode, dict[str, Any]]:
    commit_file = _require_e24_path(commit_path, label="decode commit")
    payload = load_canonical_json(commit_file, label="decode commit")
    expected_keys = {
        "schema",
        "schema_version",
        "status",
        "protocol_sha256",
        "image",
        "fold",
        "base_component_count",
        "final_component_count",
        "selected_relations",
        "attempted_relations",
        "tree_merges",
        "cycle_acceptances",
        "rejection_counts",
        "provenance",
        "artifact",
        "components_sha256",
        "margins_order_sha256",
        "labels_targets_or_metrics_opened",
        "e25_opened",
    }
    if (
        set(payload) != expected_keys
        or payload["schema"] != DECODE_COMMIT_SCHEMA
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["status"] != "complete_label_free"
        or payload["protocol_sha256"] != PROTOCOL_SHA256
        or payload["fold"] != _fold_for_image(payload["image"])
        or payload["labels_targets_or_metrics_opened"] is not False
        or payload["e25_opened"] is not False
    ):
        raise E24StagedContractError("decode commit identity drifted")
    if expected_image is not None and payload["image"] != expected_image:
        raise E24StagedContractError("decode commit belongs to the wrong scene")
    provenance = _normalize_provenance(payload["provenance"], keys=_DECODE_PROVENANCE_KEYS)
    if expected_provenance is not None and provenance != _normalize_provenance(
        dict(expected_provenance), keys=_DECODE_PROVENANCE_KEYS
    ):
        raise E24StagedContractError("decode commit provenance drifted")
    record = payload["artifact"]
    if type(record) is not dict or set(record) != {"path", "bytes", "sha256"}:
        raise E24StagedContractError("decode artifact record drifted")
    artifact = _require_e24_path(record["path"], label="decode artifact")
    artifact_sha = _lower_sha(record["sha256"], label="decode artifact SHA")
    if (
        not artifact.is_file()
        or artifact.stat().st_size != record["bytes"]
        or sha256_file(artifact) != artifact_sha
    ):
        raise E24StagedContractError("decode artifact provenance mismatch")
    try:
        with np.load(artifact, allow_pickle=False) as archive:
            if set(archive.files) != set(_DECODE_NPZ_NAMES):
                raise E24StagedContractError("decode NPZ member set drifted")
            if str(np.asarray(archive["schema"]).item()) != DECODE_ARTIFACT_SCHEMA:
                raise E24StagedContractError("decode NPZ schema drifted")
            if int(np.asarray(archive["schema_version"]).item()) != SCHEMA_VERSION:
                raise E24StagedContractError("decode NPZ version drifted")
            if str(np.asarray(archive["protocol_sha256"]).item()) != PROTOCOL_SHA256:
                raise E24StagedContractError("decode NPZ protocol drifted")
            value = FrozenDecode(
                image=int(np.asarray(archive["image"]).item()),
                fold=int(np.asarray(archive["fold"]).item()),
                base_component_count=int(np.asarray(archive["base_component_count"]).item()),
                component_offsets=np.array(archive["component_offsets"], copy=True, order="C"),
                component_tiles=np.array(archive["component_tiles"], copy=True, order="C"),
                component_positions=np.array(archive["component_positions"], copy=True, order="C"),
                selected_hypothesis_ids=np.array(
                    archive["selected_hypothesis_ids"], copy=True, order="C"
                ),
                selected_relation_ids=np.array(
                    archive["selected_relation_ids"], copy=True, order="C"
                ),
                selected_relations=np.array(archive["selected_relations"], copy=True, order="C"),
                selected_scores=np.array(archive["selected_scores"], copy=True, order="C"),
                selected_support=np.array(archive["selected_support"], copy=True, order="C"),
                attempted_count=int(np.asarray(archive["attempted_count"]).item()),
                outcome_reason_codes=np.array(
                    archive["outcome_reason_codes"], copy=True, order="C"
                ),
            )
    except E24StagedContractError:
        raise
    except Exception as exc:
        raise E24StagedContractError("decode NPZ is unreadable") from exc
    frozen = _validate_frozen_decode(value)
    if artifact.read_bytes() != decode_npz_bytes(frozen):
        raise E24StagedContractError("decode NPZ bytes are not canonical")
    reasons = [_CODE_TO_REASON[int(code)] for code in frozen.outcome_reason_codes]
    expected_summary = {
        "base_component_count": frozen.base_component_count,
        "final_component_count": len(frozen.components),
        "selected_relations": len(frozen.selected_hypothesis_ids),
        "attempted_relations": frozen.attempted_count,
        "tree_merges": sum(reason == "tree" for reason in reasons),
        "cycle_acceptances": sum(reason == "cycle" for reason in reasons),
        "rejection_counts": {
            reason: reasons.count(reason)
            for reason in ("collision", "conflict", "contact", "span")
            if reasons.count(reason)
        },
    }
    if any(payload[key] != expected_summary[key] for key in expected_summary):
        raise E24StagedContractError("decode commit count summary drifted")
    if payload["components_sha256"] != sha256_bytes(
        _canonical_npz_bytes(
            (
                ("component_offsets", frozen.component_offsets),
                ("component_tiles", frozen.component_tiles),
                ("component_positions", frozen.component_positions),
            )
        )
    ):
        raise E24StagedContractError("decode component digest drifted")
    if payload["margins_order_sha256"] != sha256_bytes(
        _canonical_npz_bytes(
            (
                ("selected_relations", frozen.selected_relations),
                ("selected_scores", frozen.selected_scores),
                (
                    "attempted_count",
                    np.asarray(frozen.attempted_count, dtype=np.int64),
                ),
                ("outcome_reason_codes", frozen.outcome_reason_codes),
            )
        )
    ):
        raise E24StagedContractError("decode margin/order digest drifted")
    return frozen, payload


@dataclass(frozen=True)
class FrozenBoardPair:
    image: int
    right: np.ndarray
    down: np.ndarray
    rr96_board: np.ndarray
    candidate_board: np.ndarray
    rr96_solved: np.ndarray
    candidate_solved: np.ndarray
    rr96_restored: np.ndarray
    candidate_restored: np.ndarray
    rr96_objective: float
    candidate_objective: float


_BOARD_NPZ_NAMES = (
    "schema",
    "schema_version",
    "protocol_sha256",
    "image",
    "right",
    "down",
    "rr96_board",
    "candidate_board",
    "rr96_solved",
    "candidate_solved",
    "rr96_restored",
    "candidate_restored",
    "objectives",
)


def _strict_board(value: object, *, label: str) -> np.ndarray:
    board = np.asarray(value)
    if board.shape != (NUM_TILES,) or not np.issubdtype(board.dtype, np.integer):
        raise E24StagedContractError(f"{label} must be an integer vector of length 576")
    board = np.ascontiguousarray(board, dtype=np.int64)
    if not np.array_equal(np.sort(board), np.arange(NUM_TILES, dtype=np.int64)):
        raise E24StagedContractError(f"{label} is not a tile permutation")
    board.setflags(write=False)
    return board


def _strict_rgb(value: object, *, size: int, label: str) -> np.ndarray:
    image = np.asarray(value)
    if image.shape != (size, size, 3) or image.dtype != np.uint8:
        raise E24StagedContractError(f"{label} must be uint8 RGB {size}x{size}")
    result = np.array(image, dtype=np.uint8, copy=True, order="C")
    result.setflags(write=False)
    return result


def _validate_board_pair(value: FrozenBoardPair) -> FrozenBoardPair:
    if type(value) is not FrozenBoardPair:
        raise E24StagedContractError("board artifact has the wrong exact type")
    _fold_for_image(value.image)
    matrices: list[np.ndarray] = []
    for label, matrix in (("right", value.right), ("down", value.down)):
        item = np.asarray(matrix)
        if (
            item.shape != (NUM_TILES, NUM_TILES)
            or item.dtype != np.float32
            or not item.flags.c_contiguous
            or not bool(np.isfinite(item).all())
            or bool(np.any(item < 0.0))
            or bool(np.any(np.diag(item) != 0.0))
        ):
            raise E24StagedContractError(f"{label} dense score matrix drifted")
        matrices.append(_readonly(item, np.dtype(np.float32)))
    objectives = (float(value.rr96_objective), float(value.candidate_objective))
    if not all(math.isfinite(item) for item in objectives):
        raise E24StagedContractError("board objective is non-finite")
    return FrozenBoardPair(
        image=value.image,
        right=matrices[0],
        down=matrices[1],
        rr96_board=_strict_board(value.rr96_board, label="RR96 board"),
        candidate_board=_strict_board(value.candidate_board, label="candidate board"),
        rr96_solved=_strict_rgb(value.rr96_solved, size=IMAGE_SIZE, label="RR96 solved canvas"),
        candidate_solved=_strict_rgb(
            value.candidate_solved, size=IMAGE_SIZE, label="candidate solved canvas"
        ),
        rr96_restored=_strict_rgb(
            value.rr96_restored, size=IMAGE_SIZE, label="RR96 NLM10 canvas"
        ),
        candidate_restored=_strict_rgb(
            value.candidate_restored, size=IMAGE_SIZE, label="candidate NLM10 canvas"
        ),
        rr96_objective=objectives[0],
        candidate_objective=objectives[1],
    )


def dense_rd_from_raw(candidate_ids: np.ndarray, raw_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(candidate_ids)
    scores = np.asarray(raw_logits)
    if ids.shape != (NUM_TILES, CANDIDATE_WIDTH) or ids.dtype != np.int64:
        raise E24StagedContractError("candidate IDs must be int64[576,128]")
    if scores.shape != (NUM_DIRECTIONS, NUM_TILES, CANDIDATE_WIDTH) or scores.dtype != np.float32:
        raise E24StagedContractError("raw logits must be float32[4,576,128]")
    if not ids.flags.c_contiguous or not scores.flags.c_contiguous:
        raise E24StagedContractError("raw graph arrays must be C-contiguous")
    if bool(np.any(np.isnan(scores))) or bool(np.any(np.isposinf(scores))):
        raise E24StagedContractError("raw logits allow only finite values or -inf")
    finite = np.isfinite(scores)
    if (
        not np.array_equal(finite[0], finite[1])
        or not np.array_equal(finite[0], finite[2])
        or not np.array_equal(finite[0], finite[3])
        or not bool(finite[0].any(axis=1).all())
        or not bool(np.isneginf(scores[~finite]).all())
    ):
        raise E24StagedContractError("raw-logit finite/padding masks drifted")
    for source in range(NUM_TILES):
        candidates = ids[source, finite[0, source]]
        if (
            bool(np.any(candidates < 0))
            or bool(np.any(candidates >= NUM_TILES))
            or bool(np.any(candidates == source))
            or len(np.unique(candidates)) != len(candidates)
        ):
            raise E24StagedContractError("raw candidate IDs are invalid/duplicate/self")
    try:
        import torch
        from eval_seeded_qap import dense_rd

        right_t, down_t = dense_rd(torch.from_numpy(ids).long(), torch.from_numpy(scores).float())
        right = np.ascontiguousarray(right_t.detach().cpu().numpy(), dtype=np.float32)
        down = np.ascontiguousarray(down_t.detach().cpu().numpy(), dtype=np.float32)
    except Exception as exc:
        raise E24StagedContractError("exact CPU-float32 dense_rd failed") from exc
    return right, down


def build_board_pair(
    *,
    image: int,
    candidate_ids: np.ndarray,
    raw_logits: np.ndarray,
    tiles: np.ndarray,
    decode: FrozenDecode,
    dense_builder: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] = dense_rd_from_raw,
    component_solver: Callable[..., tuple[np.ndarray, float]] | None = None,
    rr96_solver: Callable[..., tuple[np.ndarray, float]] | None = None,
    assembler: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    restorer: Callable[[np.ndarray], np.ndarray] | None = None,
) -> FrozenBoardPair:
    """Build both frozen arms without receiving a permutation or clean target."""

    frozen_decode = _validate_frozen_decode(decode)
    if frozen_decode.image != image:
        raise E24StagedContractError("decode and board scene identities differ")
    tile_array = np.asarray(tiles)
    if (
        tile_array.shape != (NUM_TILES, TILE_SIZE, TILE_SIZE, 3)
        or tile_array.dtype != np.uint8
        or not tile_array.flags.c_contiguous
    ):
        raise E24StagedContractError("tiles must be contiguous uint8[576,20,20,3]")
    if component_solver is None:
        from solve_buddies import solve_components_from_scores

        component_solver = solve_components_from_scores
    if rr96_solver is None:
        from solve_buddies import solve_buddies_from_scores

        rr96_solver = solve_buddies_from_scores
    if assembler is None:
        from imgio import assemble

        assembler = assemble
    if restorer is None:
        restorer = fixed_nlm10
    right, down = dense_builder(candidate_ids, raw_logits)
    validated_matrices = _validate_board_pair(
        FrozenBoardPair(
            image=image,
            right=np.ascontiguousarray(right, dtype=np.float32),
            down=np.ascontiguousarray(down, dtype=np.float32),
            rr96_board=np.arange(NUM_TILES, dtype=np.int64),
            candidate_board=np.arange(NUM_TILES, dtype=np.int64),
            rr96_solved=np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
            candidate_solved=np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
            rr96_restored=np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
            candidate_restored=np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
            rr96_objective=0.0,
            candidate_objective=0.0,
        )
    )
    right, down = validated_matrices.right, validated_matrices.down
    candidate_board, candidate_objective = component_solver(
        right,
        down,
        frozen_decode.components,
        repair_passes=0,
        restarts=1,
        seed=1234,
    )
    rr96_board, rr96_objective = rr96_solver(
        right,
        down,
        max_edges=96,
        min_margin=0.0,
        repair_passes=0,
    )
    candidate_board = _strict_board(candidate_board, label="candidate board")
    rr96_board = _strict_board(rr96_board, label="RR96 board")
    candidate_solved = _strict_rgb(
        assembler(tile_array, candidate_board),
        size=IMAGE_SIZE,
        label="candidate solved canvas",
    )
    rr96_solved = _strict_rgb(
        assembler(tile_array, rr96_board), size=IMAGE_SIZE, label="RR96 solved canvas"
    )
    # Both restoration arms are frozen before a metric/target capability opens.
    rr96_restored = _strict_rgb(
        restorer(np.array(rr96_solved, copy=True)),
        size=IMAGE_SIZE,
        label="RR96 NLM10 canvas",
    )
    candidate_restored = _strict_rgb(
        restorer(np.array(candidate_solved, copy=True)),
        size=IMAGE_SIZE,
        label="candidate NLM10 canvas",
    )
    return _validate_board_pair(
        FrozenBoardPair(
            image=image,
            right=right,
            down=down,
            rr96_board=rr96_board,
            candidate_board=candidate_board,
            rr96_solved=rr96_solved,
            candidate_solved=candidate_solved,
            rr96_restored=rr96_restored,
            candidate_restored=candidate_restored,
            rr96_objective=float(rr96_objective),
            candidate_objective=float(candidate_objective),
        )
    )


def _board_npz_members(value: FrozenBoardPair) -> tuple[tuple[str, np.ndarray], ...]:
    frozen = _validate_board_pair(value)
    return (
        ("schema", np.asarray(BOARD_ARTIFACT_SCHEMA)),
        ("schema_version", np.asarray(SCHEMA_VERSION, dtype=np.int16)),
        ("protocol_sha256", np.asarray(PROTOCOL_SHA256)),
        ("image", np.asarray(frozen.image, dtype=np.int16)),
        ("right", frozen.right),
        ("down", frozen.down),
        ("rr96_board", frozen.rr96_board),
        ("candidate_board", frozen.candidate_board),
        ("rr96_solved", frozen.rr96_solved),
        ("candidate_solved", frozen.candidate_solved),
        ("rr96_restored", frozen.rr96_restored),
        ("candidate_restored", frozen.candidate_restored),
        (
            "objectives",
            np.asarray([frozen.rr96_objective, frozen.candidate_objective], dtype=np.float64),
        ),
    )


def board_npz_bytes(value: FrozenBoardPair) -> bytes:
    return _canonical_npz_bytes(_board_npz_members(value))


_BOARD_PROVENANCE_KEYS = frozenset(
    {
        "ledger_sha256",
        "run_contract_sha256",
        "premetric_seal_sha256",
        "structural_report_sha256",
        "orchestration_receipt_sha256",
        "decode_commit_sha256",
        "decode_artifact_sha256",
        "raw_manifest_sha256",
        "tiles_file_sha256",
        "tiles_array_sha256",
    }
)


def commit_board_pair(
    *,
    artifact_path: str | os.PathLike[str],
    commit_path: str | os.PathLike[str],
    value: FrozenBoardPair,
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    frozen = _validate_board_pair(value)
    artifact = _require_e24_path(artifact_path, label="board artifact")
    commit = _require_e24_path(commit_path, label="board commit")
    if artifact == commit or artifact.suffix.lower() != ".npz":
        raise E24StagedContractError("board artifact/commit paths are invalid")
    body = board_npz_bytes(frozen)
    artifact_sha = _commit_npz_or_verify(artifact, body)
    payload: dict[str, Any] = {
        "schema": BOARD_COMMIT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete_label_free_board_and_nlm",
        "protocol_sha256": PROTOCOL_SHA256,
        "image": frozen.image,
        "fold": _fold_for_image(frozen.image),
        "provenance": _normalize_provenance(dict(provenance), keys=_BOARD_PROVENANCE_KEYS),
        "artifact": {"path": str(artifact), "bytes": len(body), "sha256": artifact_sha},
        "array_sha256": {
            "right": array_sha256(frozen.right),
            "down": array_sha256(frozen.down),
            "rr96_board": array_sha256(frozen.rr96_board),
            "candidate_board": array_sha256(frozen.candidate_board),
            "rr96_solved": array_sha256(frozen.rr96_solved),
            "candidate_solved": array_sha256(frozen.candidate_solved),
            "rr96_restored": array_sha256(frozen.rr96_restored),
            "candidate_restored": array_sha256(frozen.candidate_restored),
        },
        "objective": {
            "rr96": frozen.rr96_objective,
            "candidate": frozen.candidate_objective,
        },
        "orientation_degrees": 0,
        "reflection": False,
        "permutation_target_ssim_or_neighbour_opened": False,
        "e25_opened": False,
    }
    commit_canonical_create_once(commit, payload)
    return payload


def load_board_pair(
    commit_path: str | os.PathLike[str],
    *,
    expected_image: int | None = None,
    expected_provenance: Mapping[str, str] | None = None,
) -> tuple[FrozenBoardPair, dict[str, Any]]:
    payload = load_canonical_json(commit_path, label="board commit")
    expected_keys = {
        "schema",
        "schema_version",
        "status",
        "protocol_sha256",
        "image",
        "fold",
        "provenance",
        "artifact",
        "array_sha256",
        "objective",
        "orientation_degrees",
        "reflection",
        "permutation_target_ssim_or_neighbour_opened",
        "e25_opened",
    }
    if (
        set(payload) != expected_keys
        or payload["schema"] != BOARD_COMMIT_SCHEMA
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["status"] != "complete_label_free_board_and_nlm"
        or payload["protocol_sha256"] != PROTOCOL_SHA256
        or payload["fold"] != _fold_for_image(payload["image"])
        or payload["orientation_degrees"] != 0
        or payload["reflection"] is not False
        or payload["permutation_target_ssim_or_neighbour_opened"] is not False
        or payload["e25_opened"] is not False
    ):
        raise E24StagedContractError("board commit identity drifted")
    if expected_image is not None and payload["image"] != expected_image:
        raise E24StagedContractError("board commit belongs to the wrong scene")
    provenance = _normalize_provenance(payload["provenance"], keys=_BOARD_PROVENANCE_KEYS)
    if expected_provenance is not None and provenance != _normalize_provenance(
        dict(expected_provenance), keys=_BOARD_PROVENANCE_KEYS
    ):
        raise E24StagedContractError("board commit provenance drifted")
    record = payload["artifact"]
    if type(record) is not dict or set(record) != {"path", "bytes", "sha256"}:
        raise E24StagedContractError("board artifact record drifted")
    artifact = _require_e24_path(record["path"], label="board artifact")
    artifact_sha = _lower_sha(record["sha256"], label="board artifact SHA")
    if (
        not artifact.is_file()
        or artifact.stat().st_size != record["bytes"]
        or sha256_file(artifact) != artifact_sha
    ):
        raise E24StagedContractError("board artifact provenance mismatch")
    try:
        with np.load(artifact, allow_pickle=False) as archive:
            if set(archive.files) != set(_BOARD_NPZ_NAMES):
                raise E24StagedContractError("board NPZ member set drifted")
            if str(np.asarray(archive["schema"]).item()) != BOARD_ARTIFACT_SCHEMA:
                raise E24StagedContractError("board NPZ schema drifted")
            if int(np.asarray(archive["schema_version"]).item()) != SCHEMA_VERSION:
                raise E24StagedContractError("board NPZ version drifted")
            if str(np.asarray(archive["protocol_sha256"]).item()) != PROTOCOL_SHA256:
                raise E24StagedContractError("board NPZ protocol drifted")
            objectives = np.asarray(archive["objectives"])
            if objectives.dtype != np.float64 or objectives.shape != (2,):
                raise E24StagedContractError("board objective vector drifted")
            value = FrozenBoardPair(
                image=int(np.asarray(archive["image"]).item()),
                right=np.array(archive["right"], copy=True, order="C"),
                down=np.array(archive["down"], copy=True, order="C"),
                rr96_board=np.array(archive["rr96_board"], copy=True, order="C"),
                candidate_board=np.array(archive["candidate_board"], copy=True, order="C"),
                rr96_solved=np.array(archive["rr96_solved"], copy=True, order="C"),
                candidate_solved=np.array(archive["candidate_solved"], copy=True, order="C"),
                rr96_restored=np.array(archive["rr96_restored"], copy=True, order="C"),
                candidate_restored=np.array(archive["candidate_restored"], copy=True, order="C"),
                rr96_objective=float(objectives[0]),
                candidate_objective=float(objectives[1]),
            )
    except E24StagedContractError:
        raise
    except Exception as exc:
        raise E24StagedContractError("board NPZ is unreadable") from exc
    frozen = _validate_board_pair(value)
    if artifact.read_bytes() != board_npz_bytes(frozen):
        raise E24StagedContractError("board NPZ bytes are not canonical")
    hashes = payload["array_sha256"]
    expected_hashes = {
        "right": array_sha256(frozen.right),
        "down": array_sha256(frozen.down),
        "rr96_board": array_sha256(frozen.rr96_board),
        "candidate_board": array_sha256(frozen.candidate_board),
        "rr96_solved": array_sha256(frozen.rr96_solved),
        "candidate_solved": array_sha256(frozen.candidate_solved),
        "rr96_restored": array_sha256(frozen.rr96_restored),
        "candidate_restored": array_sha256(frozen.candidate_restored),
    }
    if hashes != expected_hashes or payload["objective"] != {
        "rr96": frozen.rr96_objective,
        "candidate": frozen.candidate_objective,
    }:
        raise E24StagedContractError("board commit hash/objective summary drifted")
    return frozen, payload


_ARM_METRIC_KEYS = frozenset(
    {
        "objective",
        "placement",
        "neighbour",
        "right",
        "down",
        "solve_only_ssim",
        "final_ssim",
        "board_sha256",
        "solved_corrupted_canvas_sha256",
        "restored_canvas_sha256",
    }
)


def _default_ssim(first: np.ndarray, second: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(first, second, channel_axis=2, data_range=255))


def _measure_arm(
    *,
    board: np.ndarray,
    solved: np.ndarray,
    restored: np.ndarray,
    objective: float,
    truth_board: np.ndarray,
    target: np.ndarray,
    ssim: Callable[[np.ndarray, np.ndarray], float],
) -> dict[str, Any]:
    from placement_metrics import neighbour_accuracy, placement_accuracy

    placement = placement_accuracy(board, truth_board)[0]
    neighbour, right, down = neighbour_accuracy(board, truth_board)
    values = {
        "objective": float(objective),
        "placement": float(placement),
        "neighbour": float(neighbour),
        "right": float(right),
        "down": float(down),
        "solve_only_ssim": float(ssim(target, solved)),
        "final_ssim": float(ssim(target, restored)),
        "board_sha256": array_sha256(board),
        "solved_corrupted_canvas_sha256": array_sha256(solved),
        "restored_canvas_sha256": array_sha256(restored),
    }
    if not all(
        math.isfinite(float(values[key]))
        for key in (
            "objective",
            "placement",
            "neighbour",
            "right",
            "down",
            "solve_only_ssim",
            "final_ssim",
        )
    ):
        raise E24StagedContractError("scene metric is non-finite")
    if any(not 0.0 <= float(values[key]) <= 1.0 for key in ("placement", "neighbour", "right", "down")):
        raise E24StagedContractError("placement/neighbour metric is outside [0,1]")
    return values


def measure_scene(
    *,
    boards: FrozenBoardPair,
    permutation: np.ndarray,
    target: np.ndarray,
    validation_name: str,
    provenance: Mapping[str, str],
    ssim: Callable[[np.ndarray, np.ndarray], float] = _default_ssim,
) -> dict[str, Any]:
    """Measure one already-frozen board pair; no loader exists in this module."""

    frozen = _validate_board_pair(boards)
    perm = np.asarray(permutation)
    if (
        perm.shape != (NUM_TILES,)
        or perm.dtype != np.int64
        or not perm.flags.c_contiguous
        or not np.array_equal(np.sort(perm), np.arange(NUM_TILES, dtype=np.int64))
    ):
        raise E24StagedContractError("metric permutation must be contiguous int64[576]")
    clean = _strict_rgb(target, size=IMAGE_SIZE, label="clean metric target")
    if type(validation_name) is not str or not validation_name:
        raise E24StagedContractError("validation name is absent")
    truth_board = np.ascontiguousarray(np.argsort(perm), dtype=np.int64)
    rr96 = _measure_arm(
        board=frozen.rr96_board,
        solved=frozen.rr96_solved,
        restored=frozen.rr96_restored,
        objective=frozen.rr96_objective,
        truth_board=truth_board,
        target=clean,
        ssim=ssim,
    )
    candidate = _measure_arm(
        board=frozen.candidate_board,
        solved=frozen.candidate_solved,
        restored=frozen.candidate_restored,
        objective=frozen.candidate_objective,
        truth_board=truth_board,
        target=clean,
        ssim=ssim,
    )
    delta = {
        "solve_only_ssim": candidate["solve_only_ssim"] - rr96["solve_only_ssim"],
        "final_ssim": candidate["final_ssim"] - rr96["final_ssim"],
        "neighbour": candidate["neighbour"] - rr96["neighbour"],
    }
    if not all(math.isfinite(float(item)) for item in delta.values()):
        raise E24StagedContractError("paired delta is non-finite")
    normalized_provenance = {
        str(key): _lower_sha(value, label=f"scene provenance {key}")
        for key, value in sorted(dict(provenance).items())
    }
    if not normalized_provenance:
        raise E24StagedContractError("scene metric provenance is empty")
    return {
        "image": frozen.image,
        "fold": _fold_for_image(frozen.image),
        "validation_name": validation_name,
        "orientation_degrees": 0,
        "reflection": False,
        "provenance": normalized_provenance,
        "permutation_sha256": array_sha256(perm),
        "target_sha256": array_sha256(clean),
        "rr96": rr96,
        "candidate": candidate,
        "delta": delta,
    }


_BROKER_ROW_PROVENANCE_KEYS = frozenset(
    {
        "board_barrier_sha256",
        "board_commit_sha256",
        "metric_broker_contract_sha256",
        "metric_request_sha256",
        "premetric_seal_sha256",
        "raw_archive_sha256",
        "e12_report_sha256",
        "calibration_report_sha256",
        "scene_provenance_digest",
    }
)


def measure_scene_with_pinned_rr96(
    *,
    boards: FrozenBoardPair,
    permutation: np.ndarray,
    target: np.ndarray,
    validation_name: str,
    provenance: Mapping[str, str],
    pinned_rr96: Mapping[str, Any],
    expected_permutation_sha256: str,
    expected_target_sha256: str,
    ssim: Callable[[np.ndarray, np.ndarray], float] = _default_ssim,
) -> dict[str, Any]:
    """Measure CRS while reusing the exact pinned E12 RR96 metric record.

    E14 froze this choice specifically to avoid a second RR96 NLM call.  The
    committed E24 RR96 board and both canvases must nevertheless match the E12
    hashes byte-for-byte, and its geometry is independently recomputed from the
    allowlisted permutation member.
    """

    frozen = _validate_board_pair(boards)
    perm = np.asarray(permutation)
    if (
        perm.shape != (NUM_TILES,)
        or perm.dtype != np.int64
        or not perm.flags.c_contiguous
        or not np.array_equal(np.sort(perm), np.arange(NUM_TILES, dtype=np.int64))
    ):
        raise E24StagedContractError("metric permutation must be contiguous int64[576]")
    clean = _strict_rgb(target, size=IMAGE_SIZE, label="clean metric target")
    if type(validation_name) is not str or not validation_name:
        raise E24StagedContractError("validation name is absent")
    if array_sha256(perm) != _lower_sha(
        expected_permutation_sha256, label="pinned permutation SHA"
    ):
        raise E24StagedContractError("permutation differs from pinned E12 lineage")
    if array_sha256(clean) != _lower_sha(expected_target_sha256, label="pinned target SHA"):
        raise E24StagedContractError("clean target differs from pinned E12 lineage")

    if type(pinned_rr96) is not dict or set(pinned_rr96) != _ARM_METRIC_KEYS:
        raise E24StagedContractError("pinned RR96 metric field set drifted")
    rr96: dict[str, Any] = {}
    for key in sorted(_ARM_METRIC_KEYS):
        value = pinned_rr96[key]
        if key.endswith("sha256"):
            rr96[key] = _lower_sha(value, label=f"pinned RR96 {key}")
        elif type(value) not in {float, int} or not math.isfinite(float(value)):
            raise E24StagedContractError(f"pinned RR96 {key} is non-finite/non-numeric")
        else:
            rr96[key] = float(value)
    expected_hashes = {
        "board_sha256": array_sha256(frozen.rr96_board),
        "solved_corrupted_canvas_sha256": array_sha256(frozen.rr96_solved),
        "restored_canvas_sha256": array_sha256(frozen.rr96_restored),
    }
    if any(rr96[key] != value for key, value in expected_hashes.items()):
        raise E24StagedContractError("committed RR96 bytes differ from pinned E12 record")
    if not math.isclose(
        rr96["objective"], frozen.rr96_objective, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise E24StagedContractError("committed RR96 objective differs from pinned E12")

    from placement_metrics import neighbour_accuracy, placement_accuracy

    truth_board = np.ascontiguousarray(np.argsort(perm), dtype=np.int64)
    placement = float(placement_accuracy(frozen.rr96_board, truth_board)[0])
    neighbour, right, down = neighbour_accuracy(frozen.rr96_board, truth_board)
    geometry = {
        "placement": placement,
        "neighbour": float(neighbour),
        "right": float(right),
        "down": float(down),
    }
    if any(rr96[key] != value for key, value in geometry.items()):
        raise E24StagedContractError("pinned RR96 geometry differs from permutation replay")

    candidate = _measure_arm(
        board=frozen.candidate_board,
        solved=frozen.candidate_solved,
        restored=frozen.candidate_restored,
        objective=frozen.candidate_objective,
        truth_board=truth_board,
        target=clean,
        ssim=ssim,
    )
    normalized_provenance = _normalize_provenance(
        dict(provenance), keys=_BROKER_ROW_PROVENANCE_KEYS
    )
    delta = {
        key: float(candidate[key]) - float(rr96[key])
        for key in ("solve_only_ssim", "final_ssim", "neighbour")
    }
    if not all(math.isfinite(value) for value in delta.values()):
        raise E24StagedContractError("paired delta is non-finite")
    return {
        "image": frozen.image,
        "fold": _fold_for_image(frozen.image),
        "validation_name": validation_name,
        "orientation_degrees": 0,
        "reflection": False,
        "provenance": normalized_provenance,
        "permutation_sha256": array_sha256(perm),
        "target_sha256": array_sha256(clean),
        "rr96": rr96,
        "candidate": candidate,
        "delta": delta,
    }


def _validate_scene_row(row: object) -> dict[str, Any]:
    expected = {
        "image",
        "fold",
        "validation_name",
        "orientation_degrees",
        "reflection",
        "provenance",
        "permutation_sha256",
        "target_sha256",
        "rr96",
        "candidate",
        "delta",
    }
    if type(row) is not dict or set(row) != expected:
        raise E24StagedContractError("staged scene row field set drifted")
    if (
        row["fold"] != _fold_for_image(row["image"])
        or type(row["validation_name"]) is not str
        or not row["validation_name"]
        or row["orientation_degrees"] != 0
        or row["reflection"] is not False
    ):
        raise E24StagedContractError("staged scene identity/orientation drifted")
    _lower_sha(row["permutation_sha256"], label="permutation SHA")
    _lower_sha(row["target_sha256"], label="target SHA")
    if type(row["provenance"]) is not dict or not row["provenance"]:
        raise E24StagedContractError("staged scene provenance is empty")
    for key, value in row["provenance"].items():
        if type(key) is not str:
            raise E24StagedContractError("staged scene provenance key is not text")
        _lower_sha(value, label=f"scene provenance {key}")
    for arm in ("rr96", "candidate"):
        values = row[arm]
        if type(values) is not dict or set(values) != _ARM_METRIC_KEYS:
            raise E24StagedContractError(f"{arm} metric field set drifted")
        for key in _ARM_METRIC_KEYS:
            if key.endswith("sha256"):
                _lower_sha(values[key], label=f"{arm} {key}")
            else:
                if type(values[key]) not in {float, int} or not math.isfinite(float(values[key])):
                    raise E24StagedContractError(f"{arm} {key} is non-finite/non-numeric")
        if any(
            not 0.0 <= float(values[key]) <= 1.0
            for key in ("placement", "neighbour", "right", "down")
        ):
            raise E24StagedContractError(f"{arm} placement/neighbour metric is invalid")
    expected_delta = {
        key: float(row["candidate"][key]) - float(row["rr96"][key])
        for key in ("solve_only_ssim", "final_ssim", "neighbour")
    }
    if type(row["delta"]) is not dict or set(row["delta"]) != set(expected_delta):
        raise E24StagedContractError("paired delta field set drifted")
    if any(
        type(row["delta"][key]) not in {float, int}
        or not math.isfinite(float(row["delta"][key]))
        or float(row["delta"][key]) != expected_delta[key]
        for key in expected_delta
    ):
        raise E24StagedContractError("paired delta value drifted")
    return dict(row)


def summarize_staged(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [_validate_scene_row(dict(row)) for row in rows]
    if len(values) != 8 or [row["image"] for row in values] != list(CALIBRATION_IDS):
        raise E24StagedContractError("staged summary requires ordered scenes 10..17 exactly once")

    def mean(items: Sequence[float]) -> float:
        return float(math.fsum(map(float, items)) / len(items))

    metric_names = (
        "placement",
        "neighbour",
        "right",
        "down",
        "solve_only_ssim",
        "final_ssim",
        "objective",
    )
    baseline_means = {
        key: mean([row["rr96"][key] for row in values]) for key in metric_names
    }
    candidate_means = {
        key: mean([row["candidate"][key] for row in values]) for key in metric_names
    }
    solve_delta = [float(row["delta"]["solve_only_ssim"]) for row in values]
    final_delta = [float(row["delta"]["final_ssim"]) for row in values]
    neighbour_delta = [float(row["delta"]["neighbour"]) for row in values]
    output = {
        "completed_scenes": 8,
        "baseline_means": baseline_means,
        "candidate_means": candidate_means,
        "mean_solve_only_ssim_delta": mean(solve_delta),
        "mean_final_ssim_delta": mean(final_delta),
        "final_ssim_wins": sum(value > 0.0 for value in final_delta),
        "worst_final_ssim_delta": min(final_delta),
        "mean_neighbour_delta": mean(neighbour_delta),
    }
    if not all(
        math.isfinite(float(value))
        for key, value in output.items()
        if key not in {"completed_scenes", "final_ssim_wins", "baseline_means", "candidate_means"}
    ):
        raise E24StagedContractError("staged summary contains a non-finite value")
    return output


def staged_decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "completed_scenes",
        "baseline_means",
        "candidate_means",
        "mean_solve_only_ssim_delta",
        "mean_final_ssim_delta",
        "final_ssim_wins",
        "worst_final_ssim_delta",
        "mean_neighbour_delta",
    }
    if type(summary) is not dict or set(summary) != expected:
        raise E24StagedContractError("staged summary field set drifted")
    numeric = {
        key: float(summary[key])
        for key in (
            "mean_solve_only_ssim_delta",
            "mean_final_ssim_delta",
            "worst_final_ssim_delta",
            "mean_neighbour_delta",
        )
    }
    if any(not math.isfinite(value) for value in numeric.values()):
        raise E24StagedContractError("staged decision input is non-finite")
    if type(summary["completed_scenes"]) is not int or type(summary["final_ssim_wins"]) is not int:
        raise E24StagedContractError("staged count is not an exact integer")
    checks = {
        "completed_scenes": summary["completed_scenes"] == 8,
        "mean_solve_only_ssim_delta": numeric["mean_solve_only_ssim_delta"]
        >= float(END_TO_END_GATES["solve_ssim_delta_mean_min"]),
        "mean_final_ssim_delta": numeric["mean_final_ssim_delta"]
        >= float(END_TO_END_GATES["final_ssim_delta_mean_min"]),
        "strict_positive_final_ssim_wins": summary["final_ssim_wins"]
        >= int(END_TO_END_GATES["final_wins_min"]),
        "worst_final_ssim_delta": numeric["worst_final_ssim_delta"]
        >= float(END_TO_END_GATES["worst_final_delta_min"]),
        "mean_neighbour_delta": numeric["mean_neighbour_delta"]
        >= float(END_TO_END_GATES["neighbour_delta_mean_min"]),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "stage": "go_final_all8_fit" if passed else "kill_crs_v1",
        "checks": checks,
        "thresholds": dict(END_TO_END_GATES),
    }


_RR96_VERIFICATION_KEYS = frozenset(
    {
        "passed",
        "source",
        "e12_report_sha256",
        "calibration_report_sha256",
        "scene_provenance_digest",
        "absolute_tolerance",
        "expected_mean_solve_ssim",
        "observed_mean_solve_ssim",
        "expected_mean_final_ssim",
        "observed_mean_final_ssim",
        "board_hashes",
        "solved_corrupted_canvas_hashes",
        "restored_canvas_hashes",
    }
)


def rr96_verification_for_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values = [_validate_scene_row(dict(row)) for row in rows]
    if [row["image"] for row in values] != list(CALIBRATION_IDS):
        raise E24StagedContractError("RR96 verification requires ordered scenes 10..17")
    solve_mean = float(
        np.mean(np.asarray([row["rr96"]["solve_only_ssim"] for row in values], dtype=np.float64))
    )
    final_mean = float(
        np.mean(np.asarray([row["rr96"]["final_ssim"] for row in values], dtype=np.float64))
    )
    if not math.isclose(
        solve_mean,
        PINNED_RR96_MEAN_SOLVE_SSIM,
        rel_tol=0.0,
        abs_tol=RR96_MEAN_ABSOLUTE_TOLERANCE,
    ) or not math.isclose(
        final_mean,
        PINNED_RR96_MEAN_FINAL_SSIM,
        rel_tol=0.0,
        abs_tol=RR96_MEAN_ABSOLUTE_TOLERANCE,
    ):
        raise E24StagedContractError("RR96 means differ from the pinned E12 record")
    return {
        "passed": True,
        "source": "exact_E12_RR_record_no_second_NLM_call",
        "e12_report_sha256": PINNED_E12_REPORT_SHA256,
        "calibration_report_sha256": PINNED_CALIBRATION_REPORT_SHA256,
        "scene_provenance_digest": PINNED_SCENE_PROVENANCE_DIGEST,
        "absolute_tolerance": RR96_MEAN_ABSOLUTE_TOLERANCE,
        "expected_mean_solve_ssim": PINNED_RR96_MEAN_SOLVE_SSIM,
        "observed_mean_solve_ssim": solve_mean,
        "expected_mean_final_ssim": PINNED_RR96_MEAN_FINAL_SSIM,
        "observed_mean_final_ssim": final_mean,
        "board_hashes": {
            str(row["image"]): row["rr96"]["board_sha256"] for row in values
        },
        "solved_corrupted_canvas_hashes": {
            str(row["image"]): row["rr96"]["solved_corrupted_canvas_sha256"]
            for row in values
        },
        "restored_canvas_hashes": {
            str(row["image"]): row["rr96"]["restored_canvas_sha256"]
            for row in values
        },
    }


def _normalize_board_commit_sha256(value: object) -> dict[str, str]:
    if type(value) is not dict or set(value) != {str(image) for image in CALIBRATION_IDS}:
        raise E24StagedContractError("board-commit SHA map must contain string IDs 10..17")
    return {
        str(image): _lower_sha(value[str(image)], label=f"scene {image} board commit SHA")
        for image in CALIBRATION_IDS
    }


def _validated_report_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    board_barrier_sha256: str,
    board_commit_sha256: Mapping[str, str],
    premetric_seal_sha256: str,
) -> list[dict[str, Any]]:
    values = [_validate_scene_row(dict(row)) for row in rows]
    if len(values) != len(CALIBRATION_IDS) or [row["image"] for row in values] != list(
        CALIBRATION_IDS
    ):
        raise E24StagedContractError("staged report requires ordered scenes 10..17")
    seen_requests: set[str] = set()
    for row in values:
        image = row["image"]
        provenance = _normalize_provenance(
            row["provenance"], keys=_BROKER_ROW_PROVENANCE_KEYS
        )
        expected = {
            "board_barrier_sha256": board_barrier_sha256,
            "board_commit_sha256": board_commit_sha256[str(image)],
            "metric_broker_contract_sha256": METRIC_BROKER_CONTRACT_SHA256,
            "premetric_seal_sha256": premetric_seal_sha256,
            "e12_report_sha256": PINNED_E12_REPORT_SHA256,
            "calibration_report_sha256": PINNED_CALIBRATION_REPORT_SHA256,
            "scene_provenance_digest": PINNED_SCENE_PROVENANCE_DIGEST,
        }
        if any(provenance[key] != value for key, value in expected.items()):
            raise E24StagedContractError("staged row authority/provenance drifted")
        request_sha = provenance["metric_request_sha256"]
        if request_sha in seen_requests:
            raise E24StagedContractError("metric request replay/duplication detected")
        seen_requests.add(request_sha)
    return values


_METRIC_REQUEST_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "image",
        "sequence_index",
        "previous_response_path",
        "previous_response_sha256",
        "ledger_sha256",
        "run_contract_sha256",
        "premetric_seal_sha256",
        "structural_report_sha256",
        "orchestration_receipt_sha256",
        "board_barrier_sha256",
        "board_commit_path",
        "board_commit_sha256",
        "raw_archive_path",
        "raw_archive_sha256",
        "validation_name",
        "metric_broker_contract_sha256",
        "e12_report_sha256",
        "calibration_report_sha256",
        "scene_provenance_digest",
        "e25_opened",
    }
)
_METRIC_RESPONSE_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        "image",
        "sequence_index",
        "request_sha256",
        "previous_response_sha256",
        "row",
        "arrays_exported",
        "e25_opened",
    }
)
DEFAULT_METRIC_RESPONSE_ROOT = Path(
    "E:/pazzle_work/posegraph_e24_selector/staged_v1/metric_broker"
)
DEFAULT_STAGED_BOARD_ROOT = Path(
    "E:/pazzle_work/posegraph_e24_selector/staged_v1"
)


def _authenticate_metric_chain(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_response_root: str | os.PathLike[str],
    ledger_sha256: str,
    run_contract_sha256: str,
    premetric_seal_sha256: str,
    structural_report_sha256: str,
    orchestration_receipt_sha256: str,
    board_barrier_sha256: str,
    board_commit_sha256: Mapping[str, str],
) -> None:
    root = _require_e24_path(metric_response_root, label="metric response root")
    production_root = root == DEFAULT_METRIC_RESPONSE_ROOT.resolve(strict=False)
    previous_sha = METRIC_CHAIN_GENESIS_SHA256
    previous_path = ""
    for sequence, row in enumerate(rows):
        image = CALIBRATION_IDS[sequence]
        scene_root = root / f"image_{image:04d}"
        request_path = _require_e24_path(
            scene_root / "request.json", label="metric request"
        )
        response_path = _require_e24_path(
            scene_root / "response.json", label="metric response"
        )
        request = load_canonical_json(request_path, label="metric request")
        response = load_canonical_json(response_path, label="metric response")
        if type(request) is not dict or set(request) != _METRIC_REQUEST_KEYS:
            raise E24StagedContractError("metric request field set drifted")
        expected_board_path = (
            DEFAULT_STAGED_BOARD_ROOT
            / f"image_{image:04d}"
            / "board_nlm.commit.json"
        ).resolve(strict=False)
        try:
            raw_path = e24_eval._require_e_drive(
                request["raw_archive_path"], label="metric raw archive"
            )
        except e24_eval.E24EvaluatorContractError as exc:
            raise E24StagedContractError(str(exc)) from exc
        if production_root and raw_path != Path(
            f"E:/pazzle_work/edge_confidence/full_graph_cache/image_{image:04d}_k64.npz"
        ).resolve(strict=False):
            raise E24StagedContractError("metric raw archive path is not the pinned RR96 cache")
        request_expected = {
            "schema": METRIC_REQUEST_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "image": image,
            "sequence_index": sequence,
            "previous_response_path": previous_path,
            "previous_response_sha256": previous_sha,
            "ledger_sha256": ledger_sha256,
            "run_contract_sha256": run_contract_sha256,
            "premetric_seal_sha256": premetric_seal_sha256,
            "structural_report_sha256": structural_report_sha256,
            "orchestration_receipt_sha256": orchestration_receipt_sha256,
            "board_barrier_sha256": board_barrier_sha256,
            "board_commit_path": str(expected_board_path),
            "board_commit_sha256": board_commit_sha256[str(image)],
            "raw_archive_path": str(raw_path),
            "raw_archive_sha256": row["provenance"]["raw_archive_sha256"],
            "validation_name": row["validation_name"],
            "metric_broker_contract_sha256": METRIC_BROKER_CONTRACT_SHA256,
            "e12_report_sha256": PINNED_E12_REPORT_SHA256,
            "calibration_report_sha256": PINNED_CALIBRATION_REPORT_SHA256,
            "scene_provenance_digest": PINNED_SCENE_PROVENANCE_DIGEST,
            "e25_opened": False,
        }
        if request != request_expected:
            raise E24StagedContractError("metric request authority/order drifted")
        if (
            not raw_path.is_file()
            or sha256_file(raw_path) != request["raw_archive_sha256"]
        ):
            raise E24StagedContractError("metric raw archive whole-file SHA drifted")
        request_sha = sha256_file(request_path)
        if request_sha != row["provenance"]["metric_request_sha256"]:
            raise E24StagedContractError("metric request SHA differs from report row")
        response_expected = {
            "schema": METRIC_RESPONSE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": "complete_row_only",
            "image": image,
            "sequence_index": sequence,
            "request_sha256": request_sha,
            "previous_response_sha256": previous_sha,
            "row": row,
            "arrays_exported": False,
            "e25_opened": False,
        }
        if (
            type(response) is not dict
            or set(response) != _METRIC_RESPONSE_KEYS
            or response != response_expected
        ):
            raise E24StagedContractError("metric response/row binding drifted")
        previous_sha = sha256_file(response_path)
        previous_path = str(response_path.resolve(strict=False))


def build_staged_report(
    *,
    ledger_sha256: str,
    run_contract_sha256: str,
    premetric_seal_sha256: str,
    structural_report_sha256: str,
    orchestration_receipt_sha256: str,
    board_barrier_sha256: str,
    board_commit_sha256: Mapping[str, str],
    rows: Sequence[Mapping[str, Any]],
    rr96_verification: Mapping[str, Any],
) -> dict[str, Any]:
    ledger = _lower_sha(ledger_sha256, label="ledger SHA")
    run_contract = _lower_sha(run_contract_sha256, label="run-contract SHA")
    seal = _lower_sha(premetric_seal_sha256, label="premetric seal SHA")
    structural = _lower_sha(structural_report_sha256, label="structural report SHA")
    receipt = _lower_sha(
        orchestration_receipt_sha256, label="orchestration receipt SHA"
    )
    barrier = _lower_sha(board_barrier_sha256, label="board barrier SHA")
    board_hashes = _normalize_board_commit_sha256(dict(board_commit_sha256))
    normalized_rows = _validated_report_rows(
        rows,
        board_barrier_sha256=barrier,
        board_commit_sha256=board_hashes,
        premetric_seal_sha256=seal,
    )
    expected_rr = rr96_verification_for_rows(normalized_rows)
    if type(rr96_verification) is not dict or set(rr96_verification) != _RR96_VERIFICATION_KEYS:
        raise E24StagedContractError("RR96 verification field set drifted")
    if dict(rr96_verification) != expected_rr:
        raise E24StagedContractError("RR96 verification differs from exact row replay")
    summary = summarize_staged(normalized_rows)
    decision = staged_decision(summary)
    return {
        "schema": STAGED_REPORT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "stage": decision["stage"],
        "staged_protocol_sha256": PROTOCOL_SHA256,
        "metric_broker_contract_sha256": METRIC_BROKER_CONTRACT_SHA256,
        "ledger_sha256": ledger,
        "run_contract_sha256": run_contract,
        "premetric_seal_sha256": seal,
        "structural_report_sha256": structural,
        "orchestration_receipt_sha256": receipt,
        "board_barrier_sha256": barrier,
        "board_commit_sha256": board_hashes,
        "rr96_verification": expected_rr,
        "rows": normalized_rows,
        "summary": summary,
        "decision": decision,
        "e25_opened": False,
    }


def validate_staged_report(
    path: str | os.PathLike[str],
    *,
    expected_ledger_sha256: str,
    expected_run_contract_sha256: str,
    expected_premetric_seal_sha256: str,
    expected_structural_report_sha256: str,
    expected_orchestration_receipt_sha256: str,
    expected_board_barrier_sha256: str,
    expected_board_commit_sha256: Mapping[str, str] | None = None,
    expected_metric_response_root: str | os.PathLike[str] = DEFAULT_METRIC_RESPONSE_ROOT,
) -> dict[str, Any]:
    payload = load_canonical_json(path, label="staged SSIM/NLM report")
    expected_keys = {
        "schema",
        "schema_version",
        "status",
        "stage",
        "staged_protocol_sha256",
        "metric_broker_contract_sha256",
        "ledger_sha256",
        "run_contract_sha256",
        "premetric_seal_sha256",
        "structural_report_sha256",
        "orchestration_receipt_sha256",
        "board_barrier_sha256",
        "board_commit_sha256",
        "rr96_verification",
        "rows",
        "summary",
        "decision",
        "e25_opened",
    }
    expected_authority = {
        "ledger_sha256": _lower_sha(expected_ledger_sha256, label="expected ledger SHA"),
        "run_contract_sha256": _lower_sha(
            expected_run_contract_sha256, label="expected run-contract SHA"
        ),
        "premetric_seal_sha256": _lower_sha(
            expected_premetric_seal_sha256, label="expected premetric seal SHA"
        ),
        "structural_report_sha256": _lower_sha(
            expected_structural_report_sha256, label="expected structural report SHA"
        ),
        "orchestration_receipt_sha256": _lower_sha(
            expected_orchestration_receipt_sha256,
            label="expected orchestration receipt SHA",
        ),
        "board_barrier_sha256": _lower_sha(
            expected_board_barrier_sha256, label="expected board barrier SHA"
        ),
    }
    if (
        set(payload) != expected_keys
        or payload["schema"] != STAGED_REPORT_SCHEMA
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["status"] != "complete"
        or payload["staged_protocol_sha256"] != PROTOCOL_SHA256
        or payload["metric_broker_contract_sha256"] != METRIC_BROKER_CONTRACT_SHA256
        or payload["e25_opened"] is not False
        or any(payload[key] != value for key, value in expected_authority.items())
    ):
        raise E24StagedContractError("staged report identity/authority drifted")
    board_hashes = _normalize_board_commit_sha256(payload["board_commit_sha256"])
    if expected_board_commit_sha256 is not None and board_hashes != _normalize_board_commit_sha256(
        dict(expected_board_commit_sha256)
    ):
        raise E24StagedContractError("staged report board-commit map drifted")
    rebuilt = build_staged_report(
        ledger_sha256=payload["ledger_sha256"],
        run_contract_sha256=payload["run_contract_sha256"],
        premetric_seal_sha256=payload["premetric_seal_sha256"],
        structural_report_sha256=payload["structural_report_sha256"],
        orchestration_receipt_sha256=payload["orchestration_receipt_sha256"],
        board_barrier_sha256=payload["board_barrier_sha256"],
        board_commit_sha256=board_hashes,
        rows=payload["rows"],
        rr96_verification=payload["rr96_verification"],
    )
    if payload != rebuilt:
        raise E24StagedContractError("staged report summary/decision was not exactly recomputed")
    _authenticate_metric_chain(
        payload["rows"],
        metric_response_root=expected_metric_response_root,
        ledger_sha256=payload["ledger_sha256"],
        run_contract_sha256=payload["run_contract_sha256"],
        premetric_seal_sha256=payload["premetric_seal_sha256"],
        structural_report_sha256=payload["structural_report_sha256"],
        orchestration_receipt_sha256=payload["orchestration_receipt_sha256"],
        board_barrier_sha256=payload["board_barrier_sha256"],
        board_commit_sha256=board_hashes,
    )
    return payload


__all__ = (
    "BOARD_ARTIFACT_SCHEMA",
    "BOARD_COMMIT_SCHEMA",
    "CALIBRATION_IDS",
    "DECODE_ARTIFACT_SCHEMA",
    "DECODE_COMMIT_SCHEMA",
    "END_TO_END_GATES",
    "E24StagedContractError",
    "FrozenBoardPair",
    "FrozenDecode",
    "OOF_FOLDS",
    "METRIC_BROKER_CONTRACT",
    "METRIC_BROKER_CONTRACT_SHA256",
    "METRIC_CHAIN_GENESIS_SHA256",
    "METRIC_REQUEST_SCHEMA",
    "METRIC_RESPONSE_SCHEMA",
    "PREMETRIC_SEAL_SCHEMA",
    "PROTOCOL_SHA256",
    "STAGED_PROTOCOL",
    "STAGED_REPORT_SCHEMA",
    "array_sha256",
    "board_npz_bytes",
    "build_board_pair",
    "build_staged_report",
    "canonical_json_bytes",
    "commit_board_pair",
    "commit_canonical_create_once",
    "commit_canonical_or_verify",
    "commit_decode",
    "decode_npz_bytes",
    "dense_rd_from_raw",
    "freeze_decode_result",
    "load_board_pair",
    "load_canonical_json",
    "load_decode",
    "measure_scene",
    "measure_scene_with_pinned_rr96",
    "rr96_verification_for_rows",
    "sha256_bytes",
    "sha256_file",
    "staged_decision",
    "summarize_staged",
    "validate_staged_report",
)
