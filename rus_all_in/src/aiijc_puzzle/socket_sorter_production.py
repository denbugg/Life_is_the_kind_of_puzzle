"""Fail-closed inference and resumable packaging for the Socket tile sorter.

This module is deliberately layout-only.  A corresponding dirty RGB input is
split into its 576 upright 20x20 tiles, SocketMatcher predicts two partial
socket assignments, decoder144 returns a strict permutation, and the optional
confirmed cyclic-border5 anchor chooses a global origin.  The raw output is
assembled only from the original tiles and audited before a separately named
pixel-tail hook runs.

The identity tail and the pinned historical RGB-offset -> bounded-luminance ->
single-pass colored-NLM h20 tail are registered here.  Further restoration
methods require a new, explicitly audited post-layout hook without changing or
obscuring the sorter.  Targets, manifests, source retrieval, centre/background
shortcuts, warps, resizing, constant canvases, and cross-board pixels are
absent from the API.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image

from aiijc_puzzle.compliant_atlas_decoder import PermutationAudit, audit_raw_permutation
from aiijc_puzzle.legacy_upgrade import atomic_write_png, layout_digest
from aiijc_puzzle.protocol import (
    GRID_SIZE,
    IMAGE_SIZE,
    RGB_CHANNELS,
    TILE_COUNT,
    assemble_tiles,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments
from aiijc_puzzle.socket_matcher import (
    BORDER_HEAD_EMBEDDING_V2,
    BORDER_HEAD_SCORE_STATS_V3,
    SocketMatcher,
)
from aiijc_puzzle.socket_pixel_tails import (
    historical_rgb_luma_nlm_h20_contract,
    historical_rgb_luma_nlm_h20_once,
)
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)

SORTER_SCHEMA = "aiijc-socket-sorter-production-v1"
RECORD_SCHEMA = "aiijc-socket-sorter-board-record-v1"
DECODER_EDGE_BUDGET = 144
DECODER_SWAP_STEPS = 24
CYCLIC_BORDER_WEIGHT = 5.0
SUPPORTED_ARCHITECTURES = {
    "board-conditioned-partial-socket-matcher-v2": BORDER_HEAD_EMBEDDING_V2,
    "board-conditioned-partial-socket-matcher-v3": BORDER_HEAD_SCORE_STATS_V3,
}


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _names_digest(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _reject_symlink_components(path: Path, *, require_leaf: bool = True) -> Path:
    absolute = _absolute(path)
    current = Path(absolute.parts[0])
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            if require_leaf or current != absolute:
                raise
            break
        if stat.S_ISLNK(mode):
            raise ValueError(f"symlink paths are forbidden: {current}")
    return absolute


def _require_regular_file(path: Path) -> Path:
    absolute = _reject_symlink_components(path)
    if not stat.S_ISREG(os.lstat(absolute).st_mode):
        raise ValueError(f"expected a regular file: {absolute}")
    return absolute


def _require_directory(path: Path) -> Path:
    absolute = _reject_symlink_components(path)
    if not stat.S_ISDIR(os.lstat(absolute).st_mode):
        raise ValueError(f"expected a directory: {absolute}")
    return absolute


def _mkdir_safe(path: Path) -> Path:
    absolute = _absolute(path)
    ancestor = absolute
    while not ancestor.exists():
        if ancestor == ancestor.parent:
            raise ValueError(f"cannot resolve an existing output ancestor: {absolute}")
        ancestor = ancestor.parent
    _reject_symlink_components(ancestor)
    absolute.mkdir(parents=True, exist_ok=True)
    return _require_directory(absolute)


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path = _absolute(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded + b"\n").hexdigest()


def _load_rgb_payload(path: Path) -> tuple[np.ndarray, str]:
    regular = _require_regular_file(path)
    payload = regular.read_bytes()
    file_hash = hashlib.sha256(payload).hexdigest()
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if (
            image.format != "PNG"
            or image.mode != "RGB"
            or image.size != (IMAGE_SIZE, IMAGE_SIZE)
        ):
            raise ValueError(f"expected RGB {IMAGE_SIZE}x{IMAGE_SIZE} PNG: {regular}")
        value = np.asarray(image, dtype=np.uint8).copy()
    expected = (IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS)
    if value.shape != expected or value.dtype != np.uint8:
        raise ValueError(f"decoded input violates RGB uint8 contract: {regular}")
    return value, file_hash


def scan_flat_png_directory(source_dir: Path) -> tuple[Path, tuple[str, ...]]:
    """Return a sorted, symlink-free roster from one flat PNG directory."""

    root = _require_directory(source_dir)
    names: list[str] = []
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise ValueError(f"source directory contains a non-regular entry: {entry.path}")
            if Path(entry.name).name != entry.name or not entry.name.endswith(".png"):
                raise ValueError(f"source directory contains a non-PNG basename: {entry.name}")
            names.append(entry.name)
    names.sort()
    if not names:
        raise ValueError("source directory contains no PNG files")
    if len(names) != len(set(names)):
        raise ValueError("source directory contains duplicate PNG basenames")
    return root, tuple(names)


def choose_deterministic_device(name: str) -> torch.device:
    """Resolve an explicit device and enable deterministic torch algorithms."""

    if name not in {"cpu", "mps"}:
        raise ValueError("device must be 'cpu' or 'mps'; implicit auto selection is forbidden")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    torch.use_deterministic_algorithms(True)
    return torch.device(name)


@dataclass(frozen=True)
class PixelTailHook:
    """Named post-layout image-to-image operation admitted by the runner."""

    name: str
    implementation: Callable[[np.ndarray], np.ndarray]
    target_blind: bool
    post_layout_only: bool
    evidence: Mapping[str, Any] | None = None

    def apply(self, raw: np.ndarray) -> np.ndarray:
        value = self.implementation(np.asarray(raw))
        output = np.asarray(value)
        expected = (IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS)
        if output.shape != expected or output.dtype != np.uint8:
            raise ValueError(
                f"pixel tail {self.name!r} must return uint8 RGB {expected}, "
                f"got {output.dtype} {output.shape}"
            )
        if not self.target_blind or not self.post_layout_only:
            raise ValueError("pixel tail must declare target-blind post-layout operation")
        return np.ascontiguousarray(output)


def _identity_tail(raw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(raw).copy()


IDENTITY_PIXEL_TAIL = PixelTailHook(
    name="identity",
    implementation=_identity_tail,
    target_blind=True,
    post_layout_only=True,
    evidence=None,
)
HISTORICAL_RGB_LUMA_NLM_H20_TAIL = PixelTailHook(
    name="historical-rgb-luma-nlm-h20-once",
    implementation=historical_rgb_luma_nlm_h20_once,
    target_blind=True,
    post_layout_only=True,
    evidence=historical_rgb_luma_nlm_h20_contract(),
)
PIXEL_TAILS = {
    IDENTITY_PIXEL_TAIL.name: IDENTITY_PIXEL_TAIL,
    HISTORICAL_RGB_LUMA_NLM_H20_TAIL.name: HISTORICAL_RGB_LUMA_NLM_H20_TAIL,
}


@dataclass(frozen=True)
class CheckpointLineage:
    train_count: int
    train_digest: str
    exposed_count: int
    exposed_digest: str
    train_filenames: tuple[str, ...]
    exposed_filenames: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        # The full rosters remain available to protocol auditors without being
        # repeated in every production board record.
        return {
            "train_count": self.train_count,
            "train_digest": self.train_digest,
            "exposed_count": self.exposed_count,
            "exposed_digest": self.exposed_digest,
        }


@dataclass(frozen=True)
class LoadedSocketCheckpoint:
    path: Path
    sha256: str
    model: SocketMatcher
    contract: dict[str, Any]
    resolved_border_head_version: str
    lineage: CheckpointLineage


def _positive_contract_integer(contract: Mapping[str, Any], key: str) -> int:
    value = contract.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"checkpoint contract {key!r} must be a positive integer")
    return value


def _checkpoint_lineage(payload: Mapping[str, Any]) -> CheckpointLineage:
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("checkpoint has no source-lineage selection mapping")

    def validated(kind: str) -> tuple[tuple[str, ...], str]:
        raw = selection.get(f"lineage_{kind}_filenames")
        digest = selection.get(f"lineage_{kind}_digest")
        if not isinstance(raw, list) or not all(isinstance(name, str) and name for name in raw):
            raise ValueError(f"checkpoint lineage_{kind}_filenames is malformed")
        names = tuple(raw)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError(f"checkpoint lineage_{kind}_filenames must be sorted and unique")
        observed = _names_digest(names)
        if not isinstance(digest, str) or digest != observed:
            raise ValueError(f"checkpoint lineage_{kind}_digest is invalid")
        return names, digest

    train, train_digest = validated("train")
    exposed, exposed_digest = validated("exposed")
    if not set(train).issubset(exposed):
        raise ValueError("checkpoint train lineage is not a subset of exposed lineage")
    return CheckpointLineage(
        len(train),
        train_digest,
        len(exposed),
        exposed_digest,
        train,
        exposed,
    )


def load_socket_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> LoadedSocketCheckpoint:
    """Load only a strict v2/v3 SocketMatcher checkpoint contract."""

    path = _require_regular_file(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("checkpoint has no architecture contract")
    contract = dict(contract)
    architecture = contract.get("architecture")
    border_version = SUPPORTED_ARCHITECTURES.get(architecture)
    if border_version is None:
        raise ValueError(f"unsupported SocketMatcher architecture: {architecture!r}")
    declared_border = contract.get("border_head_version")
    if architecture.endswith("-v3"):
        if declared_border != BORDER_HEAD_SCORE_STATS_V3:
            raise ValueError("v3 checkpoint must explicitly declare score_stats_v3 border head")
    elif declared_border not in {None, BORDER_HEAD_EMBEDDING_V2}:
        raise ValueError("v2 checkpoint declares an incompatible border head")
    dimension = _positive_contract_integer(contract, "dimension")
    heads = _positive_contract_integer(contract, "heads")
    board_layers = _positive_contract_integer(contract, "board_layers")
    socket_layers = _positive_contract_integer(contract, "socket_layers")
    sinkhorn_iterations = _positive_contract_integer(contract, "sinkhorn_iterations")
    if dimension % heads:
        raise ValueError("checkpoint dimension must be divisible by heads")
    if contract.get("synthetic_grid") != GRID_SIZE:
        raise ValueError("checkpoint must declare full-grid synthetic_grid=24")
    if contract.get("input_index_position_embedding") is not False:
        raise ValueError("checkpoint does not prove absence of shuffled-index embeddings")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint has no state_dict mapping")
    lineage = _checkpoint_lineage(payload)
    model = SocketMatcher(
        dimension=dimension,
        heads=heads,
        board_layers=board_layers,
        socket_layers=socket_layers,
        sinkhorn_iterations=sinkhorn_iterations,
        border_head_version=border_version,
    ).to(device)
    model.load_state_dict(dict(state_dict), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return LoadedSocketCheckpoint(
        path=path,
        sha256=sha256_file(path),
        model=model,
        contract=contract,
        resolved_border_head_version=border_version,
        lineage=lineage,
    )


@dataclass(frozen=True)
class SocketSorterPrediction:
    layout: np.ndarray
    raw: np.ndarray
    output: np.ndarray
    audit: PermutationAudit
    decoder_report: dict[str, Any]
    cyclic_report: dict[str, Any] | None
    matcher_seconds: float
    total_seconds: float


def assemble_audited_original_tiles(
    input_image: np.ndarray,
    layout: np.ndarray,
    *,
    restoration_applied_after_audit: bool = True,
) -> tuple[np.ndarray, PermutationAudit]:
    """Assemble a declared layout and fail on every missing/duplicate identity."""

    declared = np.asarray(layout)
    integer_dtype = declared.dtype.kind in {"i", "u"}
    layout_value = (
        np.ascontiguousarray(declared.astype(np.int64, copy=False))
        if integer_dtype
        else np.empty(0, dtype=np.int64)
    )
    if integer_dtype and layout_value.shape == (TILE_COUNT,):
        valid = layout_value[(layout_value >= 0) & (layout_value < TILE_COUNT)]
        counts = np.bincount(valid, minlength=TILE_COUNT)
        missing = np.flatnonzero(counts == 0)
        duplicates = np.flatnonzero(counts > 1)
    else:
        missing = np.arange(TILE_COUNT)
        duplicates = np.empty(0, dtype=np.int64)
    if (
        not integer_dtype
        or layout_value.shape != (TILE_COUNT,)
        or len(missing)
        or len(duplicates)
    ):
        raise ValueError(
            "layout must contain every integer tile identity exactly once: "
            f"dtype={declared.dtype}, shape={declared.shape}, missing={missing[:8].tolist()}, "
            f"duplicates={duplicates[:8].tolist()}"
        )
    layout_value = np.ascontiguousarray(layout_value.astype(np.int32))
    tiles = split_tiles(input_image)
    raw = assemble_tiles(tiles[layout_value])
    audit = audit_raw_permutation(
        input_image,
        raw,
        layout_value,
        restoration_applied_after_audit=restoration_applied_after_audit,
    )
    if not audit.passed:
        raise RuntimeError(f"raw permutation audit failed: {audit.as_dict()}")
    return raw, audit


@torch.inference_mode()
def predict_socket_sorter(
    input_image: np.ndarray,
    checkpoint: LoadedSocketCheckpoint,
    *,
    device: torch.device,
    cyclic_border5: bool,
    pixel_tail: PixelTailHook = IDENTITY_PIXEL_TAIL,
) -> SocketSorterPrediction:
    """Run target-free Socket sorting on one strict RGB input."""

    started = perf_counter()
    image = np.asarray(input_image)
    expected = (IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS)
    if image.shape != expected or image.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB input {expected}, got {image.dtype} {image.shape}")
    original_tiles = split_tiles(image)
    tensor = torch.from_numpy(original_tiles.astype(np.float32)).permute(0, 3, 1, 2) / 255.0
    matcher_started = perf_counter()
    output = checkpoint.model(tensor.unsqueeze(0).to(device), grid=GRID_SIZE)
    right = output.right_log_assignment[0].float().cpu().numpy()
    down = output.down_log_assignment[0].float().cpu().numpy()
    matcher_seconds = perf_counter() - matcher_started
    decoder = decode_socket_assignments(
        right,
        down,
        grid=GRID_SIZE,
        config=SocketDecoderConfig(
            component_edge_budget_per_axis=DECODER_EDGE_BUDGET,
            swap_edge_budget_per_axis=DECODER_EDGE_BUDGET,
            max_swap_steps=DECODER_SWAP_STEPS,
        ),
    )
    layout = decoder.layout
    cyclic_report: dict[str, Any] | None = None
    if cyclic_border5:
        cyclic = select_global_cyclic_translation(
            layout,
            right,
            down,
            grid=GRID_SIZE,
            config=CyclicTranslationConfig(border_weight=CYCLIC_BORDER_WEIGHT),
        )
        layout = cyclic.layout
        cyclic_report = cyclic.report()
    raw, audit = assemble_audited_original_tiles(image, layout)
    restored = pixel_tail.apply(raw)
    if pixel_tail.name == "identity" and not np.array_equal(restored, raw):
        raise RuntimeError("identity pixel tail altered raw assembly")
    return SocketSorterPrediction(
        layout=np.ascontiguousarray(layout, dtype=np.int32),
        raw=raw,
        output=restored,
        audit=audit,
        decoder_report=decoder.report(),
        cyclic_report=cyclic_report,
        matcher_seconds=matcher_seconds,
        total_seconds=perf_counter() - started,
    )


def _runtime_manifest() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    paths = (
        "src/aiijc_puzzle/socket_sorter_production.py",
        "src/aiijc_puzzle/socket_matcher.py",
        "src/aiijc_puzzle/socket_decoder.py",
        "src/aiijc_puzzle/socket_translation_placer.py",
        "src/aiijc_puzzle/socket_pixel_tails.py",
        "src/aiijc_puzzle/postassembly_harmonizer.py",
        "src/aiijc_puzzle/pixel_tails.py",
        "src/aiijc_puzzle/protocol.py",
    )
    files = {relative: sha256_file(project_root / relative) for relative in paths}
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "files": files,
        "files_digest": _digest_json(files),
    }


def _pipeline_contract(
    checkpoint: LoadedSocketCheckpoint,
    *,
    device: torch.device,
    cyclic_border5: bool,
    pixel_tail: PixelTailHook,
) -> dict[str, Any]:
    runtime = _runtime_manifest()
    payload: dict[str, Any] = {
        "schema": SORTER_SCHEMA,
        "checkpoint_sha256": checkpoint.sha256,
        "checkpoint_architecture": checkpoint.contract["architecture"],
        "checkpoint_lineage": checkpoint.lineage.as_dict(),
        "resolved_border_head_version": checkpoint.resolved_border_head_version,
        "device": str(device),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "decoder": {
            "name": "socket-translation-components-qap-v1",
            "component_edge_budget_per_axis": DECODER_EDGE_BUDGET,
            "swap_edge_budget_per_axis": DECODER_EDGE_BUDGET,
            "max_swap_steps": DECODER_SWAP_STEPS,
            "centre_or_background_heuristic": False,
        },
        "absolute_anchor": {
            "name": "socket-global-cyclic-translation-v1" if cyclic_border5 else "none",
            "enabled": cyclic_border5,
            "border_weight": CYCLIC_BORDER_WEIGHT if cyclic_border5 else None,
        },
        "pixel_tail": {
            "name": pixel_tail.name,
            "target_blind": pixel_tail.target_blind,
            "post_layout_only": pixel_tail.post_layout_only,
            "evidence": pixel_tail.evidence,
        },
        "policy": {
            "targets_or_manifest_labels_used": False,
            "source_lookup_used": False,
            "external_templates_used": False,
            "constant_canvas_used": False,
            "tile_warp_or_resize_used": False,
            "all_original_tiles_used_exactly_once_before_tail": True,
        },
        "runtime": runtime,
    }
    payload["pipeline_digest"] = _digest_json(payload)
    return payload


def _board_record(
    *,
    filename: str,
    input_file_sha256: str,
    input_image: np.ndarray,
    output_png_sha256: str,
    prediction: SocketSorterPrediction,
    checkpoint: LoadedSocketCheckpoint,
    pipeline: Mapping[str, Any],
    pixel_tail: PixelTailHook,
) -> dict[str, Any]:
    return {
        "schema": RECORD_SCHEMA,
        "filename": filename,
        "input": {
            "file_sha256": input_file_sha256,
            "decoded_rgb_sha256": _array_sha256(input_image),
            "shape": list(input_image.shape),
            "dtype": str(input_image.dtype),
        },
        "lineage": {
            "checkpoint_sha256": checkpoint.sha256,
            **checkpoint.lineage.as_dict(),
            "pipeline_digest": pipeline["pipeline_digest"],
        },
        "layout": {
            "tile_at_position": prediction.layout.tolist(),
            "sha256_int32_le": layout_digest(prediction.layout),
            "strict_permutation": True,
            "all_576_original_tiles_used_exactly_once": True,
        },
        "raw_assembly": {
            "array_sha256": _array_sha256(prediction.raw),
            "shape": list(prediction.raw.shape),
            "dtype": str(prediction.raw.dtype),
            "audit": prediction.audit.as_dict(),
        },
        "pixel_tail": {
            "name": pixel_tail.name,
            "target_blind": pixel_tail.target_blind,
            "post_layout_only": pixel_tail.post_layout_only,
            "output_array_sha256": _array_sha256(prediction.output),
        },
        "output_png_sha256": output_png_sha256,
        "diagnostics": {
            "decoder": prediction.decoder_report,
            "cyclic_translation": prediction.cyclic_report,
            "matcher_seconds": prediction.matcher_seconds,
            "total_seconds": prediction.total_seconds,
        },
    }


def _validate_record_and_output(
    *,
    record: Mapping[str, Any],
    filename: str,
    input_image: np.ndarray,
    input_file_sha256: str,
    output_path: Path,
    checkpoint: LoadedSocketCheckpoint,
    pipeline: Mapping[str, Any],
    pixel_tail: PixelTailHook,
) -> None:
    if record.get("schema") != RECORD_SCHEMA or record.get("filename") != filename:
        raise ValueError(f"resume record identity mismatch: {filename}")
    input_record = record.get("input")
    lineage = record.get("lineage")
    layout_record = record.get("layout")
    raw_record = record.get("raw_assembly")
    tail_record = record.get("pixel_tail")
    if not all(
        isinstance(value, Mapping)
        for value in (input_record, lineage, layout_record, raw_record, tail_record)
    ):
        raise ValueError(f"resume record structure is malformed: {filename}")
    if input_record.get("file_sha256") != input_file_sha256 or input_record.get(
        "decoded_rgb_sha256"
    ) != _array_sha256(input_image):
        raise ValueError(f"resume input hash mismatch: {filename}")
    if lineage.get("checkpoint_sha256") != checkpoint.sha256 or lineage.get(
        "pipeline_digest"
    ) != pipeline["pipeline_digest"]:
        raise ValueError(f"resume model/pipeline lineage mismatch: {filename}")
    layout = np.asarray(layout_record.get("tile_at_position"))
    raw, audit = assemble_audited_original_tiles(input_image, layout)
    if not audit.passed or raw_record.get("array_sha256") != _array_sha256(raw):
        raise ValueError(f"resume raw assembly mismatch: {filename}")
    if layout_record.get("sha256_int32_le") != layout_digest(layout.astype(np.int32)):
        raise ValueError(f"resume layout digest mismatch: {filename}")
    if tail_record.get("name") != pixel_tail.name:
        raise ValueError(f"resume pixel-tail mismatch: {filename}")
    expected_output = pixel_tail.apply(raw)
    decoded_output, observed_file_hash = _load_rgb_payload(output_path)
    if observed_file_hash != record.get("output_png_sha256"):
        raise ValueError(f"resume output PNG hash mismatch: {filename}")
    if tail_record.get("output_array_sha256") != _array_sha256(expected_output) or not (
        np.array_equal(decoded_output, expected_output)
    ):
        raise ValueError(f"resume decoded output mismatch: {filename}")


def _load_json_mapping(path: Path) -> dict[str, Any]:
    regular = _require_regular_file(path)
    value = json.loads(regular.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON payload must be an object: {regular}")
    return value


def _validate_output_directory_state(
    output_root: Path,
    records_dir: Path,
    filenames: Sequence[str],
    *,
    require_complete: bool,
) -> None:
    """Reject foreign artifacts and optionally require the full output roster."""

    expected_outputs = set(filenames)
    expected_records = {f"{name}.json" for name in filenames}
    observed_outputs: set[str] = set()
    with os.scandir(output_root) as entries:
        for entry in entries:
            if entry.name == "records":
                if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                    raise ValueError("output records entry is not a regular directory")
                continue
            if entry.name == "run.json":
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise ValueError("output run.json is not a regular file")
                continue
            if (
                entry.name not in expected_outputs
                or entry.is_symlink()
                or not entry.is_file(follow_symlinks=False)
            ):
                raise ValueError(f"output directory contains a foreign artifact: {entry.name}")
            observed_outputs.add(entry.name)
    observed_records: set[str] = set()
    with os.scandir(records_dir) as entries:
        for entry in entries:
            if (
                entry.name not in expected_records
                or entry.is_symlink()
                or not entry.is_file(follow_symlinks=False)
            ):
                raise ValueError(f"records directory contains a foreign artifact: {entry.name}")
            observed_records.add(entry.name)
    if require_complete and (
        observed_outputs != expected_outputs or observed_records != expected_records
    ):
        raise ValueError(
            "completed output roster is incomplete: "
            f"outputs={len(observed_outputs)}/{len(expected_outputs)}, "
            f"records={len(observed_records)}/{len(expected_records)}"
        )


def _run_identity(
    pipeline: Mapping[str, Any],
    source_dir: Path,
    roster: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SORTER_SCHEMA,
        "pipeline_digest": pipeline["pipeline_digest"],
        "source_dir": str(source_dir),
        "roster": list(roster),
        "roster_digest": _digest_json({"files": list(roster)}),
    }
    payload["run_digest"] = _digest_json(payload)
    return payload


def inspect_socket_sorter_run(
    *,
    checkpoint_path: Path,
    source_dir: Path,
    output_dir: Path,
    device_name: str,
    cyclic_border5: bool,
    pixel_tail_name: str,
) -> tuple[LoadedSocketCheckpoint, torch.device, PixelTailHook, dict[str, Any]]:
    """Validate checkpoint/source roster and return a no-write execution plan."""

    device = choose_deterministic_device(device_name)
    checkpoint = load_socket_checkpoint(checkpoint_path, device=device)
    if pixel_tail_name not in PIXEL_TAILS:
        raise ValueError(f"unregistered pixel tail: {pixel_tail_name!r}")
    pixel_tail = PIXEL_TAILS[pixel_tail_name]
    source_root, names = scan_flat_png_directory(source_dir)
    output_root = _absolute(output_dir)
    if _paths_overlap(source_root, output_root):
        raise ValueError("source and output directory trees must not overlap")
    roster: list[dict[str, str]] = []
    for name in names:
        path = _require_regular_file(source_root / name)
        roster.append({"filename": name, "input_file_sha256": sha256_file(path)})
    pipeline = _pipeline_contract(
        checkpoint,
        device=device,
        cyclic_border5=cyclic_border5,
        pixel_tail=pixel_tail,
    )
    identity = _run_identity(pipeline, source_root, roster)
    plan = {
        **identity,
        "status": "DRY_RUN_NO_WRITES",
        "checkpoint": {
            "path": str(checkpoint.path),
            "sha256": checkpoint.sha256,
            "contract": checkpoint.contract,
            "resolved_border_head_version": checkpoint.resolved_border_head_version,
            "lineage": checkpoint.lineage.as_dict(),
        },
        "pipeline": pipeline,
        "output_dir": str(output_root),
        "file_count": len(roster),
    }
    return checkpoint, device, pixel_tail, plan


def run_socket_sorter_directory(
    *,
    checkpoint_path: Path,
    source_dir: Path,
    output_dir: Path,
    device_name: str = "cpu",
    cyclic_border5: bool = False,
    pixel_tail_name: str = "identity",
) -> dict[str, Any]:
    """Run or resume a content-addressed directory of Socket predictions."""

    checkpoint, device, pixel_tail, plan = inspect_socket_sorter_run(
        checkpoint_path=checkpoint_path,
        source_dir=source_dir,
        output_dir=output_dir,
        device_name=device_name,
        cyclic_border5=cyclic_border5,
        pixel_tail_name=pixel_tail_name,
    )
    source_root = Path(plan["source_dir"])
    output_root = _mkdir_safe(output_dir)
    records_dir = _mkdir_safe(output_root / "records")
    run_path = output_root / "run.json"
    roster = plan["roster"]
    if not isinstance(roster, list) or not all(isinstance(item, Mapping) for item in roster):
        raise RuntimeError("internal roster contract failure")
    roster_names = tuple(str(item["filename"]) for item in roster)
    _validate_output_directory_state(
        output_root,
        records_dir,
        roster_names,
        require_complete=False,
    )
    stable_run = {
        key: plan[key]
        for key in (
            "schema",
            "pipeline_digest",
            "source_dir",
            "roster",
            "roster_digest",
            "run_digest",
        )
    }
    if run_path.exists():
        existing = _load_json_mapping(run_path)
        if any(existing.get(key) != value for key, value in stable_run.items()):
            raise ValueError("existing run.json belongs to different inputs/model/pipeline")
    else:
        unexpected = [
            path.name
            for path in output_root.iterdir()
            if path.name not in {"records", "run.json"}
        ]
        if unexpected or any(records_dir.iterdir()):
            raise ValueError("output directory is non-empty but has no matching run.json")
    progress: dict[str, Any] = {
        **stable_run,
        "status": "IN_PROGRESS",
        "checkpoint": plan["checkpoint"],
        "pipeline": plan["pipeline"],
        "output_dir": str(output_root),
        "file_count": plan["file_count"],
        "completed_filenames": [],
    }
    _atomic_write_json(run_path, progress)

    processed = 0
    resumed = 0
    completed: list[str] = []
    for index, item in enumerate(roster, start=1):
        filename = str(item["filename"])
        image, input_file_hash = _load_rgb_payload(source_root / filename)
        if input_file_hash != item["input_file_sha256"]:
            raise ValueError(f"source file changed after roster snapshot: {filename}")
        output_path = output_root / filename
        record_path = records_dir / f"{filename}.json"
        output_exists = output_path.exists()
        record_exists = record_path.exists()
        if output_exists and record_exists:
            record = _load_json_mapping(record_path)
            _validate_record_and_output(
                record=record,
                filename=filename,
                input_image=image,
                input_file_sha256=input_file_hash,
                output_path=output_path,
                checkpoint=checkpoint,
                pipeline=plan["pipeline"],
                pixel_tail=pixel_tail,
            )
            resumed += 1
        else:
            prediction = predict_socket_sorter(
                image,
                checkpoint,
                device=device,
                cyclic_border5=cyclic_border5,
                pixel_tail=pixel_tail,
            )
            expected_output_hash = _array_sha256(prediction.output)
            if output_exists:
                decoded, _ = _load_rgb_payload(output_path)
                if _array_sha256(decoded) != expected_output_hash:
                    raise ValueError(
                        f"partial resume output differs from recomputation: {filename}"
                    )
                output_png_hash = sha256_file(output_path)
            else:
                output_png_hash = atomic_write_png(output_path, prediction.output)
            record = _board_record(
                filename=filename,
                input_file_sha256=input_file_hash,
                input_image=image,
                output_png_sha256=output_png_hash,
                prediction=prediction,
                checkpoint=checkpoint,
                pipeline=plan["pipeline"],
                pixel_tail=pixel_tail,
            )
            if record_exists:
                existing_record = _load_json_mapping(record_path)
                stable_keys = (
                    "schema",
                    "filename",
                    "input",
                    "lineage",
                    "layout",
                    "raw_assembly",
                    "pixel_tail",
                    "output_png_sha256",
                )
                if any(existing_record.get(key) != record.get(key) for key in stable_keys):
                    raise ValueError(
                        f"partial resume record differs from recomputation: {filename}"
                    )
            else:
                _atomic_write_json(record_path, record)
            _validate_record_and_output(
                record=record,
                filename=filename,
                input_image=image,
                input_file_sha256=input_file_hash,
                output_path=output_path,
                checkpoint=checkpoint,
                pipeline=plan["pipeline"],
                pixel_tail=pixel_tail,
            )
            processed += 1
        completed.append(filename)
        progress["completed_filenames"] = completed.copy()
        progress["status"] = "IN_PROGRESS"
        _atomic_write_json(run_path, progress)
        print(
            json.dumps(
                {
                    "event": "socket_sorter_board",
                    "index": index,
                    "count": len(roster),
                    "filename": filename,
                    "resumed": output_exists and record_exists,
                }
            ),
            flush=True,
        )
    _validate_output_directory_state(
        output_root,
        records_dir,
        roster_names,
        require_complete=True,
    )
    progress["status"] = "COMPLETE"
    progress["processed_this_invocation"] = processed
    progress["resumed_this_invocation"] = resumed
    _atomic_write_json(run_path, progress)
    return {
        "status": "COMPLETE",
        "run_json": str(run_path),
        "run_digest": plan["run_digest"],
        "file_count": len(roster),
        "processed": processed,
        "resumed": resumed,
        "checkpoint_sha256": checkpoint.sha256,
        "pipeline_digest": plan["pipeline_digest"],
    }
