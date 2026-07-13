"""Leakage-safe visual QA for a selected tile denoiser.

The utility in this module deliberately limits itself to a small deterministic
sample from the pinned 257-source calibration partition.  Quarantined sources
and the sealed 350-source gate are represented only by counts and name hashes;
their pixels are never materialized or passed to the restorer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import re
import stat
from typing import Any, Mapping, Sequence

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
import torch

from .inference import load_restorer, restore_tiles_uint8
from .real_pairs import RealPairBatch, RealPairSampler, RealPairTable
from .real_training import (
    deterministic_contamination_aware_split,
    load_validation_quarantine,
    source_name_list_sha256,
)
from .training import load_manifest


PROTOCOL_SEED = 20260710
EXPECTED_VALIDATION_SOURCE_COUNT = 700
QUARANTINE_SOURCE_COUNT = 93
CALIBRATION_SOURCE_COUNT = 257
FROZEN_GATE_SOURCE_COUNT = 350
PRIMARY_CONFIDENCE = 1.5
SYNTHETIC_CHECKPOINT_VALIDATION_COUNT = 24
DEFAULT_VISUAL_PAIR_COUNT = 12
DEFAULT_TILE_SCALE = 6

VISUAL_QA_CODE_FILES = (
    "__init__.py",
    "degradation.py",
    "inference.py",
    "losses.py",
    "model.py",
    "real_pairs.py",
    "real_training.py",
    "tiles.py",
    "training.py",
    "visual_qa.py",
)


@dataclass(frozen=True)
class VisualQAConfig:
    data_root: str
    manifest: str
    val_pairs: str
    checkpoint: str
    quarantine_artifact: str
    output_png: str
    report_json: str
    expected_val_pairs_sha256: str
    expected_checkpoint_sha256: str
    expected_quarantine_sha256: str
    selection_seed: int = PROTOCOL_SEED
    pair_count: int = DEFAULT_VISUAL_PAIR_COUNT
    tile_scale: int = DEFAULT_TILE_SCALE
    state: str = "ema"
    device: str = "cpu"
    batch_size: int = 64
    overwrite: bool = False


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")


def validate_visual_qa_config(config: VisualQAConfig) -> None:
    for name in (
        "expected_val_pairs_sha256",
        "expected_checkpoint_sha256",
        "expected_quarantine_sha256",
    ):
        _require_sha256(name, getattr(config, name))
    if isinstance(config.selection_seed, bool) or not isinstance(config.selection_seed, int):
        raise TypeError("selection_seed must be an integer")
    if (
        isinstance(config.pair_count, bool)
        or not isinstance(config.pair_count, int)
        or not 1 <= config.pair_count <= CALIBRATION_SOURCE_COUNT
    ):
        raise ValueError(
            f"pair_count must be an integer in [1, {CALIBRATION_SOURCE_COUNT}]"
        )
    if (
        isinstance(config.tile_scale, bool)
        or not isinstance(config.tile_scale, int)
        or not 2 <= config.tile_scale <= 20
    ):
        raise ValueError("tile_scale must be an integer in [2, 20]")
    if (
        isinstance(config.batch_size, bool)
        or not isinstance(config.batch_size, int)
        or config.batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    if config.state not in {"ema", "model"}:
        raise ValueError("state must be 'ema' or 'model'")


def visual_qa_code_fingerprint(package_dir: str | Path | None = None) -> str:
    """Hash project sources that can affect selection, pixels, or rendering."""
    root = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in VISUAL_QA_CODE_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"visual-QA code file is missing: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def select_visual_qa_sources(
    source_names: tuple[str, ...],
    calibration_source_indices: Sequence[int] | np.ndarray,
    *,
    pair_count: int,
    seed: int,
) -> np.ndarray:
    """Select distinct calibration sources by a stable name-based hash rank."""
    indices = np.asarray(calibration_source_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0 or len(np.unique(indices)) != len(indices):
        raise ValueError("calibration_source_indices must be non-empty and unique")
    if indices.min() < 0 or indices.max() >= len(source_names):
        raise ValueError("calibration_source_indices contains an out-of-range source")
    if isinstance(pair_count, bool) or not isinstance(pair_count, int):
        raise TypeError("pair_count must be an integer")
    if not 1 <= pair_count <= len(indices):
        raise ValueError("pair_count must fit in the calibration source set")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    ranked = sorted(
        (int(index) for index in indices),
        key=lambda index: (
            hashlib.sha256(
                f"{seed}:visual-qa-source:{source_names[index]}".encode("utf-8")
            ).digest(),
            source_names[index],
        ),
    )
    return np.asarray(ranked[:pair_count], dtype=np.int64)


def validate_calibration_only_selection(
    selected_source_indices: Sequence[int] | np.ndarray,
    calibration_source_indices: Sequence[int] | np.ndarray,
    frozen_gate_source_indices: Sequence[int] | np.ndarray,
    quarantine_source_indices: Sequence[int] | np.ndarray,
) -> None:
    """Fail closed if any selected source is outside the calibration set."""
    selected = np.asarray(selected_source_indices, dtype=np.int64)
    calibration = np.asarray(calibration_source_indices, dtype=np.int64)
    gate = np.asarray(frozen_gate_source_indices, dtype=np.int64)
    quarantine = np.asarray(quarantine_source_indices, dtype=np.int64)
    if selected.ndim != 1 or len(selected) == 0 or len(np.unique(selected)) != len(selected):
        raise ValueError("selected_source_indices must be non-empty and unique")
    for name, values in (
        ("calibration", calibration),
        ("frozen_gate", gate),
        ("quarantine", quarantine),
    ):
        if values.ndim != 1 or len(np.unique(values)) != len(values):
            raise ValueError(f"{name} source indices must be one-dimensional and unique")
    calibration_set = set(calibration.tolist())
    gate_set = set(gate.tolist())
    quarantine_set = set(quarantine.tolist())
    if (
        calibration_set & gate_set
        or calibration_set & quarantine_set
        or gate_set & quarantine_set
    ):
        raise ValueError("calibration, frozen-gate, and quarantine partitions must be disjoint")
    selected_set = set(selected.tolist())
    if not selected_set <= calibration_set:
        raise ValueError("visual-QA selection contains a non-calibration source")
    if selected_set & gate_set:
        raise ValueError("visual-QA selection contains a frozen-gate source")
    if selected_set & quarantine_set:
        raise ValueError("visual-QA selection contains a quarantined source")


def _tensor_tiles_uint8(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim != 4 or tuple(tensor.shape[1:]) != (3, 20, 20):
        raise ValueError(f"expected Nx3x20x20 tensor, got {tuple(tensor.shape)}")
    return (
        tensor.detach()
        .cpu()
        .mul(255.0)
        .round()
        .clamp(0, 255)
        .byte()
        .permute(0, 2, 3, 1)
        .numpy()
    )


def _validate_tile_triplet(
    corrupt: np.ndarray,
    restored: np.ndarray,
    clean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(np.asarray(value) for value in (corrupt, restored, clean))
    if not arrays[0].ndim == 4 or arrays[0].shape[1:] != (20, 20, 3):
        raise ValueError(f"expected Nx20x20x3 corrupt tiles, got {arrays[0].shape}")
    if len(arrays[0]) == 0:
        raise ValueError("visual-QA tile arrays must not be empty")
    for name, array in zip(("corrupt", "restored", "clean"), arrays, strict=True):
        if array.shape != arrays[0].shape:
            raise ValueError(f"{name} tile shape {array.shape} != {arrays[0].shape}")
        if array.dtype != np.uint8:
            raise TypeError(f"{name} tiles must be uint8, got {array.dtype}")
    return arrays


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def selection_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        [dict(row) for row in rows],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def render_visual_qa_contact_sheet(
    corrupt: np.ndarray,
    restored: np.ndarray,
    clean: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint_sha256: str,
    selection_seed: int,
    tile_scale: int = DEFAULT_TILE_SCALE,
) -> Image.Image:
    """Render labelled ``corrupt | restored | clean`` rows with nearest scaling."""
    corrupt, restored, clean = _validate_tile_triplet(corrupt, restored, clean)
    _require_sha256("checkpoint_sha256", checkpoint_sha256)
    if len(rows) != len(corrupt):
        raise ValueError("rows length must match the tile arrays")
    if isinstance(tile_scale, bool) or not isinstance(tile_scale, int) or tile_scale < 2:
        raise ValueError("tile_scale must be an integer >= 2")

    margin = 16
    label_width = 218
    gap = 12
    header_height = 48
    row_gap = 12
    tile_pixels = 20 * tile_scale
    width = margin * 2 + label_width + gap + 3 * tile_pixels + 2 * gap
    height = margin * 2 + header_height + len(rows) * tile_pixels + (len(rows) - 1) * row_gap
    image = Image.new("RGB", (width, height), (245, 247, 250))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    title = (
        "Leakage-safe calibration visual QA  |  "
        f"seed {selection_seed}  |  checkpoint {checkpoint_sha256[:12]}"
    )
    draw.text((margin, margin), title, fill=(22, 29, 37), font=font)
    first_tile_x = margin + label_width + gap
    column_x = [
        first_tile_x,
        first_tile_x + tile_pixels + gap,
        first_tile_x + 2 * (tile_pixels + gap),
    ]
    for label, x in zip(("CORRUPT", "RESTORED", "CLEAN"), column_x, strict=True):
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        draw.text(
            (x + (tile_pixels - text_width) // 2, margin + 22),
            label,
            fill=(47, 56, 66),
            font=font,
        )

    row_y = margin + header_height
    resampling = Image.Resampling.NEAREST
    for ordinal, (record, tile_group) in enumerate(
        zip(rows, zip(corrupt, restored, clean, strict=True), strict=True),
        start=1,
    ):
        required = {"source_name", "input_slot", "clean_tile_index", "confidence", "pair_row"}
        if not required <= set(record):
            raise ValueError(f"row {ordinal} is missing labels {sorted(required - set(record))}")
        label = (
            f"{ordinal:02d}  {record['source_name']}\n"
            f"slot {int(record['input_slot']):03d} -> tile "
            f"{int(record['clean_tile_index']):03d}\n"
            f"confidence {float(record['confidence']):.4f}  |  row "
            f"{int(record['pair_row'])}"
        )
        draw.multiline_text(
            (margin, row_y + 4),
            label,
            fill=(30, 38, 47),
            font=font,
            spacing=4,
        )
        for tile, x in zip(tile_group, column_x, strict=True):
            enlarged = Image.fromarray(tile, mode="RGB").resize(
                (tile_pixels, tile_pixels),
                resample=resampling,
            )
            image.paste(enlarged, (x, row_y))
            draw.rectangle(
                (x - 1, row_y - 1, x + tile_pixels, row_y + tile_pixels),
                outline=(97, 107, 118),
                width=1,
            )
        row_y += tile_pixels + row_gap
    return image


def _png_bytes(image: Image.Image) -> bytes:
    stream = BytesIO()
    image.save(stream, format="PNG", optimize=False, compress_level=9)
    return stream.getvalue()


def _canonical(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _validate_output_paths(
    output_png: Path,
    report_json: Path,
    *,
    protected_paths: Sequence[Path],
    overwrite: bool,
) -> None:
    outputs = (output_png, report_json)
    canonical_outputs = tuple(_canonical(path) for path in outputs)
    if len(set(canonical_outputs)) != len(canonical_outputs):
        raise ValueError("output_png and report_json collide after canonical resolution")
    if output_png.exists() and report_json.exists() and output_png.samefile(report_json):
        raise ValueError("output_png and report_json are hard links to the same file")
    existing_protected = [path for path in protected_paths if path.exists()]
    canonical_protected = {_canonical(path) for path in existing_protected}
    for path, canonical in zip(outputs, canonical_outputs, strict=True):
        if canonical in canonical_protected:
            raise ValueError(f"refusing to overwrite a protected input: {path}")
        if path.is_symlink():
            raise ValueError(f"refusing to write through an output symlink: {path}")
        if path.exists() and path.is_dir():
            raise ValueError(f"output path is a directory: {path}")
        if path.exists() and any(path.samefile(protected) for protected in existing_protected):
            raise ValueError(f"refusing output hard-link collision with an input: {path}")
        if path.exists() and not overwrite:
            raise FileExistsError(f"output exists; enable overwrite to replace it: {path}")


def write_visual_qa_outputs(
    image: Image.Image,
    report: Mapping[str, Any],
    *,
    output_png: str | Path,
    report_json: str | Path,
    overwrite: bool = False,
    protected_paths: Sequence[str | Path] = (),
) -> dict:
    """Write a deterministic PNG and a JSON report containing its hashes."""
    png_path = Path(output_png).expanduser()
    json_path = Path(report_json).expanduser()
    protected = tuple(Path(path).expanduser() for path in protected_paths)
    _validate_output_paths(
        png_path,
        json_path,
        protected_paths=protected,
        overwrite=overwrite,
    )
    png_payload = _png_bytes(image)
    final_report = json.loads(json.dumps(dict(report), allow_nan=False))
    final_report["outputs"] = {
        "contact_sheet_png": str(png_path.resolve(strict=False)),
        "contact_sheet_png_sha256": hashlib.sha256(png_payload).hexdigest(),
        "contact_sheet_pixels_sha256": _array_sha256(np.asarray(image, dtype=np.uint8)),
        "report_json": str(json_path.resolve(strict=False)),
    }
    json_payload = json.dumps(
        final_report,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"

    png_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.write_bytes(png_payload)
    json_path.write_bytes(json_payload)
    return final_report


def _trusted_legacy_sha_from_pinned_quarantine(
    path: Path,
    expected_quarantine_sha256: str,
) -> str:
    """Read the legacy digest only after the entire quarantine file is pinned."""
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_quarantine_sha256:
        raise ValueError("validation-quarantine SHA256 does not match the pinned digest")
    try:
        decoded = json.loads(payload)
        value = decoded["legacy_checkpoint"]["sha256"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("pinned quarantine artifact has no legacy checkpoint SHA256") from error
    _require_sha256("quarantine legacy checkpoint sha256", value)
    return value


def _checkpoint_provenance(metadata: Mapping[str, Any]) -> dict[str, Any]:
    # Deliberately omit source_split and gate_validation.  A visual-QA report
    # must not disclose or evaluate frozen-gate details.
    allowed = (
        "checkpoint",
        "checkpoint_resolved",
        "checkpoint_sha256",
        "checkpoint_is_latest",
        "model_name",
        "state",
        "device",
        "step",
        "best_step",
        "rolled_back",
        "schema_version",
        "kind",
        "manifest_sha256",
        "source_code_sha256",
        "fine_tune_code_sha256",
        "init_checkpoint_sha256",
        "validation_quarantine_sha256",
        "train_pairs_sha256",
        "val_pairs_sha256",
        "promotion_status",
        "safe_for_inference",
        "runtime_versions",
        "resolved_device_fingerprint",
    )
    return {key: metadata[key] for key in allowed if key in metadata}


def _validate_selected_pixel_file_identity(
    selected_paths: Sequence[Path],
    forbidden_paths: Sequence[Path],
) -> None:
    """Reject symlinks and inode aliases without reading forbidden pixels."""

    def regular_file_identity(path: Path) -> tuple[int, int]:
        if path.is_symlink():
            raise ValueError(f"visual-QA pixel source must not be a symlink: {path}")
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"visual-QA pixel source must be a regular file: {path}")
        return int(info.st_dev), int(info.st_ino)

    forbidden_identities = {regular_file_identity(path) for path in forbidden_paths}
    selected_identities: set[tuple[int, int]] = set()
    for path in selected_paths:
        identity = regular_file_identity(path)
        if identity in forbidden_identities:
            raise ValueError(
                "selected calibration pixel file aliases a quarantine/frozen-gate file: "
                f"{path}"
            )
        if identity in selected_identities:
            raise ValueError(f"selected calibration pixel files share an inode: {path}")
        selected_identities.add(identity)


def _row_records(panel: RealPairBatch, source_names: tuple[str, ...]) -> list[dict[str, Any]]:
    records = []
    for index in range(len(panel)):
        source_index = int(panel.source_index[index])
        records.append(
            {
                "source_index": source_index,
                "source_name": source_names[source_index],
                "input_slot": int(panel.input_slot[index]),
                "clean_tile_index": int(panel.clean_tile_index[index]),
                "confidence": float(panel.confidence[index]),
                "pair_row": int(panel.pair_row[index]),
            }
        )
    return records


def run_visual_qa(config: VisualQAConfig) -> dict:
    """Generate a pinned calibration-only contact sheet and JSON provenance."""
    validate_visual_qa_config(config)
    root = Path(config.data_root).expanduser()
    manifest_path = Path(config.manifest).expanduser()
    val_pairs_path = Path(config.val_pairs).expanduser()
    checkpoint_path = Path(config.checkpoint).expanduser()
    quarantine_path = Path(config.quarantine_artifact).expanduser()
    output_png = Path(config.output_png).expanduser()
    report_json = Path(config.report_json).expanduser()
    resolved_data_root = root.resolve(strict=True)
    for output_path in (output_png, report_json):
        try:
            _canonical(output_path).relative_to(resolved_data_root)
        except ValueError:
            pass
        else:
            raise ValueError("visual-QA outputs must stay outside the puzzle data root")
    protected_paths = (manifest_path, val_pairs_path, checkpoint_path, quarantine_path)
    _validate_output_paths(
        output_png,
        report_json,
        protected_paths=protected_paths,
        overwrite=config.overwrite,
    )

    manifest = load_manifest(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    validation_names = manifest["splits"]["val"]
    if len(validation_names) != EXPECTED_VALIDATION_SOURCE_COUNT:
        raise ValueError(
            f"manifest must contain exactly {EXPECTED_VALIDATION_SOURCE_COUNT} validation sources"
        )
    table = RealPairTable.load(
        val_pairs_path,
        manifest_path=manifest_path,
        data_root=root,
        expected_split="val",
        min_confidence=PRIMARY_CONFIDENCE,
    )
    if table.npz_sha256 != config.expected_val_pairs_sha256:
        raise ValueError("validation real-pair SHA256 does not match the pinned digest")
    if set(table.source_names) != set(validation_names):
        raise ValueError("validation real-pair table does not contain the full manifest validation set")
    expected_active = np.arange(EXPECTED_VALIDATION_SOURCE_COUNT, dtype=np.int64)
    if not np.array_equal(table.active_source_indices, expected_active):
        raise ValueError("the 1.5-confidence panel must keep all 700 validation sources active")

    expected_legacy_sha256 = _trusted_legacy_sha_from_pinned_quarantine(
        quarantine_path,
        config.expected_quarantine_sha256,
    )
    quarantine, quarantine_sha256 = load_validation_quarantine(
        quarantine_path,
        config.expected_quarantine_sha256,
        manifest_sha256=manifest_sha256,
        manifest_validation_names=validation_names,
        expected_legacy_checkpoint_sha256=expected_legacy_sha256,
        expected_synthetic_validation_names=validation_names[
            :SYNTHETIC_CHECKPOINT_VALIDATION_COUNT
        ],
        gate_source_count=FROZEN_GATE_SOURCE_COUNT,
        seed=PROTOCOL_SEED,
    )
    quarantine_names = tuple(quarantine["quarantine_names"])
    calibration_sources, frozen_gate_sources = deterministic_contamination_aware_split(
        table.source_names,
        table.active_source_indices,
        quarantine_names,
        FROZEN_GATE_SOURCE_COUNT,
        PROTOCOL_SEED,
    )
    if (
        len(quarantine_names) != QUARANTINE_SOURCE_COUNT
        or len(calibration_sources) != CALIBRATION_SOURCE_COUNT
        or len(frozen_gate_sources) != FROZEN_GATE_SOURCE_COUNT
    ):
        raise RuntimeError("quarantine-aware split is not the pinned 93/257/350 partition")
    calibration_names = tuple(
        sorted(table.source_names[int(index)] for index in calibration_sources)
    )
    gate_names = tuple(sorted(table.source_names[int(index)] for index in frozen_gate_sources))
    if source_name_list_sha256(calibration_names) != quarantine["name_sha256"]["calibration"]:
        raise RuntimeError("calibration source-name hash differs from the pinned quarantine")
    if source_name_list_sha256(gate_names) != quarantine["name_sha256"]["frozen_gate"]:
        raise RuntimeError("frozen-gate source-name hash differs from the pinned quarantine")
    source_index_by_name = {name: index for index, name in enumerate(table.source_names)}
    quarantine_sources = np.asarray(
        [source_index_by_name[name] for name in quarantine_names],
        dtype=np.int64,
    )

    selected_sources = select_visual_qa_sources(
        table.source_names,
        calibration_sources,
        pair_count=config.pair_count,
        seed=config.selection_seed,
    )
    validate_calibration_only_selection(
        selected_sources,
        calibration_sources,
        frozen_gate_sources,
        quarantine_sources,
    )
    selected_names = [table.source_names[int(index)] for index in selected_sources]
    protected_pixel_paths = tuple(
        root / "train" / kind / name
        for name in selected_names
        for kind in ("inputs", "targets")
    )
    forbidden_pixel_paths = tuple(
        root / "train" / kind / table.source_names[int(index)]
        for index in np.concatenate((quarantine_sources, frozen_gate_sources))
        for kind in ("inputs", "targets")
    )
    _validate_selected_pixel_file_identity(
        protected_pixel_paths,
        forbidden_pixel_paths,
    )
    protected_paths = (*protected_paths, *protected_pixel_paths)

    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != config.expected_checkpoint_sha256:
        raise ValueError("restorer checkpoint SHA256 does not match the pinned digest")
    model, device, checkpoint_metadata = load_restorer(
        checkpoint_path,
        device=config.device,
        state=config.state,
        allow_unpromoted=False,
    )
    if checkpoint_metadata.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("checkpoint changed while the restorer was loading")

    # Only selected calibration indices are ever handed to the sampler.
    panel = RealPairSampler(
        table,
        seed=config.selection_seed,
        cache_size=min(config.pair_count, 16),
    ).materialize_validation(
        source_indices=selected_sources,
        pairs_per_source=1,
        seed=config.selection_seed,
    )
    if not np.array_equal(panel.source_index.numpy(), selected_sources):
        raise RuntimeError("visual-QA materialization changed the selected source order")
    validate_calibration_only_selection(
        panel.source_index.numpy(),
        calibration_sources,
        frozen_gate_sources,
        quarantine_sources,
    )

    corrupt = _tensor_tiles_uint8(panel.corrupt)
    clean = _tensor_tiles_uint8(panel.clean)
    restored = restore_tiles_uint8(
        model,
        corrupt,
        device,
        batch_size=config.batch_size,
    )
    corrupt, restored, clean = _validate_tile_triplet(corrupt, restored, clean)
    rows = _row_records(panel, table.source_names)
    sheet = render_visual_qa_contact_sheet(
        corrupt,
        restored,
        clean,
        rows,
        checkpoint_sha256=checkpoint_sha256,
        selection_seed=config.selection_seed,
        tile_scale=config.tile_scale,
    )

    report = {
        "schema_version": 1,
        "kind": "denoise_calibration_visual_qa_contact_sheet",
        "protocol": {
            "name": "contamination_aware_quarantine_v1",
            "split_seed": PROTOCOL_SEED,
            "selection_seed": config.selection_seed,
            "selection_ranking": (
                "ascending SHA256 of selection seed, visual-qa-source, and source name"
            ),
            "primary_confidence_floor": PRIMARY_CONFIDENCE,
            "one_pair_per_distinct_source": True,
            "calibration_only": True,
            "quarantined_pixels_materialized": False,
            "frozen_gate_pixels_materialized": False,
            "frozen_gate_passed_to_model": False,
        },
        "config": asdict(config),
        "inputs": {
            "data_root": str(root.resolve(strict=False)),
            "manifest": str(manifest_path.resolve(strict=True)),
            "manifest_sha256": manifest_sha256,
            "validation_pairs": str(val_pairs_path.resolve(strict=True)),
            "validation_pairs_sha256": table.npz_sha256,
            "validation_quarantine": str(quarantine_path.resolve(strict=True)),
            "validation_quarantine_sha256": quarantine_sha256,
            "checkpoint": _checkpoint_provenance(checkpoint_metadata),
        },
        "source_partition": {
            "manifest_validation_source_count": EXPECTED_VALIDATION_SOURCE_COUNT,
            "quarantine_source_count": QUARANTINE_SOURCE_COUNT,
            "quarantine_source_names_sha256": quarantine["name_sha256"]["quarantine"],
            "calibration_source_count": CALIBRATION_SOURCE_COUNT,
            "calibration_source_names_sha256": quarantine["name_sha256"]["calibration"],
            "frozen_gate_source_count": FROZEN_GATE_SOURCE_COUNT,
            "frozen_gate_source_names_sha256": quarantine["name_sha256"]["frozen_gate"],
            "selected_calibration_source_count": len(selected_sources),
        },
        "selection": {
            "selection_sha256": selection_sha256(rows),
            "source_indices": selected_sources.tolist(),
            "source_names": selected_names,
            "rows": rows,
        },
        "tile_pixels": {
            "corrupt_sha256": _array_sha256(corrupt),
            "restored_sha256": _array_sha256(restored),
            "clean_sha256": _array_sha256(clean),
            "corrupt_restored_clean_sha256": _array_sha256(
                np.stack((corrupt, restored, clean), axis=1)
            ),
        },
        "rendering": {
            "layout": "corrupt | restored | clean",
            "tile_size": 20,
            "tile_scale": config.tile_scale,
            "resampling": "nearest",
            "pillow_version": PIL.__version__,
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "torch": torch.__version__,
        },
        "visual_qa_code_sha256": visual_qa_code_fingerprint(),
    }
    return write_visual_qa_outputs(
        sheet,
        report,
        output_png=output_png,
        report_json=report_json,
        overwrite=config.overwrite,
        protected_paths=protected_paths,
    )


__all__ = [
    "CALIBRATION_SOURCE_COUNT",
    "DEFAULT_TILE_SCALE",
    "DEFAULT_VISUAL_PAIR_COUNT",
    "FROZEN_GATE_SOURCE_COUNT",
    "PROTOCOL_SEED",
    "VisualQAConfig",
    "render_visual_qa_contact_sheet",
    "run_visual_qa",
    "select_visual_qa_sources",
    "selection_sha256",
    "validate_calibration_only_selection",
    "validate_visual_qa_config",
    "visual_qa_code_fingerprint",
    "write_visual_qa_outputs",
]
