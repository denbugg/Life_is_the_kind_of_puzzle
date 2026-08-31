"""Independent full-pipeline validator for the DRUNet protected submission.

The validator intentionally does not call the production prediction or blend
functions.  It independently reconstructs the layout, raw assembly,
harmonizers, tilewise neural inference, all three NLM arms, the protected mask
and the final blend for every official test board.
"""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

from aiijc_puzzle.compliant_drunet_protected_submission import (
    CANONICAL_DEVICE,
    CHECKPOINT_SHA256,
    DEFAULT_CHECKPOINT,
    DRUNET_BATCH_SIZE,
    DRUNET_SIGMA_255,
    EDGE_BUDGET,
    EXPECTED_HARMONIZER_SHA256,
    EXPECTED_POLICY,
    METHOD_STATUS,
    MODEL_PARAMETER_COUNT,
    NLM_SEARCH_WINDOW,
    NLM_STRENGTHS,
    NLM_TEMPLATE_WINDOW,
    PROOF_LIMITATION,
    PROOF_SCOPE,
    RESTORATION_NAME,
    SOBEL_THRESHOLD,
    PromotionEvidence,
    _method_declaration,
    build_runtime_manifest,
    load_and_validate_attestation,
    load_promotion_evidence,
)
from aiijc_puzzle.compliant_submission import (
    InputSnapshot,
    _require_regular_file,
    array_sha256,
    build_official_input_snapshot,
    decode_rgb_png,
    load_rgb_png,
)
from aiijc_puzzle.legacy_upgrade import directional_scores, solve_buddies
from aiijc_puzzle.postassembly_harmonizer import (
    DEFAULT_LUMINANCE_GAIN_CONFIG,
    DEFAULT_SEAM_GRAPH_CONFIG,
    apply_luminance_gains,
    apply_rgb_offsets,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
)
from aiijc_puzzle.pretrained_tile_denoiser import load_drunet_color
from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    RGB_CHANNELS,
    TILE_COUNT,
    TILE_SIZE,
    sha256_file,
)


def _strict_tiles(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    expected = (IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS)
    if value.shape != expected or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB board {expected}, got {value.dtype} {value.shape}")
    return np.ascontiguousarray(
        value.reshape(24, TILE_SIZE, 24, TILE_SIZE, RGB_CHANNELS)
        .transpose(0, 2, 1, 3, 4)
        .reshape(TILE_COUNT, TILE_SIZE, TILE_SIZE, RGB_CHANNELS)
    )


def _strict_assemble(tiles: np.ndarray) -> np.ndarray:
    value = np.asarray(tiles)
    expected = (TILE_COUNT, TILE_SIZE, TILE_SIZE, RGB_CHANNELS)
    if value.shape != expected or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 tile roster {expected}, got {value.dtype} {value.shape}")
    return np.ascontiguousarray(
        value.reshape(24, 24, TILE_SIZE, TILE_SIZE, RGB_CHANNELS)
        .transpose(0, 2, 1, 3, 4)
        .reshape(IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS)
    )


def _independent_layout_digest(layout: Sequence[int]) -> str:
    value = np.asarray(layout, dtype=np.int32)
    if value.shape != (TILE_COUNT,) or not np.array_equal(
        np.sort(value), np.arange(TILE_COUNT, dtype=np.int32)
    ):
        raise ValueError("layout is not a strict permutation")
    return hashlib.sha256(value.astype("<i4", copy=False).tobytes()).hexdigest()


def _independent_tile_multiset_sha256(image: np.ndarray) -> str:
    hashes = sorted(hashlib.sha256(tile.tobytes()).digest() for tile in _strict_tiles(image))
    return hashlib.sha256(b"".join(hashes)).hexdigest()


def _independent_layout(input_image: np.ndarray) -> np.ndarray:
    tiles = _strict_tiles(input_image)
    right, down = directional_scores(tiles, views=("bilateral",))["bilateral"]
    solved = solve_buddies(right, down, max_edges=EDGE_BUDGET)
    layout = np.asarray(solved.layout, dtype=np.int32)
    _independent_layout_digest(layout)
    return layout


def _independent_raw_assembly(input_image: np.ndarray, layout: Sequence[int]) -> np.ndarray:
    value = np.asarray(layout, dtype=np.int32)
    _independent_layout_digest(value)
    return _strict_assemble(_strict_tiles(input_image)[value])


def _independent_harmonize(raw: np.ndarray) -> np.ndarray:
    tiles = _strict_tiles(raw)
    offsets, _ = seam_graph_rgb_offsets(tiles, DEFAULT_SEAM_GRAPH_CONFIG)
    rgb_tiles = apply_rgb_offsets(tiles, offsets)
    gains, _ = seam_graph_luminance_gains(rgb_tiles, DEFAULT_LUMINANCE_GAIN_CONFIG)
    return _strict_assemble(apply_luminance_gains(rgb_tiles, gains))


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def _independent_drunet_tiles(
    model: torch.nn.Module,
    tiles: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    source = np.asarray(tiles)
    expected = (TILE_COUNT, TILE_SIZE, TILE_SIZE, RGB_CHANNELS)
    if source.shape != expected or source.dtype != np.uint8:
        raise ValueError("independent DRUNet requires all 576 uint8 RGB tiles")
    outputs: list[np.ndarray] = []
    model.eval()
    _synchronize(device)
    for start in range(0, TILE_COUNT, DRUNET_BATCH_SIZE):
        batch = (
            torch.from_numpy(np.ascontiguousarray(source[start : start + DRUNET_BATCH_SIZE]))
            .permute(0, 3, 1, 2)
            .to(device=device, dtype=torch.float32)
            .div_(255.0)
        )
        padded = functional.pad(batch, (0, 4, 0, 4), mode="reflect")
        noise = torch.full(
            (len(batch), 1, 24, 24),
            DRUNET_SIGMA_255 / 255.0,
            dtype=batch.dtype,
            device=device,
        )
        prediction = model(torch.cat((padded, noise), dim=1))
        prediction = prediction[..., :TILE_SIZE, :TILE_SIZE].clamp_(0.0, 1.0)
        array = prediction.permute(0, 2, 3, 1).float().cpu().numpy()
        outputs.append(np.rint(array * 255.0).clip(0, 255).astype(np.uint8))
    _synchronize(device)
    result = np.ascontiguousarray(np.concatenate(outputs, axis=0))
    if result.shape != expected:
        raise RuntimeError("independent DRUNet changed tile roster geometry")
    return result


def _independent_colored_nlm(image: np.ndarray, h: int) -> np.ndarray:
    if h not in NLM_STRENGTHS:
        raise ValueError("independent validator only permits frozen NLM strengths")
    value = _strict_assemble(_strict_tiles(image))
    bgr = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
    filtered = cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        h,
        h,
        NLM_TEMPLATE_WINDOW,
        NLM_SEARCH_WINDOW,
    )
    return np.ascontiguousarray(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))


def _independent_masks(h20: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    value = _strict_assemble(_strict_tiles(h20))
    gray = cv2.cvtColor(value, cv2.COLOR_RGB2GRAY).astype(np.float32)
    horizontal = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    vertical = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(horizontal, vertical)
    grid = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    for coordinate in range(TILE_SIZE, IMAGE_SIZE, TILE_SIZE):
        grid[:, coordinate - 1 : coordinate + 1] = True
        grid[coordinate - 1 : coordinate + 1, :] = True
    binary = (magnitude >= SOBEL_THRESHOLD) | grid
    dilated = cv2.dilate(
        binary.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    soft = cv2.GaussianBlur(
        dilated.astype(np.float32),
        (0, 0),
        sigmaX=1.0,
        sigmaY=1.0,
    )
    soft = np.clip(soft, 0.0, 1.0)
    return dilated, soft, float(dilated.mean())


def _independent_restore(
    harmonized: np.ndarray,
    model: torch.nn.Module,
    *,
    device: torch.device,
) -> dict[str, Any]:
    restored_tiles = _independent_drunet_tiles(model, _strict_tiles(harmonized), device=device)
    canvas = _strict_assemble(restored_tiles)
    h20 = _independent_colored_nlm(canvas, 20)
    h28 = _independent_colored_nlm(canvas, 28)
    h40 = _independent_colored_nlm(canvas, 40)
    binary, soft, protected_fraction = _independent_masks(h20)
    mixed = np.rint(
        soft[..., None] * h28.astype(np.float32) + (1.0 - soft[..., None]) * h40.astype(np.float32)
    )
    output = np.ascontiguousarray(mixed.clip(0, 255).astype(np.uint8))
    return {
        "drunet_tiles": restored_tiles,
        "drunet_canvas": canvas,
        "h20": h20,
        "h28": h28,
        "h40": h40,
        "binary": binary,
        "soft": soft,
        "protected_fraction": protected_fraction,
        "output": output,
    }


def _inspect_submission_members(
    archive: zipfile.ZipFile,
    *,
    expected_names: Sequence[str],
) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise ValueError("submission ZIP contains duplicate names")
    if names != list(expected_names):
        raise ValueError("submission ZIP root roster/order differs from official inputs")
    result: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        if (
            info.is_dir()
            or "/" in info.filename
            or "\\" in info.filename
            or not info.filename.endswith(".png")
            or bool(info.flag_bits & 0x1)
            or unix_mode != 0o100644
            or info.compress_type != zipfile.ZIP_DEFLATED
            or info.file_size <= 0
        ):
            raise ValueError(f"unsafe submission member: {info.filename!r}")
        result[info.filename] = info
    return result


def _validate_restoration_metadata(restoration: Mapping[str, Any]) -> None:
    if restoration.get("name") != RESTORATION_NAME:
        raise ValueError("restoration name changed")
    fixed_flags = {
        "input_is_raw_assembly": True,
        "pixel_restoration_only": True,
        "layout_changed": False,
        "spatial_warp_used": False,
        "external_or_cross_board_pixels_used": False,
    }
    if any(restoration.get(key) != value for key, value in fixed_flags.items()):
        raise ValueError("restoration geometry/provenance declaration changed")
    harmonizers = restoration.get("harmonizers", {})
    if harmonizers != {
        "order": ["rgb_seam_offsets", "bounded_luminance_gains"],
        "config_sha256": dict(EXPECTED_HARMONIZER_SHA256),
    }:
        raise ValueError("harmonizer declaration changed")
    drunet = restoration.get("drunet", {})
    required_drunet = {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "parameter_count": MODEL_PARAMETER_COUNT,
        "sigma_255": DRUNET_SIGMA_255,
        "tile_count": TILE_COUNT,
        "tile_input_size": [TILE_SIZE, TILE_SIZE],
        "same_tile_reflect_padding_right_bottom": [4, 4],
        "exact_crop_size": [TILE_SIZE, TILE_SIZE],
        "batch_size": DRUNET_BATCH_SIZE,
        "cross_tile_context": False,
        "cross_board_context": False,
    }
    if any(drunet.get(key) != value for key, value in required_drunet.items()):
        raise ValueError("DRUNet declaration changed")
    nlm = restoration.get("nlm", {})
    required_nlm = {
        "proper_rgb_bgr_roundtrip": True,
        "independent_single_pass_strengths": list(NLM_STRENGTHS),
        "h_color_equals_h": True,
        "template_window_size": NLM_TEMPLATE_WINDOW,
        "search_window_size": NLM_SEARCH_WINDOW,
    }
    if any(nlm.get(key) != value for key, value in required_nlm.items()):
        raise ValueError("NLM declaration changed")
    mask = restoration.get("protected_mask", {})
    required_mask = {
        "source": "independent DRUNet+h20",
        "sobel_threshold": SOBEL_THRESHOLD,
        "grid_period": TILE_SIZE,
        "dilation": "3x3 one iteration",
        "softening": "Gaussian sigma1",
    }
    if any(mask.get(key) != value for key, value in required_mask.items()):
        raise ValueError("protected-mask declaration changed")
    if restoration.get("blend") != "rint(soft*h28 + (1-soft)*h40), clip uint8":
        raise ValueError("protected blend declaration changed")


def validate_against_snapshot(
    *,
    snapshot: InputSnapshot,
    inputs_dir: Path,
    submission_zip: Path,
    attestation_path: Path,
    promotion: PromotionEvidence,
    runtime_manifest: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Recompute every declared intermediate for every board in one snapshot."""

    if device.type != CANONICAL_DEVICE:
        raise ValueError("hash-exact independent validation requires canonical MPS")
    submission_zip = _require_regular_file(submission_zip)
    attestation = load_and_validate_attestation(attestation_path)
    if attestation["status"] != METHOD_STATUS:
        raise ValueError("attestation overclaims the method status")
    if attestation["scope"] != PROOF_SCOPE:
        raise ValueError("attestation scope changed")
    if attestation["correct_hidden_layout_proven"] is not False:
        raise ValueError("attestation must not claim hidden-layout correctness")
    if attestation["proof_limitation"] != PROOF_LIMITATION:
        raise ValueError("attestation proof limitation changed")
    if attestation["policy"] != EXPECTED_POLICY:
        raise ValueError("attestation policy changed")
    if attestation["promotion_evidence"] != promotion.as_dict():
        raise ValueError("attestation promotion evidence changed")
    if attestation["runtime_manifest"] != runtime_manifest:
        raise ValueError("attestation runtime manifest changed")
    if attestation["method"] != _method_declaration(promotion, runtime_manifest):
        raise ValueError("attestation method declaration changed")
    if attestation["input_snapshot"] != snapshot.attestation_record():
        raise ValueError("attestation input snapshot changed")
    if attestation["canonical_execution"] != {
        "device": CANONICAL_DEVICE,
        "cuda_or_cpu_reproduction_is_noncanonical": True,
        "noncanonical_backend_may_differ_by_one_lsb": True,
    }:
        raise ValueError("canonical/noncanonical execution disclosure changed")

    archive_sha256 = sha256_file(submission_zip)
    archive_record = attestation["archive"]
    if archive_record["sha256"] != archive_sha256:
        raise ValueError("submission ZIP hash differs from attestation")
    if archive_record["filenames"] != list(snapshot.filenames):
        raise ValueError("attested ZIP roster differs from official inputs")
    records = attestation["per_board"]
    if [record["filename"] for record in records] != list(snapshot.filenames):
        raise ValueError("per-board evidence roster differs from official inputs")

    # Deliberately load a new model instead of reusing the production instance.
    model = load_drunet_color(DEFAULT_CHECKPOINT, device)
    if sum(parameter.numel() for parameter in model.parameters()) != MODEL_PARAMETER_COUNT:
        raise ValueError("independent official DRUNet parameter count changed")
    input_hashes = snapshot.hashes_by_name
    recomputed = 0
    with zipfile.ZipFile(submission_zip) as archive:
        members = _inspect_submission_members(archive, expected_names=snapshot.filenames)
        for record in records:
            name = record["filename"]
            if record["input_sha256"] != input_hashes[name]:
                raise ValueError(f"input hash evidence changed: {name}")
            input_image = load_rgb_png(inputs_dir / name, expected_sha256=input_hashes[name])
            layout = np.asarray(record["tile_at_position"], dtype=np.int32)
            if _independent_layout_digest(layout) != record["layout_sha256"]:
                raise ValueError(f"layout digest changed: {name}")
            solver_layout = _independent_layout(input_image)
            if not np.array_equal(layout, solver_layout):
                raise ValueError(f"layout differs from frozen bilateral buddies96: {name}")
            raw = _independent_raw_assembly(input_image, solver_layout)
            if array_sha256(raw) != record["raw_assembly_sha256"]:
                raise ValueError(f"raw assembly hash changed: {name}")
            input_multiset = _independent_tile_multiset_sha256(input_image)
            if _independent_tile_multiset_sha256(raw) != input_multiset:
                raise ValueError(f"raw tile multiset differs from input: {name}")
            if record["input_and_raw_tile_multiset_sha256"] != input_multiset:
                raise ValueError(f"attested tile multiset hash changed: {name}")
            audit = record["raw_permutation_audit"]
            required_audit = {
                "grid_rows": 24,
                "grid_columns": 24,
                "tile_count": TILE_COUNT,
                "unique_tile_indices": TILE_COUNT,
                "missing_tile_indices": [],
                "duplicate_tile_indices": [],
                "exact_reassembly_from_declared_layout": True,
                "input_output_tile_multiset_equal": True,
                "raw_input_pixels_preserved": True,
                "restoration_applied_after_audit": True,
                "passed": True,
            }
            if audit != required_audit:
                raise ValueError(f"raw permutation audit declaration changed: {name}")

            harmonized = _independent_harmonize(raw)
            restoration = record["restoration"]
            _validate_restoration_metadata(restoration)
            if array_sha256(harmonized) != restoration["harmonized_array_sha256"]:
                raise ValueError(f"harmonized hash changed: {name}")
            derived = _independent_restore(harmonized, model, device=device)
            drunet = restoration["drunet"]
            nlm = restoration["nlm"]
            mask = restoration["protected_mask"]
            checks = {
                "DRUNet tiles": (
                    derived["drunet_tiles"],
                    drunet["restored_tiles_array_sha256"],
                ),
                "DRUNet canvas": (
                    derived["drunet_canvas"],
                    drunet["restored_canvas_array_sha256"],
                ),
                "NLM h20": (derived["h20"], nlm["h20_array_sha256"]),
                "NLM h28": (derived["h28"], nlm["h28_array_sha256"]),
                "NLM h40": (derived["h40"], nlm["h40_array_sha256"]),
                "binary mask": (derived["binary"], mask["binary_array_sha256"]),
                "soft mask": (derived["soft"], mask["soft_array_sha256"]),
                "output": (derived["output"], restoration["output_array_sha256"]),
            }
            for label, (array, expected_sha256) in checks.items():
                if array_sha256(array) != expected_sha256:
                    raise ValueError(f"{label} hash changed: {name}")
            if derived["protected_fraction"] != mask["protected_fraction"]:
                raise ValueError(f"protected fraction changed: {name}")

            payload = archive.read(members[name])
            if hashlib.sha256(payload).hexdigest() != record["output_png_sha256"]:
                raise ValueError(f"output PNG hash changed: {name}")
            output = decode_rgb_png(payload, context=f"{submission_zip}:{name}")
            if not np.array_equal(output, derived["output"]):
                raise ValueError(f"ZIP output differs from full independent pipeline: {name}")
            recomputed += 1

    if recomputed != snapshot.file_count:
        raise RuntimeError("independent validator did not recompute every board")
    return {
        "status": METHOD_STATUS,
        "scope": PROOF_SCOPE,
        "correct_hidden_layout_proven": False,
        "proof_limitation": PROOF_LIMITATION,
        "file_count": snapshot.file_count,
        "boards_fully_recomputed": recomputed,
        "source_archive_sha256": snapshot.source_archive_sha256,
        "filenames_sha256": snapshot.filenames_sha256,
        "submission_zip_sha256": archive_sha256,
        "promotion_config_sha256": promotion.config_sha256,
        "runtime_manifest_sha256": runtime_manifest["digest_sha256"],
        "canonical_device": CANONICAL_DEVICE,
        "all_solver_layouts_recomputed": True,
        "all_raw_reassemblies_and_tile_multisets_recomputed": True,
        "all_harmonized_arrays_recomputed": True,
        "all_independent_tile_drunet_outputs_recomputed": True,
        "all_h20_h28_h40_masks_and_blends_recomputed": True,
        "all_zip_pngs_rgb480_and_hash_matched": True,
    }


def validate_submission(
    *,
    inputs_dir: Path,
    source_archive: Path,
    submission_zip: Path,
    attestation_path: Path,
) -> dict[str, Any]:
    """Public canonical validator bound to the exact official 700-board set."""

    if not torch.backends.mps.is_available():
        raise RuntimeError("canonical independent validation requires Apple MPS")
    snapshot = build_official_input_snapshot(inputs_dir, source_archive)
    promotion = load_promotion_evidence()
    runtime_manifest = build_runtime_manifest()
    return validate_against_snapshot(
        snapshot=snapshot,
        inputs_dir=inputs_dir,
        submission_zip=submission_zip,
        attestation_path=attestation_path,
        promotion=promotion,
        runtime_manifest=runtime_manifest,
        device=torch.device(CANONICAL_DEVICE),
    )


__all__ = [
    "validate_against_snapshot",
    "validate_submission",
]
