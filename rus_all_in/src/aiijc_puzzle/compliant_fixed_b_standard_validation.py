"""Independent full recomputation for the fixed-B standard submission.

No production prediction, renderer, blend, or layout helper is called.  The
validator independently rebuilds every one of 700 test outputs and checks the
root-only ZIP, RGB geometry, filenames, hashes, strict upright tile bijection,
and all restoration intermediates.
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

from aiijc_puzzle.compliant_fixed_b_standard_submission import (
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
        raise ValueError(f"expected uint8 RGB board {expected}")
    return np.ascontiguousarray(
        value.reshape(24, TILE_SIZE, 24, TILE_SIZE, RGB_CHANNELS)
        .transpose(0, 2, 1, 3, 4)
        .reshape(TILE_COUNT, TILE_SIZE, TILE_SIZE, RGB_CHANNELS)
    )


def _strict_assemble(tiles: np.ndarray) -> np.ndarray:
    value = np.asarray(tiles)
    expected = (TILE_COUNT, TILE_SIZE, TILE_SIZE, RGB_CHANNELS)
    if value.shape != expected or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 tile roster {expected}")
    return np.ascontiguousarray(
        value.reshape(24, 24, TILE_SIZE, TILE_SIZE, RGB_CHANNELS)
        .transpose(0, 2, 1, 3, 4)
        .reshape(IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS)
    )


def _layout_digest(layout: Sequence[int]) -> str:
    value = np.asarray(layout, dtype=np.int32)
    if value.shape != (TILE_COUNT,) or not np.array_equal(
        np.sort(value), np.arange(TILE_COUNT, dtype=np.int32)
    ):
        raise ValueError("layout is not a strict 0..575 permutation")
    return hashlib.sha256(value.astype("<i4", copy=False).tobytes()).hexdigest()


def _tile_multiset_sha256(image: np.ndarray) -> str:
    hashes = sorted(hashlib.sha256(tile.tobytes()).digest() for tile in _strict_tiles(image))
    return hashlib.sha256(b"".join(hashes)).hexdigest()


def _independent_layout(input_image: np.ndarray) -> np.ndarray:
    tiles = _strict_tiles(input_image)
    right, down = directional_scores(tiles, views=("bilateral",))["bilateral"]
    solved = solve_buddies(right, down, max_edges=EDGE_BUDGET)
    layout = np.asarray(solved.layout, dtype=np.int32)
    _layout_digest(layout)
    return layout


def _independent_harmonize(raw: np.ndarray) -> np.ndarray:
    tiles = _strict_tiles(raw)
    offsets, _ = seam_graph_rgb_offsets(tiles, DEFAULT_SEAM_GRAPH_CONFIG)
    rgb = apply_rgb_offsets(tiles, offsets)
    gains, _ = seam_graph_luminance_gains(rgb, DEFAULT_LUMINANCE_GAIN_CONFIG)
    return _strict_assemble(apply_luminance_gains(rgb, gains))


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
        raise RuntimeError("independent DRUNet changed tile geometry")
    return result


def _independent_colored_nlm(image: np.ndarray, h: int) -> np.ndarray:
    if h not in NLM_STRENGTHS:
        raise ValueError("validator only permits frozen NLM strengths")
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
        binary.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1
    ).astype(bool)
    soft = cv2.GaussianBlur(dilated.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0)
    soft = np.clip(soft, 0.0, 1.0)
    return dilated, soft, float(dilated.mean())


def _typed_array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = f"{array.dtype.str}|{','.join(map(str, array.shape))}|".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def _pixel_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


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
    h50 = _independent_colored_nlm(canvas, 50)
    binary, soft, protected_fraction = _independent_masks(h20)
    mixed = np.rint(
        soft[..., None] * h28.astype(np.float32) + (1.0 - soft[..., None]) * h50.astype(np.float32)
    )
    output = np.ascontiguousarray(mixed.clip(0, 255).astype(np.uint8))
    return {
        "drunet_canvas": canvas,
        "h20": h20,
        "h28": h28,
        "h50": h50,
        "binary": binary,
        "soft": soft,
        "protected_fraction": protected_fraction,
        "output": output,
    }


def _inspect_submission_members(
    archive: zipfile.ZipFile, *, expected_names: Sequence[str]
) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)) or names != list(expected_names):
        raise ValueError("submission ZIP roster/order differs from official inputs")
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
    fixed = {
        "name": RESTORATION_NAME,
        "input_is_raw_assembly": True,
        "pixel_restoration_only": True,
        "layout_changed": False,
        "spatial_warp_used": False,
        "constant_or_near_flat_tile_substitution_used": False,
        "external_or_cross_board_pixels_used": False,
        "blend": "rint(soft*h28 + (1-soft)*h50), clip uint8",
    }
    if any(restoration.get(key) != value for key, value in fixed.items()):
        raise ValueError("restoration declaration changed")
    if restoration.get("harmonizers") != {
        "order": ["rgb_seam_offsets", "bounded_luminance_gains"],
        "config_sha256": dict(EXPECTED_HARMONIZER_SHA256),
    }:
        raise ValueError("harmonizer declaration changed")
    drunet = restoration.get("drunet", {})
    expected_drunet = {
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
    if any(drunet.get(key) != value for key, value in expected_drunet.items()):
        raise ValueError("DRUNet declaration changed")
    nlm = restoration.get("nlm", {})
    expected_nlm = {
        "proper_rgb_bgr_roundtrip": True,
        "independent_single_pass_strengths": list(NLM_STRENGTHS),
        "h_color_equals_h": True,
        "template_window_size": NLM_TEMPLATE_WINDOW,
        "search_window_size": NLM_SEARCH_WINDOW,
    }
    if any(nlm.get(key) != value for key, value in expected_nlm.items()):
        raise ValueError("NLM declaration changed")
    mask = restoration.get("protected_mask", {})
    expected_mask = {
        "source": "independent DRUNet50+h20",
        "sobel_threshold": SOBEL_THRESHOLD,
        "grid_period": TILE_SIZE,
        "dilation": "3x3 one iteration",
        "softening": "Gaussian sigma1",
    }
    if any(mask.get(key) != value for key, value in expected_mask.items()):
        raise ValueError("protected mask declaration changed")


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
    if device.type != CANONICAL_DEVICE:
        raise ValueError("hash-exact independent validation requires canonical MPS")
    submission_zip = _require_regular_file(submission_zip)
    attestation = load_and_validate_attestation(attestation_path)
    exact_top = {
        "status": METHOD_STATUS,
        "scope": PROOF_SCOPE,
        "correct_hidden_layout_proven": False,
        "proof_limitation": PROOF_LIMITATION,
        "policy": EXPECTED_POLICY,
        "promotion_evidence": promotion.as_dict(),
        "runtime_manifest": runtime_manifest,
        "input_snapshot": snapshot.attestation_record(),
        "canonical_execution": {
            "device": CANONICAL_DEVICE,
            "cuda_or_cpu_reproduction_is_noncanonical": True,
            "noncanonical_backend_may_differ_by_one_lsb": True,
        },
    }
    for key, value in exact_top.items():
        if attestation.get(key) != value:
            raise ValueError(f"attestation binding changed: {key}")
    if attestation.get("method") != _method_declaration(promotion, runtime_manifest):
        raise ValueError("attestation method changed")

    archive_sha256 = sha256_file(submission_zip)
    archive_record = attestation["archive"]
    if (
        archive_record.get("sha256") != archive_sha256
        or archive_record.get("file_count") != 700
        or archive_record.get("root_only") is not True
        or archive_record.get("format") != "PNG"
        or archive_record.get("mode") != "RGB"
        or archive_record.get("width") != IMAGE_SIZE
        or archive_record.get("height") != IMAGE_SIZE
        or archive_record.get("filenames") != list(snapshot.filenames)
    ):
        raise ValueError("attested ZIP contract changed")
    records = attestation["per_board"]
    if [record["filename"] for record in records] != list(snapshot.filenames):
        raise ValueError("per-board roster differs from official inputs")

    model = load_drunet_color(DEFAULT_CHECKPOINT, device)
    if sum(parameter.numel() for parameter in model.parameters()) != MODEL_PARAMETER_COUNT:
        raise ValueError("independent model parameter count changed")
    recomputed = 0
    with zipfile.ZipFile(submission_zip) as archive:
        members = _inspect_submission_members(archive, expected_names=snapshot.filenames)
        for record in records:
            name = record["filename"]
            input_hash = snapshot.hashes_by_name[name]
            if record["input_sha256"] != input_hash:
                raise ValueError(f"input hash evidence changed: {name}")
            image = load_rgb_png(inputs_dir / name, expected_sha256=input_hash)
            declared_layout = np.asarray(record["tile_at_position"], dtype=np.int32)
            if _layout_digest(declared_layout) != record["layout_sha256"]:
                raise ValueError(f"layout digest changed: {name}")
            solver_layout = _independent_layout(image)
            if not np.array_equal(declared_layout, solver_layout):
                raise ValueError(f"layout differs from buddies96: {name}")
            raw = _strict_assemble(_strict_tiles(image)[solver_layout])
            if array_sha256(raw) != record["raw_assembly_sha256"]:
                raise ValueError(f"raw assembly hash changed: {name}")
            multiset = _tile_multiset_sha256(image)
            if _tile_multiset_sha256(raw) != multiset:
                raise ValueError(f"raw tile multiset differs from input: {name}")
            if record["input_and_raw_tile_multiset_sha256"] != multiset:
                raise ValueError(f"attested tile multiset changed: {name}")
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
            if record["raw_permutation_audit"] != required_audit:
                raise ValueError(f"raw audit declaration changed: {name}")

            harmonized = _independent_harmonize(raw)
            restoration = record["restoration"]
            _validate_restoration_metadata(restoration)
            if array_sha256(harmonized) != restoration["harmonized_array_sha256"]:
                raise ValueError(f"harmonized hash changed: {name}")
            derived = _independent_restore(harmonized, model, device=device)
            drunet = restoration["drunet"]
            nlm = restoration["nlm"]
            mask = restoration["protected_mask"]
            pixel_checks = {
                "DRUNet50 canvas": (
                    derived["drunet_canvas"],
                    drunet["restored_canvas_pixel_sha256"],
                ),
                "NLM h20": (derived["h20"], nlm["h20_pixel_sha256"]),
                "NLM h28": (derived["h28"], nlm["h28_pixel_sha256"]),
                "NLM h50": (derived["h50"], nlm["h50_pixel_sha256"]),
            }
            for label, (array, expected) in pixel_checks.items():
                if _pixel_digest(array) != expected:
                    raise ValueError(f"{label} hash changed: {name}")
            if _typed_array_digest(derived["binary"]) != mask["binary_array_sha256"]:
                raise ValueError(f"binary mask changed: {name}")
            if _typed_array_digest(derived["soft"]) != mask["soft_array_sha256"]:
                raise ValueError(f"soft mask changed: {name}")
            if derived["protected_fraction"] != mask["protected_fraction"]:
                raise ValueError(f"protected fraction changed: {name}")
            if array_sha256(derived["output"]) != restoration["output_array_sha256"]:
                raise ValueError(f"output array hash changed: {name}")

            payload = archive.read(members[name])
            if hashlib.sha256(payload).hexdigest() != record["output_png_sha256"]:
                raise ValueError(f"output PNG hash changed: {name}")
            output = decode_rgb_png(payload, context=f"{submission_zip}:{name}")
            if not np.array_equal(output, derived["output"]):
                raise ValueError(f"ZIP output differs from independent pipeline: {name}")
            recomputed += 1

    if recomputed != 700:
        raise RuntimeError("independent validator did not recompute all 700 boards")
    return {
        "status": METHOD_STATUS,
        "scope": PROOF_SCOPE,
        "correct_hidden_layout_proven": False,
        "proof_limitation": PROOF_LIMITATION,
        "file_count": 700,
        "boards_fully_recomputed": recomputed,
        "source_archive_sha256": snapshot.source_archive_sha256,
        "filenames_sha256": snapshot.filenames_sha256,
        "submission_zip_sha256": archive_sha256,
        "promotion_config_sha256": promotion.config_sha256,
        "runtime_manifest_sha256": runtime_manifest["digest_sha256"],
        "canonical_device": CANONICAL_DEVICE,
        "all_solver_layouts_recomputed": True,
        "all_576_tile_bijections_and_raw_multisets_recomputed": True,
        "all_harmonized_arrays_recomputed": True,
        "all_independent_tile_drunet50_outputs_recomputed": True,
        "all_h20_h28_h50_t60_masks_and_blends_recomputed": True,
        "all_zip_members_root_only_rgb480_and_hash_matched": True,
        "no_constants_substitution_rotations_resize_or_warps": True,
    }


def validate_submission(
    *,
    inputs_dir: Path,
    source_archive: Path,
    submission_zip: Path,
    attestation_path: Path,
) -> dict[str, Any]:
    """Validate authorization before reading any competition-test input."""

    promotion = load_promotion_evidence()
    runtime_manifest = build_runtime_manifest()
    if not torch.backends.mps.is_available():
        raise RuntimeError("canonical independent validation requires Apple MPS")
    snapshot = build_official_input_snapshot(inputs_dir, source_archive)
    return validate_against_snapshot(
        snapshot=snapshot,
        inputs_dir=inputs_dir,
        submission_zip=submission_zip,
        attestation_path=attestation_path,
        promotion=promotion,
        runtime_manifest=runtime_manifest,
        device=torch.device(CANONICAL_DEVICE),
    )


__all__ = ["validate_against_snapshot", "validate_submission"]
