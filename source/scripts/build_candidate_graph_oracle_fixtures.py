#!/usr/bin/env python3
"""Prepare physically separated opaque fixtures for the graph-oracle audit.

This is the only target-aware preparation step.  It verifies the frozen
protocol and the already-pinned Phase-A evaluator/tests before opening a clean
target.  It then applies a second secret slot permutation, publishes an
input-only fixture root and a physically separate label root, and binds both
manifests in a durable lock file.  The master secret is never printed and is
stored only below the label root.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import stat
import sys
import uuid
import platform as platform_module
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from puzzle_assembly.geometry import TILE, TILE_COUNT, validate_permutation
from puzzle_assembly.panels import ExactPanel, make_exact_panel
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from scripts.update_candidate_graph_oracle_ledger import advance_state


SCHEMA_VERSION = 1
EXPECTED_KIND = "candidate_graph_oracle_ceiling"
INPUT_KIND = "candidate_graph_oracle_fixture_inputs"
LABEL_KIND = "candidate_graph_oracle_fixture_labels"
LOCK_KIND = "candidate_graph_oracle_fixture_lock"
PREP_MARKER_KIND = "candidate_graph_oracle_fixture_pixel_access_started"
INPUT_MANIFEST_NAME = "fixture_input_manifest.json"
LABEL_MANIFEST_NAME = "fixture_label_manifest.json"
SECRET_NAME = "FIXTURE_MASTER_SECRET.bin"
OPAQUE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
EXPECTED_TILE_SHAPE = (TILE_COUNT, TILE, TILE, 3)
EXPECTED_TARGET_SHAPE = (480, 480, 3)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _canonical_object_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"not an unlinked regular file: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.asarray(values)
    contiguous = array if array.flags.c_contiguous else np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, payload: Mapping[str, Any], *, mode: int = 0o644) -> None:
    _atomic_bytes(path, _canonical_json_bytes(payload), mode=mode)


def _exclusive_secret(path: Path, secret: bytes) -> None:
    """Create the secret directly with O_EXCL; never expose it via a temp file."""

    if len(secret) != 32:
        raise RuntimeError("fixture master secret must contain exactly 32 bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        written = 0
        while written < len(secret):
            written += os.write(descriptor, secret[written:])
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size != 32
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise RuntimeError("fixture master secret fstat contract failed")
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _require_regular_unlinked_file(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise RuntimeError(f"{label} may not be a symlink: {path}")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"{label} must be a regular file: {path}")
    if info.st_nlink != 1:
        raise RuntimeError(f"{label} may not be hardlinked: {path}")


def _resolve_repo_path(repo_root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError(f"invalid repository-relative path: {relative!r}")
    candidate = (repo_root / relative).resolve(strict=True)
    try:
        candidate.relative_to(repo_root.resolve(strict=True))
    except ValueError as error:
        raise RuntimeError(f"repository path escapes root: {relative}") from error
    _require_regular_unlinked_file(candidate, label="pinned repository artifact")
    return candidate


def _verify_hash(path: Path, expected: str, *, label: str) -> None:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise RuntimeError(f"{label} has no valid pinned SHA-256")
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: {path}")


def _assert_no_symlink_ancestors(path: Path) -> None:
    absolute = path.expanduser().absolute()
    existing = absolute
    while not existing.exists() and not existing.is_symlink():
        if existing.parent == existing:
            break
        existing = existing.parent
    cursor = existing
    while True:
        if cursor.is_symlink():
            raise RuntimeError(f"symlinked path component is forbidden: {cursor}")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent


def _assert_fresh_outputs(input_root: Path, label_root: Path, lock_path: Path, marker: Path) -> None:
    paths = [input_root, label_root, lock_path, marker]
    for path in paths:
        _assert_no_symlink_ancestors(path)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"output already exists: {path}")
    resolved = [path.expanduser().absolute() for path in paths]
    roots = resolved[:2]
    if resolved[2].parent != resolved[3].parent:
        raise RuntimeError("fixture lock and prep marker must share one control root")
    control_root = resolved[2].parent
    if roots[0] == roots[1]:
        raise RuntimeError("input and label roots must be distinct")
    for first, second in ((roots[0], roots[1]), (roots[1], roots[0])):
        try:
            second.relative_to(first)
        except ValueError:
            pass
        else:
            raise RuntimeError("input and label roots may not be nested")
    for file_path in resolved[2:]:
        for root in roots:
            try:
                file_path.relative_to(root)
            except ValueError:
                continue
            raise RuntimeError("lock and marker must live outside both fixture roots")
    for root in roots:
        for first, second in ((root, control_root), (control_root, root)):
            try:
                second.relative_to(first)
            except ValueError:
                pass
            else:
                raise RuntimeError("fixture input, label, and control roots may not be nested")


def _pinned_fixture_relative_parts(
    pins: Mapping[str, Any],
    field: str,
    *,
    expected_filename: str,
) -> tuple[str, ...]:
    value = pins.get(field)
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise RuntimeError(f"immutable fixture path is invalid: {field}")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or len(relative.parts) != 2
        or any(part in ("", ".", "..") for part in relative.parts)
        or relative.name != expected_filename
    ):
        raise RuntimeError(f"immutable fixture path is invalid: {field}")
    return relative.parts


def _expected_fixture_output_paths(
    config: Mapping[str, Any], bundle_root: Path
) -> dict[str, Path]:
    pins = config.get("runtime_pins")
    if not isinstance(pins, dict):
        raise RuntimeError("protocol has no runtime_pins")
    relative_contract = {
        "input_manifest": (
            "fixture_input_manifest_relative_path",
            INPUT_MANIFEST_NAME,
        ),
        "label_manifest": (
            "fixture_label_manifest_relative_path",
            LABEL_MANIFEST_NAME,
        ),
        "lock_path": ("fixture_lock_relative_path", "fixture_lock.json"),
        "marker_path": (
            "fixture_prep_marker_relative_path",
            "FIXTURE_PIXEL_ACCESS_STARTED.json",
        ),
    }
    expected_files: dict[str, Path] = {}
    for output_name, (field, filename) in relative_contract.items():
        parts = _pinned_fixture_relative_parts(
            pins, field, expected_filename=filename
        )
        expected_files[output_name] = bundle_root.joinpath(*parts)

    input_root = expected_files["input_manifest"].parent
    label_root = expected_files["label_manifest"].parent
    control_root = expected_files["lock_path"].parent
    if expected_files["marker_path"].parent != control_root:
        raise RuntimeError("immutable fixture lock and marker roots differ")
    roots = (input_root, label_root, control_root)
    if any(root.parent != bundle_root for root in roots) or len(set(roots)) != 3:
        raise RuntimeError(
            "immutable fixture roots must be three distinct sibling directories "
            "under one bundle root"
        )
    return {
        "input_root": input_root,
        "label_root": label_root,
        "lock_path": expected_files["lock_path"],
        "marker_path": expected_files["marker_path"],
    }


def _assert_exact_fixture_output_paths(
    *,
    config: Mapping[str, Any],
    repo_root: Path,
    input_root: Path,
    label_root: Path,
    lock_path: Path,
    marker_path: Path,
) -> Path:
    candidate_bundle_roots = {
        input_root.parent,
        label_root.parent,
        lock_path.parent.parent,
        marker_path.parent.parent,
    }
    if len(candidate_bundle_roots) != 1:
        raise RuntimeError("fixture outputs do not share one bundle root")
    bundle_root = next(iter(candidate_bundle_roots))
    if bundle_root == repo_root:
        raise RuntimeError("fixture bundle root may not be the repository root")
    expected = _expected_fixture_output_paths(config, bundle_root)
    actual = {
        "input_root": input_root,
        "label_root": label_root,
        "lock_path": lock_path,
        "marker_path": marker_path,
    }
    for name, expected_path in expected.items():
        if actual[name] != expected_path:
            raise RuntimeError(
                f"immutable fixture output path mismatch for {name}: "
                f"expected {expected_path}, got {actual[name]}"
            )
    return bundle_root


def _load_rgb_target(path: Path) -> np.ndarray:
    _require_regular_unlinked_file(path, label="clean target")
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != EXPECTED_TARGET_SHAPE:
        raise RuntimeError(f"clean target has wrong shape: {path} {values.shape}")
    return np.ascontiguousarray(values)


def _opaque_material(secret: bytes, prefix: str, source: str, panel: str) -> bytes:
    message = f"{prefix}:{source}:{panel}".encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).digest()


def _opaque_id(secret: bytes, source: str, panel: str) -> str:
    return _opaque_material(secret, "id", source, panel)[:16].hex()


def _opaque_permutation(secret: bytes, source: str, panel: str) -> np.ndarray:
    seed = int.from_bytes(
        _opaque_material(secret, "shuffle", source, panel)[:8], "big", signed=False
    )
    return np.random.Generator(np.random.PCG64(seed)).permutation(TILE_COUNT).astype(
        np.int32
    )


def _qap_seed(opaque_id: str) -> int:
    """Return a fixed nuisance seed without leaking or linking source identity."""

    if not OPAQUE_ID_RE.fullmatch(opaque_id):
        raise RuntimeError("invalid opaque id for QAP seed derivation")
    digest = hashlib.sha256(f"qap:{opaque_id}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _array_descriptor(values: np.ndarray, semantic: str) -> dict[str, Any]:
    array = np.asarray(values)
    contiguous = array if array.flags.c_contiguous else np.ascontiguousarray(array)
    return {
        "semantic": semantic,
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "c_order_sha256": _array_sha256(contiguous),
    }


def _artifact_descriptor(path: Path, root: Path) -> dict[str, Any]:
    _require_regular_unlinked_file(path, label="fixture artifact")
    if path.stat().st_dev != root.stat().st_dev:
        raise RuntimeError("fixture artifact crosses a mount boundary")
    relative = path.relative_to(root).as_posix()
    if relative.startswith("/") or ".." in Path(relative).parts:
        raise RuntimeError("fixture artifact path is not contained")
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _assert_exact_fixture_tree(
    root: Path,
    *,
    top_level_files: set[str],
    record_ids: set[str],
) -> None:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"invalid fixture root: {root}")
    expected_top = set(top_level_files) | {"records"}
    actual_top = {path.name for path in root.iterdir()}
    if actual_top != expected_top:
        raise RuntimeError(
            f"unlisted fixture root entries: expected {sorted(expected_top)}, "
            f"got {sorted(actual_top)}"
        )
    records = root / "records"
    if records.is_symlink() or not records.is_dir():
        raise RuntimeError("fixture records entry is not a real directory")
    expected_records = {f"{opaque_id}.npz" for opaque_id in record_ids}
    actual_records = {path.name for path in records.iterdir()}
    if actual_records != expected_records:
        raise RuntimeError("unlisted, missing, or duplicate fixture record entries")
    root_device = root.stat().st_dev
    for path in records.iterdir():
        _require_regular_unlinked_file(path, label="fixture record")
        if path.stat().st_dev != root_device:
            raise RuntimeError("fixture record crosses a mount boundary")
    for name in top_level_files:
        path = root / name
        _require_regular_unlinked_file(path, label="fixture control file")
        if path.stat().st_dev != root_device:
            raise RuntimeError("fixture control file crosses a mount boundary")


def _verify_protocol_before_pixels(
    config_path: Path, repo_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = config_path.resolve(strict=True)
    _require_regular_unlinked_file(config_path, label="protocol config")
    config = _load_json(config_path)
    if config.get("kind") != EXPECTED_KIND or config.get("schema_version") != 1:
        raise RuntimeError("wrong graph-oracle protocol config")
    contract = config.get("frozen_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("protocol has no frozen_contract")
    actual_contract_hash = _canonical_object_sha256(contract)
    if actual_contract_hash != config.get("frozen_contract_sha256"):
        raise RuntimeError("frozen_contract hash mismatch")
    protocol_instance_id = config.get(
        "protocol_instance_id", contract.get("protocol_instance_id")
    )
    if not isinstance(protocol_instance_id, str) or not OPAQUE_ID_RE.fullmatch(
        protocol_instance_id
    ):
        raise RuntimeError("protocol_instance_id must be 32 lowercase hex characters")
    if contract.get("protocol_instance_id") not in (None, protocol_instance_id):
        raise RuntimeError("protocol instance id drift")

    pins = config.get("runtime_pins")
    mutation_policy = config.get("runtime_pin_mutation_policy")
    if not isinstance(pins, dict):
        raise RuntimeError("protocol has no runtime_pins")
    if not isinstance(mutation_policy, dict):
        raise RuntimeError("protocol has no runtime pin mutation policy")
    code_pin_fields = mutation_policy.get("code_pin_fields")
    if not isinstance(code_pin_fields, list) or not code_pin_fields:
        raise RuntimeError("protocol has no code pin field closure")
    code_bindings: dict[str, str] = {}
    for pair in code_pin_fields:
        if not isinstance(pair, dict) or set(pair) != {"path_field", "sha256_field"}:
            raise RuntimeError("invalid code pin pair schema")
        path_field = str(pair["path_field"])
        sha_field = str(pair["sha256_field"])
        configured_path = pins.get(path_field)
        configured_sha = pins.get(sha_field)
        if not configured_path or not configured_sha:
            raise RuntimeError(f"runtime pin must be set before pixel access: {sha_field}")
        pinned_path = _resolve_repo_path(repo_root, str(configured_path))
        _verify_hash(pinned_path, str(configured_sha), label=sha_field)
        code_bindings[sha_field] = str(configured_sha)
    for field in (
        "fixture_input_manifest_sha256",
        "fixture_label_manifest_sha256",
        "fixture_lock_sha256",
    ):
        if pins.get(field) is not None:
            raise RuntimeError(f"fixture pin must still be null during preparation: {field}")

    environment_path_value = pins.get("environment_lock_path")
    environment_sha256 = pins.get("environment_lock_sha256")
    if not environment_path_value or not environment_sha256:
        raise RuntimeError("environment lock must be pinned before pixel access")
    environment_path = _resolve_repo_path(repo_root, str(environment_path_value))
    _verify_hash(environment_path, str(environment_sha256), label="environment lock")
    environment = _load_json(environment_path)
    expected_environment = environment.get("fixture_preparation_and_phase_b")
    if not isinstance(expected_environment, dict):
        raise RuntimeError("environment lock has no local prep/Phase-B contract")
    import cv2
    import kornia
    import scipy
    import skimage
    import torch

    actual_packages = {
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": Image.__version__ if hasattr(Image, "__version__") else None,
        "kornia": kornia.__version__,
        "scikit_image": skimage.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
    }
    # Pillow exposes its package version on PIL, not consistently on Image.
    from PIL import __version__ as pillow_version

    actual_packages["pillow"] = pillow_version
    runtime_environment = contract.get("runtime_environment", {}).get(
        "fixture_preparation_and_phase_b", {}
    )
    expected_prefix = runtime_environment.get("environment")
    if not isinstance(expected_prefix, str) or Path(sys.prefix).resolve() != Path(
        expected_prefix
    ).resolve():
        raise RuntimeError("fixture preparation is not using the pinned repo environment")
    if expected_environment.get("python") != platform_module.python_version():
        raise RuntimeError("fixture environment Python version mismatch")
    if expected_environment.get("platform") != platform_module.platform():
        raise RuntimeError("fixture environment platform mismatch")
    if expected_environment.get("packages") != actual_packages:
        raise RuntimeError(
            f"fixture package lock mismatch: expected {expected_environment.get('packages')}, "
            f"got {actual_packages}"
        )

    assets = contract.get("assets")
    if not isinstance(assets, dict):
        raise RuntimeError("protocol assets are missing")
    for asset_name in ("denoiser", "hbt"):
        asset = assets.get(asset_name)
        if not isinstance(asset, dict):
            raise RuntimeError(f"missing asset contract: {asset_name}")
        path = _resolve_repo_path(repo_root, asset["path"])
        _verify_hash(path, asset["sha256"], label=asset_name)
    code_pins = assets.get("known_code_sha256")
    if not isinstance(code_pins, dict) or not code_pins:
        raise RuntimeError("known code pins are missing")
    for relative, expected in sorted(code_pins.items()):
        path = _resolve_repo_path(repo_root, relative)
        _verify_hash(path, expected, label=f"code {relative}")

    selection = contract.get("source_selection")
    if not isinstance(selection, dict):
        raise RuntimeError("source selection is missing")
    for path_field, hash_field in (
        ("authoritative_manifest", "authoritative_manifest_sha256"),
        ("quarantine", "quarantine_sha256"),
    ):
        path = _resolve_repo_path(repo_root, selection[path_field])
        _verify_hash(path, selection[hash_field], label=path_field)
    sealed = contract.get("sealed_sets")
    if not isinstance(sealed, dict):
        raise RuntimeError("sealed set contract is missing")
    ledger = _resolve_repo_path(repo_root, sealed["audit_exclusion_ledger"])
    _verify_hash(ledger, sealed["audit_exclusion_ledger_sha256"], label="audit ledger")

    decision = config.get("decision_basis")
    if not isinstance(decision, dict):
        raise RuntimeError("decision basis is missing")
    for path_field, hash_field in (
        ("qap_confirmation_config", "qap_confirmation_config_sha256"),
        ("qap_confirmation_report", "qap_confirmation_report_sha256"),
    ):
        path = _resolve_repo_path(repo_root, decision[path_field])
        _verify_hash(path, decision[hash_field], label=path_field)

    private_contract = assets.get("private_function_contract")
    if not isinstance(private_contract, dict):
        raise RuntimeError("private function contract is missing")
    private_module = _resolve_repo_path(repo_root, private_contract["module"])
    _verify_hash(private_module, private_contract["module_sha256"], label="components module")
    import puzzle_assembly.components as components

    for symbol in private_contract.get("required_symbols", []):
        if not hasattr(components, str(symbol)):
            raise RuntimeError(f"required private symbol is missing: {symbol}")

    current = {
        "protocol_instance_id": protocol_instance_id,
        "frozen_contract_sha256": actual_contract_hash,
        **code_bindings,
    }
    expected_binding_fields = set(
        contract["fixture_preparation"]["exact_common_manifest_binding_field_names"]
    )
    if set(current) != expected_binding_fields:
        raise RuntimeError(
            f"fixture common binding closure drift: {sorted(set(current))} != "
            f"{sorted(expected_binding_fields)}"
        )
    return config, current


def _source_names(config: Mapping[str, Any], repo_root: Path) -> list[str]:
    selection = config["frozen_contract"]["source_selection"]
    all_names = source_names_for_split(
        selection["split"],
        manifest_path=repo_root / selection["authoritative_manifest"],
        quarantine_path=repo_root / selection["quarantine"],
        audit_exclusion_path=repo_root
        / config["frozen_contract"]["sealed_sets"]["audit_exclusion_ledger"],
    )
    offset = int(selection["offset"])
    count = int(selection["count"])
    names = all_names[offset : offset + count]
    digest = _sha256_bytes("\n".join(names).encode("utf-8"))
    if len(names) != selection["source_count_must_equal"]:
        raise RuntimeError("source count drift")
    if digest != selection["source_names_sha256"]:
        raise RuntimeError("source name hash drift")
    return names


def _publish_root(staging: Path, final: Path) -> None:
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to replace fixture root: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, final)
    _fsync_directory(final.parent)


def prepare_fixtures(
    *,
    config_path: Path,
    data_root: Path,
    input_root: Path,
    label_root: Path,
    lock_path: Path,
    marker_path: Path,
    lifecycle_ledger_root: Path,
    repo_root: Path = REPO_ROOT,
    image_loader: Callable[[Path], np.ndarray] = _load_rgb_target,
    panel_builder: Callable[..., ExactPanel] = make_exact_panel,
    master_secret: bytes | None = None,
    executing_builder_path: Path | None = None,
) -> dict[str, Any]:
    """Build and publish one frozen two-panel fixture set.

    ``image_loader``, ``panel_builder`` and ``master_secret`` exist for
    adversarial unit tests.  The CLI never accepts a caller-supplied secret.
    """

    repo_root = repo_root.expanduser().resolve(strict=True)
    config_path = config_path.expanduser().resolve(strict=True)
    data_root = data_root.expanduser().resolve(strict=True)
    input_root = input_root.expanduser().absolute()
    label_root = label_root.expanduser().absolute()
    lock_path = lock_path.expanduser().absolute()
    marker_path = marker_path.expanduser().absolute()
    lifecycle_ledger_root = lifecycle_ledger_root.expanduser().absolute()

    config, provenance = _verify_protocol_before_pixels(config_path, repo_root)
    fixture_bundle_root = _assert_exact_fixture_output_paths(
        config=config,
        repo_root=repo_root,
        input_root=input_root,
        label_root=label_root,
        lock_path=lock_path,
        marker_path=marker_path,
    )
    _assert_fresh_outputs(input_root, label_root, lock_path, marker_path)
    for first, second, label in (
        (fixture_bundle_root, data_root, "fixture bundle and source-data root"),
        (data_root, fixture_bundle_root, "source-data root and fixture bundle"),
        (
            fixture_bundle_root,
            lifecycle_ledger_root,
            "fixture bundle and lifecycle ledger",
        ),
        (
            lifecycle_ledger_root,
            fixture_bundle_root,
            "lifecycle ledger and fixture bundle",
        ),
    ):
        try:
            second.relative_to(first)
        except ValueError:
            pass
        else:
            raise RuntimeError(f"{label} may not contain one another")
    names = _source_names(config, repo_root)
    selection = config["frozen_contract"]["source_selection"]
    panels = tuple(selection["panels_in_label_order"])
    if panels != ("primary_kornia", "independent_libjpeg"):
        raise RuntimeError("panel contract drift")
    if int(selection["total_fixture_records"]) != len(names) * len(panels):
        raise RuntimeError("fixture record count contract drift")

    builder_path = (
        Path(__file__).resolve(strict=True)
        if executing_builder_path is None
        else executing_builder_path.expanduser().resolve(strict=True)
    )
    try:
        builder_relative = builder_path.relative_to(repo_root).as_posix()
    except ValueError as error:
        raise RuntimeError("fixture builder path must be contained by repo root") from error
    _require_regular_unlinked_file(builder_path, label="fixture builder")
    builder_sha256 = _sha256_file(builder_path)
    runtime_pins = config["runtime_pins"]
    if builder_relative != runtime_pins.get("fixture_builder_path"):
        raise RuntimeError("executing fixture builder path differs from runtime pin")
    if builder_sha256 != runtime_pins.get("fixture_builder_sha256"):
        raise RuntimeError("executing fixture builder SHA256 differs from runtime pin")
    config_sha256_before_pixels = _sha256_file(config_path)

    for fixture_location in (
        input_root,
        label_root,
        lock_path.parent,
    ):
        try:
            lifecycle_ledger_root.relative_to(fixture_location)
        except ValueError:
            pass
        else:
            raise RuntimeError("lifecycle ledger may not be inside a fixture root")
        try:
            fixture_location.relative_to(lifecycle_ledger_root)
        except ValueError:
            pass
        else:
            raise RuntimeError("fixture roots may not be inside the lifecycle ledger")
    prep_claim = advance_state(
        config_path=config_path,
        ledger_root=lifecycle_ledger_root,
        state="PREP",
        expected_config_sha256=config_sha256_before_pixels,
    )

    marker_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": PREP_MARKER_KIND,
        "started_utc": _utc_now(),
        **provenance,
        "builder_path": builder_relative,
        "builder_sha256": builder_sha256,
        "config_sha256_before_pixels": config_sha256_before_pixels,
        "source_names_sha256": selection["source_names_sha256"],
        "source_count": len(names),
        "expected_fixture_records": len(names) * len(panels),
        "prep_lifecycle_sha256": prep_claim["state_sha256"],
    }
    _atomic_json(marker_path, marker_payload)

    # Construct the first clean-target path only after both irreversible PREP
    # and the durable fixture pixel-access marker exist.
    target_root = data_root / "train" / "targets"
    if target_root.is_symlink() or not target_root.is_dir():
        raise RuntimeError(f"clean target root is unavailable: {target_root}")
    _assert_no_symlink_ancestors(target_root)

    secret = secrets.token_bytes(32) if master_secret is None else bytes(master_secret)
    if len(secret) != 32:
        raise RuntimeError("fixture master secret must contain exactly 32 bytes")

    token = uuid.uuid4().hex
    input_staging = input_root.with_name(f".{input_root.name}.{token}.staging")
    label_staging = label_root.with_name(f".{label_root.name}.{token}.staging")
    for staging in (input_staging, label_staging):
        if staging.exists() or staging.is_symlink():
            raise FileExistsError(f"staging path already exists: {staging}")
        (staging / "records").mkdir(parents=True, exist_ok=False)
        _fsync_directory(staging)
    os.chmod(label_staging, 0o700)
    os.chmod(label_staging / "records", 0o700)

    _exclusive_secret(label_staging / SECRET_NAME, secret)
    input_records: list[dict[str, Any]] = []
    label_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    panel_counts: Counter[str] = Counter()

    for source in names:
        target_path = target_root / source
        clean_target = np.asarray(image_loader(target_path))
        if clean_target.dtype != np.uint8 or clean_target.shape != EXPECTED_TARGET_SHAPE:
            raise RuntimeError(f"invalid clean target returned for {source}")
        clean_target = np.ascontiguousarray(clean_target)
        target_file_sha256 = _sha256_file(target_path)
        for panel in panels:
            panel_seed = per_source_seed(
                int(config["frozen_contract"]["synthetic_corruption"]["master_seed"]),
                f"candidate-graph-oracle-{panel}",
                source,
                0,
            )
            exact = panel_builder(clean_target, panel=panel, seed=panel_seed)
            exact_tiles = np.asarray(exact.slot_tiles)
            exact_truth = validate_permutation(
                np.asarray(exact.slot_to_target), name="exact slot_to_target"
            )
            if exact_tiles.dtype != np.uint8 or exact_tiles.shape != EXPECTED_TILE_SHAPE:
                raise RuntimeError("exact panel returned invalid slot tiles")

            opaque_id = _opaque_id(secret, source, panel)
            if not OPAQUE_ID_RE.fullmatch(opaque_id) or opaque_id in seen_ids:
                raise RuntimeError("opaque fixture id collision or format failure")
            seen_ids.add(opaque_id)
            permutation = validate_permutation(
                _opaque_permutation(secret, source, panel),
                name="opaque slot permutation",
            ).astype(np.int32, copy=False)
            opaque_tiles = np.ascontiguousarray(exact_tiles[permutation])
            composed_truth = validate_permutation(
                exact_truth[permutation], name="composed slot_to_target"
            ).astype(np.int32, copy=False)
            qap_seed = _qap_seed(opaque_id)

            input_path = input_staging / "records" / f"{opaque_id}.npz"
            _atomic_bytes(
                input_path,
                _npz_bytes(
                    slot_tiles=opaque_tiles,
                    qap_seed=np.asarray(qap_seed, dtype=np.uint64),
                ),
            )
            input_records.append(
                {
                    "opaque_id": opaque_id,
                    "artifact": _artifact_descriptor(input_path, input_staging),
                    "arrays": {
                        "slot_tiles": _array_descriptor(
                            opaque_tiles, "opaque corrupted input slot tiles"
                        ),
                        "qap_seed": _array_descriptor(
                            np.asarray(qap_seed, dtype=np.uint64),
                            "fixed opaque nuisance QAP seed",
                        ),
                    },
                }
            )

            label_path = label_staging / "records" / f"{opaque_id}.npz"
            _atomic_bytes(
                label_path,
                _npz_bytes(
                    opaque_slot_permutation=permutation,
                    composed_slot_to_target=composed_truth,
                    clean_target_rgb=clean_target,
                ),
                mode=0o600,
            )
            label_records.append(
                {
                    "opaque_id": opaque_id,
                    "source_name": source,
                    "panel": panel,
                    "panel_seed": panel_seed,
                    "target_file_sha256": target_file_sha256,
                    "artifact": _artifact_descriptor(label_path, label_staging),
                    "arrays": {
                        "opaque_slot_permutation": _array_descriptor(
                            permutation, "secret second-stage slot permutation"
                        ),
                        "composed_slot_to_target": _array_descriptor(
                            composed_truth, "truth mapping after opaque slot permutation"
                        ),
                        "clean_target_rgb": _array_descriptor(
                            clean_target, "clean RGB target"
                        ),
                    },
                }
            )
            panel_counts[panel] += 1

    expected_records = int(selection["total_fixture_records"])
    if len(input_records) != expected_records or len(label_records) != expected_records:
        raise RuntimeError("fixture coverage is incomplete")
    if panel_counts != Counter({panel: len(names) for panel in panels}):
        raise RuntimeError("hidden panel coverage is incomplete")

    input_records.sort(key=lambda record: record["opaque_id"])
    label_records.sort(key=lambda record: record["opaque_id"])
    input_ids = [str(record["opaque_id"]) for record in input_records]
    label_ids = [str(record["opaque_id"]) for record in label_records]
    if input_ids != label_ids or len(set(input_ids)) != expected_records:
        raise RuntimeError("opaque input/label bijection failed")
    nuisance_seeds = [_qap_seed(opaque_id) for opaque_id in input_ids]
    if len(set(nuisance_seeds)) != expected_records:
        raise RuntimeError("opaque nuisance seed collision")
    ids_sha256 = _sha256_bytes("\n".join(input_ids).encode("ascii"))

    common_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        **provenance,
        "record_count": expected_records,
        "opaque_ids_sha256": ids_sha256,
        "canonical_record_order": "ascending opaque_id",
    }
    input_manifest = {
        **common_manifest,
        "kind": INPUT_KIND,
        "allowed_record_metadata": ["opaque_id", "artifact", "arrays"],
        "records": input_records,
    }
    input_manifest_path = input_staging / INPUT_MANIFEST_NAME
    _atomic_json(input_manifest_path, input_manifest)
    input_manifest_sha256 = _sha256_file(input_manifest_path)

    secret_path = label_staging / SECRET_NAME
    label_manifest = {
        **common_manifest,
        "kind": LABEL_KIND,
        "fixture_input_manifest_sha256": input_manifest_sha256,
        "hidden_panel_counts": dict(sorted(panel_counts.items())),
        "master_secret": {
            **_artifact_descriptor(secret_path, label_staging),
            "mode": "0600",
        },
        "records": label_records,
    }
    label_manifest_path = label_staging / LABEL_MANIFEST_NAME
    _atomic_json(label_manifest_path, label_manifest, mode=0o600)

    expected_id_set = set(input_ids)
    _assert_exact_fixture_tree(
        input_staging,
        top_level_files={INPUT_MANIFEST_NAME},
        record_ids=expected_id_set,
    )
    _assert_exact_fixture_tree(
        label_staging,
        top_level_files={LABEL_MANIFEST_NAME, SECRET_NAME},
        record_ids=expected_id_set,
    )

    if _sha256_file(config_path) != config_sha256_before_pixels:
        raise RuntimeError("protocol config changed during fixture preparation")
    if _sha256_file(builder_path) != builder_sha256:
        raise RuntimeError("fixture builder changed during fixture preparation")
    _verify_protocol_before_pixels(config_path, repo_root)

    label_manifest_sha256 = _sha256_file(label_manifest_path)
    lock = {
        "schema_version": SCHEMA_VERSION,
        "kind": LOCK_KIND,
        "created_utc": _utc_now(),
        **provenance,
        "prep_marker_sha256": _sha256_file(marker_path),
        "prep_lifecycle_sha256": prep_claim["state_sha256"],
        "fixture_input_manifest_sha256": input_manifest_sha256,
        "fixture_label_manifest_sha256": label_manifest_sha256,
        "record_count": expected_records,
        "opaque_ids_sha256": ids_sha256,
        "input_and_label_roots_are_distinct_siblings": input_root.parent
        == label_root.parent,
        "phase_a_may_receive_label_root": False,
        "phase_a_may_receive_master_secret": False,
    }

    _publish_root(input_staging, input_root)
    _publish_root(label_staging, label_root)
    published_input_manifest = input_root / INPUT_MANIFEST_NAME
    published_label_manifest = label_root / LABEL_MANIFEST_NAME
    if _sha256_file(published_input_manifest) != input_manifest_sha256:
        raise RuntimeError("published input manifest changed")
    if _sha256_file(published_label_manifest) != label_manifest_sha256:
        raise RuntimeError("published label manifest changed")
    _assert_exact_fixture_tree(
        input_root,
        top_level_files={INPUT_MANIFEST_NAME},
        record_ids=expected_id_set,
    )
    _assert_exact_fixture_tree(
        label_root,
        top_level_files={LABEL_MANIFEST_NAME, SECRET_NAME},
        record_ids=expected_id_set,
    )
    _atomic_json(lock_path, lock)
    control_entries = {path.name for path in lock_path.parent.iterdir()}
    expected_control_entries = {lock_path.name, marker_path.name}
    if control_entries != expected_control_entries:
        raise RuntimeError("fixture control root contains missing or unlisted entries")
    for control_file in (lock_path, marker_path):
        _require_regular_unlinked_file(control_file, label="fixture control artifact")

    return {
        "status": "fixtures_prepared_runtime_pins_required",
        "safe_for_submission": False,
        "input_root": str(input_root),
        "label_root": str(label_root),
        "lock_path": str(lock_path),
        "prep_marker_path": str(marker_path),
        "fixture_input_manifest_sha256": input_manifest_sha256,
        "fixture_label_manifest_sha256": label_manifest_sha256,
        "fixture_lock_sha256": _sha256_file(lock_path),
        "record_count": expected_records,
        "opaque_ids_sha256": ids_sha256,
        "next_action": "pin the three fixture hashes in runtime_pins before Phase A",
        "prep_lifecycle_sha256": prep_claim["state_sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/candidate_graph_oracle_ceiling_v3.json",
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "puzzle")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--prep-marker-path", type=Path)
    parser.add_argument("--lifecycle-ledger-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    marker = args.prep_marker_path
    if marker is None:
        marker = args.lock_path.with_name("FIXTURE_PIXEL_ACCESS_STARTED.json")
    summary = prepare_fixtures(
        config_path=args.config,
        data_root=args.data_root,
        input_root=args.input_root,
        label_root=args.label_root,
        lock_path=args.lock_path,
        marker_path=marker,
        lifecycle_ledger_root=args.lifecycle_ledger_root,
    )
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
