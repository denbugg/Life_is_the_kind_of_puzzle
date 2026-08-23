"""Crash-safe production inference for E24 CRS-v1.

This entry point is intentionally *downstream* of the frozen E24 experiment.
It refuses to inspect a test image until all of the following are hash-linked
and valid: the PASS structural report, its PASS orchestration receipt, the
frozen preflight ledger/canary, and a completed final all-eight model manifest.

Generic images are processed one at a time.  The 227-column relation table is
kept only in memory for that image and is released before the next image.  The
18 verified clean-source overrides are copied byte-for-byte from the immutable
Rank96 baseline bundle.  The baseline ZIP is never a write target and remains
the fallback if this fail-closed route does not complete.

Examples::

    python src/infer_e24.py --dry-run
    python src/infer_e24.py --device cuda --resume
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import os
import sys
import tempfile
import time
import zipfile
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PRODUCTION_OUTPUT_ROOT = Path("E:/pazzle_work/submissions/e24_crs_v1")
RUNTIME_ROOT = PRODUCTION_OUTPUT_ROOT / "runtime_v1"
for _key in ("TEMP", "TMP", "TMPDIR", "JOBLIB_TEMP_FOLDER", "LIGHTGBM_TMPDIR"):
    os.environ[_key] = str(RUNTIME_ROOT / "tmp")
if sys.pycache_prefix is None or Path(sys.pycache_prefix).drive.upper() != "E:":
    sys.pycache_prefix = str(RUNTIME_ROOT / "pycache")

import numpy as np

import e23_i21_residual_candidate_oracle as e23_core
import e24_context_relation_selector as selector
import infer_rank96 as rank96


GRID = 24
NUM_TILES = 576
RAW_WIDTH = 128
FEATURE_COUNT = 227
DEFAULT_EXPECTED_COUNT = 700
DEFAULT_PAIR_BATCH = 4096
DEFAULT_SEED = 20_260_806
INCOMPLETE_EXIT_CODE = 75

STRUCTURAL_SCHEMA = "pazzle-e24-crs-v1-structural-oof-report-v1"
ORCHESTRATION_SCHEMA = "pazzle-e24-crs-v1-orchestration-receipt-v1"
LEDGER_SCHEMA = "pazzle-e24-crs-v1-runner-preflight-v1"
CANARY_SCHEMA = "pazzle-e24-crs-v1-feature-canary-gate-v1"
MANIFEST_SCHEMA = "pazzle-e24-crs-v1-production-manifest-v1"
REPORT_SCHEMA = "pazzle-e24-crs-v1-production-report-v1"
PRODUCTION_CANARY_SCHEMA = "pazzle-e24-crs-v1-production-scene17-parity-v1"

BASELINE_ZIP_SIZE = 222_050_278
BASELINE_ZIP_SHA256 = "9a2eaf962507d11f2cad0caf59af40fe9755a6f092051c9d144a5f6aca10965f"
BASELINE_MANIFEST_SHA256 = "66031c750a77b43b69bc4ce8ef44fb06bebac87951c47d7c42caa1e6dcac13cc"
BASELINE_OVERRIDE_COUNT = 18

PRODUCTION_CONTRACT: dict[str, Any] = {
    "schema": "pazzle-e24-crs-v1-production-v1",
    "grid": 24,
    "tile_size": 20,
    "image_size": 480,
    "num_tiles": 576,
    "orientation": "upright_fixed_no_rotation_or_reflection",
    "rank96_candidates": "dual_encoder_ordered_union_top64_each_raw_ranker_logits",
    "spatial_logits": "frozen_i21_cpu_float32_edge_head",
    "candidate_pool": "e23_i21_residual_k64_complete_geometry_valid",
    "spatial_pair_guard": "frozen_e23_max_100000_before_combined_pool",
    "features": "e24_crs_v1_exact_227_columns_in_memory_one_image_only",
    "selector": "final_all8_lightgbm_256_trees",
    "decoder": "strict_positive_none_margin_2C_minus_2_rollback_potential_dsu",
    "packer": "solve_buddies.solve_components_from_scores",
    "repair_passes": 0,
    "restarts": 1,
    "packer_seed": 1234,
    "restoration": "opencv_fast_nlm_colored",
    "nlm_h": 10,
    "nlm_h_color": 10,
    "nlm_template_window": 7,
    "nlm_search_window": 21,
    "verified_overrides": "exact_18_members_from_immutable_rank96_baseline",
    "generic_failure": "abort_without_new_final_zip_keep_baseline",
}


class E24InferenceError(RuntimeError):
    """A production authority, integrity, or inference invariant failed."""


class IncompleteRun(E24InferenceError):
    """A declared safe stop wrote resumable state."""


@dataclass(frozen=True)
class InferenceConfig:
    input_dir: Path
    output_dir: Path
    output_zip: Path
    baseline_zip: Path
    baseline_manifest: Path
    ledger: Path
    canary_gate: Path
    structural_report: Path
    orchestration_receipt: Path
    ranker_checkpoint: Path
    affinity_primary_checkpoint: Path
    affinity_secondary_checkpoint: Path
    i21_checkpoint: Path
    production_root: Path = PRODUCTION_OUTPUT_ROOT
    device: str = "auto"
    pair_batch: int = DEFAULT_PAIR_BATCH
    seed: int = DEFAULT_SEED
    expected_count: int = DEFAULT_EXPECTED_COUNT
    limit: int = 0
    max_runtime_seconds: float = 0.0
    resume: bool = False
    dry_run: bool = False
    manifest_path: Path | None = None
    report_path: Path | None = None


@dataclass(frozen=True)
class Authority:
    ledger_sha256: str
    run_contract_sha256: str
    structural_report_sha256: str
    orchestration_receipt_sha256: str
    canary_gate_sha256: str
    final_model_manifest_sha256: str
    ordered_feature_names_sha256: str
    final_model_path: Path
    final_model_sha256: str


@dataclass(frozen=True)
class FrozenOOFAuthority:
    ledger_sha256: str
    run_contract_sha256: str
    structural_report_sha256: str
    orchestration_receipt_sha256: str
    canary_gate_sha256: str
    ordered_feature_names_sha256: str


@dataclass(frozen=True)
class BaselineBundle:
    zip_path: Path
    zip_sha256: str
    manifest_sha256: str
    names: tuple[str, ...]
    overrides: Mapping[str, str]
    rank96_code: Mapping[str, str]
    rank96_checkpoints: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class LoadedModels:
    rank96_models: Any
    i21_model: Any
    selector_model: Any


@dataclass(frozen=True)
class InferredImage:
    output: np.ndarray
    board: np.ndarray
    objective: float
    candidate_ids_sha256: str
    raw_scores_sha256: str
    spatial_logits_sha256: str
    relation_rows: int
    relation_queries: int
    proposed_relations: int
    accepted_relations: int
    tree_merges: int
    cycle_acceptances: int


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _e24_canonical_json_bytes(value: Any) -> bytes:
    """Exact compact ASCII encoding used by the frozen E24 evaluator/runner."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise E24InferenceError("value is not canonical finite E24 JSON") from exc
    return (encoded + "\n").encode("ascii")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _e24_array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    target = _require_e_write_path(path, label="atomic output")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, _canonical_json_bytes(value))


def _load_canonical_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise E24InferenceError(f"{label} is missing or unreadable: {path}") from exc
    if type(value) is not dict or raw != _canonical_json_bytes(value):
        raise E24InferenceError(f"{label} must be canonical JSON: {path}")
    return value


def _load_e24_canonical_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except Exception as exc:
        raise E24InferenceError(f"{label} is missing or unreadable: {path}") from exc
    if type(value) is not dict or raw != _e24_canonical_json_bytes(value):
        raise E24InferenceError(f"{label} must use frozen compact E24 JSON: {path}")
    return value


def _require_e_write_path(path: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=False)
    if resolved.drive.upper() != "E:" or resolved == Path("E:/").resolve(strict=False):
        raise E24InferenceError(f"{label} must be a non-root path on E:, got {resolved}")
    return resolved


def _is_within(path: Path, root: Path, *, allow_equal: bool = False) -> bool:
    path_value = path.resolve(strict=False)
    root_value = root.resolve(strict=False)
    try:
        contained = os.path.commonpath((str(path_value), str(root_value))) == str(root_value)
    except ValueError:
        return False
    return contained and (allow_equal or path_value != root_value)


def _paths_overlap(first: Path, second: Path) -> bool:
    return _is_within(first, second, allow_equal=True) or _is_within(
        second, first, allow_equal=True
    )


def _safe_basename(name: str) -> str:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise E24InferenceError(f"image name is not a safe basename: {name!r}")
    return name


def _verify_file_record(record: object, *, label: str) -> Path:
    if type(record) is not dict or set(record) != {"path", "bytes", "sha256"}:
        raise E24InferenceError(f"{label} record field set drifted")
    path = Path(record["path"]).resolve()
    if (
        not path.is_file()
        or type(record["bytes"]) is not int
        or record["bytes"] < 1
        or path.stat().st_size != record["bytes"]
        or not _is_sha256(record["sha256"])
        or sha256_file(path) != record["sha256"]
    ):
        raise E24InferenceError(f"{label} file provenance mismatch")
    return path


def authenticate_frozen_oof_authority(config: InferenceConfig) -> FrozenOOFAuthority:
    """Verify the currently frozen structural PASS/orchestration chain."""

    ledger_path = config.ledger.resolve()
    ledger = _load_e24_canonical_json(ledger_path, label="E24 preflight ledger")
    ledger_sha = sha256_file(ledger_path)
    if (
        ledger.get("schema") != LEDGER_SCHEMA
        or ledger.get("status") != "frozen_preflight_only"
        or ledger.get("metrics_opened") is not False
        or ledger.get("target_artifacts_created") is not False
        or ledger.get("staged_board_ssim_nlm") != "sealed"
        or ledger.get("e25", {}).get("opened") is not False
        or not _is_sha256(ledger.get("run_contract_sha256"))
        or ledger.get("ordered_feature_names") != list(selector.FEATURE_NAMES)
        or len(selector.FEATURE_NAMES) != FEATURE_COUNT
        or ledger.get("core_protocol") != json.loads(
            _e24_canonical_json_bytes(selector.PROTOCOL).decode("ascii")
        )
    ):
        raise E24InferenceError("E24 preflight ledger contract drifted")
    feature_digest = hashlib.sha256(
        _e24_canonical_json_bytes({"feature_names": list(selector.FEATURE_NAMES)})
    ).hexdigest()
    if ledger.get("ordered_feature_names_sha256") != feature_digest:
        raise E24InferenceError("E24 feature schema digest drifted")
    runtime_versions = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "torch": importlib.metadata.version("torch"),
        "scikit-image": importlib.metadata.version("scikit-image"),
        "scipy": importlib.metadata.version("scipy"),
        "opencv-python": importlib.metadata.version("opencv-python"),
        "Pillow": importlib.metadata.version("Pillow"),
        "lightgbm": importlib.metadata.version("lightgbm"),
    }
    if ledger.get("runtime_versions") != runtime_versions:
        raise E24InferenceError("E24 production package/runtime versions drifted")
    sources = ledger.get("sources")
    if type(sources) is not dict or not sources:
        raise E24InferenceError("E24 ledger has no frozen source map")
    for path_text, expected in sources.items():
        if type(path_text) is not str or not _is_sha256(expected):
            raise E24InferenceError("E24 source map is malformed")
        path = Path(path_text).resolve()
        if not path.is_file() or sha256_file(path) != expected:
            raise E24InferenceError(f"frozen E24 source drifted: {path}")

    canary_path = config.canary_gate.resolve()
    canary = _load_e24_canonical_json(canary_path, label="E24 feature canary gate")
    canary_sha = sha256_file(canary_path)
    if (
        set(canary)
        != {
            "schema",
            "status",
            "passed",
            "ledger_sha256",
            "run_contract_sha256",
            "receipt_sha256",
            "thresholds",
            "observed",
            "checks",
            "labels_or_metrics_opened",
        }
        or canary.get("schema") != CANARY_SCHEMA
        or canary.get("status") != "pass"
        or canary.get("passed") is not True
        or canary.get("ledger_sha256") != ledger_sha
        or canary.get("run_contract_sha256") != ledger["run_contract_sha256"]
        or type(canary.get("checks")) is not dict
        or not canary["checks"]
        or not all(value is True for value in canary["checks"].values())
        or canary.get("labels_or_metrics_opened") is not False
    ):
        raise E24InferenceError("E24 feature canary is not an authenticated PASS")

    structural_path = config.structural_report.resolve()
    structural = _load_e24_canonical_json(structural_path, label="E24 structural report")
    structural_sha = sha256_file(structural_path)
    decision = structural.get("decision")
    if (
        set(structural)
        != {
            "schema",
            "status",
            "stage",
            "ledger_sha256",
            "run_contract_sha256",
            "fold_commit_sha256",
            "rows",
            "summary",
            "decision",
            "staged_board_ssim_nlm",
            "e25_opened",
        }
        or structural.get("schema") != STRUCTURAL_SCHEMA
        or structural.get("status") != "complete"
        or structural.get("ledger_sha256") != ledger_sha
        or structural.get("run_contract_sha256") != ledger["run_contract_sha256"]
        or structural.get("e25_opened") is not False
        or type(decision) is not dict
        or decision.get("passed") is not True
        or decision.get("stage") != "go_staged_end_to_end"
        or structural.get("stage") != decision.get("stage")
        or structural.get("staged_board_ssim_nlm") != "sealed_not_run"
        or type(decision.get("checks")) is not dict
        or not decision["checks"]
        or not all(value is True for value in decision["checks"].values())
    ):
        raise E24InferenceError("E24 structural report is not an authenticated PASS")

    receipt_path = config.orchestration_receipt.resolve()
    receipt = _load_e24_canonical_json(receipt_path, label="E24 orchestration receipt")
    receipt_sha = sha256_file(receipt_path)
    if (
        set(receipt)
        != {
            "schema",
            "status",
            "ledger_sha256",
            "run_contract_sha256",
            "canary_gate_sha256",
            "structural_report_sha256",
            "resource",
            "checks",
        }
        or receipt.get("schema") != ORCHESTRATION_SCHEMA
        or receipt.get("status") != "pass"
        or receipt.get("ledger_sha256") != ledger_sha
        or receipt.get("run_contract_sha256") != ledger["run_contract_sha256"]
        or receipt.get("canary_gate_sha256") != canary_sha
        or receipt.get("structural_report_sha256") != structural_sha
        or type(receipt.get("checks")) is not dict
        or not receipt["checks"]
        or not all(value is True for value in receipt["checks"].values())
    ):
        raise E24InferenceError("E24 orchestration receipt does not authenticate the PASS report")

    return FrozenOOFAuthority(
        ledger_sha256=ledger_sha,
        run_contract_sha256=ledger["run_contract_sha256"],
        structural_report_sha256=structural_sha,
        orchestration_receipt_sha256=receipt_sha,
        canary_gate_sha256=canary_sha,
        ordered_feature_names_sha256=feature_digest,
    )


def authenticate_e24_authority(config: InferenceConfig) -> Authority:
    """Fail closed until downstream PASS writers have exact frozen contracts.

    The current OOF artifacts are fully verified first.  The staged board/NLM,
    final-all-eight fit and independent E25 writers do not yet have frozen
    schemas or paths.  Inventing a consumer-side manifest would create fake
    authority, so the real CLI remains sealed.  Once those writers freeze, this
    function is the single integration point; synthetic tests inject an exact
    mock :class:`Authority` without weakening the CLI path.
    """

    authenticate_frozen_oof_authority(config)
    raise E24InferenceError(
        "production remains sealed: exact staged PASS, final-all8 fit and E25 PASS "
        "writer contracts are not frozen"
    )


def _list_input_names(input_dir: Path) -> list[str]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    names = sorted(
        path.name
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )
    if not names:
        raise E24InferenceError(f"no PNG inputs found in {input_dir}")
    return [_safe_basename(name) for name in names]


def _baseline_override_map(manifest: Mapping[str, Any]) -> dict[str, str]:
    contract = manifest.get("contract")
    completed = manifest.get("completed")
    if type(contract) is not dict or type(completed) is not dict:
        raise E24InferenceError("baseline manifest has no contract/completed map")
    records = contract.get("overrides")
    if type(records) is not list or len(records) != BASELINE_OVERRIDE_COUNT:
        raise E24InferenceError("baseline override inventory drifted")
    output: dict[str, str] = {}
    for record in records:
        if type(record) is not dict or set(record) != {"name", "sha256"}:
            raise E24InferenceError("baseline override record is malformed")
        name = _safe_basename(record["name"])
        digest = record["sha256"]
        if name in output or not _is_sha256(digest):
            raise E24InferenceError("baseline override identity is malformed/duplicated")
        completed_record = completed.get(name)
        if (
            type(completed_record) is not dict
            or completed_record.get("source") != "verified_source_override"
            or completed_record.get("output_sha256") != digest
            or completed_record.get("override_sha256") != digest
        ):
            raise E24InferenceError("baseline completed override record drifted")
        output[name] = digest
    return dict(sorted(output.items()))


def authenticate_baseline(
    baseline_zip: Path,
    baseline_manifest: Path,
    input_names: Sequence[str] | None,
    *,
    expected_zip_size: int = BASELINE_ZIP_SIZE,
    expected_zip_sha256: str = BASELINE_ZIP_SHA256,
    expected_manifest_sha256: str = BASELINE_MANIFEST_SHA256,
) -> BaselineBundle:
    """Authenticate the immutable fallback and its exact 18 override members."""

    zip_path = baseline_zip.resolve()
    manifest_path = baseline_manifest.resolve()
    if (
        not zip_path.is_file()
        or zip_path.stat().st_size != expected_zip_size
        or sha256_file(zip_path) != expected_zip_sha256
    ):
        raise E24InferenceError("immutable Rank96 fallback ZIP identity drifted")
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_manifest_sha256:
        raise E24InferenceError("immutable Rank96 baseline manifest identity drifted")
    manifest = _load_canonical_json(manifest_path, label="Rank96 baseline manifest")
    contract = manifest.get("contract")
    if (
        manifest.get("schema") != rank96.MANIFEST_SCHEMA
        or manifest.get("status") != "completed"
        or type(contract) is not dict
        or contract.get("pipeline") != rank96.RANK96_CONTRACT
        or contract.get("code") != rank96._code_provenance()
        or type(contract.get("checkpoints")) is not dict
    ):
        raise E24InferenceError("Rank96 baseline manifest contract drifted")
    overrides = _baseline_override_map(manifest)
    contract_inputs = contract.get("inputs")
    if type(contract_inputs) is not list or not contract_inputs:
        raise E24InferenceError("baseline contract input inventory is missing")
    expected_names = tuple(
        sorted(
            _safe_basename(record["name"])
            for record in contract_inputs
            if type(record) is dict
            and set(record) == {"name", "sha256"}
            and _is_sha256(record["sha256"])
        )
    )
    if len(expected_names) != len(contract_inputs) or len(expected_names) != len(set(expected_names)):
        raise E24InferenceError("baseline contract input inventory is malformed")
    if input_names is not None and expected_names != tuple(
        sorted(_safe_basename(name) for name in input_names)
    ):
        raise E24InferenceError("baseline inventory differs from the target input names")
    with zipfile.ZipFile(zip_path, "r") as archive:
        infos = archive.infolist()
        names = tuple(info.filename for info in infos)
        if (
            names != expected_names
            or len(names) != len(set(names))
            or any(info.is_dir() or Path(info.filename).name != info.filename for info in infos)
        ):
            raise E24InferenceError("baseline ZIP members differ from the target input set")
        for name, expected in overrides.items():
            content = archive.read(name)
            if _sha256_bytes(content) != expected:
                raise E24InferenceError(f"baseline override member hash mismatch: {name}")
            rank96._validate_rgb_array(_decode_png_bytes(content, label=name), label=name)
    return BaselineBundle(
        zip_path=zip_path,
        zip_sha256=expected_zip_sha256,
        manifest_sha256=expected_manifest_sha256,
        names=expected_names,
        overrides=overrides,
        rank96_code=dict(contract["code"]),
        rank96_checkpoints=dict(contract["checkpoints"]),
    )


def _decode_png_bytes(content: bytes, *, label: str) -> np.ndarray:
    from PIL import Image

    try:
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            if image.format != "PNG" or image.mode != "RGB" or image.size != (480, 480):
                raise E24InferenceError(f"{label} is not a strict RGB 480x480 PNG")
            value = np.asarray(image, dtype=np.uint8)
    except E24InferenceError:
        raise
    except Exception as exc:
        raise E24InferenceError(f"cannot decode baseline PNG {label}") from exc
    return np.ascontiguousarray(value)


def _read_baseline_override(bundle: BaselineBundle, name: str) -> bytes:
    expected = bundle.overrides.get(name)
    if expected is None:
        raise E24InferenceError(f"{name} is not an authenticated baseline override")
    with zipfile.ZipFile(bundle.zip_path, "r") as archive:
        content = archive.read(_safe_basename(name))
    if _sha256_bytes(content) != expected:
        raise E24InferenceError(f"baseline override changed during inference: {name}")
    _decode_png_bytes(content, label=name)
    return content


def _input_inventory(config: InferenceConfig, names: Sequence[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for name in names:
        path = config.input_dir / name
        rank96.load_rgb_strict(path)
        records.append({"name": name, "sha256": sha256_file(path)})
    return records


def _configure_i21_runtime_and_checkpoint(path: Path) -> dict[str, Any]:
    import eval_e23_i21_residual_candidate_ceiling as e23_eval

    runtime = e23_eval._runtime_provenance()
    return {
        "checkpoint": dict(e23_eval._checkpoint_record(path)),
        "runtime": runtime,
    }


def _rank96_checkpoint_provenance(config: InferenceConfig) -> dict[str, dict[str, Any]]:
    rank_config = rank96.InferenceConfig(
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        output_zip=None,
        ranker_checkpoint=config.ranker_checkpoint,
        affinity_primary_checkpoint=config.affinity_primary_checkpoint,
        affinity_secondary_checkpoint=config.affinity_secondary_checkpoint,
        device=config.device,
        pair_batch=config.pair_batch,
        expected_count=config.expected_count,
    )
    return rank96._checkpoint_provenance(rank_config)


def _code_provenance() -> dict[str, str]:
    source = Path(__file__).resolve().parent
    paths = {
        "infer_e24.py": Path(__file__).resolve(),
        "infer_rank96.py": source / "infer_rank96.py",
        "e23_i21_residual_candidate_oracle.py": source / "e23_i21_residual_candidate_oracle.py",
        "eval_e23_i21_residual_candidate_ceiling.py": source / "eval_e23_i21_residual_candidate_ceiling.py",
        "e24_context_relation_selector.py": source / "e24_context_relation_selector.py",
        "eval_seeded_qap.py": source / "eval_seeded_qap.py",
        "solve_buddies.py": source / "solve_buddies.py",
    }
    if any(not path.is_file() for path in paths.values()):
        raise E24InferenceError("an E24 production code dependency is missing")
    output = {name: sha256_file(path) for name, path in sorted(paths.items())}
    output.update(
        {f"rank96/{name}": digest for name, digest in rank96._code_provenance().items()}
    )
    return dict(sorted(output.items()))


def _build_contract(
    config: InferenceConfig,
    *,
    authority: Authority,
    baseline: BaselineBundle,
    inputs: list[dict[str, str]],
    rank96_checkpoints: Mapping[str, Mapping[str, Any]],
    i21_checkpoint: Mapping[str, Any],
    code: Mapping[str, str],
    resolved_device: str,
    production_canary_sha256: str,
) -> dict[str, Any]:
    return {
        "pipeline": PRODUCTION_CONTRACT,
        "execution": {
            "device": resolved_device,
            "pair_batch": int(config.pair_batch),
            "seed": int(config.seed),
        },
        "authority": {
            "ledger_sha256": authority.ledger_sha256,
            "run_contract_sha256": authority.run_contract_sha256,
            "structural_report_sha256": authority.structural_report_sha256,
            "orchestration_receipt_sha256": authority.orchestration_receipt_sha256,
            "canary_gate_sha256": authority.canary_gate_sha256,
            "final_model_manifest_sha256": authority.final_model_manifest_sha256,
            "final_model_sha256": authority.final_model_sha256,
            "ordered_feature_names_sha256": authority.ordered_feature_names_sha256,
            "production_scene17_parity_sha256": production_canary_sha256,
        },
        "baseline": {
            "zip_sha256": baseline.zip_sha256,
            "manifest_sha256": baseline.manifest_sha256,
            "override_count": len(baseline.overrides),
            "overrides": [
                {"name": name, "sha256": baseline.overrides[name]}
                for name in sorted(baseline.overrides)
            ],
        },
        "checkpoints": {
            "rank96": rank96_checkpoints,
            "i21": i21_checkpoint,
        },
        "code": dict(code),
        "inputs": inputs,
    }


def load_production_models(
    config: InferenceConfig, authority: Authority, resolved_device: Any
) -> LoadedModels:
    """Configure/load CPU I21 before the first GPU scoring operation."""

    import eval_e23_i21_residual_candidate_ceiling as e23_eval
    from lightgbm import Booster

    i21_model, _record = e23_eval.load_frozen_i21_model(config.i21_checkpoint)
    rank_config = rank96.InferenceConfig(
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        output_zip=None,
        ranker_checkpoint=config.ranker_checkpoint,
        affinity_primary_checkpoint=config.affinity_primary_checkpoint,
        affinity_secondary_checkpoint=config.affinity_secondary_checkpoint,
        device=config.device,
        pair_batch=config.pair_batch,
        expected_count=config.expected_count,
        seed=config.seed,
    )
    rank_models = rank96.load_models(rank_config, resolved_device)
    try:
        predictor = Booster(model_file=str(authority.final_model_path))
    except Exception as exc:
        raise E24InferenceError("final E24 LightGBM model could not be loaded") from exc
    if predictor.num_trees() != 256 or predictor.num_feature() != FEATURE_COUNT:
        raise E24InferenceError("final E24 model must be exactly 256 trees over 227 features")
    return LoadedModels(rank96_models=rank_models, i21_model=i21_model, selector_model=predictor)


def _rank96_label_free_graph(
    image: np.ndarray, models: LoadedModels, *, pair_batch: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from eval_candidate_rank import score_full_graph
    from eval_seeded_qap import dense_rd
    from train_offset_pose import mine_affinity_candidates

    tiles_uint8 = rank96.split_upright_tiles(image)
    tensor = (
        torch.from_numpy(tiles_uint8)
        .permute(0, 3, 1, 2)
        .contiguous()
        .float()
        .to(models.rank96_models.device)
        .div_(255.0)
    )
    candidates_batch, valid_batch = mine_affinity_candidates(
        models.rank96_models.affinity_primary,
        tensor.unsqueeze(0),
        candidate_k=64,
        device=models.rank96_models.device,
        affinity_secondary=models.rank96_models.affinity_secondary,
    )
    candidates = candidates_batch[0]
    valid = valid_batch[0]
    if tuple(candidates.shape) != (NUM_TILES, RAW_WIDTH) or valid.shape != candidates.shape:
        raise E24InferenceError("Rank96 candidate union shape drifted")
    raw = score_full_graph(
        models.rank96_models.ranker,
        tensor,
        candidates,
        valid,
        pair_batch=pair_batch,
        device=models.rank96_models.device,
    )
    if tuple(raw.shape) != (4, NUM_TILES, RAW_WIDTH):
        raise E24InferenceError("Rank96 raw score shape drifted")
    candidates_cpu_tensor = candidates.detach().cpu().long().contiguous()
    raw_cpu_tensor = raw.detach().float().cpu().contiguous()
    valid_cpu = valid.detach().cpu().bool().contiguous()
    finite = torch.isfinite(raw_cpu_tensor)
    expected_finite = valid_cpu.unsqueeze(0).expand_as(raw_cpu_tensor)
    if not bool(torch.equal(finite, expected_finite)) or not bool(
        torch.isneginf(raw_cpu_tensor[~expected_finite]).all()
    ):
        raise E24InferenceError("Rank96 raw score padding is not exact -inf across UDLR")
    right, down = dense_rd(candidates_cpu_tensor, raw_cpu_tensor)
    candidate_ids = np.ascontiguousarray(candidates_cpu_tensor.numpy(), dtype=np.int64)
    raw_logits = np.ascontiguousarray(raw_cpu_tensor.numpy(), dtype=np.float32)
    right_np = np.ascontiguousarray(right.detach().float().cpu().numpy(), dtype=np.float32)
    down_np = np.ascontiguousarray(down.detach().float().cpu().numpy(), dtype=np.float32)
    del tensor, candidates_batch, valid_batch, candidates, valid, raw, candidates_cpu_tensor
    del raw_cpu_tensor, valid_cpu, finite, expected_finite, right, down
    if models.rank96_models.device.type == "cuda":
        torch.cuda.empty_cache()
    return tiles_uint8, candidate_ids, raw_logits, right_np, down_np


def infer_one_e24(image: np.ndarray, models: LoadedModels, *, pair_batch: int) -> InferredImage:
    """Run one upright image; never save the in-memory 227-column table."""

    import eval_e23_i21_residual_candidate_ceiling as e23_eval
    from solve_buddies import solve_components_from_scores

    tiles, candidate_ids, raw_logits, right, down = _rank96_label_free_graph(
        image, models, pair_batch=pair_batch
    )
    spatial_logits = e23_eval.infer_spatial_logits(tiles, models.i21_model)
    e23_eval.preflight_spatial_deployability(
        image_id=0,
        candidate_ids=candidate_ids,
        raw_logits=raw_logits,
        spatial_logits=spatial_logits,
    )
    result = e23_core.run_i21_residual_candidate_oracle(
        candidate_ids, raw_logits, spatial_logits
    )
    table = selector.extract_relation_features(
        result, candidate_ids, raw_logits, spatial_logits, tiles
    )
    scores = selector.predict_scores(models.selector_model, table)
    decoded = selector.decode_relation_scores(result, table, scores)
    board, objective = solve_components_from_scores(
        right,
        down,
        decoded.components,
        repair_passes=0,
        restarts=1,
        seed=1234,
    )
    board = rank96._assert_board(board)
    if not np.isfinite(float(objective)):
        raise E24InferenceError("component solver returned a non-finite objective")
    assembled = rank96.assemble_upright_tiles(tiles, board)
    output = rank96.fixed_nlm(assembled)
    accepted = sum(outcome.accepted for outcome in decoded.outcomes)
    inferred = InferredImage(
        output=output,
        board=np.ascontiguousarray(board),
        objective=float(objective),
        candidate_ids_sha256=rank96.sha256_array(candidate_ids),
        raw_scores_sha256=rank96.sha256_array(raw_logits),
        spatial_logits_sha256=rank96.sha256_array(spatial_logits),
        relation_rows=table.rows,
        relation_queries=table.queries,
        proposed_relations=len(decoded.attempted),
        accepted_relations=int(accepted),
        tree_merges=decoded.tree_merges,
        cycle_acceptances=decoded.cycle_acceptances,
    )
    del table, scores, decoded, result, candidate_ids, raw_logits, spatial_logits
    del right, down, assembled
    gc.collect()
    return inferred


def run_label_free_production_canary(
    config: InferenceConfig,
    authority: Authority,
    models: LoadedModels,
) -> dict[str, Any]:
    """Replay opened scene17 through production code before any test inventory."""

    e24_root = config.ledger.resolve().parents[1]
    input_root = e24_root / "label_free_inputs_v1/image_0017"
    input_manifest_path = input_root / "input_manifest.json"
    feature_manifest_path = e24_root / "feature_cache_v1/image_0017_features.json"
    ledger = _load_e24_canonical_json(config.ledger, label="canary ledger")
    input_manifest = _load_e24_canonical_json(
        input_manifest_path, label="scene17 label-free input manifest"
    )
    feature_manifest = _load_e24_canonical_json(
        feature_manifest_path, label="scene17 feature manifest"
    )
    if (
        input_manifest.get("image") != 17
        or input_manifest.get("status") != "complete"
        or input_manifest.get("ledger_sha256") != authority.ledger_sha256
        or input_manifest.get("run_contract_sha256") != authority.run_contract_sha256
        or input_manifest.get("orientation_degrees") != 0
        or input_manifest.get("reflection") is not False
        or feature_manifest.get("image") != 17
        or feature_manifest.get("status") != "complete"
        or feature_manifest.get("ledger_sha256") != authority.ledger_sha256
        or feature_manifest.get("run_contract_sha256") != authority.run_contract_sha256
        or feature_manifest.get("hypotheses") != 333_080
    ):
        raise E24InferenceError("scene17 stored production-canary authority drifted")
    projection_records = ledger.get("upstream", {}).get(
        "label_free_input_projection", {}
    ).get("records")
    if type(projection_records) is not list:
        raise E24InferenceError("ledger has no scene17 source projection")
    expected = next(
        (record for record in projection_records if type(record) is dict and record.get("image") == 17),
        None,
    )
    if expected is None:
        raise E24InferenceError("ledger scene17 source projection is absent")
    tiles_record = input_manifest.get("tiles")
    if type(tiles_record) is not dict or set(tiles_record) != {
        "array_sha256",
        "bytes",
        "dtype",
        "file_sha256",
        "path",
        "shape",
    }:
        raise E24InferenceError("scene17 tile record drifted")
    tiles_path = Path(tiles_record["path"]).resolve()
    if (
        tiles_path != (input_root / "tiles_uint8.npy").resolve()
        or not tiles_path.is_file()
        or tiles_path.stat().st_size != tiles_record["bytes"]
        or sha256_file(tiles_path) != tiles_record["file_sha256"]
    ):
        raise E24InferenceError("scene17 detached tile file provenance drifted")
    tiles = np.load(tiles_path, allow_pickle=False)
    if (
        not isinstance(tiles, np.ndarray)
        or tiles.shape != (576, 20, 20, 3)
        or tiles.dtype != np.uint8
    ):
        raise E24InferenceError("scene17 detached tiles are malformed")
    tiles = np.ascontiguousarray(tiles)
    if (
        _e24_array_sha256(tiles) != tiles_record["array_sha256"]
        or tiles_record["array_sha256"] != expected.get("tiles_uint8_sha256")
    ):
        raise E24InferenceError("scene17 tile array hash drifted")
    image = rank96.assemble_upright_tiles(tiles, np.arange(NUM_TILES))
    replay_tiles, candidate_ids, raw_logits, _right, _down = _rank96_label_free_graph(
        image, models, pair_batch=config.pair_batch
    )
    if not np.array_equal(replay_tiles, tiles):
        raise E24InferenceError("production Rank96 splitter changed scene17 tile bytes")
    import eval_e23_i21_residual_candidate_ceiling as e23_eval

    spatial_logits = e23_eval.infer_spatial_logits(tiles, models.i21_model)
    e23_eval.preflight_spatial_deployability(
        image_id=17,
        candidate_ids=candidate_ids,
        raw_logits=raw_logits,
        spatial_logits=spatial_logits,
    )
    observed_arrays = {
        "candidate_ids_sha256": _e24_array_sha256(candidate_ids),
        "raw_logits_sha256": _e24_array_sha256(raw_logits),
        "spatial_logits_sha256": _e24_array_sha256(spatial_logits),
    }
    expected_arrays = {
        "candidate_ids_sha256": expected.get("candidate_ids_sha256"),
        "raw_logits_sha256": expected.get("raw_logits_sha256"),
        "spatial_logits_sha256": expected.get("spatial_logits", {}).get("array_sha256"),
    }
    if observed_arrays != expected_arrays:
        raise E24InferenceError(
            f"scene17 production tensor parity failed: {observed_arrays} != {expected_arrays}"
        )
    result = e23_core.run_i21_residual_candidate_oracle(
        candidate_ids, raw_logits, spatial_logits
    )
    if len(result.hypotheses) != 333_080:
        raise E24InferenceError("scene17 production E23 hypothesis count drifted")
    table = selector.extract_relation_features(
        result, candidate_ids, raw_logits, spatial_logits, tiles
    )
    feature_bytes = selector.feature_table_npz_bytes(table)
    feature_sha = _sha256_bytes(feature_bytes)
    feature_record = feature_manifest.get("feature_file")
    if (
        type(feature_record) is not dict
        or feature_manifest.get("rows") != table.rows
        or feature_manifest.get("queries") != table.queries
        or feature_sha != feature_record.get("sha256")
    ):
        raise E24InferenceError("scene17 production 227-feature table parity failed")
    payload = {
        "schema": PRODUCTION_CANARY_SCHEMA,
        "status": "pass",
        "ledger_sha256": authority.ledger_sha256,
        "run_contract_sha256": authority.run_contract_sha256,
        "final_model_sha256": authority.final_model_sha256,
        "infer_e24_sha256": sha256_file(Path(__file__).resolve()),
        "image": 17,
        "hypotheses": len(result.hypotheses),
        "rows": table.rows,
        "queries": table.queries,
        "arrays": observed_arrays,
        "feature_npz_sha256": feature_sha,
        "checks": {
            "upright_tiles_exact": True,
            "rank96_candidate_ids_exact": True,
            "rank96_raw_logits_exact": True,
            "i21_spatial_logits_exact": True,
            "e23_hypotheses_exact": True,
            "e24_feature_bytes_exact": True,
        },
        "target_pngs_opened": False,
    }
    receipt_path = config.production_root.resolve() / "authority/scene17_label_free_parity.json"
    if receipt_path.exists():
        if _load_canonical_json(receipt_path, label="production parity receipt") != payload:
            raise E24InferenceError("existing production parity receipt belongs to other code/model")
    else:
        _atomic_write_json(receipt_path, payload)
    del feature_bytes, table, result, candidate_ids, raw_logits, spatial_logits
    del replay_tiles, tiles, image, _right, _down
    gc.collect()
    return payload


def solve_components_tail(
    tiles: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    components: Sequence[Mapping[int, tuple[int, int]]],
    *,
    solver: Any,
    restorer: Any,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Small injectable proof of the frozen E24 packing/restoration tail."""

    board, objective = solver(
        np.ascontiguousarray(right, dtype=np.float32),
        np.ascontiguousarray(down, dtype=np.float32),
        components,
        repair_passes=0,
        restarts=1,
        seed=1234,
    )
    order = rank96._assert_board(board)
    output = rank96._validate_rgb_array(
        restorer(rank96.assemble_upright_tiles(tiles, order)), label="restored output"
    )
    if not np.isfinite(float(objective)):
        raise E24InferenceError("component solver returned a non-finite objective")
    return output, order, float(objective)


def _resolved_manifest_path(config: InferenceConfig) -> Path:
    return (config.manifest_path or config.output_dir / "e24_manifest.json").resolve()


def _resolved_report_path(config: InferenceConfig) -> Path:
    return (config.report_path or config.output_dir / "e24_report.json").resolve()


def _existing_png_names(output_dir: Path) -> set[str]:
    if not output_dir.exists():
        return set()
    if not output_dir.is_dir():
        raise E24InferenceError(f"output path is not a directory: {output_dir}")
    return {
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    }


def _initial_manifest(contract: Mapping[str, Any], digest: str) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "in_progress",
        "contract_digest": digest,
        "contract": contract,
        "completed": {},
        "final_zip": None,
    }


def _validate_resume_manifest(
    value: Any,
    *,
    contract: Mapping[str, Any],
    digest: str,
    output_dir: Path,
) -> dict[str, Any]:
    if (
        type(value) is not dict
        or value.get("schema") != MANIFEST_SCHEMA
        or value.get("contract_digest") != digest
        or value.get("contract") != contract
        or type(value.get("completed")) is not dict
        or set(value) != {
            "schema",
            "status",
            "contract_digest",
            "contract",
            "completed",
            "final_zip",
        }
        or value.get("status")
        not in {"in_progress", "partial_runtime", "partial_limit", "failed", "completed"}
    ):
        raise E24InferenceError("resume manifest belongs to a different production contract")
    input_hashes = {row["name"]: row["sha256"] for row in contract["inputs"]}
    override_hashes = {
        row["name"]: row["sha256"] for row in contract["baseline"]["overrides"]
    }
    if value["status"] != "completed" and value["final_zip"] is not None:
        raise E24InferenceError("non-completed manifest must not claim a final ZIP")
    if not set(value["completed"]).issubset(input_hashes):
        raise E24InferenceError("resume manifest contains a non-input completed key")
    for name, record in value["completed"].items():
        _safe_basename(name)
        output = output_dir / name
        common = {
            "pipeline_contract_digest",
            "input_sha256",
            "output_sha256",
            "source",
        }
        if (
            name not in input_hashes
            or type(record) is not dict
            or record.get("pipeline_contract_digest") != digest
            or record.get("input_sha256") != input_hashes[name]
            or not output.is_file()
            or record.get("output_sha256") != sha256_file(output)
            or not _is_sha256(record.get("output_sha256"))
        ):
            raise E24InferenceError(f"resume output provenance mismatch: {name}")
        if name in override_hashes:
            if (
                set(record) != common | {"baseline_member_sha256"}
                or record.get("source") != "verified_source_override"
                or record.get("baseline_member_sha256") != override_hashes[name]
                or record.get("output_sha256") != override_hashes[name]
            ):
                raise E24InferenceError(f"resume override schema/provenance mismatch: {name}")
        else:
            generic_fields = {
                "board_sha256",
                "candidate_ids_sha256",
                "raw_scores_sha256",
                "spatial_logits_sha256",
                "solver_objective",
                "relation_rows",
                "relation_queries",
                "proposed_relations",
                "accepted_relations",
                "tree_merges",
                "cycle_acceptances",
            }
            if set(record) != common | generic_fields or record.get("source") != "e24_crs_v1":
                raise E24InferenceError(f"resume generic record schema drifted: {name}")
            if not all(
                _is_sha256(record.get(field))
                for field in (
                    "board_sha256",
                    "candidate_ids_sha256",
                    "raw_scores_sha256",
                    "spatial_logits_sha256",
                )
            ):
                raise E24InferenceError(f"resume generic hash field drifted: {name}")
            objective = record.get("solver_objective")
            if isinstance(objective, bool) or not isinstance(objective, (int, float)) or not math.isfinite(float(objective)):
                raise E24InferenceError(f"resume generic objective drifted: {name}")
            count_fields = (
                "relation_rows",
                "relation_queries",
                "proposed_relations",
                "accepted_relations",
                "tree_merges",
                "cycle_acceptances",
            )
            if any(type(record.get(field)) is not int or record[field] < 0 for field in count_fields):
                raise E24InferenceError(f"resume generic count field drifted: {name}")
            if (
                record["relation_rows"] < record["relation_queries"] * 2
                or record["accepted_relations"] > record["proposed_relations"]
                or record["tree_merges"] + record["cycle_acceptances"]
                != record["accepted_relations"]
            ):
                raise E24InferenceError(f"resume generic count algebra drifted: {name}")
        rank96.load_rgb_strict(output)
    return value


def _write_report(
    path: Path,
    *,
    status: str,
    contract_digest: str,
    input_count: int,
    completed_count: int,
    skipped_count: int,
    new_count: int,
    generic_count: int,
    override_count: int,
    elapsed_seconds: float,
    output_zip: Path | None = None,
    output_zip_sha256: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    report = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "contract_digest": contract_digest,
        "input_count": int(input_count),
        "completed_count": int(completed_count),
        "skipped_count": int(skipped_count),
        "new_count": int(new_count),
        "generic_count": int(generic_count),
        "override_count": int(override_count),
        "elapsed_seconds": float(elapsed_seconds),
        "output_zip": str(output_zip) if output_zip is not None else None,
        "output_zip_sha256": output_zip_sha256
        or (sha256_file(output_zip) if output_zip is not None and output_zip.is_file() else None),
        "baseline_preserved": True,
        "baseline_zip_sha256": BASELINE_ZIP_SHA256,
        "error": error,
    }
    _atomic_write_json(path, report)
    return report


def _validate_config(config: InferenceConfig) -> None:
    if config.limit < 0 or config.expected_count < 0 or config.max_runtime_seconds < 0:
        raise E24InferenceError("limit/count/runtime values must be non-negative")
    if config.pair_batch < 1:
        raise E24InferenceError("pair_batch must be positive")
    production_root = _require_e_write_path(config.production_root, label="production root")
    output_dir = _require_e_write_path(config.output_dir, label="output directory")
    output_zip = _require_e_write_path(config.output_zip, label="new submission ZIP")
    manifest = _require_e_write_path(_resolved_manifest_path(config), label="run manifest")
    report = _require_e_write_path(_resolved_report_path(config), label="run report")
    pending_zip = output_zip.with_name(output_zip.name + ".pending")
    if output_zip == config.baseline_zip.resolve():
        raise E24InferenceError("new output ZIP must never overwrite the Rank96 fallback")
    for label, path in (
        ("output directory", output_dir),
        ("new submission ZIP", output_zip),
        ("pending submission ZIP", pending_zip),
        ("run manifest", manifest),
        ("run report", report),
    ):
        if not _is_within(path, production_root):
            raise E24InferenceError(
                f"{label} must live below dedicated production root {production_root}"
            )
    read_targets = (
        config.input_dir,
        config.baseline_zip,
        config.baseline_manifest,
        config.ledger,
        config.canary_gate,
        config.structural_report,
        config.orchestration_receipt,
        config.ranker_checkpoint,
        config.affinity_primary_checkpoint,
        config.affinity_secondary_checkpoint,
        config.i21_checkpoint,
    )
    for path in read_targets:
        if _paths_overlap(production_root, path):
            raise E24InferenceError(
                f"production root overlaps a read-only input/authority path: {path.resolve(strict=False)}"
            )
    if output_zip in {manifest, report} or manifest == report or output_dir == output_zip:
        raise E24InferenceError("production write targets overlap")
    if config.limit and config.output_zip is not None:
        # The configured path is retained in the contract, but a partial run
        # never creates it.  This condition documents that invariant only.
        pass


def _build_verified_pending_zip(
    output_dir: Path,
    names: Sequence[str],
    pending_zip: Path,
    baseline: BaselineBundle,
    expected_output_sha256: Mapping[str, str],
) -> str:
    """Build a cross-zlib deterministic ZIP_STORED transaction candidate."""

    pending = _require_e_write_path(pending_zip, label="pending submission ZIP")
    pending.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{pending.name}.", dir=pending.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, mode="w", compression=zipfile.ZIP_STORED) as archive:
            for name in names:
                path = output_dir / _safe_basename(name)
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_STORED)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, pending)
    finally:
        if temporary.exists():
            temporary.unlink()
    with zipfile.ZipFile(pending, "r") as archive:
        if archive.namelist() != list(names):
            raise E24InferenceError("new ZIP member order/set drifted")
        bad = archive.testzip()
        if bad is not None:
            raise E24InferenceError(f"new ZIP CRC failure: {bad}")
        if set(expected_output_sha256) != set(names):
            raise E24InferenceError("completed output hash map differs from ZIP inventory")
        for name in names:
            expected = expected_output_sha256[name]
            if not _is_sha256(expected) or _sha256_bytes(archive.read(name)) != expected:
                raise E24InferenceError(f"new ZIP member differs from completed record: {name}")
        for name, expected in baseline.overrides.items():
            if _sha256_bytes(archive.read(name)) != expected:
                raise E24InferenceError(f"new ZIP changed verified override bytes: {name}")
    return sha256_file(pending)


def run_inference(config: InferenceConfig) -> dict[str, Any]:
    """Run/resume E24, aborting generics on the first error without a new ZIP."""

    _validate_config(config)
    started = time.perf_counter()
    # Critical ordering: authority is authenticated before target inventory.
    authority = authenticate_e24_authority(config)
    if _paths_overlap(config.production_root, authority.final_model_path):
        raise E24InferenceError("production root overlaps the authenticated final model")
    baseline = authenticate_baseline(config.baseline_zip, config.baseline_manifest, None)
    i21_checkpoint = _configure_i21_runtime_and_checkpoint(config.i21_checkpoint)
    rank96_checkpoints = _rank96_checkpoint_provenance(config)
    if rank96_checkpoints != baseline.rank96_checkpoints:
        raise E24InferenceError(
            "current Rank96 checkpoints differ from the immutable baseline contract"
        )
    resolved_device = rank96.resolve_device(config.device)
    code = _code_provenance()
    models = load_production_models(config, authority, resolved_device)
    production_canary = run_label_free_production_canary(config, authority, models)
    production_canary_sha = _canonical_digest(production_canary)
    # The first target-directory access occurs only after every authority and
    # opened-scene deployment-equivalence check above has passed.
    names = _list_input_names(config.input_dir)
    if config.expected_count and len(names) != config.expected_count:
        raise E24InferenceError(
            f"expected exactly {config.expected_count} inputs, found {len(names)}"
        )
    if tuple(names) != baseline.names:
        raise E24InferenceError("target input names differ from the immutable baseline inventory")
    inputs = _input_inventory(config, names)
    contract = _build_contract(
        config,
        authority=authority,
        baseline=baseline,
        inputs=inputs,
        rank96_checkpoints=rank96_checkpoints,
        i21_checkpoint=i21_checkpoint,
        code=code,
        resolved_device=str(resolved_device),
        production_canary_sha256=production_canary_sha,
    )
    contract_digest = _canonical_digest(contract)
    if config.dry_run:
        return {
            "status": "dry_run",
            "contract_digest": contract_digest,
            "input_count": len(names),
            "override_count": len(baseline.overrides),
            "device": str(resolved_device),
            "baseline_preserved": True,
        }

    output_dir = config.output_dir.resolve()
    output_zip = config.output_zip.resolve()
    pending_zip = output_zip.with_name(output_zip.name + ".pending")
    manifest_path = _resolved_manifest_path(config)
    report_path = _resolved_report_path(config)
    existing = _existing_png_names(output_dir)
    extras = sorted(existing - set(names))
    if extras:
        raise E24InferenceError(f"output directory has extra PNGs: {extras[:8]}")
    if manifest_path.exists():
        if not config.resume:
            raise E24InferenceError(f"manifest exists; use --resume: {manifest_path}")
        manifest = _validate_resume_manifest(
            _load_canonical_json(manifest_path, label="E24 production manifest"),
            contract=contract,
            digest=contract_digest,
            output_dir=output_dir,
        )
        if manifest.get("status") != "completed" and config.output_zip.exists():
            raise E24InferenceError(
                "a non-completed run has a stale submit-looking final ZIP; move it aside before resume"
            )
        if manifest.get("status") == "completed":
            record = manifest.get("final_zip")
            if (
                type(record) is not dict
                or set(record) != {"path", "bytes", "sha256"}
                or Path(record.get("path", "")).resolve() != output_zip
                or type(record.get("bytes")) is not int
                or not _is_sha256(record.get("sha256"))
            ):
                raise E24InferenceError("completed manifest final ZIP record drifted")
            completed_report = _load_canonical_json(
                report_path, label="completed E24 report"
            )
            if (
                completed_report.get("schema") != REPORT_SCHEMA
                or completed_report.get("status") != "completed"
                or completed_report.get("contract_digest") != contract_digest
                or completed_report.get("output_zip") != str(output_zip)
                or completed_report.get("output_zip_sha256") != record["sha256"]
            ):
                raise E24InferenceError("completed report does not bind the final ZIP record")
            if output_zip.is_file():
                if (
                    output_zip.stat().st_size != record["bytes"]
                    or sha256_file(output_zip) != record["sha256"]
                ):
                    raise E24InferenceError("published final ZIP does not match its commit record")
                return completed_report
            if pending_zip.is_file():
                if (
                    pending_zip.stat().st_size != record["bytes"]
                    or sha256_file(pending_zip) != record["sha256"]
                ):
                    raise E24InferenceError("pending ZIP does not match completed publish record")
                os.replace(pending_zip, output_zip)
                return completed_report
    else:
        if existing:
            raise E24InferenceError("existing outputs have no authenticated resume manifest")
        if config.output_zip.exists():
            raise E24InferenceError("new ZIP path already exists without this run manifest")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = _initial_manifest(contract, contract_digest)
        _atomic_write_json(manifest_path, manifest)

    completed: dict[str, Any] = manifest["completed"]
    skipped_count = len(completed)
    new_count = 0
    generic_count = sum(row.get("source") == "e24_crs_v1" for row in completed.values())
    override_count = sum(
        row.get("source") == "verified_source_override" for row in completed.values()
    )
    input_hashes = {row["name"]: row["sha256"] for row in inputs}
    try:
        for name in names:
            if name in completed:
                continue
            if config.max_runtime_seconds and time.perf_counter() - started >= config.max_runtime_seconds:
                manifest["status"] = "partial_runtime"
                _atomic_write_json(manifest_path, manifest)
                raise IncompleteRun("E24 stopped at a safe runtime boundary; rerun with --resume")
            if config.limit and new_count >= config.limit:
                manifest["status"] = "partial_limit"
                _atomic_write_json(manifest_path, manifest)
                raise IncompleteRun("E24 stopped at a safe limit boundary; rerun with --resume")
            image_started = time.perf_counter()
            if name in baseline.overrides:
                content = _read_baseline_override(baseline, name)
                _atomic_write_bytes(output_dir / name, content)
                output_sha = _sha256_bytes(content)
                record: dict[str, Any] = {
                    "pipeline_contract_digest": contract_digest,
                    "input_sha256": input_hashes[name],
                    "output_sha256": output_sha,
                    "source": "verified_source_override",
                    "baseline_member_sha256": baseline.overrides[name],
                }
                override_count += 1
            else:
                image = rank96.load_rgb_strict(config.input_dir / name)
                inferred = infer_one_e24(image, models, pair_batch=config.pair_batch)
                output_sha = rank96._atomic_write_png(output_dir / name, inferred.output)
                record = {
                    "pipeline_contract_digest": contract_digest,
                    "input_sha256": input_hashes[name],
                    "output_sha256": output_sha,
                    "source": "e24_crs_v1",
                    "board_sha256": rank96.sha256_array(inferred.board.astype(np.int16)),
                    "candidate_ids_sha256": inferred.candidate_ids_sha256,
                    "raw_scores_sha256": inferred.raw_scores_sha256,
                    "spatial_logits_sha256": inferred.spatial_logits_sha256,
                    "solver_objective": inferred.objective,
                    "relation_rows": inferred.relation_rows,
                    "relation_queries": inferred.relation_queries,
                    "proposed_relations": inferred.proposed_relations,
                    "accepted_relations": inferred.accepted_relations,
                    "tree_merges": inferred.tree_merges,
                    "cycle_acceptances": inferred.cycle_acceptances,
                }
                generic_count += 1
                del image, inferred
                gc.collect()
            completed[name] = record
            manifest["completed"] = completed
            manifest["status"] = "in_progress"
            _atomic_write_json(manifest_path, manifest)
            new_count += 1
            elapsed = time.perf_counter() - image_started
            _write_report(
                report_path,
                status="in_progress",
                contract_digest=contract_digest,
                input_count=len(names),
                completed_count=len(completed),
                skipped_count=skipped_count,
                new_count=new_count,
                generic_count=generic_count,
                override_count=override_count,
                elapsed_seconds=time.perf_counter() - started,
            )
            print(
                json.dumps(
                    {
                        "completed": len(completed),
                        "total": len(names),
                        "name": name,
                        "source": record["source"],
                        "seconds": round(elapsed, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if set(completed) != set(names):
            raise IncompleteRun("E24 run is incomplete; rerun with --resume")
        for name in names:
            rank96.load_rgb_strict(output_dir / name)
            if sha256_file(output_dir / name) != completed[name]["output_sha256"]:
                raise E24InferenceError(f"completed output hash drifted: {name}")
        pending_sha = _build_verified_pending_zip(
            output_dir,
            names,
            pending_zip,
            baseline,
            {name: completed[name]["output_sha256"] for name in names},
        )
        if (
            config.baseline_zip.stat().st_size != BASELINE_ZIP_SIZE
            or sha256_file(config.baseline_zip) != BASELINE_ZIP_SHA256
        ):
            raise E24InferenceError("Rank96 fallback changed during E24 production")
        final_zip_record = {
            "path": str(output_zip),
            "bytes": pending_zip.stat().st_size,
            "sha256": pending_sha,
        }
        report = _write_report(
            report_path,
            status="completed",
            contract_digest=contract_digest,
            input_count=len(names),
            completed_count=len(completed),
            skipped_count=skipped_count,
            new_count=new_count,
            generic_count=generic_count,
            override_count=override_count,
            elapsed_seconds=time.perf_counter() - started,
            output_zip=output_zip,
            output_zip_sha256=pending_sha,
        )
        manifest["status"] = "completed"
        manifest["final_zip"] = final_zip_record
        _atomic_write_json(manifest_path, manifest)
        # Final ZIP publication is the last filesystem mutation in a successful
        # run.  Before this replace there is no submit-looking new ZIP.
        os.replace(pending_zip, output_zip)
        return report
    except IncompleteRun:
        _write_report(
            report_path,
            status=manifest["status"],
            contract_digest=contract_digest,
            input_count=len(names),
            completed_count=len(completed),
            skipped_count=skipped_count,
            new_count=new_count,
            generic_count=generic_count,
            override_count=override_count,
            elapsed_seconds=time.perf_counter() - started,
        )
        raise
    except Exception as exc:
        manifest["status"] = "failed"
        _atomic_write_json(manifest_path, manifest)
        _write_report(
            report_path,
            status="failed",
            contract_digest=contract_digest,
            input_count=len(names),
            completed_count=len(completed),
            skipped_count=skipped_count,
            new_count=new_count,
            generic_count=generic_count,
            override_count=override_count,
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def smoke_contract() -> dict[str, Any]:
    image = np.zeros((480, 480, 3), dtype=np.uint8)
    image[..., 0] = np.arange(480, dtype=np.uint16)[:, None] % 251
    image[..., 1] = np.arange(480, dtype=np.uint16)[None, :] % 253
    tiles = rank96.split_upright_tiles(image)
    rebuilt = rank96.assemble_upright_tiles(tiles, np.arange(NUM_TILES))
    if not np.array_equal(image, rebuilt):
        raise AssertionError("upright E24 split/assembly roundtrip failed")
    return {
        "status": "smoke_pass",
        "contract_digest": _canonical_digest(PRODUCTION_CONTRACT),
        "feature_count": FEATURE_COUNT,
        "orientation": PRODUCTION_CONTRACT["orientation"],
        "baseline_sha256": BASELINE_ZIP_SHA256,
    }


def _defaults() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    e24_root = Path("E:/pazzle_work/posegraph_e24_selector")
    outputs = Path("E:/pazzle_work/submissions/e24_crs_v1")
    rank_defaults = rank96._default_checkpoints()
    return {
        "input_dir": Path("E:/pazzle_data/test"),
        "production_root": outputs,
        "output_dir": outputs / "png",
        "output_zip": outputs / "submission_e24_crs_v1.zip",
        "baseline_zip": Path("E:/pazzle_work/submission_rank96_v1.zip"),
        "baseline_manifest": root / "artifacts/rank96_submission_v1/rank96_manifest.json",
        "ledger": e24_root / "preflight/e24_crs_v1_preflight.json",
        "canary_gate": e24_root / "canary/scene_0017_gate.json",
        "structural_report": e24_root / "contextual_relation_selector_oof_v1.json",
        "orchestration_receipt": e24_root / "oof_orchestration_receipt.json",
        "ranker_checkpoint": rank_defaults["ranker"],
        "affinity_primary_checkpoint": rank_defaults["affinity_primary"],
        "affinity_secondary_checkpoint": rank_defaults["affinity_secondary"],
        "i21_checkpoint": Path("E:/pazzle_work/positional_ddpm/positional_ddpm_train_latest.pt"),
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = _defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=defaults["input_dir"])
    parser.add_argument("--production-root", type=Path, default=defaults["production_root"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--output-zip", type=Path, default=defaults["output_zip"])
    parser.add_argument("--baseline-zip", type=Path, default=defaults["baseline_zip"])
    parser.add_argument("--baseline-manifest", type=Path, default=defaults["baseline_manifest"])
    parser.add_argument("--ledger", type=Path, default=defaults["ledger"])
    parser.add_argument("--canary-gate", type=Path, default=defaults["canary_gate"])
    parser.add_argument("--structural-report", type=Path, default=defaults["structural_report"])
    parser.add_argument("--orchestration-receipt", type=Path, default=defaults["orchestration_receipt"])
    parser.add_argument("--ranker-ckpt", type=Path, default=defaults["ranker_checkpoint"])
    parser.add_argument("--affinity-ckpt", type=Path, default=defaults["affinity_primary_checkpoint"])
    parser.add_argument("--affinity-ckpt2", type=Path, default=defaults["affinity_secondary_checkpoint"])
    parser.add_argument("--i21-ckpt", type=Path, default=defaults["i21_checkpoint"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pair-batch", type=int, default=DEFAULT_PAIR_BATCH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-count", type=int, default=DEFAULT_EXPECTED_COUNT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-runtime-seconds", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> InferenceConfig:
    return InferenceConfig(
        input_dir=args.input_dir.resolve(),
        production_root=args.production_root.resolve(),
        output_dir=args.output_dir.resolve(),
        output_zip=args.output_zip.resolve(),
        baseline_zip=args.baseline_zip.resolve(),
        baseline_manifest=args.baseline_manifest.resolve(),
        ledger=args.ledger.resolve(),
        canary_gate=args.canary_gate.resolve(),
        structural_report=args.structural_report.resolve(),
        orchestration_receipt=args.orchestration_receipt.resolve(),
        ranker_checkpoint=args.ranker_ckpt.resolve(),
        affinity_primary_checkpoint=args.affinity_ckpt.resolve(),
        affinity_secondary_checkpoint=args.affinity_ckpt2.resolve(),
        i21_checkpoint=args.i21_ckpt.resolve(),
        device=args.device,
        pair_batch=args.pair_batch,
        seed=args.seed,
        expected_count=args.expected_count,
        limit=args.limit,
        max_runtime_seconds=args.max_runtime_seconds,
        resume=args.resume,
        dry_run=args.dry_run,
        manifest_path=args.manifest.resolve() if args.manifest is not None else None,
        report_path=args.report.resolve() if args.report is not None else None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        print(json.dumps(smoke_contract(), indent=2, sort_keys=True), flush=True)
        return 0
    try:
        result = run_inference(_config_from_args(args))
    except IncompleteRun as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return INCOMPLETE_EXIT_CODE
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
