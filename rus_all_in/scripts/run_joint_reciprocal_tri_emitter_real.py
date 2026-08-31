#!/usr/bin/env python3
"""Train and evaluate one signed joint reciprocal tri-emitter endpoint.

The real experiment is deliberately split into three irreversible stages:

``fit``
    Train one endpoint from scratch from the immutable tri-emitter FIT caches.
``freeze-fit-heads`` / ``score-fit-heads``
    Keep fixed 5% FIT selections target-free, then score labels separately.
``freeze-dev``
    Materialise target-free DEV candidates, scores and fixed 5% heads.
``score-dev``
    Verify the pre-score freeze hashes first, and only then reconstruct labels.

There is intentionally no terminal or competition-test mode in this runner.
The checked-in config is a populated but unsigned, blocked template; final
review must create a separate signed protocol before any stage runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar

import numpy as np
import torch

from aiijc_puzzle.joint_reciprocal_tri_emitter_verifier import (
    CONFIDENCE_BCE_WEIGHT,
    DELTA_REGULARIZATION_WEIGHT,
    RECIPROCAL_HEAD_FRACTION,
    SOFTMIN_TAU,
    JointReciprocalTriEmitterVerifier,
    exact_joint_targets,
    fixed_fraction_reciprocal_head,
    joint_assignment_loss,
    joint_verifier_contract,
)
from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file
from aiijc_puzzle.restoration_r6 import distort_tiles
from aiijc_puzzle.synthetic_socket_evaluation import (
    DEFAULT_SYNTHETIC_NAMESPACE,
    ExactSyntheticReference,
    SyntheticSocketInput,
    exact_local_retrieval_metrics,
    make_exact_synthetic_case,
    names_digest,
)
from aiijc_puzzle.tri_emitter_edge_verifier import (
    AUXILIARY_DIM,
    DINO_PROJECTION_DIM,
    EMITTERS,
    TOP_K,
    candidate_pool_digest,
)

try:
    from scripts import run_fullres_boundary_denoiser as boundary
    from scripts import run_tri_emitter_edge_verifier as prior
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_fullres_boundary_denoiser as boundary
    import run_tri_emitter_edge_verifier as prior

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/joint_reciprocal_tri_emitter_real_unsigned_template_v1.json"
)
DEFAULT_EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "outputs/joint-reciprocal-tri-emitter-verifier/real-fit32-draw2-dev32-development-v1"
)
FIT_ENDPOINT = Path("fit/joint_reciprocal_endpoint.pt")
FIT_REPORT = Path("fit/report.json")
FIT_HEAD_ARCHIVE = Path("fit/frozen-target-free-reciprocal-heads.npz")
FIT_HEAD_METADATA = Path("fit/frozen-target-free-reciprocal-heads.json")
FIT_HEAD_PRE_SCORE_FREEZE = Path("fit/reciprocal-heads-pre-score-freeze.json")
FIT_HEAD_SCORE = Path("fit/reciprocal-heads-score.json")
DEV_PANEL = Path("dev")
FREEZE_ARCHIVE = Path("frozen-target-free-predictions.npz")
FREEZE_METADATA = Path("frozen-target-free-predictions.json")
PRE_SCORE_FREEZE = Path("pre-score-freeze.json")
DEV_SCORE = Path("score.json")

FIT_CACHE_KEYS = frozenset(
    {
        "raw_sides",
        "dino_sides",
        "candidates",
        "valid",
        "auxiliary",
        "raw_baseline",
        "emitter_topk",
        "target_slots",
    }
)
SIGNED_STATUS = "signed-fixed-protocol"
BLOCKED_STATUS = "unsigned-template-blocked-awaiting-final-review"
CHECKPOINT_SCHEMA = "aiijc-joint-reciprocal-real-fit-checkpoint-v1"
CONFIG_SCHEMA = "aiijc-joint-reciprocal-tri-emitter-real-protocol-v1"


@dataclass(frozen=True)
class VerifiedFreeze:
    """A freeze whose archive and metadata hashes were verified before labels."""

    panel_dir: Path
    archive: Path
    metadata: Path
    archive_sha256: str
    metadata_sha256: str
    rows: tuple[dict[str, Any], ...]


T = TypeVar("T")
R = TypeVar("R")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "fit",
            "freeze-fit-heads",
            "score-fit-heads",
            "freeze-dev",
            "score-dev",
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--manifest", type=Path, default=prior.roster.DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=prior.roster.DEFAULT_TARGETS)
    parser.add_argument("--socket-checkpoint", type=Path, default=prior.SOCKET_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        label = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        label = str(resolved)
    return {"path": label, "sha256": sha256_file(resolved)}


def _require_exact_contract(config: Mapping[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise RuntimeError("joint reciprocal real protocol schema changed")
    model = config.get("fixed_model", {})
    if model.get("architecture") != "joint-reciprocal-tri-emitter-verifier-v1":
        raise RuntimeError("joint reciprocal architecture is not fixed")
    if model.get("candidate_roster") != "raw+adapter1600+DINO-top32-stable-union":
        raise RuntimeError("candidate identity contract changed")
    if int(model.get("dino_projection_dim", -1)) != DINO_PROJECTION_DIM:
        raise RuntimeError("DINO projection contract changed")
    if int(model.get("auxiliary_dim", -1)) != AUXILIARY_DIM:
        raise RuntimeError("auxiliary feature contract changed")
    objective = config.get("objective", {})
    expected = {
        "row_cross_entropy_weight": 1.0,
        "column_cross_entropy_weight": 1.0,
        "confidence_bce_weight": CONFIDENCE_BCE_WEIGHT,
        "delta_l2_weight": DELTA_REGULARIZATION_WEIGHT,
        "softmin_tau": SOFTMIN_TAU,
        "fixed_reciprocal_fraction_per_axis_per_board": RECIPROCAL_HEAD_FRACTION,
    }
    for key, value in expected.items():
        if not math.isclose(float(objective.get(key, math.nan)), value, abs_tol=1e-12):
            raise RuntimeError(f"fixed objective changed: {key}")
    if objective.get("learned_none") is not True:
        raise RuntimeError("learned NONE contract is required")
    gate = config.get("dev_gate", {})
    expected_gate = {
        "pooled_r1_gain_minimum": 0.005,
        "pooled_r5_gain_minimum": 0.0,
        "pooled_fixed_head_precision_gain_minimum": 0.02,
        "per_axis_gain_minimum": 0.0,
    }
    for key, value in expected_gate.items():
        if not math.isclose(float(gate.get(key, math.nan)), value, abs_tol=1e-12):
            raise RuntimeError(f"fixed discovery gate changed: {key}")
    training = config.get("training", {})
    if int(training.get("seed", -1)) != 20260913:
        raise RuntimeError("audited real-FIT seed changed")
    if int(training.get("epochs", -1)) != 3:
        raise RuntimeError("audited nominal epoch count changed")
    if int(training.get("optimizer_updates", -1)) != 1752:
        raise RuntimeError("audited optimizer update count changed")
    source = config.get("source_protocol", {})
    if int(source.get("dev_case_seed", -1)) != 20260908:
        raise RuntimeError("audited DEV case seed changed")
    if int(source.get("dev_draw_index", -1)) != 0:
        raise RuntimeError("audited DEV draw index changed")


def _validate_frozen_evidence(config: Mapping[str, Any]) -> None:
    frozen = config.get("frozen_inputs", {})
    required = {
        "fit_cache_report",
        "roster_audit_report",
        "capacity_report",
        "socket_checkpoint",
        "adapter1600_checkpoint",
        "dino_checkpoint",
        "module",
        "runner",
    }
    missing = required - set(frozen)
    if missing:
        raise RuntimeError(f"signed protocol omits frozen evidence: {sorted(missing)}")
    if "legacy_tri_checkpoint" in frozen or "capacity_checkpoint" in frozen:
        raise RuntimeError("real FIT protocol must not freeze a warm-start checkpoint")
    capacity = json.loads(
        _project_path(frozen["capacity_report"]["path"]).read_text(encoding="utf-8")
    )
    if capacity.get("status") != "pass" or capacity.get("gate", {}).get("passed") is not True:
        raise RuntimeError("reviewed joint reciprocal capacity gate did not pass")
    if capacity.get("real_fit_or_dev_or_terminal_panel_opened") is not False:
        raise RuntimeError("capacity evidence unexpectedly opened a real panel")
    audit = json.loads(
        _project_path(frozen["roster_audit_report"]["path"]).read_text(encoding="utf-8")
    )
    proposal = audit.get("proposal", {})
    source = config["source_protocol"]
    if proposal.get("fit", {}).get("source_filenames") != source["fit_filenames"]:
        raise RuntimeError("signed FIT roster differs from roster audit")
    if proposal.get("dev", {}).get("source_filenames") != source["dev_filenames"]:
        raise RuntimeError("signed DEV roster differs from roster audit")
    if (
        audit.get("protected_terminal16", {}).get("source_filenames")
        != source["terminal16_owned_filenames"]
    ):
        raise RuntimeError("signed protocol lost terminal16 ownership exclusion")
    parent = json.loads(
        _project_path(frozen["fit_cache_report"]["path"]).read_text(encoding="utf-8")
    )
    if (
        parent.get("protocol", {}).get("local_filenames")
        != source["opened_local16_owned_filenames"]
    ):
        raise RuntimeError("signed protocol lost opened-local16 exclusion")


def _load_signed_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"joint reciprocal protocol is missing: {resolved}")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if config.get("status") == BLOCKED_STATUS:
        raise RuntimeError(
            "joint reciprocal real protocol is intentionally blocked until the "
            "populated template receives final review and a separate signed config"
        )
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if config.get("status") != SIGNED_STATUS or not sidecar.is_file():
        raise RuntimeError("joint reciprocal real protocol is not signed/fixed")
    digest = sha256_file(resolved)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise RuntimeError("joint reciprocal real protocol sidecar mismatch")
    _require_exact_contract(config)
    for artifact in config.get("frozen_inputs", {}).values():
        target = _project_path(artifact["path"])
        if not target.is_file() or sha256_file(target) != artifact["sha256"]:
            raise RuntimeError(f"frozen joint reciprocal input changed: {target}")
    _validate_frozen_evidence(config)
    return config, digest


def validate_source_rosters(config: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Fail closed on overlap with FIT, opened-local or terminal ownership."""

    source = config.get("source_protocol", {})

    def names(key: str, *, allow_empty: bool = False) -> tuple[str, ...]:
        value = source.get(key)
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
            raise RuntimeError(f"source protocol {key} must be a filename list")
        result = tuple(value)
        if not allow_empty and not result:
            raise RuntimeError(f"source protocol {key} is empty")
        if len(set(result)) != len(result):
            raise RuntimeError(f"source protocol {key} contains duplicates")
        return result

    fit = names("fit_filenames")
    dev = names("dev_filenames")
    opened = names("opened_local16_owned_filenames", allow_empty=True)
    terminal = names("terminal16_owned_filenames", allow_empty=True)
    audit_excluded = names("source_audit_excluded_filenames", allow_empty=True)
    if names_digest(fit) != source.get("fit_digest"):
        raise RuntimeError("signed FIT roster digest mismatch")
    if names_digest(dev) != source.get("dev_digest"):
        raise RuntimeError("signed DEV roster digest mismatch")
    groups = {"fit": set(fit), "dev": set(dev)}
    if groups["fit"] & groups["dev"]:
        raise RuntimeError("FIT and DEV sources overlap")
    forbidden = set(opened) | set(terminal) | set(audit_excluded)
    overlap = groups["dev"] & forbidden
    if overlap:
        raise RuntimeError(
            "DEV reuses opened-local/terminal/audited source ownership: "
            + ", ".join(sorted(overlap))
        )
    return {
        "fit": fit,
        "dev": dev,
        "opened_local16_owned": opened,
        "terminal16_owned": terminal,
        "source_audit_excluded": audit_excluded,
    }


def validate_fit_cache_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_tile_count: int = 576,
) -> None:
    """Validate the immutable prior tri-emitter FIT cache schema and labels."""

    if set(arrays) != FIT_CACHE_KEYS:
        missing = sorted(FIT_CACHE_KEYS - set(arrays))
        extra = sorted(set(arrays) - FIT_CACHE_KEYS)
        raise RuntimeError(f"FIT cache keys changed; missing={missing}, extra={extra}")
    count = expected_tile_count
    candidates = np.asarray(arrays["candidates"])
    valid = np.asarray(arrays["valid"])
    if candidates.ndim != 3 or candidates.shape[:2] != (2, count):
        raise RuntimeError("FIT cache candidates must be 2 x N x K")
    width = candidates.shape[2]
    expected_shapes = {
        "raw_sides": (4, count, 20, 6),
        "dino_sides": (4, count, 14, DINO_PROJECTION_DIM),
        "valid": (2, count, width),
        "auxiliary": (2, count, width, AUXILIARY_DIM),
        "raw_baseline": (2, count, width),
        "emitter_topk": (len(EMITTERS), 2, count, min(TOP_K, count - 1)),
        "target_slots": (2, count),
    }
    for key, shape in expected_shapes.items():
        if np.asarray(arrays[key]).shape != shape:
            raise RuntimeError(f"FIT cache {key} shape changed")
    if candidates.dtype not in (np.int32, np.int64):
        raise RuntimeError("FIT cache candidates must be int32/int64")
    if valid.dtype != np.bool_:
        raise RuntimeError("FIT cache valid mask must be boolean")
    if np.asarray(arrays["target_slots"]).dtype not in (
        np.int16,
        np.int32,
        np.int64,
    ):
        raise RuntimeError("FIT cache target slots must be integer")
    for key in ("raw_sides", "dino_sides", "auxiliary", "raw_baseline"):
        value = np.asarray(arrays[key])
        if value.dtype not in (np.float16, np.float32) or not np.isfinite(value).all():
            raise RuntimeError(f"FIT cache {key} must be finite float16/float32")
    if np.any(valid & ((candidates < 0) | (candidates >= count))):
        raise RuntimeError("FIT cache candidate identity is out of range")
    for axis in range(2):
        for source in range(count):
            row = candidates[axis, source, valid[axis, source]]
            if len(row) != len(np.unique(row)):
                raise RuntimeError("FIT cache contains a duplicate candidate row")
    slots = np.asarray(arrays["target_slots"], dtype=np.int64)
    if np.any((slots < -1) | (slots >= width)):
        raise RuntimeError("FIT cache target slot is out of range")
    present = slots >= 0
    axis, source = np.nonzero(present)
    if len(axis) and not valid[axis, source, slots[present]].all():
        raise RuntimeError("FIT cache target points to an invalid candidate slot")


def _load_validated_fit_cache(path: Path) -> dict[str, np.ndarray]:
    values = prior._load_fit_cache(path)
    validate_fit_cache_arrays(values)
    return values


def _fit_cache_manifest(
    config: Mapping[str, Any], rosters: Mapping[str, tuple[str, ...]]
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    artifact = config["frozen_inputs"]["fit_cache_report"]
    path = _project_path(artifact["path"])
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "aiijc-tri-emitter-edge-verifier-report-v1":
        raise RuntimeError("FIT cache parent report schema changed")
    rows = tuple(report.get("fit_cache", {}).get("rows", ()))
    expected_draws = tuple(int(value) for value in config["source_protocol"]["fit_draw_indices"])
    observed_order = tuple((row["source_filename"], int(row["draw_index"])) for row in rows)
    expected_order = tuple((name, draw) for name in rosters["fit"] for draw in expected_draws)
    if observed_order != expected_order:
        raise RuntimeError("FIT cache cases do not match the signed FIT roster/draws")
    for row in rows:
        path = _project_path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"immutable FIT cache changed: {path}")
    return report, rows


def _truth_from_target_slots(values: Mapping[str, np.ndarray], axis: int) -> np.ndarray:
    slots = np.asarray(values["target_slots"][axis], dtype=np.int64)
    truth = np.full(len(slots), -1, dtype=np.int64)
    present = slots >= 0
    sources = np.flatnonzero(present)
    truth[sources] = values["candidates"][axis, sources, slots[present]]
    return truth


def _device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def _new_model(
    config: Mapping[str, Any], device: torch.device
) -> JointReciprocalTriEmitterVerifier:
    fixed = config["fixed_model"]
    return JointReciprocalTriEmitterVerifier(
        width=int(fixed["width"]), hidden=int(fixed["hidden"])
    ).to(device)


def _joint_outputs(
    model: JointReciprocalTriEmitterVerifier,
    values: Mapping[str, np.ndarray],
    *,
    device: torch.device,
) -> tuple[Any, Any]:
    case = prior._to_device_case(values, device)
    return tuple(
        model(
            case["raw_sides"],
            case["dino_sides"],
            case["candidates"][axis],
            case["valid"][axis],
            case["auxiliary"][axis],
            case["raw_baseline"][axis],
            direction=axis,
        )
        for axis in range(2)
    )


def _validate_real_checkpoint_payload(payload: Mapping[str, Any], path: Path) -> None:
    lowered = {part.lower() for part in path.parts}
    if any("capacity" in part for part in lowered):
        raise RuntimeError("synthetic capacity checkpoint cannot seed real FIT")
    if payload.get("capacity_only_not_reusable_for_real_fit") is True:
        raise RuntimeError("capacity-only checkpoint is forbidden for real FIT")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("checkpoint is not a real joint reciprocal FIT endpoint")
    training = payload.get("training", {})
    if training.get("from_scratch") is not True:
        raise RuntimeError("real joint reciprocal endpoint was not trained from scratch")
    if training.get("checkpoint_selection") != "single-final-endpoint-no-selection":
        raise RuntimeError("checkpoint selection contract changed")


def _load_real_checkpoint(
    path: Path,
    config: Mapping[str, Any],
    config_sha: str,
    *,
    device: torch.device,
) -> JointReciprocalTriEmitterVerifier:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _validate_real_checkpoint_payload(payload, path)
    if payload.get("config_sha256") != config_sha:
        raise RuntimeError("real FIT endpoint belongs to another signed protocol")
    model = _new_model(config, device)
    model.load_state_dict(payload["state_dict"], strict=True)
    if payload.get("contract") != joint_verifier_contract(model):
        raise RuntimeError("real FIT endpoint architecture contract changed")
    model.eval()
    return model


def fixed_cache_schedule(*, case_count: int, optimizer_updates: int, seed: int) -> tuple[int, ...]:
    """Cycle deterministic fresh permutations until the signed update count."""

    if case_count <= 0 or optimizer_updates <= 0:
        raise ValueError("case_count and optimizer_updates must be positive")
    generator = np.random.default_rng(seed + 1)
    order: list[int] = []
    while len(order) < optimizer_updates:
        order.extend(int(value) for value in generator.permutation(case_count))
    return tuple(order[:optimizer_updates])


def run_fit(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    config_sha: str,
    rosters: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    """Train exactly one final endpoint; no resume or capacity weights exist."""

    _, rows = _fit_cache_manifest(config, rosters)
    cache_values = [_load_validated_fit_cache(_project_path(row["path"])) for row in rows]
    fit_dir = args.experiment_dir.resolve() / "fit"
    fit_dir.mkdir(parents=True, exist_ok=False)
    training = config["training"]
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=bool(args.allow_nondeterministic_mps))
    torch.set_num_threads(int(training["torch_threads"]))
    device = _device(args.device)
    model = _new_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(training["epochs"])
    total_steps = int(training["optimizer_updates"])
    schedule = fixed_cache_schedule(case_count=len(rows), optimizer_updates=total_steps, seed=seed)
    schedule_sha256 = hashlib.sha256(np.asarray(schedule, dtype=np.int16).tobytes()).hexdigest()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        total_steps,
        eta_min=float(training["learning_rate"]) * 0.05,
    )
    history: list[dict[str, Any]] = []
    losses: list[float] = []
    started = perf_counter()
    model.train()
    for step, row_index in enumerate(schedule, start=1):
        # Nominal epoch is diagnostic only: joint full-board assignments
        # deliberately retain the audited 1,752 optimizer-update budget.
        epoch = min(epochs - 1, (step - 1) * epochs // total_steps)
        values = cache_values[row_index]
        outputs = _joint_outputs(model, values, device=device)
        axis_losses = []
        for axis, output in enumerate(outputs):
            candidates = torch.from_numpy(values["candidates"][axis].astype(np.int64)).to(device)
            valid = torch.from_numpy(values["valid"][axis]).to(device)
            truth = torch.from_numpy(_truth_from_target_slots(values, axis)).to(device)
            targets = exact_joint_targets(candidates, valid, truth)
            axis_losses.append(joint_assignment_loss(output, targets, valid))
        loss = torch.stack([value.total for value in axis_losses]).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite training loss at optimizer step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(training["gradient_clip"]),
                error_if_nonfinite=True,
            )
        )
        optimizer.step()
        scheduler.step()
        if device.type == "mps":
            torch.mps.synchronize()
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % 25 == 0 or step == total_steps:
            current = {
                "step": step,
                "epoch": epoch,
                "loss": float(np.mean(losses[-min(25, len(losses)) :])),
                "row_cross_entropy": float(
                    torch.stack([value.row_cross_entropy for value in axis_losses])
                    .mean()
                    .detach()
                    .cpu()
                ),
                "column_cross_entropy": float(
                    torch.stack([value.column_cross_entropy for value in axis_losses])
                    .mean()
                    .detach()
                    .cpu()
                ),
                "confidence_bce": float(
                    torch.stack([value.confidence_bce for value in axis_losses])
                    .mean()
                    .detach()
                    .cpu()
                ),
                "delta_regularization": float(
                    torch.stack([value.delta_regularization for value in axis_losses])
                    .mean()
                    .detach()
                    .cpu()
                ),
                "grad_norm": grad_norm,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": perf_counter() - started,
            }
            history.append(current)
            print(json.dumps({"event": "joint_fit", **current}), flush=True)
    endpoint = args.experiment_dir.resolve() / FIT_ENDPOINT
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "contract": joint_verifier_contract(model),
            "config_sha256": config_sha,
            "capacity_only_not_reusable_for_real_fit": False,
            "training": {
                "from_scratch": True,
                "capacity_checkpoint_loaded": False,
                "seed": seed,
                "epochs": epochs,
                "optimizer_updates": total_steps,
                "steps": total_steps,
                "cache_schedule_sha256": schedule_sha256,
                "checkpoint_selection": "single-final-endpoint-no-selection",
                "history": history,
            },
            "selection": {
                "fit_filenames": list(rosters["fit"]),
                "fit_digest": config["source_protocol"]["fit_digest"],
                "dev_filenames_seen": [],
            },
        },
        endpoint,
    )
    report = {
        "schema": "aiijc-joint-reciprocal-real-fit-report-v1",
        "status": "complete-single-endpoint-ready-for-target-free-dev-freeze",
        "config_sha256": config_sha,
        "cache_case_count": len(rows),
        "fit_filenames": list(rosters["fit"]),
        "fit_draw_indices": config["source_protocol"]["fit_draw_indices"],
        "training": {
            "from_scratch": True,
            "capacity_checkpoint_loaded": False,
            "single_final_endpoint": True,
            "fit_caches_preloaded_once": True,
            "steps": total_steps,
            "cache_schedule_sha256": schedule_sha256,
            "seconds": perf_counter() - started,
            "initial_loss": losses[0],
            "final_25_mean_loss": float(np.mean(losses[-25:])),
            "history": history,
        },
        "endpoint": _record(endpoint),
        "dev_pixels_or_labels_opened": False,
        "terminal16_opened": False,
        "competition_test_accessed": False,
        "weco_logged": False,
    }
    _write_json(args.experiment_dir.resolve() / FIT_REPORT, report)
    return report


@torch.inference_mode()
def _freeze_fit_head_case(
    model: JointReciprocalTriEmitterVerifier,
    values: Mapping[str, np.ndarray],
    *,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Freeze inference-visible fixed heads without copying FIT target slots."""

    target_free = {key: value for key, value in values.items() if key != "target_slots"}
    before = candidate_pool_digest(
        target_free["candidates"], target_free["valid"], target_free["emitter_topk"]
    )
    outputs = _joint_outputs(model, target_free, device=device)
    after = candidate_pool_digest(
        target_free["candidates"], target_free["valid"], target_free["emitter_topk"]
    )
    if before != after:
        raise RuntimeError("joint verifier mutated FIT candidate identities")
    arrays: dict[str, np.ndarray] = {
        "union_identity_digest_ascii": np.frombuffer(before.encode(), dtype=np.uint8)
    }
    for axis, output in enumerate(outputs):
        axis_name = ("right", "down")[axis]
        head = fixed_fraction_reciprocal_head(
            output,
            target_free["candidates"][axis],
            target_free["valid"][axis],
        )
        if len(head.sources) != head.requested_count:
            raise RuntimeError(f"FIT {axis_name} reciprocal head cannot fill fixed 5% coverage")
        arrays[f"selected_sources__{axis_name}"] = head.sources.astype(np.int32)
        arrays[f"selected_targets__{axis_name}"] = head.targets.astype(np.int32)
        arrays[f"selected_joint_confidences__{axis_name}"] = head.confidences.astype(np.float32)
        arrays[f"requested_count__{axis_name}"] = np.asarray(head.requested_count, dtype=np.int32)
        arrays[f"reciprocal_count__{axis_name}"] = np.asarray(
            int(head.reciprocal.sum()), dtype=np.int32
        )
    return arrays


def run_freeze_fit_heads(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    config_sha: str,
    rosters: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    """Freeze the real endpoint's 5% FIT heads without label/truth arrays."""

    experiment = args.experiment_dir.resolve()
    fit_report = json.loads((experiment / FIT_REPORT).read_text(encoding="utf-8"))
    if fit_report.get("config_sha256") != config_sha:
        raise RuntimeError("FIT report belongs to another signed protocol")
    _, rows = _fit_cache_manifest(config, rosters)
    endpoint = experiment / FIT_ENDPOINT
    device = _device(args.device)
    model = _load_real_checkpoint(endpoint, config, config_sha, device=device)
    arrays: dict[str, np.ndarray] = {}
    metadata_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        cache_path = _project_path(row["path"])
        values = _load_validated_fit_cache(cache_path)
        frozen = _freeze_fit_head_case(model, values, device=device)
        prefix = f"case_{index:04d}"
        arrays.update({f"{prefix}__{key}": value for key, value in frozen.items()})
        digest = bytes(frozen["union_identity_digest_ascii"]).decode()
        metadata_rows.append(
            {
                "prefix": prefix,
                "case_id": row["case_id"],
                "source_filename": row["source_filename"],
                "draw_index": int(row["draw_index"]),
                "dirty_sha256": row["dirty_sha256"],
                "fit_cache": {"path": row["path"], "sha256": row["sha256"]},
                "union_identity_digest": digest,
            }
        )
        print(
            json.dumps(
                {
                    "event": "joint_freeze_fit_head",
                    "case": index + 1,
                    "count": len(rows),
                    "source": row["source_filename"],
                    "draw": row["draw_index"],
                }
            ),
            flush=True,
        )
    archive = experiment / FIT_HEAD_ARCHIVE
    metadata = experiment / FIT_HEAD_METADATA
    freeze = experiment / FIT_HEAD_PRE_SCORE_FREEZE
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-joint-reciprocal-target-free-fit-heads-v1",
            "config_sha256": config_sha,
            "contains_target_slots_truth_or_reference_labels": False,
            "contains_pixels": False,
            "tile_id_space": "immutable-shuffled-tile-bag-identity",
            "candidate_identities_immutable": True,
            "fixed_fraction_per_axis_per_board": RECIPROCAL_HEAD_FRACTION,
            "expected_requested_count_for_576_tiles": math.ceil(RECIPROCAL_HEAD_FRACTION * 576),
            "rows": metadata_rows,
        },
    )
    _write_json(
        freeze,
        {
            "schema": "aiijc-joint-reciprocal-fit-heads-pre-score-freeze-v1",
            "created_before_fit_head_label_scoring": True,
            "contains_target_slots_truth_or_reference_labels": False,
            "config_sha256": config_sha,
            "artifacts": {
                "archive": _record(archive),
                "metadata": _record(metadata),
                "fit_endpoint": _record(endpoint),
                "runner": _record(Path(__file__)),
                "module": _record(
                    PROJECT_ROOT / "src/aiijc_puzzle/joint_reciprocal_tri_emitter_verifier.py"
                ),
            },
        },
    )
    return {
        "schema": "aiijc-joint-reciprocal-fit-heads-freeze-result-v1",
        "status": "target-free-fit-heads-frozen-label-scoring-not-run",
        "case_count": len(rows),
        "archive": _record(archive),
        "metadata": _record(metadata),
        "pre_score_freeze": _record(freeze),
        "fit_head_labels_scored": False,
        "dev_pixels_or_labels_opened": False,
        "terminal16_opened": False,
        "competition_test_accessed": False,
    }


def verify_fit_head_freeze(experiment_dir: Path, config_sha: str) -> VerifiedFreeze:
    experiment = experiment_dir.resolve()
    freeze_path = experiment / FIT_HEAD_PRE_SCORE_FREEZE
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema") != "aiijc-joint-reciprocal-fit-heads-pre-score-freeze-v1":
        raise RuntimeError("FIT-head pre-score freeze schema changed")
    if freeze.get("created_before_fit_head_label_scoring") is not True:
        raise RuntimeError("FIT heads were not frozen before label scoring")
    if freeze.get("contains_target_slots_truth_or_reference_labels") is not False:
        raise RuntimeError("FIT-head pre-score freeze unexpectedly contains labels")
    if freeze.get("config_sha256") != config_sha:
        raise RuntimeError("FIT-head freeze belongs to another protocol")
    archive = experiment / FIT_HEAD_ARCHIVE
    metadata = experiment / FIT_HEAD_METADATA
    expected = freeze["artifacts"]
    for key, path in (("archive", archive), ("metadata", metadata)):
        if not path.is_file() or sha256_file(path) != expected[key]["sha256"]:
            raise RuntimeError(f"target-free FIT-head {key} changed after freeze")
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    if payload.get("contains_target_slots_truth_or_reference_labels") is not False:
        raise RuntimeError("target-free FIT-head metadata contains labels")
    if payload.get("config_sha256") != config_sha:
        raise RuntimeError("target-free FIT-head metadata belongs to another protocol")
    return VerifiedFreeze(
        panel_dir=experiment / "fit",
        archive=archive,
        metadata=metadata,
        archive_sha256=sha256_file(archive),
        metadata_sha256=sha256_file(metadata),
        rows=tuple(payload["rows"]),
    )


def run_score_fit_heads(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    config_sha: str,
    rosters: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    """Score FIT precision only after the target-free freeze is verified."""

    experiment = args.experiment_dir.resolve()
    verified = verify_fit_head_freeze(experiment, config_sha)
    _, cache_rows = _fit_cache_manifest(config, rosters)
    if len(verified.rows) != len(cache_rows):
        raise RuntimeError("FIT-head freeze and immutable cache have different case counts")
    totals = {axis: {"selected": 0, "correct": 0, "requested": 0} for axis in ("right", "down")}
    with np.load(verified.archive, allow_pickle=False) as archive:
        for metadata, row in zip(verified.rows, cache_rows, strict=True):
            identity = (
                row["case_id"],
                row["source_filename"],
                int(row["draw_index"]),
                row["dirty_sha256"],
                row["sha256"],
            )
            observed = (
                metadata["case_id"],
                metadata["source_filename"],
                int(metadata["draw_index"]),
                metadata["dirty_sha256"],
                metadata["fit_cache"]["sha256"],
            )
            if observed != identity:
                raise RuntimeError("FIT-head metadata/cache identity mismatch")
            values = _load_validated_fit_cache(_project_path(row["path"]))
            prefix = metadata["prefix"]
            for axis_index, axis in enumerate(("right", "down")):
                sources = archive[f"{prefix}__selected_sources__{axis}"]
                targets = archive[f"{prefix}__selected_targets__{axis}"]
                requested = int(archive[f"{prefix}__requested_count__{axis}"])
                truth = _truth_from_target_slots(values, axis_index)
                totals[axis]["selected"] += len(sources)
                totals[axis]["requested"] += requested
                totals[axis]["correct"] += int(np.count_nonzero(targets == truth[sources]))
    for axis in ("right", "down"):
        current = totals[axis]
        current["precision"] = (
            current["correct"] / current["selected"] if current["selected"] else 0.0
        )
        current["coverage_complete"] = current["selected"] == current["requested"]
    pooled_selected = sum(totals[axis]["selected"] for axis in ("right", "down"))
    pooled_correct = sum(totals[axis]["correct"] for axis in ("right", "down"))
    pooled_requested = sum(totals[axis]["requested"] for axis in ("right", "down"))
    report = {
        "schema": "aiijc-joint-reciprocal-fit-heads-score-v1",
        "status": "complete-fit-diagnostic-only",
        "config_sha256": config_sha,
        "target_freeze_verified_before_fit_label_loading": True,
        "fixed_5_percent_head": {
            **totals,
            "pooled": {
                "selected": pooled_selected,
                "correct": pooled_correct,
                "requested": pooled_requested,
                "precision": pooled_correct / pooled_selected if pooled_selected else 0.0,
                "coverage_complete": pooled_selected == pooled_requested,
            },
        },
        "artifacts": {
            "target_free_archive": _record(verified.archive),
            "target_free_metadata": _record(verified.metadata),
            "pre_score_freeze": _record(experiment / FIT_HEAD_PRE_SCORE_FREEZE),
        },
        "dev_pixels_or_labels_opened": False,
        "terminal16_opened": False,
        "competition_test_accessed": False,
        "promotion_claim": False,
    }
    _write_json(experiment / FIT_HEAD_SCORE, report)
    return report


def _manifest_records(manifest_path: Path, filenames: Sequence[str]) -> tuple[dict[str, Any], ...]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise RuntimeError("validation manifest protocol digest mismatch")
    by_name = {str(record["filename"]): record for record in manifest["splits"]["train"]}
    try:
        return tuple(by_name[name] for name in filenames)
    except KeyError as error:
        raise RuntimeError("signed DEV source is absent from organizer-train manifest") from error


def make_target_free_synthetic_case(
    clean_tiles: np.ndarray,
    *,
    source_filename: str,
    draw_index: int,
    seed: int,
) -> SyntheticSocketInput:
    """Create the dirty tile bag without constructing the inverse-shuffle label."""

    clean = np.asarray(clean_tiles)
    if clean.ndim != 4 or clean.shape[1:] != (20, 20, 3) or clean.dtype != np.uint8:
        raise ValueError("clean_tiles must be uint8 N x 20 x 20 x 3")
    grid = round(len(clean) ** 0.5)
    if grid < 2 or grid * grid != len(clean):
        raise ValueError("clean tile count must be a square board with grid >= 2")
    if not source_filename:
        raise ValueError("source_filename must be non-empty")
    digest = hashlib.sha256(
        f"{DEFAULT_SYNTHETIC_NAMESPACE}\0{seed}\0{source_filename}\0{draw_index}".encode()
    ).digest()
    corruption_seed = int.from_bytes(digest[:8], "little")
    permutation_seed = int.from_bytes(digest[8:16], "little")
    corrupted = distort_tiles(clean, np.random.default_rng(corruption_seed))
    permutation = np.random.default_rng(permutation_seed).permutation(len(clean))
    case_digest = hashlib.sha256(f"{source_filename}\0{draw_index}\0{seed}".encode()).hexdigest()[
        :16
    ]
    return SyntheticSocketInput(
        case_id=f"synthetic-{case_digest}",
        source_filename=source_filename,
        draw_index=draw_index,
        corruption_seed=corruption_seed,
        permutation_seed=permutation_seed,
        tiles=np.ascontiguousarray(corrupted[permutation]),
    )


def _fixed_raw_head(scores: np.ndarray) -> dict[str, np.ndarray | int]:
    evidence = boundary._reciprocal_evidence(scores)
    count = len(scores)
    requested = max(1, math.ceil(RECIPROCAL_HEAD_FRACTION * count))
    sources = np.flatnonzero(evidence["reciprocal"])
    entries = sorted(
        (
            float(evidence["confidence"][source]),
            int(source),
            int(evidence["target"][source]),
        )
        for source in sources
    )
    entries.sort(key=lambda value: (-value[0], value[1], value[2]))
    chosen = entries[:requested]
    return {
        "sources": np.asarray([value[1] for value in chosen], dtype=np.int32),
        "targets": np.asarray([value[2] for value in chosen], dtype=np.int32),
        "confidences": np.asarray([value[0] for value in chosen], dtype=np.float32),
        "requested_count": requested,
        "reciprocal_count": len(entries),
    }


@torch.inference_mode()
def _freeze_one_case(
    model: JointReciprocalTriEmitterVerifier,
    values: Mapping[str, np.ndarray],
    *,
    device: torch.device,
) -> dict[str, np.ndarray]:
    before = candidate_pool_digest(values["candidates"], values["valid"], values["emitter_topk"])
    outputs = _joint_outputs(model, values, device=device)
    after = candidate_pool_digest(values["candidates"], values["valid"], values["emitter_topk"])
    if before != after:
        raise RuntimeError("joint verifier mutated frozen candidate identities")
    arrays: dict[str, np.ndarray] = {
        "union_candidates": values["candidates"].astype(np.int32),
        "union_valid": values["valid"].astype(bool),
        "emitter_topk": values["emitter_topk"].astype(np.int32),
        "raw_top32": values["emitter_topk"][0].astype(np.int32),
    }
    learned_top32 = []
    for axis, output in enumerate(outputs):
        logits = output.edge_logits.float().cpu().numpy()
        learned_top32.append(
            prior._rank_topk(
                values["candidates"][axis],
                values["valid"][axis],
                logits,
                k=TOP_K,
            )
        )
        learned_head = fixed_fraction_reciprocal_head(
            output, values["candidates"][axis], values["valid"][axis]
        )
        raw_head = _fixed_raw_head(values["raw_dense"][axis])
        axis_name = ("right", "down")[axis]
        arrays[f"learned_logits__{axis_name}"] = logits.astype(np.float32)
        arrays[f"learned_joint_confidence__{axis_name}"] = (
            output.joint_confidence.float().cpu().numpy().astype(np.float32)
        )
        for prefix, head in (("raw", raw_head), ("learned", learned_head)):
            if prefix == "raw":
                sources = head["sources"]
                targets = head["targets"]
                confidences = head["confidences"]
                requested = int(head["requested_count"])
                reciprocal_count = int(head["reciprocal_count"])
            else:
                sources = head.sources
                targets = head.targets
                confidences = head.confidences
                requested = head.requested_count
                reciprocal_count = int(head.reciprocal.sum())
            arrays[f"{prefix}_head_sources__{axis_name}"] = np.asarray(sources, dtype=np.int32)
            arrays[f"{prefix}_head_targets__{axis_name}"] = np.asarray(targets, dtype=np.int32)
            arrays[f"{prefix}_head_confidences__{axis_name}"] = np.asarray(
                confidences, dtype=np.float32
            )
            arrays[f"{prefix}_head_requested__{axis_name}"] = np.asarray(requested, dtype=np.int32)
            arrays[f"{prefix}_reciprocal_count__{axis_name}"] = np.asarray(
                reciprocal_count, dtype=np.int32
            )
    arrays["learned_top32"] = np.stack(learned_top32).astype(np.int32)
    arrays["union_identity_digest_ascii"] = np.frombuffer(before.encode(), dtype=np.uint8)
    return arrays


def run_freeze_dev(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    config_sha: str,
    rosters: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    experiment = args.experiment_dir.resolve()
    fit_report_path = experiment / FIT_REPORT
    fit_report = json.loads(fit_report_path.read_text(encoding="utf-8"))
    endpoint = experiment / FIT_ENDPOINT
    if fit_report.get("config_sha256") != config_sha:
        raise RuntimeError("FIT report belongs to another signed protocol")
    device = _device(args.device)
    model = _load_real_checkpoint(endpoint, config, config_sha, device=device)
    records = _manifest_records(args.manifest, rosters["dev"])
    boards = boundary._prepare_boards(records, args.targets)
    expected_socket = config["frozen_inputs"].get("socket_checkpoint")
    if expected_socket is None or sha256_file(args.socket_checkpoint) != expected_socket["sha256"]:
        raise RuntimeError("Socket checkpoint differs from signed real protocol")
    socket, adapter, dino, projection, feature_device = prior._make_models(args)
    if feature_device != device:
        raise RuntimeError("feature and verifier devices diverged")
    panel = experiment / DEV_PANEL
    panel.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    draw_index = int(config["source_protocol"]["dev_draw_index"])
    case_seed = int(config["source_protocol"]["dev_case_seed"])
    with ThreadPoolExecutor(max_workers=1) as executor:
        for index, board in enumerate(boards):
            item = make_target_free_synthetic_case(
                board.tiles,
                source_filename=board.filename,
                draw_index=draw_index,
                seed=case_seed,
            )
            values, runtime = prior._extract_case(
                item,
                socket=socket,
                adapter=adapter,
                dino=dino,
                projection=projection,
                device=device,
                executor=executor,
            )
            frozen = _freeze_one_case(model, values, device=device)
            frozen_digest = bytes(frozen["union_identity_digest_ascii"]).decode()
            if frozen_digest != runtime["union_identity_digest"]:
                raise RuntimeError("feature runtime and frozen union identity diverged")
            prefix = f"case_{index:04d}"
            arrays.update({f"{prefix}__{key}": value for key, value in frozen.items()})
            rows.append(
                {
                    "prefix": prefix,
                    "case_id": item.case_id,
                    "source_filename": item.source_filename,
                    "draw_index": item.draw_index,
                    "dirty_sha256": hashlib.sha256(item.tiles.tobytes()).hexdigest(),
                    "union_identity_digest": runtime["union_identity_digest"],
                    "runtime": runtime,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "joint_freeze_dev",
                        "case": index + 1,
                        "count": len(boards),
                        "source": board.filename,
                    }
                ),
                flush=True,
            )
    archive = panel / FREEZE_ARCHIVE
    metadata = panel / FREEZE_METADATA
    freeze = panel / PRE_SCORE_FREEZE
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-joint-reciprocal-target-free-dev-v1",
            "config_sha256": config_sha,
            "contains_exact_references_or_labels": False,
            "contains_clean_dirty_or_output_pixels": False,
            "candidate_identities_immutable": True,
            "fixed_fraction_per_axis_per_board": RECIPROCAL_HEAD_FRACTION,
            "rows": rows,
        },
    )
    _write_json(
        freeze,
        {
            "schema": "aiijc-joint-reciprocal-pre-score-freeze-v1",
            "created_before_exact_reference_scoring": True,
            "contains_exact_references_or_labels": False,
            "config_sha256": config_sha,
            "artifacts": {
                "archive": _record(archive),
                "metadata": _record(metadata),
                "fit_endpoint": _record(endpoint),
                "runner": _record(Path(__file__)),
                "module": _record(
                    PROJECT_ROOT / "src/aiijc_puzzle/joint_reciprocal_tri_emitter_verifier.py"
                ),
            },
        },
    )
    return {
        "schema": "aiijc-joint-reciprocal-freeze-result-v1",
        "status": "target-free-dev-frozen-label-scoring-not-run",
        "case_count": len(rows),
        "archive": _record(archive),
        "metadata": _record(metadata),
        "pre_score_freeze": _record(freeze),
        "dev_labels_scored": False,
        "terminal16_opened": False,
        "competition_test_accessed": False,
    }


def verify_pre_score_freeze(panel_dir: Path, config_sha: str) -> VerifiedFreeze:
    """Verify all target-free hashes before a caller is allowed to load labels."""

    panel = panel_dir.resolve()
    freeze_path = panel / PRE_SCORE_FREEZE
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema") != "aiijc-joint-reciprocal-pre-score-freeze-v1":
        raise RuntimeError("pre-score freeze schema changed")
    if freeze.get("created_before_exact_reference_scoring") is not True:
        raise RuntimeError("target-free freeze was not created before scoring")
    if freeze.get("contains_exact_references_or_labels") is not False:
        raise RuntimeError("pre-score freeze unexpectedly contains labels")
    if freeze.get("config_sha256") != config_sha:
        raise RuntimeError("pre-score freeze belongs to another protocol")
    archive = panel / FREEZE_ARCHIVE
    metadata = panel / FREEZE_METADATA
    expected = freeze["artifacts"]
    for key, path in (("archive", archive), ("metadata", metadata)):
        if not path.is_file() or sha256_file(path) != expected[key]["sha256"]:
            raise RuntimeError(f"target-free {key} changed after freeze")
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    if payload.get("contains_exact_references_or_labels") is not False:
        raise RuntimeError("target-free metadata contains labels")
    if payload.get("config_sha256") != config_sha:
        raise RuntimeError("target-free metadata belongs to another protocol")
    return VerifiedFreeze(
        panel_dir=panel,
        archive=archive,
        metadata=metadata,
        archive_sha256=sha256_file(archive),
        metadata_sha256=sha256_file(metadata),
        rows=tuple(payload["rows"]),
    )


def score_after_verified_freeze(
    panel_dir: Path,
    config_sha: str,
    *,
    reference_loader: Callable[[VerifiedFreeze], T],
    scorer: Callable[[VerifiedFreeze, T], R],
) -> R:
    """Dependency-injected sequencing helper used by the real scorer and tests."""

    verified = verify_pre_score_freeze(panel_dir, config_sha)
    references = reference_loader(verified)
    return scorer(verified, references)


def _load_dev_references(
    verified: VerifiedFreeze,
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    rosters: Mapping[str, tuple[str, ...]],
) -> dict[str, ExactSyntheticReference]:
    records = _manifest_records(args.manifest, rosters["dev"])
    boards = boundary._prepare_boards(records, args.targets)
    if [row["source_filename"] for row in verified.rows] != list(rosters["dev"]):
        raise RuntimeError("frozen DEV order differs from signed roster")
    references: dict[str, ExactSyntheticReference] = {}
    draw_index = int(config["source_protocol"]["dev_draw_index"])
    case_seed = int(config["source_protocol"]["dev_case_seed"])
    for board, row in zip(boards, verified.rows, strict=True):
        item, reference = make_exact_synthetic_case(
            board.tiles,
            source_filename=board.filename,
            draw_index=draw_index,
            seed=case_seed,
        )
        observed = {
            "case_id": item.case_id,
            "source_filename": item.source_filename,
            "draw_index": item.draw_index,
            "dirty_sha256": hashlib.sha256(item.tiles.tobytes()).hexdigest(),
        }
        if any(row[key] != value for key, value in observed.items()):
            raise RuntimeError("reference reconstruction differs from target-free freeze")
        references[item.case_id] = reference
    return references


def _score_frozen_panel(
    verified: VerifiedFreeze,
    references: Mapping[str, ExactSyntheticReference],
) -> dict[str, Any]:
    totals = {
        variant: {
            f"{axis}_{field}": 0
            for axis in ("right", "down", "pooled")
            for field in ("total", "hits_at_1", "hits_at_5")
        }
        for variant in ("raw_d64_ot", "joint_reciprocal")
    }
    heads = {
        variant: {axis: {"selected": 0, "correct": 0, "requested": 0} for axis in ("right", "down")}
        for variant in ("raw_d64_ot", "joint_reciprocal")
    }
    identity_unchanged = True
    raw_top32_preserved = True
    with np.load(verified.archive, allow_pickle=False) as archive:
        for row in verified.rows:
            prefix = row["prefix"]
            reference = references[row["case_id"]].tile_at_position
            digest = bytes(archive[f"{prefix}__union_identity_digest_ascii"]).decode()
            identity_unchanged &= digest == row["union_identity_digest"]
            union = archive[f"{prefix}__union_candidates"]
            union_valid = archive[f"{prefix}__union_valid"]
            raw_top32 = archive[f"{prefix}__raw_top32"]
            for axis in range(2):
                for source in range(len(raw_top32[axis])):
                    available = union[axis, source, union_valid[axis, source]]
                    raw_top32_preserved &= bool(np.isin(raw_top32[axis, source], available).all())
            for variant, key in (
                ("raw_d64_ot", "raw_top32"),
                ("joint_reciprocal", "learned_top32"),
            ):
                candidates = archive[f"{prefix}__{key}"]
                metrics = exact_local_retrieval_metrics(
                    candidates[0], candidates[1], reference, ks=(1, 5)
                )
                for metric in totals[variant]:
                    totals[variant][metric] += int(metrics[metric])
            for axis_index, axis in enumerate(("right", "down")):
                truth = prior._truth_by_anchor(reference, axis=axis_index)
                for variant, prefix_name in (
                    ("raw_d64_ot", "raw"),
                    ("joint_reciprocal", "learned"),
                ):
                    sources = archive[f"{prefix}__{prefix_name}_head_sources__{axis}"]
                    targets = archive[f"{prefix}__{prefix_name}_head_targets__{axis}"]
                    requested = int(archive[f"{prefix}__{prefix_name}_head_requested__{axis}"])
                    heads[variant][axis]["selected"] += len(sources)
                    heads[variant][axis]["requested"] += requested
                    heads[variant][axis]["correct"] += int(
                        np.count_nonzero(targets == truth[sources])
                    )
    for variant in totals:
        for axis in ("right", "down", "pooled"):
            total = totals[variant][f"{axis}_total"]
            for k in (1, 5):
                totals[variant][f"{axis}_r{k}"] = totals[variant][f"{axis}_hits_at_{k}"] / total
    for variant in heads:
        for axis in ("right", "down"):
            current = heads[variant][axis]
            current["precision"] = (
                current["correct"] / current["selected"] if current["selected"] else 0.0
            )
            current["coverage_complete"] = current["selected"] == current["requested"]
        selected = sum(heads[variant][axis]["selected"] for axis in ("right", "down"))
        correct = sum(heads[variant][axis]["correct"] for axis in ("right", "down"))
        requested = sum(heads[variant][axis]["requested"] for axis in ("right", "down"))
        heads[variant]["pooled"] = {
            "selected": selected,
            "correct": correct,
            "requested": requested,
            "precision": correct / selected if selected else 0.0,
            "coverage_complete": selected == requested,
        }
    return {
        "case_count": len(verified.rows),
        "retrieval": totals,
        "fixed_5_percent_reciprocal_head": heads,
        "union": {
            "identities_unchanged": identity_unchanged,
            "raw_top32_preserved": raw_top32_preserved,
            "coverage_nonregression": identity_unchanged and raw_top32_preserved,
        },
        "freeze": {
            "archive_sha256": verified.archive_sha256,
            "metadata_sha256": verified.metadata_sha256,
            "verified_before_reference_loading": True,
        },
    }


def joint_discovery_gate(
    metrics: Mapping[str, Any], thresholds: Mapping[str, float]
) -> dict[str, Any]:
    """Apply the preregistered pooled and per-axis non-regression gate."""

    retrieval = metrics["retrieval"]
    raw = retrieval["raw_d64_ot"]
    learned = retrieval["joint_reciprocal"]
    heads = metrics["fixed_5_percent_reciprocal_head"]
    gains = {
        f"{axis}_r{k}": learned[f"{axis}_r{k}"] - raw[f"{axis}_r{k}"]
        for axis in ("right", "down", "pooled")
        for k in (1, 5)
    }
    head_gains = {
        axis: heads["joint_reciprocal"][axis]["precision"] - heads["raw_d64_ot"][axis]["precision"]
        for axis in ("right", "down", "pooled")
    }
    coverage_complete = all(
        heads[variant][axis]["coverage_complete"]
        for variant in ("raw_d64_ot", "joint_reciprocal")
        for axis in ("right", "down", "pooled")
    )
    per_axis_minimum = float(thresholds["per_axis_gain_minimum"])
    per_axis_nonnegative = all(
        gains[f"{axis}_r{k}"] >= per_axis_minimum for axis in ("right", "down") for k in (1, 5)
    ) and all(head_gains[axis] >= per_axis_minimum for axis in ("right", "down"))
    union_nonregression = all(
        bool(metrics["union"][key])
        for key in (
            "identities_unchanged",
            "raw_top32_preserved",
            "coverage_nonregression",
        )
    )
    passed = bool(
        gains["pooled_r1"] >= float(thresholds["pooled_r1_gain_minimum"])
        and gains["pooled_r5"] >= float(thresholds["pooled_r5_gain_minimum"])
        and head_gains["pooled"] >= float(thresholds["pooled_fixed_head_precision_gain_minimum"])
        and per_axis_nonnegative
        and coverage_complete
        and union_nonregression
    )
    return {
        "retrieval_gains": gains,
        "fixed_head_precision_gains": head_gains,
        "fixed_head_coverage_complete": coverage_complete,
        "each_axis_nonnegative": per_axis_nonnegative,
        "union_identity_and_coverage_nonregression": union_nonregression,
        "thresholds": dict(thresholds),
        "passed": passed,
    }


def run_score_dev(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    config_sha: str,
    rosters: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    panel = args.experiment_dir.resolve() / DEV_PANEL
    metrics = score_after_verified_freeze(
        panel,
        config_sha,
        reference_loader=lambda verified: _load_dev_references(
            verified, args=args, config=config, rosters=rosters
        ),
        scorer=_score_frozen_panel,
    )
    gate = joint_discovery_gate(metrics, config["dev_gate"])
    report = {
        "schema": "aiijc-joint-reciprocal-real-dev-score-v1",
        "status": "pass-emitter-eligible-decoder-still-separate" if gate["passed"] else "fail-stop",
        "config_sha256": config_sha,
        "metrics": metrics,
        "gate": gate,
        "target_freeze_verified_before_labels": True,
        "terminal16_opened": False,
        "decoder_run": False,
        "competition_test_accessed": False,
        "submission_or_production_modified": False,
        "weco_logged": False,
    }
    _write_json(panel / DEV_SCORE, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config, config_sha = _load_signed_config(args.config)
    rosters = validate_source_rosters(config)
    if args.mode == "fit":
        report = run_fit(args, config, config_sha, rosters)
    elif args.mode == "freeze-fit-heads":
        report = run_freeze_fit_heads(args, config, config_sha, rosters)
    elif args.mode == "score-fit-heads":
        report = run_score_fit_heads(args, config, config_sha, rosters)
    elif args.mode == "freeze-dev":
        report = run_freeze_dev(args, config, config_sha, rosters)
    else:
        report = run_score_dev(args, config, config_sha, rosters)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.mode == "score-dev" and not report["gate"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
