"""Frozen, resumable trainer for the E26 contextual directional edge network.

The production protocol is intentionally narrow:

* authenticate the already frozen 7,000-row source-group manifest;
* admit exactly ``img_000000.png`` .. ``img_006699.png`` and require 6,700
  distinct source groups;
* deterministically divide them into FIT/CAL/DEV (5,360/670/670);
* train only on FIT for eight fixed epochs, with stateless corruption and source
  order; and
* publish recovery checkpoints at optimizer boundaries and one fixed-final
  scientific checkpoint.  Validation is never used to select a checkpoint.

Importing this module performs no data access and creates no directories.  A
production invocation fails closed unless every mutable/cache path is on E:.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

# Do this before importing project modules.  The autonomous launcher also points
# PYTHONPYCACHEPREFIX at E:, but the trainer must remain safe when inspected.
sys.dont_write_bytecode = True

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from e26_contextual_edge_net import (
    CHECKPOINT_SCHEMA,
    ContextualDirectionalEdgeNet,
    ContextualEdgeConfig,
    boundary_reconstruction_loss,
    clean_boundary_targets,
    directional_neighbour_labels,
    listwise_directional_ce,
    model_config_dict,
    model_from_checkpoint_payload,
)


TRAIN_SOURCE_COUNT = 6_700
FIT_SOURCE_COUNT = 5_360
CAL_SOURCE_COUNT = 670
DEV_SOURCE_COUNT = 670
FROZEN_EPOCHS = 8
FROZEN_BATCH_SIZE = 1
FROZEN_ACCUMULATE = 8
FROZEN_SEED = 2_601
RECOVERY_INTERVAL = 100
TOTAL_OPTIMIZER_STEPS = FROZEN_EPOCHS * (FIT_SOURCE_COUNT // FROZEN_ACCUMULATE)

DEFAULT_OUTPUT_DIR = Path("E:/pazzle_work/e26_contextual_edge")
DEFAULT_SOURCE_MANIFEST = Path("E:/pazzle_work/rank96_e11_v3/source_groups_v3.json")
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "fa142c5f9c4fa17671b60d72b9acedff0eafcad4e77afac2b17a9649adfbfbd9"
)
EXPECTED_SOURCE_MAPPING_SHA256 = (
    "62e32bc6b8b9ae320abec9db8cfebe263a30cf476b761887bbf902bedbeabde0"
)
SOURCE_MAPPING_SERIALIZATION = (
    "utf8-name-tab-source_group-lf-every-record-including-final-v1"
)
SPLIT_SCHEMA = "pazzle-e26-source-disjoint-split-v1"
TRAINING_SCHEMA = "pazzle-e26-contextual-edge-training-v2"
RECOVERY_SCHEMA = "pazzle-e26-contextual-edge-recovery-v1"
REPORT_SCHEMA = "pazzle-e26-contextual-edge-training-report-v1"

SPLIT_PREFIX = "E26-split-v1"
RNG_PREFIX = "E26-rng-v1"
ORDER_PREFIX = "E26-edge-order-v1"
TAG_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,79}\Z")
REQUIRED_E_ENV = (
    "TEMP",
    "TMP",
    "PYTHONPYCACHEPREFIX",
    "TORCH_EXTENSIONS_DIR",
    "CUDA_CACHE_PATH",
)
OPTIONAL_E_ENV = (
    "TMPDIR",
    "JOBLIB_TEMP_FOLDER",
    "TORCH_HOME",
    "HF_HOME",
    "XDG_CACHE_HOME",
)


@dataclass(frozen=True)
class AuthenticatedTrainingSources:
    names: tuple[str, ...]
    group_for_name: Mapping[str, str]
    content_sha256_for_name: Mapping[str, str]
    manifest_sha256: str
    mapping_sha256: str


@dataclass(frozen=True)
class SourceGroupSplit:
    fit_names: tuple[str, ...]
    calibration_names: tuple[str, ...]
    development_names: tuple[str, ...]
    fit_groups: tuple[str, ...]
    calibration_groups: tuple[str, ...]
    development_groups: tuple[str, ...]
    mapping_sha256: str

    # Compatibility aliases are deliberately read-only.  DEV is the only edge
    # validation partition; CAL belongs to the relation verifier.
    @property
    def validation_names(self) -> tuple[str, ...]:
        return self.development_names

    @property
    def validation_groups(self) -> tuple[str, ...]:
        return self.development_groups

    def validate(self) -> None:
        partitions = (self.fit_names, self.calibration_names, self.development_names)
        group_partitions = (self.fit_groups, self.calibration_groups, self.development_groups)
        if tuple(map(len, partitions)) != (
            FIT_SOURCE_COUNT,
            CAL_SOURCE_COUNT,
            DEV_SOURCE_COUNT,
        ):
            raise ValueError("E26 split must contain exactly 5360 FIT, 670 CAL, 670 DEV")
        if len(set().union(*map(set, partitions))) != TRAIN_SOURCE_COUNT:
            raise ValueError("E26 split name partitions are not an exact disjoint union")
        if len(set().union(*map(set, group_partitions))) != TRAIN_SOURCE_COUNT:
            raise ValueError("E26 split group partitions are not an exact disjoint union")
        for left in range(3):
            for right in range(left + 1, 3):
                if set(partitions[left]) & set(partitions[right]):
                    raise ValueError("E26 split names overlap")
                if set(group_partitions[left]) & set(group_partitions[right]):
                    raise ValueError("E26 split source groups overlap")


@dataclass(frozen=True)
class LossConfig:
    label_smoothing: float = 0.02
    boundary_weight: float = 0.25
    boundary_gradient_weight: float = 0.50

    def validate(self) -> None:
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.boundary_weight < 0.0 or self.boundary_gradient_weight < 0.0:
            raise ValueError("boundary loss weights cannot be negative")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _expected_train_names(count: int = TRAIN_SOURCE_COUNT) -> tuple[str, ...]:
    return tuple(f"img_{index:06d}.png" for index in range(count))


def _record_group(record: Any) -> str | None:
    if isinstance(record, str) and record:
        return record
    if isinstance(record, Mapping):
        group = record.get("source_group")
        if isinstance(group, str) and group:
            return group
    return None


def source_mapping_records(
    names: Sequence[str], group_for_name: Mapping[str, str]
) -> list[dict[str, str]]:
    """Return diagnostic records (the seal itself uses TAB/LF bytes below)."""

    return [
        {"name": str(name), "source_group": str(group_for_name[name])}
        for name in names
    ]


def source_mapping_sha256(
    names: Sequence[str], group_for_name: Mapping[str, str]
) -> str:
    rows: list[str] = []
    for name in names:
        group = str(group_for_name[name])
        if any(character in str(name) or character in group for character in ("\t", "\r", "\n")):
            raise ValueError("source mapping names/groups cannot contain TAB, CR, or LF")
        rows.append(f"{name}\t{group}\n")
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def load_authenticated_training_sources(
    path: Path,
    *,
    train_source_count: int = TRAIN_SOURCE_COUNT,
    expected_manifest_sha256: str = EXPECTED_SOURCE_MANIFEST_SHA256,
    expected_mapping_sha256: str = EXPECTED_SOURCE_MAPPING_SHA256,
) -> AuthenticatedTrainingSources:
    """Authenticate the frozen manifest and the exact E26 train namespace.

    Test callers may pass digests of an explicit synthetic manifest.  Production
    does not expose CLI overrides for either frozen digest.
    """

    if train_source_count < 2:
        raise ValueError("train_source_count must be at least two")
    path = Path(path)
    raw = path.read_bytes()
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "source-group manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256}, observed {manifest_sha256}"
        )
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("source-group manifest root must be an object")
    records = payload.get("files")
    if not isinstance(records, Mapping):
        records = payload.get("group_for_name")
    if not isinstance(records, Mapping):
        raise ValueError("source-group manifest needs files or group_for_name mapping")

    names = _expected_train_names(train_source_count)
    missing = [name for name in names if name not in records]
    if missing:
        raise ValueError(f"frozen E26 namespace is missing {len(missing)} rows")
    groups: dict[str, str] = {}
    content_sha256: dict[str, str] = {}
    for name in names:
        record = records[name]
        group = _record_group(record)
        if group is None:
            raise ValueError(f"missing source_group for train source {name!r}")
        groups[name] = group
        if isinstance(record, Mapping):
            digest = record.get("sha256")
            if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
                content_sha256[name] = digest
    if len(set(groups.values())) != train_source_count:
        raise ValueError("E26 requires one distinct source group per admitted source")
    mapping_sha256 = source_mapping_sha256(names, groups)
    if mapping_sha256 != expected_mapping_sha256:
        raise ValueError(
            "source name->group mapping SHA-256 mismatch under "
            f"{SOURCE_MAPPING_SERIALIZATION}: expected {expected_mapping_sha256}, "
            f"observed {mapping_sha256}"
        )
    return AuthenticatedTrainingSources(
        names=names,
        group_for_name=groups,
        content_sha256_for_name=content_sha256,
        manifest_sha256=manifest_sha256,
        mapping_sha256=mapping_sha256,
    )


def load_training_source_groups(
    path: Path,
    *,
    train_source_count: int = TRAIN_SOURCE_COUNT,
    expected_manifest_sha256: str = EXPECTED_SOURCE_MANIFEST_SHA256,
    expected_mapping_sha256: str = EXPECTED_SOURCE_MAPPING_SHA256,
) -> tuple[tuple[str, ...], dict[str, str], str]:
    """Compatibility wrapper around strict source authentication."""

    sources = load_authenticated_training_sources(
        path,
        train_source_count=train_source_count,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_mapping_sha256=expected_mapping_sha256,
    )
    return sources.names, dict(sources.group_for_name), sources.manifest_sha256


def _split_rank(mapping_sha256: str, name: str, source_group: str) -> str:
    material = f"{SPLIT_PREFIX}|{mapping_sha256}|{name}|{source_group}".encode("ascii")
    return hashlib.sha256(material).hexdigest()


def split_source_groups(
    names: Sequence[str],
    group_for_name: Mapping[str, str],
    *,
    mapping_sha256: str = EXPECTED_SOURCE_MAPPING_SHA256,
) -> SourceGroupSplit:
    """Apply the immutable 5,360/670/670 E26 source-disjoint split."""

    names = tuple(names)
    if names != _expected_train_names():
        raise ValueError("E26 split input is not exact img_000000..img_006699 namespace")
    if set(group_for_name) != set(names):
        raise ValueError("E26 split mapping has missing or extra train names")
    groups = tuple(str(group_for_name[name]) for name in names)
    if any(not group for group in groups) or len(set(groups)) != TRAIN_SOURCE_COUNT:
        raise ValueError("E26 split requires 6,700 non-empty distinct source groups")
    observed_mapping_sha256 = source_mapping_sha256(names, group_for_name)
    if observed_mapping_sha256 != mapping_sha256:
        raise ValueError("E26 split mapping digest does not match its authenticated seal")
    ranked = sorted(
        names,
        key=lambda name: (
            _split_rank(mapping_sha256, name, group_for_name[name]),
            name,
        ),
    )
    fit = tuple(ranked[:FIT_SOURCE_COUNT])
    calibration = tuple(ranked[FIT_SOURCE_COUNT : FIT_SOURCE_COUNT + CAL_SOURCE_COUNT])
    development = tuple(ranked[FIT_SOURCE_COUNT + CAL_SOURCE_COUNT :])
    split = SourceGroupSplit(
        fit_names=fit,
        calibration_names=calibration,
        development_names=development,
        fit_groups=tuple(group_for_name[name] for name in fit),
        calibration_groups=tuple(group_for_name[name] for name in calibration),
        development_groups=tuple(group_for_name[name] for name in development),
        mapping_sha256=mapping_sha256,
    )
    split.validate()
    return split


def build_split_manifest(split: SourceGroupSplit) -> dict[str, Any]:
    """Build the exact canonical split artifact; its byte SHA seeds all draws."""

    split.validate()
    partitions: dict[str, list[dict[str, str | int]]] = {}
    for label, names, groups in (
        ("FIT", split.fit_names, split.fit_groups),
        ("CAL", split.calibration_names, split.calibration_groups),
        ("DEV", split.development_names, split.development_groups),
    ):
        partitions[label] = [
            {
                "partition_rank": rank,
                "name": name,
                "source_group": group,
                "split_rank_sha256": _split_rank(split.mapping_sha256, name, group),
            }
            for rank, (name, group) in enumerate(zip(names, groups, strict=True))
        ]
    return {
        "schema": SPLIT_SCHEMA,
        "algorithm": (
            "sort (sha256(ascii('E26-split-v1|<mapping_sha>|<name>|<group>')),name); "
            "ranks [0,5360)=FIT [5360,6030)=CAL [6030,6700)=DEV"
        ),
        "mapping_sha256": split.mapping_sha256,
        "counts": {"FIT": FIT_SOURCE_COUNT, "CAL": CAL_SOURCE_COUNT, "DEV": DEV_SOURCE_COUNT},
        "partitions": partitions,
    }


def split_manifest_sha256(split_manifest: Mapping[str, Any]) -> str:
    return _sha256_json(split_manifest)


def deterministic_seed(
    split_sha256: str,
    stage: str,
    epoch_or_zero: int,
    name: str,
    purpose: str,
) -> int:
    """Derive the exact PCG64 seed: LE uint64(first eight SHA-256 bytes)."""

    if not re.fullmatch(r"[0-9a-f]{64}", split_sha256):
        raise ValueError("split SHA-256 must be lowercase hexadecimal")
    if stage not in {"FIT", "CAL", "DEV"}:
        raise ValueError("stage must be FIT, CAL, or DEV")
    if purpose not in {"corrupt", "perm"}:
        raise ValueError("purpose must be corrupt or perm")
    if epoch_or_zero < 0 or (stage != "FIT" and epoch_or_zero != 0):
        raise ValueError("CAL/DEV draws are fixed at epoch zero")
    material = (
        f"{RNG_PREFIX}|{split_sha256}|{stage}|{epoch_or_zero}|{name}|{purpose}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little", signed=False)


def epoch_source_order(
    fit_names: Sequence[str], split_sha256: str, epoch: int
) -> tuple[str, ...]:
    if epoch < 0:
        raise ValueError("epoch cannot be negative")
    return tuple(
        sorted(
            fit_names,
            key=lambda name: (
                hashlib.sha256(
                    f"{ORDER_PREFIX}|{split_sha256}|{epoch}|{name}".encode("ascii")
                ).digest(),
                name,
            ),
        )
    )


class DeterministicE26Dataset(Dataset[dict[str, Tensor]]):
    """Stateless synthetic draws; workers and prefetch cannot change samples."""

    def __init__(
        self,
        names: Sequence[str],
        *,
        clean_root: Path,
        split_sha256: str,
        stage: str,
        epoch_or_zero: int,
    ) -> None:
        self.names = tuple(names)
        self.clean_root = Path(clean_root)
        self.split_sha256 = split_sha256
        self.stage = stage
        self.epoch_or_zero = int(epoch_or_zero)
        if not self.names:
            raise ValueError("deterministic E26 dataset needs at least one source")
        # Validate before a worker is spawned.
        deterministic_seed(split_sha256, stage, epoch_or_zero, self.names[0], "corrupt")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        # Lazy imports keep source inspection/test discovery data-free.
        from distort import distort_frags
        from imgio import load, to_frags

        name = self.names[index]
        clean = load(str(self.clean_root / name))
        if clean.shape != (480, 480, 3) or clean.dtype != np.uint8:
            raise ValueError(f"invalid clean E26 source {name}: {clean.shape} {clean.dtype}")
        canonical = to_frags(clean)
        corrupt_rng = np.random.Generator(
            np.random.PCG64(
                deterministic_seed(
                    self.split_sha256,
                    self.stage,
                    self.epoch_or_zero,
                    name,
                    "corrupt",
                )
            )
        )
        perm_rng = np.random.Generator(
            np.random.PCG64(
                deterministic_seed(
                    self.split_sha256,
                    self.stage,
                    self.epoch_or_zero,
                    name,
                    "perm",
                )
            )
        )
        dirty = distort_frags(canonical, corrupt_rng)
        permutation = perm_rng.permutation(len(canonical)).astype(np.int64)
        dirty = dirty[permutation]
        return {
            "tiles": torch.from_numpy(np.ascontiguousarray(dirty))
            .permute(0, 3, 1, 2)
            .float()
            .div_(255.0),
            "clean": torch.from_numpy(np.ascontiguousarray(clean))
            .permute(2, 0, 1)
            .float()
            .div_(255.0),
            "perm": torch.from_numpy(permutation),
            "name": name,
        }


def clean_canvas_to_tiles(clean: Tensor, grid_height: int, grid_width: int) -> Tensor:
    if clean.ndim != 4 or clean.shape[1] != 3:
        raise ValueError("clean canvas must have shape (batch, 3, height, width)")
    batch, channels, height, width = clean.shape
    if height % grid_height or width % grid_width:
        raise ValueError("clean canvas dimensions must be divisible by the E26 grid")
    tile_height = height // grid_height
    tile_width = width // grid_width
    return (
        clean.reshape(batch, channels, grid_height, tile_height, grid_width, tile_width)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, grid_height * grid_width, channels, tile_height, tile_width)
        .contiguous()
    )


def clean_tiles_in_input_order(
    clean: Tensor,
    permutation: Tensor,
    grid_height: int,
    grid_width: int,
) -> Tensor:
    canonical = clean_canvas_to_tiles(clean, grid_height, grid_width)
    if permutation.ndim != 2 or tuple(permutation.shape) != tuple(canonical.shape[:2]):
        raise ValueError("permutation shape must match clean canvas tile count")
    gather = permutation.long()[..., None, None, None].expand_as(canonical)
    return canonical.gather(1, gather)


def compute_loss(
    output: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    model_config: ContextualEdgeConfig,
    loss_config: LossConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    loss_config.validate()
    logits = output.get("logits")
    reconstruction = output.get("boundary_reconstruction")
    if not isinstance(logits, Tensor) or not isinstance(reconstruction, Tensor):
        raise KeyError("model output must contain logits and boundary_reconstruction")
    if "perm" not in batch or "clean" not in batch:
        raise KeyError("synthetic E26 batch requires perm and clean")
    device = logits.device
    permutation = batch["perm"].to(device=device, dtype=torch.long, non_blocking=True)
    labels = directional_neighbour_labels(
        permutation, model_config.grid_height, model_config.grid_width
    )
    primary = listwise_directional_ce(
        logits.float(), labels, label_smoothing=loss_config.label_smoothing
    )
    clean = batch["clean"].to(device=device, dtype=torch.float32, non_blocking=True)
    aligned_clean = clean_tiles_in_input_order(
        clean, permutation, model_config.grid_height, model_config.grid_width
    )
    target = clean_boundary_targets(
        aligned_clean,
        model_config.reconstruction_samples,
        band=model_config.boundary_band,
    )
    auxiliary, auxiliary_terms = boundary_reconstruction_loss(
        reconstruction.float(),
        target,
        gradient_weight=loss_config.boundary_gradient_weight,
    )
    total = primary + loss_config.boundary_weight * auxiliary
    return total, {
        "total": total,
        "listwise_ce": primary,
        "boundary": auxiliary,
        **auxiliary_terms,
    }


@torch.no_grad()
def batch_metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
    prediction = logits.argmax(dim=-1)
    count = logits.shape[-1] - 1
    none = labels.eq(count)
    neighbour = ~none
    correct = prediction.eq(labels)
    return {
        "row_accuracy": float(correct.float().mean().cpu()),
        "neighbour_r1": float(correct[neighbour].float().mean().cpu()),
        "none_accuracy": float(correct[none].float().mean().cpu()),
        "mean_ce": float(
            torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, count + 1), labels.reshape(-1)
            ).cpu()
        ),
    }


def _loader(dataset: Dataset[Any], *, workers: int, device: torch.device) -> DataLoader:
    # DataLoader draws a worker base seed even with shuffle=False/workers=0.
    # Giving it a private generator prevents iterator construction (including a
    # mid-epoch resume iterator) from shifting the model/dropout RNG stream.
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(FROZEN_SEED)
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": FROZEN_BATCH_SIZE,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
        "generator": loader_generator,
    }
    if workers:
        arguments.update(persistent_workers=False, prefetch_factor=2)
    return DataLoader(**arguments)


@torch.inference_mode()
def evaluate(
    model: ContextualDirectionalEdgeNet,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    sums: dict[str, float] = {}
    rows = 0
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    for batch in loader:
        tiles = batch["tiles"].to(device, non_blocking=True)
        labels = directional_neighbour_labels(
            batch["perm"].to(device), model.config.grid_height, model.config.grid_width
        )
        with autocast:
            output = model(tiles)
        metrics = batch_metrics(output["logits"], labels)
        for name, value in metrics.items():
            sums[name] = sums.get(name, 0.0) + value
        rows += 1
    model.train(was_training)
    if rows == 0:
        raise ValueError("evaluation loader yielded no batches")
    return {name: value / rows for name, value in sums.items()} | {"scenes": rows}


def _is_e_drive(path: str | Path) -> bool:
    return PureWindowsPath(str(path)).drive.upper() == "E:"


def validate_tag(tag: str) -> str:
    if not TAG_RE.fullmatch(tag):
        raise ValueError("tag must match [a-z0-9][a-z0-9_.-]{0,79}")
    return tag


def validate_e_only_runtime(
    *,
    source_manifest: Path,
    output_dir: Path,
    resume: Path | None,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Reject any production path/cache contract that can write to C:."""

    env = os.environ if environment is None else environment
    paths = {"source_manifest": source_manifest, "output_dir": output_dir}
    if resume is not None:
        paths["resume"] = resume
    for name, path in paths.items():
        if not _is_e_drive(path):
            raise ValueError(f"{name} must be an explicit E: path, observed {path}")
    for name in REQUIRED_E_ENV:
        value = env.get(name, "").strip()
        if not value or not _is_e_drive(value):
            raise ValueError(f"{name} must be explicitly configured on E:")
    for name in OPTIONAL_E_ENV:
        value = env.get(name, "").strip()
        if value and not _is_e_drive(value):
            raise ValueError(f"{name} is configured outside E: ({value})")
    for name in ("PAZZLE_DATA", "PAZZLE_WORK"):
        value = env.get(name, "").strip()
        if not value or not _is_e_drive(value):
            raise ValueError(f"{name} must be explicitly configured on E:")
    if env.get("PYTHONHASHSEED") != str(FROZEN_SEED):
        raise ValueError(f"PYTHONHASHSEED must be exactly {FROZEN_SEED}")
    if env.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be exactly :4096:8")


def verify_clean_source_inventory(
    clean_root: Path,
    names: Sequence[str],
    expected_sha256_for_name: Mapping[str, str],
) -> str:
    """Hash-authenticate every admitted clean source before model construction."""

    if set(expected_sha256_for_name) != set(names):
        raise ValueError("frozen source manifest lacks an image SHA-256 for every E26 source")
    records: list[dict[str, str]] = []
    for name in names:
        path = Path(clean_root) / name
        observed = sha256_file(path)
        expected = expected_sha256_for_name[name]
        if observed != expected:
            raise ValueError(f"clean source SHA-256 mismatch: {name}")
        records.append({"name": name, "sha256": observed})
    return _sha256_json(records)


def dependency_provenance() -> dict[str, Any]:
    import cv2
    import PIL

    source_dir = Path(__file__).resolve().parent
    source_files = (
        "train_e26_contextual_edge.py",
        "e26_contextual_edge_net.py",
        "distort.py",
        "imgio.py",
        "config.py",
    )
    return {
        "source_sha256": {
            name: sha256_file(source_dir / name) for name in source_files
        },
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "opencv": cv2.__version__,
        "pillow": PIL.__version__,
    }


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise ValueError("checkpoint RNG state is incomplete or contains unknown members")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available():
        cuda_state = state["torch_cuda"]
        if len(cuda_state) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG device count differs from current runtime")
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])
    elif state["torch_cuda"]:
        raise ValueError("CUDA RNG state cannot be restored without CUDA")


def checkpoint_payload(
    model: ContextualDirectionalEdgeNet,
    optimizer: torch.optim.Optimizer | None,
    *,
    step: int,
    training_config: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    split_contract: Mapping[str, Any] | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    progress: Mapping[str, int] | None = None,
    rng_state: Mapping[str, Any] | None = None,
    run_contract: Mapping[str, Any] | None = None,
    dependencies: Mapping[str, Any] | None = None,
    history: Mapping[str, Any] | None = None,
    checkpoint_kind: str = "generic",
) -> dict[str, Any]:
    if step < 0:
        raise ValueError("checkpoint step cannot be negative")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "training_schema": TRAINING_SCHEMA,
        "recovery_schema": RECOVERY_SCHEMA if checkpoint_kind == "recovery" else None,
        "checkpoint_kind": checkpoint_kind,
        "model_config": model_config_dict(model),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": int(step),
        "progress": dict(progress or {}),
        "rng_state": dict(rng_state or {}),
        "training_config": dict(training_config or {}),
        "metrics": dict(metrics or {}),
        "split_contract": dict(split_contract or {}),
        "run_contract": dict(run_contract or {}),
        "dependencies": dict(dependencies or {}),
        "history": dict(history or {}),
    }


def save_checkpoint_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.stage-", suffix=".pt", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        # Windows' CRT rejects fsync on a read-only descriptor.
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.stage-", suffix=".json", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint does not implement the E26 schema")
    model_from_checkpoint_payload(payload)
    return dict(payload)


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    """Explicitly place every nested optimizer tensor with its CUDA parameters."""

    def move(value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.to(device=device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    for parameter, state in tuple(optimizer.state.items()):
        optimizer.state[parameter] = move(state)


def validate_recovery_checkpoint(
    payload: Mapping[str, Any],
    *,
    run_contract_sha256: str,
    split_manifest_sha256_value: str,
    source_manifest_sha256: str,
) -> dict[str, int]:
    if payload.get("training_schema") != TRAINING_SCHEMA:
        raise ValueError("recovery checkpoint training schema mismatch")
    if payload.get("recovery_schema") != RECOVERY_SCHEMA or payload.get("checkpoint_kind") != "recovery":
        raise ValueError("--resume accepts only E26 recovery checkpoints")
    for name in (
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "rng_state",
        "history",
        "dependencies",
        "training_config",
    ):
        if payload.get(name) is None:
            raise ValueError(f"recovery checkpoint lacks {name}")
    run_contract = payload.get("run_contract")
    split_contract = payload.get("split_contract")
    if not isinstance(run_contract, Mapping) or run_contract.get("sha256") != run_contract_sha256:
        raise ValueError("recovery checkpoint run-contract SHA-256 mismatch")
    run_contract_body = dict(run_contract)
    run_contract_body.pop("sha256", None)
    if _sha256_json(run_contract_body) != run_contract_sha256:
        raise ValueError("recovery checkpoint run-contract body is not hash-authentic")
    if not isinstance(split_contract, Mapping):
        raise ValueError("recovery checkpoint lacks split contract")
    if dict(split_contract) != run_contract.get("split_contract"):
        raise ValueError("recovery checkpoint duplicated split contract drifted")
    if payload.get("dependencies") != run_contract.get("dependencies"):
        raise ValueError("recovery checkpoint dependency provenance drifted")
    if payload.get("training_config") != run_contract.get("training_config"):
        raise ValueError("recovery checkpoint training configuration drifted")
    if split_contract.get("split_manifest_sha256") != split_manifest_sha256_value:
        raise ValueError("recovery checkpoint split SHA-256 mismatch")
    if split_contract.get("source_manifest_sha256") != source_manifest_sha256:
        raise ValueError("recovery checkpoint source-manifest SHA-256 mismatch")
    rng_state = payload.get("rng_state")
    if not isinstance(rng_state, Mapping) or set(rng_state) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise ValueError("recovery checkpoint RNG state is incomplete")
    history = payload.get("history")
    if not isinstance(history, Mapping) or not {
        "started_utc",
        "recovery_commits",
        "loss_log",
        "epoch_end",
    }.issubset(history):
        raise ValueError("recovery checkpoint historical state is incomplete")
    progress = payload.get("progress")
    if not isinstance(progress, Mapping):
        raise ValueError("recovery checkpoint lacks progress")
    try:
        epoch = int(progress["next_epoch"])
        cursor = int(progress["next_sample_cursor"])
        optimizer_steps = int(progress["optimizer_steps_completed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("recovery checkpoint progress is invalid") from exc
    if not (0 <= epoch <= FROZEN_EPOCHS):
        raise ValueError("recovery checkpoint epoch is out of range")
    if not (0 <= cursor < FIT_SOURCE_COUNT) or cursor % FROZEN_ACCUMULATE:
        raise ValueError("recovery checkpoint cursor is not an optimizer boundary")
    expected_steps = epoch * (FIT_SOURCE_COUNT // FROZEN_ACCUMULATE) + cursor // FROZEN_ACCUMULATE
    if optimizer_steps != expected_steps or int(payload.get("step", -1)) != optimizer_steps:
        raise ValueError("recovery checkpoint step/cursor arithmetic is inconsistent")
    return {
        "next_epoch": epoch,
        "next_sample_cursor": cursor,
        "optimizer_steps_completed": optimizer_steps,
    }


def _warmup_cosine_factor(
    optimizer_steps_completed: int,
    *,
    warmup_steps: int = 1_000,
    total_steps: int = TOTAL_OPTIMIZER_STEPS,
    minimum_ratio: float = 0.10,
) -> float:
    if optimizer_steps_completed < warmup_steps:
        return float(optimizer_steps_completed + 1) / float(warmup_steps)
    progress = min(
        1.0,
        (optimizer_steps_completed - warmup_steps) / max(1, total_steps - warmup_steps),
    )
    return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def _artifact_paths(output_dir: Path, tag: str) -> dict[str, Path]:
    validate_tag(tag)
    return {
        "split_manifest": output_dir / f"{tag}_split_manifest.json",
        "recovery": output_dir / f"{tag}_recovery.pt",
        "final": output_dir / f"{tag}_final_epoch08.pt",
        "report": output_dir / f"{tag}_training_report.json",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="e26_contextual_edge_v2")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--cnn-width", type=int, default=48)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--local-dim", type=int, default=96)
    parser.add_argument("--match-dim", type=int, default=64)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--encoder-chunk-size", type=int, default=144)
    parser.add_argument("--log-every", type=int, default=25)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    try:
        validate_tag(args.tag)
    except ValueError as exc:
        parser.error(str(exc))
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    for name in (
        "cnn_width",
        "d_model",
        "local_dim",
        "match_dim",
        "transformer_layers",
        "attention_heads",
        "encoder_chunk_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")


def _set_deterministic_runtime() -> None:
    random.seed(FROZEN_SEED)
    np.random.seed(FROZEN_SEED)
    torch.manual_seed(FROZEN_SEED)
    torch.cuda.manual_seed_all(FROZEN_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _publish_recovery(
    path: Path,
    model: ContextualDirectionalEdgeNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    *,
    progress: Mapping[str, int],
    training_config: Mapping[str, Any],
    split_contract: Mapping[str, Any],
    run_contract: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    history: Mapping[str, Any],
) -> None:
    save_checkpoint_atomic(
        path,
        checkpoint_payload(
            model,
            optimizer,
            scheduler=scheduler,
            scaler=scaler,
            step=progress["optimizer_steps_completed"],
            progress=progress,
            rng_state=capture_rng_state(),
            training_config=training_config,
            split_contract=split_contract,
            run_contract=run_contract,
            dependencies=dependencies,
            history=history,
            checkpoint_kind="recovery",
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    try:
        validate_e_only_runtime(
            source_manifest=args.source_manifest,
            output_dir=args.output_dir,
            resume=args.resume,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not torch.cuda.is_available():
        parser.error("frozen E26 training requires CUDA with bfloat16 support")
    if not torch.cuda.is_bf16_supported():
        parser.error("frozen E26 training requires CUDA bfloat16 support")
    device = torch.device("cuda")
    _set_deterministic_runtime()

    sources = load_authenticated_training_sources(args.source_manifest)
    split = split_source_groups(
        sources.names,
        sources.group_for_name,
        mapping_sha256=sources.mapping_sha256,
    )
    split_manifest = build_split_manifest(split)
    split_sha256 = split_manifest_sha256(split_manifest)
    clean_root = Path(os.environ["PAZZLE_DATA"]) / "train" / "targets"
    if not _is_e_drive(clean_root):
        parser.error("clean train target root must be on E:")
    source_inventory_sha256 = verify_clean_source_inventory(
        clean_root,
        sources.names,
        sources.content_sha256_for_name,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _artifact_paths(args.output_dir, args.tag)
    split_bytes = _canonical_json(split_manifest)
    if artifacts["split_manifest"].exists():
        if artifacts["split_manifest"].read_bytes() != split_bytes:
            raise RuntimeError("existing split manifest conflicts with frozen E26 split")
    else:
        save_json_atomic(artifacts["split_manifest"], split_manifest)
    if sha256_file(artifacts["split_manifest"]) != split_sha256:
        raise RuntimeError("published split manifest SHA-256 drifted")
    if args.resume is None and any(artifacts[name].exists() for name in ("recovery", "final", "report")):
        raise RuntimeError("E26 artifacts already exist; provide --resume or choose a new frozen tag")

    model_config = ContextualEdgeConfig(
        cnn_width=args.cnn_width,
        d_model=args.d_model,
        local_dim=args.local_dim,
        match_dim=args.match_dim,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
        encoder_chunk_size=args.encoder_chunk_size,
    )
    loss_config = LossConfig()
    loss_config.validate()
    dependencies = dependency_provenance()
    split_contract = {
        "source_manifest_path": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sources.manifest_sha256,
        "source_mapping_sha256": sources.mapping_sha256,
        "source_mapping_serialization": SOURCE_MAPPING_SERIALIZATION,
        "source_inventory_sha256": source_inventory_sha256,
        "split_manifest_path": str(artifacts["split_manifest"].resolve()),
        "split_manifest_sha256": split_sha256,
        "counts": {"FIT": FIT_SOURCE_COUNT, "CAL": CAL_SOURCE_COUNT, "DEV": DEV_SOURCE_COUNT},
    }
    training_config = {
        "epochs": FROZEN_EPOCHS,
        "batch_size": FROZEN_BATCH_SIZE,
        "accumulate": FROZEN_ACCUMULATE,
        "seed": FROZEN_SEED,
        "workers": args.workers,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 3.0e-4,
            "betas": [0.9, 0.95],
            "eps": 1.0e-8,
            "weight_decay": 0.01,
        },
        "schedule": {
            "name": "linear-warmup-cosine",
            "warmup_steps": 1_000,
            "total_steps": TOTAL_OPTIMIZER_STEPS,
            "minimum_learning_rate": 3.0e-5,
        },
        "precision": "cuda-bfloat16",
        "recovery_interval_optimizer_steps": RECOVERY_INTERVAL,
        "gradient_clip_norm": 1.0,
        "model": asdict(model_config),
        "loss": asdict(loss_config),
        "draw": {
            "rng_prefix": RNG_PREFIX,
            "order_prefix": ORDER_PREFIX,
            "pcg64_seed": "little-endian uint64(first8(SHA256(ascii(material))))",
            "rotation": False,
            "reflection": False,
            "brightness": [-30.0, 30.0],
            "contrast": [0.70, 1.30],
            "noise_sigma": [40.0, 55.0],
            "blur": "separable reflect [0.25,0.5,0.25]",
            "jpeg_quality_inclusive": [35, 50],
        },
        "selection": "fixed final epoch 8; DEV metrics cannot select a checkpoint",
        "output_dir": str(args.output_dir.resolve()),
        "tag": args.tag,
    }
    run_contract_base = {
        "training_schema": TRAINING_SCHEMA,
        "training_config": training_config,
        "split_contract": split_contract,
        "dependencies": dependencies,
    }
    run_contract = dict(run_contract_base)
    run_contract["sha256"] = _sha256_json(run_contract_base)

    model = ContextualDirectionalEdgeNet(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda completed: _warmup_cosine_factor(completed),
    )
    # BF16 has no FP16-style dynamic scaling, but a (disabled) scaler state is
    # still checkpointed and verified as part of the optimizer-boundary state.
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    progress = {
        "next_epoch": 0,
        "next_sample_cursor": 0,
        "optimizer_steps_completed": 0,
    }
    history: dict[str, Any] = {
        "started_utc": _utc_now(),
        "recovery_commits": [],
        "loss_log": [],
        "epoch_end": [],
    }
    if args.resume is not None:
        # Deserialize on CPU so CPU/CUDA RNG byte tensors preserve their
        # semantic role.  Model and optimizer state are then moved explicitly.
        resume = load_checkpoint(args.resume, map_location="cpu")
        restored = model_from_checkpoint_payload(resume)
        if restored.config != model.config:
            parser.error("--resume model configuration differs from frozen run contract")
        progress = validate_recovery_checkpoint(
            resume,
            run_contract_sha256=run_contract["sha256"],
            split_manifest_sha256_value=split_sha256,
            source_manifest_sha256=sources.manifest_sha256,
        )
        model.load_state_dict(restored.state_dict())
        optimizer.load_state_dict(resume["optimizer"])
        move_optimizer_state_to_device(optimizer, device)
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        history = dict(resume["history"])
        restore_rng_state(resume["rng_state"])

    print(
        f"device={device} params={sum(p.numel() for p in model.parameters()):,} "
        f"FIT={FIT_SOURCE_COUNT} CAL={CAL_SOURCE_COUNT} DEV={DEV_SOURCE_COUNT} "
        f"split_sha256={split_sha256} run_sha256={run_contract['sha256']} "
        f"resume_epoch={progress['next_epoch']} resume_cursor={progress['next_sample_cursor']}",
        flush=True,
    )
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    started_monotonic = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(progress["next_epoch"], FROZEN_EPOCHS):
        order = epoch_source_order(split.fit_names, split_sha256, epoch)
        cursor = progress["next_sample_cursor"] if epoch == progress["next_epoch"] else 0
        if cursor % FROZEN_ACCUMULATE:
            raise RuntimeError("resume cursor is not an optimizer boundary")
        remaining = order[cursor:]
        dataset = DeterministicE26Dataset(
            remaining,
            clean_root=clean_root,
            split_sha256=split_sha256,
            stage="FIT",
            epoch_or_zero=epoch,
        )
        iterator = iter(_loader(dataset, workers=args.workers, device=device))
        while cursor < FIT_SOURCE_COUNT:
            sums: dict[str, float] = {}
            for _ in range(FROZEN_ACCUMULATE):
                batch = next(iterator)
                tiles = batch["tiles"].to(device, non_blocking=True)
                with autocast:
                    output = model(tiles)
                    loss, terms = compute_loss(output, batch, model_config, loss_config)
                    scaled_loss = loss / FROZEN_ACCUMULATE
                scaler.scale(scaled_loss).backward()
                for name, value in terms.items():
                    sums[name] = sums.get(name, 0.0) + float(value.detach()) / FROZEN_ACCUMULATE
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            cursor += FROZEN_ACCUMULATE
            progress = {
                "next_epoch": epoch,
                "next_sample_cursor": cursor,
                "optimizer_steps_completed": progress["optimizer_steps_completed"] + 1,
            }
            if cursor == FIT_SOURCE_COUNT:
                progress = {
                    "next_epoch": epoch + 1,
                    "next_sample_cursor": 0,
                    "optimizer_steps_completed": progress["optimizer_steps_completed"],
                }
            if progress["optimizer_steps_completed"] % args.log_every == 0:
                row = {
                    "epoch": epoch,
                    "sample_cursor": cursor,
                    "optimizer_steps_completed": progress["optimizer_steps_completed"],
                    "total": sums["total"],
                    "listwise_ce": sums["listwise_ce"],
                    "boundary": sums["boundary"],
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                history["loss_log"].append(row)
                print(
                    f"epoch={epoch + 1}/{FROZEN_EPOCHS} sample={cursor}/{FIT_SOURCE_COUNT} "
                    f"step={row['optimizer_steps_completed']}/{TOTAL_OPTIMIZER_STEPS} "
                    f"total={row['total']:.5f} ce={row['listwise_ce']:.5f} "
                    f"boundary={row['boundary']:.5f} lr={row['learning_rate']:.3e} "
                    f"elapsed={time.monotonic() - started_monotonic:.1f}s",
                    flush=True,
                )
            is_epoch_end = progress["next_epoch"] == epoch + 1
            if progress["optimizer_steps_completed"] % RECOVERY_INTERVAL == 0 or is_epoch_end:
                if is_epoch_end:
                    history["epoch_end"].append(
                        {
                            "epoch_completed": epoch + 1,
                            "optimizer_steps_completed": progress["optimizer_steps_completed"],
                            "utc": _utc_now(),
                        }
                    )
                history["recovery_commits"].append(
                    {
                        **progress,
                        "reason": "epoch_end" if is_epoch_end else "interval_100",
                        "utc": _utc_now(),
                    }
                )
                _publish_recovery(
                    artifacts["recovery"],
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    progress=progress,
                    training_config=training_config,
                    split_contract=split_contract,
                    run_contract=run_contract,
                    dependencies=dependencies,
                    history=history,
                )
        progress["next_sample_cursor"] = 0

    if progress != {
        "next_epoch": FROZEN_EPOCHS,
        "next_sample_cursor": 0,
        "optimizer_steps_completed": TOTAL_OPTIMIZER_STEPS,
    }:
        raise RuntimeError(f"frozen E26 training terminated at invalid progress: {progress}")

    # One deterministic DEV draw per source at epoch zero.  It is evaluated only
    # after the fixed epoch-8 weights already exist in memory and cannot choose
    # an epoch, threshold, seed, or model variant.
    development_dataset = DeterministicE26Dataset(
        split.development_names,
        clean_root=clean_root,
        split_sha256=split_sha256,
        stage="DEV",
        epoch_or_zero=0,
    )
    metrics = evaluate(
        model,
        _loader(development_dataset, workers=min(args.workers, 2), device=device),
        device,
    )
    final_payload = checkpoint_payload(
        model,
        None,
        step=TOTAL_OPTIMIZER_STEPS,
        progress=progress,
        training_config=training_config,
        metrics=metrics,
        split_contract=split_contract,
        run_contract=run_contract,
        dependencies=dependencies,
        history=history,
        checkpoint_kind="scientific_final_epoch08",
    )
    save_checkpoint_atomic(artifacts["final"], final_payload)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "completed_fixed_final",
        "completed_utc": _utc_now(),
        "scientific_selection": "fixed epoch 8 only",
        "final_checkpoint": {
            "path": str(artifacts["final"].resolve()),
            "sha256": sha256_file(artifacts["final"]),
        },
        "recovery_checkpoint": {
            "path": str(artifacts["recovery"].resolve()),
            "sha256": sha256_file(artifacts["recovery"]),
        },
        "split_contract": split_contract,
        "run_contract": run_contract,
        "metrics": metrics,
        "progress": progress,
        "history": history,
    }
    save_json_atomic(artifacts["report"], report)
    print(
        "fixed-final DEV " + " ".join(f"{name}={value:.6f}" for name, value in metrics.items()),
        flush=True,
    )
    print(f"final_checkpoint={artifacts['final']} report={artifacts['report']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
