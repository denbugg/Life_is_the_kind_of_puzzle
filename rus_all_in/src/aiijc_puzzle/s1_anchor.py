"""Auditable reconstruction of the historical S1 restoration anchor.

The externally scored S1 pipeline used a frozen rank96 solver followed by an
R5 ``RestoreNet`` and OpenCV coloured NLM.  The four historical checkpoints
were not committed to Git, so this module deliberately separates two things:

* an exact, hash-aware description of the missing production artifacts; and
* the complete post-layout R5 -> NLM arm, which can be run on an input-only
  board permutation without consulting a clean image or target.

It is not a replacement solver.  A board supplied here must have been inferred
from the corrupted image alone (for example by the original rank96 code).
"""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

GRID_SIZE = 24
TILE_SIZE = 20
TILE_COUNT = GRID_SIZE * GRID_SIZE
IMAGE_SIZE = GRID_SIZE * TILE_SIZE

S1_OFFICIAL_SSIM = 0.23748525732559034
S1_RUNNER_COMMIT = "3c2f0b8"
S1_RESULT_COMMIT = "d7826aa"
S1_ARCHIVE_BRANCH = "origin/autoresearch/pazzle-fixed-orientation-cb1"

RANK96_CONTRACT: dict[str, Any] = {
    "orientation": "fixed_upright_tiles_no_rotation",
    "candidate_k_per_encoder": 64,
    "candidate_union": "ordered_deduplicated_primary_then_secondary",
    "candidate_score": "candidate_ranker_raw_logits",
    "dense_conversion": "cpu_float32_dense_rd",
    "solver": "corrected_best_buddies",
    "max_edges": 96,
    "min_margin": 0.0,
    "repair_passes": 0,
}

S1_TAIL_CONTRACT: dict[str, Any] = {
    "r5_architecture": "RestoreNet",
    "r5_base": 32,
    "r5_depth": 4,
    "r5_dtype": "float32",
    "r5_input": "assembled_480x480_rgb_raw_rank96_layout",
    "nlm_operator": "cv2.fastNlMeansDenoisingColored",
    "nlm_h": 10,
    "nlm_h_color": 10,
    "nlm_template_window": 7,
    "nlm_search_window": 21,
    # Historical infer_rank96.fixed_nlm passed the repository RGB ndarray
    # directly to OpenCV.  Do not insert an RGB<->BGR conversion when trying
    # to reproduce the byte-level S1 tail.
    "opencv_input_contract": "historical_direct_rgb_ndarray_no_channel_swap",
}

EXPECTED_RANK96_SHA256: dict[str, str] = {
    "ranker": "42685373b1a450a4cb3d7a9b22370dfcfaa2335e9e8ada609f21b7cc64abbfbc",
    "affinity_primary": "708565329c7661a965215d98e85f462a90930071f36a0f75b4813c0c5797ec4f",
    "affinity_secondary": "0fceafdb110bde59149fe1ad1e800a69d116041bc627af369aaecd60be53b6c8",
}


@dataclass(frozen=True)
class S1ArtifactPaths:
    """The four checkpoint paths needed for exact historical S1 inference."""

    ranker: Path
    affinity_primary: Path
    affinity_secondary: Path
    r5: Path


@dataclass(frozen=True)
class ArtifactStatus:
    """One checkpoint's local availability and integrity."""

    role: str
    path: str
    exists: bool
    sha256: str | None
    expected_sha256: str | None
    integrity: str


def default_artifact_paths(project_root: Path) -> S1ArtifactPaths:
    """Return stable workspace-local destinations for recovered S1 artifacts."""
    root = project_root / "artifacts" / "s1_anchor"
    return S1ArtifactPaths(
        ranker=root / "rank_v2w64_best.pt",
        affinity_primary=root / "affinity_r1_1200_best.pt",
        affinity_secondary=root / "affinity_r3_1000_best.pt",
        r5=root / "r5_capacity_fp32.pt",
    )


def sha256_file(path: Path) -> str:
    """Hash a file without loading a checkpoint into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def audit_artifacts(
    paths: S1ArtifactPaths,
    *,
    expected_rank96: Mapping[str, str] = EXPECTED_RANK96_SHA256,
    expected_r5_sha256: str | None = None,
) -> dict[str, Any]:
    """Audit availability and hashes without deserialising untrusted weights.

    Git never recorded the R5 digest.  Its status is therefore ``present_unpinned``
    unless the caller supplies a digest recovered from the original S1 manifest.
    That distinction prevents a same-named retrain from being labelled the
    externally scored model.
    """
    records: list[ArtifactStatus] = []
    path_map = asdict(paths)
    for role in ("ranker", "affinity_primary", "affinity_secondary", "r5"):
        path = Path(path_map[role])
        expected = expected_r5_sha256 if role == "r5" else expected_rank96.get(role)
        if not path.is_file():
            record = ArtifactStatus(role, str(path.resolve()), False, None, expected, "missing")
        else:
            actual = sha256_file(path)
            if expected is None:
                integrity = "present_unpinned"
            elif actual == expected:
                integrity = "verified"
            else:
                integrity = "hash_mismatch"
            record = ArtifactStatus(role, str(path.resolve()), True, actual, expected, integrity)
        records.append(record)

    exact_ready = all(record.integrity == "verified" for record in records)
    tail_runnable = next(record for record in records if record.role == "r5").exists
    return {
        "schema": "aiijc-s1-artifact-audit-v1",
        "historical_runner_commit": S1_RUNNER_COMMIT,
        "historical_result_commit": S1_RESULT_COMMIT,
        "official_ssim": S1_OFFICIAL_SSIM,
        "records": [asdict(record) for record in records],
        "exact_s1_ready": exact_ready,
        "tail_runnable": tail_runnable,
        "note": (
            "Exact readiness requires the three pinned rank96 hashes and a recovered "
            "R5 hash. Git records the R5 filename and architecture, but not its digest."
        ),
    }


class ConvBlock(nn.Module):
    """The exact residual block used by the historical R5 model."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.c2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )
        self.act = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.act(self.c1(value))
        hidden = self.c2(hidden)
        return self.act(hidden + self.skip(value))


class RestoreNet(nn.Module):
    """Exact R5 U-Net topology, including its global input residual."""

    def __init__(self, base: int = 32, depth: int = 4) -> None:
        super().__init__()
        channels = [base * (2**index) for index in range(depth)]
        self.stem = nn.Conv2d(3, base, 3, padding=1)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for index in range(depth - 1):
            self.enc.append(ConvBlock(channels[index], channels[index]))
            self.down.append(nn.Conv2d(channels[index], channels[index + 1], 2, stride=2))
        self.mid = ConvBlock(channels[-1], channels[-1])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for index in range(depth - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(channels[index], channels[index - 1], 2, stride=2))
            self.dec.append(ConvBlock(channels[index - 1] * 2, channels[index - 1]))
        self.head = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(value)
        skips: list[torch.Tensor] = []
        for encoder, downsample in zip(self.enc, self.down, strict=True):
            hidden = encoder(hidden)
            skips.append(hidden)
            hidden = downsample(hidden)
        hidden = self.mid(hidden)
        for upsample, decoder, skip in zip(self.up, self.dec, reversed(skips), strict=True):
            hidden = upsample(hidden)
            hidden = decoder(torch.cat([hidden, skip], dim=1))
        return torch.clamp(value + self.head(hidden), 0.0, 1.0)


def _checkpoint_state(payload: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("R5 checkpoint must be a mapping")
    for key in ("model", "model_state_dict", "state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return candidate  # type: ignore[return-value]
    if payload and all(isinstance(key, str) for key in payload):
        return payload  # type: ignore[return-value]
    raise TypeError("R5 checkpoint has no model/model_state_dict/state_dict mapping")


def load_r5_checkpoint(path: Path, device: torch.device) -> RestoreNet:
    """Load an explicitly trusted local R5 checkpoint with strict shapes."""
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with old torch
        payload = torch.load(path, map_location=device)
    model = RestoreNet(base=32, depth=4).to(device)
    model.load_state_dict(_checkpoint_state(payload), strict=True)
    model.eval()
    return model


def validate_rgb(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    expected = (IMAGE_SIZE, IMAGE_SIZE, 3)
    if value.shape != expected or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB {expected}, got {value.dtype} {value.shape}")
    return np.ascontiguousarray(value)


def split_tiles(image: np.ndarray) -> np.ndarray:
    """Split a strict canvas into unchanged upright 20x20 tiles."""
    value = validate_rgb(image)
    return (
        value.reshape(GRID_SIZE, TILE_SIZE, GRID_SIZE, TILE_SIZE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(TILE_COUNT, TILE_SIZE, TILE_SIZE, 3)
    )


def validate_board(board: np.ndarray) -> np.ndarray:
    """Validate the historical slot->input-tile permutation convention."""
    value = np.asarray(board)
    if value.shape != (TILE_COUNT,) or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"board must be an integer vector of length {TILE_COUNT}")
    value = value.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(value), np.arange(TILE_COUNT)):
        raise ValueError("board must be a permutation of 0..575")
    return value


def assemble_board(image: np.ndarray, board: np.ndarray) -> np.ndarray:
    """Assemble an input image under rank96's slot->input board convention."""
    tiles = split_tiles(image)
    order = validate_board(board)
    return (
        tiles[order]
        .reshape(GRID_SIZE, GRID_SIZE, TILE_SIZE, TILE_SIZE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
    )


@torch.inference_mode()
def restore_r5(image: np.ndarray, model: RestoreNet, device: torch.device) -> np.ndarray:
    """Apply the historical FP32 R5 conversion and uint8 rounding contract."""
    value = validate_rgb(image)
    source = torch.from_numpy(value).to(device=device, dtype=torch.float32)
    source = source.permute(2, 0, 1).unsqueeze(0).div_(255.0)
    output = model(source).clamp_(0.0, 1.0)
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return validate_rgb(np.rint(output * 255.0).clip(0, 255).astype(np.uint8))


def canonical_historical_nlm(image: np.ndarray) -> np.ndarray:
    """Apply the exact S1 OpenCV call, including its direct channel convention."""
    value = validate_rgb(image)
    cv2.setNumThreads(1)
    restored = cv2.fastNlMeansDenoisingColored(
        value,
        None,
        S1_TAIL_CONTRACT["nlm_h"],
        S1_TAIL_CONTRACT["nlm_h_color"],
        S1_TAIL_CONTRACT["nlm_template_window"],
        S1_TAIL_CONTRACT["nlm_search_window"],
    )
    return validate_rgb(np.asarray(restored, dtype=np.uint8))


def apply_s1_tail(
    raw_layout: np.ndarray,
    model: RestoreNet,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(R5 layout, R5->historical-NLM output)``."""
    r5_layout = restore_r5(raw_layout, model, device)
    return r5_layout, canonical_historical_nlm(r5_layout)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or image.size != (480, 480):
            raise ValueError(f"expected strict 480x480 RGB PNG: {path}")
        return validate_rgb(np.asarray(image, dtype=np.uint8))


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(validate_rgb(image), mode="RGB").save(path, format="PNG", optimize=False)


def deterministic_zip(output_dir: Path, names: Sequence[str], destination: Path) -> str:
    """Package root-level PNGs under the historical S1 timestamp contract."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in names:
            source = output_dir / name
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 14, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(
                info,
                source.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return sha256_file(destination)
