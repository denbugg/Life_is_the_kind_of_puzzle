"""Legal, target-free frontend for the historical TASKA seam matchers.

This module is a deliberately narrow port of the useful matcher path at
``pazzle_will_be_killed@ae9d231``.  It accepts only the current bag of raw
20x20 RGB fragments and produces:

* calibrated right/down log-assignment matrices (higher is better);
* the exact lower-is-better matrices expected by the raw-tail solver; and
* a mutual-best edge harvest voted across v3/local, raw/median/bilateral, and
  two orientations.

There is no target, recovered label map, filename, border oracle, chooser,
verifier, or pixel renderer in this API.  Checkpoint pickle deserialization is
allowed only after a known SHA-256 digest has been verified, then architecture
metadata and the state dictionary are loaded strictly.

The historical 2x2 ``quad_rerank`` is intentionally not reproduced.  It
excluded rows using ``tile_id % 24`` and ``tile_id // 24`` after historical
validation had target-relabelled the input bag.  That path is not invariant to
renumbering an otherwise identical input bag.  ``quad_weight`` therefore has
to stay zero in the legal frontend.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge

RING_SIGMA = 13.4
TILE_SIZE = 20
ORIENTATIONS: tuple[tuple[int, int, int], ...] = tuple(
    (transpose, left_right, up_down)
    for transpose in (0, 1)
    for left_right in (0, 1)
    for up_down in (0, 1)
)


class TaskaCheckpointError(ValueError):
    """A checkpoint is not the exact audited artifact it claims to be."""


@dataclass(frozen=True)
class TaskaCheckpointSpec:
    """Immutable identity and architecture manifest for an audited weight file."""

    kind: Literal["v3", "local"]
    filename: str
    sha256: str
    args: Mapping[str, Any]
    step: int


_V3_ARGS: dict[str, Any] = {
    "ch": 96,
    "blocks": 6,
    "dim": 192,
    "strip": 3,
    "head": "global",
    "restored": "",
    "norm_only": False,
    "invariance_weight": 0.0,
    "photo_jitter": 0.0,
    "batch": 1,
    "steps": 20000,
    "lr": 0.00015,
    "real_prob": 0.0,
    "predict_weight": 0.3,
    "twin_thr": 0.0,
    "mix": 0.0,
    "workers": 2,
    "eval_every": 2500,
    "eval_boards": 6,
    "calibrate": 0,
    "init": "seam_embed_v2.pt",
    "out": "seam_embed_v3.pt",
}

_LOCAL_ARGS: dict[str, Any] = {
    "ch": 96,
    "blocks": 6,
    "dim": 192,
    "strip": 3,
    "head": "local",
    "restored": "",
    "norm_only": False,
    "invariance_weight": 0.0,
    "photo_jitter": 0.0,
    "batch": 1,
    "steps": 24000,
    "lr": 0.0003,
    "real_prob": 0.0,
    "predict_weight": 0.3,
    "twin_thr": 0.0,
    "mix": 0.0,
    "workers": 2,
    "eval_every": 2000,
    "eval_boards": 6,
    "calibrate": 0,
    "init": "seam_embed_v3_trunk.pt",
    "out": "seam_embed_local.pt",
}

TASKA_CHECKPOINTS: Mapping[str, TaskaCheckpointSpec] = {
    "v3": TaskaCheckpointSpec(
        kind="v3",
        filename="seam_embed_v3.pt",
        sha256="6f0917d66d908f6cc0f4c1fcb949d3bcbadcba2490a6f7b5a12596e61de9730e",
        args=_V3_ARGS,
        step=20000,
    ),
    "local": TaskaCheckpointSpec(
        kind="local",
        filename="seam_embed_local.pt",
        sha256="5932853a73961d261b494368a4db04633fecc5996771c14d64f49ef00c7cfe73",
        args=_LOCAL_ARGS,
        step=24000,
    ),
}

DEFAULT_CHECKPOINT_DIR = (
    Path(__file__).resolve().parents[2] / "artifacts" / "prior-taska" / "ckpt"
)


class _Block(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.n1 = nn.GroupNorm(8, channels)
        self.n2 = nn.GroupNorm(8, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.n1(self.c1(inputs)))
        return inputs + self.n2(self.c2(hidden))


class SeamEmbed(nn.Module):
    """Checkpoint-exact seam network plus a public raw-logit protocol.

    ``right_down_logits`` consumes BCHW, uint8-range tensors and returns two
    scaled, high-is-good NxN matrices before diagonal masking or Sinkhorn.
    Callers control inference/autocast context so the same method remains useful
    to adjacent target-free modules such as a structural border prior.
    """

    checkpoint_spec: TaskaCheckpointSpec | None
    checkpoint_path: Path | None

    def __init__(
        self,
        channels: int = 64,
        blocks: int = 4,
        dimension: int = 128,
        strip: int = 3,
        head: str = "global",
        *,
        predict: bool = False,
        norm_only: bool = False,
        restored: bool = False,
    ) -> None:
        super().__init__()
        self.strip = strip
        self.predict = predict
        self.norm_only = norm_only
        self.restored = restored
        input_channels = 3 if norm_only else 6
        if restored:
            input_channels += 3
        self.stem = nn.Conv2d(input_channels, channels, 3, padding=1)
        self.trunk = nn.Sequential(*[_Block(channels) for _ in range(blocks)])
        self.head_kind = head
        if head == "local":
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(channels, channels, 3, padding=1),
                        nn.GELU(),
                        nn.Conv2d(channels, dimension, (1, strip)),
                    )
                    for _ in range(4)
                ]
            )
        elif head == "global":
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(channels, channels, 3, padding=1),
                        nn.GELU(),
                        nn.Flatten(),
                        nn.Linear(channels * strip * TILE_SIZE, dimension),
                    )
                    for _ in range(4)
                ]
            )
        else:
            raise ValueError("head must be 'global' or 'local'")
        self.rows: Sequence[int] | None = None
        self.logit_scale = nn.Parameter(torch.tensor(2.5))
        self.modes = 1
        if predict:
            self.pred = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(channels, channels, 3, padding=1),
                        nn.GELU(),
                        nn.Conv2d(channels, 3 * strip, (1, strip)),
                    )
                    for _ in range(2)
                ]
            )
        self.checkpoint_spec = None
        self.checkpoint_path = None

    def prep(
        self,
        inputs: torch.Tensor,
        restored: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flat = inputs.flatten(2)
        mean = flat.mean(-1)[:, :, None, None]
        variance = flat.var(-1)[:, :, None, None] - RING_SIGMA**2
        minimum = (0.25 * RING_SIGMA) ** 2
        standard_deviation = torch.sqrt(torch.clamp(variance, min=minimum))
        normalised = (inputs - mean) / standard_deviation / 4.0
        views = [normalised] if self.norm_only else [inputs / 255.0 - 0.5, normalised]
        if self.restored:
            if restored is None:
                raise ValueError("this matcher requires its restored input view")
            views.append(restored / 255.0 - 0.5)
        return views[0] if len(views) == 1 else torch.cat(views, dim=1)

    def forward(
        self,
        inputs: torch.Tensor,
        restored: torch.Tensor | None = None,
    ) -> list[Any]:
        features = self.trunk(self.stem(self.prep(inputs, restored)))
        strip = self.strip
        strips = (
            features[:, :, :, -strip:],
            features[:, :, :, :strip],
            features[:, :, -strip:, :].transpose(2, 3),
            features[:, :, :strip, :].transpose(2, 3),
        )
        outputs: list[Any] = []
        for head, current_strip in zip(self.heads, strips, strict=True):
            descriptor = head(current_strip)
            if self.head_kind == "local":
                if self.rows is not None:
                    descriptor = descriptor[:, :, self.rows]
                descriptor = descriptor.flatten(1)
            outputs.append(F.normalize(descriptor, dim=-1))
        if self.predict:
            outputs.append(
                [
                    predictor(current_strip).reshape(
                        features.shape[0], 3, strip, TILE_SIZE
                    )
                    for predictor, current_strip in zip(
                        self.pred,
                        (strips[0], strips[2]),
                        strict=True,
                    )
                ]
            )
        return outputs

    def right_down_logits(
        self,
        tiles_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw scaled right/down logits before mask and Sinkhorn."""

        descriptors = [value.float() for value in self(tiles_tensor)[:4]]
        scale = self.logit_scale.exp().detach().float()
        right = board_logits(descriptors, "right", self.modes).float() * scale
        down = board_logits(descriptors, "down", self.modes).float() * scale
        return right, down


def board_logits(
    descriptors: Sequence[torch.Tensor],
    axis: Literal["right", "down", "h", "v"],
    modes: int = 1,
    mode_tau: float = 0.0,
) -> torch.Tensor:
    """Entry (i,j) scores fragment ``j`` after fragment ``i`` on ``axis``."""

    right, left, down, up = descriptors
    first, second = (right, left) if axis in {"right", "h"} else (down, up)
    if modes <= 1:
        return first @ second.t()
    count, dimension = first.shape
    if dimension % modes:
        raise ValueError("descriptor dimension must be divisible by modes")
    per_mode = dimension // modes
    first = first.reshape(count, modes, per_mode)
    second = second.reshape(count, modes, per_mode)
    scores = torch.einsum("ikd,jkd->kij", first, second)
    if mode_tau > 0:
        return mode_tau * torch.logsumexp(scores / mode_tau, dim=0)
    return scores.amax(0)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        while block := checkpoint_file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or str(device) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    resolved = torch.device(device)
    if resolved.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be cpu, cuda, mps, or auto")
    return resolved


def _autocast_context(device: torch.device) -> AbstractContextManager[Any]:
    # The historical implementation hard-coded CUDA autocast.  CPU and MPS
    # stay in float32: it is supported everywhere and avoids unsupported/low
    # precision normalization and convolution paths on Apple hardware.
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _validate_checkpoint_payload(
    payload: Any,
    spec: TaskaCheckpointSpec,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TaskaCheckpointError("checkpoint payload must be a mapping")
    if set(payload) != {"model", "args", "eval", "step"}:
        raise TaskaCheckpointError(
            "checkpoint top-level keys must be exactly model/args/eval/step"
        )
    args = payload["args"]
    if not isinstance(args, Mapping) or dict(args) != dict(spec.args):
        raise TaskaCheckpointError(f"{spec.kind} checkpoint metadata does not match manifest")
    if payload["step"] != spec.step:
        raise TaskaCheckpointError(f"{spec.kind} checkpoint step does not match manifest")
    evaluation = payload["eval"]
    if not isinstance(evaluation, Mapping) or set(evaluation) != {
        "R@1",
        "R@20",
        "twinR@1",
    }:
        raise TaskaCheckpointError("checkpoint evaluation metadata is malformed")
    if not all(np.isfinite(float(value)) for value in evaluation.values()):
        raise TaskaCheckpointError("checkpoint evaluation metadata must be finite")
    state = payload["model"]
    if not isinstance(state, Mapping) or not state:
        raise TaskaCheckpointError("checkpoint model state must be a non-empty mapping")
    predict_keys = any(str(key).startswith("pred.") for key in state)
    if not predict_keys:
        raise TaskaCheckpointError("audited seam checkpoint must include prediction heads")
    return state


def load_taska_checkpoint(
    path: str | Path,
    kind: Literal["v3", "local"],
    *,
    device: str | torch.device | None = "auto",
) -> SeamEmbed:
    """Verify and strictly load one audited matcher checkpoint.

    SHA-256 is checked before ``torch.load(weights_only=False)``.  This ordering
    is security-sensitive because these historical files contain NumPy scalar
    metadata and cannot be read by PyTorch's restricted weights-only unpickler.
    """

    if kind not in TASKA_CHECKPOINTS:
        raise ValueError("kind must be 'v3' or 'local'")
    spec = TASKA_CHECKPOINTS[kind]
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    actual_sha256 = _file_sha256(checkpoint_path)
    if actual_sha256 != spec.sha256:
        raise TaskaCheckpointError(
            f"{kind} SHA-256 mismatch: expected {spec.sha256}, got {actual_sha256}"
        )
    # The exact known digest is the trust gate for this otherwise unrestricted
    # pickle load. Loading to CPU first also makes CUDA/MPS behavior predictable.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _validate_checkpoint_payload(payload, spec)
    args = spec.args
    model = SeamEmbed(
        channels=int(args["ch"]),
        blocks=int(args["blocks"]),
        dimension=int(args["dim"]),
        strip=int(args["strip"]),
        head=str(args["head"]),
        predict=True,
        norm_only=bool(args["norm_only"]),
        restored=bool(args["restored"]),
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise TaskaCheckpointError(
            f"{kind} state dictionary does not match its declared architecture"
        ) from error
    model.checkpoint_spec = spec
    model.checkpoint_path = checkpoint_path
    model.requires_grad_(False)
    model.eval()
    return model.to(_resolve_device(device))


def load_default_taska_ensemble(
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    *,
    device: str | torch.device | None = "auto",
) -> tuple[SeamEmbed, SeamEmbed]:
    """Load the audited v3 + local pair used by the historical ensemble."""

    root = Path(checkpoint_dir)
    return tuple(
        load_taska_checkpoint(root / TASKA_CHECKPOINTS[kind].filename, kind, device=device)
        for kind in ("v3", "local")
    )  # type: ignore[return-value]


def analytic_view(
    name: Literal["median", "bilateral"],
    tiles: np.ndarray,
) -> np.ndarray:
    """Apply one historical non-generative filter independently to each tile."""

    source = _validated_tiles(tiles)
    output = np.empty_like(source)
    for index, tile in enumerate(source):
        uint8_tile = np.clip(tile, 0, 255).astype(np.uint8)
        if name == "median":
            filtered = cv2.medianBlur(uint8_tile, 3)
        elif name == "bilateral":
            filtered = cv2.bilateralFilter(uint8_tile, 7, 50, 7)
        else:
            raise ValueError("analytic view must be 'median' or 'bilateral'")
        output[index] = filtered.astype(np.float32)
    return output


def _validated_tiles(value: Any) -> np.ndarray:
    tiles = np.asarray(value)
    if tiles.ndim != 4 or tiles.shape[1:] != (TILE_SIZE, TILE_SIZE, 3):
        raise ValueError(f"tiles must have shape (N,{TILE_SIZE},{TILE_SIZE},3)")
    if tiles.shape[0] < 2:
        raise ValueError("at least two tiles are required")
    if not np.issubdtype(tiles.dtype, np.number):
        raise TypeError("tiles must be numeric")
    tiles = np.asarray(tiles, dtype=np.float32)
    if not np.isfinite(tiles).all():
        raise ValueError("tiles must contain only finite values")
    if float(tiles.min()) < 0.0 or float(tiles.max()) > 255.0:
        raise ValueError("tiles must be in the uint8 value range [0, 255]")
    return np.ascontiguousarray(tiles)


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("matcher must contain at least one parameter") from error


def _require_shared_device(
    matchers: Sequence[nn.Module],
    requested: str | torch.device | None,
) -> torch.device:
    devices = {_model_device(model) for model in matchers}
    if len(devices) != 1:
        raise ValueError("all matchers must be on the same device")
    actual = next(iter(devices))
    if requested is not None and str(requested) != "auto":
        expected = _resolve_device(requested)
        same_index = expected.index is None or actual.index == expected.index
        if actual.type != expected.type or not same_index:
            raise ValueError(f"matchers are on {actual}, not requested device {expected}")
    return actual


def _sink(logits: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    """Historical log-space Sinkhorn with no non-equivariant slack row."""

    result = logits
    for _ in range(iterations):
        result = result - torch.logsumexp(result, dim=1, keepdim=True)
        result = result - torch.logsumexp(result, dim=0, keepdim=True)
    return result


def _acyclic(probability: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp(1.0 - probability.t(), min=1e-6)) + torch.log(
        torch.clamp(1.0 - (probability @ probability).t(), min=1e-6)
    )


def cycle_consistency(
    right_logits: torch.Tensor,
    down_logits: torch.Tensor,
    *,
    rounds: int = 3,
    weight: float = 0.35,
    sinkhorn_iterations: int = 20,
    acyclic_weight: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Historical Sinkhorn + commuting-axis + acyclicity calibration."""

    right = _sink(right_logits, sinkhorn_iterations)
    down = _sink(down_logits, sinkhorn_iterations)
    for _ in range(rounds):
        right_probability = right.exp()
        down_probability = down.exp()
        right_evidence = torch.log(
            torch.clamp(
                down_probability @ right_probability @ down_probability.t(),
                min=1e-12,
            )
        )
        down_evidence = torch.log(
            torch.clamp(
                right_probability @ down_probability @ right_probability.t(),
                min=1e-12,
            )
        )
        if acyclic_weight > 0:
            right_evidence = right_evidence + acyclic_weight * _acyclic(right_probability)
            down_evidence = down_evidence + acyclic_weight * _acyclic(down_probability)
        right = _sink(right + weight * right_evidence, sinkhorn_iterations)
        down = _sink(down + weight * down_evidence, sinkhorn_iterations)
    return right, down


def _oriented_tiles(
    tiles: np.ndarray,
    orientation: tuple[int, int, int],
) -> np.ndarray:
    transpose, left_right, up_down = orientation
    result = tiles
    if left_right:
        result = result[:, :, ::-1]
    if up_down:
        result = result[:, ::-1]
    if transpose:
        result = result.transpose(0, 2, 1, 3)
    return np.ascontiguousarray(result)


@torch.inference_mode()
def _initial_log_assignments(
    model: nn.Module,
    tiles: np.ndarray,
    orientation: tuple[int, int, int],
    *,
    device: torch.device,
    sinkhorn_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    transformed = _oriented_tiles(tiles, orientation)
    tensor = torch.from_numpy(transformed).permute(0, 3, 1, 2).to(device)
    with _autocast_context(device):
        right, down = model.right_down_logits(tensor)  # type: ignore[attr-defined]
    right = right.float()
    down = down.float()
    right.fill_diagonal_(-1e4)
    down.fill_diagonal_(-1e4)
    right = _sink(right, sinkhorn_iterations)
    down = _sink(down, sinkhorn_iterations)
    transpose, left_right, up_down = orientation
    if transpose:
        right, down = down, right
    if left_right:
        right = right.t().contiguous()
    if up_down:
        down = down.t().contiguous()
    return right, down


@torch.inference_mode()
def calibrated_log_assignments(
    model: nn.Module,
    tiles: Any,
    *,
    device: str | torch.device | None = None,
    orientation: tuple[int, int, int] = (0, 0, 0),
    rounds: int = 3,
    cycle_weight: float = 0.35,
    sinkhorn_iterations: int = 20,
    acyclic_weight: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """One model/view/orientation as calibrated high-is-good log matrices."""

    source = _validated_tiles(tiles)
    if orientation not in ORIENTATIONS:
        raise ValueError("orientation must be one of the eight board symmetries")
    actual_device = _require_shared_device([model], device)
    right, down = _initial_log_assignments(
        model,
        source,
        orientation,
        device=actual_device,
        sinkhorn_iterations=sinkhorn_iterations,
    )
    right, down = cycle_consistency(
        right,
        down,
        rounds=rounds,
        weight=cycle_weight,
        sinkhorn_iterations=sinkhorn_iterations,
        acyclic_weight=acyclic_weight,
    )
    return (
        np.ascontiguousarray(right.cpu().numpy(), dtype=np.float64),
        np.ascontiguousarray(down.cpu().numpy(), dtype=np.float64),
    )


@torch.inference_mode()
def pessimistic_log_assignments(
    matchers: Sequence[nn.Module],
    tiles: Any,
    *,
    device: str | torch.device | None = None,
    rounds: int = 3,
    cycle_weight: float = 0.35,
    sinkhorn_iterations: int = 20,
    acyclic_weight: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse two comparable models by the historical elementwise minimum."""

    if len(matchers) != 2:
        raise ValueError("pessimistic TASKA fusion requires exactly two matchers")
    source = _validated_tiles(tiles)
    actual_device = _require_shared_device(matchers, device)
    initial = [
        _initial_log_assignments(
            model,
            source,
            (0, 0, 0),
            device=actual_device,
            sinkhorn_iterations=sinkhorn_iterations,
        )
        for model in matchers
    ]
    right = torch.stack([pair[0] for pair in initial]).amin(0)
    down = torch.stack([pair[1] for pair in initial]).amin(0)
    right, down = cycle_consistency(
        right,
        down,
        rounds=rounds,
        weight=cycle_weight,
        sinkhorn_iterations=sinkhorn_iterations,
        acyclic_weight=acyclic_weight,
    )
    return (
        np.ascontiguousarray(right.cpu().numpy(), dtype=np.float64),
        np.ascontiguousarray(down.cpu().numpy(), dtype=np.float64),
    )


@dataclass(frozen=True)
class MutualVote:
    """Target-free evidence retained for one voted candidate edge."""

    edge: RawTailEdge
    vote_count: int
    minimum_margin: float
    maximum_margin: float


@dataclass(frozen=True)
class TaskaSeamConfig:
    """The legal replay knobs and their historical fixed defaults."""

    views: tuple[str, ...] = ("raw", "median", "bilateral")
    orientations: int = 2
    votes: int = 10
    vote_target: int = 350
    margin: float = 0.0
    depth: int = 1
    quad_weight: float = 0.0
    rounds: int = 3
    cycle_weight: float = 0.35
    sinkhorn_iterations: int = 20
    acyclic_weight: float = 3.0

    def validate(self, *, scorer_count: int) -> None:
        if not self.views or self.views[0] != "raw":
            raise ValueError("views must start with 'raw'")
        if len(set(self.views)) != len(self.views) or any(
            view not in {"raw", "median", "bilateral"} for view in self.views
        ):
            raise ValueError("views may contain raw, median, and bilateral once each")
        if (
            isinstance(self.orientations, bool)
            or not isinstance(self.orientations, int)
            or not 1 <= self.orientations <= 8
        ):
            raise ValueError("orientations must be an integer in [1, 8]")
        if (
            isinstance(self.votes, bool)
            or not isinstance(self.votes, int)
            or not 1 <= self.votes <= scorer_count
        ):
            raise ValueError("votes must be an integer no larger than the scorer count")
        if (
            isinstance(self.vote_target, bool)
            or not isinstance(self.vote_target, int)
            or self.vote_target < 0
        ):
            raise ValueError("vote_target must be a non-negative integer")
        if not np.isfinite(self.margin):
            raise ValueError("margin must be finite")
        if self.depth != 1:
            raise ValueError("this legal frontend implements mutual-best depth=1 only")
        if self.quad_weight != 0.0:
            raise ValueError(
                "quad_weight must be zero: historical quad masks used target-position ids"
            )
        if (
            isinstance(self.rounds, bool)
            or not isinstance(self.rounds, int)
            or self.rounds < 0
        ):
            raise ValueError("rounds must be a non-negative integer")
        if not np.isfinite(self.cycle_weight) or self.cycle_weight < 0:
            raise ValueError("cycle_weight must be finite and non-negative")
        if (
            isinstance(self.sinkhorn_iterations, bool)
            or not isinstance(self.sinkhorn_iterations, int)
            or self.sinkhorn_iterations < 1
        ):
            raise ValueError("sinkhorn_iterations must be a positive integer")
        if not np.isfinite(self.acyclic_weight) or self.acyclic_weight < 0:
            raise ValueError("acyclic_weight must be finite and non-negative")


@dataclass(frozen=True)
class TaskaSeamMatchResult:
    """Matrices and mutual-vote candidates consumed by the discrete solver."""

    right_log: np.ndarray
    down_log: np.ndarray
    cost_right: np.ndarray
    cost_down: np.ndarray
    candidate_edges: tuple[RawTailEdge, ...]
    vote_records: tuple[MutualVote, ...]
    chosen_vote_threshold: int
    scorer_count: int
    checkpoint_sha256: tuple[str, ...]
    config: TaskaSeamConfig


def _mutual(
    matrix: np.ndarray,
    axis: Literal["right", "down"],
) -> dict[RawTailEdge, float]:
    scores = np.array(matrix, np.float64, copy=True)
    np.fill_diagonal(scores, -np.inf)
    forward = scores.argmax(axis=1)
    backward = scores.argmax(axis=0)
    partition = np.partition(scores, -2, axis=1)
    return {
        RawTailEdge(int(source), int(forward[source]), axis): float(
            partition[source, -1] - partition[source, -2]
        )
        for source in range(len(scores))
        if int(backward[int(forward[source])]) == source
    }


def _votes_for_target(
    scorer_sets: Sequence[Mapping[RawTailEdge, float]],
    target: int,
) -> int:
    counts: dict[RawTailEdge, int] = {}
    for scorer in scorer_sets:
        for edge in scorer:
            counts[edge] = counts.get(edge, 0) + 1
    for threshold in range(len(scorer_sets), 0, -1):
        if sum(count >= threshold for count in counts.values()) >= target:
            return threshold
    return 1


def _cost_from_log_assignment(log_assignment: np.ndarray) -> np.ndarray:
    cost = -np.asarray(log_assignment, dtype=np.float64)
    cost -= cost.min()
    np.fill_diagonal(cost, 0.0)
    return np.ascontiguousarray(cost)


def _read_only(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _verified_provenance(
    matchers: Sequence[nn.Module],
    *,
    require_verified: bool,
) -> tuple[str, ...]:
    specs = [getattr(model, "checkpoint_spec", None) for model in matchers]
    if all(isinstance(spec, TaskaCheckpointSpec) for spec in specs):
        typed_specs = [spec for spec in specs if isinstance(spec, TaskaCheckpointSpec)]
        if {spec.kind for spec in typed_specs} != {"v3", "local"}:
            raise TaskaCheckpointError("ensemble must contain one v3 and one local checkpoint")
        return tuple(spec.sha256 for spec in typed_specs)
    if require_verified:
        raise TaskaCheckpointError(
            "matchers must come from load_taska_checkpoint; use require_verified=False "
            "only for synthetic tests"
        )
    return tuple("unverified" for _ in matchers)


@torch.inference_mode()
def match_taska_tiles(
    tiles: Any,
    matchers: Sequence[nn.Module],
    *,
    config: TaskaSeamConfig | None = None,
    device: str | torch.device | None = None,
    require_verified: bool = True,
) -> TaskaSeamMatchResult:
    """Run the complete legal matcher/vote frontend on one unordered tile bag."""

    if len(matchers) != 2:
        raise ValueError("TASKA frontend requires the v3 + local matcher pair")
    if config is None:
        config = TaskaSeamConfig()
    source = _validated_tiles(tiles)
    actual_device = _require_shared_device(matchers, device)
    scorer_count = len(matchers) * len(config.views) * config.orientations
    config.validate(scorer_count=scorer_count)
    provenance = _verified_provenance(matchers, require_verified=require_verified)

    views: list[np.ndarray] = []
    for name in config.views:
        if name == "raw":
            views.append(source)
        else:
            views.append(analytic_view(name, source))  # type: ignore[arg-type]

    # Identity/raw initial assignments are shared with the vote pass, avoiding
    # two duplicate full-board forwards while preserving the historical order
    # of Sinkhorn, minimum fusion, and cycle consistency.
    raw_initial = [
        _initial_log_assignments(
            model,
            source,
            (0, 0, 0),
            device=actual_device,
            sinkhorn_iterations=config.sinkhorn_iterations,
        )
        for model in matchers
    ]
    fused_right = torch.stack([pair[0] for pair in raw_initial]).amin(0)
    fused_down = torch.stack([pair[1] for pair in raw_initial]).amin(0)
    fused_right, fused_down = cycle_consistency(
        fused_right,
        fused_down,
        rounds=config.rounds,
        weight=config.cycle_weight,
        sinkhorn_iterations=config.sinkhorn_iterations,
        acyclic_weight=config.acyclic_weight,
    )
    right_log = np.ascontiguousarray(fused_right.cpu().numpy(), dtype=np.float64)
    down_log = np.ascontiguousarray(fused_down.cpu().numpy(), dtype=np.float64)

    scorer_sets: list[dict[RawTailEdge, float]] = []
    for model_index, model in enumerate(matchers):
        for view_index, current_view in enumerate(views):
            for orientation_index, orientation in enumerate(
                ORIENTATIONS[: config.orientations]
            ):
                if view_index == 0 and orientation_index == 0:
                    initial_right, initial_down = raw_initial[model_index]
                else:
                    initial_right, initial_down = _initial_log_assignments(
                        model,
                        current_view,
                        orientation,
                        device=actual_device,
                        sinkhorn_iterations=config.sinkhorn_iterations,
                    )
                calibrated_right, calibrated_down = cycle_consistency(
                    initial_right,
                    initial_down,
                    rounds=config.rounds,
                    weight=config.cycle_weight,
                    sinkhorn_iterations=config.sinkhorn_iterations,
                    acyclic_weight=config.acyclic_weight,
                )
                right_numpy = calibrated_right.cpu().numpy().astype(np.float64)
                down_numpy = calibrated_down.cpu().numpy().astype(np.float64)
                scorer_sets.append(
                    {
                        **_mutual(right_numpy, "right"),
                        **_mutual(down_numpy, "down"),
                    }
                )

    threshold = (
        _votes_for_target(scorer_sets, config.vote_target)
        if config.vote_target
        else config.votes
    )
    all_edges = set().union(*(set(scorer) for scorer in scorer_sets))
    records: list[MutualVote] = []
    for edge in all_edges:
        margins = [scorer[edge] for scorer in scorer_sets if edge in scorer]
        if len(margins) >= threshold and min(margins) >= config.margin:
            records.append(
                MutualVote(
                    edge=edge,
                    vote_count=len(margins),
                    minimum_margin=float(min(margins)),
                    maximum_margin=float(max(margins)),
                )
            )
    axis_order = {"right": 0, "down": 1}
    records.sort(
        key=lambda record: (
            axis_order[record.edge.axis],
            record.edge.source,
            record.edge.target,
        )
    )
    immutable_records = tuple(records)
    return TaskaSeamMatchResult(
        right_log=_read_only(right_log),
        down_log=_read_only(down_log),
        cost_right=_read_only(_cost_from_log_assignment(right_log)),
        cost_down=_read_only(_cost_from_log_assignment(down_log)),
        candidate_edges=tuple(record.edge for record in immutable_records),
        vote_records=immutable_records,
        chosen_vote_threshold=threshold,
        scorer_count=len(scorer_sets),
        checkpoint_sha256=provenance,
        config=config,
    )


__all__ = [
    "DEFAULT_CHECKPOINT_DIR",
    "MutualVote",
    "SeamEmbed",
    "TASKA_CHECKPOINTS",
    "TaskaCheckpointError",
    "TaskaCheckpointSpec",
    "TaskaSeamConfig",
    "TaskaSeamMatchResult",
    "analytic_view",
    "board_logits",
    "calibrated_log_assignments",
    "cycle_consistency",
    "load_default_taska_ensemble",
    "load_taska_checkpoint",
    "match_taska_tiles",
    "pessimistic_log_assignments",
]
