"""Directional full-tile local-offset classifier for shuffled puzzle fragments.

``OffsetPoseNet`` scores an *ordered* pair of corrupted 20x20 fragments.  Its
49-way label space is deliberately relative: the forty-eight non-zero offsets
in ``[-3, 3] x [-3, 3]`` each get a separate class, while every other relative
position maps to ``far``.  The raw logits are interpreted hierarchically:
``logsumexp`` over the 48 local directions competes with the far logit, then a
conditional distribution chooses the direction only when the pair is local.
Swapping the two inputs therefore changes the target from ``(dr, dc)`` to
``(-dr, -dc)``; this is not an undirected compatibility score.

The network consumes both complete tiles (raw RGB plus per-tile exposure
normalised RGB).  It does not crop seams or restrict its receptive field to
edges, so it can use texture, colour continuation, and local scene structure
when deciding a direction.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


OFFSET_RADIUS = 3
OFFSET_SIDE = 2 * OFFSET_RADIUS + 1
LOCAL_CLASS_COUNT = OFFSET_SIDE * OFFSET_SIDE - 1
FAR_CLASS = LOCAL_CLASS_COUNT
NUM_CLASSES = FAR_CLASS + 1

# Stable row-major mapping, with the zero displacement omitted.  Persisting
# this exact order in checkpoints makes a trained classifier easy to consume by
# a later graph optimizer without relying on a source-code version.
CLASS_OFFSETS: tuple[tuple[int, int], ...] = tuple(
    (delta_row, delta_col)
    for delta_row in range(-OFFSET_RADIUS, OFFSET_RADIUS + 1)
    for delta_col in range(-OFFSET_RADIUS, OFFSET_RADIUS + 1)
    if (delta_row, delta_col) != (0, 0)
)
OFFSET_TO_CLASS = {offset: index for index, offset in enumerate(CLASS_OFFSETS)}

if len(CLASS_OFFSETS) != LOCAL_CLASS_COUNT or len(OFFSET_TO_CLASS) != LOCAL_CLASS_COUNT:
    raise RuntimeError("relative-offset class mapping is malformed")


def _groups(channels: int, maximum: int = 8) -> int:
    """Return a small GroupNorm group count that divides ``channels``."""
    for groups in range(min(channels, maximum), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def offsets_to_classes(delta_row: Tensor, delta_col: Tensor) -> Tensor:
    """Map relative row/column tensors to local-direction classes or ``far``.

    Values with either magnitude above three, and the zero displacement, return
    ``FAR_CLASS``.  Inputs must be integer-like tensors with identical shapes;
    the returned long tensor has that same shape and lives on the input device.
    """
    if delta_row.shape != delta_col.shape:
        raise ValueError(
            "delta_row and delta_col must have equal shapes, got "
            f"{tuple(delta_row.shape)} and {tuple(delta_col.shape)}"
        )
    if delta_row.device != delta_col.device:
        raise ValueError("delta_row and delta_col must live on the same device")
    row = delta_row.long()
    col = delta_col.long()
    local = (
        row.abs().le(OFFSET_RADIUS)
        & col.abs().le(OFFSET_RADIUS)
        & (row.ne(0) | col.ne(0))
    )
    # Flatten the 7x7 coordinate plane then close the hole at (0,0).  The
    # arithmetic agrees exactly with CLASS_OFFSETS' row-major construction.
    flat = (row + OFFSET_RADIUS) * OFFSET_SIDE + (col + OFFSET_RADIUS)
    local_class = torch.where(flat < FAR_CLASS // 2, flat, flat - 1)
    return torch.where(local, local_class, torch.full_like(local_class, FAR_CLASS))


def classes_to_offsets(classes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Decode classes to ``(delta_row, delta_col, is_local)`` tensors.

    Far predictions receive zero deltas and ``is_local=False``.  Keeping the
    far sentinel explicit prevents an optimiser from accidentally treating it
    as the geometric centre of the local 7x7 neighbourhood.
    """
    labels = classes.long()
    if torch.any(labels < 0) or torch.any(labels >= NUM_CLASSES):
        raise ValueError(f"class labels must lie in [0,{NUM_CLASSES - 1}]")
    device = labels.device
    table = torch.tensor(CLASS_OFFSETS, dtype=torch.long, device=device)
    local = labels.ne(FAR_CLASS)
    safe = labels.clamp(max=FAR_CLASS - 1)
    offsets = table[safe]
    zeros = torch.zeros_like(offsets[..., 0])
    return (
        torch.where(local, offsets[..., 0], zeros),
        torch.where(local, offsets[..., 1], zeros),
        local,
    )


def inverse_classes(classes: Tensor) -> Tensor:
    """Return the directional inverse class; ``far`` remains ``far``."""
    row, col, local = classes_to_offsets(classes)
    inverse = offsets_to_classes(-row, -col)
    return torch.where(local, inverse, torch.full_like(inverse, FAR_CLASS))


def aggregate_local_logit(logits: Tensor) -> Tensor:
    """Return the one local-vs-far aggregate logit for raw 49-way outputs.

    The 48 directional logits collectively represent the local hypothesis, so
    comparing only their individual argmax to ``far`` would bias prediction
    toward the single far class.  This helper intentionally preserves the raw
    49-logit checkpoint contract while exposing the correct binary evidence.
    """
    if logits.ndim < 1 or logits.shape[-1] != NUM_CLASSES:
        raise ValueError(f"logits must end in {NUM_CLASSES} classes, got {tuple(logits.shape)}")
    return torch.logsumexp(logits[..., :LOCAL_CLASS_COUNT].float(), dim=-1)


def hierarchical_probabilities(logits: Tensor) -> dict[str, Tensor]:
    """Decode raw logits into local/far and conditional-direction probabilities.

    ``local_probability`` is a binary softmax between the aggregate local
    evidence and the single far logit.  ``conditional_offset_probabilities``
    is a 48-way softmax conditional on being local.  Their product recovers
    the local entries of the ordinary 49-way softmax exactly (up to rounding).
    """
    local_logit = aggregate_local_logit(logits)
    far_logit = logits[..., FAR_CLASS].float()
    binary_logits = torch.stack((far_logit, local_logit), dim=-1)
    binary_probabilities = binary_logits.softmax(dim=-1)
    local_probability = binary_probabilities[..., 1]
    far_probability = binary_probabilities[..., 0]
    conditional_offset_probabilities = logits[..., :LOCAL_CLASS_COUNT].float().softmax(dim=-1)
    raw_probabilities = logits.float().softmax(dim=-1)
    return {
        "aggregate_local_logit": local_logit,
        "far_logit": far_logit,
        "binary_logits": binary_logits,
        "local_probability": local_probability,
        "far_probability": far_probability,
        "conditional_offset_probabilities": conditional_offset_probabilities,
        "raw_probabilities": raw_probabilities,
    }


def hierarchical_predictions(
    logits: Tensor, *, local_threshold: float = 0.5
) -> dict[str, Tensor]:
    """Apply a local-confidence threshold then decode a conditional direction.

    ``classes`` remains in the original 49-class space for downstream
    checkpoint/inference compatibility.  ``confidence`` is geometric-edge
    confidence: ``P(local) * max P(offset | local)`` rather than confidence in
    the dominant far class.
    """
    if not 0.0 <= local_threshold <= 1.0:
        raise ValueError("local_threshold must lie in [0,1]")
    output = hierarchical_probabilities(logits)
    conditional_confidence, conditional_offset_class = output[
        "conditional_offset_probabilities"
    ].max(dim=-1)
    predicted_local = output["local_probability"].ge(float(local_threshold))
    classes = torch.where(
        predicted_local,
        conditional_offset_class,
        torch.full_like(conditional_offset_class, FAR_CLASS),
    )
    output.update(
        {
            "conditional_offset_class": conditional_offset_class,
            "conditional_offset_confidence": conditional_confidence,
            "predicted_local": predicted_local,
            "classes": classes,
            "confidence": output["local_probability"] * conditional_confidence,
        }
    )
    return output


class _ResidualBlock(nn.Module):
    """Compact full-field residual block for tiny paired images."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _groups(channels)
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.layers(x))


class OffsetPoseNet(nn.Module):
    """Classify the exact local displacement of an ordered pair of full tiles.

    ``left`` is tile ``i`` and ``right`` is tile ``j``.  Both inputs have shape
    ``(pairs, 3, 20, 20)`` and are concatenated in that order, so a prediction
    is intentionally direction-sensitive.  Each tile also contributes a
    per-tile zero-mean, unit-RMS version; that counteracts independent exposure
    changes while the raw channels preserve useful colour and brightness cues.
    """

    def __init__(
        self,
        *,
        tile_size: int = 20,
        width: int = 48,
        dropout: float = 0.10,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        if tile_size < 4:
            raise ValueError("tile_size must be at least 4")
        if width < 4:
            raise ValueError("width must be at least 4")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if num_classes != NUM_CLASSES:
            raise ValueError(
                f"OffsetPoseNet's fixed label space has {NUM_CLASSES} classes, got {num_classes}"
            )

        middle = width * 2
        final = width * 3
        self.tile_size = int(tile_size)
        self.width = int(width)
        self.dropout = float(dropout)
        self.num_classes = int(num_classes)

        # 12 channels: ordered raw RGB i/j plus independently normalised RGB
        # i/j.  Every convolution has access to the entire 20x20 pair field.
        self.stem = nn.Sequential(
            nn.Conv2d(12, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
        )
        self.block1 = _ResidualBlock(width)
        self.down1 = nn.Sequential(
            nn.Conv2d(width, middle, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(middle), middle),
            nn.GELU(),
        )
        self.block2 = _ResidualBlock(middle)
        self.down2 = nn.Sequential(
            nn.Conv2d(middle, final, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(final), final),
            nn.GELU(),
        )
        self.block3 = _ResidualBlock(final)
        self.head = nn.Sequential(
            nn.LayerNorm(final * 3),
            nn.Linear(final * 3, final * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final * 2, num_classes),
        )

    def _check_pair(self, left: Tensor, right: Tensor) -> None:
        expected_tail = (3, self.tile_size, self.tile_size)
        if left.ndim != 4 or tuple(left.shape[1:]) != expected_tail:
            raise ValueError(
                f"left must have shape (pairs,{expected_tail[0]},{self.tile_size},{self.tile_size}), "
                f"got {tuple(left.shape)}"
            )
        if right.shape != left.shape:
            raise ValueError(
                f"right must have exactly left's shape, got {tuple(right.shape)} vs {tuple(left.shape)}"
            )
        if left.device != right.device:
            raise ValueError("left and right must live on the same device")
        if not torch.is_floating_point(left) or not torch.is_floating_point(right):
            raise TypeError("left and right must be floating point tensors")

    @staticmethod
    def _exposure_normalize(tile: Tensor) -> Tensor:
        """Normalise per-tile exposure without discarding ordered raw RGB."""
        mean = tile.mean(dim=(-3, -2, -1), keepdim=True)
        rms = (tile - mean).square().mean(dim=(-3, -2, -1), keepdim=True).add(1.0e-5).sqrt()
        # Soft clipping avoids an occasional nearly-flat JPEG tile dominating a
        # pair while retaining the spatial signal across its full field.
        return ((tile - mean) / rms).clamp(-5.0, 5.0)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        """Return ordered-pair logits with shape ``(pairs, 49)``."""
        self._check_pair(left, right)
        if left.shape[0] == 0:
            return left.new_empty((0, self.num_classes))
        pair = torch.cat(
            (left, right, self._exposure_normalize(left), self._exposure_normalize(right)),
            dim=1,
        )
        x = self.block1(self.stem(pair))
        x = self.block2(self.down1(x))
        x = self.block3(self.down2(x))
        spatial = x.flatten(start_dim=2)
        # Mean, standard deviation, and maximum all aggregate *all* spatial
        # locations.  The representation is not an edge-strip descriptor.
        pooled = torch.cat(
            (
                spatial.mean(dim=-1),
                spatial.var(dim=-1, unbiased=False).add(1.0e-6).sqrt(),
                spatial.amax(dim=-1),
            ),
            dim=-1,
        )
        return self.head(pooled)

    @torch.no_grad()
    def predict(
        self, left: Tensor, right: Tensor, *, local_threshold: float = 0.5
    ) -> dict[str, Tensor]:
        """Return hierarchical local/far and conditional-direction predictions.

        The legacy-compatible ``probabilities`` key remains the ordinary raw
        49-way softmax.  Prefer ``local_probability`` and
        ``conditional_offset_probabilities`` for geometric decisions.
        """
        logits = self(left, right)
        output = hierarchical_predictions(logits, local_threshold=local_threshold)
        output["logits"] = logits
        output["probabilities"] = output["raw_probabilities"]
        return output


def count_params(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def class_offsets_metadata() -> list[list[int] | None]:
    """Return JSON/torch-save friendly class-to-offset metadata including far."""
    return [[int(row), int(col)] for row, col in CLASS_OFFSETS] + [None]


def smoke(device: torch.device | str = "cpu") -> dict[str, object]:
    """Run shape and directional-label checks without puzzle data."""
    torch.manual_seed(1234)
    model = OffsetPoseNet().to(device).eval()
    left = torch.rand(3, 3, 20, 20, device=device)
    right = torch.rand(3, 3, 20, 20, device=device)
    with torch.inference_mode():
        logits = model(left, right)
        prediction = model.predict(left, right)
    if tuple(logits.shape) != (3, NUM_CLASSES):
        raise AssertionError(f"unexpected logits shape {tuple(logits.shape)}")
    labels = offsets_to_classes(
        torch.tensor([-3, 0, 3, 4], device=device),
        torch.tensor([2, -1, -3, 0], device=device),
    )
    expected = torch.tensor(
        [OFFSET_TO_CLASS[(-3, 2)], OFFSET_TO_CLASS[(0, -1)], OFFSET_TO_CLASS[(3, -3)], FAR_CLASS],
        device=device,
    )
    if not torch.equal(labels, expected):
        raise AssertionError(f"directional class mapping mismatch: {labels} vs {expected}")
    if not torch.equal(inverse_classes(labels[:3]), offsets_to_classes(torch.tensor([3, 0, -3], device=device), torch.tensor([-2, 1, 3], device=device))):
        raise AssertionError("inverse class mapping is inconsistent")
    # The key hierarchical guard: no individual local direction has to exceed
    # the far logit for their *aggregate* local evidence to win.
    aggregate_probe = torch.full((1, NUM_CLASSES), -2.0, device=device)
    aggregate_probe[..., FAR_CLASS] = 0.0
    hierarchy = hierarchical_predictions(aggregate_probe)
    if not bool(hierarchy["predicted_local"].item()):
        raise AssertionError("aggregate local evidence did not beat the far logit")
    if int(hierarchy["classes"].item()) == FAR_CLASS:
        raise AssertionError("hierarchical decoder incorrectly emitted far")
    return {
        "logits": tuple(logits.shape),
        "prediction": tuple(prediction["classes"].shape),
        "hierarchical_local_probability": float(hierarchy["local_probability"].item()),
        "parameters": count_params(model),
    }


__all__: Sequence[str] = (
    "OFFSET_RADIUS",
    "OFFSET_SIDE",
    "LOCAL_CLASS_COUNT",
    "FAR_CLASS",
    "NUM_CLASSES",
    "CLASS_OFFSETS",
    "OFFSET_TO_CLASS",
    "offsets_to_classes",
    "classes_to_offsets",
    "inverse_classes",
    "aggregate_local_logit",
    "hierarchical_probabilities",
    "hierarchical_predictions",
    "OffsetPoseNet",
    "count_params",
    "class_offsets_metadata",
    "smoke",
)


if __name__ == "__main__":
    print(smoke())
