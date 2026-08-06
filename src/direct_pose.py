"""Directional *direct-neighbour* classifier for shuffled puzzle fragments.

``DirectPoseNet`` scores an ordered pair of full corrupted 20x20 fragments.
It deliberately answers a much narrower and more useful question than the
49-way local-offset classifier: is ``j`` immediately above, below, left, or
right of ``i``?  Every diagonal, radius-two/radius-three, and distant pair is
the one ``NON_DIRECT`` class.

The five raw logits are never treated as a flat five-way softmax during
training.  The four direct-direction logits first aggregate with
``logsumexp`` and compete against the one non-direct logit.  Only examples
known to be direct supervise the conditional four-way direction head.  This
prevents the abundant non-direct candidates from winning merely because they
occupy one class while direct evidence is split four ways.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# Stable checkpoint contract.  For an ordered pair (i, j), the label denotes
# the clean-grid displacement of j relative to i.
DIRECT_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, 0),  # j is directly above i
    (+1, 0),  # j is directly below i
    (0, -1),  # j is directly left of i
    (0, +1),  # j is directly right of i
)
DIRECT_CLASS_COUNT = len(DIRECT_OFFSETS)
NON_DIRECT_CLASS = DIRECT_CLASS_COUNT
NUM_CLASSES = NON_DIRECT_CLASS + 1
OFFSET_TO_CLASS = {offset: index for index, offset in enumerate(DIRECT_OFFSETS)}

if len(OFFSET_TO_CLASS) != DIRECT_CLASS_COUNT:
    raise RuntimeError("direct-neighbour class mapping is malformed")


def _groups(channels: int, maximum: int = 8) -> int:
    """Return a small GroupNorm group count that divides ``channels``."""
    for groups in range(min(channels, maximum), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def offsets_to_direct_classes(delta_row: Tensor, delta_col: Tensor) -> Tensor:
    """Map offsets to the four cardinal classes or ``NON_DIRECT_CLASS``.

    Inputs may have any matching shape.  The resulting long tensor shares the
    input device and treats diagonals, zero displacement, and all larger
    offsets as non-direct.
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
    classes = torch.full_like(row, NON_DIRECT_CLASS)
    for index, (expected_row, expected_col) in enumerate(DIRECT_OFFSETS):
        classes[(row == expected_row) & (col == expected_col)] = index
    return classes


def classes_to_offsets(classes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Decode classes to ``(delta_row, delta_col, is_direct)`` tensors."""
    labels = classes.long()
    if torch.any(labels < 0) or torch.any(labels >= NUM_CLASSES):
        raise ValueError(f"class labels must lie in [0,{NUM_CLASSES - 1}]")
    table = torch.tensor(DIRECT_OFFSETS, dtype=torch.long, device=labels.device)
    direct = labels.ne(NON_DIRECT_CLASS)
    offsets = table[labels.clamp(max=NON_DIRECT_CLASS - 1)]
    zeros = torch.zeros_like(offsets[..., 0])
    return (
        torch.where(direct, offsets[..., 0], zeros),
        torch.where(direct, offsets[..., 1], zeros),
        direct,
    )


def inverse_classes(classes: Tensor) -> Tensor:
    """Return the cardinal inverse class; non-direct remains non-direct."""
    labels = classes.long()
    if torch.any(labels < 0) or torch.any(labels >= NUM_CLASSES):
        raise ValueError(f"class labels must lie in [0,{NUM_CLASSES - 1}]")
    # above <-> below, left <-> right, non-direct -> non-direct
    table = torch.tensor((1, 0, 3, 2, NON_DIRECT_CLASS), device=labels.device)
    return table[labels]


def aggregate_direct_logit(logits: Tensor) -> Tensor:
    """Aggregate the four direct hypotheses into one binary-evidence logit."""
    if logits.ndim < 1 or logits.shape[-1] != NUM_CLASSES:
        raise ValueError(f"logits must end in {NUM_CLASSES} classes, got {tuple(logits.shape)}")
    return torch.logsumexp(logits[..., :DIRECT_CLASS_COUNT].float(), dim=-1)


def hierarchical_probabilities(logits: Tensor) -> dict[str, Tensor]:
    """Decode raw logits into direct/non-direct and conditional directions.

    ``direct_probability`` is the calibrated binary probability from
    ``[non_direct_logit, logsumexp(direct_logits)]``.  The four-way direction
    distribution is explicitly conditional on a pair being direct.
    """
    direct_logit = aggregate_direct_logit(logits)
    non_direct_logit = logits[..., NON_DIRECT_CLASS].float()
    binary_logits = torch.stack((non_direct_logit, direct_logit), dim=-1)
    binary_probabilities = binary_logits.softmax(dim=-1)
    direct_probability = binary_probabilities[..., 1]
    non_direct_probability = binary_probabilities[..., 0]
    conditional_direction_probabilities = logits[..., :DIRECT_CLASS_COUNT].float().softmax(dim=-1)
    return {
        "aggregate_direct_logit": direct_logit,
        "non_direct_logit": non_direct_logit,
        "binary_logits": binary_logits,
        "direct_probability": direct_probability,
        "non_direct_probability": non_direct_probability,
        "conditional_direction_probabilities": conditional_direction_probabilities,
        # Retained for diagnostics only.  Do not use this flat five-way softmax
        # to decide whether a candidate is direct.
        "raw_probabilities": logits.float().softmax(dim=-1),
    }


def hierarchical_predictions(
    logits: Tensor, *, direct_threshold: float = 0.5
) -> dict[str, Tensor]:
    """Threshold direct evidence and then decode the conditional direction."""
    if not 0.0 <= direct_threshold <= 1.0:
        raise ValueError("direct_threshold must lie in [0,1]")
    output = hierarchical_probabilities(logits)
    conditional_confidence, conditional_direction_class = output[
        "conditional_direction_probabilities"
    ].max(dim=-1)
    predicted_direct = output["direct_probability"].ge(float(direct_threshold))
    classes = torch.where(
        predicted_direct,
        conditional_direction_class,
        torch.full_like(conditional_direction_class, NON_DIRECT_CLASS),
    )
    output.update(
        {
            "conditional_direction_class": conditional_direction_class,
            "conditional_direction_confidence": conditional_confidence,
            "predicted_direct": predicted_direct,
            "classes": classes,
            # A directed assembly-edge confidence: a pair must be direct *and*
            # have one confident conditional cardinal direction.
            "confidence": output["direct_probability"] * conditional_confidence,
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


class DirectPoseNet(nn.Module):
    """Classify the direct cardinal relation of an ordered full-tile pair.

    Both inputs have shape ``(pairs, 3, 20, 20)`` and remain ordered: swapping
    them changes above to below and left to right.  The CNN sees raw RGB and a
    per-tile exposure-normalised copy of each full tile.  It is intentionally
    not an edge-strip matcher.
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
                f"DirectPoseNet's fixed label space has {NUM_CLASSES} classes, got {num_classes}"
            )

        middle = width * 2
        final = width * 3
        self.tile_size = int(tile_size)
        self.width = int(width)
        self.dropout = float(dropout)
        self.num_classes = int(num_classes)

        # 12 channels: ordered raw RGB i/j plus independently normalised RGB
        # i/j.  Every convolution can use the entire 20x20 pair field.
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
            nn.Linear(final * 2, NUM_CLASSES),
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
        return ((tile - mean) / rms).clamp(-5.0, 5.0)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        """Return ordered-pair logits with shape ``(pairs, 5)``."""
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
        self, left: Tensor, right: Tensor, *, direct_threshold: float = 0.5
    ) -> dict[str, Tensor]:
        """Return hierarchical direct/non-direct and direction predictions."""
        logits = self(left, right)
        output = hierarchical_predictions(logits, direct_threshold=direct_threshold)
        output["logits"] = logits
        output["probabilities"] = output["raw_probabilities"]
        return output


def count_params(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def class_offsets_metadata() -> list[list[int] | None]:
    """Return JSON/torch-save friendly class-to-offset metadata."""
    return [[int(row), int(col)] for row, col in DIRECT_OFFSETS] + [None]


def smoke(device: torch.device | str = "cpu") -> dict[str, object]:
    """Run shape, inverse-label, and hierarchical-evidence checks."""
    torch.manual_seed(1234)
    model = DirectPoseNet(width=8).to(device).eval()
    left = torch.rand(3, 3, 20, 20, device=device)
    right = torch.rand(3, 3, 20, 20, device=device)
    with torch.inference_mode():
        logits = model(left, right)
        prediction = model.predict(left, right)
    if tuple(logits.shape) != (3, NUM_CLASSES):
        raise AssertionError(f"unexpected logits shape {tuple(logits.shape)}")
    labels = offsets_to_direct_classes(
        torch.tensor([-1, 1, 0, 0, 2, 1], device=device),
        torch.tensor([0, 0, -1, 1, 0, 1], device=device),
    )
    expected = torch.tensor((0, 1, 2, 3, NON_DIRECT_CLASS, NON_DIRECT_CLASS), device=device)
    if not torch.equal(labels, expected):
        raise AssertionError(f"direct class mapping mismatch: {labels} vs {expected}")
    inverses = inverse_classes(labels[:4])
    if not torch.equal(inverses, torch.tensor((1, 0, 3, 2), device=device)):
        raise AssertionError("direct inverse mapping is inconsistent")
    # Critical guard against flat-5-way collapse: four individually weak direct
    # logits together can still beat the one non-direct logit.
    aggregate_probe = torch.full((1, NUM_CLASSES), -1.0, device=device)
    aggregate_probe[..., NON_DIRECT_CLASS] = 0.0
    hierarchy = hierarchical_predictions(aggregate_probe)
    if not bool(hierarchy["predicted_direct"].item()):
        raise AssertionError("aggregate direct evidence did not beat non-direct")
    return {
        "logits": tuple(logits.shape),
        "prediction": tuple(prediction["classes"].shape),
        "hierarchical_direct_probability": float(hierarchy["direct_probability"].item()),
        "parameters": count_params(model),
    }


__all__: Sequence[str] = (
    "DIRECT_OFFSETS",
    "DIRECT_CLASS_COUNT",
    "NON_DIRECT_CLASS",
    "NUM_CLASSES",
    "OFFSET_TO_CLASS",
    "offsets_to_direct_classes",
    "classes_to_offsets",
    "inverse_classes",
    "aggregate_direct_logit",
    "hierarchical_probabilities",
    "hierarchical_predictions",
    "DirectPoseNet",
    "count_params",
    "class_offsets_metadata",
    "smoke",
)


if __name__ == "__main__":
    print(smoke())
