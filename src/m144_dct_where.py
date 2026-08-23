"""Scientific core for the M144 deterministic DCT ``what -> where`` gate.

This module is deliberately free of dataset paths, checkpoint I/O, training
loops, and report writing.  It contains only the representation, rendering,
metric, and decision primitives shared by the M144 runner and verifier.

The experiment predicts a low-frequency residual above an input-derived flat
colour.  A zero prediction must therefore render *exactly* as that flat image;
this makes the generic-prior baseline explicit and prevents interpolation from
creating an accidental positive score.  Tile embeddings are consumed as an
unordered set by learned semantic slots.  There are no tile-position
embeddings, rotations, or hidden coordinate inputs in this core.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


DCT_SIDE = 16
DCT_COEFFICIENTS = 32
CHANNELS = 3
DCT_OUTPUT_DIM = CHANNELS * DCT_COEFFICIENTS
RGB_FIELD_SIDE = 8
RGB_OUTPUT_DIM = CHANNELS * RGB_FIELD_SIDE * RGB_FIELD_SIDE

SSIM_WINDOW = 7
BOOTSTRAP_SEED = 144_032
BOOTSTRAP_SAMPLES = 10_000


@dataclass(frozen=True)
class GateThresholds:
    """Predeclared scalar thresholds for one M144 decision stage."""

    oracle_gain: float | None
    full_gain: float
    full_minus_blind: float
    full_minus_swapped: float
    representation_delta: float
    win_fraction: float | None
    confidence: float
    require_swap_lower: bool
    require_representation_lower: bool


CAL_THRESHOLDS = GateThresholds(
    oracle_gain=0.040,
    full_gain=0.008,
    full_minus_blind=0.003,
    full_minus_swapped=0.002,
    representation_delta=0.001,
    win_fraction=None,
    confidence=0.90,
    require_swap_lower=False,
    require_representation_lower=False,
)

DEV_THRESHOLDS = GateThresholds(
    oracle_gain=None,
    full_gain=0.012,
    full_minus_blind=0.005,
    full_minus_swapped=0.003,
    representation_delta=0.003,
    win_fraction=0.60,
    confidence=0.95,
    require_swap_lower=True,
    require_representation_lower=True,
)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def zigzag_indices(side: int, count: int | None = None) -> tuple[tuple[int, int], ...]:
    """Return the canonical top-left-first JPEG-style zigzag coordinates.

    The first entries are ``(0,0), (0,1), (1,0), (2,0), (1,1), (0,2)``.
    ``count`` defaults to the complete square.
    """

    side = _positive_int(side, "side")
    if count is None:
        count = side * side
    count = _positive_int(count, "count")
    if count > side * side:
        raise ValueError("count cannot exceed side * side")

    result: list[tuple[int, int]] = []
    for diagonal in range(2 * side - 1):
        row_min = max(0, diagonal - side + 1)
        row_max = min(side - 1, diagonal)
        rows = range(row_max, row_min - 1, -1) if diagonal % 2 == 0 else range(row_min, row_max + 1)
        for row in rows:
            result.append((row, diagonal - row))
            if len(result) == count:
                return tuple(result)
    raise AssertionError("zigzag construction did not produce the requested count")


def orthonormal_dct_matrix(size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Construct the orthonormal DCT-II analysis matrix."""

    size = _positive_int(size, "size")
    if not dtype.is_floating_point:
        raise TypeError("DCT requires a floating-point dtype")
    samples = torch.arange(size, device=device, dtype=dtype) + 0.5
    frequencies = torch.arange(size, device=device, dtype=dtype).unsqueeze(1)
    matrix = torch.cos((torch.pi / float(size)) * frequencies * samples)
    scale = torch.full((size,), (2.0 / float(size)) ** 0.5, device=device, dtype=dtype)
    scale[0] = (1.0 / float(size)) ** 0.5
    return matrix * scale.unsqueeze(1)


def _spatial_matrix_pair(x: Tensor) -> tuple[Tensor, Tensor]:
    if x.ndim < 2:
        raise ValueError("DCT input needs at least two spatial dimensions")
    if not x.dtype.is_floating_point:
        raise TypeError("DCT input must be floating point")
    height, width = int(x.shape[-2]), int(x.shape[-1])
    if height <= 0 or width <= 0:
        raise ValueError("DCT spatial dimensions must be non-empty")
    return (
        orthonormal_dct_matrix(height, device=x.device, dtype=x.dtype),
        orthonormal_dct_matrix(width, device=x.device, dtype=x.dtype),
    )


def dct_2d(x: Tensor) -> Tensor:
    """Apply an orthonormal DCT-II over the final two dimensions."""

    vertical, horizontal = _spatial_matrix_pair(x)
    return torch.matmul(torch.matmul(vertical, x), horizontal.transpose(0, 1))


def idct_2d(coefficients: Tensor) -> Tensor:
    """Invert :func:`dct_2d` over the final two dimensions."""

    vertical, horizontal = _spatial_matrix_pair(coefficients)
    return torch.matmul(
        torch.matmul(vertical.transpose(0, 1), coefficients),
        horizontal,
    )


def _flat_rgb(flat_rgb: Tensor, *, batch: int, channels: int = CHANNELS) -> Tensor:
    if flat_rgb.ndim == 4 and tuple(flat_rgb.shape[-2:]) == (1, 1):
        flat_rgb = flat_rgb[..., 0, 0]
    if flat_rgb.ndim != 2 or tuple(flat_rgb.shape) != (batch, channels):
        raise ValueError(f"flat_rgb must have shape ({batch},{channels})")
    if not flat_rgb.dtype.is_floating_point:
        raise TypeError("flat_rgb must be floating point")
    return flat_rgb


def flat_rgb_from_tiles(tiles: Tensor) -> Tensor:
    """Return each dirty bag's exact per-channel mean.

    ``tiles`` is ``(B,T,C,H,W)``.  No clipping, rounding, or colour transform is
    applied, so callers control whether values live in 0..1 or 0..255.
    """

    if tiles.ndim != 5 or tiles.shape[2] != CHANNELS:
        raise ValueError("tiles must have shape (B,T,3,H,W)")
    if not tiles.dtype.is_floating_point:
        raise TypeError("tiles must be floating point")
    return tiles.mean(dim=(1, 3, 4))


def _validate_image(target: Tensor) -> None:
    if target.ndim != 4 or target.shape[1] != CHANNELS:
        raise ValueError("image must have shape (B,3,H,W)")
    if min(int(target.shape[-2]), int(target.shape[-1])) < 1:
        raise ValueError("image spatial dimensions cannot be empty")
    if not target.dtype.is_floating_point:
        raise TypeError("image must be floating point")


def _resize_shape(size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(size, int):
        value = _positive_int(size, "size")
        return value, value
    if len(size) != 2:
        raise ValueError("size sequence must contain height and width")
    return _positive_int(int(size[0]), "height"), _positive_int(int(size[1]), "width")


@lru_cache(maxsize=32)
def _cpu_bicubic_interpolation_matrix(
    source_size: int, target_size: int, dtype: torch.dtype
) -> Tensor:
    """Build one axis of PyTorch's bicubic resize from one-hot basis vectors.

    Matrix construction is deliberately CPU-only and outside autograd.  The
    resulting resize is a pair of ordinary matrix multiplications, avoiding
    CUDA's non-deterministic ``upsample_bicubic2d_backward`` while preserving
    its ``align_corners=False`` forward convention.
    """

    if dtype not in (torch.float32, torch.float64):
        raise TypeError("cached bicubic bases support float32 or float64")
    basis = torch.eye(source_size, dtype=dtype).reshape(source_size, 1, source_size, 1)
    with torch.no_grad():
        resized = F.interpolate(
            basis,
            size=(target_size, 1),
            mode="bicubic",
            align_corners=False,
        )
    # Batch item j is the output produced by source basis vector j, hence the
    # transpose gives W[target_coordinate, source_coordinate].
    return resized[:, 0, :, 0].transpose(0, 1).contiguous()


def bicubic_interpolation_matrix(
    source_size: int,
    target_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    """Return the frozen 1-D bicubic interpolation matrix ``(target, source)``.

    Float32 is the production M144 rendering contract.  Float64 keeps the core
    useful for high-precision parity tests; lower-precision callers receive the
    frozen float32 weights cast to their requested dtype.
    """

    source_size = _positive_int(source_size, "source_size")
    target_size = _positive_int(target_size, "target_size")
    if not dtype.is_floating_point:
        raise TypeError("bicubic interpolation requires a floating-point dtype")
    basis_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    matrix = _cpu_bicubic_interpolation_matrix(source_size, target_size, basis_dtype)
    return matrix.to(device=torch.device(device), dtype=dtype)


def fixed_bicubic_resize(value: Tensor, size: int | Sequence[int]) -> Tensor:
    """Deterministically resize the final two dimensions with frozen bicubic W.

    For production sides 8 and 16 at 480 px this separable implementation
    agrees with ``torch.nn.functional.interpolate(..., mode='bicubic',
    align_corners=False)`` to at most 2e-6 in the predeclared float32 parity
    test.  Gradients flow only through deterministic matrix multiplications,
    never through an upsample kernel.
    """

    if value.ndim < 2 or min(int(value.shape[-2]), int(value.shape[-1])) < 1:
        raise ValueError("bicubic input needs two non-empty spatial dimensions")
    if not value.dtype.is_floating_point:
        raise TypeError("bicubic input must be floating point")
    target_height, target_width = _resize_shape(size)
    source_height, source_width = int(value.shape[-2]), int(value.shape[-1])
    if (source_height, source_width) == (target_height, target_width):
        return value
    vertical = bicubic_interpolation_matrix(
        source_height, target_height, device=value.device, dtype=value.dtype
    )
    horizontal = bicubic_interpolation_matrix(
        source_width, target_width, device=value.device, dtype=value.dtype
    )
    return torch.matmul(torch.matmul(vertical, value), horizontal.transpose(0, 1))


def encode_dct_residual(
    target: Tensor,
    flat_rgb: Tensor,
    *,
    side: int = DCT_SIDE,
    count: int = DCT_COEFFICIENTS,
) -> Tensor:
    """Encode a target as retained low-frequency coefficients above its flat.

    Returns ``(B,3,count)``.  Area resampling is the only target reduction.
    """

    _validate_image(target)
    side = _positive_int(side, "side")
    indices = zigzag_indices(side, count)
    flat = _flat_rgb(flat_rgb, batch=int(target.shape[0])).to(device=target.device, dtype=target.dtype)
    # Subtract first.  Area resampling is linear, but this order additionally
    # preserves the scientific zero exactly: a target equal to its flat colour
    # becomes a byte-for-byte zero tensor before any reduction arithmetic.
    residual = F.interpolate(
        target - flat[:, :, None, None], size=(side, side), mode="area"
    )
    spectrum = dct_2d(residual)
    rows = torch.tensor([row for row, _ in indices], device=target.device)
    columns = torch.tensor([column for _, column in indices], device=target.device)
    return spectrum[:, :, rows, columns]


def _coefficient_tensor(
    coefficients: Tensor,
    *,
    count: int,
    channels: int = CHANNELS,
) -> Tensor:
    if coefficients.ndim == 2:
        if coefficients.shape[1] != channels * count:
            raise ValueError(f"flattened coefficients must have width {channels * count}")
        coefficients = coefficients.reshape(coefficients.shape[0], channels, count)
    if coefficients.ndim != 3 or tuple(coefficients.shape[1:]) != (channels, count):
        raise ValueError(f"coefficients must have shape (B,{channels},{count})")
    if not coefficients.dtype.is_floating_point:
        raise TypeError("coefficients must be floating point")
    return coefficients


def render_dct_residual(
    coefficients: Tensor,
    flat_rgb: Tensor,
    *,
    size: int | Sequence[int] = (480, 480),
    side: int = DCT_SIDE,
    count: int = DCT_COEFFICIENTS,
    clamp: bool = True,
) -> Tensor:
    """Render retained DCT residuals above an input-derived flat colour."""

    side = _positive_int(side, "side")
    indices = zigzag_indices(side, count)
    coefficients = _coefficient_tensor(coefficients, count=count)
    flat = _flat_rgb(flat_rgb, batch=int(coefficients.shape[0])).to(
        device=coefficients.device, dtype=coefficients.dtype
    )
    spectrum = coefficients.new_zeros((coefficients.shape[0], CHANNELS, side, side))
    rows = torch.tensor([row for row, _ in indices], device=coefficients.device)
    columns = torch.tensor([column for _, column in indices], device=coefficients.device)
    spectrum[:, :, rows, columns] = coefficients
    residual = idct_2d(spectrum)
    output_size = _resize_shape(size)
    if tuple(residual.shape[-2:]) != output_size:
        residual = fixed_bicubic_resize(residual, output_size)
    rendered = residual + flat[:, :, None, None]
    return rendered.clamp(0.0, 1.0) if clamp else rendered


def encode_rgb_residual(
    target: Tensor,
    flat_rgb: Tensor,
    *,
    side: int = RGB_FIELD_SIDE,
) -> Tensor:
    """Encode the matched RGB-field comparator as an area-resampled residual."""

    _validate_image(target)
    side = _positive_int(side, "side")
    flat = _flat_rgb(flat_rgb, batch=int(target.shape[0])).to(device=target.device, dtype=target.dtype)
    return F.interpolate(
        target - flat[:, :, None, None], size=(side, side), mode="area"
    )


def render_rgb_residual(
    field: Tensor,
    flat_rgb: Tensor,
    *,
    size: int | Sequence[int] = (480, 480),
    side: int = RGB_FIELD_SIDE,
    clamp: bool = True,
) -> Tensor:
    """Render a direct RGB residual field above the same exact flat baseline."""

    side = _positive_int(side, "side")
    if field.ndim == 2:
        expected = CHANNELS * side * side
        if field.shape[1] != expected:
            raise ValueError(f"flattened RGB field must have width {expected}")
        field = field.reshape(field.shape[0], CHANNELS, side, side)
    if field.ndim != 4 or tuple(field.shape[1:]) != (CHANNELS, side, side):
        raise ValueError(f"field must have shape (B,{CHANNELS},{side},{side})")
    if not field.dtype.is_floating_point:
        raise TypeError("field must be floating point")
    flat = _flat_rgb(flat_rgb, batch=int(field.shape[0])).to(device=field.device, dtype=field.dtype)
    output_size = _resize_shape(size)
    residual = field
    if tuple(field.shape[-2:]) != output_size:
        residual = fixed_bicubic_resize(field, output_size)
    rendered = residual + flat[:, :, None, None]
    return rendered.clamp(0.0, 1.0) if clamp else rendered


class _AttentionMLPBlock(nn.Module):
    """Pre-norm slot self-attention followed by a compact residual MLP."""

    def __init__(self, dimension: int, heads: int, ffn_dimension: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, batch_first=True, dropout=0.0)
        self.mlp_norm = nn.LayerNorm(dimension)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, ffn_dimension),
            nn.GELU(),
            nn.Linear(ffn_dimension, dimension),
        )

    def forward(self, slots: Tensor) -> Tensor:
        normalized = self.attention_norm(slots)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        slots = slots + attended
        return slots + self.mlp(self.mlp_norm(slots))


class M144WhereModel(nn.Module):
    """Permutation-invariant 16-slot ``what -> where`` predictor.

    The final layer is zero-initialised.  Consequently both full and blind
    models emit an exact zero residual before learning, regardless of their
    randomly initialised attention stack.
    """

    def __init__(
        self,
        output_dim: int = DCT_OUTPUT_DIM,
        *,
        embedding_dim: int = 128,
        num_slots: int = 16,
        num_heads: int = 4,
        self_layers: int = 2,
        ffn_dim: int = 256,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        output_dim = _positive_int(output_dim, "output_dim")
        embedding_dim = _positive_int(embedding_dim, "embedding_dim")
        num_slots = _positive_int(num_slots, "num_slots")
        num_heads = _positive_int(num_heads, "num_heads")
        self_layers = _positive_int(self_layers, "self_layers")
        ffn_dim = _positive_int(ffn_dim, "ffn_dim")
        hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.num_slots = num_slots
        self.input_norm = nn.LayerNorm(embedding_dim)
        self.slot_queries = nn.Parameter(torch.empty(1, num_slots, embedding_dim))
        nn.init.trunc_normal_(self.slot_queries, std=0.02)

        self.cross_query_norm = nn.LayerNorm(embedding_dim)
        self.cross_key_norm = nn.LayerNorm(embedding_dim)
        self.cross_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, batch_first=True, dropout=0.0
        )
        self.cross_mlp_norm = nn.LayerNorm(embedding_dim)
        self.cross_mlp = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embedding_dim),
        )
        self.slot_blocks = nn.ModuleList(
            _AttentionMLPBlock(embedding_dim, num_heads, ffn_dim) for _ in range(self_layers)
        )
        head_input = num_slots * embedding_dim + CHANNELS
        self.head = nn.Sequential(
            nn.LayerNorm(head_input),
            nn.Linear(head_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        final = self.head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _validate_inputs(self, tile_embeddings: Tensor, flat_rgb: Tensor) -> Tensor:
        if tile_embeddings.ndim != 3 or tile_embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"tile_embeddings must have shape (B,T,{self.embedding_dim})"
            )
        if tile_embeddings.shape[1] < 1:
            raise ValueError("each board needs at least one tile embedding")
        if not tile_embeddings.dtype.is_floating_point:
            raise TypeError("tile_embeddings must be floating point")
        return _flat_rgb(flat_rgb, batch=int(tile_embeddings.shape[0])).to(
            device=tile_embeddings.device, dtype=tile_embeddings.dtype
        )

    def semantic_slots(self, tile_embeddings: Tensor, *, blind: bool = False) -> Tensor:
        """Aggregate an unordered tile set into learned semantic slots."""

        if tile_embeddings.ndim != 3 or tile_embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"tile_embeddings must have shape (B,T,{self.embedding_dim})"
            )
        if tile_embeddings.shape[1] < 1:
            raise ValueError("each board needs at least one tile embedding")
        if not tile_embeddings.dtype.is_floating_point:
            raise TypeError("tile_embeddings must be floating point")
        if blind:
            tile_embeddings = torch.zeros_like(tile_embeddings)
        tokens = self.input_norm(tile_embeddings)
        slots = self.slot_queries.expand(tile_embeddings.shape[0], -1, -1)
        attended, _ = self.cross_attention(
            self.cross_query_norm(slots),
            self.cross_key_norm(tokens),
            self.cross_key_norm(tokens),
            need_weights=False,
        )
        slots = slots + attended
        slots = slots + self.cross_mlp(self.cross_mlp_norm(slots))
        for block in self.slot_blocks:
            slots = block(slots)
        return slots

    def forward(self, tile_embeddings: Tensor, flat_rgb: Tensor, *, blind: bool = False) -> Tensor:
        flat = self._validate_inputs(tile_embeddings, flat_rgb)
        slots = self.semantic_slots(tile_embeddings, blind=blind)
        features = torch.cat((slots.flatten(1), flat), dim=1)
        return self.head(features)

    def forward_full(self, tile_embeddings: Tensor, flat_rgb: Tensor) -> Tensor:
        return self.forward(tile_embeddings, flat_rgb, blind=False)

    def forward_blind(self, tile_embeddings: Tensor, flat_rgb: Tensor) -> Tensor:
        return self.forward(tile_embeddings, flat_rgb, blind=True)


def uniform_ssim(
    first: Tensor,
    second: Tensor,
    *,
    data_range: float = 1.0,
    win_size: int = SSIM_WINDOW,
    use_sample_covariance: bool = True,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tensor:
    """Differentiable uniform-window SSIM compatible with scikit-image.

    The returned tensor contains one mean RGB SSIM per image.  Valid pooling is
    equivalent to scikit-image's reflect-filter followed by cropping the
    ``(win_size - 1) // 2`` border, while avoiding irrelevant border work.
    """

    _validate_image(first)
    _validate_image(second)
    if first.shape != second.shape:
        raise ValueError("SSIM inputs must share shape")
    if first.device != second.device:
        raise ValueError("SSIM inputs must share a device")
    win_size = _positive_int(win_size, "win_size")
    if win_size % 2 != 1:
        raise ValueError("win_size must be odd")
    if min(int(first.shape[-2]), int(first.shape[-1])) < win_size:
        raise ValueError("win_size exceeds an image dimension")
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be finite and positive")
    if k1 < 0 or k2 < 0:
        raise ValueError("k1 and k2 cannot be negative")

    average = lambda value: F.avg_pool2d(value, kernel_size=win_size, stride=1)
    mean_first = average(first)
    mean_second = average(second)
    mean_first_sq = mean_first.square()
    mean_second_sq = mean_second.square()
    mean_product = mean_first * mean_second

    covariance_norm = 1.0
    if use_sample_covariance:
        pixels = float(win_size * win_size)
        covariance_norm = pixels / (pixels - 1.0)
    variance_first = covariance_norm * (average(first.square()) - mean_first_sq)
    variance_second = covariance_norm * (average(second.square()) - mean_second_sq)
    covariance = covariance_norm * (average(first * second) - mean_product)

    c1 = float(k1 * data_range) ** 2
    c2 = float(k2 * data_range) ** 2
    numerator = (2.0 * mean_product + c1) * (2.0 * covariance + c2)
    denominator = (mean_first_sq + mean_second_sq + c1) * (
        variance_first + variance_second + c2
    )
    score_map = numerator / denominator
    return score_map.mean(dim=(1, 2, 3))


def skimage_ssim_parity(
    first: Tensor | np.ndarray,
    second: Tensor | np.ndarray,
    *,
    data_range: float = 1.0,
    win_size: int = SSIM_WINDOW,
    use_sample_covariance: bool = True,
) -> np.ndarray:
    """Reference per-image SSIM using scikit-image for parity tests/reports."""

    from skimage.metrics import structural_similarity

    def as_numpy(value: Tensor | np.ndarray) -> np.ndarray:
        if isinstance(value, Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.ndim != 4 or array.shape[1] != CHANNELS:
            raise ValueError("SSIM reference inputs must have shape (B,3,H,W)")
        return array

    first_np, second_np = as_numpy(first), as_numpy(second)
    if first_np.shape != second_np.shape:
        raise ValueError("SSIM reference inputs must share shape")
    scores = [
        structural_similarity(
            np.moveaxis(first_np[index], 0, -1),
            np.moveaxis(second_np[index], 0, -1),
            channel_axis=2,
            data_range=data_range,
            win_size=win_size,
            gaussian_weights=False,
            use_sample_covariance=use_sample_covariance,
        )
        for index in range(first_np.shape[0])
    ]
    return np.asarray(scores, dtype=np.float64)


def _float_vector(values: Sequence[float] | np.ndarray | Tensor, name: str) -> np.ndarray:
    if isinstance(values, Tensor):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _group_ids(groups: Sequence[Hashable] | np.ndarray | None, count: int) -> tuple[np.ndarray, int]:
    if groups is None:
        return np.arange(count, dtype=np.int64), count
    if len(groups) != count:
        raise ValueError("groups must have one entry per observation")
    mapping: dict[Hashable, int] = {}
    ids = np.empty(count, dtype=np.int64)
    for index, raw_group in enumerate(groups):
        group: Hashable = raw_group.item() if isinstance(raw_group, np.generic) else raw_group
        try:
            identifier = mapping.setdefault(group, len(mapping))
        except TypeError as error:
            raise ValueError("group identifiers must be hashable") from error
        ids[index] = identifier
    return ids, len(mapping)


def one_sided_bootstrap_lower(
    values: Sequence[float] | np.ndarray | Tensor,
    *,
    groups: Sequence[Hashable] | np.ndarray | None = None,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> float:
    """Return a deterministic lower cluster-bootstrap bound for the mean."""

    vector = _float_vector(values, "values")
    samples = _positive_int(samples, "samples")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    group_ids, group_count = _group_ids(groups, vector.size)
    if group_count < 1:
        raise ValueError("at least one bootstrap group is required")
    group_sums = np.bincount(group_ids, weights=vector, minlength=group_count)
    group_sizes = np.bincount(group_ids, minlength=group_count).astype(np.float64)
    rng = np.random.default_rng(seed)
    replicates = np.empty(samples, dtype=np.float64)
    chunk = 1_024
    for start in range(0, samples, chunk):
        stop = min(samples, start + chunk)
        selected = rng.integers(0, group_count, size=(stop - start, group_count))
        sampled_sum = group_sums[selected].sum(axis=1)
        sampled_size = group_sizes[selected].sum(axis=1)
        replicates[start:stop] = sampled_sum / sampled_size
    return float(np.quantile(replicates, alpha, method="linear"))


def paired_lift(
    candidate: Sequence[float] | np.ndarray | Tensor,
    baseline: Sequence[float] | np.ndarray | Tensor,
) -> np.ndarray:
    """Return a validated per-board paired difference without mutating inputs."""

    candidate_array = _float_vector(candidate, "candidate")
    baseline_array = _float_vector(baseline, "baseline")
    if candidate_array.shape != baseline_array.shape:
        raise ValueError("candidate and baseline must have the same length")
    return candidate_array - baseline_array


def _contrast_summary(
    values: np.ndarray,
    *,
    groups: Sequence[Hashable] | np.ndarray | None,
    samples: int,
    seed: int,
    alpha: float,
    include_win_fraction: bool = True,
) -> dict[str, float]:
    result = {
        "mean": float(values.mean()),
        "lower": one_sided_bootstrap_lower(
            values, groups=groups, samples=samples, seed=seed, alpha=alpha
        ),
        "confidence": float(1.0 - alpha),
    }
    if include_win_fraction:
        result["win_fraction"] = float(np.mean(values > 0.0))
    return result


def summarize_arm_metrics(
    *,
    flat_ssim: Sequence[float] | np.ndarray | Tensor,
    dct_full_ssim: Sequence[float] | np.ndarray | Tensor,
    dct_blind_ssim: Sequence[float] | np.ndarray | Tensor,
    dct_swapped_ssim: Sequence[float] | np.ndarray | Tensor,
    rgb8_full_ssim: Sequence[float] | np.ndarray | Tensor,
    rgb8_blind_ssim: Sequence[float] | np.ndarray | Tensor,
    target_oracle_dct_ssim: Sequence[float] | np.ndarray | Tensor | None = None,
    source_groups: Sequence[Hashable] | np.ndarray | None = None,
    swap_groups: Sequence[Hashable] | np.ndarray | None = None,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute the canonical nested M144 summary from raw per-board SSIM."""

    raw = {
        "flat": _float_vector(flat_ssim, "flat_ssim"),
        "dct_full": _float_vector(dct_full_ssim, "dct_full_ssim"),
        "dct_blind": _float_vector(dct_blind_ssim, "dct_blind_ssim"),
        "dct_swapped": _float_vector(dct_swapped_ssim, "dct_swapped_ssim"),
        "rgb8_full": _float_vector(rgb8_full_ssim, "rgb8_full_ssim"),
        "rgb8_blind": _float_vector(rgb8_blind_ssim, "rgb8_blind_ssim"),
    }
    if target_oracle_dct_ssim is not None:
        raw["target_oracle_dct"] = _float_vector(
            target_oracle_dct_ssim, "target_oracle_dct_ssim"
        )
    lengths = {array.size for array in raw.values()}
    if len(lengths) != 1:
        raise ValueError("all raw SSIM arms must have the same length")
    count = next(iter(lengths))
    source_ids, source_count = _group_ids(source_groups, count)
    swap_ids, swap_count = _group_ids(swap_groups, count)
    # Preserve caller labels for the bootstrap; generated integer IDs are used
    # when labels were omitted.
    source_bootstrap_groups: Sequence[Hashable] | np.ndarray = (
        source_ids if source_groups is None else source_groups
    )
    swap_bootstrap_groups: Sequence[Hashable] | np.ndarray = (
        swap_ids if swap_groups is None else swap_groups
    )

    means = {name: float(array.mean()) for name, array in raw.items()}
    gains = {
        name: float((array - raw["flat"]).mean())
        for name, array in raw.items()
        if name != "flat"
    }
    full_minus_blind = raw["dct_full"] - raw["dct_blind"]
    full_minus_swapped = raw["dct_full"] - raw["dct_swapped"]
    representation_delta = full_minus_blind - (
        raw["rgb8_full"] - raw["rgb8_blind"]
    )
    return {
        "n_boards": int(count),
        "n_source_groups": int(source_count),
        "n_swap_cycles": int(swap_count),
        "bootstrap": {
            "samples": int(bootstrap_samples),
            "seed": int(bootstrap_seed),
            "confidence": float(1.0 - alpha),
        },
        "means": means,
        "gains": gains,
        "contrasts": {
            "full_minus_blind": _contrast_summary(
                full_minus_blind,
                groups=source_bootstrap_groups,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
                alpha=alpha,
            ),
            "full_minus_swapped": _contrast_summary(
                full_minus_swapped,
                groups=swap_bootstrap_groups,
                samples=bootstrap_samples,
                seed=bootstrap_seed + 1,
                alpha=alpha,
            ),
            "representation_delta": _contrast_summary(
                representation_delta,
                groups=source_bootstrap_groups,
                samples=bootstrap_samples,
                seed=bootstrap_seed + 2,
                alpha=alpha,
                include_win_fraction=False,
            ),
        },
    }


def _gate_record(
    observed: float,
    threshold: float,
    *,
    lower: float | None = None,
    require_lower_positive: bool = False,
) -> dict[str, Any]:
    passed = bool(observed >= threshold)
    result: dict[str, Any] = {
        "observed": float(observed),
        "threshold": float(threshold),
        "passed": passed,
    }
    if lower is not None:
        result["lower"] = float(lower)
        result["lower_threshold"] = 0.0
        result["lower_passed"] = bool(lower > 0.0)
        if require_lower_positive:
            result["passed"] = bool(passed and lower > 0.0)
    return result


def _evaluate_gates(summary: Mapping[str, Any], thresholds: GateThresholds) -> dict[str, Any]:
    try:
        gains = summary["gains"]
        contrasts = summary["contrasts"]
        full_blind = contrasts["full_minus_blind"]
        full_swapped = contrasts["full_minus_swapped"]
        representation = contrasts["representation_delta"]
    except (KeyError, TypeError) as error:
        raise ValueError("summary does not follow the canonical M144 schema") from error

    checks: dict[str, dict[str, Any]] = {}
    if thresholds.oracle_gain is not None:
        if "target_oracle_dct" not in gains:
            raise ValueError("CAL decision requires target_oracle_dct gain")
        checks["oracle_gain"] = _gate_record(
            gains["target_oracle_dct"], thresholds.oracle_gain
        )
    checks["full_gain"] = _gate_record(gains["dct_full"], thresholds.full_gain)
    checks["full_blind"] = _gate_record(
        full_blind["mean"],
        thresholds.full_minus_blind,
        lower=full_blind["lower"],
        require_lower_positive=True,
    )
    checks["full_swapped"] = _gate_record(
        full_swapped["mean"],
        thresholds.full_minus_swapped,
        lower=full_swapped["lower"],
        require_lower_positive=thresholds.require_swap_lower,
    )
    checks["representation_delta"] = _gate_record(
        representation["mean"],
        thresholds.representation_delta,
        lower=representation["lower"],
        require_lower_positive=thresholds.require_representation_lower,
    )
    if thresholds.win_fraction is not None:
        checks["full_blind_win"] = _gate_record(
            full_blind["win_fraction"], thresholds.win_fraction
        )
        checks["full_swapped_win"] = _gate_record(
            full_swapped["win_fraction"], thresholds.win_fraction
        )
    return {
        "confidence": float(thresholds.confidence),
        "checks": checks,
        "passed": bool(all(record["passed"] for record in checks.values())),
    }


def evaluate_cal_gates(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen CAL opening gates to an alpha=0.10 summary."""

    confidence = float(summary.get("bootstrap", {}).get("confidence", float("nan")))
    if not np.isclose(confidence, CAL_THRESHOLDS.confidence, rtol=0.0, atol=1.0e-12):
        raise ValueError("CAL summary must use a 90% one-sided bootstrap bound")
    return _evaluate_gates(summary, CAL_THRESHOLDS)


def evaluate_dev_gates(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen one-shot DEV gates to an alpha=0.05 summary."""

    confidence = float(summary.get("bootstrap", {}).get("confidence", float("nan")))
    if not np.isclose(confidence, DEV_THRESHOLDS.confidence, rtol=0.0, atol=1.0e-12):
        raise ValueError("DEV summary must use a 95% one-sided bootstrap bound")
    return _evaluate_gates(summary, DEV_THRESHOLDS)


__all__ = [
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "CAL_THRESHOLDS",
    "CHANNELS",
    "DCT_COEFFICIENTS",
    "DCT_OUTPUT_DIM",
    "DCT_SIDE",
    "DEV_THRESHOLDS",
    "GateThresholds",
    "M144WhereModel",
    "RGB_FIELD_SIDE",
    "RGB_OUTPUT_DIM",
    "SSIM_WINDOW",
    "bicubic_interpolation_matrix",
    "dct_2d",
    "encode_dct_residual",
    "encode_rgb_residual",
    "evaluate_cal_gates",
    "evaluate_dev_gates",
    "flat_rgb_from_tiles",
    "fixed_bicubic_resize",
    "idct_2d",
    "one_sided_bootstrap_lower",
    "orthonormal_dct_matrix",
    "paired_lift",
    "render_dct_residual",
    "render_rgb_residual",
    "skimage_ssim_parity",
    "summarize_arm_metrics",
    "uniform_ssim",
    "zigzag_indices",
]
