"""Full-resolution perimeter encoder with an absolute pointer decoder.

The challenge tiles are upright 20x20 pixel squares rather than shaped jigsaw
pieces.  This model therefore keeps a stride-one 20x20 feature field, encodes
the ordered 76-pixel perimeter, and combines it with a frozen SocketMatcher
summary.  A permutation-equivariant board encoder has no shuffled-input index
embedding.  Absolute position enters only through the *output* slot queried by
the autoregressive pointer decoder.

The decoder masks every previously selected tile identity.  Greedy inference
is consequently a strict ``tile_at_position`` permutation without repairing or
resampling any tile pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from aiijc_puzzle.socket_matcher import SIDE_NAMES, SocketMatcher, robust_tile_views


class _FullResolutionResidual(nn.Module):
    """A residual image block that never changes the 20x20 lattice."""

    def __init__(self, width: int) -> None:
        super().__init__()
        groups = math.gcd(width, max(1, width // 8))
        self.network = nn.Sequential(
            nn.GroupNorm(groups, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(groups, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, padding_mode="replicate"),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.network(value)


class FullResolutionPerimeterEncoder(nn.Module):
    """Encode raw tiles while retaining all ordered boundary samples."""

    def __init__(self, *, width: int = 48, blocks: int = 3) -> None:
        super().__init__()
        if width < 8 or blocks < 1:
            raise ValueError("width must be at least eight and blocks must be positive")
        self.width = width
        self.blocks = blocks
        self.raw_skip = nn.Conv2d(3, width, 1)
        # Replication avoids presenting the matcher with a synthetic black
        # frame which is much easier to classify than a true image edge.
        self.stem = nn.Conv2d(6, width, 3, padding=1, padding_mode="replicate")
        self.residual = nn.Sequential(*[_FullResolutionResidual(width) for _ in range(blocks)])
        self.perimeter_local = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1, padding_mode="circular"),
            nn.GELU(),
            nn.Conv1d(width, width, 3, padding=1, padding_mode="circular"),
        )
        self.side_projection = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.tile_projection = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.side_identity = nn.Parameter(torch.randn(1, 1, 4, width) * 0.02)

    @staticmethod
    def _normalise(tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tiles.ndim != 5 or tiles.shape[2:] != (3, 20, 20):
            raise ValueError(f"expected B x N x 3 x 20 x 20, got {tuple(tiles.shape)}")
        raw = tiles.float()
        if bool((raw.detach().amax() > 1.5).item()):
            raw = raw / 255.0
        raw = raw.clamp(0.0, 1.0)
        flat = raw.reshape(-1, 3, 20, 20)
        mean = flat.mean(dim=(1, 2, 3), keepdim=True)
        scale = flat.std(dim=(1, 2, 3), keepdim=True).clamp_min(1.0 / 255.0)
        local = ((flat - mean) / scale).clamp(-4.0, 4.0) / 4.0
        return flat, local

    @staticmethod
    def ordered_perimeter(field: torch.Tensor) -> torch.Tensor:
        """Return the clockwise non-duplicated 76-pixel perimeter."""

        if field.ndim != 4 or field.shape[-2:] != (20, 20):
            raise ValueError(f"expected K x C x 20 x 20 field, got {tuple(field.shape)}")
        top = field[:, :, 0, :]
        right = field[:, :, 1:, -1]
        bottom = field[:, :, -1, :-1].flip(2)
        left = field[:, :, 1:-1, 0].flip(2)
        perimeter = torch.cat((top, right, bottom, left), dim=2)
        if perimeter.shape[2] != 76:
            raise RuntimeError("clockwise perimeter does not contain 76 samples")
        return perimeter

    @staticmethod
    def _side_sequences(perimeter: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Each side has 20 samples including both corners.  The final side
        # wraps across the end of the unique clockwise perimeter.
        return (
            perimeter[:, :, 0:20],
            perimeter[:, :, 19:39],
            perimeter[:, :, 38:58],
            torch.cat((perimeter[:, :, 57:76], perimeter[:, :, 0:1]), dim=2),
        )

    def forward(self, tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count = tiles.shape[:2]
        raw, local = self._normalise(tiles)
        field = self.raw_skip(raw) + self.residual(self.stem(torch.cat((raw, local), dim=1)))
        if field.shape[-2:] != (20, 20):
            raise RuntimeError("full-resolution encoder changed the spatial lattice")
        perimeter = self.ordered_perimeter(field)
        perimeter = perimeter + self.perimeter_local(perimeter)
        side_tokens = []
        for sequence in self._side_sequences(perimeter):
            pooled = torch.cat((sequence.mean(2), sequence.amax(2)), dim=1)
            side_tokens.append(self.side_projection(pooled))
        sides = torch.stack(side_tokens, dim=1).reshape(batch, count, 4, self.width)
        sides = sides + self.side_identity
        tile_pooled = torch.cat((field.mean((2, 3)), field.amax((2, 3))), dim=1)
        tile = self.tile_projection(tile_pooled).reshape(batch, count, self.width)
        return tile, sides


class _BoardBlock(nn.Module):
    """One input-permutation-equivariant board update."""

    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension,
            heads,
            dropout=0.05,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, 4 * dimension),
            nn.GELU(),
            nn.Linear(4 * dimension, dimension),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalised = self.norm(tokens)
        update, _ = self.attention(normalised, normalised, normalised, need_weights=False)
        value = tokens + update
        return value + self.feed_forward(value)


class AbsolutePointerDecoder(nn.Module):
    """Causal tile-identity pointer for fixed row-major output slots."""

    def __init__(self, *, dimension: int, max_grid: int = 24, layers: int = 4) -> None:
        super().__init__()
        if dimension < 8 or max_grid < 2 or layers < 1:
            raise ValueError("dimension must be >=8, max_grid >=2 and layers positive")
        self.dimension = dimension
        self.max_grid = max_grid
        self.layers = layers
        self.row_position = nn.Parameter(torch.randn(max_grid, dimension) * 0.02)
        self.column_position = nn.Parameter(torch.randn(max_grid, dimension) * 0.02)
        self.start = nn.Parameter(torch.randn(dimension) * 0.02)
        self.initial = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.Tanh(),
        )
        self.input_projection = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, dimension)
        )
        self.distance_projection = nn.Linear(4, dimension, bias=False)
        self.recurrent = nn.ModuleList(
            [nn.GRUCell(dimension, dimension) for _ in range(layers)]
        )
        self.query = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, dimension))
        self.key = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, dimension))
        self.border_key = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 4))
        self.border_query = nn.Linear(4, 4, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.left_edge_log_weight = nn.Parameter(torch.tensor(0.0))
        self.up_edge_log_weight = nn.Parameter(torch.tensor(0.0))

    def _slot_distances(self, grid: int) -> torch.Tensor:
        """Return normalised top/left/bottom/right distances for each slot."""

        coordinates = torch.arange(grid, device=self.row_position.device).float()
        normaliser = float(grid - 1)
        row = coordinates[:, None].expand(grid, grid) / normaliser
        column = coordinates[None, :].expand(grid, grid) / normaliser
        return torch.stack((row, column, 1.0 - row, 1.0 - column), dim=2)

    def _slot_tokens(self, grid: int) -> torch.Tensor:
        if not 2 <= grid <= self.max_grid:
            raise ValueError(f"grid must be in [2, {self.max_grid}]")
        rows = self.row_position[:grid, None, :].expand(grid, grid, -1)
        columns = self.column_position[None, :grid, :].expand(grid, grid, -1)
        distance = self.distance_projection(self._slot_distances(grid)).to(rows)
        return (rows + columns + distance).reshape(grid * grid, self.dimension)

    def _border_logits(self, memory: torch.Tensor, *, grid: int) -> torch.Tensor:
        """Candidate border compatibility for every fixed raster slot."""

        candidate_sides = self.border_key(memory)
        distances = self._slot_distances(grid).to(memory).reshape(-1, 4)
        slot_query = self.border_query(distances)
        return torch.einsum("bnk,sk->bsn", candidate_sides, slot_query) / 2.0

    @staticmethod
    def _validate_layout(layout: torch.Tensor, count: int) -> torch.Tensor:
        value = layout.long()
        if value.ndim != 2 or value.shape[1] != count:
            raise ValueError(f"layout must have shape B x {count}, got {tuple(value.shape)}")
        expected = torch.arange(count, device=value.device).expand(len(value), -1)
        if not torch.equal(value.sort(dim=1).values, expected):
            raise ValueError("every teacher layout must be a strict permutation")
        return value

    def _step_logits(
        self,
        hidden: torch.Tensor,
        keys: torch.Tensor,
        used: torch.Tensor,
        *,
        position: int,
        grid: int,
        prefix: list[torch.Tensor],
        right_logits: torch.Tensor,
        down_logits: torch.Tensor,
        border_logits: torch.Tensor,
    ) -> torch.Tensor:
        query = F.normalize(self.query(hidden), dim=1)
        scale = self.log_scale.exp().clamp(1.0, 100.0)
        logits = scale * torch.einsum("bd,bnd->bn", query, keys)
        batch_index = torch.arange(len(hidden), device=hidden.device)
        if position % grid:
            left = prefix[position - 1]
            logits = logits + F.softplus(self.left_edge_log_weight) * right_logits[
                batch_index, left
            ]
        if position >= grid:
            up = prefix[position - grid]
            logits = logits + F.softplus(self.up_edge_log_weight) * down_logits[batch_index, up]
        logits = logits + border_logits[:, position]
        return logits.masked_fill(used, -1e4)

    def teacher_forced(
        self,
        memory: torch.Tensor,
        layout: torch.Tensor,
        *,
        grid: int,
        right_logits: torch.Tensor,
        down_logits: torch.Tensor,
    ) -> torch.Tensor:
        batch, count, dimension = memory.shape
        if dimension != self.dimension or count != grid * grid:
            raise ValueError("memory shape is incompatible with decoder grid/dimension")
        target = self._validate_layout(layout, count)
        slots = self._slot_tokens(grid).to(memory)
        keys = F.normalize(self.key(memory), dim=2)
        initial = self.initial(memory.mean(1))
        hidden = [initial for _ in self.recurrent]
        previous = self.start.unsqueeze(0).expand(batch, -1)
        used = torch.zeros(batch, count, dtype=torch.bool, device=memory.device)
        rows = []
        prefix: list[torch.Tensor] = []
        border_logits = self._border_logits(memory, grid=grid)
        batch_index = torch.arange(batch, device=memory.device)
        for position in range(count):
            recurrent_input = self.input_projection(previous + slots[position])
            for layer, recurrent in enumerate(self.recurrent):
                hidden[layer] = recurrent(recurrent_input, hidden[layer])
                recurrent_input = hidden[layer]
            rows.append(
                self._step_logits(
                    hidden[-1],
                    keys,
                    used,
                    position=position,
                    grid=grid,
                    prefix=prefix,
                    right_logits=right_logits,
                    down_logits=down_logits,
                    border_logits=border_logits,
                )
            )
            chosen = target[:, position]
            prefix.append(chosen)
            used = used.scatter(1, chosen[:, None], True)
            previous = memory[batch_index, chosen]
        return torch.stack(rows, dim=1)

    @torch.no_grad()
    def greedy(
        self,
        memory: torch.Tensor,
        *,
        grid: int,
        right_logits: torch.Tensor,
        down_logits: torch.Tensor,
    ) -> torch.Tensor:
        batch, count, dimension = memory.shape
        if dimension != self.dimension or count != grid * grid:
            raise ValueError("memory shape is incompatible with decoder grid/dimension")
        slots = self._slot_tokens(grid).to(memory)
        keys = F.normalize(self.key(memory), dim=2)
        initial = self.initial(memory.mean(1))
        hidden = [initial for _ in self.recurrent]
        previous = self.start.unsqueeze(0).expand(batch, -1)
        used = torch.zeros(batch, count, dtype=torch.bool, device=memory.device)
        choices: list[torch.Tensor] = []
        border_logits = self._border_logits(memory, grid=grid)
        batch_index = torch.arange(batch, device=memory.device)
        for position in range(count):
            recurrent_input = self.input_projection(previous + slots[position])
            for layer, recurrent in enumerate(self.recurrent):
                hidden[layer] = recurrent(recurrent_input, hidden[layer])
                recurrent_input = hidden[layer]
            chosen = self._step_logits(
                hidden[-1],
                keys,
                used,
                position=position,
                grid=grid,
                prefix=choices,
                right_logits=right_logits,
                down_logits=down_logits,
                border_logits=border_logits,
            ).argmax(1)
            choices.append(chosen)
            used = used.scatter(1, chosen[:, None], True)
            previous = memory[batch_index, chosen]
        layout = torch.stack(choices, dim=1)
        expected = torch.arange(count, device=memory.device).expand(batch, -1)
        if not torch.equal(layout.sort(dim=1).values, expected):
            raise RuntimeError("masked pointer decoder did not produce a strict permutation")
        return layout

    @torch.no_grad()
    def beam(
        self,
        memory: torch.Tensor,
        *,
        grid: int,
        right_logits: torch.Tensor,
        down_logits: torch.Tensor,
        width: int = 4,
    ) -> torch.Tensor:
        """Fixed-width autoregressive search for a single puzzle board."""

        batch, count, dimension = memory.shape
        if batch != 1:
            raise ValueError("beam decoding currently requires batch size one")
        if dimension != self.dimension or count != grid * grid:
            raise ValueError("memory shape is incompatible with decoder grid/dimension")
        if not 1 <= width <= count:
            raise ValueError("beam width must be in [1, tile_count]")
        slots = self._slot_tokens(grid).to(memory)
        keys = F.normalize(self.key(memory), dim=2)
        initial = self.initial(memory.mean(1))
        border_logits = self._border_logits(memory, grid=grid)
        start = self.start.unsqueeze(0)
        empty_used = torch.zeros(1, count, dtype=torch.bool, device=memory.device)
        # score, hidden layers, previous memory, used identities, prefix
        beams: list[
            tuple[float, list[torch.Tensor], torch.Tensor, torch.Tensor, list[torch.Tensor]]
        ] = [(0.0, [initial for _ in self.recurrent], start, empty_used, [])]
        for position in range(count):
            extensions: list[
                tuple[
                    float,
                    list[torch.Tensor],
                    torch.Tensor,
                    torch.Tensor,
                    list[torch.Tensor],
                ]
            ] = []
            remaining = count - position
            for score, hidden, previous, used, prefix in beams:
                recurrent_input = self.input_projection(previous + slots[position])
                updated_hidden: list[torch.Tensor] = []
                for layer, recurrent in enumerate(self.recurrent):
                    recurrent_input = recurrent(recurrent_input, hidden[layer])
                    updated_hidden.append(recurrent_input)
                logits = self._step_logits(
                    updated_hidden[-1],
                    keys,
                    used,
                    position=position,
                    grid=grid,
                    prefix=prefix,
                    right_logits=right_logits,
                    down_logits=down_logits,
                    border_logits=border_logits,
                )
                log_probability = F.log_softmax(logits, dim=1)[0]
                values, candidates = log_probability.topk(min(width, remaining))
                for value, candidate in zip(values, candidates, strict=True):
                    candidate_index = int(candidate)
                    candidate_tensor = candidate.reshape(1)
                    next_used = used.clone()
                    next_used[0, candidate_index] = True
                    extensions.append(
                        (
                            score + float(value),
                            updated_hidden,
                            memory[:, candidate_index],
                            next_used,
                            [*prefix, candidate_tensor],
                        )
                    )
            extensions.sort(key=lambda item: item[0], reverse=True)
            beams = extensions[:width]
        layout = torch.stack(beams[0][4], dim=1)
        expected = torch.arange(count, device=memory.device).expand(1, -1)
        if not torch.equal(layout.sort(dim=1).values, expected):
            raise RuntimeError("beam pointer decoder did not produce a strict permutation")
        return layout


@dataclass(frozen=True)
class BorderPointerOutput:
    memory: torch.Tensor
    pointer_logits: torch.Tensor | None
    right_logits: torch.Tensor
    down_logits: torch.Tensor


class BorderPointerSorter(nn.Module):
    """Full-resolution, board-conditioned, absolute permutation sorter."""

    def __init__(
        self,
        *,
        socket_backbone: SocketMatcher | None = None,
        feature_width: int = 48,
        feature_blocks: int = 3,
        dimension: int = 128,
        heads: int = 8,
        board_layers: int = 4,
        pointer_layers: int = 4,
        max_grid: int = 24,
        freeze_socket: bool = True,
    ) -> None:
        super().__init__()
        if dimension % heads or board_layers < 1:
            raise ValueError("dimension must divide heads and board_layers must be positive")
        self.socket_backbone = socket_backbone
        self.freeze_socket = freeze_socket
        self.feature_width = feature_width
        self.dimension = dimension
        self.heads = heads
        self.board_layers = board_layers
        self.pointer_layers = pointer_layers
        self.max_grid = max_grid
        self.perimeter = FullResolutionPerimeterEncoder(width=feature_width, blocks=feature_blocks)
        socket_dimension = 0 if socket_backbone is None else 5 * socket_backbone.dimension
        input_dimension = 5 * feature_width + socket_dimension
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )
        self.board = nn.ModuleList([_BoardBlock(dimension, heads) for _ in range(board_layers)])
        self.pointer = AbsolutePointerDecoder(
            dimension=dimension,
            max_grid=max_grid,
            layers=pointer_layers,
        )
        self.right_source = nn.Linear(dimension, dimension)
        self.right_target = nn.Linear(dimension, dimension)
        self.down_source = nn.Linear(dimension, dimension)
        self.down_target = nn.Linear(dimension, dimension)
        self.right_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.down_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        # Start as a small trainable residual over the retained frozen d64
        # directional evidence instead of replacing its known useful ranking.
        self.right_residual_log_weight = nn.Parameter(torch.tensor(-2.25))
        self.down_residual_log_weight = nn.Parameter(torch.tensor(-2.25))
        if socket_backbone is not None and freeze_socket:
            for parameter in socket_backbone.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> BorderPointerSorter:
        super().train(mode)
        if self.socket_backbone is not None and self.freeze_socket:
            self.socket_backbone.eval()
        return self

    def _socket_evidence(
        self,
        tiles: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if self.socket_backbone is None:
            return None, None, None
        context_manager = torch.no_grad() if self.freeze_socket else torch.enable_grad()
        with context_manager:
            views = robust_tile_views(tiles)
            context = self.socket_backbone.tile_context(views)
            sides = self.socket_backbone._side_embeddings(views, context)  # noqa: SLF001
            right_source, left_target = self.socket_backbone.horizontal(
                sides["right"], sides["left"]
            )
            down_source, top_target = self.socket_backbone.vertical(
                sides["bottom"], sides["top"]
            )
            right = self.socket_backbone._similarity(  # noqa: SLF001
                right_source,
                left_target,
                self.socket_backbone.horizontal_scale,
            )
            down = self.socket_backbone._similarity(  # noqa: SLF001
                down_source,
                top_target,
                self.socket_backbone.vertical_scale,
            )
            features = torch.cat((context, *(sides[name] for name in SIDE_NAMES)), dim=2)
            return features, right, down

    @staticmethod
    def _pair_scores(
        source: torch.Tensor,
        target: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        source = F.normalize(source, dim=2)
        target = F.normalize(target, dim=2)
        logits = scale.exp().clamp(1.0, 100.0) * source @ target.transpose(1, 2)
        diagonal = torch.eye(logits.shape[1], dtype=torch.bool, device=logits.device)
        return logits.masked_fill(diagonal.unsqueeze(0), -1e4)

    def _encode_evidence(
        self,
        tiles: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        tile, sides = self.perimeter(tiles)
        features = [tile, sides.flatten(2)]
        socket, socket_right, socket_down = self._socket_evidence(tiles)
        if socket is not None:
            features.append(socket)
        tokens = self.input_projection(torch.cat(features, dim=2))
        for block in self.board:
            tokens = block(tokens)
        return tokens, socket_right, socket_down

    def encode(self, tiles: torch.Tensor) -> torch.Tensor:
        memory, _, _ = self._encode_evidence(tiles)
        return memory

    def _directional_scores(
        self,
        memory: torch.Tensor,
        socket_right: torch.Tensor | None,
        socket_down: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        learned_right = self._pair_scores(
            self.right_source(memory), self.right_target(memory), self.right_scale
        )
        learned_down = self._pair_scores(
            self.down_source(memory), self.down_target(memory), self.down_scale
        )
        if socket_right is None or socket_down is None:
            return learned_right, learned_down
        right = socket_right + F.softplus(self.right_residual_log_weight) * learned_right
        down = socket_down + F.softplus(self.down_residual_log_weight) * learned_down
        return right, down

    def forward(
        self,
        tiles: torch.Tensor,
        *,
        teacher_layout: torch.Tensor | None = None,
        grid: int | None = None,
    ) -> BorderPointerOutput:
        count = tiles.shape[1]
        inferred = round(math.sqrt(count))
        grid = inferred if grid is None else grid
        if grid * grid != count:
            raise ValueError(f"tile count {count} is not grid^2 for grid={grid}")
        memory, socket_right, socket_down = self._encode_evidence(tiles)
        right, down = self._directional_scores(memory, socket_right, socket_down)
        pointer_logits = (
            None
            if teacher_layout is None
            else self.pointer.teacher_forced(
                memory,
                teacher_layout,
                grid=grid,
                right_logits=right,
                down_logits=down,
            )
        )
        return BorderPointerOutput(memory, pointer_logits, right, down)

    @torch.no_grad()
    def decode(self, tiles: torch.Tensor, *, grid: int | None = None) -> torch.Tensor:
        count = tiles.shape[1]
        inferred = round(math.sqrt(count))
        grid = inferred if grid is None else grid
        if grid * grid != count:
            raise ValueError(f"tile count {count} is not grid^2 for grid={grid}")
        memory, socket_right, socket_down = self._encode_evidence(tiles)
        right, down = self._directional_scores(memory, socket_right, socket_down)
        return self.pointer.greedy(
            memory,
            grid=grid,
            right_logits=right,
            down_logits=down,
        )

    @torch.no_grad()
    def decode_beam(
        self,
        tiles: torch.Tensor,
        *,
        grid: int | None = None,
        width: int = 4,
    ) -> torch.Tensor:
        """Decode a strict permutation with the fixed-width pointer beam."""

        count = tiles.shape[1]
        inferred = round(math.sqrt(count))
        grid = inferred if grid is None else grid
        if grid * grid != count:
            raise ValueError(f"tile count {count} is not grid^2 for grid={grid}")
        memory, socket_right, socket_down = self._encode_evidence(tiles)
        right, down = self._directional_scores(memory, socket_right, socket_down)
        return self.pointer.beam(
            memory,
            grid=grid,
            right_logits=right,
            down_logits=down,
            width=width,
        )


def border_pointer_loss(
    output: BorderPointerOutput,
    tile_at_position: torch.Tensor,
    *,
    grid: int,
    adjacency_weight: float = 0.15,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Teacher-forced permutation CE plus exact right/down adjacency CE."""

    if output.pointer_logits is None:
        raise ValueError("teacher-forced pointer logits are required for training")
    if not math.isfinite(adjacency_weight) or adjacency_weight < 0:
        raise ValueError("adjacency_weight must be finite and non-negative")
    layout = tile_at_position.long()
    batch, count = layout.shape
    if count != grid * grid or output.pointer_logits.shape != (batch, count, count):
        raise ValueError("layout/pointer shapes are incompatible with grid")
    expected = torch.arange(count, device=layout.device).expand(batch, -1)
    if not torch.equal(layout.sort(dim=1).values, expected):
        raise ValueError("tile_at_position must contain strict permutations")

    pointer = F.cross_entropy(output.pointer_logits.reshape(-1, count), layout.reshape(-1))
    positions = torch.arange(count, device=layout.device).reshape(grid, grid)
    right_positions = positions[:, :-1].reshape(-1)
    down_positions = positions[:-1, :].reshape(-1)
    right_sources = layout[:, right_positions]
    right_targets = layout[:, right_positions + 1]
    down_sources = layout[:, down_positions]
    down_targets = layout[:, down_positions + grid]
    batch_index = torch.arange(batch, device=layout.device)[:, None]
    right = F.cross_entropy(
        output.right_logits[batch_index, right_sources].reshape(-1, count),
        right_targets.reshape(-1),
    )
    down = F.cross_entropy(
        output.down_logits[batch_index, down_sources].reshape(-1, count),
        down_targets.reshape(-1),
    )
    adjacency = 0.5 * (right + down)
    loss = pointer + adjacency_weight * adjacency
    with torch.no_grad():
        pointer_accuracy = (output.pointer_logits.argmax(2) == layout).float().mean()
        right_accuracy = (
            output.right_logits[batch_index, right_sources].argmax(2) == right_targets
        ).float().mean()
        down_accuracy = (
            output.down_logits[batch_index, down_sources].argmax(2) == down_targets
        ).float().mean()
    return loss, {
        "loss": float(loss.detach()),
        "pointer_nll": float(pointer.detach()),
        "adjacency_nll": float(adjacency.detach()),
        "teacher_pointer_accuracy": float(pointer_accuracy),
        "right_r1": float(right_accuracy),
        "down_r1": float(down_accuracy),
    }


__all__ = [
    "AbsolutePointerDecoder",
    "BorderPointerOutput",
    "BorderPointerSorter",
    "FullResolutionPerimeterEncoder",
    "border_pointer_loss",
]
