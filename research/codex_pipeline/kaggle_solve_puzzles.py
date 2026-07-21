import math
import os
import random
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity
from tqdm.auto import tqdm


GRID = 24
TILE = 20
N = GRID * GRID


def select_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        probe = torch.linspace(0, 1, 8, device="cuda")
        _ = (probe * probe).sum().item()
        print(f"gpu={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
        return torch.device("cuda")
    except Exception as exc:
        print(f"cuda_unusable={type(exc).__name__}: {exc}")
        return torch.device("cpu")


DEVICE = select_device()
OUT_DIR = Path(os.getenv("OUT_DIR", "/kaggle/working"))
VALIDATE_IMAGES = int(os.getenv("VALIDATE_IMAGES", "2"))
SOLVE_TEST = os.getenv("SOLVE_TEST", "1") == "1"
EDGE_BATCH = int(os.getenv("EDGE_BATCH", "4096"))
EDGE_TOPK = int(os.getenv("EDGE_TOPK", "2"))
SWAP_STEPS = int(os.getenv("SWAP_STEPS", "20000"))
POSITION_WEIGHT = float(os.getenv("POSITION_WEIGHT", "0.12"))
RESTORE_BATCH = int(os.getenv("RESTORE_BATCH", "512"))
USE_RL = os.getenv("USE_RL", "1") == "1"
RL_PROPOSALS = int(os.getenv("RL_PROPOSALS", "48"))
RL_STEPS = int(os.getenv("RL_STEPS", "800"))
RL_STAGNATION = int(os.getenv("RL_STAGNATION", "120"))
RL_POLICY_BLEND = float(os.getenv("RL_POLICY_BLEND", "0.75"))
RL_VALIDATION_MIN_DELTA = float(os.getenv("RL_VALIDATION_MIN_DELTA", "0.0"))
DATA_ROOT = os.getenv("DATA_ROOT")
ASSEMBLY_MODEL_ROOT = Path(os.getenv("ASSEMBLY_MODEL_ROOT", "/kaggle/input/pazzle-puzzle-assembly-models"))
RESTORER_MODEL_ROOT = Path(os.getenv("RESTORER_MODEL_ROOT", "/kaggle/input/pazzle-fragment-restorer"))
RL_MODEL_ROOT = Path(os.getenv("RL_MODEL_ROOT", "/kaggle/input/pazzle-rl-puzzle-assembler"))
RL_CHECKPOINT_NAME = os.getenv("RL_CHECKPOINT_NAME", "rl_swap_actor_critic_epoch1.pt")
SEED = int(os.getenv("SEED", "2026"))

RL_FEATURE_NAMES = [
    "visual_delta", "position_delta",
    "a_local_before", "b_local_before", "a_local_after", "b_local_after",
    "a_left_before", "a_right_before", "a_up_before", "a_down_before",
    "b_left_before", "b_right_before", "b_up_before", "b_down_before",
    "a_left_after", "a_right_after", "a_up_after", "a_down_after",
    "b_left_after", "b_right_after", "b_up_after", "b_down_after",
    "a_position_before", "b_position_before", "a_position_after", "b_position_after",
    "a_row", "a_col", "b_row", "b_col", "distance", "same_row", "same_col",
    "board_visual", "board_position", "progress", "stagnation", "bias",
]


class EdgeMatcher(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(7, base, 3, padding=1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1), nn.GroupNorm(8, base * 2), nn.SiLU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.GroupNorm(8, base * 2), nn.SiLU(),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1), nn.GroupNorm(8, base * 4), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(base * 4, base * 2),
            nn.SiLU(), nn.Dropout(0.1), nn.Linear(base * 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class PositionPrior(nn.Module):
    def __init__(self, base=48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1), nn.GroupNorm(8, base * 2), nn.SiLU(),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1), nn.GroupNorm(8, base * 4), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.row_head = nn.Linear(base * 4, GRID)
        self.col_head = nn.Linear(base * 4, GRID)

    def forward(self, x):
        h = self.encoder(x)
        return self.row_head(h), self.col_head(h)


class SwapActorCritic(nn.Module):
    def __init__(self, feature_dim=len(RL_FEATURE_NAMES), hidden=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1),
        )

    def forward(self, proposal_features):
        logits = self.actor(proposal_features).squeeze(-1)
        pooled = torch.cat([proposal_features.mean(dim=-2), proposal_features.amax(dim=-2)], dim=-1)
        return logits, self.value(pooled).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x):
        return F.silu(x + self.body(x))


class FragmentRestorer(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.stem = nn.Conv2d(3, base, 3, padding=1)
        self.enc1 = nn.Sequential(ResidualBlock(base), ResidualBlock(base))
        self.down = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.mid = nn.Sequential(
            ResidualBlock(base * 2), ResidualBlock(base * 2), ResidualBlock(base * 2)
        )
        self.up = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec = nn.Sequential(
            ResidualBlock(base * 2),
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.SiLU(),
        )
        self.residual = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, x):
        skip = self.enc1(self.stem(x))
        h = self.up(self.mid(self.down(skip)))
        correction = 0.35 * torch.tanh(
            self.residual(self.dec(torch.cat([h, skip], dim=1)))
        )
        return (x + correction).clamp(0.0, 1.0)


def find_latest(root, pattern):
    files = list(root.rglob(pattern)) if root.exists() else []
    if not files:
        raise FileNotFoundError(f"Checkpoint not found under {root}: {pattern}")

    def epoch(path):
        try:
            return int(path.stem.rsplit("epoch", 1)[1])
        except Exception:
            return -1

    parsed = [(epoch(path), path) for path in files]
    if max(value for value, _ in parsed) < 0:
        raise ValueError(f"No checkpoint has a parseable epoch under {root}: {pattern}")
    return max(parsed, key=lambda item: (item[0], str(item[1])))[1]


def resolve_model_root(configured_root, patterns, label):
    candidates = []
    search_roots = [configured_root]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        search_roots.extend(path for path in kaggle_input.iterdir() if path.is_dir())
    for root in search_roots:
        if root in candidates or not root.is_dir():
            continue
        if all(any(root.rglob(pattern)) for pattern in patterns):
            candidates.append(root)
    unique = list(dict.fromkeys(candidates))
    if len(unique) == 1:
        return unique[0]
    if not unique:
        raise FileNotFoundError(f"No {label} model root contains all patterns {patterns}")
    raise RuntimeError(f"Ambiguous {label} model roots; configure one explicitly: {unique}")


def find_rl_checkpoint():
    local_root = Path(__file__).resolve().parent / "models"
    if local_root.is_dir() and any(local_root.rglob(RL_CHECKPOINT_NAME)):
        root = local_root
    else:
        root = resolve_model_root(RL_MODEL_ROOT, [RL_CHECKPOINT_NAME], "RL")
    files = list(root.rglob(RL_CHECKPOINT_NAME))
    if not files:
        raise FileNotFoundError(f"RL checkpoint not found in configured roots: {RL_CHECKPOINT_NAME}")
    if len(files) > 1:
        raise RuntimeError(f"Ambiguous RL checkpoint {RL_CHECKPOINT_NAME}: {files}")
    return files[0]


def find_data_root():
    if DATA_ROOT:
        root = Path(DATA_ROOT)
        if not (root / "train" / "inputs").is_dir() or not (root / "train" / "targets").is_dir():
            raise FileNotFoundError(f"Invalid DATA_ROOT, expected train/inputs and train/targets: {root}")
        return root
    roots = sorted({
        path.parent.parent
        for path in Path("/kaggle/input").rglob("train/inputs")
        if (path.parent / "targets").is_dir()
    })
    if len(roots) == 1:
        return roots[0]
    if len(roots) > 1:
        raise RuntimeError(f"Multiple datasets found; set DATA_ROOT explicitly: {roots}")
    raise FileNotFoundError("Dataset with train/inputs and train/targets not found")


def load_models():
    assembly_root = resolve_model_root(
        ASSEMBLY_MODEL_ROOT,
        ["edge_matcher_epoch*.pt", "position_prior_epoch*.pt"],
        "assembly",
    )
    restorer_root = resolve_model_root(
        RESTORER_MODEL_ROOT, ["fragment_restorer_epoch*.pt"], "fragment restorer"
    )
    edge_path = find_latest(assembly_root, "edge_matcher_epoch*.pt")
    pos_path = find_latest(assembly_root, "position_prior_epoch*.pt")
    edge_ckpt = torch.load(edge_path, map_location="cpu")
    pos_ckpt = torch.load(pos_path, map_location="cpu")
    edge = EdgeMatcher()
    pos = PositionPrior()
    edge.load_state_dict(edge_ckpt["model"])
    pos.load_state_dict(pos_ckpt["model"])
    del edge_ckpt, pos_ckpt
    edge = edge.to(DEVICE).eval()
    pos = pos.to(DEVICE).eval()
    print(f"edge_checkpoint={edge_path}")
    print(f"position_checkpoint={pos_path}")
    restorer_path = find_latest(restorer_root, "fragment_restorer_epoch*.pt")
    restorer_ckpt = torch.load(restorer_path, map_location="cpu")
    restorer_config = restorer_ckpt.get("config", {})
    checkpoint_tile = int(restorer_config.get("tile", TILE))
    if checkpoint_tile != TILE:
        raise ValueError(f"Restorer tile contract mismatch: checkpoint={checkpoint_tile}, solver={TILE}")
    restorer = FragmentRestorer(base=int(restorer_config.get("base_channels", 64)))
    restorer.load_state_dict(restorer_ckpt["model"])
    restorer_epoch = restorer_ckpt.get("epoch")
    restorer_metrics = restorer_ckpt.get("metrics")
    del restorer_ckpt
    restorer = restorer.to(DEVICE).eval()
    print(
        f"restorer_checkpoint={restorer_path} epoch={restorer_epoch} "
        f"metrics={restorer_metrics}"
    )
    rl = None
    if USE_RL:
        rl_path = find_rl_checkpoint()
        rl_ckpt = torch.load(rl_path, map_location="cpu")
        if rl_ckpt.get("feature_names") != RL_FEATURE_NAMES:
            raise ValueError("RL checkpoint feature contract does not match inference code")
        rl = SwapActorCritic()
        rl.load_state_dict(rl_ckpt["model"])
        rl_epoch = rl_ckpt.get("epoch")
        del rl_ckpt
        rl = rl.to(DEVICE).eval()
        print(f"rl_checkpoint={rl_path} epoch={rl_epoch}")
    return edge, pos, restorer, restorer_config, rl


def validate_config():
    integer_ranges = {
        "VALIDATE_IMAGES": (VALIDATE_IMAGES, 1, None),
        "EDGE_BATCH": (EDGE_BATCH, 1, None),
        "EDGE_TOPK": (EDGE_TOPK, 1, N - 1),
        "SWAP_STEPS": (SWAP_STEPS, 0, None),
        "RESTORE_BATCH": (RESTORE_BATCH, 1, None),
        "RL_PROPOSALS": (RL_PROPOSALS, 2, N * (N - 1) // 2),
        "RL_STEPS": (RL_STEPS, 0, None),
        "RL_STAGNATION": (RL_STAGNATION, 1, None),
    }
    for name, (value, lower, upper) in integer_ranges.items():
        if value < lower or (upper is not None and value > upper):
            limit = f"[{lower}, {upper}]" if upper is not None else f">= {lower}"
            raise ValueError(f"{name} must be {limit}, got {value}")
    for name, value in {
        "POSITION_WEIGHT": POSITION_WEIGHT,
        "RL_POLICY_BLEND": RL_POLICY_BLEND,
        "RL_VALIDATION_MIN_DELTA": RL_VALIDATION_MIN_DELTA,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")


@torch.inference_mode()
def restore_tiles(model, tiles_t):
    outputs = []
    for batch in tiles_t.split(RESTORE_BATCH):
        restored = model((batch.to(DEVICE) + 1.0) * 0.5)
        outputs.append((restored * 2.0 - 1.0).clamp(-1, 1).cpu())
    return torch.cat(outputs)


def load_tiles(path):
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    expected = (GRID * TILE, GRID * TILE, 3)
    if image.shape != expected:
        raise ValueError(f"Expected image shape {expected}, got {image.shape}: {path}")
    return image.reshape(GRID, TILE, GRID, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(N, TILE, TILE, 3)


def tiles_tensor(tiles):
    x = np.ascontiguousarray(tiles.transpose(0, 3, 1, 2))
    return torch.from_numpy(x).float().div_(127.5).sub_(1.0)


@torch.inference_mode()
def position_scores(model, tiles_t):
    rows, cols = [], []
    for batch in tiles_t.split(EDGE_BATCH):
        r, c = model(batch.to(DEVICE))
        rows.append(F.log_softmax(r, 1).cpu())
        cols.append(F.log_softmax(c, 1).cpu())
    row = torch.cat(rows).numpy()
    col = torch.cat(cols).numpy()
    rr, cc = np.divmod(np.arange(N), GRID)
    return row[:, rr] + col[:, cc], row, col


def seam_distance(tiles, direction):
    x = tiles.astype(np.float32) / 255.0
    if direction == 0:
        a, b = x[:, :, -2:, :], x[:, :, :2, :]
    else:
        a, b = x[:, -2:, :, :], x[:, :2, :, :]
    af = a.reshape(N, -1)
    bf = b.reshape(N, -1)
    aa = np.sum(af * af, axis=1)[:, None]
    bb = np.sum(bf * bf, axis=1)[None, :]
    dist = np.maximum(aa + bb - 2.0 * af @ bf.T, 0.0) / af.shape[1]
    np.fill_diagonal(dist, np.inf)
    return dist


@torch.inference_mode()
def compatibility_scores(model, tiles, tiles_t, direction):
    seam = seam_distance(tiles, direction)
    k = min(EDGE_TOPK, N - 1)
    candidates = np.argpartition(seam, k, axis=1)[:, :k]
    src = np.repeat(np.arange(N), k)
    dst = candidates.reshape(-1)
    logits = []
    for start in range(0, len(src), EDGE_BATCH):
        ia = torch.from_numpy(src[start:start + EDGE_BATCH])
        ib = torch.from_numpy(dst[start:start + EDGE_BATCH])
        a = tiles_t[ia]
        b = tiles_t[ib]
        d = torch.full((len(a), 1, TILE, TILE), 1.0 if direction == 0 else -1.0)
        logits.append(model(torch.cat([a, b, d], 1).to(DEVICE)).flatten().float().cpu())
    logits = torch.cat(logits).numpy()

    finite = seam[np.isfinite(seam)]
    scale = max(float(np.median(finite)), 1e-5)
    score = np.clip(-seam / scale, -8.0, 0.0).astype(np.float32)
    score[src, dst] += 1.5 * np.clip(logits, -6.0, 6.0)
    np.fill_diagonal(score, -50.0)
    return score


def initial_layout(pos_score):
    tile_idx, position_idx = linear_sum_assignment(-pos_score)
    layout = np.empty(N, dtype=np.int32)
    layout[position_idx] = tile_idx
    return layout


def local_value(layout, positions, right, down, pos_score):
    edges = set()
    for p in positions:
        r, c = divmod(int(p), GRID)
        if c > 0: edges.add((p - 1, p, 0))
        if c + 1 < GRID: edges.add((p, p + 1, 0))
        if r > 0: edges.add((p - GRID, p, 1))
        if r + 1 < GRID: edges.add((p, p + GRID, 1))
    value = POSITION_WEIGHT * sum(pos_score[layout[p], p] for p in positions)
    for a, b, direction in edges:
        value += (right if direction == 0 else down)[layout[a], layout[b]]
    return float(value)


def optimize_layout(layout, right, down, pos_score, seed):
    rng = np.random.default_rng(seed)
    layout = layout.copy()
    best = layout.copy()
    current_total = layout_objective(layout, right, down, pos_score)
    best_total = current_total
    for step in range(SWAP_STEPS):
        a = int(rng.integers(N))
        if rng.random() < 0.7:
            row, col = divmod(a, GRID)
            neighbors = [
                (row + dr) * GRID + col + dc
                for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
                if 0 <= row + dr < GRID and 0 <= col + dc < GRID
            ]
            b = int(rng.choice(neighbors))
        else:
            b = int(rng.integers(N - 1))
            if b >= a:
                b += 1
        positions = {a, b}
        before = local_value(layout, positions, right, down, pos_score)
        layout[a], layout[b] = layout[b], layout[a]
        after = local_value(layout, positions, right, down, pos_score)
        delta = after - before
        temperature = 0.35 * (1.0 - step / max(SWAP_STEPS, 1)) + 0.01
        accepted = delta >= 0 or rng.random() < np.exp(delta / temperature)
        if accepted:
            current_total += delta
            if current_total > best_total:
                best_total = current_total
                best = layout.copy()
        else:
            layout[a], layout[b] = layout[b], layout[a]
        if step > 0 and step % 5000 == 0:
            exact = layout_objective(layout, right, down, pos_score)
            if not np.isclose(current_total, exact, rtol=1e-5, atol=1e-4):
                raise RuntimeError(f"Incremental layout objective drifted: {current_total} vs {exact}")
            current_total = exact
    return best


def layout_objective(layout, right, down, pos_score):
    board = layout.reshape(GRID, GRID)
    value = right[board[:, :-1], board[:, 1:]].sum()
    value += down[board[:-1, :], board[1:, :]].sum()
    value += POSITION_WEIGHT * pos_score[layout, np.arange(N)].sum()
    return float(value)


def edge_cost_matrix(a, b):
    a = a.reshape(len(a), -1).astype(np.float32) / 255.0
    b = b.reshape(len(b), -1).astype(np.float32) / 255.0
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * a @ b.T, 0.0) / a.shape[1]


def rl_visual_matrices(tiles):
    right = edge_cost_matrix(tiles[:, :, -2:, :], tiles[:, :, :2, :])
    down = edge_cost_matrix(tiles[:, -2:, :, :], tiles[:, :2, :, :])
    for cost in (right, down):
        cost /= max(float(np.median(cost[np.isfinite(cost)])), 1e-6)
        np.fill_diagonal(cost, 12.0)
    return -right.astype(np.float32), -down.astype(np.float32)


class RLPuzzleRefiner:
    """Inference-only environment matching the trained policy's observable state."""

    def __init__(self, tiles, row_logp, col_logp, layout, rng):
        self.n = N
        self.h = self.w = GRID
        self.right, self.down = rl_visual_matrices(tiles)
        self.row_logp = row_logp
        self.col_logp = col_logp
        self.board = layout.copy()
        self.rng = rng
        self.steps = 0
        self.stagnation = 0
        self.current_visual = self.visual_score()
        self.current_position = self.position_score()

    def neighbors(self, p):
        r, c = divmod(int(p), self.w)
        if c > 0:
            yield p - 1, p, 0
        if c + 1 < self.w:
            yield p, p + 1, 0
        if r > 0:
            yield p - self.w, p, 1
        if r + 1 < self.h:
            yield p, p + self.w, 1

    def edge_value(self, board, edges):
        value = 0.0
        for p, q, direction in edges:
            matrix = self.right if direction == 0 else self.down
            value += matrix[board[p], board[q]]
        return value

    def tile_position_score(self, tile, position):
        r, c = divmod(int(position), self.w)
        return float(self.row_logp[tile, r] + self.col_logp[tile, c])

    def visual_score(self):
        board = self.board.reshape(self.h, self.w)
        return float(
            self.right[board[:, :-1], board[:, 1:]].sum()
            + self.down[board[:-1], board[1:]].sum()
        )

    def position_score(self):
        return sum(self.tile_position_score(int(tile), p) for p, tile in enumerate(self.board))

    def observable_score(self):
        return self.current_visual + 0.15 * self.current_position

    def local_signature(self, board, p):
        r, c = divmod(int(p), self.w)
        tile = int(board[p])
        missing = -4.0
        left = self.right[int(board[p - 1]), tile] if c > 0 else missing
        right = self.right[tile, int(board[p + 1])] if c + 1 < self.w else missing
        up = self.down[int(board[p - self.w]), tile] if r > 0 else missing
        down = self.down[tile, int(board[p + self.w])] if r + 1 < self.h else missing
        return np.asarray([left, right, up, down], dtype=np.float32)

    def slot_candidate_scores(self, position):
        r, c = divmod(int(position), self.w)
        scores = np.zeros(self.n, dtype=np.float32)
        if c > 0:
            scores += self.right[int(self.board[position - 1])]
        if c + 1 < self.w:
            scores += self.right[:, int(self.board[position + 1])]
        if r > 0:
            scores += self.down[int(self.board[position - self.w])]
        if r + 1 < self.h:
            scores += self.down[:, int(self.board[position + self.w])]
        scores += 0.15 * (self.row_logp[:, r] + self.col_logp[:, c])
        return scores

    def proposal_pairs(self, k):
        max_pairs = self.n * (self.n - 1) // 2
        if not 1 <= k <= max_pairs:
            raise ValueError(f"proposal count must be in [1, {max_pairs}], got {k}")
        pairs = []
        seen = set()
        local = np.asarray([
            self.edge_value(self.board, set(self.neighbors(p))) for p in range(self.n)
        ], dtype=np.float32)
        bad_pool = np.argsort(local)[:max(8, self.n // 3)]
        inverse = np.empty(self.n, dtype=np.int32)
        inverse[self.board] = np.arange(self.n)
        anchors = self.rng.choice(bad_pool, size=min(8, len(bad_pool)), replace=False)
        per_anchor = max(2, (k * 2 // 3) // max(len(anchors), 1))
        for a in anchors:
            scores = self.slot_candidate_scores(int(a)).copy()
            scores[int(self.board[a])] = -np.inf
            take = min(per_anchor, self.n - 1)
            candidate_tiles = np.argpartition(scores, -take)[-take:]
            candidate_tiles = candidate_tiles[np.argsort(scores[candidate_tiles])[::-1]]
            for tile in candidate_tiles:
                pair = tuple(sorted((int(a), int(inverse[int(tile)]))))
                if pair[0] != pair[1] and pair not in seen:
                    pairs.append(pair)
                    seen.add(pair)
        attempts = 0
        while len(pairs) < k and attempts < max(1000, k * 20):
            attempts += 1
            a = int(self.rng.choice(bad_pool)) if len(pairs) < k * 5 // 6 else int(self.rng.integers(self.n))
            b = int(self.rng.integers(self.n - 1))
            if b >= a:
                b += 1
            pair = tuple(sorted((a, b)))
            if pair not in seen:
                pairs.append(pair)
                seen.add(pair)
        if len(pairs) < k:
            rows, cols = np.triu_indices(self.n, 1)
            remaining = [
                (int(a), int(b)) for a, b in zip(rows, cols)
                if (int(a), int(b)) not in seen
            ]
            take = self.rng.choice(len(remaining), size=k - len(pairs), replace=False)
            pairs.extend(remaining[int(index)] for index in np.atleast_1d(take))
        return np.asarray(pairs[:k], dtype=np.int32)

    def swap_stats(self, a, b):
        edges = set(self.neighbors(a)) | set(self.neighbors(b))
        before_visual = self.edge_value(self.board, edges)
        ta, tb = int(self.board[a]), int(self.board[b])
        before_pos_a = self.tile_position_score(ta, a)
        before_pos_b = self.tile_position_score(tb, b)
        before_sig_a = self.local_signature(self.board, a)
        before_sig_b = self.local_signature(self.board, b)
        self.board[a], self.board[b] = self.board[b], self.board[a]
        after_visual = self.edge_value(self.board, edges)
        after_pos_a = self.tile_position_score(tb, a)
        after_pos_b = self.tile_position_score(ta, b)
        after_sig_a = self.local_signature(self.board, a)
        after_sig_b = self.local_signature(self.board, b)
        self.board[a], self.board[b] = self.board[b], self.board[a]
        return {
            "visual_delta": after_visual - before_visual,
            "position_delta": after_pos_a + after_pos_b - before_pos_a - before_pos_b,
            "before_visual": before_visual,
            "after_visual": after_visual,
            "before_pos_a": before_pos_a,
            "before_pos_b": before_pos_b,
            "after_pos_a": after_pos_a,
            "after_pos_b": after_pos_b,
            "before_sig_a": before_sig_a,
            "before_sig_b": before_sig_b,
            "after_sig_a": after_sig_a,
            "after_sig_b": after_sig_b,
        }

    def proposal_features(self, pairs):
        features, stats = [], []
        board_visual = self.current_visual / max(2 * self.n - self.h - self.w, 1)
        board_position = self.current_position / self.n
        for a, b in pairs:
            s = self.swap_stats(int(a), int(b))
            stats.append(s)
            ar, ac = divmod(int(a), self.w)
            br, bc = divmod(int(b), self.w)
            features.append([
                np.tanh(s["visual_delta"] / 4), np.tanh(s["position_delta"] / 4),
                # Schema v1 intentionally duplicates the union score; the promoted checkpoint was trained this way.
                np.tanh(s["before_visual"] / 8), np.tanh(s["before_visual"] / 8),
                np.tanh(s["after_visual"] / 8), np.tanh(s["after_visual"] / 8),
                *np.tanh(s["before_sig_a"] / 4), *np.tanh(s["before_sig_b"] / 4),
                *np.tanh(s["after_sig_a"] / 4), *np.tanh(s["after_sig_b"] / 4),
                np.tanh(s["before_pos_a"] / 8), np.tanh(s["before_pos_b"] / 8),
                np.tanh(s["after_pos_a"] / 8), np.tanh(s["after_pos_b"] / 8),
                ar / (self.h - 1), ac / (self.w - 1), br / (self.h - 1), bc / (self.w - 1),
                (abs(ar - br) + abs(ac - bc)) / (self.h + self.w - 2),
                float(ar == br), float(ac == bc), np.tanh(board_visual / 4),
                np.tanh(board_position / 8), self.steps / max(self.n * 4, 1),
                min(self.stagnation / 100, 1.0), 1.0,
            ])
        return np.asarray(features, dtype=np.float32), stats

    def apply(self, pair, stats):
        a, b = map(int, pair)
        self.board[a], self.board[b] = self.board[b], self.board[a]
        self.current_visual += stats["visual_delta"]
        self.current_position += stats["position_delta"]
        self.steps += 1


@torch.inference_mode()
def refine_layout_rl(model, tiles, row_logp, col_logp, layout, seed):
    env = RLPuzzleRefiner(tiles, row_logp, col_logp, layout, np.random.default_rng(seed))
    best_board = env.board.copy()
    best_score = env.observable_score()
    no_gain = 0
    for _ in range(RL_STEPS):
        pairs = env.proposal_pairs(RL_PROPOSALS)
        features, stats = env.proposal_features(pairs)
        logits, _ = model(torch.from_numpy(features).unsqueeze(0).to(DEVICE))
        heuristic = torch.tensor(
            [s["visual_delta"] + 0.15 * s["position_delta"] for s in stats],
            device=DEVICE,
        )
        heuristic = (heuristic - heuristic.mean()) / heuristic.std(correction=0).clamp_min(1e-6)
        action = int((logits[0] + RL_POLICY_BLEND * heuristic).argmax().item())
        env.apply(pairs[action], stats[action])
        score = env.observable_score()
        if score > best_score:
            best_score = score
            best_board = env.board.copy()
            no_gain = 0
        else:
            no_gain += 1
        env.stagnation = no_gain
        if no_gain >= RL_STAGNATION:
            break
    return best_board, best_score, env.steps


def assemble(tiles, layout):
    layout = np.asarray(layout)
    if layout.shape != (N,) or not np.array_equal(np.sort(layout), np.arange(N)):
        raise ValueError("layout must be a permutation of 0..N-1")
    board = tiles[layout].reshape(GRID, GRID, TILE, TILE, 3)
    return board.transpose(0, 2, 1, 3, 4).reshape(GRID * TILE, GRID * TILE, 3)


def select_layout_candidate(baseline_layout, candidate_layout, right, down, pos_score):
    baseline_value = layout_objective(baseline_layout, right, down, pos_score)
    candidate_value = layout_objective(candidate_layout, right, down, pos_score)
    accepted = candidate_value >= baseline_value
    return (candidate_layout if accepted else baseline_layout), baseline_value, candidate_value, accepted


def solve_one(path, edge_model, pos_model, restorer_model, restorer_config, rl_model, seed):
    tiles = load_tiles(path)
    tiles_t = restore_tiles(restorer_model, tiles_tensor(tiles))
    clean_tiles = ((tiles_t.numpy().transpose(0, 2, 3, 1) + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)
    pos, row_logp, col_logp = position_scores(pos_model, tiles_t)
    right = compatibility_scores(edge_model, clean_tiles, tiles_t, 0)
    down = compatibility_scores(edge_model, clean_tiles, tiles_t, 1)
    baseline_layout = optimize_layout(initial_layout(pos), right, down, pos, seed)
    layout = baseline_layout
    if rl_model is not None:
        rl_layout, rl_objective, rl_steps = refine_layout_rl(
            rl_model, clean_tiles, row_logp, col_logp, baseline_layout, seed + 50000
        )
        layout, baseline_value, candidate_value, accepted = select_layout_candidate(
            baseline_layout, rl_layout, right, down, pos
        )
        print(
            f"rl_refine steps={rl_steps}/{RL_STEPS} proposals={RL_PROPOSALS} "
            f"rl_objective={rl_objective:.3f} baseline_value={baseline_value:.3f} "
            f"candidate_value={candidate_value:.3f} accepted={int(accepted)}"
        )
    return assemble(clean_tiles, layout), layout, assemble(clean_tiles, baseline_layout)


def validate(data_root, edge_model, pos_model, restorer_model, restorer_config, rl_model):
    inputs = sorted((data_root / "train" / "inputs").glob("*.png"))[:VALIDATE_IMAGES]
    if not inputs:
        raise FileNotFoundError(f"No validation PNG files found in {data_root / 'train' / 'inputs'}")
    rl_scores, baseline_scores = [], []
    for i, path in enumerate(inputs):
        pred, _, baseline_pred = solve_one(
            path, edge_model, pos_model, restorer_model, restorer_config, rl_model, SEED + i
        )
        target = np.asarray(Image.open(data_root / "train" / "targets" / path.name).convert("RGB"))
        rl_score = structural_similarity(target, pred, channel_axis=2, data_range=255)
        baseline_score = structural_similarity(target, baseline_pred, channel_axis=2, data_range=255)
        rl_scores.append(rl_score)
        baseline_scores.append(baseline_score)
        Image.fromarray(pred).save(OUT_DIR / f"validation_{path.stem}.png")
        print(
            f"validation file={path.name} rl_ssim={rl_score:.6f} "
            f"baseline_ssim={baseline_score:.6f} delta={rl_score - baseline_score:+.6f}"
        )
    mean_rl = float(np.mean(rl_scores))
    mean_baseline = float(np.mean(baseline_scores))
    mean_delta = mean_rl - mean_baseline
    print(
        f"validation_mean_rl_ssim={mean_rl:.6f} "
        f"validation_mean_baseline_ssim={mean_baseline:.6f} delta={mean_delta:+.6f}"
    )
    return {"rl_ssim": mean_rl, "baseline_ssim": mean_baseline, "delta": mean_delta}


def finalize_submission_zip(created, expected_names, zip_path):
    temp_zip = zip_path.with_suffix(zip_path.suffix + ".tmp")
    with zipfile.ZipFile(temp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in created:
            zf.write(path, arcname=path.name)
        archived = set(zf.namelist())
    expected = set(expected_names)
    if archived != expected:
        raise RuntimeError(
            f"Submission archive mismatch: missing={sorted(expected - archived)} extra={sorted(archived - expected)}"
        )
    os.replace(temp_zip, zip_path)
    return archived


def solve_test(data_root, edge_model, pos_model, restorer_model, restorer_config, rl_model):
    candidates = [data_root / "test", data_root / "test" / "inputs"]
    test_dir = next((path for path in candidates if list(path.glob("*.png"))), None)
    if test_dir is None:
        raise FileNotFoundError(f"No test PNG files found in: {candidates}")
    files = sorted(test_dir.glob("*.png"))
    zip_path = OUT_DIR / "submission.zip"
    with tempfile.TemporaryDirectory(prefix="submission-", dir=OUT_DIR) as temp_dir:
        temp_dir = Path(temp_dir)
        created = []
        for i, path in enumerate(tqdm(files, desc="test puzzles")):
            pred, _, _ = solve_one(
                path, edge_model, pos_model, restorer_model, restorer_config, rl_model, SEED + 1000 + i
            )
            output = temp_dir / path.name
            Image.fromarray(pred).save(output)
            created.append(output)
        archived = finalize_submission_zip(created, (path.name for path in files), zip_path)
    print(f"submission={zip_path} files={len(files)} archived={len(archived)}")


def main():
    validate_config()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE}")
    data_root = find_data_root()
    edge, pos, restorer, restorer_config, rl = load_models()
    metrics = validate(data_root, edge, pos, restorer, restorer_config, rl)
    if SOLVE_TEST:
        test_rl = rl
        if rl is not None and metrics["delta"] < RL_VALIDATION_MIN_DELTA:
            print(
                f"rl_disabled_for_test delta={metrics['delta']:+.6f} "
                f"required={RL_VALIDATION_MIN_DELTA:+.6f}"
            )
            test_rl = None
        solve_test(data_root, edge, pos, restorer, restorer_config, test_rl)


if __name__ == "__main__":
    main()
