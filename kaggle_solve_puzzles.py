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

import kaggle_e14_solver
import kaggle_e18b_postprocess


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
VALIDATE_IMAGES = int(os.getenv("VALIDATE_IMAGES", "64"))
SOLVE_TEST = os.getenv("SOLVE_TEST", "1") == "1"
EDGE_BATCH = int(os.getenv("EDGE_BATCH", "4096"))
RELATION_BATCH = int(os.getenv("RELATION_BATCH", "512"))
EDGE_TOPK = int(os.getenv("EDGE_TOPK", "2"))
RELATION_TOPK = int(os.getenv("RELATION_TOPK", "2"))
RELATION_MIN_PROB = float(os.getenv("RELATION_MIN_PROB", "0.65"))
RELATION_MIN_MARGIN = float(os.getenv("RELATION_MIN_MARGIN", "0.50"))
RELATION_GUARD_WEIGHT = float(os.getenv("RELATION_GUARD_WEIGHT", "0.25"))
RELATION_MIN_GAIN = float(os.getenv("RELATION_MIN_GAIN", "0.75"))
RELATION_BASE_TOL = float(os.getenv("RELATION_BASE_TOL", "0.05"))
RELATION_POSITION_TOL = float(os.getenv("RELATION_POSITION_TOL", "0.25"))
RELATION_MAX_SWAPS = int(os.getenv("RELATION_MAX_SWAPS", "64"))
# Retained only for the isolated regression test of the retired v2 initializer.
GREEDY_TOPK = 4
SWAP_STEPS = int(os.getenv("SWAP_STEPS", "20000"))
POSITION_WEIGHT = float(os.getenv("POSITION_WEIGHT", "0.12"))
RESTORE_BATCH = int(os.getenv("RESTORE_BATCH", "512"))
USE_RL = os.getenv("USE_RL", "1") == "1"
USE_E14 = os.getenv("USE_E14", "1") == "1"
E14_FALLBACK_ON_ERROR = os.getenv("E14_FALLBACK_ON_ERROR", "1") == "1"
USE_E18B = os.getenv("USE_E18B", "1") == "1"
E18B_FALLBACK_ON_ERROR = os.getenv("E18B_FALLBACK_ON_ERROR", "1") == "1"
RL_PROPOSALS = int(os.getenv("RL_PROPOSALS", "48"))
RL_STEPS = int(os.getenv("RL_STEPS", "800"))
RL_STAGNATION = int(os.getenv("RL_STAGNATION", "120"))
RL_POLICY_BLEND = float(os.getenv("RL_POLICY_BLEND", "0.75"))
RELATION_VALIDATION_MIN_DELTA = float(os.getenv("RELATION_VALIDATION_MIN_DELTA", "0.0"))
DATA_ROOT = os.getenv("DATA_ROOT")
ASSEMBLY_MODEL_ROOT = Path(os.getenv("ASSEMBLY_MODEL_ROOT", "/kaggle/input/pazzle-puzzle-assembly-models"))
RESTORER_MODEL_ROOT = Path(os.getenv("RESTORER_MODEL_ROOT", "/kaggle/input/pazzle-fragment-restorer"))
RL_MODEL_ROOT = Path(os.getenv("RL_MODEL_ROOT", "/kaggle/input/pazzle-rl-puzzle-assembler"))
RELATION_MODEL_ROOT = Path(os.getenv(
    "RELATION_MODEL_ROOT", "/kaggle/input/pazzle-continue-pair-on-restorer"
))
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


class SeamScorer(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1), nn.GroupNorm(8, base * 2), nn.SiLU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.GroupNorm(8, base * 2), nn.SiLU(),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1), nn.GroupNorm(8, base * 4), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(base * 4, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class PairRelationClassifier(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.horizontal = SeamScorer(base)
        self.vertical = SeamScorer(base)
        self.not_adjacent_logit = nn.Parameter(torch.zeros(()))

    def forward(self, a, b):
        horizontal = torch.cat(
            [torch.cat([b, a], dim=3), torch.cat([a, b], dim=3)], dim=0
        )
        left, right = self.horizontal(horizontal).chunk(2, dim=0)
        vertical = torch.cat(
            [torch.cat([b, a], dim=2), torch.cat([a, b], dim=2)], dim=0
        )
        up, down = self.vertical(vertical).chunk(2, dim=0)
        none = self.not_adjacent_logit.expand_as(left)
        return torch.stack([none, left, right, up, down], dim=1)


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


def find_relation_checkpoint():
    name = "pair_relation_restorer_continued_best.pt"
    root = resolve_model_root(RELATION_MODEL_ROOT, [name], "pair relation")
    files = list(root.rglob(name))
    if len(files) != 1:
        raise RuntimeError(f"Expected one relation checkpoint, found: {files}")
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
    relation_path = find_relation_checkpoint()
    relation_ckpt = torch.load(relation_path, map_location="cpu")
    relation = PairRelationClassifier()
    relation.load_state_dict(relation_ckpt["model"])
    relation_epoch = relation_ckpt.get("epoch")
    del relation_ckpt
    relation = relation.to(DEVICE).eval()
    print(f"relation_checkpoint={relation_path} epoch={relation_epoch}")
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
    return edge, relation, pos, restorer, restorer_config, rl


def validate_config():
    integer_ranges = {
        "VALIDATE_IMAGES": (VALIDATE_IMAGES, 1, None),
        "EDGE_BATCH": (EDGE_BATCH, 1, None),
        "RELATION_BATCH": (RELATION_BATCH, 1, None),
        "EDGE_TOPK": (EDGE_TOPK, 1, N - 1),
        "RELATION_TOPK": (RELATION_TOPK, 1, N - 1),
        "RELATION_MAX_SWAPS": (RELATION_MAX_SWAPS, 0, None),
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
        "RELATION_MIN_PROB": RELATION_MIN_PROB,
        "RELATION_MIN_MARGIN": RELATION_MIN_MARGIN,
        "RELATION_GUARD_WEIGHT": RELATION_GUARD_WEIGHT,
        "RELATION_MIN_GAIN": RELATION_MIN_GAIN,
        "RELATION_BASE_TOL": RELATION_BASE_TOL,
        "RELATION_POSITION_TOL": RELATION_POSITION_TOL,
        "RL_POLICY_BLEND": RL_POLICY_BLEND,
        "RELATION_VALIDATION_MIN_DELTA": RELATION_VALIDATION_MIN_DELTA,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
    if not 0.0 <= RELATION_MIN_PROB <= 1.0:
        raise ValueError(f"RELATION_MIN_PROB must be in [0, 1], got {RELATION_MIN_PROB}")


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


@torch.inference_mode()
def relation_confidence_scores(model, tiles, tiles_t, direction):
    """Sparse high-precision directional scores with explicit reverse confirmation."""
    seam = seam_distance(tiles, direction)
    k = min(RELATION_TOPK, N - 1)
    candidates = np.argpartition(seam, k, axis=1)[:, :k]
    src = np.repeat(np.arange(N), k)
    dst = candidates.reshape(-1)
    pair_src = np.concatenate([src, dst])
    pair_dst = np.concatenate([dst, src])
    outputs = []
    for start in range(0, len(pair_src), RELATION_BATCH):
        ia = torch.from_numpy(pair_src[start:start + RELATION_BATCH])
        ib = torch.from_numpy(pair_dst[start:start + RELATION_BATCH])
        outputs.append(model(tiles_t[ia].to(DEVICE), tiles_t[ib].to(DEVICE)).float().cpu())
    logits = torch.cat(outputs).numpy()
    forward, reverse = np.split(logits, 2)
    forward_class = 2 if direction == 0 else 4
    reverse_class = 1 if direction == 0 else 3
    forward_prob = torch.softmax(torch.from_numpy(forward), 1).numpy()[:, forward_class]
    reverse_prob = torch.softmax(torch.from_numpy(reverse), 1).numpy()[:, reverse_class]
    forward_other = np.max(np.delete(forward, forward_class, axis=1), axis=1)
    reverse_other = np.max(np.delete(reverse, reverse_class, axis=1), axis=1)
    margin = np.minimum(
        forward[:, forward_class] - forward_other,
        reverse[:, reverse_class] - reverse_other,
    )
    strength = np.minimum(
        forward[:, forward_class] - forward[:, 0],
        reverse[:, reverse_class] - reverse[:, 0],
    )
    keep = (
        (forward_prob >= RELATION_MIN_PROB)
        & (reverse_prob >= RELATION_MIN_PROB)
        & (margin >= RELATION_MIN_MARGIN)
    )
    score = np.zeros((N, N), dtype=np.float32)
    score[src[keep], dst[keep]] = np.clip(strength[keep], 0.0, 6.0)
    return score


def greedy_graph_layout(right, down, pos_score):
    """Build coordinate-consistent tile components from the best directed edges."""
    components = {tile: {tile: (0, 0)} for tile in range(N)}
    root_of = np.arange(N, dtype=np.int32)
    edges = []
    k = min(GREEDY_TOPK, N - 1)
    for matrix, (dr, dc) in ((right, (0, 1)), (down, (1, 0))):
        for source in range(N):
            candidates = np.argpartition(matrix[source], -k)[-k:]
            for target in candidates:
                if source != int(target):
                    edges.append((float(matrix[source, target]), source, int(target), dr, dc))
    edges.sort(key=lambda item: item[0], reverse=True)

    accepted_edges = 0
    for _, source, target, dr, dc in edges:
        root_a, root_b = int(root_of[source]), int(root_of[target])
        comp_a, comp_b = components[root_a], components[root_b]
        ar, ac = comp_a[source]
        br, bc = comp_b[target]
        shift = (ar + dr - br, ac + dc - bc)
        if root_a == root_b:
            if shift == (0, 0):
                accepted_edges += 1
            continue
        translated = {
            tile: (row + shift[0], col + shift[1])
            for tile, (row, col) in comp_b.items()
        }
        occupied = set(comp_a.values())
        if occupied.intersection(translated.values()):
            continue
        combined_coords = list(comp_a.values()) + list(translated.values())
        rows = [coord[0] for coord in combined_coords]
        cols = [coord[1] for coord in combined_coords]
        if max(rows) - min(rows) + 1 > GRID or max(cols) - min(cols) + 1 > GRID:
            continue
        comp_a.update(translated)
        for tile in translated:
            root_of[tile] = root_a
        del components[root_b]
        accepted_edges += 1

    layout = np.full(N, -1, dtype=np.int32)
    occupied = np.zeros(N, dtype=bool)
    deferred = []
    ordered_components = sorted(components.values(), key=len, reverse=True)
    placed_components = 0
    largest_component = len(ordered_components[0]) if ordered_components else 0
    for component in ordered_components:
        min_row = min(row for row, _ in component.values())
        min_col = min(col for _, col in component.values())
        normalized = {
            tile: (row - min_row, col - min_col)
            for tile, (row, col) in component.items()
        }
        height = max(row for row, _ in normalized.values()) + 1
        width = max(col for _, col in normalized.values()) + 1
        best = None
        for top in range(GRID - height + 1):
            for left in range(GRID - width + 1):
                placements = [
                    (tile, (top + row) * GRID + left + col)
                    for tile, (row, col) in normalized.items()
                ]
                if any(occupied[position] for _, position in placements):
                    continue
                value = sum(float(pos_score[tile, position]) for tile, position in placements)
                if best is None or value > best[0]:
                    best = (value, placements)
        if best is None:
            deferred.extend(component)
            continue
        for tile, position in best[1]:
            layout[position] = tile
            occupied[position] = True
        placed_components += 1

    remaining_tiles = np.asarray(
        deferred + [tile for tile in range(N) if tile not in set(layout[layout >= 0])],
        dtype=np.int32,
    )
    if len(remaining_tiles):
        remaining_tiles = np.unique(remaining_tiles)
        remaining_positions = np.flatnonzero(~occupied)
        if len(remaining_tiles) != len(remaining_positions):
            raise RuntimeError("Greedy component placement lost or duplicated tiles")
        tile_idx, position_idx = linear_sum_assignment(
            -pos_score[np.ix_(remaining_tiles, remaining_positions)]
        )
        layout[remaining_positions[position_idx]] = remaining_tiles[tile_idx]
    if not np.array_equal(np.sort(layout), np.arange(N)):
        raise RuntimeError("Greedy graph layout is not a complete permutation")
    stats = {
        "components": len(ordered_components),
        "largest": largest_component,
        "placed": placed_components,
        "accepted_edges": accepted_edges,
    }
    return layout, stats


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


def relation_local_value(layout, positions, right, down):
    edges = set()
    for p in positions:
        r, c = divmod(int(p), GRID)
        if c > 0: edges.add((p - 1, p, 0))
        if c + 1 < GRID: edges.add((p, p + 1, 0))
        if r > 0: edges.add((p - GRID, p, 1))
        if r + 1 < GRID: edges.add((p, p + GRID, 1))
    return float(sum(
        (right if direction == 0 else down)[layout[a], layout[b]]
        for a, b, direction in edges
    ))


def relation_guarded_refine(layout, right, down, pos_score, relation_right, relation_down):
    """Apply only confident relation moves that preserve the trusted v5 objective."""
    layout = layout.copy()
    inverse = np.empty(N, dtype=np.int32)
    inverse[layout] = np.arange(N)
    edges = []
    for matrix, direction in ((relation_right, 0), (relation_down, 1)):
        for source, target in np.argwhere(matrix > 0):
            edges.append((float(matrix[source, target]), int(source), int(target), direction))
    edges.sort(reverse=True)
    attempted = accepted = 0
    for strength, source, target, direction in edges:
        if accepted >= RELATION_MAX_SWAPS:
            break
        source_pos, target_pos = int(inverse[source]), int(inverse[target])
        sr, sc = divmod(source_pos, GRID)
        tr, tc = divmod(target_pos, GRID)
        if (direction == 0 and sr == tr and sc + 1 == tc) or (
            direction == 1 and sr + 1 == tr and sc == tc
        ):
            continue
        proposals = []
        if direction == 0:
            if sc + 1 < GRID: proposals.append((source_pos + 1, target_pos))
            if tc > 0: proposals.append((source_pos, target_pos - 1))
        else:
            if sr + 1 < GRID: proposals.append((source_pos + GRID, target_pos))
            if tr > 0: proposals.append((source_pos, target_pos - GRID))
        best = None
        for a, b in dict.fromkeys(tuple(sorted(pair)) for pair in proposals):
            if a == b:
                continue
            attempted += 1
            positions = {a, b}
            before_base = local_value(layout, positions, right, down, pos_score)
            before_relation = relation_local_value(
                layout, positions, relation_right, relation_down
            )
            ta, tb = int(layout[a]), int(layout[b])
            before_position = float(pos_score[ta, a] + pos_score[tb, b])
            layout[a], layout[b] = layout[b], layout[a]
            after_base = local_value(layout, positions, right, down, pos_score)
            after_relation = relation_local_value(
                layout, positions, relation_right, relation_down
            )
            after_position = float(pos_score[tb, a] + pos_score[ta, b])
            layout[a], layout[b] = layout[b], layout[a]
            base_delta = after_base - before_base
            relation_delta = after_relation - before_relation
            position_delta = after_position - before_position
            combined = base_delta + RELATION_GUARD_WEIGHT * relation_delta
            allowed = (
                relation_delta >= RELATION_MIN_GAIN
                and base_delta >= -RELATION_BASE_TOL
                and position_delta >= -RELATION_POSITION_TOL
                and combined > 0
            )
            if allowed and (best is None or combined > best[0]):
                best = (combined, a, b, base_delta, relation_delta, position_delta, strength)
        if best is None:
            continue
        _, a, b, *_ = best
        ta, tb = int(layout[a]), int(layout[b])
        layout[a], layout[b] = layout[b], layout[a]
        inverse[ta], inverse[tb] = b, a
        accepted += 1
    if not np.array_equal(np.sort(layout), np.arange(N)):
        raise RuntimeError("Relation guard produced an invalid permutation")
    return layout, {"confident_edges": len(edges), "attempted": attempted, "accepted": accepted}


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

    def _local_after_score(self, a, b, position):
        """Local score at one endpoint on the temporary board after swapping."""
        self.board[a], self.board[b] = self.board[b], self.board[a]
        value = self.edge_value(self.board, set(self.neighbors(position)))
        self.board[a], self.board[b] = self.board[b], self.board[a]
        return value

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
                np.tanh(self.edge_value(self.board, set(self.neighbors(int(a)))) / 8),
                np.tanh(self.edge_value(self.board, set(self.neighbors(int(b)))) / 8),
                np.tanh(self._local_after_score(int(a), int(b), int(a)) / 8),
                np.tanh(self._local_after_score(int(a), int(b), int(b)) / 8),
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


def select_e14_or_fallback(raw_tiles, right, down, pos_score, seed, fallback_layout):
    """Run target-free E14, retaining the previous solver as a safe fallback."""
    fallback_layout = np.asarray(fallback_layout, dtype=np.int32)
    if not USE_E14:
        return fallback_layout, False, "disabled"
    try:
        fused_right, fused_down = kaggle_e14_solver.fused_directional_scores(
            raw_tiles,
            right,
            down,
            # EdgeMatcher emits finite compatibility logits, while verified E14
            # receives learned log-probabilities. Normalize once before fusion.
            learned_are_logp=False,
        )
        candidate = kaggle_e14_solver.solve_layout(
            fused_right, fused_down, pos_score, seed
        )
        if not kaggle_e14_solver.is_valid_layout(candidate):
            raise ValueError("E14 returned an invalid tile permutation")
        return np.asarray(candidate, dtype=np.int32), True, None
    except Exception as exc:
        if not E14_FALLBACK_ON_ERROR:
            raise
        return fallback_layout, False, f"{type(exc).__name__}: {exc}"


def select_e18b_or_raw(raw_assembled):
    """Apply target-free E18b, retaining raw E14 pixels as a safe fallback."""
    return kaggle_e18b_postprocess.polish_or_raw(
        raw_assembled,
        enabled=USE_E18B,
        fallback_on_error=E18B_FALLBACK_ON_ERROR,
    )


def solve_one(path, edge_model, relation_model, pos_model, restorer_model, restorer_config,
              rl_model, seed, use_relation_guard=True):
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
    v5_layout = layout.copy()
    if use_relation_guard:
        relation_right = relation_confidence_scores(
            relation_model, clean_tiles, tiles_t, 0
        )
        relation_down = relation_confidence_scores(
            relation_model, clean_tiles, tiles_t, 1
        )
        layout, guard_stats = relation_guarded_refine(
            v5_layout, right, down, pos, relation_right, relation_down
        )
        print(
            f"relation_guard confident_edges={guard_stats['confident_edges']} "
            f"attempted={guard_stats['attempted']} accepted={guard_stats['accepted']}"
        )
    layout, e14_selected, e14_reason = select_e14_or_fallback(
        tiles, right, down, pos, seed, layout
    )
    print(
        f"e14_fusion_relaxation selected={int(e14_selected)} "
        f"fallback_reason={e14_reason or 'none'}"
    )
    raw_e14 = assemble(tiles, layout)
    pred, e18b_selected, e18b_reason, e18b_stats = select_e18b_or_raw(raw_e14)
    print(
        f"e18b_guarded_nlm selected={int(e18b_selected)} "
        f"fallback_reason={e18b_reason or 'none'} "
        f"raw_gray={e18b_stats['raw_gray_count']} "
        f"unguarded_gray={e18b_stats['unguarded_gray_count']} "
        f"guarded_gray={e18b_stats['guarded_gray_count']} "
        f"reverted={e18b_stats['reverted_new_gray_cells']}"
    )
    raw_v5 = assemble(tiles, v5_layout)
    baseline_pred, _, _, _ = select_e18b_or_raw(raw_v5)
    return pred, layout, baseline_pred


def validate(data_root, edge_model, relation_model, pos_model, restorer_model, restorer_config,
             rl_model):
    available = sorted((data_root / "train" / "inputs").glob("*.png"))
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(len(available), size=min(VALIDATE_IMAGES, len(available)), replace=False)
    inputs = [available[int(index)] for index in sorted(chosen)]
    if not inputs:
        raise FileNotFoundError(f"No validation PNG files found in {data_root / 'train' / 'inputs'}")
    rl_scores, baseline_scores = [], []
    for i, path in enumerate(inputs):
        pred, _, baseline_pred = solve_one(
            path, edge_model, relation_model, pos_model, restorer_model, restorer_config,
            rl_model, SEED + i,
        )
        target = np.asarray(Image.open(data_root / "train" / "targets" / path.name).convert("RGB"))
        rl_score = structural_similarity(target, pred, channel_axis=2, data_range=255)
        baseline_score = structural_similarity(target, baseline_pred, channel_axis=2, data_range=255)
        rl_scores.append(rl_score)
        baseline_scores.append(baseline_score)
        Image.fromarray(pred).save(OUT_DIR / f"validation_{path.stem}.png")
        print(
            f"validation file={path.name} solver_ssim={rl_score:.6f} "
            f"v5_baseline_ssim={baseline_score:.6f} delta={rl_score - baseline_score:+.6f}"
        )
    mean_rl = float(np.mean(rl_scores))
    mean_baseline = float(np.mean(baseline_scores))
    mean_delta = mean_rl - mean_baseline
    print(
        f"validation_mean_solver_ssim={mean_rl:.6f} "
        f"validation_mean_v5_baseline_ssim={mean_baseline:.6f} delta={mean_delta:+.6f}"
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


def solve_test(data_root, edge_model, relation_model, pos_model, restorer_model, restorer_config,
               rl_model, use_relation_guard):
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
                path, edge_model, relation_model, pos_model, restorer_model, restorer_config,
                rl_model, SEED + 1000 + i, use_relation_guard=use_relation_guard
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
    edge, relation, pos, restorer, restorer_config, rl = load_models()
    metrics = validate(data_root, edge, relation, pos, restorer, restorer_config, rl)
    if SOLVE_TEST:
        use_relation_guard = metrics["delta"] >= RELATION_VALIDATION_MIN_DELTA
        if not use_relation_guard:
            print(
                f"relation_guard_disabled_for_test delta={metrics['delta']:+.6f} "
                f"required={RELATION_VALIDATION_MIN_DELTA:+.6f}"
            )
        solve_test(
            data_root, edge, relation, pos, restorer, restorer_config, rl,
            use_relation_guard=use_relation_guard,
        )


if __name__ == "__main__":
    main()
