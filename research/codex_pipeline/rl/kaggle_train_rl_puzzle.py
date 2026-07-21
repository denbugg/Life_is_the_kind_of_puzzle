import io
import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from skimage.metrics import structural_similarity
from torch.distributions import Categorical
from tqdm.auto import tqdm


GRID = 24
TILE = 20
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "7000"))
EPOCHS = int(os.getenv("EPOCHS", "12"))
STEPS_PER_EPOCH = int(os.getenv("STEPS_PER_EPOCH", "8192"))
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "4096"))
PROPOSALS = int(os.getenv("PROPOSALS", "96"))
ROLLOUT_STEPS = int(os.getenv("ROLLOUT_STEPS", "512"))
PPO_EPOCHS = int(os.getenv("PPO_EPOCHS", "4"))
MINIBATCH = int(os.getenv("MINIBATCH", "64"))
LR = float(os.getenv("LR", "3e-4"))
GAMMA = float(os.getenv("GAMMA", "0.995"))
GAE_LAMBDA = float(os.getenv("GAE_LAMBDA", "0.95"))
CLIP_EPS = float(os.getenv("CLIP_EPS", "0.2"))
ENTROPY_COEF = float(os.getenv("ENTROPY_COEF", "0.005"))
VALUE_COEF = float(os.getenv("VALUE_COEF", "0.5"))
BC_COEF = float(os.getenv("BC_COEF", "0.25"))
POLICY_BLEND = float(os.getenv("POLICY_BLEND", "0.75"))
VALIDATE_IMAGES = int(os.getenv("VALIDATE_IMAGES", "8"))
VALIDATE_STEPS = int(os.getenv("VALIDATE_STEPS", "5000"))
SEED = int(os.getenv("SEED", "20260719"))
OUT_DIR = Path(os.getenv("OUT_DIR", "/kaggle/working"))


FEATURE_NAMES = [
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
FEATURE_DIM = len(FEATURE_NAMES)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        probe = torch.ones(4, device="cuda").sum().item()
        print(f"gpu={torch.cuda.get_device_name(0)} probe={probe}")
        return torch.device("cuda")
    except Exception as exc:
        print(f"cuda_unusable={type(exc).__name__}: {exc}")
        return torch.device("cpu")


def find_data_root():
    for root in [Path("/kaggle/input"), Path(".")]:
        if not root.exists():
            continue
        for targets in root.rglob("train/targets"):
            if targets.exists():
                return targets.parent.parent
    raise FileNotFoundError("train/targets not found")


def find_latest(pattern):
    paths = list(Path("/kaggle/input").rglob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    return max(paths, key=lambda p: int(p.stem.rsplit("epoch", 1)[-1]))


def split_tiles(image):
    return image.reshape(GRID, TILE, GRID, TILE, 3).transpose(0, 2, 1, 3, 4)


def assemble(tiles, board, h, w):
    x = tiles[board].reshape(h, w, TILE, TILE, 3)
    return x.transpose(0, 2, 1, 3, 4).reshape(h * TILE, w * TILE, 3)


class TargetCache:
    def __init__(self, files, capacity=24):
        self.files = files
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        image = np.asarray(Image.open(self.files[index]).convert("RGB"), dtype=np.uint8)
        tiles = split_tiles(image)
        self.cache[index] = tiles
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return tiles


class PositionPrior(nn.Module):
    def __init__(self, base=48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base * 2, 4, 2, 1), nn.GroupNorm(8, base * 2), nn.SiLU(),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1), nn.GroupNorm(8, base * 4), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.row_head = nn.Linear(base * 4, GRID)
        self.col_head = nn.Linear(base * 4, GRID)

    def forward(self, x):
        h = self.encoder(x)
        return self.row_head(h), self.col_head(h)


class SwapActorCritic(nn.Module):
    """A fully connected policy over a fixed set of proposed swaps."""

    def __init__(self, feature_dim=FEATURE_DIM, hidden=256):
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
        value = self.value(pooled).squeeze(-1)
        return logits, value


@torch.inference_mode()
def position_log_scores(model, tiles, device):
    x = torch.from_numpy(np.ascontiguousarray(tiles.transpose(0, 3, 1, 2))).float() / 127.5 - 1.0
    rows, cols = [], []
    for batch in x.split(512):
        r, c = model(batch.to(device))
        rows.append(F.log_softmax(r, 1).cpu())
        cols.append(F.log_softmax(c, 1).cpu())
    return torch.cat(rows).numpy(), torch.cat(cols).numpy()


def edge_cost_matrix(a, b):
    a = a.reshape(len(a), -1).astype(np.float32) / 255.0
    b = b.reshape(len(b), -1).astype(np.float32) / 255.0
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * a @ b.T, 0.0) / a.shape[1]


def visual_matrices(tiles):
    right_cost = edge_cost_matrix(tiles[:, :, -2:, :], tiles[:, :, :2, :])
    down_cost = edge_cost_matrix(tiles[:, -2:, :, :], tiles[:, :2, :, :])
    for cost in [right_cost, down_cost]:
        finite = cost[np.isfinite(cost)]
        cost /= max(float(np.median(finite)), 1e-6)
        np.fill_diagonal(cost, 12.0)
    return -right_cost.astype(np.float32), -down_cost.astype(np.float32)


class PuzzleSwapEnv:
    def __init__(self, tiles, h, w, row_logp, col_logp, scramble, rng):
        self.h, self.w = h, w
        self.n = h * w
        self.tiles = tiles.reshape(self.n, TILE, TILE, 3)
        self.right, self.down = visual_matrices(self.tiles)
        self.row_logp, self.col_logp = row_logp, col_logp
        self.rng = rng
        self.board = np.arange(self.n, dtype=np.int32)
        if scramble >= self.n:
            self.rng.shuffle(self.board)
        else:
            for _ in range(scramble):
                a, b = self.rng.choice(self.n, 2, replace=False)
                self.board[a], self.board[b] = self.board[b], self.board[a]
        self.initial_errors = max(int(np.sum(self.board != np.arange(self.n))), 1)
        self.steps = 0
        self.stagnation = 0
        self.current_visual = self.visual_score()
        self.current_position = self.position_score()

    def neighbors(self, p):
        r, c = divmod(int(p), self.w)
        if c > 0: yield p - 1, p, 0
        if c + 1 < self.w: yield p, p + 1, 0
        if r > 0: yield p - self.w, p, 1
        if r + 1 < self.h: yield p, p + self.w, 1

    def affected_edges(self, a, b):
        edges = set(self.neighbors(a)) | set(self.neighbors(b))
        return edges

    def edge_value(self, board, edges):
        value = 0.0
        for p, q, direction in edges:
            matrix = self.right if direction == 0 else self.down
            value += matrix[board[p], board[q]]
        return value

    def correct_edges(self, board, edges):
        value = 0
        for p, q, direction in edges:
            ta, tb = int(board[p]), int(board[q])
            if direction == 0:
                value += int(ta // self.w == tb // self.w and tb == ta + 1)
            else:
                value += int(tb == ta + self.w)
        return value

    def tile_position_score(self, tile, position):
        r, c = divmod(int(position), self.w)
        # Crop coordinates retain their absolute 24x24 labels.
        return float(self.row_logp[tile, r] + self.col_logp[tile, c])

    def visual_score(self):
        board = self.board.reshape(self.h, self.w)
        return float(self.right[board[:, :-1], board[:, 1:]].sum() + self.down[board[:-1], board[1:]].sum())

    def position_score(self):
        return sum(self.tile_position_score(int(t), p) for p, t in enumerate(self.board))

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

    def tile_distance(self, tile, position):
        tr, tc = divmod(int(tile), self.w)
        pr, pc = divmod(int(position), self.w)
        return abs(tr - pr) + abs(tc - pc)

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

    def metrics(self):
        board = self.board.reshape(self.h, self.w)
        h_ok = ((board[:, 1:] == board[:, :-1] + 1) & (board[:, 1:] // self.w == board[:, :-1] // self.w)).mean()
        v_ok = (board[1:] == board[:-1] + self.w).mean()
        return {
            "position_acc": float(np.mean(self.board == np.arange(self.n))),
            "adjacency_acc": float((h_ok + v_ok) * 0.5),
            "visual": self.visual_score() / max(2 * self.n - self.h - self.w, 1),
        }

    def proposal_pairs(self, k):
        # Guided proposals place a visually compatible tile into a weak slot.
        pairs = []
        seen = set()
        local = np.zeros(self.n, dtype=np.float32)
        for p in range(self.n):
            local[p] = self.edge_value(self.board, set(self.neighbors(p)))
        bad_pool = np.argsort(local)[:max(8, self.n // 3)]
        inverse = np.empty(self.n, dtype=np.int32)
        inverse[self.board] = np.arange(self.n)
        anchors = self.rng.choice(bad_pool, size=min(8, len(bad_pool)), replace=False)
        candidates_per_anchor = max(2, (k * 2 // 3) // max(len(anchors), 1))
        for a in anchors:
            scores = self.slot_candidate_scores(int(a)).copy()
            scores[int(self.board[a])] = -np.inf
            take = min(candidates_per_anchor, self.n - 1)
            candidate_tiles = np.argpartition(scores, -take)[-take:]
            candidate_tiles = candidate_tiles[np.argsort(scores[candidate_tiles])[::-1]]
            for tile in candidate_tiles:
                b = int(inverse[int(tile)])
                pair = tuple(sorted((int(a), b)))
                if pair[0] != pair[1] and pair not in seen:
                    pairs.append(pair)
                    seen.add(pair)
        while len(pairs) < k:
            a = int(self.rng.choice(bad_pool)) if len(pairs) < k * 5 // 6 else int(self.rng.integers(self.n))
            b = int(self.rng.integers(self.n - 1))
            if b >= a:
                b += 1
            pair = tuple(sorted((a, b)))
            if pair not in seen:
                pairs.append(pair)
                seen.add(pair)
        return np.asarray(pairs[:k], dtype=np.int32)

    def swap_stats(self, a, b):
        edges = self.affected_edges(a, b)
        before_visual = self.edge_value(self.board, edges)
        before_correct = self.correct_edges(self.board, edges)
        before_exact = int(self.board[a] == a) + int(self.board[b] == b)
        ta, tb = int(self.board[a]), int(self.board[b])
        before_distance = self.tile_distance(ta, a) + self.tile_distance(tb, b)
        before_pos_a = self.tile_position_score(ta, a)
        before_pos_b = self.tile_position_score(tb, b)
        before_sig_a = self.local_signature(self.board, a)
        before_sig_b = self.local_signature(self.board, b)

        self.board[a], self.board[b] = self.board[b], self.board[a]
        after_visual = self.edge_value(self.board, edges)
        after_correct = self.correct_edges(self.board, edges)
        after_exact = int(self.board[a] == a) + int(self.board[b] == b)
        after_distance = self.tile_distance(tb, a) + self.tile_distance(ta, b)
        after_pos_a = self.tile_position_score(tb, a)
        after_pos_b = self.tile_position_score(ta, b)
        after_sig_a = self.local_signature(self.board, a)
        after_sig_b = self.local_signature(self.board, b)
        self.board[a], self.board[b] = self.board[b], self.board[a]
        return {
            "visual_delta": after_visual - before_visual,
            "correct_edge_delta": after_correct - before_correct,
            "correct_position_delta": after_exact - before_exact,
            "distance_delta": before_distance - after_distance,
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
        feats, stats = [], []
        board_visual = self.current_visual / max(2 * self.n - self.h - self.w, 1)
        board_position = self.current_position / self.n
        for a, b in pairs:
            s = self.swap_stats(int(a), int(b))
            stats.append(s)
            ar, ac = divmod(int(a), self.w)
            br, bc = divmod(int(b), self.w)
            feats.append([
                np.tanh(s["visual_delta"] / 4), np.tanh(s["position_delta"] / 4),
                np.tanh(s["before_visual"] / 8), np.tanh(s["before_visual"] / 8),
                np.tanh(s["after_visual"] / 8), np.tanh(s["after_visual"] / 8),
                *np.tanh(s["before_sig_a"] / 4), *np.tanh(s["before_sig_b"] / 4),
                *np.tanh(s["after_sig_a"] / 4), *np.tanh(s["after_sig_b"] / 4),
                np.tanh(s["before_pos_a"] / 8), np.tanh(s["before_pos_b"] / 8),
                np.tanh(s["after_pos_a"] / 8), np.tanh(s["after_pos_b"] / 8),
                ar / max(self.h - 1, 1), ac / max(self.w - 1, 1),
                br / max(self.h - 1, 1), bc / max(self.w - 1, 1),
                (abs(ar - br) + abs(ac - bc)) / max(self.h + self.w - 2, 1),
                float(ar == br), float(ac == bc), np.tanh(board_visual / 4),
                np.tanh(board_position / 8), self.steps / max(self.n * 4, 1),
                min(self.stagnation / 100, 1.0), 1.0,
            ])
        return np.asarray(feats, dtype=np.float32), stats

    def step(self, pair, stats):
        a, b = map(int, pair)
        self.board[a], self.board[b] = self.board[b], self.board[a]
        self.current_visual += stats["visual_delta"]
        self.current_position += stats["position_delta"]
        reward = training_action_score(stats) - 0.002
        self.steps += 1
        if reward > 0:
            self.stagnation = 0
        else:
            self.stagnation += 1
        done = bool(np.all(self.board == np.arange(self.n)))
        if done:
            reward += 10.0
        return float(reward), done


def crop_episode(cache, index, size, scramble, pos_model, device, rng):
    full = cache.get(index)
    r0 = int(rng.integers(0, GRID - size + 1))
    c0 = int(rng.integers(0, GRID - size + 1))
    tiles = full[r0:r0 + size, c0:c0 + size].copy()
    flat = tiles.reshape(size * size, TILE, TILE, 3)
    row_logp, col_logp = position_log_scores(pos_model, flat, device)
    # PositionPrior predicts absolute coordinates; remap local tile labels to absolute rows/cols.
    row_logp = row_logp[:, np.r_[r0:r0 + size, np.zeros(GRID - size, dtype=int)]]
    col_logp = col_logp[:, np.r_[c0:c0 + size, np.zeros(GRID - size, dtype=int)]]
    return PuzzleSwapEnv(tiles, size, size, row_logp, col_logp, scramble, rng)


def curriculum(epoch):
    stages = [
        (6, 12), (8, 32), (12, 96), (16, 200),
        (20, 350), (24, 576), (24, 576), (24, 576),
        (24, 576), (24, 576), (24, 576), (24, 576),
    ]
    return stages[min(epoch - 1, len(stages) - 1)]


def training_action_score(stats):
    return (
        1.50 * stats["correct_edge_delta"]
        + 0.50 * stats["correct_position_delta"]
        + 0.08 * stats["distance_delta"]
        + 0.12 * np.clip(stats["visual_delta"], -10, 10)
        + 0.04 * np.clip(stats["position_delta"], -10, 10)
    )


def observable_action_score(stats):
    return stats["visual_delta"] + 0.15 * stats["position_delta"]


def warm_start(model, optimizer, cache, train_count, pos_model, device, rng):
    model.train()
    losses, accs = [], []
    pbar = tqdm(range(WARMUP_STEPS), desc="reward-guided warm start")
    env = None
    for step in pbar:
        if env is None or env.stagnation > 80:
            env = crop_episode(cache, int(rng.integers(train_count)), 8, 40, pos_model, device, rng)
        pairs = env.proposal_pairs(PROPOSALS)
        features, stats = env.proposal_features(pairs)
        targets = np.asarray([training_action_score(s) for s in stats])
        target = int(np.argmax(targets))
        x = torch.from_numpy(features).unsqueeze(0).to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits, torch.tensor([target], device=device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        reward, done = env.step(pairs[target], stats[target])
        losses.append(float(loss.item()))
        accs.append(float(logits.argmax(1).item() == target))
        if done:
            env = None
        if (step + 1) % 100 == 0:
            pbar.set_postfix(loss=f"{np.mean(losses[-100:]):.3f}", acc=f"{np.mean(accs[-100:]):.3f}")
    return float(np.mean(losses[-500:])), float(np.mean(accs[-500:]))


def collect_rollout(model, cache, train_count, pos_model, device, rng, size, scramble):
    storage = {k: [] for k in ["features", "actions", "oracle_actions", "logp", "rewards", "values", "dones"]}
    env = crop_episode(cache, int(rng.integers(train_count)), size, scramble, pos_model, device, rng)
    model.eval()
    for _ in range(ROLLOUT_STEPS):
        pairs = env.proposal_pairs(PROPOSALS)
        features, stats = env.proposal_features(pairs)
        x = torch.from_numpy(features).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, value = model(x)
            dist = Categorical(logits=logits)
            action = dist.sample()
        a = int(action.item())
        oracle_action = int(np.argmax([training_action_score(s) for s in stats]))
        reward, done = env.step(pairs[a], stats[a])
        storage["features"].append(features)
        storage["actions"].append(a)
        storage["oracle_actions"].append(oracle_action)
        storage["logp"].append(float(dist.log_prob(action).item()))
        storage["rewards"].append(reward)
        storage["values"].append(float(value.item()))
        storage["dones"].append(float(done))
        if done or env.stagnation > 150:
            env = crop_episode(cache, int(rng.integers(train_count)), size, scramble, pos_model, device, rng)
    with torch.no_grad():
        pairs = env.proposal_pairs(PROPOSALS)
        features, _ = env.proposal_features(pairs)
        _, last_value = model(torch.from_numpy(features).unsqueeze(0).to(device))
    storage["last_value"] = float(last_value.item())
    return storage


def advantages(storage):
    rewards = np.asarray(storage["rewards"], dtype=np.float32)
    values = np.asarray(storage["values"] + [storage["last_value"]], dtype=np.float32)
    dones = np.asarray(storage["dones"], dtype=np.float32)
    adv = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * values[t + 1] * mask - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * mask * gae
        adv[t] = gae
    returns = adv + values[:-1]
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    return adv, returns


def ppo_update(model, optimizer, storage, device):
    x = torch.from_numpy(np.asarray(storage["features"])).to(device)
    actions = torch.tensor(storage["actions"], device=device)
    oracle_actions = torch.tensor(storage["oracle_actions"], device=device)
    old_logp = torch.tensor(storage["logp"], device=device)
    adv, returns = advantages(storage)
    adv = torch.from_numpy(adv).to(device)
    returns = torch.from_numpy(returns).to(device)
    metrics = []
    for _ in range(PPO_EPOCHS):
        for idx in torch.randperm(len(x), device=device).split(MINIBATCH):
            logits, values = model(x[idx])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(actions[idx])
            ratio = torch.exp(logp - old_logp[idx])
            policy_loss = -torch.min(ratio * adv[idx], torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv[idx]).mean()
            value_loss = F.smooth_l1_loss(values, returns[idx])
            entropy = dist.entropy().mean()
            imitation_loss = F.cross_entropy(logits, oracle_actions[idx])
            loss = policy_loss + VALUE_COEF * value_loss + BC_COEF * imitation_loss - ENTROPY_COEF * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            metrics.append((float(policy_loss.item()), float(value_loss.item()), float(entropy.item()), float(imitation_loss.item())))
    return np.mean(metrics, axis=0)


@torch.inference_mode()
def solve_env(model, env, device, max_steps, use_policy=True):
    model.eval()
    best_board = env.board.copy()
    best_score = env.observable_score()
    no_gain = 0
    for _ in range(max_steps):
        pairs = env.proposal_pairs(PROPOSALS)
        features, stats = env.proposal_features(pairs)
        if use_policy:
            logits, _ = model(torch.from_numpy(features).unsqueeze(0).to(device))
            heuristic = torch.tensor([observable_action_score(s) for s in stats], device=device)
            heuristic = (heuristic - heuristic.mean()) / heuristic.std().clamp_min(1e-6)
            action = int((logits[0] + POLICY_BLEND * heuristic).argmax().item())
        else:
            action = int(np.argmax([observable_action_score(s) for s in stats]))
        reward, done = env.step(pairs[action], stats[action])
        score = env.observable_score()
        if score > best_score:
            best_score, best_board, no_gain = score, env.board.copy(), 0
        else:
            no_gain += 1
        if done or no_gain > 300:
            break
    env.board = best_board
    return env.metrics()


def validate(model, cache, val_indices, pos_model, device, epoch, rng):
    rows = []
    preview = None
    for j, image_idx in enumerate(val_indices):
        tiles = cache.get(image_idx).copy()
        flat = tiles.reshape(GRID * GRID, TILE, TILE, 3)
        row_logp, col_logp = position_log_scores(pos_model, flat, device)
        seed = SEED + 10000 + image_idx
        env_rl = PuzzleSwapEnv(tiles, GRID, GRID, row_logp, col_logp, GRID * GRID, np.random.default_rng(seed))
        initial_board = env_rl.board.copy()
        rl_metrics = solve_env(model, env_rl, device, VALIDATE_STEPS, use_policy=True)
        env_base = PuzzleSwapEnv(tiles, GRID, GRID, row_logp, col_logp, 0, np.random.default_rng(seed))
        env_base.board = initial_board.copy()
        env_base.current_visual = env_base.visual_score()
        env_base.current_position = env_base.position_score()
        base_metrics = solve_env(model, env_base, device, VALIDATE_STEPS, use_policy=False)
        target = assemble(flat, np.arange(GRID * GRID), GRID, GRID)
        pred_rl = assemble(flat, env_rl.board, GRID, GRID)
        pred_base = assemble(flat, env_base.board, GRID, GRID)
        rl_ssim = structural_similarity(target, pred_rl, channel_axis=2, data_range=255)
        base_ssim = structural_similarity(target, pred_base, channel_axis=2, data_range=255)
        rows.append((rl_ssim, base_ssim, rl_metrics["adjacency_acc"], base_metrics["adjacency_acc"], rl_metrics["position_acc"]))
        print(
            f"val_image={j} rl_ssim={rl_ssim:.6f} heuristic_ssim={base_ssim:.6f} "
            f"rl_adj={rl_metrics['adjacency_acc']:.6f} heuristic_adj={base_metrics['adjacency_acc']:.6f} "
            f"rl_pos={rl_metrics['position_acc']:.6f}"
        )
        if preview is None:
            preview = np.concatenate([assemble(flat, initial_board, GRID, GRID), pred_base, pred_rl, target], axis=1)
    Image.fromarray(preview).save(OUT_DIR / f"rl_preview_epoch{epoch}.png")
    mean = np.mean(rows, axis=0)
    print(
        f"validation_epoch={epoch} rl_ssim={mean[0]:.6f} heuristic_ssim={mean[1]:.6f} "
        f"rl_adj={mean[2]:.6f} heuristic_adj={mean[3]:.6f} rl_pos={mean[4]:.6f}"
    )
    return mean


def main():
    seed_everything(SEED)
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    data_root = find_data_root()
    files = sorted((data_root / "train" / "targets").glob("*.png"))[:MAX_IMAGES]
    if len(files) < 2:
        raise RuntimeError("Not enough target images")
    order = np.arange(len(files))
    rng.shuffle(order)
    files = [files[i] for i in order]
    n_val = min(VALIDATE_IMAGES, max(1, len(files) // 20))
    train_count = len(files) - n_val
    val_indices = list(range(train_count, len(files)))
    cache = TargetCache(files)

    pos_path = find_latest("position_prior_epoch*.pt")
    pos_model = PositionPrior().to(device)
    pos_model.load_state_dict(torch.load(pos_path, map_location=device)["model"])
    pos_model.eval()
    print(f"device={device} train_images={train_count} val_images={n_val} position_checkpoint={pos_path}")

    model = SwapActorCritic().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    warm_loss, warm_acc = warm_start(model, optimizer, cache, train_count, pos_model, device, rng)
    print(f"warm_start_loss={warm_loss:.6f} warm_start_acc={warm_acc:.6f}")

    best_score = -1.0
    total_updates = max(1, STEPS_PER_EPOCH // ROLLOUT_STEPS)
    for epoch in range(1, EPOCHS + 1):
        size, scramble = curriculum(epoch)
        epoch_rewards, epoch_metrics = [], []
        pbar = tqdm(range(total_updates), desc=f"ppo epoch {epoch}/{EPOCHS} crop={size} scramble={scramble}")
        for _ in pbar:
            storage = collect_rollout(model, cache, train_count, pos_model, device, rng, size, scramble)
            metrics = ppo_update(model, optimizer, storage, device)
            epoch_rewards.extend(storage["rewards"])
            epoch_metrics.append(metrics)
            pbar.set_postfix(reward=f"{np.mean(epoch_rewards[-512:]):.3f}", policy=f"{metrics[0]:.3f}")
        p, v, e, bc = np.mean(epoch_metrics, axis=0)
        print(
            f"rl_epoch={epoch} crop={size} scramble={scramble} reward={np.mean(epoch_rewards):.6f} "
            f"policy_loss={p:.6f} value_loss={v:.6f} entropy={e:.6f} imitation_loss={bc:.6f}"
        )
        val = validate(model, cache, val_indices, pos_model, device, epoch, rng)
        checkpoint = {
            "model": model.state_dict(), "epoch": epoch, "feature_names": FEATURE_NAMES,
            "config": {"grid": GRID, "tile": TILE, "proposals": PROPOSALS},
            "validation": {"rl_ssim": float(val[0]), "heuristic_ssim": float(val[1]), "rl_adj": float(val[2])},
        }
        torch.save(checkpoint, OUT_DIR / f"rl_swap_actor_critic_epoch{epoch}.pt")
        if val[0] > best_score:
            best_score = float(val[0])
            torch.save(checkpoint, OUT_DIR / "rl_swap_actor_critic_best.pt")
            print(f"new_best_epoch={epoch} rl_ssim={best_score:.6f}")
    print(f"done best_validation_rl_ssim={best_score:.6f}")


if __name__ == "__main__":
    main()
