"""Train a swap policy on synthetically shuffled, already assembled 480px images.

This is an offline actor-critic / DAgger trainer: the actor ranks a set of swap
proposals, while the critic predicts the discounted improvement available from
the current proposal set.  Ground-truth tile coordinates provide a dense,
noise-free training reward; pixels are used for the observations exactly as at
inference time.
"""
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

GRID = int(os.getenv("GRID", "24"))
TILE = int(os.getenv("TILE", "20"))
N = GRID * GRID
EPOCHS = int(os.getenv("EPOCHS", "12"))
IMAGES_PER_EPOCH = int(os.getenv("IMAGES_PER_EPOCH", "320"))
ROLLOUT_STEPS = int(os.getenv("ROLLOUT_STEPS", "96"))
PROPOSALS = int(os.getenv("PROPOSALS", "64"))
UPDATE_BATCH = int(os.getenv("UPDATE_BATCH", "2048"))
LR = float(os.getenv("LR", "3e-4"))
SEED = int(os.getenv("SEED", "20260726"))
OUT_DIR = Path(os.getenv("OUT_DIR", "outputs"))
IMAGE_DIR = Path(os.getenv("IMAGE_DIR", "pseudo_targets"))
RESUME = os.getenv("RESUME", "")

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


class SwapActorCritic(nn.Module):
    def __init__(self, feature_dim=len(FEATURE_NAMES), hidden=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(0.05),
            nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        logits = self.actor(x).squeeze(-1)
        pooled = torch.cat([x.mean(-2), x.amax(-2)], -1)
        return logits, self.value(pooled).squeeze(-1)


def split_tiles(path):
    im = Image.open(path).convert("RGB").resize((GRID * TILE, GRID * TILE))
    x = np.asarray(im, dtype=np.uint8)
    return x.reshape(GRID, TILE, GRID, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(N, TILE, TILE, 3)


def visual_matrices(tiles):
    def cost(a, b):
        a = a.reshape(N, -1).astype(np.float32) / 255
        b = b.reshape(N, -1).astype(np.float32) / 255
        z = np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T, 0) / a.shape[1]
        z /= max(float(np.median(z)), 1e-6)
        np.fill_diagonal(z, 12)
        return -z.astype(np.float32)
    return cost(tiles[:, :, -2:], tiles[:, :, :2]), cost(tiles[:, -2:], tiles[:, :2])


class Env:
    def __init__(self, tiles, rng, corruption):
        self.rng, self.steps, self.stagnation = rng, 0, 0
        self.right, self.down = visual_matrices(tiles)
        self.board = np.arange(N, dtype=np.int32)
        # Curriculum: local swaps first, approaching a random permutation.
        for _ in range(corruption):
            a, b = rng.choice(N, 2, replace=False)
            self.board[a], self.board[b] = self.board[b], self.board[a]
        rr = np.arange(N) // GRID
        cc = np.arange(N) % GRID
        # A noisy proxy for the separately trained position prior.
        self.row_logp = -0.65 * np.abs(rr[:, None] - np.arange(GRID)[None]) + rng.normal(0, .35, (N, GRID))
        self.col_logp = -0.65 * np.abs(cc[:, None] - np.arange(GRID)[None]) + rng.normal(0, .35, (N, GRID))
        self.current_visual = self.visual_score()
        self.current_position = self.position_score()

    def neighbors(self, p):
        r, c = divmod(int(p), GRID)
        if c: yield p - 1, p, 0
        if c + 1 < GRID: yield p, p + 1, 0
        if r: yield p - GRID, p, 1
        if r + 1 < GRID: yield p, p + GRID, 1

    def edge_value(self, board, edges):
        return sum((self.right if d == 0 else self.down)[board[p], board[q]] for p, q, d in edges)

    def pos(self, tile, p):
        r, c = divmod(int(p), GRID)
        return float(self.row_logp[tile, r] + self.col_logp[tile, c])

    def visual_score(self):
        b = self.board.reshape(GRID, GRID)
        return float(self.right[b[:, :-1], b[:, 1:]].sum() + self.down[b[:-1], b[1:]].sum())

    def position_score(self):
        return sum(self.pos(int(t), p) for p, t in enumerate(self.board))

    def signature(self, board, p):
        r, c, t = p // GRID, p % GRID, int(board[p])
        return np.asarray([
            self.right[int(board[p-1]), t] if c else -4,
            self.right[t, int(board[p+1])] if c + 1 < GRID else -4,
            self.down[int(board[p-GRID]), t] if r else -4,
            self.down[t, int(board[p+GRID])] if r + 1 < GRID else -4,
        ], np.float32)

    def stats(self, a, b):
        ea, eb = set(self.neighbors(a)), set(self.neighbors(b))
        union = ea | eb
        av, bv, uv = self.edge_value(self.board, ea), self.edge_value(self.board, eb), self.edge_value(self.board, union)
        ta, tb = int(self.board[a]), int(self.board[b])
        pa, pb, sa, sb = self.pos(ta, a), self.pos(tb, b), self.signature(self.board, a), self.signature(self.board, b)
        self.board[a], self.board[b] = self.board[b], self.board[a]
        a2, b2, u2 = self.edge_value(self.board, ea), self.edge_value(self.board, eb), self.edge_value(self.board, union)
        qa, qb, s2a, s2b = self.pos(tb, a), self.pos(ta, b), self.signature(self.board, a), self.signature(self.board, b)
        self.board[a], self.board[b] = self.board[b], self.board[a]
        exact_delta = int(tb == a) + int(ta == b) - int(ta == a) - int(tb == b)
        man_before = abs(ta//GRID-a//GRID)+abs(ta%GRID-a%GRID)+abs(tb//GRID-b//GRID)+abs(tb%GRID-b%GRID)
        man_after = abs(tb//GRID-a//GRID)+abs(tb%GRID-a%GRID)+abs(ta//GRID-b//GRID)+abs(ta%GRID-b%GRID)
        reward = 2.0 * exact_delta + 0.08 * (man_before - man_after) + 0.03 * (u2 - uv)
        return (u2-uv, qa+qb-pa-pb, av, bv, a2, b2, sa, sb, s2a, s2b, pa, pb, qa, qb, reward)

    def proposals(self):
        wrong = np.flatnonzero(self.board != np.arange(N))
        pairs = set()
        # Oracle-like hard candidates are mixed with exploration, but labels decide the action.
        for a in self.rng.choice(wrong if len(wrong) else N, min(16, max(len(wrong), 1)), replace=True):
            pairs.add(tuple(sorted((int(a), int(np.where(self.board == a)[0][0])))))
        while len(pairs) < PROPOSALS:
            a, b = self.rng.choice(N, 2, replace=False)
            pairs.add(tuple(sorted((int(a), int(b)))))
        return np.asarray(list(pairs)[:PROPOSALS], np.int32)

    def features(self, pairs):
        out, stats = [], []
        for a, b in pairs:
            s = self.stats(int(a), int(b)); stats.append(s)
            vd, pd, av, bv, a2, b2, sa, sb, s2a, s2b, pa, pb, qa, qb, _ = s
            ar, ac, br, bc = a//GRID, a%GRID, b//GRID, b%GRID
            out.append([
                np.tanh(vd/4), np.tanh(pd/4), np.tanh(av/8), np.tanh(bv/8),
                np.tanh(a2/8), np.tanh(b2/8), *np.tanh(sa/4), *np.tanh(sb/4),
                *np.tanh(s2a/4), *np.tanh(s2b/4), np.tanh(pa/8), np.tanh(pb/8),
                np.tanh(qa/8), np.tanh(qb/8), ar/23, ac/23, br/23, bc/23,
                (abs(ar-br)+abs(ac-bc))/46, float(ar==br), float(ac==bc),
                np.tanh(self.current_visual/(2*N-2*GRID)/4), np.tanh(self.current_position/N/8),
                self.steps/(N*4), min(self.stagnation/100, 1), 1.,
            ])
        return np.asarray(out, np.float32), stats

    def apply(self, pair, stat):
        a, b = map(int, pair)
        self.board[a], self.board[b] = self.board[b], self.board[a]
        self.current_visual += stat[0]; self.current_position += stat[1]; self.steps += 1


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(IMAGE_DIR.glob("*.png"))
    if not files: raise FileNotFoundError(f"No PNGs in {IMAGE_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = SwapActorCritic().to(device)
    start = 1
    if RESUME:
        ck = torch.load(RESUME, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); start = int(ck.get("epoch", 0)) + 1
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(SEED)
    history = []
    for epoch in range(start, EPOCHS + 1):
        model.train(); buf_x, buf_y, buf_v = [], [], []
        losses, accs, rewards = [], [], []
        corruption = int(24 + (N * .9 - 24) * min(epoch / max(EPOCHS-2, 1), 1))
        chosen = rng.choice(files, min(IMAGES_PER_EPOCH, len(files)), replace=len(files)<IMAGES_PER_EPOCH)
        for image_i, path in enumerate(chosen):
            env = Env(split_tiles(path), rng, corruption)
            for _ in range(ROLLOUT_STEPS):
                pairs = env.proposals(); x, stats = env.features(pairs)
                rs = np.asarray([s[-1] for s in stats], np.float32)
                target = int(rs.argmax())
                buf_x.append(x); buf_y.append(target); buf_v.append(float(rs[target]))
                # DAgger: 85% oracle, 15% current policy.
                if rng.random() < .85:
                    action = target
                else:
                    with torch.inference_mode():
                        action = int(model(torch.from_numpy(x)[None].to(device))[0].argmax())
                env.apply(pairs[action], stats[action]); rewards.append(float(rs[action]))
                env.stagnation = env.stagnation + 1 if rs[action] <= 0 else 0
                if len(buf_x) >= UPDATE_BATCH:
                    xx = torch.from_numpy(np.stack(buf_x)).to(device)
                    yy = torch.tensor(buf_y, device=device)
                    vv = torch.tensor(buf_v, device=device)
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=device.type=="cuda"):
                        logits, value = model(xx)
                        actor = F.cross_entropy(logits, yy, label_smoothing=.02)
                        critic = F.smooth_l1_loss(value, vv)
                        # Encourage a margin between the oracle and all other proposals.
                        best = logits.gather(1, yy[:, None])
                        rank = F.relu(.25 - best + logits).mean()
                        loss = actor + .25*critic + .15*rank
                    scaler.scale(loss).backward(); scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    scaler.step(opt); scaler.update()
                    losses.append(float(loss.detach()))
                    accs.append(float((logits.argmax(1) == yy).float().mean().detach()))
                    buf_x.clear(); buf_y.clear(); buf_v.clear()
        row = {"epoch":epoch, "loss":float(np.mean(losses)), "policy_acc":float(np.mean(accs)),
               "mean_reward":float(np.mean(rewards)), "corruption_swaps":corruption}
        history.append(row); print(json.dumps(row), flush=True)
        torch.save({"model":model.state_dict(), "epoch":epoch, "feature_names":FEATURE_NAMES,
                    "schema_version":2, "config":vars_config, "history":history},
                   OUT_DIR/f"rl_swap_actor_critic_v2_epoch{epoch}.pt")
        (OUT_DIR/"metrics.json").write_text(json.dumps(history, indent=2))


vars_config = {"grid":GRID, "tile":TILE, "proposals":PROPOSALS, "rollout_steps":ROLLOUT_STEPS,
               "images_per_epoch":IMAGES_PER_EPOCH, "seed":SEED}
if __name__ == "__main__":
    main()
