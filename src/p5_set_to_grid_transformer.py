"""P5: FIT-only set-to-grid Transformer with Hungarian assignment.

The input bag is a shuffled set of independently corrupted upright tiles.  The
model intentionally has no input tile-order positional embedding.  G0 validates
permutation equivariance and bijective Hungarian decoding.  G1 trains a set
Transformer and an equally sized independent-CNN comparator using only pinned
FIT source targets; it cannot load CAL/DEV/test files or assemble boards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F

from train_eval_cb1_g1_capacity import distort_frags, load_rgb, sha256_file, to_frags

GRID = 24
N = GRID * GRID
FIT_TARGETS = Path(r"E:\pazzle_data\train\targets")
SPLIT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json")
WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P5_set_to_grid\g0_g1_capacity")


@dataclass(frozen=True)
class ModelSpec:
    width: int = 192
    blocks: int = 6
    heads: int = 8


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("g0", "g1"))
    p.add_argument("--targets", type=Path, default=FIT_TARGETS)
    p.add_argument("--split", type=Path, default=SPLIT)
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260817)
    p.add_argument("--train-sources", type=int, default=256)
    p.add_argument("--eval-sources", type=int, default=32)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--width", type=int, default=192)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    return p.parse_args()


class TileStem(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        c1, c2 = width // 3, width // 2
        self.net = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1, bias=False), nn.GroupNorm(max(1, c1 // 16), c1), nn.GELU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False), nn.GroupNorm(max(1, c2 // 16), c2), nn.GELU(),
            nn.Conv2d(c2, width, 3, stride=2, padding=1, bias=False), nn.GroupNorm(max(1, width // 16), width), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.LayerNorm(width),
        )

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = tiles.shape
        return self.net(tiles.reshape(b * n, c, h, w)).reshape(b, n, -1)


class SetBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width); self.attn = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(width); self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x); x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.ff(self.norm2(x))


class SetToGrid(nn.Module):
    def __init__(self, spec: ModelSpec, use_set_attention: bool) -> None:
        super().__init__()
        self.stem = TileStem(spec.width)
        self.use_set_attention = use_set_attention
        self.blocks = nn.ModuleList([SetBlock(spec.width, spec.heads) for _ in range(spec.blocks)]) if use_set_attention else nn.ModuleList()
        self.independent_blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(spec.width), nn.Linear(spec.width, 4 * spec.width), nn.GELU(), nn.Linear(4 * spec.width, spec.width)) for _ in range(spec.blocks)]) if not use_set_attention else nn.ModuleList()
        self.slot_queries = nn.Parameter(torch.randn(N, spec.width) * 0.02)
        self.slot_norm = nn.LayerNorm(spec.width)
        self.cross = nn.MultiheadAttention(spec.width, spec.heads, batch_first=True, dropout=0.0) if use_set_attention else None
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        tokens = self.stem(tiles)
        for block in self.blocks: tokens = block(tokens)
        for block in self.independent_blocks: tokens = tokens + block(tokens)
        b = tokens.shape[0]; slots = self.slot_queries.unsqueeze(0).expand(b, -1, -1)
        if self.cross is not None:
            slots = slots + self.cross(self.slot_norm(slots), tokens, tokens, need_weights=False)[0]
        scores = torch.einsum("bnd,bmd->bnm", F.normalize(tokens, dim=-1), F.normalize(slots, dim=-1))
        return scores * self.logit_scale.exp().clamp(max=20.0)


def source_sets(cfg: argparse.Namespace) -> tuple[list[str], list[str]]:
    payload = json.loads(cfg.split.read_text(encoding="utf-8")); fit = list(payload["splits"]["fit"]); cal = set(payload["splits"]["cal"]); dev = set(payload["splits"]["dev"])
    if len(fit) != 5360 or cfg.train_sources != 256 or cfg.eval_sources != 32 or set(fit) & (cal | dev): raise RuntimeError("P5 split contract violated")
    train, heldout = fit[:256], fit[256:288]
    if set(train) & set(heldout) or any(name in cal or name in dev for name in train + heldout): raise RuntimeError("P5 non-FIT source")
    for name in train + heldout:
        if not (cfg.targets / name).is_file(): raise FileNotFoundError(cfg.targets / name)
    return train, heldout


def make_bag(targets: Path, name: str, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    clean = load_rgb(targets / name); fragments = distort_frags(to_frags(clean), np.random.default_rng(seed * 1009 + int(name[4:10])))
    permutation = np.random.default_rng(seed * 2029 + int(name[4:10])).permutation(N).astype(np.int64)
    bag = fragments[permutation]
    tiles = torch.from_numpy(bag).permute(0, 3, 1, 2).contiguous().float().div_(255.0).unsqueeze(0).to(device)
    labels = torch.from_numpy(permutation).long().unsqueeze(0).to(device)
    return tiles, labels, permutation


def hungarian_board(scores: torch.Tensor) -> np.ndarray:
    matrix = scores.detach().float().cpu().numpy()
    row, col = linear_sum_assignment(-matrix)
    board = np.empty(N, dtype=np.int16); board[col] = row
    if np.unique(board).size != N or board.min() != 0 or board.max() != N - 1: raise RuntimeError("nonbijective Hungarian board")
    return board


def slot_accuracy(scores: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    independent = float((scores.argmax(dim=-1) == labels).float().mean().item())
    boards = [hungarian_board(scores[b]) for b in range(scores.shape[0])]
    # labels[tile] is its source slot; board[slot] is tile.
    hungarian = float(np.mean([np.mean(labels[b].detach().cpu().numpy()[board] == np.arange(N)) for b, board in enumerate(boards)]))
    return independent, hungarian


def config_hash(spec: ModelSpec, use_set: bool) -> str:
    value = json.dumps({"spec":spec.__dict__,"use_set_attention":use_set}, sort_keys=True).encode(); return hashlib.sha256(value).hexdigest()


def g0(cfg: argparse.Namespace) -> None:
    if cfg.device != "cuda" or not torch.cuda.is_available(): raise RuntimeError("P5 requires local CUDA")
    train, _ = source_sets(cfg); device = torch.device("cuda"); spec = ModelSpec(cfg.width,cfg.blocks,cfg.heads)
    torch.manual_seed(cfg.seed); model = SetToGrid(spec, True).to(device).eval(); cfg.work.mkdir(parents=True,exist_ok=True)
    checks = []
    with torch.no_grad():
        for index, name in enumerate(train[:4]):
            tiles, labels, _ = make_bag(cfg.targets,name,cfg.seed + index,device); scores = model(tiles)[0]; board = hungarian_board(scores)
            perm = torch.from_numpy(np.random.default_rng(cfg.seed + 9000 + index).permutation(N)).long().to(device)
            reordered = tiles[:,perm]; reordered_labels = labels[:,perm]; score2 = model(reordered)[0]
            # score2 row k must equal original score at the tile index perm[k].
            max_abs = float((score2 - scores[perm]).abs().max().item())
            labels_ok = bool(torch.equal(reordered_labels[0].cpu(), labels[0,perm].cpu()))
            checks.append({"source":name,"max_equivariance_abs":max_abs,"labels_permute":labels_ok,"hungarian_bijection":bool(np.unique(board).size == N),"board_sha256":hashlib.sha256(board.tobytes()).hexdigest()})
    passed = all(c["max_equivariance_abs"] < 1e-5 and c["labels_permute"] and c["hungarian_bijection"] for c in checks)
    report = {"experiment":"P5_set_to_grid_transformer","gate":"G0_equivariance_label_contract","passes":passed,"decision":"pass_to_G1_capacity" if passed else "reject_P5_before_training","checks":checks,"model_config":{"width":cfg.width,"blocks":cfg.blocks,"heads":cfg.heads,"config_sha256":config_hash(spec,True)},"split_sha256":sha256_file(cfg.split),"CAL_target_opened":False,"DEV_targets_opened":False,"test_accessed":False,"layouts_assembled":False,"restorer_used":False}
    (cfg.work / "p5_g0_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2),flush=True)


def train_one(cfg: argparse.Namespace, use_set: bool, train: list[str], heldout: list[str], device: torch.device) -> dict[str, object]:
    spec = ModelSpec(cfg.width,cfg.blocks,cfg.heads); torch.manual_seed(cfg.seed + (0 if use_set else 1)); model = SetToGrid(spec,use_set).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); rng = np.random.default_rng(cfg.seed + (11 if use_set else 13)); losses=[]
    model.train()
    for step in range(cfg.steps):
        name=train[int(rng.integers(len(train)))]; tiles,labels,_=make_bag(cfg.targets,name,cfg.seed*100000 + step,device); scores=model(tiles)
        loss=F.cross_entropy(scores.reshape(-1,N),labels.reshape(-1)); optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
        if (step+1)%250==0: print(f"model={'set' if use_set else 'independent'} step={step+1} loss={np.mean(losses[-100:]):.6f}",flush=True)
    model.eval(); independent=[]; hungarian=[]
    with torch.no_grad():
        for index,name in enumerate(heldout):
            tiles,labels,_=make_bag(cfg.targets,name,cfg.seed+777+index,device); a,b=slot_accuracy(model(tiles),labels); independent.append(a); hungarian.append(b)
    ckpt=cfg.work/("p5_g1_set_transformer.pt" if use_set else "p5_g1_independent_cnn.pt"); torch.save({"state_dict":model.state_dict(),"spec":spec.__dict__,"use_set_attention":use_set,"seed":cfg.seed,"steps":cfg.steps},ckpt)
    return {"use_set_attention":use_set,"loss_first_100":float(np.mean(losses[:100])),"loss_last_100":float(np.mean(losses[-100:])),"heldout_independent_slot_accuracy":float(np.mean(independent)),"heldout_hungarian_slot_accuracy":float(np.mean(hungarian)),"checkpoint":str(ckpt),"checkpoint_sha256":sha256_file(ckpt),"model_config_sha256":config_hash(spec,use_set)}


def g1(cfg: argparse.Namespace) -> None:
    if cfg.device != "cuda" or not torch.cuda.is_available(): raise RuntimeError("P5 requires local CUDA")
    if cfg.steps != 4000 or (cfg.width,cfg.blocks,cfg.heads)!=(192,6,8): raise ValueError("P5 G1 fixed model/budget contract violated")
    train, heldout=source_sets(cfg); device=torch.device("cuda"); cfg.work.mkdir(parents=True,exist_ok=True)
    set_result=train_one(cfg,True,train,heldout,device); independent_result=train_one(cfg,False,train,heldout,device)
    set_acc=float(set_result["heldout_hungarian_slot_accuracy"]); base_acc=float(independent_result["heldout_hungarian_slot_accuracy"])
    passed=bool(set_acc>0.10 and set_acc>=base_acc+0.05 and float(set_result["loss_last_100"])<float(set_result["loss_first_100"]))
    report={"experiment":"P5_set_to_grid_transformer","gate":"G1_FIT_capacity","set_transformer":set_result,"independent_cnn":independent_result,"heldout_hungarian_delta_pp":100*(set_acc-base_acc),"pass_criteria":"set Hungarian >10.0%, delta >=+5.0pp vs independent CNN, loss decreases","passes_G1":passed,"decision":"pass_to_full_FIT_scale" if passed else "reject_P5_before_scale_CAL","train_sources":train,"heldout_sources":heldout,"split_sha256":sha256_file(cfg.split),"CAL_target_opened":False,"DEV_targets_opened":False,"test_accessed":False,"layouts_assembled":False,"restorer_used":False}
    (cfg.work/"p5_g1_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2),flush=True)


def main() -> None:
    cfg=args(); random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
    if cfg.phase=="g0": g0(cfg)
    else: g1(cfg)


if __name__=="__main__": main()
