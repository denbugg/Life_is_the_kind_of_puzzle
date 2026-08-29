"""V24 cross-attention reranker over the calibrated V23 candidate generator."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, "/home/kva/pazzle_boundary_biencoder_v23_xl")
import train_boundary_biencoder_v23 as v23


SMALL_PATH = Path("/home/kva/pazzle_boundary_biencoder_v23/outputs/boundary_biencoder_best.pt")
XL_PATH = Path("/home/kva/pazzle_boundary_biencoder_v23_xl/outputs/boundary_biencoder_best.pt")
DATA_DIR = Path("/home/kva/pazzle_directional_transformer/data/real/restored_target_order")
OUT_DIR = Path(os.getenv("OUT_DIR", "/home/kva/pazzle_boundary_reranker_v24/outputs"))
TOPK = 32
WEIGHTS = (0.25, 0.75, 0.50)
FUSION_ALPHA = 0.18
SEED = 20260909


@dataclass(frozen=True)
class ModelConfig:
    token_dim: int = int(os.getenv("TOKEN_DIM", "160"))
    heads: int = int(os.getenv("HEADS", "5"))
    pair_ff: int = int(os.getenv("PAIR_FF", "640"))
    dropout: float = 0.08
    widths: tuple[int, ...] = (2, 4, 8)


@dataclass(frozen=True)
class TrainConfig:
    steps: int = int(os.getenv("STEPS", "2400"))
    warmup: int = int(os.getenv("WARMUP", "120"))
    lr: float = float(os.getenv("LR", "0.00025"))
    min_lr: float = float(os.getenv("MIN_LR", "0.000006"))
    weight_decay: float = 0.04
    grad_accum: int = int(os.getenv("GRAD_ACCUM", "2"))
    log_every: int = int(os.getenv("LOG_EVERY", "10"))
    validate_every: int = int(os.getenv("VALIDATE_EVERY", "400"))
    validation_boards: int = int(os.getenv("VALIDATION_BOARDS", "8"))
    holdout_boards: int = int(os.getenv("HOLDOUT_BOARDS", "16"))
    hard_weight: float = 0.10
    reverse_weight: float = 0.20
    consistency_weight: float = 0.12
    hard_margin: float = 0.45
    clean_probability: float = 0.20


def log(**payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def row_z(matrix):
    result = matrix.copy()
    np.fill_diagonal(result, np.nan)
    mean = np.nanmean(result, axis=1, keepdims=True)
    std = np.nanstd(result, axis=1, keepdims=True) + 1e-6
    result = (result - mean) / std
    np.fill_diagonal(result, -1e4)
    return result


def load_frozen(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = v23.BoundaryBiEncoder(v23.ModelConfig(**state["model_config"]))
    model.load_state_dict(state["model"], strict=True)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, state


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(16, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(16, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class BoundaryReranker(nn.Module):
    def __init__(self, config):
        super().__init__(); self.config = config
        channels = 12 * len(config.widths)
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 96, 3, padding=1), nn.SiLU(), ResidualBlock(96),
            nn.Conv2d(96, 128, 3, stride=(2, 1), padding=1), nn.SiLU(), ResidualBlock(128),
            nn.Conv2d(128, 160, 3, stride=(2, 1), padding=1), nn.SiLU(), ResidualBlock(160))
        self.project = nn.Linear(160, config.token_dim)
        self.pair_input = nn.Sequential(nn.LayerNorm(4 * config.token_dim),
                                        nn.Linear(4 * config.token_dim, config.token_dim), nn.GELU())
        self.pair_layer = nn.TransformerEncoderLayer(
            config.token_dim, config.heads, config.pair_ff, config.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * config.token_dim), nn.Linear(2 * config.token_dim, config.token_dim),
            nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.token_dim, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def encode_sides(self, x):
        views = v23.BoundaryBiEncoder.robust_views(x)
        sides = []
        for side in ("right", "left", "bottom", "top"):
            scales = []
            for width in self.config.widths:
                if side == "right": strip = views[:, :, :, -width:]
                elif side == "left": strip = views[:, :, :, :width]
                elif side == "bottom": strip = views[:, :, -width:, :].transpose(-2, -1)
                else: strip = views[:, :, :width, :].transpose(-2, -1)
                mean = strip.mean(-1, keepdim=True).expand_as(strip)
                strip = torch.cat((strip, mean), 1)
                scales.append(F.interpolate(strip, size=(32, 12), mode="bilinear", align_corners=False))
            sides.append(torch.cat(scales, 1))
        stacked = torch.stack(sides, 1)
        n = len(x)
        encoded = self.encoder(stacked.reshape(n * 4, -1, 32, 12)).mean(-1).transpose(1, 2)
        tokens = self.project(encoded)
        return tokens.reshape(n, 4, tokens.shape[1], tokens.shape[2])

    def score_edges(self, tokens, source, target, direction, chunk=4096):
        output = []
        for start in range(0, len(source), chunk):
            stop = min(len(source), start + chunk); d = direction[start:stop]
            source_side = torch.where(d == 0, 0, 2)
            target_side = torch.where(d == 0, 1, 3)
            a = tokens[source[start:stop], source_side]
            b = tokens[target[start:stop], target_side]
            h = self.pair_input(torch.cat((a, b, b - a, a * b), -1))
            h = checkpoint(self.pair_layer, h, use_reentrant=False) if self.training else self.pair_layer(h)
            output.append(self.head(torch.cat((h.mean(1), h.amax(1)), -1)).squeeze(-1))
        return torch.cat(output)


@torch.inference_mode()
def candidate_scores(small, xl, tiles):
    score_sets = []
    for model in (small, xl):
        e = model(tiles)
        score_sets.append([
            row_z((e["right"] @ e["left"].t()).float().cpu().numpy()),
            row_z((e["bottom"] @ e["top"].t()).float().cpu().numpy())])
    seam = []
    for source_side, target_side in (("right", "left"), ("bottom", "top")):
        source = small.side_features(tiles, source_side).flatten(1)
        target = small.side_features(tiles, target_side).flatten(1)
        source = F.normalize(source - source.mean(1, keepdim=True), dim=1)
        target = F.normalize(target - target.mean(1, keepdim=True), dim=1)
        seam.append(row_z((source @ target.t()).float().cpu().numpy()))
    return [WEIGHTS[0] * score_sets[0][d] + WEIGHTS[1] * score_sets[1][d] + WEIGHTS[2] * seam[d]
            for d in range(2)]


def candidates_from(matrices, device):
    n = len(matrices[0]); indices=[]; sources=[]; directions=[]
    for direction, matrix in enumerate(matrices):
        candidate = np.argsort(-matrix, axis=1)[:, :TOPK].astype(np.int64)
        indices.append(candidate)
        sources.append(np.repeat(np.arange(n, dtype=np.int64), TOPK))
        directions.append(np.full(n * TOPK, direction, np.int64))
    source = torch.from_numpy(np.concatenate(sources)).to(device)
    target = torch.from_numpy(np.concatenate([x.reshape(-1) for x in indices])).to(device)
    direction = torch.from_numpy(np.concatenate(directions)).to(device)
    return indices, source, target, direction


def neighbour_targets(side):
    grid = np.arange(side * side).reshape(side, side); output=[]
    right = np.full(side * side, -100, np.int64); right[grid[:, :-1].reshape(-1)] = grid[:, 1:].reshape(-1)
    down = np.full(side * side, -100, np.int64); down[grid[:-1].reshape(-1)] = grid[1:].reshape(-1)
    return [right, down]


def fused_logits(matrices, candidates, residual, device):
    n = len(matrices[0]); residual = residual.reshape(2, n, TOPK); output=[]
    for d in range(2):
        values = np.take_along_axis(matrices[d], candidates[d], axis=1)
        base = torch.from_numpy(values).to(device).float()
        output.append(base + FUSION_ALPHA * residual[d].float())
    return output


def ranking_loss(logits, candidates, targets, config):
    row_losses=[]; reverse_losses=[]; hard_losses=[]; coverage=[]
    for d in range(2):
        target = targets[d]; valid = target >= 0
        matches = candidates[d] == target[:, None]; included = valid & matches.any(1)
        coverage.append(float(included.sum() / max(1, valid.sum())))
        rows = np.flatnonzero(included)
        if len(rows) == 0:
            zero = logits[d].sum() * 0; row_losses.append(zero); reverse_losses.append(zero); hard_losses.append(zero)
            continue
        labels = matches[rows].argmax(1).astype(np.int64)
        rows_t = torch.from_numpy(rows).to(logits[d].device)
        labels_t = torch.from_numpy(labels).to(logits[d].device)
        selected = logits[d][rows_t]
        row_losses.append(F.cross_entropy(selected, labels_t))
        positive = selected[torch.arange(len(rows_t), device=selected.device), labels_t]
        negative = selected.clone(); negative[torch.arange(len(rows_t), device=selected.device), labels_t] = -1e4
        hard_losses.append(F.relu(config.hard_margin + negative.max(1).values - positive).mean())
        dense = torch.full((len(target), len(target)), -1e4, device=selected.device)
        dense.scatter_(1, torch.from_numpy(candidates[d]).to(selected.device), logits[d])
        true_target = torch.from_numpy(target[rows]).to(selected.device)
        reverse_losses.append(F.cross_entropy(dense.t()[true_target], rows_t))
    row = torch.stack(row_losses).mean(); reverse = torch.stack(reverse_losses).mean(); hard = torch.stack(hard_losses).mean()
    total = row + config.reverse_weight * reverse + config.hard_weight * hard
    return total, {"row_ce":float(row.detach()), "reverse_ce":float(reverse.detach()),
                   "hard":float(hard.detach()), "candidate_coverage":float(np.mean(coverage))}


def refined_matrices(reranker, small, xl, tiles):
    base = candidate_scores(small, xl, tiles)
    candidates, source, target, direction = candidates_from(base, tiles.device)
    tokens = reranker.encode_sides(tiles)
    residual = reranker.score_edges(tokens, source, target, direction).reshape(2, len(tiles), TOPK)
    refined=[]
    for d in range(2):
        matrix=base[d].copy(); values=np.take_along_axis(matrix,candidates[d],1)
        updated=values + FUSION_ALPHA * residual[d].float().cpu().numpy()
        matrix[np.arange(len(matrix))[:,None],candidates[d]]=updated
        np.fill_diagonal(matrix,-1e4); refined.append(matrix)
    return base, refined


@torch.inference_mode()
def evaluate(reranker, small, xl, scenes, device):
    reranker.eval(); rows=[]
    for scene in scenes:
        board=v23.load_board(DATA_DIR/f"img_{scene:06d}.png")
        tiles=board.reshape(v23.GRID**2,3,v23.TILE,v23.TILE).to(device)
        base,new=refined_matrices(reranker,small,xl,tiles)
        bm=[]; nm=[]
        for d,direction in enumerate(("right","down")):
            bm.append(v23.retrieval(base[d].copy(),v23.GRID,direction))
            nm.append(v23.retrieval(new[d].copy(),v23.GRID,direction))
        row={}
        for key in bm[0]:
            row["base_"+key]=float(np.mean([x[key] for x in bm])); row[key]=float(np.mean([x[key] for x in nm]))
        rows.append(row)
    aggregate={key:float(np.mean([row[key] for row in rows])) for key in rows[0]}
    aggregate["score"]=.40*aggregate["top1"]+.25*aggregate["top5"]+.20*aggregate["mrr"]+.15*aggregate["top32"]
    return {"boards":len(rows),**aggregate,"rows":rows}


def lr_at(step,config):
    if step<=config.warmup:return config.lr*step/config.warmup
    progress=(step-config.warmup)/(config.steps-config.warmup)
    return config.min_lr+.5*(config.lr-config.min_lr)*(1+math.cos(math.pi*progress))


def save(path,model,optimizer,step,best,model_config,train_config):
    temporary=path.with_suffix(".tmp")
    torch.save({"schema":"puzzle-boundary-reranker-v24","step":step,"model":model.state_dict(),
                "optimizer":optimizer.state_dict(),"best_validation_score":best,
                "model_config":asdict(model_config),"train_config":asdict(train_config),
                "topk":TOPK,"weights":WEIGHTS,"fusion_alpha":FUSION_ALPHA},temporary)
    temporary.replace(path)


def main():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    device=torch.device("cuda");torch.backends.cuda.matmul.allow_tf32=True;OUT_DIR.mkdir(parents=True,exist_ok=True)
    small,small_state=load_frozen(SMALL_PATH,device);xl,xl_state=load_frozen(XL_PATH,device)
    model_config=ModelConfig();config=TrainConfig();reranker=BoundaryReranker(model_config).to(device)
    optimizer=torch.optim.AdamW(reranker.parameters(),lr=config.lr,weight_decay=config.weight_decay,betas=(.9,.95))
    scaler=torch.amp.GradScaler("cuda",init_scale=128.0,growth_interval=1000)
    generator=torch.Generator(device=device).manual_seed(SEED+1);rng=np.random.default_rng(SEED+2)
    optimizer.zero_grad(set_to_none=True);best=-1.;accumulated=0.;started=time.perf_counter()
    log(event="start",device=torch.cuda.get_device_name(),parameters=sum(p.numel() for p in reranker.parameters()),
        small_step=small_state["step"],xl_step=xl_state["step"],model_config=asdict(model_config),train_config=asdict(config))
    for step in range(1,config.steps+1):
        side=12 if step<=config.steps//2 else 16;scene=int(rng.integers(6700))
        clean=v23.crop_board(v23.load_board(DATA_DIR/f"img_{scene:06d}.png"),side,rng).to(device)
        is_clean=bool(rng.random()<config.clean_probability);tiles=clean if is_clean else v23.augment_tiles(clean,generator,True)
        matrices=candidate_scores(small,xl,tiles);candidates,source,target,direction=candidates_from(matrices,device)
        targets=neighbour_targets(side);reranker.train()
        with torch.autocast("cuda",dtype=torch.bfloat16):
            noisy_tokens=reranker.encode_sides(tiles)
            residual=reranker.score_edges(noisy_tokens,source,target,direction)
            logits=fused_logits(matrices,candidates,residual,device)
            loss,details=ranking_loss(logits,candidates,targets,config)
            consistency=torch.zeros((),device=device)
            if not is_clean:
                with torch.no_grad():clean_tokens=reranker.encode_sides(clean)
                consistency=(1-F.cosine_similarity(noisy_tokens.mean(-2),clean_tokens.mean(-2),dim=-1)).mean()
                loss=loss+config.consistency_weight*consistency
        scaler.scale(loss/config.grad_accum).backward();accumulated+=float(loss.detach())
        if step%config.grad_accum==0:
            scaler.unscale_(optimizer);grad_norm=float(nn.utils.clip_grad_norm_(reranker.parameters(),1.0));lr=lr_at(step,config)
            for group in optimizer.param_groups:group["lr"]=lr
            scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
        else:grad_norm=0.;lr=optimizer.param_groups[0]["lr"]
        if step==1 or step%config.log_every==0:
            log(event="train",step=step,scene=scene,side=side,clean=is_clean,
                loss=accumulated/(1 if step==1 else config.log_every),consistency=float(consistency.detach()),
                lr=lr,grad_norm=grad_norm,gpu_gb=torch.cuda.max_memory_allocated()/2**30,
                seconds=time.perf_counter()-started,**details);accumulated=0.
        if step%config.validate_every==0:
            result=evaluate(reranker,small,xl,range(6756,6756+config.validation_boards),device);log(event="validation",step=step,**result)
            if result["score"]>best:
                best=result["score"];save(OUT_DIR/"reranker_best.pt",reranker,optimizer,step,best,model_config,config)
            save(OUT_DIR/"reranker_latest.pt",reranker,optimizer,step,best,model_config,config)
    holdout=evaluate(reranker,small,xl,range(6957,6957+config.holdout_boards),device)
    report={"schema":"puzzle-boundary-reranker-v24","model_config":asdict(model_config),
            "train_config":asdict(config),"best_validation_score":best,"holdout":holdout,
            "seconds":time.perf_counter()-started}
    (OUT_DIR/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    save(OUT_DIR/"reranker_final.pt",reranker,optimizer,config.steps,best,model_config,config)
    log(event="complete",report=report)


if __name__=="__main__":main()
