"""P37 RIT-24: raw-RGB relational transformer, FP32 only; no score/DINO/P8 inputs."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

N, GRID = 576, 24


def seed(value=20260817):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def raw_tiles(input_dir: Path, source: str) -> torch.Tensor:
    image = cv2.imread(str(input_dir / source), cv2.IMREAD_COLOR)
    if image is None or image.shape != (480, 480, 3):
        raise RuntimeError("invalid raw input image")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tiles = image.reshape(GRID, 20, GRID, 20, 3).transpose(0, 2, 4, 1, 3).reshape(N, 3, 20, 20)
    return torch.from_numpy(tiles.copy()).float().div_(255.0)


def splits(manifest: Path):
    data = json.loads(manifest.read_text())
    train, held = list(data["train_sources"]), list(data["held_sources"])
    if len(train) != 128 or len(held) != 32 or len(set(train + held)) != 160:
        raise RuntimeError("pinned source manifest mismatch")
    return train[:96], train[96:], held


def labels(label_dir: Path, source: str):
    with np.load(label_dir / (Path(source).stem + ".npz"), allow_pickle=False) as data:
        position = data["target_tile_to_slot"].astype(np.int64)
        cached_source = str(data["source"])
    if cached_source != source or position.shape != (N,) or np.unique(position).size != N:
        raise RuntimeError("invalid cached FIT label")
    inv = np.empty(N, np.int64)
    inv[position] = np.arange(N, dtype=np.int64)
    right = np.full(N, -1, np.int64)
    down = np.full(N, -1, np.int64)
    for tile, slot in enumerate(position):
        row, col = divmod(int(slot), GRID)
        if col + 1 < GRID:
            right[tile] = inv[slot + 1]
        if row + 1 < GRID:
            down[tile] = inv[slot + GRID]
    return torch.from_numpy(right), torch.from_numpy(down)


class RawRelationalTransformer(nn.Module):
    """Raw pixels -> contextual tile set -> directional neighbor matrices. No positional tokens."""
    def __init__(self, dim=384, depth=8, heads=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128, dim, 5), nn.GELU(),
        )
        block = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.05, activation="gelu", batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(block, num_layers=depth, norm=nn.LayerNorm(dim))
        self.rq, self.rk = nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False)
        self.dq, self.dk = nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5

    def forward(self, pixels):
        batch, count = pixels.shape[:2]
        token = self.encoder(pixels.reshape(batch * count, 3, 20, 20)).flatten(1).reshape(batch, count, -1)
        token = self.context(token)
        right = (self.rq(token) @ self.rk(token).transpose(1, 2)) * self.scale
        down = (self.dq(token) @ self.dk(token).transpose(1, 2)) * self.scale
        eye = torch.eye(count, dtype=torch.bool, device=pixels.device).unsqueeze(0)
        right = right.masked_fill(eye, -1e4)
        down = down.masked_fill(eye, -1e4)
        return right, down


def equivariance(model, tiles, device):
    model.eval()
    with torch.no_grad():
        x = tiles.unsqueeze(0).to(device)
        order = torch.randperm(N, device=device)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(N, device=device)
        r0, d0 = model(x)
        r1, d1 = model(x[:, order])
        r1 = r1[:, inv][:, :, inv]
        d1 = d1[:, inv][:, :, inv]
        return float(max((r0 - r1).abs().max().cpu(), (d0 - d1).abs().max().cpu()))


def top20_recall(right_scores, down_scores, truth_right, truth_down):
    recall = []
    for matrix, truth in ((right_scores, truth_right), (down_scores, truth_down)):
        candidates = matrix.topk(20, dim=-1).indices.cpu()
        valid = truth >= 0
        if bool(valid.any()):
            recall.append(float((candidates[valid] == truth[valid, None]).any(1).float().mean()))
    return float(np.mean(recall))


def gate_g0(args):
    seed()
    model = RawRelationalTransformer(dim=64, depth=2, heads=4).eval()
    tiles = torch.rand(N, 3, 20, 20)
    error = equivariance(model, tiles, torch.device("cpu"))
    return {"experiment":"P37_RIT24","gate":"G0","equivariance_error":error,"invalid":0,"targets_opened":False,"p8_imported":False,"passes_G0":bool(error < 1e-5)}


def gate_g1(args):
    seed()
    device = torch.device("cuda")
    model = RawRelationalTransformer().to(device).eval()
    rows = []
    for source in args.sources:
        error = equivariance(model, raw_tiles(args.inputs, source), device)
        rows.append({"source":source,"equivariance_error":error,"finite":bool(np.isfinite(error))})
    maximum = max(row["equivariance_error"] for row in rows)
    invalid = sum(not row["finite"] for row in rows)
    return {"experiment":"P37_RIT24","gate":"G1","sources":len(rows),"max_equivariance_error":maximum,"invalid":invalid,"targets_opened":False,"p8_imported":False,"passes_G1":bool(maximum < 1e-5 and invalid == 0),"rows":rows}


def gate_g2(args):
    seed()
    train, _, _ = splits(args.manifest)
    if len(args.sources) != 96 or set(args.sources) != set(train):
        raise RuntimeError("G2 must use exactly locked 96 FIT-train sources")
    device = torch.device("cuda")
    model = RawRelationalTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)
    data = [(source, raw_tiles(args.inputs, source), *labels(args.labels, source)) for source in args.sources]
    started = time.perf_counter()
    model.train()
    epochs = 4
    for epoch in range(epochs):
        order = list(range(len(data))); random.Random(20260817 + epoch).shuffle(order)
        losses=[]
        for index in order:
            source, pixels, right, down = data[index]
            permutation = torch.randperm(N)
            pixels = pixels[permutation].unsqueeze(0).to(device)
            inverse = torch.empty_like(permutation); inverse[permutation] = torch.arange(N)
            right = right[permutation]; down = down[permutation]
            vr, vd = right >= 0, down >= 0
            right = torch.where(vr, inverse[right.clamp_min(0)], right).to(device)
            down = torch.where(vd, inverse[down.clamp_min(0)], down).to(device)
            optimizer.zero_grad(set_to_none=True)
            sr, sd = model(pixels)
            loss = F.cross_entropy(sr[0, vr], right[vr]) + F.cross_entropy(sd[0, vd], down[vd])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(json.dumps({"stage":"train","epoch":epoch+1,"epochs":epochs,"loss":float(np.mean(losses))}),flush=True)
    model.eval(); rows=[]
    with torch.no_grad():
        for source, pixels, right, down in data:
            sr, sd = model(pixels.unsqueeze(0).to(device))
            value = top20_recall(sr[0], sd[0], right, down)
            rows.append({"source":source,"top20_recall":value,"valid":bool(np.isfinite(value))})
    recall=float(np.mean([row["top20_recall"] for row in rows])); invalid=sum(not row["valid"] for row in rows); seconds=time.perf_counter()-started
    args.model_out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"sources":args.sources},args.model_out)
    return {"experiment":"P37_RIT24","gate":"G2","sources":len(rows),"top20_recall":recall,"invalid":invalid,"seconds":seconds,"targets_opened":False,"p8_imported":False,"selection_opened":False,"held_opened":False,"passes_G2":bool(recall >= .20 and invalid == 0 and seconds < 1500),"rows":rows}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("g0","g1","g2"),required=True)
    parser.add_argument("--inputs",type=Path)
    parser.add_argument("--labels",type=Path)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--sources",nargs="*",default=[])
    parser.add_argument("--model-out",type=Path)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args()
    report={"g0":gate_g0,"g1":gate_g1,"g2":gate_g2}[args.mode](args)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)
    if not report[f"passes_{args.mode.upper()}"]:
        raise RuntimeError(f"P37 {args.mode.upper()} rejected")

if __name__=="__main__": main()
