"""Train a colour-robust directional transformer for 24x24 jigsaw tiles.

The model embeds the left/right/up/down sides of every tile. Training uses
in-image negatives and independent corruption per tile, so colour agreement
cannot become a shortcut. Validation reports neighbour retrieval recall.
"""
import json
import math
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

GRID = int(os.getenv("GRID", "24"))
TILE = int(os.getenv("TILE", "20"))
N = GRID * GRID
IMAGE_DIR = Path(os.getenv("IMAGE_DIR", "pseudo_targets"))
OUT_DIR = Path(os.getenv("OUT_DIR", "directional_transformer_outputs"))
EPOCHS = int(os.getenv("EPOCHS", "18"))
STEPS_PER_EPOCH = int(os.getenv("STEPS_PER_EPOCH", "500"))
VAL_STEPS = int(os.getenv("VAL_STEPS", "80"))
TILES_PER_IMAGE = int(os.getenv("TILES_PER_IMAGE", "144"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "3"))
DIM = int(os.getenv("DIM", "128"))
LR = float(os.getenv("LR", "3e-4"))
TAU = float(os.getenv("TAU", "0.10"))
SEED = int(os.getenv("SEED", "20260814"))


def split_tiles(path):
    x = np.asarray(Image.open(path).convert("RGB").resize((GRID*TILE, GRID*TILE)), np.uint8)
    return x.reshape(GRID,TILE,GRID,TILE,3).transpose(0,2,1,3,4).reshape(N,TILE,TILE,3)


def corrupt(tile, rng):
    """Independent strong colour corruption, plus mild spatial damage."""
    x = tile.astype(np.float32) / 255.0
    scale = rng.uniform(.62, 1.42, (1,1,3)); bias = rng.uniform(-.20, .20, (1,1,3))
    mix = np.eye(3, dtype=np.float32) + rng.normal(0, .055, (3,3)).astype(np.float32)
    x = np.einsum("hwc,dc->hwd", x, mix) * scale + bias
    gamma = rng.uniform(.72, 1.38); x = np.clip(x, 0, 1) ** gamma
    if rng.random() < .35:
        c = int(rng.integers(3)); x[...,c] = x[...,c].mean()
    if rng.random() < .35:
        h,w = x.shape[:2]; y0=int(rng.integers(h)); x0=int(rng.integers(w))
        y1=min(h,y0+int(rng.integers(3,9))); x1=min(w,x0+int(rng.integers(3,9)))
        alpha=rng.uniform(.15,.55); x[y0:y1,x0:x1]=(1-alpha)*x[y0:y1,x0:x1]+alpha*rng.random(3)
    x += rng.normal(0, rng.uniform(0,.045), x.shape).astype(np.float32)
    if rng.random() < .20:
        x=np.asarray(Image.fromarray((np.clip(x,0,1)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(.25,.8))),np.float32)/255
    return np.clip(x,0,1)


def structural_channels(x):
    rgb=torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).float()
    gray=.299*rgb[0]+.587*rgb[1]+.114*rgb[2]
    norm=(gray-gray.mean())/(gray.std()+1e-4)
    sx=F.pad(gray[:,2:]-gray[:,:-2],(1,1,0,0))
    sy=F.pad(gray[2:]-gray[:-2],(0,0,1,1))
    return torch.cat([rgb, norm[None].clamp(-4,4)/4, sx[None], sy[None]],0)


class PuzzleSamples(Dataset):
    def __init__(self, files, steps, seed, train): self.files,self.steps,self.seed,self.train=files,steps,seed,train
    def __len__(self): return self.steps
    def __getitem__(self, idx):
        rng=np.random.default_rng(None if self.train else self.seed+idx)
        path=self.files[int(rng.integers(len(self.files)))]
        tiles=split_tiles(path)
        # Sample anchors first and force all available right/down neighbours into the candidate set.
        anchors=rng.choice(N, min(TILES_PER_IMAGE//3,N), replace=False)
        ids=set(map(int,anchors))
        for i in anchors:
            r,c=divmod(int(i),GRID)
            if c+1<GRID: ids.add(int(i)+1)
            if r+1<GRID: ids.add(int(i)+GRID)
            if c: ids.add(int(i)-1)
            if r: ids.add(int(i)-GRID)
        remaining=np.setdiff1d(np.arange(N),np.fromiter(ids,np.int64),assume_unique=False)
        need=max(0,TILES_PER_IMAGE-len(ids))
        if need: ids.update(map(int,rng.choice(remaining,min(need,len(remaining)),replace=False)))
        ids=np.asarray(sorted(ids),np.int64)[:TILES_PER_IMAGE]
        # Each tile receives its own transform.
        xx=torch.stack([structural_channels(corrupt(tiles[i],rng)) for i in ids])
        return xx,torch.from_numpy(ids)


class ResBlock(nn.Module):
    def __init__(self,c):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GroupNorm(8,c),nn.SiLU(),nn.Conv2d(c,c,3,padding=1),nn.GroupNorm(8,c))
    def forward(self,x): return F.silu(x+self.net(x))


class DirectionalTransformer(nn.Module):
    def __init__(self,dim=DIM):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv2d(6,64,3,padding=1),nn.GroupNorm(8,64),nn.SiLU(),ResBlock(64),nn.Conv2d(64,96,3,stride=2,padding=1),nn.GroupNorm(8,96),nn.SiLU(),ResBlock(96))
        self.proj=nn.Linear(96,dim); self.kind=nn.Parameter(torch.randn(5,dim)*.02)
        layer=nn.TransformerEncoderLayer(dim,4,dim*3,.08,"gelu",batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,3,norm=nn.LayerNorm(dim))
        self.border=nn.Linear(dim,1)
    def forward(self,x):
        n=x.shape[0]; f=self.cnn(x) # N,C,10,10
        tokens=torch.stack([f[:,:,:,0:3].mean((2,3)),f[:,:,:,-3:].mean((2,3)),f[:,:,0:3,:].mean((2,3)),f[:,:,-3:,:].mean((2,3)),f.mean((2,3))],1)
        z=self.transformer(self.proj(tokens)+self.kind[None])
        sides=F.normalize(z[:,:4]+.25*z[:,4,None],dim=-1)
        return sides,self.border(z[:,:4]).squeeze(-1)


def neighbour_targets(ids, direction):
    # directions L,R,U,D; return candidate index or -1.
    delta=(-1,1,-GRID,GRID)[direction]
    r=ids//GRID; c=ids%GRID
    valid=(c>0,c+1<GRID,r>0,r+1<GRID)[direction]
    target_id=ids+delta
    eq=target_id[:,None].eq(ids[None,:]) & valid[:,None]
    found=eq.any(1); target=eq.float().argmax(1).long(); target[~found]=-1
    return target,~valid


def batch_loss(model,x,ids):
    losses=[]; recalls=[]; border_losses=[]; border_acc=[]
    opposite=(1,0,3,2)
    for xb,ib in zip(x,ids):
        sides,border=model(xb)
        for d in range(4):
            target,is_border=neighbour_targets(ib,d)
            logits=sides[:,d]@sides[:,opposite[d]].T/TAU
            logits.fill_diagonal_(-1e4)
            keep=target.ge(0)
            if keep.any():
                losses.append(F.cross_entropy(logits[keep],target[keep]))
                rank=logits[keep].topk(min(5,len(ib)),1).indices
                recalls.append(torch.stack([(rank[:,0]==target[keep]).float().mean(),(rank==target[keep,None]).any(1).float().mean()]))
            border_losses.append(F.binary_cross_entropy_with_logits(border[:,d],is_border.float()))
            border_acc.append(((border[:,d]>0)==is_border).float().mean())
    return torch.stack(losses).mean()+.25*torch.stack(border_losses).mean(),torch.stack(recalls).mean(0),torch.stack(border_acc).mean()


@torch.inference_mode()
def validate(model,loader,device):
    model.eval(); ls=[]; rs=[]; bs=[]
    for x,ids in loader:
        loss,r,b=batch_loss(model,x.to(device),ids.to(device)); ls.append(float(loss)); rs.append(r.cpu()); bs.append(float(b))
    r=torch.stack(rs).mean(0)
    return {"val_loss":float(np.mean(ls)),"recall_at_1":float(r[0]),"recall_at_5":float(r[1]),"border_acc":float(np.mean(bs))}


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    files=sorted(IMAGE_DIR.glob("*.png")); rng=random.Random(SEED); rng.shuffle(files)
    nval=max(20,len(files)//10); train,val=files[:-nval],files[-nval:]
    if not train: raise RuntimeError(f"No training images in {IMAGE_DIR}")
    tr=DataLoader(PuzzleSamples(train,STEPS_PER_EPOCH*BATCH_SIZE,SEED,True),BATCH_SIZE,num_workers=4,pin_memory=True,drop_last=True)
    va=DataLoader(PuzzleSamples(val,VAL_STEPS*BATCH_SIZE,SEED+999,False),BATCH_SIZE,num_workers=2,pin_memory=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=DirectionalTransformer().to(device)
    opt=torch.optim.AdamW(model.parameters(),LR,weight_decay=2e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=LR*.05)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda"); OUT_DIR.mkdir(parents=True,exist_ok=True)
    history=[]; best=-1
    print(json.dumps({"device":str(device),"gpu":torch.cuda.get_device_name(0) if device.type=="cuda" else None,"train_images":len(train),"val_images":len(val),"params":sum(p.numel() for p in model.parameters())}),flush=True)
    for epoch in range(1,EPOCHS+1):
        model.train(); losses=[]; r1=[]
        for x,ids in tr:
            x,ids=x.to(device,non_blocking=True),ids.to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=device.type=="cuda",dtype=torch.bfloat16): loss,r,_=batch_loss(model,x,ids)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),1); scaler.step(opt); scaler.update()
            losses.append(float(loss.detach())); r1.append(float(r[0]))
        sched.step(); m=validate(model,va,device); m.update(epoch=epoch,train_loss=float(np.mean(losses)),train_recall_at_1=float(np.mean(r1)),lr=opt.param_groups[0]["lr"]); history.append(m)
        print(json.dumps(m),flush=True); payload={"model":model.state_dict(),"config":{"grid":GRID,"tile":TILE,"dim":DIM},"epoch":epoch,"metrics":m,"schema_version":1}
        torch.save(payload,OUT_DIR/f"epoch_{epoch}.pt")
        if m["recall_at_1"]>best: best=m["recall_at_1"]; torch.save(payload,OUT_DIR/"best.pt")
        (OUT_DIR/"metrics.json").write_text(json.dumps(history,indent=2))


if __name__=="__main__": main()
