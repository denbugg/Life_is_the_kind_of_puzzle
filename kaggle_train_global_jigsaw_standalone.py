"""Permutation-equivariant global jigsaw transformer for all 576 tiles."""
import json
import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset


def find_full_targets():
    candidates = []
    for path in Path("/kaggle/input").rglob("train/targets"):
        if sum(1 for _ in path.glob("*.png")) >= 1000:
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one full train/targets directory, got {candidates}")
    return candidates[0]


if Path("/kaggle/input").exists():
    os.environ.setdefault("IMAGE_DIR", str(find_full_targets()))
    os.environ.setdefault("OUT_DIR", "/kaggle/working")
    os.environ.setdefault("EPOCHS", "20")
    os.environ.setdefault("SAMPLES_PER_EPOCH", "7000")
    os.environ.setdefault("VAL_SAMPLES", "512")
    os.environ.setdefault("VAL_SOURCES", "512")
    os.environ.setdefault("BATCH_SIZE", "4")
    os.environ.setdefault("DIM", "192")
    os.environ.setdefault("LAYERS", "6")
    os.environ.setdefault("LR", "2e-4")


GRID, TILE, N = 24, 20, 576
IMAGE_DIR = Path(os.getenv("IMAGE_DIR", "restored_rl_targets"))
OUT_DIR = Path(os.getenv("OUT_DIR", "global_jigsaw_outputs"))
EPOCHS = int(os.getenv("EPOCHS", "16"))
SAMPLES_PER_EPOCH = int(os.getenv("SAMPLES_PER_EPOCH", "1600"))
VAL_SAMPLES = int(os.getenv("VAL_SAMPLES", "64"))
VAL_SOURCES = int(os.getenv("VAL_SOURCES", "512"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
LR = float(os.getenv("LR", "2e-4"))
DIM = int(os.getenv("DIM", "192"))
LAYERS = int(os.getenv("LAYERS", "6"))
SEED = int(os.getenv("SEED", "27072031"))


def source_stem(path):
    return path.stem.rsplit("_v", 1)[0]


def split_tiles(image):
    return image.reshape(GRID,TILE,GRID,TILE,3).transpose(0,2,1,3,4).reshape(N,TILE,TILE,3)


class PuzzleDataset(Dataset):
    def __init__(self, files, samples, seed, deterministic=False):
        self.files,self.samples,self.seed,self.deterministic=files,samples,seed,deterministic
        self.cache=OrderedDict()

    def __len__(self): return self.samples

    def load(self,index):
        path=self.files[index]
        if path in self.cache:
            self.cache.move_to_end(path); return self.cache[path]
        image=np.asarray(Image.open(path).convert("RGB").resize((480,480)),np.uint8)
        tiles=split_tiles(image)
        self.cache[path]=tiles
        while len(self.cache)>16: self.cache.popitem(last=False)
        return tiles

    def __getitem__(self,index):
        rng=np.random.default_rng(self.seed+index) if self.deterministic else np.random.default_rng()
        fi=index%len(self.files) if self.deterministic else int(rng.integers(len(self.files)))
        tiles=self.load(fi)
        perm=rng.permutation(N)
        x=torch.from_numpy(np.ascontiguousarray(tiles[perm].transpose(0,3,1,2))).float()/127.5-1
        labels=torch.from_numpy(perm.astype(np.int64))
        return x,labels//GRID,labels%GRID,torch.from_numpy(perm.astype(np.int64))


class TileEncoder(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,48,3,padding=1),nn.GroupNorm(8,48),nn.SiLU(),
            nn.Conv2d(48,64,3,stride=2,padding=1),nn.GroupNorm(8,64),nn.SiLU(),
            nn.Conv2d(64,96,3,padding=1),nn.GroupNorm(8,96),nn.SiLU(),
            nn.Conv2d(96,128,3,stride=2,padding=1),nn.GroupNorm(8,128),nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(128,dim),nn.LayerNorm(dim),
        )
    def forward(self,x): return self.net(x)


class GlobalJigsawTransformer(nn.Module):
    def __init__(self,dim=DIM,layers=LAYERS):
        super().__init__()
        self.encoder=TileEncoder(dim)
        block=nn.TransformerEncoderLayer(
            d_model=dim,nhead=6,dim_feedforward=dim*3,dropout=.08,
            activation="gelu",batch_first=True,norm_first=True,
        )
        self.transformer=nn.TransformerEncoder(block,layers,norm=nn.LayerNorm(dim))
        self.row_head=nn.Linear(dim,GRID); self.col_head=nn.Linear(dim,GRID)
    def forward(self,x):
        b,n,c,h,w=x.shape
        z=self.encoder(x.reshape(b*n,c,h,w)).reshape(b,n,-1)
        z=self.transformer(z)
        return self.row_head(z),self.col_head(z)


def assignment(row_logits,col_logits):
    row=F.log_softmax(row_logits,1).cpu().numpy()
    col=F.log_softmax(col_logits,1).cpu().numpy()
    rr,cc=np.divmod(np.arange(N),GRID)
    score=row[:,rr]+col[:,cc]
    tile_idx,pos_idx=linear_sum_assignment(-score)
    layout=np.empty(N,dtype=np.int32); layout[pos_idx]=tile_idx
    return layout


def adjacency(layout,perm):
    orig=perm[layout].reshape(GRID,GRID)
    r=orig[:,1:]==orig[:,:-1]+1
    d=orig[1:]==orig[:-1]+GRID
    return float((r.sum()+d.sum())/(r.size+d.size))


@torch.inference_mode()
def validate(model,loader,device):
    model.eval(); exact=[]; adj=[]; row_acc=[]; col_acc=[]
    for x,row,col,perm in loader:
        x,row,col=x.to(device),row.to(device),col.to(device)
        a,b=model(x)
        row_acc.append(float((a.argmax(2)==row).float().mean()))
        col_acc.append(float((b.argmax(2)==col).float().mean()))
        for i in range(len(x)):
            layout=assignment(a[i],b[i]); truth=np.empty(N,dtype=np.int32)
            p=perm[i].numpy(); truth[p]=np.arange(N)
            exact.append(float(np.mean(layout==truth))); adj.append(adjacency(layout,p))
    return {"tile_exact":float(np.mean(exact)),"adjacency":float(np.mean(adj)),
            "row_acc":float(np.mean(row_acc)),"col_acc":float(np.mean(col_acc))}


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    files=sorted(IMAGE_DIR.glob("*_v*.png"))
    if not files:
        files=sorted(IMAGE_DIR.glob("*.png"))
    stems=sorted({source_stem(x) for x in files})
    n_val=min(VAL_SOURCES,max(1,len(stems)//10))
    val_stems=set(stems[-n_val:])
    train=[x for x in files if source_stem(x) not in val_stems]
    val=[x for x in files if source_stem(x) in val_stems]
    if not train or not val: raise RuntimeError("invalid grouped split")
    train_ds=PuzzleDataset(train,SAMPLES_PER_EPOCH,SEED)
    val_ds=PuzzleDataset(val,VAL_SAMPLES,SEED+999,True)
    train_loader=DataLoader(train_ds,BATCH_SIZE,shuffle=False,num_workers=2,pin_memory=True,drop_last=True)
    val_loader=DataLoader(val,BATCH_SIZE,shuffle=False) if False else DataLoader(val_ds,BATCH_SIZE,shuffle=False,num_workers=0)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=GlobalJigsawTransformer().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=2e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=LR*.05)
    OUT_DIR.mkdir(parents=True,exist_ok=True); history=[]
    print(json.dumps({"train_sources":len(stems)-n_val,"val_sources":n_val,"train_variants":len(train),
                      "val_variants":len(val),"parameters":sum(p.numel() for p in model.parameters())}),flush=True)
    for epoch in range(1,EPOCHS+1):
        model.train(); losses=[]
        for x,row,col,_ in train_loader:
            x,row,col=x.to(device,non_blocking=True),row.to(device,non_blocking=True),col.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=device.type=="cuda",dtype=torch.bfloat16):
                a,b=model(x)
                ce=F.cross_entropy(a.flatten(0,1),row.flatten())+F.cross_entropy(b.flatten(0,1),col.flatten())
                # Every row/column must receive exactly GRID tiles.
                balance=(a.softmax(2).sum(1)-GRID).square().mean()+(b.softmax(2).sum(1)-GRID).square().mean()
                loss=ce+.02*balance
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1)
            opt.step(); losses.append(float(loss.detach()))
        sched.step(); m=validate(model,val_loader,device)
        m.update(epoch=epoch,train_loss=float(np.mean(losses)),lr=opt.param_groups[0]["lr"])
        history.append(m); print(json.dumps(m),flush=True)
        torch.save({"model":model.state_dict(),"epoch":epoch,"schema_version":1,"metrics":m,
                    "config":{"dim":DIM,"layers":LAYERS,"grid":GRID,"tile":TILE}},
                   OUT_DIR/f"global_jigsaw_transformer_epoch{epoch}.pt")
        (OUT_DIR/"metrics.json").write_text(json.dumps(history,indent=2))


if __name__=="__main__":
    main()
