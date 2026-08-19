"""Train edge and position models on correctly laid-out Diffusion-v2 outputs."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import kaggle_train_puzzle_assembly as base

IMAGE_DIR = Path(os.getenv("IMAGE_DIR", "restored_rl_targets"))
OUT_DIR = Path(os.getenv("OUT_DIR", "assembly_outputs"))
EDGE_EPOCHS = int(os.getenv("EDGE_EPOCHS", "4"))
POS_EPOCHS = int(os.getenv("POS_EPOCHS", "4"))
SAMPLES = int(os.getenv("SAMPLES", "240000"))
VAL_SAMPLES = int(os.getenv("VAL_SAMPLES", "10000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "512"))
LR = float(os.getenv("LR", "2e-4"))
SEED = int(os.getenv("SEED", "27072029"))


def source_stem(path):
    return path.stem.rsplit("_v", 1)[0]


def train_model(kind, train_files, val_files, device):
    if kind == "edge":
        train_ds = base.EdgePairDataset(train_files, SAMPLES, augment=True)
        val_ds = base.EdgePairDataset(val_files, VAL_SAMPLES, augment=False)
        model = base.EdgeMatcher().to(device)
    else:
        train_ds = base.PositionDataset(train_files, SAMPLES, augment=True)
        val_ds = base.PositionDataset(val_files, VAL_SAMPLES, augment=False)
        model = base.PositionPrior().to(device)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    epochs = EDGE_EPOCHS if kind == "edge" else POS_EPOCHS
    history = []
    for epoch in range(1, epochs+1):
        model.train(); losses=[]
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            if kind == "edge":
                x,y=(z.to(device,non_blocking=True) for z in batch)
                with torch.amp.autocast("cuda",enabled=device.type=="cuda"):
                    loss=F.binary_cross_entropy_with_logits(model(x),y)
            else:
                x,row,col=(z.to(device,non_blocking=True) for z in batch)
                with torch.amp.autocast("cuda",enabled=device.type=="cuda"):
                    a,b=model(x); loss=F.cross_entropy(a,row)+F.cross_entropy(b,col)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            scaler.step(opt); scaler.update(); losses.append(float(loss.detach()))
        if kind == "edge":
            val_loss,acc=base.eval_edge(model,val_loader,device,max_batches=10000)
            metrics={"epoch":epoch,"train_loss":float(np.mean(losses)),"val_loss":val_loss,"val_acc":acc}
            name=f"edge_matcher_diffusion_v3_epoch{epoch}.pt"
        else:
            val_loss,row_acc,col_acc=base.eval_position(model,val_loader,device,max_batches=10000)
            metrics={"epoch":epoch,"train_loss":float(np.mean(losses)),"val_loss":val_loss,
                     "row_acc":row_acc,"col_acc":col_acc}
            name=f"position_prior_diffusion_v3_epoch{epoch}.pt"
        history.append(metrics); print(json.dumps({"kind":kind,**metrics}),flush=True)
        torch.save({"model":model.state_dict(),"epoch":epoch,"schema_version":3,"metrics":metrics,
                    "config":{"grid":24,"tile":20,"source":"diffusion_v2_outputs"}},OUT_DIR/name)
    return history


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    files=sorted(IMAGE_DIR.glob("*_v*.png")); stems=sorted({source_stem(x) for x in files})
    val=set(stems[-4:])
    train_files=[x for x in files if source_stem(x) not in val]
    val_files=[x for x in files if source_stem(x) in val]
    if not train_files or not val_files: raise RuntimeError("invalid grouped split")
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    print(json.dumps({"train_sources":len(stems)-4,"val_sources":4,
                      "train_images":len(train_files),"val_images":len(val_files)}),flush=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result={"edge":train_model("edge",train_files,val_files,device),
            "position":train_model("position",train_files,val_files,device)}
    (OUT_DIR/"metrics.json").write_text(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
