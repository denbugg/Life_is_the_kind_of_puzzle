"""Fine-tune directional transformer on genuinely noisy, shuffled train inputs."""
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset

import train_directional_jigsaw_transformer as base

DATA_ROOT=Path(os.getenv("DATA_ROOT","data/real/train"))
MAP_FILE=Path(os.getenv("MAP_FILE","real_tile_maps.npz"))
TEACHER=Path(os.getenv("TEACHER","outputs_real/best.pt"))
OUT_DIR=Path(os.getenv("OUT_DIR","outputs_real_student"))
EPOCHS=int(os.getenv("EPOCHS","14")); STEPS=int(os.getenv("STEPS_PER_EPOCH","1000"))
VAL_STEPS=int(os.getenv("VAL_STEPS","120")); BATCH=int(os.getenv("BATCH_SIZE","3"))
TILES=int(os.getenv("TILES_PER_IMAGE","144")); LR=float(os.getenv("LR","1.2e-4"))
DISTILL=float(os.getenv("DISTILL_WEIGHT","0.35")); SEED=int(os.getenv("SEED","20260817"))


def split(path):
    x=np.asarray(Image.open(path).convert("RGB").resize((480,480)),np.uint8)
    return x.reshape(24,20,24,20,3).transpose(0,2,1,3,4).reshape(576,20,20,3)


def select_ids(rng):
    anchors=rng.choice(576,min(TILES//3,576),replace=False); ids=set(map(int,anchors))
    for i in anchors:
        r,c=divmod(int(i),24)
        if c: ids.add(int(i)-1)
        if c<23: ids.add(int(i)+1)
        if r: ids.add(int(i)-24)
        if r<23: ids.add(int(i)+24)
    rem=np.setdiff1d(np.arange(576),np.fromiter(ids,np.int64)); need=max(0,TILES-len(ids))
    if need: ids.update(map(int,rng.choice(rem,min(need,len(rem)),replace=False)))
    return np.asarray(sorted(ids),np.int64)[:TILES]


class RealPairs(Dataset):
    def __init__(self,stems,maps,samples,seed,train): self.stems,self.maps,self.samples,self.seed,self.train=stems,maps,samples,seed,train
    def __len__(self): return self.samples
    def __getitem__(self,index):
        rng=np.random.default_rng(None if self.train else self.seed+index)
        j=int(rng.integers(len(self.stems))) if self.train else index%len(self.stems)
        ids=select_ids(rng); stem=str(self.stems[j])
        noisy=split(DATA_ROOT/"inputs"/(stem+".png")); clean=split(DATA_ROOT/"targets"/(stem+".png"))
        # maps[j, position] gives the source index in the shuffled noisy image.
        real=noisy[self.maps[j,ids]]
        x=torch.stack([base.structural_channels(t.astype(np.float32)/255) for t in real])
        teacher_x=torch.stack([base.structural_channels(t.astype(np.float32)/255) for t in clean[ids]])
        return x,teacher_x,torch.from_numpy(ids)


def student_loss(student,teacher,x,clean,ids):
    task,r,b=base.batch_loss(student,x,ids)
    with torch.no_grad(): tz=torch.stack([teacher(c)[0] for c in clean])
    sz=torch.stack([student(a)[0] for a in x])
    distill=(1-(sz*tz).sum(-1)).mean()
    return task+DISTILL*distill,r,b,distill


@torch.inference_mode()
def validate(student,teacher,loader,device):
    student.eval(); vals=[]
    for x,c,ids in loader:
        loss,r,b,d=student_loss(student,teacher,x.to(device),c.to(device),ids.to(device)); vals.append((float(loss),float(r[0]),float(r[1]),float(b),float(d)))
    a=np.asarray(vals); return {"val_loss":float(a[:,0].mean()),"recall_at_1":float(a[:,1].mean()),"recall_at_5":float(a[:,2].mean()),"border_acc":float(a[:,3].mean()),"distill_loss":float(a[:,4].mean())}


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    z=np.load(MAP_FILE); stems=z["stems"]; maps=z["maps"]
    order=np.arange(len(stems)); np.random.default_rng(SEED).shuffle(order); nv=max(100,len(order)//10); vi,ti=order[-nv:],order[:-nv]
    tr=DataLoader(RealPairs(stems[ti],maps[ti],STEPS*BATCH,SEED,True),BATCH,num_workers=4,pin_memory=True,drop_last=True)
    va=DataLoader(RealPairs(stems[vi],maps[vi],VAL_STEPS*BATCH,SEED+999,False),BATCH,num_workers=2,pin_memory=True)
    device=torch.device("cuda"); ck=torch.load(TEACHER,map_location="cpu",weights_only=False)
    teacher=base.DirectionalTransformer().to(device); teacher.load_state_dict(ck["model"]); teacher.eval(); teacher.requires_grad_(False)
    student=base.DirectionalTransformer().to(device); student.load_state_dict(ck["model"])
    opt=torch.optim.AdamW(student.parameters(),LR,weight_decay=2e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=LR*.05)
    scaler=torch.amp.GradScaler("cuda"); OUT_DIR.mkdir(parents=True,exist_ok=True); hist=[]; best=-1
    print(json.dumps({"device":torch.cuda.get_device_name(0),"pairs":len(stems),"train_pairs":len(ti),"val_pairs":len(vi),"teacher_epoch":ck.get("epoch"),"parameters":sum(p.numel() for p in student.parameters())}),flush=True)
    for epoch in range(1,EPOCHS+1):
        student.train(); ls=[]; rs=[]; ds=[]
        for x,c,ids in tr:
            x,c,ids=x.to(device,non_blocking=True),c.to(device,non_blocking=True),ids.to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",dtype=torch.bfloat16): loss,r,_,d=student_loss(student,teacher,x,c,ids)
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(student.parameters(),1); scaler.step(opt); scaler.update()
            ls.append(float(loss.detach())); rs.append(float(r[0])); ds.append(float(d))
        sched.step(); m=validate(student,teacher,va,device); m.update(epoch=epoch,train_loss=float(np.mean(ls)),train_recall_at_1=float(np.mean(rs)),train_distill_loss=float(np.mean(ds)),lr=opt.param_groups[0]["lr"]); hist.append(m); print(json.dumps(m),flush=True)
        payload={"model":student.state_dict(),"epoch":epoch,"metrics":m,"teacher":str(TEACHER),"schema_version":2}
        torch.save(payload,OUT_DIR/f"epoch_{epoch}.pt")
        if m["recall_at_1"]>best: best=m["recall_at_1"]; torch.save(payload,OUT_DIR/"best.pt")
        (OUT_DIR/"metrics.json").write_text(json.dumps(hist,indent=2))


if __name__=="__main__": main()
