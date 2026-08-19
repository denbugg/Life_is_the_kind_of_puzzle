"""Context-aware second-pass diffusion over overlapping 3x3 tile patches."""
import json
import math
import os
import random
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from kaggle_ddpm_denoise_fragments import Diffusion, TinyCondUNet

PATCH = 60
GRID_PATCHES = 22
TIMESTEPS = 200
START_T = int(os.getenv("START_T", "199"))
EPOCHS = int(os.getenv("EPOCHS", "14"))
SAMPLES_PER_EPOCH = int(os.getenv("SAMPLES_PER_EPOCH", "30000"))
VAL_SAMPLES = int(os.getenv("VAL_SAMPLES", "1024"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LR = float(os.getenv("LR", "4e-5"))
EMA_RATE = float(os.getenv("EMA_RATE", "0.01"))
SEED = int(os.getenv("SEED", "27072028"))
RESTORED_DIR = Path(os.getenv("RESTORED_DIR", "restored_rl_targets"))
CLEAN_DIR = Path(os.getenv("CLEAN_DIR", "clean_targets"))
RESUME = Path(os.getenv("RESUME", "ddpm_restorer_v2_epoch18.pt"))
OUT_DIR = Path(os.getenv("OUT_DIR", "context_outputs"))


def source_stem(path):
    return path.stem.rsplit("_v", 1)[0]


class ContextPatchDataset(Dataset):
    def __init__(self, restored_files, clean_dir, samples, seed, deterministic=False):
        self.files, self.clean_dir = restored_files, clean_dir
        self.samples, self.seed, self.deterministic = samples, seed, deterministic
        self.cache = OrderedDict()

    def __len__(self):
        return self.samples

    def load_pair(self, index):
        path = self.files[index]
        key = str(path)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        cond = np.asarray(Image.open(path).convert("RGB").resize((480, 480)), np.uint8)
        clean = np.asarray(Image.open(self.clean_dir/f"{source_stem(path)}.png").convert("RGB").resize((480, 480)), np.uint8)
        self.cache[key] = (cond, clean)
        while len(self.cache) > 12:
            self.cache.popitem(last=False)
        return cond, clean

    def __getitem__(self, index):
        rng = random.Random(self.seed + index) if self.deterministic else random
        file_i = index % len(self.files) if self.deterministic else rng.randrange(len(self.files))
        cond, clean = self.load_pair(file_i)
        r, c = rng.randrange(GRID_PATCHES), rng.randrange(GRID_PATCHES)
        y, x = r*20, c*20
        a = np.ascontiguousarray(cond[y:y+PATCH, x:x+PATCH].transpose(2,0,1))
        b = np.ascontiguousarray(clean[y:y+PATCH, x:x+PATCH].transpose(2,0,1))
        a, b = torch.from_numpy(a).float()/127.5-1, torch.from_numpy(b).float()/127.5-1
        if not self.deterministic:
            if random.random() < .5: a, b = a.flip(2), b.flip(2)
            if random.random() < .5: a, b = a.flip(1), b.flip(1)
            k = random.randrange(4)
            if k: a, b = torch.rot90(a,k,(1,2)), torch.rot90(b,k,(1,2))
        return a, b


def ssim(a, b):
    a, b = (a+1)/2, (b+1)/2
    ma, mb = F.avg_pool2d(a,5,1,2), F.avg_pool2d(b,5,1,2)
    va = F.avg_pool2d(a*a,5,1,2)-ma*ma
    vb = F.avg_pool2d(b*b,5,1,2)-mb*mb
    vab = F.avg_pool2d(a*b,5,1,2)-ma*mb
    return (((2*ma*mb+.01**2)*(2*vab+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2)+1e-8)).mean()


def gradient_loss(a,b):
    return F.smooth_l1_loss(a[...,1:]-a[...,:-1],b[...,1:]-b[...,:-1],beta=.03) + \
           F.smooth_l1_loss(a[...,1:,:]-a[...,:-1,:],b[...,1:,:]-b[...,:-1,:],beta=.03)


@torch.inference_mode()
def ddim(model, diffusion, cond, steps=24, seed=123):
    gen = torch.Generator(device=cond.device).manual_seed(seed)
    # High-noise x0 estimator: target leakage through x_t is negligible, so
    # the correction must come from context. One-step inference is stable and
    # avoids accumulating tile-colour drift.
    x = torch.randn(cond.shape, generator=gen, device=cond.device, dtype=cond.dtype)
    t = torch.full((len(cond),),START_T,device=cond.device,dtype=torch.long)
    pred_residual = model(x,cond,t).clamp(-1,1)
    return (cond + pred_residual).clamp(-1, 1)


def metric(pred,target):
    mse = F.mse_loss((pred+1)/2,(target+1)/2).item()
    return -10*math.log10(max(mse,1e-12)),float(ssim(pred,target))


def preview(cond,pred,clean,path,n=6):
    scale=2; label=24
    canvas=Image.new("RGB",(n*PATCH*scale,3*(PATCH*scale+label)),"white")
    draw=ImageDraw.Draw(canvas)
    for row,(name,batch) in enumerate((("INPUT 3x3",cond),("CONTEXT DDIM",pred),("TARGET",clean))):
        y=row*(PATCH*scale+label); draw.text((4,y+5),name,fill="black")
        for j in range(n):
            arr=((batch[j].cpu().permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
            canvas.paste(Image.fromarray(arr).resize((PATCH*scale,PATCH*scale)),(j*PATCH*scale,y+label))
    canvas.save(path)


@torch.inference_mode()
def validate(model,diffusion,loader,device,out):
    model.eval(); raw=[]; restored=[]; first=None
    for i,(cond,clean) in enumerate(loader):
        cond,clean=cond.to(device),clean.to(device)
        pred=ddim(model,diffusion,cond,seed=SEED+i)
        raw.append(metric(cond,clean)); restored.append(metric(pred,clean))
        if first is None: first=(cond,pred,clean)
        if i>=7: break
    preview(*first,out)
    return {"raw_psnr":float(np.mean([x[0] for x in raw])),
            "raw_ssim":float(np.mean([x[1] for x in raw])),
            "restored_psnr":float(np.mean([x[0] for x in restored])),
            "restored_ssim":float(np.mean([x[1] for x in restored]))}


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    files=sorted(RESTORED_DIR.glob("*_v*.png"))
    stems=sorted({source_stem(x) for x in files})
    if len(stems)<5: raise RuntimeError("need at least five source images")
    val_stems=set(stems[-4:])
    train_files=[x for x in files if source_stem(x) not in val_stems]
    val_files=[x for x in files if source_stem(x) in val_stems]
    train_ds=ContextPatchDataset(train_files,CLEAN_DIR,SAMPLES_PER_EPOCH,SEED)
    val_ds=ContextPatchDataset(val_files,CLEAN_DIR,VAL_SAMPLES,SEED+999,True)
    train_loader=DataLoader(train_ds,BATCH_SIZE,shuffle=False,num_workers=2,pin_memory=True,drop_last=True)
    val_loader=DataLoader(val_ds,32,shuffle=False,num_workers=0)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base=torch.load(RESUME,map_location="cpu",weights_only=False)
    model=TinyCondUNet(base=64).to(device); model.load_state_dict(base["model"])
    # Safe identity initialization for direct residual prediction.
    torch.nn.init.zeros_(model.out.weight)
    torch.nn.init.zeros_(model.out.bias)
    ema=deepcopy(model).eval().requires_grad_(False)
    diffusion=Diffusion(TIMESTEPS,device)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=LR*.08)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    OUT_DIR.mkdir(parents=True,exist_ok=True); history=[]
    print(json.dumps({"train_sources":len(stems)-4,"val_sources":4,"train_variants":len(train_files),
                      "val_variants":len(val_files),"patch":PATCH}),flush=True)
    for epoch in range(1,EPOCHS+1):
        model.train(); losses=[]
        for cond,clean in train_loader:
            cond,clean=cond.to(device,non_blocking=True),clean.to(device,non_blocking=True)
            residual=(clean-cond).clamp(-1,1)
            t=torch.randint(max(START_T//2,1),START_T+1,(len(clean),),device=device)
            noise=torch.randn_like(clean); xt=diffusion.q_sample(residual,t,noise)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=device.type=="cuda"):
                pred_residual=model(xt,cond,t).clamp(-1,1)
                restored=(cond+pred_residual).clamp(-1,1)
                charb=torch.sqrt((restored-clean).square()+1e-6).mean()
                loss=F.smooth_l1_loss(pred_residual,residual,beta=.04)+6*charb+1.2*gradient_loss(restored,clean)+.6*(1-ssim(restored,clean))
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            scaler.step(opt); scaler.update(); losses.append(float(loss.detach()))
            with torch.no_grad():
                for ep,p in zip(ema.parameters(),model.parameters()): ep.lerp_(p,EMA_RATE)
        sched.step(); m=validate(ema,diffusion,val_loader,device,OUT_DIR/f"preview_epoch{epoch}.png")
        m.update(epoch=epoch,train_loss=float(np.mean(losses)),lr=opt.param_groups[0]["lr"])
        history.append(m); print(json.dumps(m),flush=True)
        torch.save({"model":ema.state_dict(),"epoch":epoch,"schema_version":3,"metrics":m,
                    "config":{"patch":PATCH,"context_tiles":3,"timesteps":TIMESTEPS,
                              "cond_start_t":START_T,"sampler":"context_high_noise_x0_residual"}},
                   OUT_DIR/f"context_diffusion_v3_epoch{epoch}.pt")
        (OUT_DIR/"metrics.json").write_text(json.dumps(history,indent=2))


if __name__=="__main__":
    main()
