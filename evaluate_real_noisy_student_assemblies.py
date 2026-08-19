"""Create honest validation assembly examples for the real-noise student."""
import json, os, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
import train_directional_jigsaw_transformer as base
from global_solver_candidate import solve_layout

GRID,TILE,N=24,20,576
ROOT=Path(os.getenv("DATA_ROOT","data/real/train")); MAP_FILE=Path(os.getenv("MAP_FILE","real_tile_maps.npz"))
CKPT=Path(os.getenv("CKPT","outputs_real_student/best.pt")); OUT=Path(os.getenv("OUT_DIR","assembly_examples_val"))
POS_CKPT=Path(os.getenv("POS_CKPT","/home/kva/pazzle_assembly_diffusion_v3/outputs/position_prior_diffusion_v3_epoch4.pt"))
SEED=20260817; POSITION_WEIGHT=.20; STEPS=120000

class PositionPrior(nn.Module):
    def __init__(self,base_ch=48):
        super().__init__(); self.encoder=nn.Sequential(nn.Conv2d(3,base_ch,3,padding=1),nn.GroupNorm(8,base_ch),nn.SiLU(),nn.Conv2d(base_ch,base_ch,3,padding=1),nn.GroupNorm(8,base_ch),nn.SiLU(),nn.Conv2d(base_ch,base_ch*2,4,stride=2,padding=1),nn.GroupNorm(8,base_ch*2),nn.SiLU(),nn.Conv2d(base_ch*2,base_ch*4,4,stride=2,padding=1),nn.GroupNorm(8,base_ch*4),nn.SiLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten()); self.row_head=nn.Linear(base_ch*4,24); self.col_head=nn.Linear(base_ch*4,24)
    def forward(self,x): h=self.encoder(x); return self.row_head(h),self.col_head(h)

def split(path):
    x=np.asarray(Image.open(path).convert("RGB"),np.uint8)
    return x.reshape(24,20,24,20,3).transpose(0,2,1,3,4).reshape(576,20,20,3)

def assemble(tiles,layout): return tiles[layout].reshape(24,24,20,20,3).transpose(0,2,1,3,4).reshape(480,480,3)

@torch.inference_mode()
def matrices(model,pos_model,tiles,device):
    x=torch.stack([base.structural_channels(t.astype(np.float32)/255) for t in tiles]).to(device)
    sides,_=model(x); right=(sides[:,1]@sides[:,0].T/.10).float().cpu().numpy(); down=(sides[:,3]@sides[:,2].T/.10).float().cpu().numpy()
    np.fill_diagonal(right,-1e4); np.fill_diagonal(down,-1e4)
    rgb=torch.from_numpy(np.ascontiguousarray(tiles.transpose(0,3,1,2))).float().to(device)/127.5-1
    rr,cc=pos_model(rgb); row=F.log_softmax(rr,1).cpu().numpy(); col=F.log_softmax(cc,1).cpu().numpy(); pr,pc=np.divmod(np.arange(N),24); pos=row[:,pr]+col[:,pc]
    return right,down,pos

def objective(layout,right,down,pos):
    b=layout.reshape(24,24); return float(right[b[:,:-1],b[:,1:]].sum()+down[b[:-1],b[1:]].sum()+POSITION_WEIGHT*pos[layout,np.arange(N)].sum())

def local_opt(layout,right,down,pos,seed):
    rng=np.random.default_rng(seed); cur=objective(layout,right,down,pos); best=layout.copy(); bv=cur
    for step in range(STEPS):
        a=int(rng.integers(N)); b=int(rng.integers(N-1)); b+=b>=a
        affected={a,b}
        for p in (a,b):
            r,c=divmod(p,24)
            if c: affected.add(p-1)
            if c<23: affected.add(p+1)
            if r: affected.add(p-24)
            if r<23: affected.add(p+24)
        def val():
            v=0.
            for p in affected:
                r,c=divmod(p,24); t=layout[p]; v+=POSITION_WEIGHT*pos[t,p]
                if c<23: v+=right[t,layout[p+1]]
                if r<23: v+=down[t,layout[p+24]]
            return v
        before=val(); layout[a],layout[b]=layout[b],layout[a]; delta=val()-before
        temp=.18*(1-step/STEPS)+.005
        if delta>=0 or rng.random()<np.exp(delta/temp):
            cur+=delta
            if cur>bv: bv=cur; best=layout.copy()
        else: layout[a],layout[b]=layout[b],layout[a]
    return best

def metrics(layout,true_layout):
    target_of=np.empty(N,np.int32); target_of[true_layout]=np.arange(N); x=target_of[layout].reshape(24,24)
    r=(x[:,1:]==x[:,:-1]+1)&(x[:,:-1]//24==x[:,1:]//24); d=x[1:]==x[:-1]+24
    return {"tile_exact":float(np.mean(layout==true_layout)),"adjacency":float((r.sum()+d.sum())/(r.size+d.size))}

def panel(inp,pred,target,title):
    canvas=Image.new("RGB",(1440,520),"white"); draw=ImageDraw.Draw(canvas)
    canvas.paste(Image.fromarray(inp),(0,40)); canvas.paste(Image.fromarray(pred),(480,40)); canvas.paste(Image.fromarray(target),(960,40))
    draw.text((10,12),"NOISY SHUFFLED INPUT",fill="black"); draw.text((490,12),"STUDENT + GLOBAL SOLVER",fill="black"); draw.text((970,12),"CLEAN TARGET",fill="black"); draw.text((650,12),title,fill=(20,80,160))
    return canvas

def main():
    device=torch.device("cuda"); model=base.DirectionalTransformer().to(device); ck=torch.load(CKPT,map_location="cpu",weights_only=False); model.load_state_dict(ck["model"]); model.eval()
    pos_model=PositionPrior().to(device); pos_model.load_state_dict(torch.load(POS_CKPT,map_location="cpu",weights_only=False)["model"]); pos_model.eval()
    z=np.load(MAP_FILE); stems=z["stems"]; maps=z["maps"]; order=np.arange(len(stems)); np.random.default_rng(SEED).shuffle(order); vi=order[-max(100,len(order)//10):]
    chosen=vi[[0,len(vi)//3,2*len(vi)//3,-1]]; OUT.mkdir(parents=True,exist_ok=True); reports=[]; panels=[]
    for k,j in enumerate(chosen):
        stem=str(stems[j]); inp=np.asarray(Image.open(ROOT/"inputs"/(stem+".png")).convert("RGB"),np.uint8); target=np.asarray(Image.open(ROOT/"targets"/(stem+".png")).convert("RGB"),np.uint8); tiles=split(ROOT/"inputs"/(stem+".png"))
        right,down,pos=matrices(model,pos_model,tiles,device); layout=solve_layout(right,down,pos,SEED+k*100); m=metrics(layout,maps[j]); m.update(stem=stem,split="validation"); reports.append(m)
        p=panel(inp,assemble(tiles,layout),target,f"{stem}  adjacency={m['adjacency']:.3f}"); p.save(OUT/f"{stem}_comparison.png"); panels.append(p)
    montage=Image.new("RGB",(1440,520*len(panels)),"white")
    for i,p in enumerate(panels): montage.paste(p,(0,i*520))
    montage.save(OUT/"validation_assembly_examples.png"); (OUT/"metrics.json").write_text(json.dumps(reports,indent=2)); print(json.dumps(reports),flush=True)

if __name__=="__main__": main()
