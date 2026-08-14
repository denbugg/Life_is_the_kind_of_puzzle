"""P6: FIT-only conditional positional diffusion for shuffled 24x24 tile bags.

No CAL/DEV/test source, rank96 solver, output assembly, restorer, or submission
is imported.  The denoiser sees an arbitrary tile-set ordering plus a noised 2D
position state per tile; only FIT labels supply training u_0 coordinates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F

from train_eval_cb1_g1_capacity import distort_frags, load_rgb, sha256_file, to_frags

GRID=24; N=GRID*GRID; T=32
FIT_TARGETS=Path(r"E:\pazzle_data\train\targets")
SPLIT=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json")
WORK=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P6_positional_diffusion\g0_g1_capacity")

@dataclass(frozen=True)
class Spec: width:int=192; blocks:int=6; heads:int=8

def args()->argparse.Namespace:
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('phase',choices=('g0','g1')); p.add_argument('--targets',type=Path,default=FIT_TARGETS); p.add_argument('--split',type=Path,default=SPLIT); p.add_argument('--work',type=Path,default=WORK); p.add_argument('--device',default='cuda'); p.add_argument('--seed',type=int,default=20260818); p.add_argument('--train-sources',type=int,default=256); p.add_argument('--eval-sources',type=int,default=32); p.add_argument('--steps',type=int,default=8000); p.add_argument('--width',type=int,default=192); p.add_argument('--blocks',type=int,default=6); p.add_argument('--heads',type=int,default=8); return p.parse_args()

def source_sets(cfg:argparse.Namespace)->tuple[list[str],list[str]]:
 x=json.loads(cfg.split.read_text(encoding='utf-8')); fit=list(x['splits']['fit']); cal=set(x['splits']['cal']); dev=set(x['splits']['dev'])
 if len(fit)!=5360 or cfg.train_sources!=256 or cfg.eval_sources!=32 or set(fit)&(cal|dev): raise RuntimeError('P6 split contract')
 train,hold=fit[288:544],fit[544:576]
 if set(train)&set(hold) or any(k in cal or k in dev for k in train+hold): raise RuntimeError('non-FIT source')
 for k in train+hold:
  if not (cfg.targets/k).is_file(): raise FileNotFoundError(cfg.targets/k)
 return train,hold

def make_bag(targets:Path,name:str,seed:int,device:torch.device)->tuple[torch.Tensor,torch.Tensor]:
 clean=load_rgb(targets/name); frags=distort_frags(to_frags(clean),np.random.default_rng(seed*1009+int(name[4:10]))); perm=np.random.default_rng(seed*2029+int(name[4:10])).permutation(N).astype(np.int64)
 tiles=torch.from_numpy(frags[perm]).permute(0,3,1,2).contiguous().float().div_(255).unsqueeze(0).to(device)
 slots=torch.from_numpy(perm).long().to(device); row=(slots//GRID).float(); col=(slots%GRID).float(); pos=torch.stack((2*col/(GRID-1)-1,2*row/(GRID-1)-1),dim=-1).unsqueeze(0)
 return tiles,pos

def schedule(device:torch.device)->torch.Tensor:
 # Cosine alpha-bar, t=0 exact clean; t=T retains a small but nonzero signal.
 steps=torch.arange(T+1,dtype=torch.float32,device=device); s=0.008; values=torch.cos(((steps/T+s)/(1+s))*math.pi/2).pow(2); return values/values[0]

def time_features(t:torch.Tensor,width:int)->torch.Tensor:
 half=width//2; f=torch.exp(-math.log(10000)*torch.arange(half,device=t.device,dtype=torch.float32)/max(half-1,1)); a=t.float().unsqueeze(-1)*f.unsqueeze(0); return torch.cat((torch.sin(a),torch.cos(a)),dim=-1)

class Stem(nn.Module):
 def __init__(self,w:int)->None:
  super().__init__(); a,b=w//3,w//2; self.net=nn.Sequential(nn.Conv2d(3,a,3,padding=1,bias=False),nn.GroupNorm(max(1,a//16),a),nn.SiLU(),nn.Conv2d(a,b,3,2,1,bias=False),nn.GroupNorm(max(1,b//16),b),nn.SiLU(),nn.Conv2d(b,w,3,2,1,bias=False),nn.GroupNorm(max(1,w//16),w),nn.SiLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.LayerNorm(w))
 def forward(self,x:torch.Tensor)->torch.Tensor:
  b,n,c,h,w=x.shape; return self.net(x.reshape(b*n,c,h,w)).reshape(b,n,-1)
class Block(nn.Module):
 def __init__(self,w:int,h:int)->None:
  super().__init__(); self.n1=nn.LayerNorm(w); self.a=nn.MultiheadAttention(w,h,batch_first=True,dropout=0.0); self.n2=nn.LayerNorm(w); self.f=nn.Sequential(nn.Linear(w,4*w),nn.SiLU(),nn.Linear(4*w,w))
 def forward(self,x:torch.Tensor)->torch.Tensor:
  y=self.n1(x); x=x+self.a(y,y,y,need_weights=False)[0]; return x+self.f(self.n2(x))
class Denoiser(nn.Module):
 def __init__(self,s:Spec,use_set:bool)->None:
  super().__init__(); self.use_set=use_set; self.stem=Stem(s.width); self.pos=nn.Sequential(nn.Linear(2,s.width),nn.SiLU(),nn.Linear(s.width,s.width)); self.time=nn.Sequential(nn.Linear(s.width,s.width),nn.SiLU(),nn.Linear(s.width,s.width)); self.blocks=nn.ModuleList([Block(s.width,s.heads) for _ in range(s.blocks)]) if use_set else nn.ModuleList(); self.local=nn.ModuleList([nn.Sequential(nn.LayerNorm(s.width),nn.Linear(s.width,4*s.width),nn.SiLU(),nn.Linear(4*s.width,s.width)) for _ in range(s.blocks)]) if not use_set else nn.ModuleList(); self.out=nn.Sequential(nn.LayerNorm(s.width),nn.Linear(s.width,s.width),nn.SiLU(),nn.Linear(s.width,2))
 def forward(self,tiles:torch.Tensor,u:torch.Tensor,t:torch.Tensor)->torch.Tensor:
  x=self.stem(tiles)+self.pos(u)+self.time(time_features(t,self.stem.net[-1].normalized_shape[0])).unsqueeze(1)
  for b in self.blocks:x=b(x)
  for b in self.local:x=x+b(x)
  return self.out(x)

def denoise_loss(model:Denoiser,tiles:torch.Tensor,u0:torch.Tensor,bar:torch.Tensor,t:torch.Tensor)->torch.Tensor:
 eps=torch.randn_like(u0); a=bar[t].view(-1,1,1); ut=a.sqrt()*u0+(1-a).sqrt()*eps; return F.mse_loss(model(tiles,ut,t),eps)

def reverse(model:Denoiser,tiles:torch.Tensor,bar:torch.Tensor)->torch.Tensor:
 # Deterministic epsilon prediction: convert current u_t to x0 then step to t-1.
 u=torch.randn((tiles.shape[0],N,2),device=tiles.device); model.eval()
 with torch.no_grad():
  for k in range(T,0,-1):
   t=torch.full((tiles.shape[0],),k,device=tiles.device,dtype=torch.long); a=bar[k]; prev=bar[k-1]; eps=model(tiles,u,t); x0=(u-(1-a).sqrt()*eps)/a.sqrt().clamp_min(1e-6); u=prev.sqrt()*x0+(1-prev).sqrt()*eps
 return u

def hungarian_accuracy(points:torch.Tensor,u0:torch.Tensor)->float:
 slots=torch.stack(torch.meshgrid(torch.linspace(-1,1,GRID,device=points.device),torch.linspace(-1,1,GRID,device=points.device),indexing='ij'),dim=-1).reshape(N,2)[:,[1,0]]
 scores=-torch.cdist(points,slots).pow(2); values=scores.detach().cpu().numpy(); total=[]
 for b in range(values.shape[0]):
  row,col=linear_sum_assignment(-values[b]); board=np.empty(N,dtype=np.int16); board[col]=row
  target=((u0[b,:,1]+1)*(GRID-1)/2).round().long()*GRID+((u0[b,:,0]+1)*(GRID-1)/2).round().long(); tgt=target.cpu().numpy(); total.append(float(np.mean(tgt[board]==np.arange(N))))
 return float(np.mean(total))
def config_hash(s:Spec,use_set:bool)->str:return hashlib.sha256(json.dumps({'spec':s.__dict__,'set':use_set,'T':T},sort_keys=True).encode()).hexdigest()

def g0(cfg:argparse.Namespace)->None:
 if cfg.device!='cuda' or not torch.cuda.is_available():raise RuntimeError('local CUDA required')
 train,_=source_sets(cfg); d=torch.device('cuda'); s=Spec(cfg.width,cfg.blocks,cfg.heads); torch.manual_seed(cfg.seed); m=Denoiser(s,True).to(d).eval(); bar=schedule(d); rows=[]
 with torch.no_grad():
  for i,name in enumerate(train[:4]):
   tiles,u0=make_bag(cfg.targets,name,cfg.seed+i,d); t=torch.tensor([17],device=d); eps=torch.randn_like(u0); a=bar[t].view(1,1,1); ut=a.sqrt()*u0+(1-a).sqrt()*eps; out=m(tiles,ut,t)
   p=torch.from_numpy(np.random.default_rng(cfg.seed+8000+i).permutation(N)).long().to(d); outp=m(tiles[:,p],ut[:,p],t); permerr=float((outp-out[:,p]).abs().max()); decoded=reverse(m,tiles,bar); finite=bool(torch.isfinite(decoded).all()); acc=hungarian_accuracy(decoded,u0); rows.append({'source':name,'max_equivariance_abs':permerr,'finite_reverse':finite,'hungarian_bijection':True,'untrained_reverse_accuracy':acc})
 passed=all(r['max_equivariance_abs']<1e-5 and r['finite_reverse'] and r['hungarian_bijection'] for r in rows); cfg.work.mkdir(parents=True,exist_ok=True); rep={'experiment':'P6_conditional_positional_diffusion','gate':'G0_equivariance_diffusion_contract','passes':passed,'decision':'pass_to_G1_capacity' if passed else 'reject_P6_before_training','checks':rows,'config_sha256':config_hash(s,True),'split_sha256':sha256_file(cfg.split),'CAL_target_opened':False,'DEV_targets_opened':False,'test_accessed':False,'layouts_assembled':False,'restorer_used':False}; (cfg.work/'p6_g0_report.json').write_text(json.dumps(rep,indent=2),encoding='utf-8');print(json.dumps(rep,indent=2),flush=True)

def train_one(cfg:argparse.Namespace,use_set:bool,train:list[str],hold:list[str],d:torch.device)->dict[str,object]:
 s=Spec(cfg.width,cfg.blocks,cfg.heads); torch.manual_seed(cfg.seed+(0 if use_set else 1)); m=Denoiser(s,use_set).to(d); opt=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-4); rng=np.random.default_rng(cfg.seed+(41 if use_set else 43)); bar=schedule(d); losses=[];m.train()
 for step in range(cfg.steps):
  name=train[int(rng.integers(len(train)))]; tiles,u0=make_bag(cfg.targets,name,cfg.seed*100000+step,d); t=torch.tensor([int(rng.integers(1,T+1))],device=d); loss=denoise_loss(m,tiles,u0,bar,t);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();losses.append(float(loss.detach().cpu()))
  if (step+1)%250==0:print(f"model={'set' if use_set else 'independent'} step={step+1} loss={np.mean(losses[-100:]):.6f}",flush=True)
 scores=[]
 for i,name in enumerate(hold):
  tiles,u0=make_bag(cfg.targets,name,cfg.seed+7000+i,d);scores.append(hungarian_accuracy(reverse(m,tiles,bar),u0))
 ckpt=cfg.work/('p6_g1_set_diffusion.pt' if use_set else 'p6_g1_independent_diffusion.pt');torch.save({'state_dict':m.state_dict(),'spec':s.__dict__,'set':use_set,'seed':cfg.seed,'steps':cfg.steps,'T':T},ckpt)
 return {'use_set_attention':use_set,'loss_first_100':float(np.mean(losses[:100])),'loss_last_100':float(np.mean(losses[-100:])),'heldout_reverse_hungarian_accuracy':float(np.mean(scores)),'checkpoint':str(ckpt),'checkpoint_sha256':sha256_file(ckpt),'config_sha256':config_hash(s,use_set)}
def g1(cfg:argparse.Namespace)->None:
 if cfg.device!='cuda' or not torch.cuda.is_available():raise RuntimeError('local CUDA required')
 if cfg.steps!=8000 or (cfg.width,cfg.blocks,cfg.heads)!=(192,6,8):raise ValueError('P6 fixed G1 contract')
 train,hold=source_sets(cfg);d=torch.device('cuda');cfg.work.mkdir(parents=True,exist_ok=True);setr=train_one(cfg,True,train,hold,d);ind=train_one(cfg,False,train,hold,d);a=float(setr['heldout_reverse_hungarian_accuracy']);b=float(ind['heldout_reverse_hungarian_accuracy']);passed=bool(a>=.01 and a>=b+.005 and float(setr['loss_last_100'])<float(setr['loss_first_100']));rep={'experiment':'P6_conditional_positional_diffusion','gate':'G1_FIT_global_position_capacity','set_denoiser':setr,'independent_denoiser':ind,'heldout_reverse_delta_pp':100*(a-b),'pass_criteria':'set reverse Hungarian >=1.0%, delta >=+0.5pp, loss decreases','passes_G1':passed,'decision':'pass_to_full_FIT_scale' if passed else 'reject_P6_before_scale_CAL','train_sources':train,'heldout_sources':hold,'split_sha256':sha256_file(cfg.split),'CAL_target_opened':False,'DEV_targets_opened':False,'test_accessed':False,'layouts_assembled':False,'restorer_used':False};(cfg.work/'p6_g1_report.json').write_text(json.dumps(rep,indent=2),encoding='utf-8');print(json.dumps(rep,indent=2),flush=True)
def main()->None:
 cfg=args();random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed);g0(cfg) if cfg.phase=='g0' else g1(cfg)
if __name__=='__main__':main()
