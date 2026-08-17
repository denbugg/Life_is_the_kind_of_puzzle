"""P31 BHCS-24: seam-only boundary hard-contrastive scorer, G0/G1 harness."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parent))
from p29_dpcg import N,load_tiles
from p12_loop_consensus import solve_buddies_from_scores

class SeamCNN(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.GELU(),nn.Conv2d(32,32,3,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(32,1))
 def forward(self,x):return self.net(x).squeeze(1)
def bad(*x):
 if 'p8' in '\n'.join(map(lambda z:str(z).lower(),x)):raise RuntimeError('P8 prohibited')
def seam(t,i,j,d):
 # canonical 3x20x8: source boundary then target opposite boundary.
 if d==1:return seam(t.transpose(-2,-1).contiguous(),i,j,0)
 if d==2:return seam(t,j,i,0)
 if d==3:return seam(t.transpose(-2,-1).contiguous(),j,i,0)
 if d!=0:raise ValueError(d)
 return torch.cat((t[i,:,:,-4:],t[j,:,:,:4]),dim=-1)
def g0(a):
 t=torch.arange(4*3*20*20,dtype=torch.float32).reshape(4,3,20,20);x=seam(t,0,1,0);y=seam(t.transpose(-2,-1).contiguous(),0,1,1)
 raw=np.full((2,N,N),-20,np.float32)
 for r in range(24):
  for c in range(24):
   i=r*24+c
   if c<23:raw[0,i,i+1]=20
   if r<23:raw[1,i,i+24]=20
 for k in range(2):np.fill_diagonal(raw[k],-np.inf)
 p,o=solve_buddies_from_scores(raw[0],raw[1],max_edges=2208,min_margin=0,repair_passes=2);p=np.asarray(p,np.int32)
 return {'experiment':'P31_BHCS24','gate':'G0','seam_shape':list(x.shape),'vertical_shape':list(y.shape),'valid_bijection':bool(p.shape==(N,) and np.unique(p).size==N),'exact_synthetic':bool(np.array_equal(p,np.arange(N,dtype=np.int32))),'finite_seam':bool(torch.isfinite(x).all() and torch.isfinite(y).all())}
def g1(a):
 dev=torch.device('cuda');m=SeamCNN().to(dev).eval();rows=[]
 with torch.no_grad():
  for k,n in enumerate(a.sources):
   st=time.perf_counter();t=load_tiles(a.inputs,n).to(dev);a0=torch.arange(16,device=dev);b0=(a0+1)%N;x=torch.stack([seam(t,int(i),int(j),0) for i,j in zip(a0,b0)]);z=m(x);dt=time.perf_counter()-st
   ok=bool(x.shape==(16,3,20,8) and torch.isfinite(z).all() and dt<=90)
   rows.append({'source':n,'seconds':dt,'shape':list(x.shape),'ok':ok})
   if (k+1)%4==0:print(json.dumps({'stage':'g1','done':k+1,'total':len(a.sources)}),flush=True)
 return {'experiment':'P31_BHCS24','gate':'G1','rows':rows,'labels_used':False,'targets_opened':False,'p8_imported':False,'passes_G1':bool(all(r['ok'] for r in rows))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('g0','g1'),required=True);p.add_argument('--inputs',type=Path,default=Path(r'E:\pazzle_data\train\inputs'));p.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P31_bhcs'));p.add_argument('--sources',nargs='*',default=('img_000002.png','img_000025.png','img_000098.png','img_000168.png','img_000172.png','img_000194.png','img_000223.png','img_000243.png','img_000267.png','img_000304.png','img_000344.png','img_000384.png','img_000426.png','img_000457.png','img_000480.png','img_000513.png'));a=p.parse_args();bad(a.inputs,a.work);a.work.mkdir(parents=True,exist_ok=True);r=g0(a) if a.mode=='g0' else g1(a);r['passes_G0']=r.get('exact_synthetic',False) and r.get('valid_bijection',False) and r.get('finite_seam',False) if a.mode=='g0' else r.get('passes_G1');(a.work/f'p31_{a.mode}_report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
 if not r[f'passes_{a.mode.upper()}']:raise RuntimeError(f'P31 {a.mode} rejected')
if __name__=='__main__':main()
