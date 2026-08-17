"""P21 GBLS-24 input-only bridge contracts. See committed P21_PRE_REGISTRATION.md."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
GRID,TILE,N,W=24,20,576,128

def sha(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+repr(a.shape).encode()+a.tobytes()).hexdigest()
def safe(*paths):
 if 'p8' in '\n'.join(str(x).lower() for x in paths):raise RuntimeError('P8 prohibited')
def tiles(root,name):
 x=np.asarray(Image.open(root/name).convert('RGB'),np.float32)/255.
 if x.shape!=(480,480,3):raise RuntimeError(f'bad board shape {x.shape}')
 return torch.from_numpy(x.reshape(GRID,TILE,GRID,TILE,3).transpose(0,2,4,1,3).reshape(N,3,TILE,TILE).copy())
def orient(t,a,b,d):
 """Return directed pair in physical left-to-right orientation, BxCxHxW."""
 left,right=t[a],t[b]
 if d==0: pass
 elif d==1: left,right=left.transpose(-2,-1),right.transpose(-2,-1)
 elif d==2: left,right=right,left
 elif d==3: left,right=right.transpose(-2,-1),left.transpose(-2,-1)
 else:raise ValueError(d)
 return left,right
def bridge(t,anchors,candidates,d):
 """Masked input (B,3,20,16) and withheld 4-column bridge (B,3,20,4)."""
 out=[];truth=[]
 for a,b in zip(anchors,candidates):
  left,right=orient(t,int(a),int(b),d)
  # Last/first two true boundary columns are withheld. Context never includes them.
  x=torch.cat([left[:,:,-8:-2],torch.zeros((3,TILE,4),dtype=left.dtype),right[:,:,2:8]],dim=-1)
  y=torch.cat([left[:,:,-2:],right[:,:,:2]],dim=-1)
  out.append(x);truth.append(y)
 return torch.stack(out),torch.stack(truth)
def g0(a):
 # Synthetic tile: boundary content and axis transpose are exactly traceable.
 z=torch.zeros((3,3,TILE,TILE),dtype=torch.float32);r=torch.arange(TILE,dtype=torch.float32)[None,:]/19
 z[0,0]=r;z[1,0]=1+r;z[2,0]=2+r
 x,y=bridge(z,[0],[1],0);xt,yt=bridge(z.transpose(-2,-1).contiguous(),[0],[1],1)
 order=torch.tensor([2,0,1]);xp,yp=bridge(z,[0,0,0],order.tolist(),0);xi,yi=bridge(z,[0,0,0],torch.arange(3).tolist(),0);inv=torch.argsort(order)
 cand=np.array([[1,2,3],[3,2,1]],np.int32);raw=np.array([[.2,.4,.6],[.1,.3,.5]],np.float32);res=np.array([[.7,.2,.9],[.5,.8,.1]],np.float32);fuse=lambda alpha:raw-alpha*res
 report={'experiment':'P21_GBLS24','gate':'G0','shape':list(x.shape),'mask_is_zero':bool(torch.count_nonzero(x[:,:,: ,6:10])==0),'target_alignment':bool(torch.allclose(y[:,:,:,:2],z[0:1,:,:,-2:]) and torch.allclose(y[:,:,:,2:],z[1:2,:,:,:2])),'transpose_consistent':bool(torch.allclose(x,xt) and torch.allclose(y,yt)),'candidate_row_invariant':bool(torch.allclose(xp[inv],xi) and torch.allclose(yp[inv],yi)),'alpha_zero_identity':bool(np.array_equal(fuse(0),raw)),'finite':bool(torch.isfinite(x).all() and torch.isfinite(y).all() and np.isfinite(fuse(.2)).all()),'labels_used':False,'targets_opened':False,'p8_imported':False}
 report['passes_G0']=all(report[k] for k in ['mask_is_zero','target_alignment','transpose_consistent','candidate_row_invariant','alpha_zero_identity','finite']);a.work.mkdir(parents=True,exist_ok=True);(a.work/'p21_g0_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
 if not report['passes_G0']:raise RuntimeError('P21 G0 failed')
def g1(a):
 # Input-only: score cache gives frozen candidate IDs, but neither label cache nor target PNG is opened.
 import sys
 sys.path.insert(0,str(Path(__file__).resolve().parent))
 import p13_component_pose as p13
 fit,_=p13.source_lists(a.manifest);rows=[]
 for name in sorted(fit)[:4]:
  t=tiles(a.inputs,name);c,v,s=p13.load_score_cache(a.score_dir,name);xx=[];yy=[]
  for d in range(4):
   for i in range(N):
    slots=np.flatnonzero(v[i])[:16]
    if slots.size:
     x,y=bridge(t,[i]*len(slots),c[i,slots].tolist(),d);xx.append(x.numpy());yy.append(y.numpy())
  x=np.concatenate(xx);y=np.concatenate(yy);rows.append({'source':name,'count':int(len(x)),'input_sha256':sha(x),'target_tensor_sha256':sha(y),'finite':bool(np.isfinite(x).all() and np.isfinite(y).all()),'masked_zero':bool(np.count_nonzero(x[:,:,:,6:10])==0)})
 report={'experiment':'P21_GBLS24','gate':'G1','rows':rows,'labels_used':False,'targets_opened':False,'p8_imported':False};report['passes_G1']=bool(all(r['count']>0 and r['finite'] and r['masked_zero'] for r in rows));a.work.mkdir(parents=True,exist_ok=True);(a.work/'p21_g1_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
 if not report['passes_G1']:raise RuntimeError('P21 G1 failed')
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['g0','g1']);p.add_argument('--inputs',type=Path,default=Path(r'E:\pazzle_data\train\inputs'));p.add_argument('--score-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache'));p.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));p.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P21_gbls'));a=p.parse_args();safe(a.inputs,a.score_dir,a.manifest,a.work);g0(a) if a.mode=='g0' else g1(a)
if __name__=='__main__':main()
