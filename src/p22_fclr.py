"""P22 FCLR-24 input-only contracts. See committed P22_PRE_REGISTRATION.md."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
GRID,TILE,N=24,20,576

def sha(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+repr(a.shape).encode()+a.tobytes()).hexdigest()
def safe(*p):
 if 'p8' in '\n'.join(str(x).lower() for x in p):raise RuntimeError('P8 prohibited')
def tiles(root,name):
 x=np.asarray(Image.open(root/name).convert('RGB'),np.float32)/255.
 if x.shape!=(480,480,3):raise RuntimeError(x.shape)
 return torch.from_numpy(x.reshape(GRID,TILE,GRID,TILE,3).transpose(0,2,4,1,3).reshape(N,3,TILE,TILE).copy())
def bands(t,anchors,candidates,d):
 out=[]
 for a,b in zip(anchors,candidates):
  left,right=t[int(a)],t[int(b)]
  if d==1:left,right=left.transpose(-2,-1),right.transpose(-2,-1)
  elif d==2:left,right=right,left
  elif d==3:left,right=right.transpose(-2,-1),left.transpose(-2,-1)
  elif d!=0:raise ValueError(d)
  out.append(torch.cat([left[:,:,-10:],right[:,:,:10]],dim=-1))
 return torch.stack(out)
def g0(a):
 z=torch.zeros((4,3,TILE,TILE),dtype=torch.float32);q=torch.arange(TILE,dtype=torch.float32)[None,:]/19
 for i in range(4):z[i,0]=i+q
 x=bands(z,[0],[1],0);xt=bands(z.transpose(-2,-1).contiguous(),[0],[1],1);base=torch.tensor([1,2,3]);order=torch.tensor([2,0,1]);xp=bands(z,[0,0,0],base[order].tolist(),0);xi=bands(z,[0,0,0],base.tolist(),0);raw=np.array([[.2,.4,.6],[.1,.3,.5]],np.float32);logit=np.array([[.7,.2,.9],[.5,.8,.1]],np.float32);f=lambda alpha:raw+alpha*logit
 r={'experiment':'P22_FCLR24','gate':'G0','shape':list(x.shape),'transpose_consistent':bool(torch.allclose(x,xt)),'candidate_row_invariant':bool(torch.allclose(xp[torch.argsort(order)],xi)),'alpha_zero_identity':bool(np.array_equal(f(0),raw)),'finite':bool(torch.isfinite(x).all() and np.isfinite(f(.2)).all()),'labels_used':False,'targets_opened':False,'p8_imported':False};r['passes_G0']=all(r[k] for k in ['transpose_consistent','candidate_row_invariant','alpha_zero_identity','finite']);a.work.mkdir(parents=True,exist_ok=True);(a.work/'p22_g0_report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
 if not r['passes_G0']:raise RuntimeError('P22 G0 failed')
def g1(a):
 sys.path.insert(0,str(Path(__file__).resolve().parent));import p13_component_pose as p13
 fit,_=p13.source_lists(a.manifest);rows=[]
 for name in sorted(fit)[:4]:
  t=tiles(a.inputs,name);c,v,s=p13.load_score_cache(a.score_dir,name);parts=[]
  for d in range(4):
   for i in range(N):
    slots=np.flatnonzero(v[i])[:16]
    if slots.size:parts.append(bands(t,[i]*len(slots),c[i,slots].tolist(),d).numpy())
  z=np.concatenate(parts);rows.append({'source':name,'count':int(len(z)),'band_sha256':sha(z),'finite':bool(np.isfinite(z).all())})
 r={'experiment':'P22_FCLR24','gate':'G1','rows':rows,'labels_used':False,'targets_opened':False,'p8_imported':False};r['passes_G1']=bool(all(x['count']>0 and x['finite'] for x in rows));a.work.mkdir(parents=True,exist_ok=True);(a.work/'p22_g1_report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
 if not r['passes_G1']:raise RuntimeError('P22 G1 failed')
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['g0','g1']);p.add_argument('--inputs',type=Path,default=Path(r'E:\pazzle_data\train\inputs'));p.add_argument('--score-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache'));p.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));p.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P22_fclr'));a=p.parse_args();safe(a.inputs,a.score_dir,a.manifest,a.work);g0(a) if a.mode=='g0' else g1(a)
if __name__=='__main__':main()
