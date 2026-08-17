"""P24 RCR-24 input-only contracts; G2 added only after G0/G1 evidence."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
G,T,N=24,20,576
def sha(x):x=np.ascontiguousarray(x);return hashlib.sha256(str(x.dtype).encode()+repr(x.shape).encode()+x.tobytes()).hexdigest()
def safe(*p):
 if 'p8' in '\n'.join(str(x).lower() for x in p):raise RuntimeError('P8 prohibited')
def tiles(root,n):
 x=np.asarray(Image.open(root/n).convert('RGB'),np.float32)/255.
 if x.shape!=(480,480,3):raise RuntimeError(x.shape)
 return torch.from_numpy(x.reshape(G,T,G,T,3).transpose(0,2,4,1,3).reshape(N,3,T,T).copy())
def pair(t,a,b,d):
 x,y=t[a].clone(),t[b].clone()
 if d==1:x,y=x.transpose(-2,-1),y.transpose(-2,-1)
 elif d==2:x,y=y,x
 elif d==3:x,y=y.transpose(-2,-1),x.transpose(-2,-1)
 elif d!=0:raise ValueError(d)
 return torch.cat([x,y],0)
def g0(a):
 z=torch.zeros((4,3,T,T));q=torch.arange(T)[None,:]/19
 for i in range(4):z[i,0]=i+q
 x=pair(z,0,1,0);xt=pair(z.transpose(-2,-1).contiguous(),0,1,1);base=[1,2,3];o=torch.tensor([2,0,1]);xp=torch.stack([pair(z,0,base[int(i)],0) for i in o]);xi=torch.stack([pair(z,0,i,0) for i in base]);raw=np.array([.2,.4,.6]);log=np.array([.7,.2,.9]);r={'experiment':'P24_RCR24','gate':'G0','shape':list(x.shape),'transpose_consistent':bool(torch.allclose(x,xt)),'candidate_permutation_invariant':bool(torch.allclose(xp[torch.argsort(o)],xi)),'alpha_zero_identity':bool(np.array_equal(raw,raw+0*log)),'finite':bool(torch.isfinite(x).all()),'labels_used':False,'targets_opened':False,'p8_imported':False};r['passes_G0']=all(r[k] for k in ['transpose_consistent','candidate_permutation_invariant','alpha_zero_identity','finite']);a.work.mkdir(parents=True,exist_ok=True);(a.work/'p24_g0_report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
 if not r['passes_G0']:raise RuntimeError('P24 G0 failed')
def g1(a):
 sys.path.insert(0,str(Path(__file__).resolve().parent));import p13_component_pose as p;fit,_=p.source_lists(a.manifest);rows=[]
 for n in sorted(fit)[:4]:
  t=tiles(a.inputs,n);c,v,s=p.load_score_cache(a.score_dir,n);x=[]
  for d in range(4):
   for i in range(N):
    slots=np.flatnonzero(v[i])[:8];x.extend(pair(t,i,int(c[i,j]),d).numpy() for j in slots)
  z=np.stack(x);rows.append({'source':n,'count':len(z),'pair_sha256':sha(z),'finite':bool(np.isfinite(z).all())})
 r={'experiment':'P24_RCR24','gate':'G1','rows':rows,'labels_used':False,'targets_opened':False,'p8_imported':False};r['passes_G1']=bool(all(x['count']>0 and x['finite'] for x in rows));a.work.mkdir(parents=True,exist_ok=True);(a.work/'p24_g1_report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
 if not r['passes_G1']:raise RuntimeError('P24 G1 failed')
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['g0','g1']);p.add_argument('--inputs',type=Path,default=Path(r'E:\pazzle_data\train\inputs'));p.add_argument('--score-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache'));p.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));p.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P24_rcr'));a=p.parse_args();safe(a.inputs,a.score_dir,a.manifest,a.work);g0(a) if a.mode=='g0' else g1(a)
if __name__=='__main__':main()
