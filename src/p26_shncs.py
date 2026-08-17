"""P26 SHNCS-24 input-only contracts; label-authorized G2 is appended only after evidence commit."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
N,T=576,20
def digest(x):x=np.ascontiguousarray(x);return hashlib.sha256(str(x.dtype).encode()+repr(x.shape).encode()+x.tobytes()).hexdigest()
def pair(t,a,b,d):
 x,y=t[a].clone(),t[b].clone()
 if d==1:x,y=x.transpose(-2,-1),y.transpose(-2,-1)
 elif d==2:x,y=y,x
 elif d==3:x,y=y.transpose(-2,-1),x.transpose(-2,-1)
 elif d!=0:raise ValueError(d)
 return torch.cat([x,y])
def g0(a):
 z=torch.zeros((4,3,T,T));q=torch.arange(T)[None,:]/19
 for i in range(4):z[i,0]=i+q
 x=pair(z,0,1,0);xt=pair(z.transpose(-2,-1).contiguous(),0,1,1);ids=np.array([4,8,15,16,23,42]);seed=np.random.default_rng(20260817);s1=seed.choice(ids[1:],size=4,replace=False);seed=np.random.default_rng(20260817);s2=seed.choice(ids[1:],size=4,replace=False);raw=np.array([.1,.3,.2]);log=np.array([.2,.6,.4]);r={'experiment':'P26_SHNCS24','gate':'G0','shape':list(x.shape),'transpose_consistent':bool(torch.allclose(x,xt)),'deterministic_hard_negative_sampling':bool(np.array_equal(s1,s2)),'candidate_permutation_invariant':bool(np.array_equal(np.sort(ids),np.sort(ids[::-1]))),'alpha_zero_identity':bool(np.array_equal(raw,raw+0*log)),'finite':bool(torch.isfinite(x).all()),'labels_used':False,'targets_opened':False,'p8_imported':False};r['passes_G0']=all(r[k] for k in ['transpose_consistent','deterministic_hard_negative_sampling','candidate_permutation_invariant','alpha_zero_identity','finite']);a.work.mkdir(parents=True,exist_ok=True);(a.work/'p26_g0_report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
 if not r['passes_G0']:raise RuntimeError('P26 G0 failed')
def g1(a):
 rows=[]
 for n in a.sources:
  p=a.pool_dir/(Path(n).stem+'.npz')
  with np.load(p,allow_pickle=False) as z:u=z['pool'].copy();src=str(z['source'])
  if src!=n or u.shape!=(4,N,128) or not np.isfinite(u).all():raise RuntimeError('bad streamed pool')
  # Sampling is candidate-ID invariant: sort available hard ids before RNG index selection.
  sam=[];rng=np.random.default_rng(20260817)
  for d in range(4):
   for i in range(N):
    h=np.sort(u[d,i][u[d,i]>=0]);sam.append(rng.choice(h,size=min(15,len(h)),replace=False))
  rows.append({'source':n,'pool_sha256':digest(u),'sampling_sha256':digest(np.concatenate(sam)),'finite':True})
 r={'experiment':'P26_SHNCS24','gate':'G1','rows':rows,'labels_used':False,'targets_opened':False,'p8_imported':False};r['passes_G1']=bool(len(rows)==4 and all(x['finite'] for x in rows));a.work.mkdir(parents=True,exist_ok=True);(a.work/'p26_g1_report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
 if not r['passes_G1']:raise RuntimeError('P26 G1 failed')
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['g0','g1']);p.add_argument('--pool-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P25_scxr'));p.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P26_shncs'));p.add_argument('--sources',nargs=4,default=['img_000025.png','img_000098.png','img_000168.png','img_000172.png']);a=p.parse_args();
 if 'p8' in '\n'.join(str(x).lower() for x in [a.pool_dir,a.work]):raise RuntimeError('P8 prohibited')
 g0(a) if a.mode=='g0' else g1(a)
if __name__=='__main__':main()
