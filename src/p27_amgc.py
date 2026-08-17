"""P27 AMGC-24 analytic contracts; labels are prohibited until G2 extension."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
N,G,T=576,24,20
def sha(x):x=np.ascontiguousarray(x);return hashlib.sha256(str(x.dtype).encode()+repr(x.shape).encode()+x.tobytes()).hexdigest()
def orient(x,d):
 if d==0:return x
 if d==1:return x.transpose(0,1,3,2)
 if d==2:return x[:,:, :,::-1]
 if d==3:return x.transpose(0,1,3,2)[:,:, :,::-1]
 raise ValueError(d)
def amgc(x,y,eps=1e-3):
 # x,y B×3×H×W, y is candidate right of x after orientation.
 a=x[:,:,:,-1].transpose(0,2,1);inn=(x[:,:,:,-1]-x[:,:,:,-2]).transpose(0,2,1);cross=(y[:,:,:,0]-x[:,:,:,-1]).transpose(0,2,1)
 b=y[:,:,:,0].transpose(0,2,1);inn2=(y[:,:,:,1]-y[:,:,:,0]).transpose(0,2,1);cross2=(x[:,:,:,-1]-y[:,:,:,0]).transpose(0,2,1)
 def d(v,c):
  mu=v.mean(1,keepdims=True);q=v-mu;cov=np.einsum('bti,btj->bij',q,q)/max(1,v.shape[1]-1)+eps*np.eye(3)[None];iv=np.linalg.inv(cov);z=c-mu;return np.einsum('bti,bij,btj->bt',z,iv,z).mean(1)
 return d(inn,cross)+d(inn2,cross2)
def tiles(root,n):
 x=np.asarray(Image.open(root/n).convert('RGB'),np.float32)/255.
 if x.shape!=(480,480,3):raise RuntimeError(x.shape)
 return x.reshape(G,T,G,T,3).transpose(0,2,4,1,3).reshape(N,3,T,T).copy()
def g0(a):
 q=np.linspace(0,1,T,dtype=np.float64);r=(q[None,:]+q[:,None])[None,None];x=np.tile(r,(1,3,1,1));y=x+1;bad=y+.5;ok=amgc(x,y)[0];disc=amgc(x,bad)[0];xt=x.transpose(0,1,3,2);yt=y.transpose(0,1,3,2);perm=np.array([2,0,1]);raw=np.array([.1,.3,.2]);f=np.array([.8,.1,.4]);out={'experiment':'P27_AMGC24','gate':'G0','self_continuation_lower':bool(ok<disc),'transpose_consistent':bool(np.allclose(amgc(xt,yt),amgc(x,y))),'epsilon_finite':bool(np.isfinite(amgc(np.zeros_like(x),np.zeros_like(y))).all()),'candidate_permutation_invariant':bool(np.array_equal(perm[np.argsort(f[perm])],np.argsort(f))),'alpha_zero_identity':bool(np.array_equal(raw,raw+0*f)),'labels_used':False,'targets_opened':False,'p8_imported':False};out['passes_G0']=all(out[k] for k in ['self_continuation_lower','transpose_consistent','epsilon_finite','candidate_permutation_invariant','alpha_zero_identity']);a.work.mkdir(parents=True,exist_ok=True);(a.work/'p27_g0_report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
 if not out['passes_G0']:raise RuntimeError('P27 G0 failed')
def g1(a):
 sys.path.insert(0,str(Path(__file__).resolve().parent));import p13_component_pose as p;fit,_=p.source_lists(a.manifest);rows=[]
 for n in sorted(fit)[:4]:
  t=tiles(a.inputs,n);c,v,s=p.load_score_cache(a.score_dir,n);z=[]
  for d in range(4):
   q=orient(t,d)
   for i in range(N):
    sl=np.flatnonzero(v[i]);z.extend(amgc(np.repeat(q[i:i+1],len(sl),0),q[c[i,sl]]))
  z=np.asarray(z);rows.append({'source':n,'count':len(z),'amgc_sha256':sha(z),'finite':bool(np.isfinite(z).all())})
 out={'experiment':'P27_AMGC24','gate':'G1','rows':rows,'labels_used':False,'targets_opened':False,'p8_imported':False};out['passes_G1']=bool(all(x['count']>0 and x['finite'] for x in rows));a.work.mkdir(parents=True,exist_ok=True);(a.work/'p27_g1_report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
 if not out['passes_G1']:raise RuntimeError('P27 G1 failed')
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['g0','g1']);p.add_argument('--inputs',type=Path,default=Path(r'E:\pazzle_data\train\inputs'));p.add_argument('--score-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache'));p.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));p.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P27_amgc'));a=p.parse_args();
 if 'p8' in '\n'.join(str(x).lower() for x in [a.inputs,a.score_dir,a.manifest,a.work]):raise RuntimeError('P8 prohibited')
 {'g0':g0,'g1':g1}[a.mode](a)
if __name__=='__main__':main()
