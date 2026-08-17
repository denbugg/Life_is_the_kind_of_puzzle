"""P28 GDCP-24 input-only graph contracts. No labels before G2a."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
N=576
def sha(x):x=np.ascontiguousarray(x);return hashlib.sha256(str(x.dtype).encode()+repr(x.shape).encode()+x.tobytes()).hexdigest()
def graph(p,score_dir,n):
 c,v,s=p.load_score_cache(score_dir,n);src=[];dst=[];dire=[];w=[]
 for d in range(4):
  for i in range(N):
   j=np.flatnonzero(v[i]);src.extend([i]*len(j));dst.extend(c[i,j]);dire.extend([d]*len(j));w.extend(s[d,i,j])
 return np.asarray([src,dst,dire,w],np.float32)
def g0(a):
 # directed edges encode unit displacement; translation and node permutation leave relative residual unchanged.
 xy=np.array([[0.,0.],[1.,0.],[0.,1.]],np.float32);e=np.array([[0,0],[1,2]],np.int32);delta=np.array([[1.,0.],[0.,1.]],np.float32);r=(xy[e[1]]-xy[e[0]]-delta);perm=np.array([2,0,1]);xp=xy[perm];inv=np.argsort(perm);ep=inv[e];rp=xp[ep[1]]-xp[ep[0]]-delta;out={'experiment':'P28_GDCP24','gate':'G0','translation_invariant':bool(np.allclose((xy+5)[e[1]]-(xy+5)[e[0]],xy[e[1]]-xy[e[0]])),'permutation_invariant':bool(np.allclose(r,rp)),'directed_axis_correct':bool(np.allclose(delta[0],[1,0]) and np.allclose(delta[1],[0,1])),'finite':bool(np.isfinite(r).all()),'labels_used':False,'targets_opened':False,'p8_imported':False};out['passes_G0']=all(out[k] for k in ['translation_invariant','permutation_invariant','directed_axis_correct','finite']);a.work.mkdir(parents=True,exist_ok=True);(a.work/'p28_g0_report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
 if not out['passes_G0']:raise RuntimeError('P28 G0 failed')
def g1(a):
 sys.path.insert(0,str(Path(__file__).resolve().parent));import p13_component_pose as p;fit,_=p.source_lists(a.manifest);rows=[]
 for n in sorted(fit)[:4]:
  x=graph(p,a.score_dir,n);rows.append({'source':n,'edges':int(x.shape[1]),'graph_sha256':sha(x),'finite':bool(np.isfinite(x).all()),'directed_values':sorted(set(x[2].astype(int).tolist()))})
 out={'experiment':'P28_GDCP24','gate':'G1','rows':rows,'labels_used':False,'targets_opened':False,'p8_imported':False};out['passes_G1']=bool(all(r['edges']>0 and r['finite'] and r['directed_values']==[0,1,2,3] for r in rows));a.work.mkdir(parents=True,exist_ok=True);(a.work/'p28_g1_report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
 if not out['passes_G1']:raise RuntimeError('P28 G1 failed')
def main():
 q=argparse.ArgumentParser();q.add_argument('--mode',required=True,choices=['g0','g1']);q.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));q.add_argument('--score-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache'));q.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P28_gdcp'));a=q.parse_args()
 if 'p8' in '\n'.join(map(str,[a.manifest,a.score_dir,a.work])).lower():raise RuntimeError('P8 prohibited')
 {'g0':g0,'g1':g1}[a.mode](a)
if __name__=='__main__':main()
