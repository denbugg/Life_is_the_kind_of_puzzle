"""P23 DCTR-24 input-only directional retrieval contracts."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
GRID=TILE=24; TILE=20; N=576

def sha(a):a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+repr(a.shape).encode()+a.tobytes()).hexdigest()
def safe(*p):
 if 'p8' in '\n'.join(map(lambda x:str(x).lower(),p)):raise RuntimeError('P8 prohibited')
def split(root,n):
 x=np.asarray(Image.open(root/n).convert('RGB'),np.float32)/255.
 if x.shape!=(480,480,3):raise RuntimeError(x.shape)
 return torch.from_numpy(x.reshape(24,20,24,20,3).transpose(0,2,4,1,3).reshape(N,3,20,20).copy())
def roles(t,ids,d,source=True):
 z=t[ids]
 # source right/down/left/up role has a deterministic direction transform; candidate gets complementary role.
 rot=(d if source else (d+2)%4)
 if rot in (1,3):z=z.transpose(-2,-1)
 if rot in (2,3):z=torch.flip(z,[-1])
 return z
def g0(a):
 z=torch.zeros((5,3,20,20));q=torch.arange(20)[None,:]/19
 for i in range(5):z[i,0]=i+q
 s=roles(z,[0],0,True);c=roles(z,[1],0,False);st=roles(z.transpose(-2,-1).contiguous(),[0],1,True);ct=roles(z.transpose(-2,-1).contiguous(),[1],1,False);ids=torch.tensor([3,1,4]);order=torch.tensor([2,0,1]);r={'experiment':'P23_DCTR24','gate':'G0','source_candidate_shapes':list(s.shape)+list(c.shape),'transpose_consistent':bool(torch.allclose(s,st) and torch.allclose(c,ct)),'candidate_permutation_invariant':bool(torch.allclose(roles(z,ids[order],0,False)[torch.argsort(order)],roles(z,ids,0,False))),'self_exclusion_contract':bool(not bool(torch.any(ids==0))),'finite':bool(torch.isfinite(torch.cat([s,c])).all()),'labels_used':False,'targets_opened':False,'p8_imported':False};r['passes_G0']=all(r[k] for k in ['transpose_consistent','candidate_permutation_invariant','self_exclusion_contract','finite']);a.work.mkdir(parents=True,exist_ok=True);(a.work/'p23_g0_report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
 if not r['passes_G0']:raise RuntimeError('P23 G0 failed')
def g1(a):
 sys.path.insert(0,str(Path(__file__).resolve().parent));import p13_component_pose as p13;fit,_=p13.source_lists(a.manifest);rows=[]
 for n in sorted(fit)[:4]:
  t=split(a.inputs,n);c,v,s=p13.load_score_cache(a.score_dir,n);x=[]
  for d in range(4):
   x.extend([roles(t,torch.arange(N),d,True).numpy(),roles(t,torch.tensor(c[:,0]),d,False).numpy()])
  z=np.concatenate(x);rows.append({'source':n,'count':int(len(z)),'tensor_sha256':sha(z),'finite':bool(np.isfinite(z).all())})
 r={'experiment':'P23_DCTR24','gate':'G1','rows':rows,'labels_used':False,'targets_opened':False,'p8_imported':False};r['passes_G1']=bool(all(x['count']>0 and x['finite'] for x in rows));a.work.mkdir(parents=True,exist_ok=True);(a.work/'p23_g1_report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
 if not r['passes_G1']:raise RuntimeError('P23 G1 failed')
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['g0','g1']);p.add_argument('--inputs',type=Path,default=Path(r'E:\pazzle_data\train\inputs'));p.add_argument('--score-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache'));p.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));p.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P23_dctr'));a=p.parse_args();safe(a.inputs,a.score_dir,a.manifest,a.work);g0(a) if a.mode=='g0' else g1(a)
if __name__=='__main__':main()
