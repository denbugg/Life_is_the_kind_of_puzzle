"""P20 DDCC-24 input-only contracts. P20_PRE_REGISTRATION.md was published first."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
SRC=Path(__file__).resolve().parent
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
import p13_component_pose as p13
GRID,TILE,N=24,20,576

def sha(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+repr(a.shape).encode()+a.tobytes()).hexdigest()
def safe(*paths):
 if 'p8' in '\n'.join(map(lambda x:str(x).lower(),paths)):raise RuntimeError('P8 prohibited')
def board(root,name):
 x=np.asarray(Image.open(root/name).convert('RGB'),np.float32)/255.
 if x.shape!=(480,480,3):raise RuntimeError(x.shape)
 return x.reshape(GRID,TILE,GRID,TILE,3).transpose(0,2,1,3,4).reshape(N,TILE,TILE,3)
def feats(a,b,axis):
 # Inputs B,20,20,3. axis=0 (right) or 1 (down). Return B,3 analytic features.
 if axis==0:
  av,bv=a[:,-1,:],b[:,0,:];at,bt=a[:,-1,:]-a[:,-2,:],b[:,1,:]-b[:,0,:];an,bn=a[:,-1,:]-a[:,-2,:],b[:,1,:]-b[:,0,:]
 else:
  av,bv=a[-1,:,:],b[0,:,:];at,bt=a[-1,:,:]-a[-2,:,:],b[1,:,:]-b[0,:,:];an,bn=a[-1,:,:]-a[-2,:,:],b[1,:,:]-b[0,:,:]
 value=np.abs(av-bv).mean();tangent=np.abs(np.diff(av,axis=0)-np.diff(bv,axis=0)).mean();normal=np.abs(an-bn).mean()
 return np.array([value,tangent,normal],np.float32)
def direction_features(tiles,anchor,candidate,direction):return feats(tiles[anchor],tiles[candidate],0 if direction in (0,2) else 1)
def g0(args):
 # smooth horizontal gradient: true right seam is less discontinuous than a reversed non-neighbor; vertical transpose is equivalent.
 tile=np.zeros((20,20,3),np.float32);tile[:,:,0]=np.arange(20)[None,:]/19
 nxt=tile.copy();nxt[:,:,0]+=20/19; wrong=np.flip(nxt,axis=1).copy()
 ftrue=feats(tile,nxt,0);fwrong=feats(tile,wrong,0);fv=feats(np.transpose(tile,(1,0,2)),np.transpose(nxt,(1,0,2)),1)
 cand=np.array([[2,5,7],[5,2,7]],np.int64);score=np.array([[.1,.2,.3],[.4,.5,.6]],np.float32);aux=np.array([.3,.1,.2,.7,.5,.4,.9,.6],np.float32)
 order=np.array([2,0,1]);add=lambda c,s,a:s+a*aux[c];invariant=np.allclose(add(cand[:,order],score[:,order],.2)[:,np.argsort(order)],add(cand,score,.2))
 r={'experiment':'P20_DDCC24','gate':'G0_analytic_features','true_less_discontinuous':bool(ftrue[0]<fwrong[0]),'transpose_consistent':bool(np.allclose(ftrue,fv,atol=1e-6)),'alpha_zero_identity':bool(np.array_equal(add(cand,score,0),score)),'candidate_id_invariant':bool(invariant),'finite':bool(np.isfinite(np.r_[ftrue,fwrong,fv]).all()),'labels_used':False,'targets_opened':False,'p8_imported':False}
 r['passes_G0']=all([r[k] for k in ['true_less_discontinuous','transpose_consistent','alpha_zero_identity','candidate_id_invariant','finite']]);args.work.mkdir(parents=True,exist_ok=True);(args.work/'p20_g0_report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
 if not r['passes_G0']:raise RuntimeError('P20 G0 failed')
def g1(args):
 tr,_=p13.source_lists(args.manifest); names=sorted(tr)[:4];rows=[]
 for name in names:
  t=board(args.inputs,name);c,v,s=p13.load_score_cache(args.score_dir,name);parts=[]
  for d in range(4):
   for i in range(N):
    valid_slots=np.flatnonzero(v[i]); ordered=valid_slots[np.argsort(-s[d,i,valid_slots],kind='stable')]
    rank_by_slot={int(slot):float(j/max(1,ordered.size-1)) for j,slot in enumerate(ordered)}
    for slot in valid_slots[:16]:
     analytic=direction_features(t,i,int(c[i,slot]),d)
     parts.append(np.r_[s[d,i,slot],analytic,rank_by_slot[int(slot)]].astype(np.float32))
  z=np.stack(parts);rows.append({'source':name,'count':int(len(z)),'feature_sha256':sha(z),'finite':bool(np.isfinite(z).all()),'mean':z.mean(0).tolist(),'std':z.std(0).tolist()})
 r={'experiment':'P20_DDCC24','gate':'G1_input_score_features','rows':rows,'labels_used':False,'targets_opened':False,'p8_imported':False};r['passes_G1']=bool(all(x['finite'] and x['count']>0 for x in rows));args.work.mkdir(parents=True,exist_ok=True);(args.work/'p20_g1_report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
 if not r['passes_G1']:raise RuntimeError('P20 G1 failed')
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['g0','g1']);p.add_argument('--inputs',type=Path,default=Path(r'E:\pazzle_data\train\inputs'));p.add_argument('--score-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache'));p.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));p.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P20_ddcc'));a=p.parse_args();safe(a.inputs,a.score_dir,a.manifest,a.work);g0(a) if a.mode=='g0' else g1(a)
if __name__=='__main__':main()
