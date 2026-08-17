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
# Legacy entrypoint deferred until the G2 extension below.

# G2 extension authorized only after the committed G0/G1 evidence.
class Enc(torch.nn.Module):
 def __init__(self):
  super().__init__();self.c1=torch.nn.Conv2d(3,48,3,padding=1);self.c2=torch.nn.Conv2d(48,64,3,padding=1);self.o=torch.nn.Linear(64,48)
 def forward(self,x):
  z=torch.nn.functional.silu(self.c1(x));z=torch.nn.functional.silu(self.c2(z));return torch.nn.functional.normalize(self.o(z.mean((-2,-1))),dim=-1)
def p13():
 sys.path.insert(0,str(Path(__file__).resolve().parent));import p13_component_pose as x;return x
def labs(ld,n):
 with np.load(ld/(Path(n).stem+'.npz'),allow_pickle=False) as z:pos=z['target_tile_to_slot'].astype(np.int32,copy=True);src=str(z['source'])
 if src!=n or np.unique(pos).size!=N or pos.min()!=0 or pos.max()!=N-1:raise RuntimeError('bad label cache')
 inv=np.empty(N,np.int32);inv[pos]=np.arange(N,dtype=np.int32);return pos,inv
def neighbor(pos,inv,i,d):
 r,c=divmod(int(inv[i]),24);r+=d==1;r-=d==3;c+=d==0;c-=d==2
 return -1 if r<0 or r>=24 or c<0 or c>=24 else int(pos[r*24+c])
def cache(ld,names,dev):
 q=[]
 for n in names:
  with np.load(ld/(Path(n).stem+'.npz'),allow_pickle=False) as z:x=z['tiles_uint8'].copy()
  q.append(torch.from_numpy(x.transpose(0,3,1,2).copy()))
 return torch.stack(q).to(dev,dtype=torch.float32).div_(255.)
def role_gpu(bank,bi,ids,d,source):
 z=bank[bi,ids].clone();rot=d if torch.is_tensor(d) else torch.full_like(bi,d)
 if not source:rot=(rot+2)%4
 swap=(rot==1)|(rot==3);flip=(rot==2)|(rot==3)
 if swap.any():z[swap]=z[swap].transpose(-2,-1)
 if flip.any():z[flip]=torch.flip(z[flip],[-1])
 return z
def g2(a):
 if not torch.cuda.is_available():raise RuntimeError('P23 requires interactive local CUDA')
 torch.manual_seed(20260817);np.random.seed(20260817);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;p=p13();fit,_=p.source_lists(a.manifest);fit=sorted(fit);tr,sel=fit[:96],fit[96:128];dev=torch.device('cuda');a.work.mkdir(parents=True,exist_ok=True);bank=cache(a.label_dir,tr+sel,dev);edges=[]
 for bi,n in enumerate(tr):
  pos,inv=labs(a.label_dir,n)
  for d in range(4):
   for i in range(N):
    j=neighbor(pos,inv,i,d)
    if j>=0:edges.append((bi,i,j,d))
 e=torch.tensor(edges,device=dev);m=Enc().to(dev);opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4)
 for st in range(a.steps):
  b=e[torch.randint(len(e),(a.batch_size,),device=dev)];u=m(role_gpu(bank,b[:,0],b[:,1],b[:,3],True));v=m(role_gpu(bank,b[:,0],b[:,2],b[:,3],False));loss=torch.nn.functional.cross_entropy((u@v.T)/.07,torch.arange(len(b),device=dev));opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  if (st+1)%250==0:print(json.dumps({'stage':'train','step':st+1,'loss':float(loss.item())}),flush=True)
 ck=a.work/'p23_g2_retriever_fp32.pt';torch.save({'state':m.state_dict(),'steps':a.steps},ck);m.eval();results=[]
 with torch.no_grad():
  for M in [16,32,48,64]:
   covs=[];recs=[];basecovs=[]
   for bi,n in enumerate(sel):
    c,v,s=p.load_score_cache(a.score_dir,n);pos,inv=labs(a.label_dir,n);covered=base=hit=basehit=tot=0
    for d in range(4):
     ids=torch.arange(N,device=dev);cand=m(role_gpu(bank[96:],torch.full((N,),bi,device=dev),ids,d,False));src=m(role_gpu(bank[96:],torch.full((N,),bi,device=dev),ids,d,True));sim=src@cand.T;sim.fill_diagonal_(-1e9);top=sim.topk(M,dim=1).indices.cpu().numpy()
     for i in range(N):
      truth=neighbor(pos,inv,i,d)
      if truth<0:continue
      tot+=1;base+=int(np.any(c[i,v[i]]==truth));union=np.unique(np.r_[c[i,v[i]],top[i]])[:128];covered+=int(np.any(union==truth));hit+=int(np.any(top[i,:20]==truth));basehit+=int(np.any(c[i,np.flatnonzero(v[i])[np.argsort(-s[d,i,np.flatnonzero(v[i])],kind='stable')[:20]]]==truth))
    covs.append(covered/tot);basecovs.append(base/tot);recs.append((hit/tot,basehit/tot));
   results.append({'M':M,'coverage':float(np.mean(covs)),'baseline_coverage':float(np.mean(basecovs)),'coverage_gain_pp':100*(np.mean(covs)-np.mean(basecovs)),'retrieval_recall20':float(np.mean([x[0] for x in recs])),'baseline_recall20':float(np.mean([x[1] for x in recs])),'retrieval_gain_pp':100*float(np.mean([x[0]-x[1] for x in recs]))});print(json.dumps(results[-1]),flush=True)
 best=max(results,key=lambda z:z['coverage_gain_pp']);rep={'experiment':'P23_DCTR24','gate':'G2','train_sources':96,'selection_sources':32,'edges':len(edges),'steps':a.steps,'results':results,'selected':best,'checkpoint':str(ck),'labels_used':True,'targets_opened':False,'p8_imported':False,'passes_G2':bool(best['coverage_gain_pp']>=3 and best['retrieval_gain_pp']>=1)};(a.work/'p23_g2_report.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep),flush=True)
 if not rep['passes_G2']:raise RuntimeError('P23 G2 fast-futility reject')
def main2():
 p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['g0','g1','g2']);p.add_argument('--inputs',type=Path,default=Path(r'E:\pazzle_data\train\inputs'));p.add_argument('--score-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache'));p.add_argument('--label-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache'));p.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));p.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P23_dctr'));p.add_argument('--steps',type=int,default=1500);p.add_argument('--batch-size',type=int,default=128);a=p.parse_args();safe(a.inputs,a.score_dir,a.label_dir,a.manifest,a.work);{'g0':g0,'g1':g1,'g2':g2}[a.mode](a)
if __name__=='__main__':main2()
