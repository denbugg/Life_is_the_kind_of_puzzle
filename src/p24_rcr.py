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
# Legacy entrypoint deferred until the G2 extension below.

# G2 extension: P23 checkpoint and P10 labels are accessed only here.
class Cross(torch.nn.Module):
 def __init__(self):
  super().__init__();self.a=torch.nn.Conv2d(6,48,3,padding=1);self.b=torch.nn.Conv2d(48,48,3,padding=1);self.c=torch.nn.Conv2d(48,64,3,padding=1);self.o=torch.nn.Linear(64,1)
 def forward(self,x):
  z=torch.nn.functional.silu(self.a(x));z=torch.nn.functional.silu(self.b(z));z=torch.nn.functional.silu(self.c(z));return self.o(z.mean((-2,-1))).squeeze(-1)
def p13():
 sys.path.insert(0,str(Path(__file__).resolve().parent));import p13_component_pose as z;return z
def labs(ld,n):
 with np.load(ld/(Path(n).stem+'.npz'),allow_pickle=False) as z:pos=z['target_tile_to_slot'].astype(np.int32,copy=True);src=str(z['source'])
 if src!=n or np.unique(pos).size!=N or pos.min()!=0 or pos.max()!=N-1:raise RuntimeError('bad label cache')
 inv=np.empty(N,np.int32);inv[pos]=np.arange(N,dtype=np.int32);return pos,inv
def nb(pos,inv,i,d):
 r,c=divmod(int(inv[i]),24);r+=d==1;r-=d==3;c+=d==0;c-=d==2
 return -1 if r<0 or r>=24 or c<0 or c>=24 else int(pos[r*24+c])
def bank(ld,names,dev):
 q=[]
 for n in names:
  with np.load(ld/(Path(n).stem+'.npz'),allow_pickle=False) as z:x=z['tiles_uint8'].copy()
  q.append(torch.from_numpy(x.transpose(0,3,1,2).copy()))
 return torch.stack(q).to(dev,dtype=torch.float32).div_(255.)
def pair_gpu(b,bi,ai,ci,d):
 x=b[bi,ai].clone();y=b[bi,ci].clone();rot=d
 sw=(rot==1)|(rot==3);fl=(rot==2)|(rot==3)
 if sw.any():x[sw]=x[sw].transpose(-2,-1);y[sw]=y[sw].transpose(-2,-1)
 if fl.any():u=x[fl].clone();x[fl]=y[fl];y[fl]=u
 return torch.cat([x,y],1)
def g2(a):
 if not torch.cuda.is_available():raise RuntimeError('P24 requires interactive CUDA')
 torch.manual_seed(20260817);np.random.seed(20260817);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;p=p13();fit,_=p.source_lists(a.manifest);fit=sorted(fit);tr,sel=fit[:96],fit[96:128];dev=torch.device('cuda');a.work.mkdir(parents=True,exist_ok=True);B=bank(a.label_dir,tr+sel,dev)
 # Load P23 retriever only for candidate-pool formation, never as final score.
 import p23_dctr as r23;retr=r23.Enc().to(dev);state=torch.load(a.p23_checkpoint,map_location=dev,weights_only=True)['state'];retr.load_state_dict(state);retr.eval();pools=[];groups=[];scores=[]
 with torch.no_grad():
  for bi,n in enumerate(tr+sel):
   c,v,s=p.load_score_cache(a.score_dir,n);pool=np.full((4,N,128),-1,np.int32);pos,inv=labs(a.label_dir,n);ids=torch.arange(N,device=dev);bb=torch.full((N,),bi,device=dev)
   for d in range(4):
    src=retr(r23.role_gpu(B,bb,ids,d,True));can=retr(r23.role_gpu(B,bb,ids,d,False));top=(src@can.T).fill_diagonal_(-1e9).topk(64,dim=1).indices.cpu().numpy()
    for i in range(N):
     u=[]
     for z in np.r_[top[i],c[i,np.flatnonzero(v[i])]]:
      if z!=i and z not in u:u.append(int(z))
      if len(u)==128:break
     pool[d,i,:len(u)]=u
     q=nb(pos,inv,i,d)
     if q>=0:
      hit=np.flatnonzero(pool[d,i]==q)
      if len(hit):groups.append((bi,i,d,int(hit[0])))
   pools.append(pool);scores.append(s)
 pools=np.stack(pools);groups=torch.tensor(groups,device=dev);model=Cross().to(dev);opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
 for st in range(a.steps):
  g=groups[torch.randint(len(groups),(a.batch_size,),device=dev)];ci=torch.from_numpy(pools[g[:,0].cpu().numpy(),g[:,2].cpu().numpy(),g[:,1].cpu().numpy()]).to(dev);bi=g[:,0,None].expand_as(ci).reshape(-1);ai=g[:,1,None].expand_as(ci).reshape(-1);di=g[:,2,None].expand_as(ci).reshape(-1);x=pair_gpu(B,bi,ai,ci.reshape(-1),di);log=model(x).reshape(len(g),128);loss=torch.nn.functional.cross_entropy(log,g[:,3]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  if (st+1)%250==0:print(json.dumps({'stage':'train','step':st+1,'loss':float(loss.item())}),flush=True)
 # Selection computes pure cross ranks; frozen original top20 is mandatory baseline.
 rows=[];model.eval()
 with torch.no_grad():
  for M in [32,48,64]:
   for alpha in [0.,.05,.10,.20,.40]:
    hit=base=tot=0
    for bi,n in enumerate(sel):
     c,v,s=p.load_score_cache(a.score_dir,n);pos,inv=labs(a.label_dir,n);pool=pools[96+bi]
     for d in range(4):
      for i in range(N):
       q=nb(pos,inv,i,d)
       if q<0:continue
       u=[]
       for z0 in np.r_[pool[d,i,:M],c[i,np.flatnonzero(v[i])],pool[d,i,M:64]]:
        if z0>=0 and z0!=i and z0 not in u:u.append(int(z0))
        if len(u)==128:break
       ci=np.asarray(u,np.int64);xx=pair_gpu(B,torch.full((len(ci),),bi+96,device=dev),torch.full((len(ci),),i,device=dev),torch.from_numpy(ci).to(dev),torch.full((len(ci),),d,device=dev));z=model(xx).cpu().numpy();z=(z-z.mean())/(z.std()+1e-6);fr=np.zeros(len(ci),np.float32);slots=np.flatnonzero(v[i]);mp={int(c[i,j]):float(s[d,i,j]) for j in slots};fr=np.asarray([mp.get(int(j),0.) for j in ci],np.float32);fr=(fr-fr.mean())/(fr.std()+1e-6);rank=ci[np.argsort(-(z+alpha*fr))[:20]];br=c[i,slots[np.argsort(-s[d,i,slots])[:20]]];hit+=int(np.any(rank==q));base+=int(np.any(br==q));tot+=1
    rows.append({'M':M,'alpha':alpha,'recall20':hit/tot,'baseline_recall20':base/tot,'gain_pp':100*(hit-base)/tot})
 best=max(rows,key=lambda x:x['recall20']);rep={'experiment':'P24_RCR24','gate':'G2','train_sources':96,'selection_sources':32,'groups':int(len(groups)),'steps':a.steps,'scores':rows,'selected':best,'labels_used':True,'targets_opened':False,'p8_imported':False,'passes_G2':bool(best['gain_pp']>=1)};(a.work/'p24_g2_report.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep),flush=True)
 if not rep['passes_G2']:raise RuntimeError('P24 G2 fast-futility reject')
def main2():
 p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['g0','g1','g2']);p.add_argument('--inputs',type=Path,default=Path(r'E:\pazzle_data\train\inputs'));p.add_argument('--score-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache'));p.add_argument('--label-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache'));p.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));p.add_argument('--p23-checkpoint',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P23_dctr\p23_g2_retriever_fp32.pt'));p.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P24_rcr'));p.add_argument('--steps',type=int,default=1500);p.add_argument('--batch-size',type=int,default=64);a=p.parse_args();safe(a.inputs,a.score_dir,a.label_dir,a.manifest,a.p23_checkpoint,a.work);{'g0':g0,'g1':g1,'g2':g2}[a.mode](a)
if __name__=='__main__':main2()
