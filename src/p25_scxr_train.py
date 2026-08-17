"""P25 SCXR-24 G2b: bounded full-pair cross-reranker over streamed candidate pools."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
N,T=576,20
def p13():sys.path.insert(0,str(Path(__file__).resolve().parent));import p13_component_pose as p;return p
def lab(ld,n):
 with np.load(ld/(Path(n).stem+'.npz'),allow_pickle=False) as z:pos=z['target_tile_to_slot'].astype(np.int32);src=str(z['source'])
 if src!=n or np.unique(pos).size!=N:raise RuntimeError('bad label cache')
 inv=np.empty(N,np.int32);inv[pos]=np.arange(N,dtype=np.int32);return pos,inv
def nb(pos,inv,i,d):
 r,c=divmod(int(inv[i]),24);r+=d==1;r-=d==3;c+=d==0;c-=d==2
 return -1 if r<0 or r>=24 or c<0 or c>=24 else int(pos[r*24+c])
def load_bank(ld,names):
 q=[]
 for n in names:
  with np.load(ld/(Path(n).stem+'.npz'),allow_pickle=False) as z:q.append(z['tiles_uint8'].transpose(0,3,1,2).copy())
 return np.stack(q)
def pairs(bank,bi,ai,ci,di,dev):
 x=torch.from_numpy(bank[bi,ai]).to(dev,dtype=torch.float32).div_(255.);y=torch.from_numpy(bank[bi,ci]).to(dev,dtype=torch.float32).div_(255.);sw=(di==1)|(di==3);fl=(di==2)|(di==3)
 if sw.any():x[sw]=x[sw].transpose(-2,-1);y[sw]=y[sw].transpose(-2,-1)
 if fl.any():z=x[fl].clone();x[fl]=y[fl];y[fl]=z
 return torch.cat([x,y],1)
class Cross(torch.nn.Module):
 def __init__(self):
  super().__init__();self.a=torch.nn.Conv2d(6,48,3,padding=1);self.b=torch.nn.Conv2d(48,48,3,padding=1);self.c=torch.nn.Conv2d(48,64,3,padding=1);self.o=torch.nn.Linear(64,1)
 def forward(self,x):
  z=torch.nn.functional.silu(self.a(x));z=torch.nn.functional.silu(self.b(z));z=torch.nn.functional.silu(self.c(z));return self.o(z.mean((-2,-1))).squeeze(-1)
def pools(w,n):
 with np.load(w/(Path(n).stem+'.npz'),allow_pickle=False) as z:return z['pool'].astype(np.int32,copy=True)
def build_groups(w,ld,names):
 out=[]
 for bi,n in enumerate(names):
  po,iv=lab(ld,n);u=pools(w,n)
  for d in range(4):
   for i in range(N):
    q=nb(po,iv,i,d)
    if q>=0:
     h=np.flatnonzero(u[d,i]==q)
     if len(h):out.append((bi,i,d,int(h[0])))
 return np.asarray(out,np.int32)
def eval_sel(m,bank,w,ld,sd,names,dev):
 p=p13();rows=[]
 with torch.no_grad():
  for alpha in [0.,.05,.10,.20,.40]:
   hit=base=tot=0
   for bi,n in enumerate(names):
    po,iv=lab(ld,n);u=pools(w,n);c,v,s=p.load_score_cache(sd,n)
    for d in range(4):
     for i in range(N):
      q=nb(po,iv,i,d)
      if q<0:continue
      ci=u[d,i];ci=ci[ci>=0];L=len(ci);xx=pairs(bank,np.full(L,bi,np.int64),np.full(L,i,np.int64),ci,np.full(L,d,np.int64),dev);z=m(xx).cpu().numpy();z=(z-z.mean())/(z.std()+1e-6);sl=np.flatnonzero(v[i]);mp={int(c[i,j]):float(s[d,i,j]) for j in sl};f=np.asarray([mp.get(int(x),0.) for x in ci]);f=(f-f.mean())/(f.std()+1e-6);rank=ci[np.argsort(-(z+alpha*f))[:20]];br=c[i,sl[np.argsort(-s[d,i,sl])[:20]]];hit+=int(np.any(rank==q));base+=int(np.any(br==q));tot+=1
   rows.append({'alpha':alpha,'recall20':hit/tot,'baseline_recall20':base/tot,'gain_pp':100*(hit-base)/tot})
 return rows
def main():
 q=argparse.ArgumentParser();q.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P25_scxr'));q.add_argument('--pool-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P25_scxr'));q.add_argument('--label-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache'));q.add_argument('--score-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache'));q.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));q.add_argument('--steps',type=int,default=1500);q.add_argument('--batch',type=int,default=64);a=q.parse_args()
 if 'p8' in '\n'.join(map(lambda x:str(x).lower(),[a.work,a.pool_dir,a.label_dir,a.score_dir])):raise RuntimeError('P8 prohibited')
 if not torch.cuda.is_available():raise RuntimeError('interactive CUDA required')
 torch.manual_seed(20260817);np.random.seed(20260817);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;p=p13();fit,_=p.source_lists(a.manifest);fit=sorted(fit);tr,se=fit[:96],fit[96:128];dev=torch.device('cuda');bt=load_bank(a.label_dir,tr);bs=load_bank(a.label_dir,se);pt=np.stack([pools(a.pool_dir,n) for n in tr]);ps=np.stack([pools(a.pool_dir,n) for n in se]);g=build_groups(a.pool_dir,a.label_dir,tr);m=Cross().to(dev);opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4)
 for st in range(a.steps):
  z=g[np.random.randint(len(g),size=a.batch)];bi,ai,di,slot=z.T;ci=pt[bi,di,ai];x=pairs(bt,np.repeat(bi,128),np.repeat(ai,128),ci.reshape(-1),np.repeat(di,128),dev);log=m(x).reshape(a.batch,128);loss=torch.nn.functional.cross_entropy(log,torch.from_numpy(slot).to(dev));opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  if (st+1)%250==0:print(json.dumps({'stage':'train','step':st+1,'loss':float(loss.item())}),flush=True)
 m.eval();rows=eval_sel(m,bs,a.pool_dir,a.label_dir,a.score_dir,se,dev);best=max(rows,key=lambda x:x['recall20']);rep={'experiment':'P25_SCXR24','gate':'G2b','groups':int(len(g)),'steps':a.steps,'scores':rows,'selected':best,'labels_used':True,'targets_opened':False,'p8_imported':False,'passes_G2b':bool(best['gain_pp']>=1)};(a.work/'p25_g2b_report.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep),flush=True)
 if not rep['passes_G2b']:raise RuntimeError('P25 G2b fast-futility reject')
if __name__=='__main__':main()
