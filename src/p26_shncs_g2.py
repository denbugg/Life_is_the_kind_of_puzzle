"""P26 SHNCS-24 G2 after G0/G1. Full pairs, sampled hard negatives, no target PNGs."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
N,T=576,20
def p13():sys.path.insert(0,str(Path(__file__).resolve().parent));import p13_component_pose as p;return p
def lab(ld,n):
 with np.load(ld/(Path(n).stem+'.npz'),allow_pickle=False) as z:po=z['target_tile_to_slot'].astype(np.int32);src=str(z['source'])
 if src!=n or np.unique(po).size!=N:raise RuntimeError('bad label cache')
 iv=np.empty(N,np.int32);iv[po]=np.arange(N,dtype=np.int32);return po,iv
def nb(po,iv,i,d):
 r,c=divmod(int(iv[i]),24);r+=d==1;r-=d==3;c+=d==0;c-=d==2
 return -1 if r<0 or r>=24 or c<0 or c>=24 else int(po[r*24+c])
def bank(ld,names):
 q=[]
 for n in names:
  with np.load(ld/(Path(n).stem+'.npz'),allow_pickle=False) as z:q.append(z['tiles_uint8'].transpose(0,3,1,2).copy())
 return np.stack(q)
def pool(w,n):
 with np.load(w/(Path(n).stem+'.npz'),allow_pickle=False) as z:return z['pool'].astype(np.int32,copy=True)
def pairs(b,bi,ai,ci,di,dev):
 x=torch.from_numpy(b[bi,ai]).to(dev,dtype=torch.float32).div_(255.);y=torch.from_numpy(b[bi,ci]).to(dev,dtype=torch.float32).div_(255.);sw=(di==1)|(di==3);fl=(di==2)|(di==3)
 if sw.any():x[sw]=x[sw].transpose(-2,-1);y[sw]=y[sw].transpose(-2,-1)
 if fl.any():z=x[fl].clone();x[fl]=y[fl];y[fl]=z
 return torch.cat([x,y],1)
class Net(torch.nn.Module):
 def __init__(self):
  super().__init__();self.a=torch.nn.Conv2d(6,24,3,padding=1);self.b=torch.nn.Conv2d(24,32,3,padding=1);self.o=torch.nn.Linear(32,1)
 def forward(self,x):z=torch.nn.functional.silu(self.a(x));z=torch.nn.functional.silu(self.b(z));return self.o(z.mean((-2,-1))).squeeze(-1)
def groups(w,ld,names):
 g=[]
 for bi,n in enumerate(names):
  po,iv=lab(ld,n);u=pool(w,n)
  for d in range(4):
   for i in range(N):
    q=nb(po,iv,i,d)
    if q>=0 and np.any(u[d,i]==q):g.append((bi,i,d,q))
 return np.asarray(g,np.int32)
def score_board(m,b,u,bi,d,dev):
 ci=u[d].reshape(-1);ai=np.repeat(np.arange(N,dtype=np.int64),128);di=np.full(len(ci),d,np.int64);out=np.empty(len(ci),np.float32)
 for st in range(0,len(ci),4096):
  en=min(len(ci),st+4096);out[st:en]=m(pairs(b,np.full(en-st,bi,np.int64),ai[st:en],ci[st:en],di[st:en],dev)).cpu().numpy()
 return out.reshape(N,128)
def evaluate(m,b,w,ld,sd,names,dev):
 p=p13();cached=[]
 with torch.no_grad():
  for bi,n in enumerate(names):
   u=pool(w,n);cached.append([score_board(m,b,u,bi,d,dev) for d in range(4)]);print(json.dumps({'stage':'selection_scores','done':bi+1,'total':len(names)}),flush=True)
 rs=[]
 for al in [0.,.05,.10,.20,.40]:
  hit=base=tot=0
  for bi,n in enumerate(names):
   po,iv=lab(ld,n);u=pool(w,n);c,v,s=p.load_score_cache(sd,n)
   for d in range(4):
    z=cached[bi][d]
    for i in range(N):
     q=nb(po,iv,i,d)
     if q<0:continue
     ci=u[d,i];zz=(z[i]-z[i].mean())/(z[i].std()+1e-6);sl=np.flatnonzero(v[i]);mp={int(c[i,j]):float(s[d,i,j]) for j in sl};f=np.asarray([mp.get(int(x),0.) for x in ci]);f=(f-f.mean())/(f.std()+1e-6);rank=ci[np.argsort(-(zz+al*f))[:20]];br=c[i,sl[np.argsort(-s[d,i,sl])[:20]]];hit+=int(np.any(rank==q));base+=int(np.any(br==q));tot+=1
  rs.append({'alpha':al,'recall20':hit/tot,'baseline_recall20':base/tot,'gain_pp':100*(hit-base)/tot})
 return rs
def main():
 q=argparse.ArgumentParser();q.add_argument('--pool-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P25_scxr'));q.add_argument('--label-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache'));q.add_argument('--score-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache'));q.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));q.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P26_shncs'));q.add_argument('--steps',type=int,default=2000);q.add_argument('--groups-per-step',type=int,default=16);a=q.parse_args()
 if 'p8' in '\n'.join(str(x).lower() for x in [a.pool_dir,a.label_dir,a.score_dir,a.work]):raise RuntimeError('P8 prohibited')
 if not torch.cuda.is_available():raise RuntimeError('interactive CUDA required')
 torch.manual_seed(20260817);np.random.seed(20260817);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;p=p13();fit,_=p.source_lists(a.manifest);fit=sorted(fit);tr,se=fit[:96],fit[96:128];dev=torch.device('cuda');bt=bank(a.label_dir,tr);bs=bank(a.label_dir,se);pt=np.stack([pool(a.pool_dir,n) for n in tr]);g=groups(a.pool_dir,a.label_dir,tr);m=Net().to(dev);opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4);rng=np.random.default_rng(20260817)
 for st in range(a.steps):
  z=g[rng.integers(len(g),size=a.groups_per_step)];bi,ai,di,pos=z.T;cs=[];ys=[];bb=[];aa=[];dd=[]
  for B,A,D,P in zip(bi,ai,di,pos):
   u=pt[int(B),int(D),int(A)];neg=np.asarray([x for x in u if x>=0 and x!=P],np.int32);ng=rng.choice(neg,size=15,replace=False);cs.extend([int(P),*ng]);ys.extend([1.,*([0.]*15)]);bb.extend([B]*16);aa.extend([A]*16);dd.extend([D]*16)
  log=m(pairs(bt,np.asarray(bb),np.asarray(aa),np.asarray(cs),np.asarray(dd),dev));loss=torch.nn.functional.binary_cross_entropy_with_logits(log,torch.tensor(ys,device=dev));opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  if (st+1)%250==0:print(json.dumps({'stage':'train','step':st+1,'loss':float(loss.item())}),flush=True)
 m.eval();rows=evaluate(m,bs,a.pool_dir,a.label_dir,a.score_dir,se,dev);best=max(rows,key=lambda x:x['recall20']);rep={'experiment':'P26_SHNCS24','gate':'G2','train_groups':int(len(g)),'steps':a.steps,'scores':rows,'selected':best,'labels_used':True,'targets_opened':False,'p8_imported':False,'passes_G2':bool(best['gain_pp']>=1)};a.work.mkdir(parents=True,exist_ok=True);(a.work/'p26_g2_report.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep),flush=True)
 if not rep['passes_G2']:raise RuntimeError('P26 G2 fast-futility reject')
if __name__=='__main__':main()
