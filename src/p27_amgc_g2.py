"""P27 AMGC-24 G2: covariance-aware analytic feature calibration after G0/G1."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent));from p27_amgc import tiles,orient,amgc,N

def p13():import p13_component_pose as p;return p
def labels(d,n):
 with np.load(d/(Path(n).stem+'.npz'),allow_pickle=False) as z:po=z['target_tile_to_slot'].astype(np.int32);src=str(z['source'])
 if src!=n or np.unique(po).size!=N:raise RuntimeError('invalid label cache')
 iv=np.empty(N,np.int32);iv[po]=np.arange(N,dtype=np.int32);return po,iv
def nei(po,iv,i,d):
 r,c=divmod(int(iv[i]),24);r+=d==1;r-=d==3;c+=d==0;c-=d==2
 return -1 if r<0 or r>=24 or c<0 or c>=24 else int(po[r*24+c])
def features(root,sd,n):
 p=p13();t=tiles(root,n);c,v,s=p.load_score_cache(sd,n);out=[]
 for d in range(4):
  q=orient(t,d)
  for i in range(N):
   sl=np.flatnonzero(v[i]);ci=c[i,sl];sc=s[d,i,sl];f=amgc(np.repeat(q[i:i+1],len(ci),0),q[ci]);rk=np.empty(len(sc),np.float32);rk[np.argsort(np.argsort(-sc))]=np.linspace(0,1,len(sc),endpoint=False);out.append((d,i,ci.astype(np.int32),np.stack([sc,f,rk],1).astype(np.float32)))
 return out
def rows(ft,po,iv,seed):
 x=[];y=[];rng=np.random.default_rng(seed)
 for d,i,ci,z in ft:
  q=nei(po,iv,i,d);h=np.flatnonzero(ci==q)
  if q<0 or not len(h):continue
  neg=np.flatnonzero(ci!=q);k=rng.choice(neg,size=min(15,len(neg)),replace=False);x.append(z[h[0]]);y.append(1.);x.extend(z[k]);y.extend([0.]*len(k))
 return np.asarray(x,np.float32),np.asarray(y,np.float32)
def fit(x,y,C):
 mu=x.mean(0);sd=x.std(0)+1e-6;z=(x-mu)/sd;w=np.zeros(3,np.float32);b=np.float32(0)
 for _ in range(500):
  pr=1/(1+np.exp(-(z@w+b)));e=pr-y;w-=.08*((z.T@e)/len(y)+w/C/len(y));b-=.08*e.mean()
 return mu,sd,w,b
def score(ft,par,po,iv):
 mu,sd,w,b=par;hit=base=tot=0
 for d,i,ci,z in ft:
  q=nei(po,iv,i,d)
  if q<0:continue
  p=(z-mu)/sd@w+b;pn=(p-p.mean())/(p.std()+1e-6);fr=(z[:,0]-z[:,0].mean())/(z[:,0].std()+1e-6)
  for a in [0.,.05,.1,.2,.4]:
   rank=ci[np.argsort(-(fr+a*pn))[:20]];# record externally
  tot+=1
 return tot
def evaluate(fmap,ld,names,pars):
 out=[]
 for C,par in pars.items():
  mu,sd,w,b=par;vals=[]
  for a in [0.,.05,.1,.2,.4]:
   h=ba=tot=0
   for n in names:
    po,iv=labels(ld,n)
    for d,i,ci,z in fmap[n]:
     q=nei(po,iv,i,d)
     if q<0:continue
     p=((z-mu)/sd)@w+b;pn=(p-p.mean())/(p.std()+1e-6);fr=(z[:,0]-z[:,0].mean())/(z[:,0].std()+1e-6);h+=int(np.any(ci[np.argsort(-(fr+a*pn))[:20]]==q));ba+=int(np.any(ci[np.argsort(-fr)[:20]]==q));tot+=1
   vals.append({'C':C,'alpha':a,'recall20':h/tot,'baseline_recall20':ba/tot,'gain_pp':100*(h-ba)/tot})
  out.extend(vals)
 return out
def main():
 q=argparse.ArgumentParser();q.add_argument('--inputs',type=Path,default=Path(r'E:\pazzle_data\train\inputs'));q.add_argument('--score-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache'));q.add_argument('--label-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache'));q.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));q.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P27_amgc'));a=q.parse_args()
 if 'p8' in '\n'.join(str(x).lower() for x in [a.inputs,a.score_dir,a.label_dir,a.manifest,a.work]):raise RuntimeError('P8 prohibited')
 p=p13();fitn,_=p.source_lists(a.manifest);fitn=sorted(fitn);tr,se=fitn[:96],fitn[96:128];fmap={};xs=[];ys=[]
 for k,n in enumerate(tr+se):
  ft=features(a.inputs,a.score_dir,n);fmap[n]=ft
  if n in tr:
   po,iv=labels(a.label_dir,n);x,y=rows(ft,po,iv,20260817+k);xs.append(x);ys.append(y)
  if (k+1)%16==0:print(json.dumps({'stage':'features','done':k+1,'total':128}),flush=True)
 x=np.concatenate(xs);y=np.concatenate(ys);pars={C:fit(x,y,C) for C in [.01,.1,1.]};res=evaluate(fmap,a.label_dir,se,pars);best=max(res,key=lambda z:z['recall20']);rep={'experiment':'P27_AMGC24','gate':'G2','train_rows':int(len(y)),'scores':res,'selected':best,'labels_used':True,'targets_opened':False,'p8_imported':False,'passes_G2':bool(best['gain_pp']>=1)};a.work.mkdir(parents=True,exist_ok=True);(a.work/'p27_g2_report.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep),flush=True)
 if not rep['passes_G2']:raise RuntimeError('P27 G2 fast-futility reject')
if __name__=='__main__':main()
