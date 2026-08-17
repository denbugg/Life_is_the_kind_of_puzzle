"""P32 DSCP-24 G2. FIT-train cached labels only; no target images/rank96/P8."""
from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from p32_dscp import N,dino,features,DSCP,bad
import p13_component_pose as p13


def load_label(d:Path,n:str):
 with np.load(d/(Path(n).stem+'.npz'),allow_pickle=False) as z:
  x=z['target_tile_to_slot'].astype(np.int64);s=str(z['source'])
 if s!=n or x.shape!=(N,) or np.unique(x).size!=N:raise RuntimeError('invalid FIT label')
 return torch.from_numpy(x)
def cache_features(a,sources):
 dev=torch.device('cuda');m=dino(dev);out={};a.cache.mkdir(parents=True,exist_ok=True)
 for k,n in enumerate(sources,1):
  p=a.cache/(Path(n).stem+'.pt')
  if p.exists():z=torch.load(p,map_location='cpu',weights_only=True)
  else:
   from p29_dpcg import load_tiles
   z=features(m,load_tiles(a.inputs,n),dev).half();torch.save(z,p)
  if z.shape!=(N,384):raise RuntimeError('bad DINO feature cache')
  out[n]=z.float()
  if k%4==0:print(json.dumps({'stage':'features','done':k,'total':len(sources)}),flush=True)
 return out
def sinkhorn_loss(logits,y):
 x=logits/0.5
 for _ in range(4):x=x-torch.logsumexp(x,1,keepdim=True);x=x-torch.logsumexp(x,0,keepdim=True)
 return F.nll_loss(x,y)
def train(a,feat,labels,sources):
 dev=torch.device('cuda');torch.manual_seed(20260817);m=DSCP().to(dev);opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=1e-4);hist=[];start=time.perf_counter()
 for ep in range(a.epochs):
  order=list(sources);random.Random(20260817+ep).shuffle(order);ls=[];m.train()
  for n in order:
   z=feat[n].to(dev);y=labels[n].to(dev);log=m(z);loss=F.cross_entropy(log,y)+0.15*sinkhorn_loss(log,y);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();ls.append(float(loss.item()))
  h={'epoch':ep+1,'loss':float(np.mean(ls))};hist.append(h);print(json.dumps({'stage':'fit',**h}),flush=True)
  if time.perf_counter()-start>a.minutes*60:break
 return m.eval(),{'epochs_run':len(hist),'history':hist,'minutes':(time.perf_counter()-start)/60}
def evaluate(m,feat,labels,sources):
 dev=torch.device('cuda');top=[];place=[];invalid=0
 with torch.no_grad():
  for k,n in enumerate(sources,1):
   logits=m(feat[n].to(dev)).float().cpu().numpy();y=labels[n].numpy();
   if not np.isfinite(logits).all():invalid+=1;continue
   t=np.argsort(-logits,axis=1,kind='stable')[:,:20];top.append(float(np.mean([y[i] in t[i] for i in range(N)])))
   r,c=linear_sum_assignment(-logits);p=np.empty(N,np.int32);p[r]=c;place.append(float(np.mean(p==y)))
   print(json.dumps({'stage':'eval','done':k,'total':len(sources)}),flush=True)
 return float(np.mean(top)),float(np.mean(place)),invalid
def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,default=Path(r'E:\pazzle_data\train\inputs'));p.add_argument('--labels',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache'));p.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));p.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P32_dscp'));p.add_argument('--epochs',type=int,default=8);p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--minutes',type=float,default=35.0);a=p.parse_args();a.cache=a.work/'feature_cache';bad(a.inputs,a.labels,a.manifest,a.work);a.work.mkdir(parents=True,exist_ok=True);train_sources,_=p13.source_lists(a.manifest);sources=sorted(train_sources)[:96];lab={n:load_label(a.labels,n) for n in sources};feat=cache_features(a,sources);m,fit=train(a,feat,lab,sources);top,placement,invalid=evaluate(m,feat,lab,sources);torch.save({'state_dict':m.state_dict(),'fit':fit},a.work/'p32_g2_checkpoint.pt');rep={'experiment':'P32_DSCP24','gate':'G2','fit':fit,'top20':top,'placement':placement,'invalid':invalid,'labels_used':True,'targets_opened':False,'p8_imported':False,'passes_G2':bool(top>=0.05 and placement>=0.005 and invalid==0)};(a.work/'p32_g2_report.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep),flush=True)
 if not rep['passes_G2']:raise RuntimeError('P32 G2 rejected')
if __name__=='__main__':main()
