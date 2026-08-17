"""P19 MDEC-24: input-only masked directional edge contrastive scoring.
Pre-registered before this file was created. G0/G1 never opens targets or label cache.
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

SRC=Path(__file__).resolve().parent
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
import p13_component_pose as p13

GRID,TILE,SEED=24,20,20260817

def seed_all():
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

def assert_safe(*p):
 if 'p8' in '\n'.join(map(lambda x:str(x).lower(),p)): raise RuntimeError('P8 prohibited')

def read_board(root:Path,name:str)->np.ndarray:
 a=np.asarray(Image.open(root/name).convert('RGB'),dtype=np.float32)/255.
 if a.shape!=(480,480,3): raise RuntimeError(f'bad input shape {a.shape}')
 return a.reshape(GRID,TILE,GRID,TILE,3).transpose(0,2,1,3,4).reshape(576,TILE,TILE,3)

def side(tile:np.ndarray, direction:int, cut:int, right:bool)->np.ndarray:
 # all directions normalized to H=20,W=7: right is the prospective neighbor-side strip
 if direction==0: x=cut; out=tile[:, x:x+7] if right else tile[:,x-7:x]
 else:
  y=cut; out=tile[y:y+7,:] if right else tile[y-7:y,:]; out=np.transpose(out,(1,0,2))
 return out.transpose(2,0,1).copy()

def seam_raw(a,b): return -np.abs(a[:,:,:,-1]-b[:,:,:,0]).mean((1,2))

def masked(a,rng):
 a=a.copy(); w=int(rng.integers(1,3));a[:,:,-w:]=0.;return a

class EdgePairs(Dataset):
 def __init__(self,root,names,per_source=128,training=True):
  self.root,self.names,self.per_source,self.training=root,names,per_source,training
  self.boards=[read_board(root,n) for n in names]
 def __len__(self): return len(self.names)*self.per_source
 def __getitem__(self,i):
  rng=np.random.default_rng(SEED+i+(0 if self.training else 999999)); b=self.boards[i//self.per_source]; t=int(rng.integers(576));d=int(rng.integers(2));cut=int(rng.integers(7,14));pos=bool(i%2==0)
  a=side(b[t],d,cut,False); bb=side(b[t],d,cut,True)
  if not pos:
   u=int(rng.integers(575));u+=u>=t;bb=side(b[u],d,cut,True)
  if self.training:
   scale=float(rng.uniform(.9,1.1));a=np.clip(masked(a,rng)*scale,0,1);bb=np.clip(masked(bb,rng)*scale,0,1)
  return torch.from_numpy(a),torch.from_numpy(bb),torch.tensor(float(pos))

class EdgeNet(nn.Module):
 def __init__(self):
  super().__init__();self.f=nn.Sequential(nn.Conv2d(6,48,3,padding=1),nn.GELU(),nn.Conv2d(48,96,3,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d(1));self.h=nn.Linear(96,1)
 def forward(self,a,b): return self.h(self.f(torch.cat([a,b],1)).flatten(1)).squeeze(1)

def auc(y,s):
 y=np.asarray(y);s=np.asarray(s);order=np.argsort(s);rank=np.empty(len(s));rank[order]=np.arange(1,len(s)+1);p=y.sum();n=len(y)-p
 return float((rank[y==1].sum()-p*(p+1)/2)/(p*n))

def eval_model(model,dl,dev):
 model.eval();ys=[];raw=[];sc=[]
 with torch.no_grad():
  for a,b,y in dl:
   ys.append(y.numpy());raw.append(seam_raw(a.numpy(),b.numpy()));sc.append(model(a.to(dev),b.to(dev)).cpu().numpy())
 y=np.concatenate(ys);return {'cnn_auc':auc(y,np.concatenate(sc)),'raw_auc':auc(y,np.concatenate(raw)),'count':int(len(y)),'finite':bool(np.isfinite(np.concatenate(sc)).all())}

def g0(args):
 seed_all();rng=np.random.default_rng(SEED); tile=np.zeros((20,20,3),np.float32);tile[:,:,0]=np.arange(20)[None,:]/19
 a=side(tile,0,10,False);b=side(tile,0,10,True);wrong=side(np.zeros_like(tile),0,10,True);v=side(tile,1,10,False)
 # Explicit candidate-ID, not slot, augmentation identity contract.
 cand=np.array([[2,5,7],[5,2,7]],np.int64); score=np.array([[.1,.2,.3],[.4,.5,.6]],np.float32); extra=np.array([.7,.1,.9,.2,.3,.4,.8,.6],np.float32)
 def add(c,s,alpha): return s+alpha*extra[c]
 order=np.array([2,0,1]); shuffled=add(cand[:,order],score[:,order],.2); restored=shuffled[:,np.argsort(order)]
 report={'experiment':'P19_MDEC24','gate':'G0_input_only_contract','positive_internal_continuity':bool(float(seam_raw(a[None],b[None])[0]) > float(seam_raw(a[None],wrong[None])[0])),'axis_exchange_shape':list(v.shape),'alpha_zero_identity':bool(np.array_equal(add(cand,score,0.),score)),'candidate_id_invariant':bool(np.allclose(restored,add(cand,score,.2))),'finite':bool(np.isfinite(add(cand,score,.2)).all()),'targets_opened':False,'labels_used':False,'p8_imported':False}
 report['passes_G0']=all([report['positive_internal_continuity'],report['axis_exchange_shape']==[3,20,7],report['alpha_zero_identity'],report['candidate_id_invariant'],report['finite']])
 args.work.mkdir(parents=True,exist_ok=True);(args.work/'p19_g0_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
 if not report['passes_G0']:raise RuntimeError('G0 failed')

def train(args):
 seed_all();dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 if dev.type!='cuda':raise RuntimeError('P19 requires interactive local GPU')
 tr,he=p13.source_lists(args.manifest); tr,he=sorted(tr)[:128],sorted(he)[:32]
 ds=EdgePairs(args.inputs,tr);ev=EdgePairs(args.inputs,he,per_source=64,training=False)
 dl=DataLoader(ds,batch_size=512,shuffle=True,num_workers=0,pin_memory=True);edl=DataLoader(ev,batch_size=512,num_workers=0)
 m=EdgeNet().to(dev);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);lossf=nn.BCEWithLogitsLoss();hist=[]
 for epoch in range(12):
  m.train();losses=[]
  for a,b,y in dl:
   opt.zero_grad(set_to_none=True);z=m(a.to(dev),b.to(dev));loss=lossf(z,y.to(dev));loss.backward();opt.step();losses.append(float(loss.detach().cpu()))
  met=eval_model(m,edl,dev);met.update(epoch=epoch+1,loss=float(np.mean(losses)));hist.append(met);print(json.dumps(met),flush=True)
 out={'experiment':'P19_MDEC24','gate':'G1_input_only_auroc','history':hist,'best':max(hist,key=lambda x:x['cnn_auc']),'targets_opened':False,'labels_used':False,'p8_imported':False,'device':str(dev)}
 out['passes_G1']=bool(out['best']['finite'] and out['best']['cnn_auc']>=out['best']['raw_auc']+.03)
 args.work.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':m.state_dict(),'history':hist},args.work/'p19_mdec.pt');(args.work/'p19_g1_report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
 if not out['passes_G1']:raise RuntimeError('P19 G1 AUROC gate failed')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['g0','train'],required=True);ap.add_argument('--inputs',type=Path,default=Path(r'E:\pazzle_data\train\inputs'));ap.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));ap.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P19_mdec'));args=ap.parse_args();assert_safe(args.inputs,args.manifest,args.work);g0(args) if args.mode=='g0' else train(args)
if __name__=='__main__':main()
