"""P7 FIT-only paired clean-corruption encoder pretraining G0/G1.

Only FIT source target crops are loaded.  Two independently challenge-matched
corruptions of an exact clean tile crop produce the contrastive pair.  No CAL,
DEV, test, solver, layout, or image-output dependency exists in this file.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from train_eval_cb1_g1_capacity import distort_frags, load_rgb, sha256_file, to_frags

GRID=24;N=576; BATCH=256
FIT_TARGETS=Path(r"E:\pazzle_data\train\targets")
SPLIT=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json")
WORK=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P7_pretrain_then_assemble\g0_g1_representation")

def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('g0','g1'));p.add_argument('--targets',type=Path,default=FIT_TARGETS);p.add_argument('--split',type=Path,default=SPLIT);p.add_argument('--work',type=Path,default=WORK);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=20260819);p.add_argument('--train-sources',type=int,default=256);p.add_argument('--eval-sources',type=int,default=32);p.add_argument('--steps',type=int,default=12000);p.add_argument('--width',type=int,default=128);return p.parse_args()

def sets(c):
 x=json.loads(c.split.read_text(encoding='utf-8'));fit=list(x['splits']['fit']);cal=set(x['splits']['cal']);dev=set(x['splits']['dev']);
 if len(fit)!=5360 or c.train_sources!=256 or c.eval_sources!=32 or set(fit)&(cal|dev):raise RuntimeError('P7 split contract')
 tr,ev=fit[576:832],fit[832:864]
 if set(tr)&set(ev) or any(k in cal or k in dev for k in tr+ev):raise RuntimeError('non-FIT source')
 for n in tr+ev:
  if not (c.targets/n).is_file():raise FileNotFoundError(c.targets/n)
 return tr,ev

class Encoder(nn.Module):
 def __init__(self,w):
  super().__init__();a=w//3;b=w//2;self.net=nn.Sequential(nn.Conv2d(3,a,3,padding=1,bias=False),nn.GroupNorm(max(1,a//16),a),nn.SiLU(),nn.Conv2d(a,b,3,2,1,bias=False),nn.GroupNorm(max(1,b//16),b),nn.SiLU(),nn.Conv2d(b,w,3,2,1,bias=False),nn.GroupNorm(max(1,w//16),w),nn.SiLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.LayerNorm(w))
 def forward(self,x):return self.net(x)
class Model(nn.Module):
 def __init__(self,w):
  super().__init__();self.encoder=Encoder(w);self.projector=nn.Sequential(nn.Linear(w,w),nn.SiLU(),nn.Linear(w,w));self.decoder=nn.Sequential(nn.Linear(w,w*5*5),nn.SiLU(),nn.Unflatten(1,(w,5,5)),nn.ConvTranspose2d(w,w//2,4,2,1),nn.SiLU(),nn.ConvTranspose2d(w//2,w//4,4,2,1),nn.SiLU(),nn.Conv2d(w//4,3,3,padding=1),nn.Sigmoid())
 def forward(self,x):
  h=self.encoder(x);return h,F.normalize(self.projector(h),dim=-1),self.decoder(h)

def tensor(frags,idx,d):return torch.from_numpy(frags[idx]).permute(0,3,1,2).contiguous().float().div_(255).to(d)
def paired(targets,name,seed,idx,d):
 clean=to_frags(load_rgb(targets/name));a=distort_frags(clean,np.random.default_rng(seed*1009+int(name[4:10])));b=distort_frags(clean,np.random.default_rng(seed*2029+int(name[4:10])));return tensor(clean,idx,d),tensor(a,idx,d),tensor(b,idx,d)
def loss(model,clean,a,b):
 _,za,recon=model(a);_,zb,_=model(b);lrec=torch.sqrt((recon-clean).pow(2)+1e-6).mean();logits=za@zb.T/.10;labels=torch.arange(za.shape[0],device=za.device);nce=(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))/2;return lrec+.25*nce,lrec,nce,recon

def g0(c):
 if c.device!='cuda' or not torch.cuda.is_available():raise RuntimeError('local CUDA required')
 tr,_=sets(c);d=torch.device('cuda');torch.manual_seed(c.seed);m=Model(c.width).to(d);rows=[]
 for i,n in enumerate(tr[:4]):
  idx=np.random.default_rng(c.seed+i).choice(N,BATCH,replace=False);clean,a,b=paired(c.targets,n,c.seed+i,idx,d);total,rec,nce,out=loss(m,clean,a,b);m.zero_grad(set_to_none=True);total.backward();finite=all(torch.isfinite(q).all() for q in (total,rec,nce,out));grad=all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());rows.append({'source':n,'unique_clean_indices':int(np.unique(idx).size),'loss':float(total.detach()),'reconstruction':float(rec.detach()),'infonce':float(nce.detach()),'finite':bool(finite),'finite_gradients':bool(grad)})
 passed=all(r['unique_clean_indices']==BATCH and r['finite'] and r['finite_gradients'] for r in rows);c.work.mkdir(parents=True,exist_ok=True);rep={'experiment':'P7_paired_encoder_pretraining','gate':'G0_pair_contract','passes':passed,'decision':'pass_to_G1_representation' if passed else 'reject_P7_before_training','checks':rows,'split_sha256':sha256_file(c.split),'CAL_target_opened':False,'DEV_targets_opened':False,'test_accessed':False,'layouts_assembled':False,'restorer_used':False};(c.work/'p7_g0_report.json').write_text(json.dumps(rep,indent=2),encoding='utf-8');print(json.dumps(rep,indent=2),flush=True)

def retrieval(embq,embclean,rawq,clean):
 sim=F.normalize(embq,dim=-1)@F.normalize(embclean,dim=-1).T;rank=torch.argsort(sim,dim=1,descending=True);truth=torch.arange(N,device=sim.device).unsqueeze(1);top20=float((rank[:,:20]==truth).any(dim=1).float().mean())
 l1=torch.abs(rawq.unsqueeze(1)-clean.unsqueeze(0)).mean(dim=(2,3,4));rankl=torch.argsort(l1,dim=1);l1top=float((rankl[:,:20]==truth).any(dim=1).float().mean());return top20,l1top

def g1(c):
 if c.device!='cuda' or not torch.cuda.is_available():raise RuntimeError('local CUDA required')
 if c.steps!=12000 or c.width!=128:raise ValueError('P7 fixed G1 contract')
 tr,ev=sets(c);d=torch.device('cuda');torch.manual_seed(c.seed);m=Model(c.width).to(d);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);rng=np.random.default_rng(c.seed+31);losses=[];recs=[];nces=[];m.train()
 for step in range(c.steps):
  n=tr[int(rng.integers(len(tr)))];idx=rng.choice(N,BATCH,replace=False);clean,a,b=paired(c.targets,n,c.seed*100000+step,idx,d);total,rec,nce,_=loss(m,clean,a,b);opt.zero_grad(set_to_none=True);total.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();losses.append(float(total.detach()));recs.append(float(rec.detach()));nces.append(float(nce.detach()));
  if (step+1)%250==0:print(f'step={step+1} loss={np.mean(losses[-100:]):.6f} rec={np.mean(recs[-100:]):.6f} nce={np.mean(nces[-100:]):.6f}',flush=True)
 m.eval();tops=[];l1tops=[];outl=[];idl=[]
 with torch.no_grad():
  idx=np.arange(N)
  for i,n in enumerate(ev):
   clean,a,b=paired(c.targets,n,c.seed+8000+i,idx,d);hqa,_,out=m(a);hclean,_,_=m(clean);top,l1top=retrieval(hqa,hclean,a,clean);tops.append(top);l1tops.append(l1top);outl.append(float(torch.abs(out-clean).mean()));idl.append(float(torch.abs(a-clean).mean()))
 ck=c.work/'p7_g1_encoder.pt';c.work.mkdir(parents=True,exist_ok=True);torch.save({'encoder':m.encoder.state_dict(),'model':m.state_dict(),'width':c.width,'seed':c.seed,'steps':c.steps},ck);retr=float(np.mean(tops));l1=float(np.mean(l1tops));out=float(np.mean(outl));identity=float(np.mean(idl));passed=bool(retr>=l1+.05 and out<=.9*identity);rep={'experiment':'P7_paired_encoder_pretraining','gate':'G1_FIT_representation_capacity','loss_first_100':float(np.mean(losses[:100])),'loss_last_100':float(np.mean(losses[-100:])),'reconstruction_first_100':float(np.mean(recs[:100])),'reconstruction_last_100':float(np.mean(recs[-100:])),'heldout_embedding_top20':retr,'heldout_raw_L1_top20':l1,'top20_delta_pp':100*(retr-l1),'heldout_decoder_clean_L1':out,'heldout_corrupted_identity_L1':identity,'reconstruction_relative_improvement':1-out/identity,'pass_criteria':'embedding top20 >= raw L1 +5pp; decoder L1 <=90% identity','passes_G1':passed,'decision':'pass_to_frozen_encoder_position_gate' if passed else 'reject_P7_before_global_position_CAL','checkpoint':str(ck),'checkpoint_sha256':sha256_file(ck),'train_sources':tr,'heldout_sources':ev,'split_sha256':sha256_file(c.split),'CAL_target_opened':False,'DEV_targets_opened':False,'test_accessed':False,'layouts_assembled':False,'restorer_used':False};(c.work/'p7_g1_report.json').write_text(json.dumps(rep,indent=2),encoding='utf-8');print(json.dumps(rep,indent=2),flush=True)
def main():
 c=args();random.seed(c.seed);np.random.seed(c.seed);torch.manual_seed(c.seed);g0(c) if c.phase=='g0' else g1(c)
if __name__=='__main__':main()
