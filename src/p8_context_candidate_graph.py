"""P8 FIT-only context-aware virtual-halo candidate-graph capacity harness.

No CAL/DEV/test path, image assembly, restorer, NLM, or submission is imported.
The graph is generated from frozen rank96 candidate utilities on independently
corrupted FIT bags.  P7's frozen encoder is used only as a visual feature map.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import infer_rank96 as rank96
import p3_g1_cdcs_capacity as p3
from eval_candidate_rank import score_full_graph
from train_eval_cb1_g1_capacity import distort_frags, load_rgb, sha256_file, to_frags
from train_offset_pose import mine_affinity_candidates
from p7_pretrain_encoder import Encoder

GRID=24;N=576;K=32
FIT_TARGETS=Path(r"E:\pazzle_data\train\targets")
SPLIT=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json")
P7_CKPT=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P7_pretrain_then_assemble\g0_g1_representation\p7_g1_encoder.pt")
WORK=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P8_context_candidate_graph\g0_g1_capacity")

def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('g0','prepare','train'));p.add_argument('--targets',type=Path,default=FIT_TARGETS);p.add_argument('--split',type=Path,default=SPLIT);p.add_argument('--p7',type=Path,default=P7_CKPT);p.add_argument('--work',type=Path,default=WORK);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=20260820);p.add_argument('--train-sources',type=int,default=128);p.add_argument('--eval-sources',type=int,default=32);p.add_argument('--steps',type=int,default=4000);p.add_argument('--batch-queries',type=int,default=12);p.add_argument('--eval-queries',type=int,default=256);return p.parse_args()

def sources(c):
 x=json.loads(c.split.read_text(encoding='utf-8'));fit=list(x['splits']['fit']);cal=set(x['splits']['cal']);dev=set(x['splits']['dev'])
 if len(fit)!=5360 or c.train_sources!=128 or c.eval_sources!=32 or set(fit)&(cal|dev):raise RuntimeError('P8 fixed source contract')
 tr,ev=fit[864:992],fit[992:1024]
 if set(tr)&set(ev) or any(n in cal or n in dev for n in tr+ev):raise RuntimeError('non-FIT source')
 for n in tr+ev:
  if not(c.targets/n).is_file():raise FileNotFoundError(c.targets/n)
 return tr,ev

def sha(a):return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def neighbor(slot,d):
 r,c=divmod(int(slot),GRID)
 if d==0:return None if c==GRID-1 else slot+1
 if d==1:return None if r==GRID-1 else slot+GRID
 if d==2:return None if c==0 else slot-1
 if d==3:return None if r==0 else slot-GRID
 raise ValueError(d)
def cache_path(c,name):return c.work/'cache'/name.replace('.png','.npz')

def build_one(c,name,index,models,device):
 dst=cache_path(c,name)
 if dst.is_file():
  with np.load(dst,allow_pickle=False) as x:
   required={'tiles','anchors','directions','members','baseline','labels','permutation'}
   if required.issubset(x.files) and tuple(x['members'].shape)==(4*GRID*(GRID-1),K):return {'source':name,'cached':True,'cache':str(dst),'cache_sha256':sha256_file(dst)}
 print(f'cache_start source={name} index={index}',flush=True)
 clean=load_rgb(c.targets/name); frags=distort_frags(to_frags(clean),np.random.default_rng(c.seed*1009+index));perm=np.random.default_rng(c.seed*2029+index).permutation(N).astype(np.int32);tiles=frags[perm]
 ten=torch.from_numpy(tiles).permute(0,3,1,2).contiguous().float().to(device)
 with torch.no_grad():
  cand,valid=mine_affinity_candidates(models.affinity_primary,ten.unsqueeze(0),candidate_k=64,device=device,affinity_secondary=models.affinity_secondary)
  scores=score_full_graph(models.ranker,ten,cand[0],valid[0],pair_batch=4096,device=device)
 cn=cand[0].detach().cpu().numpy();sc=scores.detach().cpu().numpy();anchors,dirs,members=p3.hardlists(cn,valid[0].detach().cpu().numpy(),sc,perm)
 inv=np.empty(N,dtype=np.int32);inv[perm]=np.arange(N,dtype=np.int32); labels=np.empty(len(anchors),dtype=np.int16);base=np.empty((len(anchors),K),dtype=np.float32)
 for q,(a,d) in enumerate(zip(anchors,dirs)):
  truth=inv[neighbor(int(perm[int(a)]),int(d))];hits=np.flatnonzero(members[q]==truth)
  if len(hits)!=1:raise RuntimeError('P8 target absent/duplicate')
  labels[q]=int(hits[0])
  for j,m in enumerate(members[q]):
   pos=np.flatnonzero(cn[int(a),int(d)]==m)
   if len(pos)==1:
    base[q,j]=sc[int(a),int(d),int(pos[0])]
   elif int(m)==int(truth):
    # P3 hardlists legally injects an absent true neighbour; rank66 has no
    # score for it, so represent its frozen baseline as strictly worst.
    base[q,j]=-1.0e9
   else:
    raise RuntimeError('P8 unexpected non-rank96 non-target member')
 dst.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dst,tiles=tiles,anchors=anchors,directions=dirs,members=members,baseline=base,labels=labels,permutation=perm,source=np.array(name),seed=np.array(c.seed,dtype=np.int64))
 print(f'cache_done source={name} lists={len(anchors)}',flush=True)
 return {'source':name,'cached':False,'cache':str(dst),'cache_sha256':sha256_file(dst),'lists':int(len(anchors)),'target_label_minmax':[int(labels.min()),int(labels.max())]}

def load_p7(path,device):
 x=torch.load(path,map_location='cpu',weights_only=False);e=Encoder(int(x['width']));e.load_state_dict(x['encoder'],strict=True);e.to(device).eval()
 for p in e.parameters():p.requires_grad_(False)
 return e

def bands(tiles,anchors,dirs,members):
 # Canonical 3x20x4 directional boundary bands, using candidate pixels as legal virtual halo.
 out=[]
 for a,d,row in zip(anchors.tolist(),dirs.tolist(),members.tolist()):
  A=tiles[int(a)]
  for m in row:
   B=tiles[int(m)]
   if d==0:z=torch.cat((A[:,:,-2:],B[:,:,:2]),dim=2)
   elif d==1:z=torch.cat((A[:,:,-2:],B[:,:,:2]),dim=1).transpose(1,2)
   elif d==2:z=torch.cat((B[:,:,-2:],A[:,:,:2]),dim=2)
   else:z=torch.cat((B[:,:,-2:],A[:,:,:2]),dim=1).transpose(1,2)
   out.append(z)
 return torch.stack(out,0)
class Band(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.GroupNorm(4,32),nn.SiLU(),nn.Conv2d(32,64,3,padding=1),nn.GroupNorm(8,64),nn.SiLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
 def forward(self,x):return self.net(x)
class Block(nn.Module):
 def __init__(self,w,h):
  super().__init__();self.n1=nn.LayerNorm(w);self.a=nn.MultiheadAttention(w,h,batch_first=True,dropout=0.0);self.n2=nn.LayerNorm(w);self.f=nn.Sequential(nn.Linear(w,4*w),nn.SiLU(),nn.Linear(4*w,w))
 def forward(self,x):
  y=self.n1(x);x=x+self.a(y,y,y,need_weights=False)[0];return x+self.f(self.n2(x))
class Scorer(nn.Module):
 def __init__(self,context):
  super().__init__();self.context=context;self.band=Band();self.rank=nn.Embedding(K,16);self.direction=nn.Embedding(4,16);self.inp=nn.Sequential(nn.Linear(128+128+64+16+16,192),nn.SiLU(),nn.LayerNorm(192));self.blocks=nn.ModuleList([Block(192,8) for _ in range(4)]) if context else nn.ModuleList();self.local=nn.ModuleList([nn.Sequential(nn.LayerNorm(192),nn.Linear(192,768),nn.SiLU(),nn.Linear(768,192)) for _ in range(4)]) if not context else nn.ModuleList();self.out=nn.Sequential(nn.LayerNorm(192),nn.Linear(192,1))
 def forward(self,anchor,candidate,band,dirs):
  q,k=anchor.shape[0],candidate.shape[1];r=torch.arange(K,device=anchor.device).unsqueeze(0).expand(q,-1);x=torch.cat((anchor.unsqueeze(1).expand(-1,k,-1),candidate,self.band(band.reshape(q*k,3,20,4)).reshape(q,k,-1),self.rank(r),self.direction(dirs).unsqueeze(1).expand(-1,k,-1)),dim=-1);x=self.inp(x)
  for b in self.blocks:x=b(x)
  for b in self.local:x=x+b(x)
  return self.out(x).squeeze(-1)

def materialize(cache,ids,encoder,device):
 t=torch.from_numpy(cache['tiles']).permute(0,3,1,2).contiguous().float().div_(255).to(device);a=torch.from_numpy(cache['anchors'][ids].astype(np.int64)).to(device);d=torch.from_numpy(cache['directions'][ids].astype(np.int64)).to(device);m=torch.from_numpy(cache['members'][ids].astype(np.int64)).to(device);lab=torch.from_numpy(cache['labels'][ids].astype(np.int64)).to(device);base=torch.from_numpy(cache['baseline'][ids].astype(np.float32)).to(device)
 with torch.no_grad():all_e=F.normalize(encoder(t),dim=-1)
 return all_e[a],all_e[m],bands(t,a.cpu(),d.cpu(),m.cpu()).to(device),d,lab,base

def contract(c):
 if c.device!='cuda' or not torch.cuda.is_available():raise RuntimeError('local CUDA required')
 tr,_=sources(c);d=torch.device('cuda');models=rank96.load_models(p3.config(),d);rows=[]
 for i,n in enumerate(tr[:4]):rows.append(build_one(c,n,i,models,d))
 enc=load_p7(c.p7,d);cache=np.load(cache_path(c,tr[0]),allow_pickle=False);ids=np.arange(8);a,ca,b,di,la,ba=materialize(cache,ids,enc,d);torch.manual_seed(c.seed);m=Scorer(True).to(d);z=m(a,ca,b,di);loss=F.cross_entropy(z,la);m.zero_grad(set_to_none=True);loss.backward();p=torch.randperm(K,device=d);zp=m(a,ca[:,p],b.reshape(len(ids),K,3,20,4)[:,p].reshape(len(ids)*K,3,20,4),di);equiv=float((zp-z[:,p]).abs().max());finite=bool(torch.isfinite(z).all() and torch.isfinite(loss));unique=bool(all(len(np.unique(r))==K for r in cache['members'][ids]));target=bool(np.all((cache['members'][ids]==cache['members'][ids][np.arange(len(ids)),cache['labels'][ids],None]).sum(1)==1));rep={'experiment':'P8_context_aware_candidate_graph','gate':'G0_candidate_context_contract','passes':bool(finite and unique and target and equiv<1e-5),'decision':'pass_to_G1_capacity' if finite and unique and target and equiv<1e-5 else 'reject_P8_before_training','checks':rows,'max_candidate_permutation_abs':equiv,'finite':finite,'unique_members':unique,'target_present_once':target,'p7_checkpoint_sha256':sha256_file(c.p7),'split_sha256':sha256_file(c.split),'CAL_target_opened':False,'DEV_targets_opened':False,'test_accessed':False,'layouts_assembled':False,'restorer_used':False};c.work.mkdir(parents=True,exist_ok=True);(c.work/'p8_g0_report.json').write_text(json.dumps(rep,indent=2),encoding='utf-8');print(json.dumps(rep,indent=2),flush=True)

def prepare(c):
 if c.device!='cuda' or not torch.cuda.is_available():raise RuntimeError('local CUDA required')
 tr,ev=sources(c);d=torch.device('cuda');models=rank96.load_models(p3.config(),d);rows=[]
 for i,n in enumerate(tr+ev):
  rows.append(build_one(c,n,i,models,d))
 rep={'experiment':'P8_context_aware_candidate_graph','gate':'G1_cache_prepare','decision':'ready_for_G1_training','train_sources':tr,'heldout_sources':ev,'cache_rows':rows,'split_sha256':sha256_file(c.split),'CAL_target_opened':False,'DEV_targets_opened':False,'test_accessed':False,'layouts_assembled':False,'restorer_used':False};c.work.mkdir(parents=True,exist_ok=True);(c.work/'p8_prepare_report.json').write_text(json.dumps(rep,indent=2),encoding='utf-8');print(json.dumps({'experiment':rep['experiment'],'gate':rep['gate'],'decision':rep['decision'],'sources':len(rows)},indent=2),flush=True)

def train_model(c,context,train,held,enc,d):
 torch.manual_seed(c.seed+(0 if context else 1));m=Scorer(context).to(d);opt=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-4);rng=np.random.default_rng(c.seed+(11 if context else 13));losses=[];m.train()
 for step in range(c.steps):
  cache=train[int(rng.integers(len(train)))];ids=rng.choice(cache['anchors'].shape[0],size=c.batch_queries,replace=False);a,ca,b,di,la,_=materialize(cache,ids,enc,d);log=m(a,ca,b,di);loss=F.cross_entropy(log,la);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();losses.append(float(loss.detach()))
  if(step+1)%250==0:print(f"model={'context' if context else 'local'} step={step+1} loss={np.mean(losses[-100:]):.6f}",flush=True)
 m.eval();hit1=[];hit20=[];base1=[];base20=[]
 with torch.no_grad():
  for cache in held:
   rng2=np.random.default_rng(c.seed+int(cache['seed'])+(0 if context else 100));ids=rng2.choice(cache['anchors'].shape[0],size=c.eval_queries,replace=False);a,ca,b,di,la,base=materialize(cache,ids,enc,d);log=m(a,ca,b,di);rank=log.argsort(dim=1,descending=True);brank=base.argsort(dim=1,descending=True);hit1.append(float((rank[:,0]==la).float().mean()));hit20.append(float((rank[:,:20]==la[:,None]).any(1).float().mean()));base1.append(float((brank[:,0]==la).float().mean()));base20.append(float((brank[:,:20]==la[:,None]).any(1).float().mean()))
 ck=c.work/('p8_g1_context.pt' if context else 'p8_g1_local.pt');torch.save({'state_dict':m.state_dict(),'context':context,'seed':c.seed,'steps':c.steps},ck)
 return {'context':context,'loss_first_100':float(np.mean(losses[:100])),'loss_last_100':float(np.mean(losses[-100:])),'heldout_top1':float(np.mean(hit1)),'heldout_top20':float(np.mean(hit20)),'baseline_rank96_top1':float(np.mean(base1)),'baseline_rank96_top20':float(np.mean(base20)),'checkpoint':str(ck),'checkpoint_sha256':sha256_file(ck)}
def train(c):
 if c.device!='cuda' or not torch.cuda.is_available():raise RuntimeError('local CUDA required')
 if c.steps!=4000:raise ValueError('P8 fixed G1 step contract')
 trn,ev=sources(c);d=torch.device('cuda');enc=load_p7(c.p7,d);train=[dict(np.load(cache_path(c,n),allow_pickle=False)) for n in trn];held=[dict(np.load(cache_path(c,n),allow_pickle=False)) for n in ev]
 ctx=train_model(c,True,train,held,enc,d);loc=train_model(c,False,train,held,enc,d);delta_base=100*(ctx['heldout_top1']-ctx['baseline_rank96_top1']);delta_local=100*(ctx['heldout_top1']-loc['heldout_top1']);passed=bool(delta_base>=5 and delta_local>=3 and ctx['heldout_top20']>=ctx['baseline_rank96_top20']);rep={'experiment':'P8_context_aware_candidate_graph','gate':'G1_FIT_context_capacity','context_model':ctx,'local_only_ablation':loc,'context_vs_rank96_top1_pp':delta_base,'context_vs_local_top1_pp':delta_local,'pass_criteria':'context top1 >= rank96 +5pp and local +3pp; context top20 non-decreasing','passes_G1':passed,'decision':'pass_to_G2_FIT_score_decoder_alignment' if passed else 'reject_P8_before_solver_CAL','split_sha256':sha256_file(c.split),'p7_checkpoint_sha256':sha256_file(c.p7),'CAL_target_opened':False,'DEV_targets_opened':False,'test_accessed':False,'layouts_assembled':False,'restorer_used':False};(c.work/'p8_g1_report.json').write_text(json.dumps(rep,indent=2),encoding='utf-8');print(json.dumps(rep,indent=2),flush=True)
def main():
 c=args();random.seed(c.seed);np.random.seed(c.seed);torch.manual_seed(c.seed); contract(c) if c.phase=='g0' else prepare(c) if c.phase=='prepare' else train(c)
if __name__=='__main__':main()
