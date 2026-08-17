"""P22 FCLR-24 input-only contracts. See committed P22_PRE_REGISTRATION.md."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
GRID,TILE,N=24,20,576

def sha(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+repr(a.shape).encode()+a.tobytes()).hexdigest()
def safe(*p):
 if 'p8' in '\n'.join(str(x).lower() for x in p):raise RuntimeError('P8 prohibited')
def tiles(root,name):
 x=np.asarray(Image.open(root/name).convert('RGB'),np.float32)/255.
 if x.shape!=(480,480,3):raise RuntimeError(x.shape)
 return torch.from_numpy(x.reshape(GRID,TILE,GRID,TILE,3).transpose(0,2,4,1,3).reshape(N,3,TILE,TILE).copy())
def bands(t,anchors,candidates,d):
 out=[]
 for a,b in zip(anchors,candidates):
  left,right=t[int(a)],t[int(b)]
  if d==1:left,right=left.transpose(-2,-1),right.transpose(-2,-1)
  elif d==2:left,right=right,left
  elif d==3:left,right=right.transpose(-2,-1),left.transpose(-2,-1)
  elif d!=0:raise ValueError(d)
  out.append(torch.cat([left[:,:,-10:],right[:,:,:10]],dim=-1))
 return torch.stack(out)
def g0(a):
 z=torch.zeros((4,3,TILE,TILE),dtype=torch.float32);q=torch.arange(TILE,dtype=torch.float32)[None,:]/19
 for i in range(4):z[i,0]=i+q
 x=bands(z,[0],[1],0);xt=bands(z.transpose(-2,-1).contiguous(),[0],[1],1);base=torch.tensor([1,2,3]);order=torch.tensor([2,0,1]);xp=bands(z,[0,0,0],base[order].tolist(),0);xi=bands(z,[0,0,0],base.tolist(),0);raw=np.array([[.2,.4,.6],[.1,.3,.5]],np.float32);logit=np.array([[.7,.2,.9],[.5,.8,.1]],np.float32);f=lambda alpha:raw+alpha*logit
 r={'experiment':'P22_FCLR24','gate':'G0','shape':list(x.shape),'transpose_consistent':bool(torch.allclose(x,xt)),'candidate_row_invariant':bool(torch.allclose(xp[torch.argsort(order)],xi)),'alpha_zero_identity':bool(np.array_equal(f(0),raw)),'finite':bool(torch.isfinite(x).all() and np.isfinite(f(.2)).all()),'labels_used':False,'targets_opened':False,'p8_imported':False};r['passes_G0']=all(r[k] for k in ['transpose_consistent','candidate_row_invariant','alpha_zero_identity','finite']);a.work.mkdir(parents=True,exist_ok=True);(a.work/'p22_g0_report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
 if not r['passes_G0']:raise RuntimeError('P22 G0 failed')
def g1(a):
 sys.path.insert(0,str(Path(__file__).resolve().parent));import p13_component_pose as p13
 fit,_=p13.source_lists(a.manifest);rows=[]
 for name in sorted(fit)[:4]:
  t=tiles(a.inputs,name);c,v,s=p13.load_score_cache(a.score_dir,name);parts=[]
  for d in range(4):
   for i in range(N):
    slots=np.flatnonzero(v[i])[:16]
    if slots.size:parts.append(bands(t,[i]*len(slots),c[i,slots].tolist(),d).numpy())
  z=np.concatenate(parts);rows.append({'source':name,'count':int(len(z)),'band_sha256':sha(z),'finite':bool(np.isfinite(z).all())})
 r={'experiment':'P22_FCLR24','gate':'G1','rows':rows,'labels_used':False,'targets_opened':False,'p8_imported':False};r['passes_G1']=bool(all(x['count']>0 and x['finite'] for x in rows));a.work.mkdir(parents=True,exist_ok=True);(a.work/'p22_g1_report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
 if not r['passes_G1']:raise RuntimeError('P22 G1 failed')
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['g0','g1']);p.add_argument('--inputs',type=Path,default=Path(r'E:\pazzle_data\train\inputs'));p.add_argument('--score-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache'));p.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));p.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P22_fclr'));a=p.parse_args();safe(a.inputs,a.score_dir,a.manifest,a.work);g0(a) if a.mode=='g0' else g1(a)
# Legacy entrypoint deferred until the G2 extension below.

# P22 G2 extension appended after G0/G1 evidence commit.
class RankNet(torch.nn.Module):
 def __init__(self):
  super().__init__();self.a=torch.nn.Conv2d(3,32,3,padding=1);self.b=torch.nn.Conv2d(32,32,3,padding=1,groups=32);self.c=torch.nn.Conv2d(32,32,1);self.o=torch.nn.Linear(32,1)
 def forward(self,x):
  z=torch.nn.functional.silu(self.a(x));z=torch.nn.functional.silu(self.b(z));z=torch.nn.functional.silu(self.c(z));return self.o(z.mean((-2,-1))).squeeze(-1)
def p13mod():
 sys.path.insert(0,str(Path(__file__).resolve().parent));import p13_component_pose as p13;return p13
def labels(ld,n):
 with np.load(ld/(Path(n).stem+'.npz'),allow_pickle=False) as z:pos=z['target_tile_to_slot'].astype(np.int32,copy=True);src=str(z['source'])
 if src!=n or np.unique(pos).size!=N or pos.min()!=0 or pos.max()!=N-1:raise RuntimeError('bad approved label cache')
 inv=np.empty(N,np.int32);inv[pos]=np.arange(N,dtype=np.int32);return pos,inv
def nbr(pos,inv,i,d):
 r,c=divmod(int(inv[i]),GRID);r+=d==1;r-=d==3;c+=d==0;c-=d==2
 return -1 if r<0 or r>=GRID or c<0 or c>=GRID else int(pos[r*GRID+c])
def bank_from_labels(ld,names,dev):
 z=[]
 for n in names:
  with np.load(ld/(Path(n).stem+'.npz'),allow_pickle=False) as q:x=q['tiles_uint8'].copy()
  z.append(torch.from_numpy(x.transpose(0,3,1,2).copy()))
 return torch.stack(z).to(dev,dtype=torch.float32).div_(255.)
def band_gpu(bank,candbank,rows):
 # rows [B,3] = board, anchor, direction; result B,W,3,20,20.
 bi,ai,di=(rows[:,j] for j in range(3));ca=candbank[bi,ai];left=bank[bi,ai][:,None].expand(-1,ca.shape[1],-1,-1,-1).clone();right=bank[bi[:,None],ca].clone();down=(di==1)|(di==3);swap=(di==2)|(di==3)
 if down.any():left[down]=left[down].transpose(-2,-1);right[down]=right[down].transpose(-2,-1)
 if swap.any():u=left[swap].clone();left[swap]=right[swap];right[swap]=u
 return torch.cat([left[:,:,:,:,-10:],right[:,:,:,:,:10]],dim=-1)
def setup(a,names,dev):
 p13=p13mod();cb=[];vb=[];sb=[];groups=[]
 for bi,n in enumerate(names):
  c,v,s=p13.load_score_cache(a.score_dir,n);pos,inv=labels(a.label_dir,n);cb.append(c);vb.append(v);sb.append(s)
  for d in range(4):
   for i in range(N):
    q=nbr(pos,inv,i,d)
    if q<0:continue
    slots=np.flatnonzero(v[i]);m=slots[c[i,slots]==q]
    if len(m):groups.append((bi,i,d,int(m[0])))
 return torch.from_numpy(np.stack(cb)).to(dev),torch.from_numpy(np.stack(vb)).to(dev),np.stack(sb),torch.tensor(groups,device=dev,dtype=torch.long)
def eval_recall(a,names,bank,cand,valid,model,alpha,dev):
 p13=p13mod();vals=[];bases=[];cov=[];model.eval()
 with torch.no_grad():
  for bi,n in enumerate(names):
   c,v,s=p13.load_score_cache(a.score_dir,n);pos,inv=labels(a.label_dir,n);hit=base=covered=tot=0;rows=[];meta=[]
   for d in range(4):
    for i in range(N):
     q=nbr(pos,inv,i,d)
     if q<0:continue
     tot+=1;slots=np.flatnonzero(v[i]);truth=int(pos[q]);covered+=int(np.any(c[i,slots]==truth));rows.append((bi,i,d));meta.append((i,d,slots,truth))
   out=[]
   for st in range(0,len(rows),a.eval_batch):
    rr=torch.tensor(rows[st:st+a.eval_batch],device=dev);bb=band_gpu(bank,cand,rr).flatten(0,1);out.append(model(bb).reshape(len(rr),-1).cpu().numpy())
   out=np.concatenate(out)
   for k,(i,d,slots,truth) in enumerate(meta):
    z=out[k,slots];z=(z-z.mean())/(z.std()+1e-6);rank=slots[np.argsort(-(s[d,i,slots]+alpha*z),kind='stable')[:20]];brank=slots[np.argsort(-s[d,i,slots],kind='stable')[:20]];hit+=int(np.any(c[i,rank]==truth));base+=int(np.any(c[i,brank]==truth))
   vals.append(hit/tot);bases.append(base/tot);cov.append(covered/tot);print(json.dumps({'stage':'recall','done':bi+1,'total':len(names),'alpha':alpha}),flush=True)
 return float(np.mean(vals)),float(np.mean(bases)),float(np.mean(cov))
def g2(a):
 if not torch.cuda.is_available():raise RuntimeError('P22 G2 requires interactive local CUDA')
 torch.manual_seed(20260817);np.random.seed(20260817);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;p13=p13mod();fit,_=p13.source_lists(a.manifest);fit=sorted(fit);tr,sel=fit[:96],fit[96:128];dev=torch.device('cuda');a.work.mkdir(parents=True,exist_ok=True)
 bank=bank_from_labels(a.label_dir,tr+sel,dev);cand,valid,sc,grp=setup(a,tr+sel,dev);grp=grp[grp[:,0]<len(tr)];model=RankNet().to(dev);opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4);best=1e9
 for step in range(a.steps):
  g=grp[torch.randint(len(grp),(a.batch_size,),device=dev)];b=band_gpu(bank,cand,g[:,:3]).flatten(0,1);log=model(b).reshape(len(g),-1);mask=valid[g[:,0],g[:,1]];log=log.masked_fill(~mask,-1e9);loss=torch.nn.functional.cross_entropy(log,g[:,3]);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();best=min(best,float(loss.item()))
  if (step+1)%250==0:print(json.dumps({'stage':'train','step':step+1,'loss':float(loss.item()),'best':best}),flush=True)
 ck=a.work/'p22_g2_ranker_fp32.pt';torch.save({'state':model.state_dict(),'steps':a.steps,'best_loss':best},ck);scores=[]
 for alpha in [0.,.05,.10,.20,.40]:
  r,b,c=eval_recall(a,sel,bank[96:],cand[96:],valid[96:],model,alpha,dev);scores.append({'alpha':alpha,'recall20':r,'baseline_recall20':b,'gain_pp':100*(r-b),'coverage':c});print(json.dumps(scores[-1]),flush=True)
 z=max(scores,key=lambda x:(x['recall20'],-x['alpha']));rep={'experiment':'P22_FCLR24','gate':'G2','train_sources':96,'selection_sources':32,'groups':int(len(grp)),'steps':a.steps,'best_loss':best,'scores':scores,'selected':z,'checkpoint':str(ck),'labels_used':True,'targets_opened':False,'p8_imported':False,'passes_G2':bool(z['alpha']>0 and z['gain_pp']>=1.)};(a.work/'p22_g2_report.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep),flush=True)
 if not rep['passes_G2']:raise RuntimeError('P22 G2 fast-futility reject')
# Replace the original entrypoint only after retaining its G0/G1 functions.
def main2():
 p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['g0','g1','g2']);p.add_argument('--inputs',type=Path,default=Path(r'E:\pazzle_data\train\inputs'));p.add_argument('--score-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache'));p.add_argument('--label-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache'));p.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));p.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P22_fclr'));p.add_argument('--steps',type=int,default=2000);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--eval-batch',type=int,default=128);a=p.parse_args();safe(a.inputs,a.score_dir,a.label_dir,a.manifest,a.work);{'g0':g0,'g1':g1,'g2':g2}[a.mode](a)
if __name__=='__main__':main2()
