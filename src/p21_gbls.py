"""P21 GBLS-24. G0/G1 are input-only; G2 may read only the approved P10 label cache."""
from __future__ import annotations
import argparse, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
GRID,TILE,N,W=24,20,576,128

def sha(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+repr(a.shape).encode()+a.tobytes()).hexdigest()
def safe(*paths):
 if 'p8' in '\n'.join(str(x).lower() for x in paths):raise RuntimeError('P8 prohibited')
def tiles(root,name):
 x=np.asarray(Image.open(root/name).convert('RGB'),np.float32)/255.
 if x.shape!=(480,480,3):raise RuntimeError(f'bad board shape {x.shape}')
 return torch.from_numpy(x.reshape(GRID,TILE,GRID,TILE,3).transpose(0,2,4,1,3).reshape(N,3,TILE,TILE).copy())
def orient(t,a,b,d):
 left,right=t[a],t[b]
 if d==0:pass
 elif d==1:left,right=left.transpose(-2,-1),right.transpose(-2,-1)
 elif d==2:left,right=right,left
 elif d==3:left,right=right.transpose(-2,-1),left.transpose(-2,-1)
 else:raise ValueError(d)
 return left,right
def bridge(t,anchors,candidates,d):
 out=[];truth=[]
 for a,b in zip(anchors,candidates):
  left,right=orient(t,int(a),int(b),d)
  out.append(torch.cat([left[:,:,-8:-2],torch.zeros((3,TILE,4),dtype=left.dtype,device=left.device),right[:,:,2:8]],dim=-1))
  truth.append(torch.cat([left[:,:,-2:],right[:,:,:2]],dim=-1))
 return torch.stack(out),torch.stack(truth)
def bridge_gpu(tile_bank,rows):
 """rows Bx4 = source_index,input_anchor,input_candidate,direction."""
 board,anchor,cand,direction=(rows[:,i] for i in range(4));a=tile_bank[board,anchor];b=tile_bank[board,cand]
 left=a.clone();right=b.clone();down=(direction==1)|(direction==3);swap=(direction==2)|(direction==3)
 if down.any():left[down]=left[down].transpose(-2,-1);right[down]=right[down].transpose(-2,-1)
 if swap.any():tmp=left[swap].clone();left[swap]=right[swap];right[swap]=tmp
 x=torch.cat([left[:,:,:,-8:-2],torch.zeros((len(rows),3,TILE,4),device=a.device,dtype=a.dtype),right[:,:,:,2:8]],dim=-1)
 y=torch.cat([left[:,:,:,-2:],right[:,:,:,:2]],dim=-1)
 return x,y
class BridgeNet(nn.Module):
 """Tiny depthwise-separable FP32 bridge predictor; no discriminator or candidate label enters input."""
 def __init__(self):
  super().__init__();self.inp=nn.Conv2d(3,16,1);self.dw=nn.Conv2d(16,16,3,padding=1,groups=16);self.out=nn.Conv2d(16,3,1)
 def forward(self,x):return self.out(F.silu(self.dw(F.silu(self.inp(x)))))[:,:,:,6:10]
def p13mod():
 sys.path.insert(0,str(Path(__file__).resolve().parent));import p13_component_pose as p13;return p13
def label_map(label_dir,name):
 with np.load(label_dir/(Path(name).stem+'.npz'),allow_pickle=False) as z:
  pos=z['target_tile_to_slot'].astype(np.int32,copy=True);source=str(z['source'])
 if source!=name:raise RuntimeError(f'label cache source mismatch {source} != {name}')
 if np.unique(pos).size!=N or pos.min()!=0 or pos.max()!=N-1:raise RuntimeError('invalid target_tile_to_slot')
 inv=np.empty(N,np.int32);inv[pos]=np.arange(N,dtype=np.int32);return pos,inv
def directed_edges(pos):
 rows=[]
 for q in range(N):
  r,c=divmod(q,GRID);src=int(pos[q])
  if c<GRID-1:rows.append((src,int(pos[q+1]),0))
  if r<GRID-1:rows.append((src,int(pos[q+GRID]),1))
  if c>0:rows.append((src,int(pos[q-1]),2))
  if r>0:rows.append((src,int(pos[q-GRID]),3))
 return np.asarray(rows,np.int64)
def cache_tiles(label_dir,names,device):
 out=[]
 for name in names:
  with np.load(label_dir/(Path(name).stem+'.npz'),allow_pickle=False) as z: x=z['tiles_uint8'].copy()
  if x.shape!=(N,TILE,TILE,3):raise RuntimeError(f'bad cache tiles {name} {x.shape}')
  out.append(torch.from_numpy(x.transpose(0,3,1,2).copy()))
 return torch.stack(out).to(device=device,dtype=torch.float32).div_(255.)
def g0(a):
 z=torch.zeros((3,3,TILE,TILE),dtype=torch.float32);r=torch.arange(TILE,dtype=torch.float32)[None,:]/19;z[0,0]=r;z[1,0]=1+r;z[2,0]=2+r
 x,y=bridge(z,[0],[1],0);xt,yt=bridge(z.transpose(-2,-1).contiguous(),[0],[1],1);order=torch.tensor([2,0,1]);xp,yp=bridge(z,[0,0,0],order.tolist(),0);xi,yi=bridge(z,[0,0,0],torch.arange(3).tolist(),0);inv=torch.argsort(order)
 raw=np.array([[.2,.4,.6],[.1,.3,.5]],np.float32);res=np.array([[.7,.2,.9],[.5,.8,.1]],np.float32);fuse=lambda alpha:raw-alpha*res
 report={'experiment':'P21_GBLS24','gate':'G0','shape':list(x.shape),'mask_is_zero':bool(torch.count_nonzero(x[:,:,:,6:10])==0),'target_alignment':bool(torch.allclose(y[:,:,:,:2],z[0:1,:,:,-2:]) and torch.allclose(y[:,:,:,2:],z[1:2,:,:,:2])),'transpose_consistent':bool(torch.allclose(x,xt) and torch.allclose(y,yt)),'candidate_row_invariant':bool(torch.allclose(xp[inv],xi) and torch.allclose(yp[inv],yi)),'alpha_zero_identity':bool(np.array_equal(fuse(0),raw)),'finite':bool(torch.isfinite(x).all() and torch.isfinite(y).all() and np.isfinite(fuse(.2)).all()),'labels_used':False,'targets_opened':False,'p8_imported':False}
 report['passes_G0']=all(report[k] for k in ['mask_is_zero','target_alignment','transpose_consistent','candidate_row_invariant','alpha_zero_identity','finite']);a.work.mkdir(parents=True,exist_ok=True);(a.work/'p21_g0_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
 if not report['passes_G0']:raise RuntimeError('P21 G0 failed')
def g1(a):
 p13=p13mod();fit,_=p13.source_lists(a.manifest);rows=[]
 for name in sorted(fit)[:4]:
  t=tiles(a.inputs,name);c,v,s=p13.load_score_cache(a.score_dir,name);xx=[];yy=[]
  for d in range(4):
   for i in range(N):
    slots=np.flatnonzero(v[i])[:16]
    if slots.size:x,y=bridge(t,[i]*len(slots),c[i,slots].tolist(),d);xx.append(x.numpy());yy.append(y.numpy())
  x=np.concatenate(xx);y=np.concatenate(yy);rows.append({'source':name,'count':int(len(x)),'input_sha256':sha(x),'target_tensor_sha256':sha(y),'finite':bool(np.isfinite(x).all() and np.isfinite(y).all()),'masked_zero':bool(np.count_nonzero(x[:,:,:,6:10])==0)})
 report={'experiment':'P21_GBLS24','gate':'G1','rows':rows,'labels_used':False,'targets_opened':False,'p8_imported':False};report['passes_G1']=bool(all(r['count']>0 and r['finite'] and r['masked_zero'] for r in rows));a.work.mkdir(parents=True,exist_ok=True);(a.work/'p21_g1_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
 if not report['passes_G1']:raise RuntimeError('P21 G1 failed')
def recall_scores(a,names,bank,model,alpha,device):
 p13=p13mod();model.eval();values=[];bases=[];coverage=[]
 with torch.no_grad():
  for bi,name in enumerate(names):
   c,v,s=p13.load_score_cache(a.score_dir,name);pos,inv=label_map(a.label_dir,name);hit=base=covered=total=0
   for d in range(4):
    rows=[];meta=[]
    for i in range(N):
     slots=np.flatnonzero(v[i]);
     if slots.size:
      rows.extend((bi,i,int(c[i,j]),d) for j in slots);meta.append((i,slots.copy(),len(rows)-len(slots),len(rows)))
    residual=np.empty(len(rows),np.float32)
    for start in range(0,len(rows),a.eval_batch):
     z=torch.tensor(rows[start:start+a.eval_batch],device=device,dtype=torch.long);x,y=bridge_gpu(bank,z);pred=model(x);residual[start:start+len(z)]=(-F.smooth_l1_loss(pred,y,reduction='none').mean((1,2,3))).cpu().numpy()
    for i,slots,lo,hi in meta:
     r,c0=divmod(int(inv[i]),GRID);q=-1
     if d==0 and c0<GRID-1:q=r*GRID+c0+1
     elif d==1 and r<GRID-1:q=(r+1)*GRID+c0
     elif d==2 and c0>0:q=r*GRID+c0-1
     elif d==3 and r>0:q=(r-1)*GRID+c0
     if q<0:continue
     truth=int(pos[q]);total+=1;covered+=int(np.any(c[i,slots]==truth));z=(residual[lo:hi]-residual[lo:hi].mean())/(residual[lo:hi].std()+1e-6);rank=slots[np.argsort(-(s[d,i,slots]+alpha*z),kind='stable')[:20]];brank=slots[np.argsort(-s[d,i,slots],kind='stable')[:20]];hit+=int(np.any(c[i,rank]==truth));base+=int(np.any(c[i,brank]==truth))
   values.append(hit/total);bases.append(base/total);coverage.append(covered/total);print(json.dumps({'stage':'recall','source_done':bi+1,'sources_total':len(names),'alpha':alpha}),flush=True)
 return float(np.mean(values)),float(np.mean(bases)),float(np.mean(coverage))
def g2(a):
 if not torch.cuda.is_available():raise RuntimeError('P21 G2 requires local CUDA interactive session')
 torch.manual_seed(20260817);np.random.seed(20260817);random.seed(20260817);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
 p13=p13mod();fit,_=p13.source_lists(a.manifest);fit=sorted(fit);train_names,sel_names=fit[:96],fit[96:128];device=torch.device('cuda');a.work.mkdir(parents=True,exist_ok=True)
 bank=cache_tiles(a.label_dir,train_names+sel_names,device);train_rows=[]
 for bi,name in enumerate(train_names):
  pos,_=label_map(a.label_dir,name);e=directed_edges(pos);train_rows.append(np.column_stack([np.full(len(e),bi,np.int64),e]))
 rows=torch.from_numpy(np.concatenate(train_rows)).to(device=device,dtype=torch.long);model=BridgeNet().to(device);opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-4);best=float('inf');t0=time.time()
 for step in range(a.steps):
  idx=torch.randint(len(rows),(a.batch_size,),device=device);x,y=bridge_gpu(bank,rows[idx]);pred=model(x);loss=F.smooth_l1_loss(pred,y);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();best=min(best,float(loss.item()))
  if (step+1)%200==0:print(json.dumps({'stage':'train','step':step+1,'steps':a.steps,'loss':float(loss.item()),'best_loss':best,'seconds':round(time.time()-t0,1)}),flush=True)
 ckpt=a.work/'p21_g2_bridge_fp32.pt';torch.save({'state_dict':model.state_dict(),'steps':a.steps,'best_loss':best,'seed':20260817},ckpt)
 scores=[]
 for alpha in [0.0,0.05,0.10,0.20,0.40]:
  rec,base,cov=recall_scores(a,sel_names,bank[96:],model,alpha,device);scores.append({'alpha':alpha,'recall20':rec,'baseline_recall20':base,'gain_pp':100*(rec-base),'coverage':cov});print(json.dumps(scores[-1]),flush=True)
 selected=max(scores,key=lambda z:(z['recall20'],-z['alpha']));report={'experiment':'P21_GBLS24','gate':'G2','train_sources':len(train_names),'selection_sources':len(sel_names),'steps':a.steps,'batch_size':a.batch_size,'best_train_loss':best,'scores':scores,'selected':selected,'checkpoint':str(ckpt),'labels_used':True,'targets_opened':False,'p8_imported':False};report['passes_G2']=bool(selected['alpha']>0 and selected['gain_pp']>=1.0 and np.isfinite(selected['recall20']));(a.work/'p21_g2_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
 if not report['passes_G2']:raise RuntimeError('P21 G2 fast-futility reject')
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['g0','g1','g2']);p.add_argument('--inputs',type=Path,default=Path(r'E:\pazzle_data\train\inputs'));p.add_argument('--score-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache'));p.add_argument('--label-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache'));p.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));p.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P21_gbls'));p.add_argument('--steps',type=int,default=2000);p.add_argument('--batch-size',type=int,default=512);p.add_argument('--eval-batch',type=int,default=4096);a=p.parse_args();safe(a.inputs,a.score_dir,a.label_dir,a.manifest,a.work);{'g0':g0,'g1':g1,'g2':g2}[a.mode](a)
if __name__=='__main__':main()
