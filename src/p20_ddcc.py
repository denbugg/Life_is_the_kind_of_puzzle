"""P20 DDCC-24. Pre-registered in P20_PRE_REGISTRATION.md before implementation."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
SRC=Path(__file__).resolve().parent
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
import p13_component_pose as p13
GRID,TILE,N,W=24,20,576,128

def sha(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+repr(a.shape).encode()+a.tobytes()).hexdigest()
def safe(*paths):
 if 'p8' in '\n'.join(str(x).lower() for x in paths): raise RuntimeError('P8 assets are prohibited')
def board(root,name):
 x=np.asarray(Image.open(root/name).convert('RGB'),np.float32)/255.
 if x.shape!=(480,480,3):raise RuntimeError(f'bad board shape {x.shape}')
 return x.reshape(GRID,TILE,GRID,TILE,3).transpose(0,2,1,3,4).reshape(N,TILE,TILE,3)
def analytic(a,b,d):
 """Three directional, orientation-respecting seam features for B candidate tiles."""
 if d==0: av,bv=a[:,-1,:],b[:,:,0,:]; an=a[:,-1,:]-a[:,-2,:];bn=b[:,:,1,:]-b[:,:,0,:]
 elif d==1: av,bv=a[-1,:,:],b[:,0,:,:]; an=a[-1,:,:]-a[-2,:,:];bn=b[:,1,:,:]-b[:,0,:,:]
 elif d==2: av,bv=a[:,0,:],b[:,:,-1,:]; an=a[:,1,:]-a[:,0,:];bn=b[:,:,-1,:]-b[:,:,-2,:]
 else: av,bv=a[0,:,:],b[:,-1,:,:]; an=a[1,:,:]-a[0,:,:];bn=b[:,-1,:,:]-b[:,-2,:,:]
 av=np.asarray(av); bv=np.asarray(bv)
 return np.stack([np.abs(av-bv).mean(axis=(-2,-1)),np.abs(np.diff(av,axis=-2)-np.diff(bv,axis=-2)).mean(axis=(-2,-1)),np.abs(an-bn).mean(axis=(-2,-1))],axis=-1).astype(np.float32)
def feature_block(tiles,cand,scores,valid,i,d,slots):
 ids=cand[i,slots]; raw=analytic(tiles[i],tiles[ids],d)
 good=np.flatnonzero(valid[i]);ordered=good[np.argsort(-scores[d,i,good],kind='stable')]
 ranks=np.empty(W,np.float32); ranks.fill(np.nan); ranks[ordered]=np.arange(len(ordered),dtype=np.float32)/max(1,len(ordered)-1)
 return np.column_stack([scores[d,i,slots],raw,ranks[slots]]).astype(np.float32)
def target_map(label_dir,name):
 with np.load(label_dir/(Path(name).stem+'.npz'),allow_pickle=False) as z:
  pos=z['target_tile_to_slot'].astype(np.int32,copy=True);recorded=str(z['source'])
 if recorded!=name:raise RuntimeError(f'label cache source mismatch {recorded} != {name}')
 if np.unique(pos).size!=N or pos.min()!=0 or pos.max()!=N-1:raise RuntimeError('label target permutation invalid')
 inv=np.empty(N,np.int32);inv[pos]=np.arange(N,dtype=np.int32);return pos,inv
def neighbor(pos,inv,i,d):
 r,c=divmod(int(pos[i]),GRID)
 if d==0: c+=1
 elif d==1:r+=1
 elif d==2:c-=1
 else:r-=1
 return -1 if r<0 or r>=GRID or c<0 or c>=GRID else int(inv[r*GRID+c])
def g0(a):
 tile=np.zeros((20,20,3),np.float32);tile[:,:,0]=np.arange(20,dtype=np.float32)[None,:]/19
 nxt=tile.copy();nxt[:,:,0]+=20/19;wrong=np.flip(nxt,axis=1).copy()
 ftrue=analytic(tile,nxt[None],0)[0];fwrong=analytic(tile,wrong[None],0)[0];fv=analytic(np.transpose(tile,(1,0,2)),np.transpose(nxt,(1,0,2))[None],1)[0]
 cand=np.array([[2,5,7],[5,2,7]],np.int64);score=np.array([[.1,.2,.3],[.4,.5,.6]],np.float32);aux=np.array([.3,.1,.2,.7,.5,.4,.9,.6],np.float32);order=np.array([2,0,1]);add=lambda c,s,z:s+z*aux[c]
 r={'experiment':'P20_DDCC24','gate':'G0','true_less_discontinuous':bool(ftrue[0]<fwrong[0]),'transpose_consistent':bool(np.allclose(ftrue,fv,atol=1e-6)),'alpha_zero_identity':bool(np.array_equal(add(cand,score,0),score)),'candidate_id_invariant':bool(np.allclose(add(cand[:,order],score[:,order],.2)[:,np.argsort(order)],add(cand,score,.2))),'finite':bool(np.isfinite(np.r_[ftrue,fwrong,fv]).all()),'ftrue':ftrue.tolist(),'fwrong':fwrong.tolist(),'transpose':fv.tolist(),'labels_used':False,'targets_opened':False,'p8_imported':False}
 r['passes_G0']=all(r[k] for k in ['true_less_discontinuous','transpose_consistent','alpha_zero_identity','candidate_id_invariant','finite']);a.work.mkdir(parents=True,exist_ok=True);(a.work/'p20_g0_report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
 if not r['passes_G0']:raise RuntimeError('P20 G0 failed')
def g1(a):
 fit,_=p13.source_lists(a.manifest);rows=[]
 for name in sorted(fit)[:4]:
  t=board(a.inputs,name);c,v,s=p13.load_score_cache(a.score_dir,name);parts=[]
  for d in range(4):
   for i in range(N):
    slots=np.flatnonzero(v[i])[:16]
    if slots.size:parts.append(feature_block(t,c,s,v,i,d,slots))
  z=np.concatenate(parts);rows.append({'source':name,'count':int(len(z)),'feature_sha256':sha(z),'finite':bool(np.isfinite(z).all()),'mean':z.mean(0).tolist(),'std':z.std(0).tolist()})
 r={'experiment':'P20_DDCC24','gate':'G1','rows':rows,'labels_used':False,'targets_opened':False,'p8_imported':False};r['passes_G1']=bool(all(x['finite'] and x['count']>0 for x in rows));a.work.mkdir(parents=True,exist_ok=True);(a.work/'p20_g1_report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
 if not r['passes_G1']:raise RuntimeError('P20 G1 failed')
def train_rows(a,names):
 xx=[];yy=[];coverage=[]
 for k,name in enumerate(names):
  t=board(a.inputs,name);c,v,s=p13.load_score_cache(a.score_dir,name);pos,inv=target_map(a.label_dir,name);hits=total=0
  for d in range(4):
   for i in range(N):
    truth=neighbor(pos,inv,i,d)
    if truth<0:continue
    total+=1; good=np.flatnonzero(v[i]); match=good[c[i,good]==truth]
    if not len(match):continue
    hits+=1;positive=int(match[0]); hard=good[np.argsort(-s[d,i,good],kind='stable')]; hard=hard[hard!=positive][:15];slots=np.r_[positive,hard].astype(np.int32)
    xx.append(feature_block(t,c,s,v,i,d,slots));yy.append(np.r_[1,np.zeros(len(hard),np.int8)])
  coverage.append({'source':name,'positive_coverage':hits/total if total else 0.0,'positives':hits})
  if (k+1)%8==0:print(json.dumps({'stage':'rows','sources_done':k+1,'samples':sum(len(y) for y in yy)}),flush=True)
 return np.concatenate(xx).astype(np.float32),np.concatenate(yy).astype(np.int8),coverage
def model_fit(x,y,C):
 try:from sklearn.linear_model import LogisticRegression
 except ImportError as e:raise RuntimeError('scikit-learn required for pre-registered L2 logistic G2') from e
 mu=x.mean(0);sd=x.std(0);sd=np.maximum(sd,1e-6);z=(x-mu)/sd
 m=LogisticRegression(C=C,penalty='l2',solver='lbfgs',class_weight='balanced',max_iter=250,tol=1e-5,n_jobs=1,random_state=20260817).fit(z,y)
 return {'mean':mu,'std':sd,'coef':m.coef_[0].astype(np.float32),'intercept':float(m.intercept_[0]),'C':float(C)}
def score_features(f,m):return (f-m['mean'])@m['coef']/m['std']+m['intercept']
def recall(a,names,m,labels=True):
 rows=[];allhits=[];basehits=[]
 for k,name in enumerate(names):
  t=board(a.inputs,name);c,v,s=p13.load_score_cache(a.score_dir,name);pos,inv=target_map(a.label_dir,name) if labels else (None,None);hit=base=tot=0
  for d in range(4):
   for i in range(N):
    truth=neighbor(pos,inv,i,d)
    if truth<0:continue
    good=np.flatnonzero(v[i]);f=feature_block(t,c,s,v,i,d,good);rank=good[np.argsort(-score_features(f,m),kind='stable')[:20]];brank=good[np.argsort(-s[d,i,good],kind='stable')[:20]];hit+=int(np.any(c[i,rank]==truth));base+=int(np.any(c[i,brank]==truth));tot+=1
  rows.append({'source':name,'recall20':hit/tot,'baseline_recall20':base/tot,'count':tot});allhits.append(hit/tot);basehits.append(base/tot)
  if (k+1)%8==0:print(json.dumps({'stage':'recall','sources_done':k+1}),flush=True)
 return float(np.mean(allhits)),float(np.mean(basehits)),rows
def g2(a):
 fit,held=p13.source_lists(a.manifest);x,y,cov=train_rows(a,sorted(fit));models=[]
 for C in [0.01,0.1,1.0]:
  m=model_fit(x,y,C);r,b,_=recall(a,sorted(fit),m);models.append((r,C,m));print(json.dumps({'C':C,'fit_recall20':r,'fit_baseline_recall20':b}),flush=True)
 r,C,m=max(models,key=lambda q:(q[0],-q[1]));np.savez(a.work/'p20_g2_model.npz',mean=m['mean'],std=m['std'],coef=m['coef'],intercept=np.float32(m['intercept']),C=np.float32(C));report={'experiment':'P20_DDCC24','gate':'G2','selected_C':C,'fit_recall20':r,'fit_baseline_recall20':b,'gain_pp':100*(r-b),'samples':int(len(y)),'positives':int(y.sum()),'coverage':cov,'labels_used':True,'targets_opened':False,'p8_imported':False,'passes_G2':bool(np.isfinite(r) and len(y)>0)};(a.work/'p20_g2_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
 if not report['passes_G2']:raise RuntimeError('P20 G2 failed')
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['g0','g1','g2']);p.add_argument('--inputs',type=Path,default=Path(r'E:\pazzle_data\train\inputs'));p.add_argument('--score-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache'));p.add_argument('--label-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache'));p.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));p.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P20_ddcc'));a=p.parse_args();safe(a.inputs,a.score_dir,a.label_dir,a.manifest,a.work);{'g0':g0,'g1':g1,'g2':g2}[a.mode](a)
if __name__=='__main__':main()
