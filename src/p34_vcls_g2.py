"""P34 VCLS-24 G2: vectorized FIT-only consensus-loop coverage gate."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from p34_vcls import N,RIGHT,DOWN,LEFT,UP,witness_masks
from p29_dpcg import load_tiles,desc,model,topm
import p13_component_pose as p13
GRID=24

def bad(*x):
 if 'p8' in '\n'.join(map(lambda q:str(q).lower(),x)):raise RuntimeError('P8 prohibited')
def labels(root,n):
 with np.load(root/(Path(n).stem+'.npz'),allow_pickle=False) as z:po=z['target_tile_to_slot'].astype(np.int32);src=str(z['source'])
 if src!=n or po.shape!=(N,) or np.unique(po).size!=N:raise RuntimeError('bad FIT labels')
 inv=np.empty(N,np.int32);inv[po]=np.arange(N,dtype=np.int32);return po,inv
def gt_neighbors(po,inv):
 out=np.full((2,N),-1,np.int32)
 for i in range(N):
  r,c=divmod(int(po[i]),GRID)
  if c+1<GRID:out[0,i]=inv[r*GRID+c+1]
  if r+1<GRID:out[1,i]=inv[(r+1)*GRID+c]
 return out
def masks(m,inputs,scores,n,dev):
 c,v,_=p13.load_score_cache(scores,n);z=desc(m,load_tiles(inputs,n),dev);b=torch.stack([z[:,:,-1,:],z[:,-1,:,:],z[:,:,0,:],z[:,0,:,:]]);dn=[topm(b[d],b[(d+2)%4],64)[0] for d in range(4)];u=np.zeros((4,N,N),bool);f=np.zeros((4,N,N),bool)
 for d in range(4):
  for i in range(N):
   fr=[int(q) for q in c[i,v[i]] if int(q)!=i];f[d,i,np.asarray(fr,np.int32)]=True
   row=[]
   for q in list(map(int,dn[d][i]))+fr:
    if q!=i and q not in row:row.append(q)
    if len(row)==128:break
   u[d,i,np.asarray(row,np.int32)]=True
 rw,dw=witness_masks(u);selr=rw & u[LEFT].T; seld=dw & u[UP].T
 return f,selr,seld
def cov(mask_r,mask_d,gt):
 total=0;hit=0
 for d,m in enumerate((mask_r,mask_d)):
  for i in range(N):
   q=int(gt[d,i])
   if q>=0:total+=1;hit+=int(m[i,q])
 return hit/total if total else 0.0
def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,default=Path(r'E:\pazzle_data\train\inputs'));p.add_argument('--scores',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache'));p.add_argument('--labels',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache'));p.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));p.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P34_vcls'));a=p.parse_args();bad(a.inputs,a.scores,a.labels,a.manifest,a.work);a.work.mkdir(parents=True,exist_ok=True);src,_=p13.source_lists(a.manifest);src=sorted(src)[:96];dev=torch.device('cuda');m=model(dev);base=[];sel=[];rows=[];start=time.perf_counter()
 for k,n in enumerate(src,1):
  st=time.perf_counter();f,r,d=masks(m,a.inputs,a.scores,n,dev);po,inv=labels(a.labels,n);gt=gt_neighbors(po,inv);br=f[RIGHT]&f[LEFT].T;bd=f[DOWN]&f[UP].T;b=cov(br,bd,gt);s=cov(r,d,gt);sec=time.perf_counter()-st;base.append(b);sel.append(s);rows.append({'source':n,'baseline':b,'selected':s,'seconds':sec,'valid':bool(sec<=90 and np.isfinite(s))})
  if k%4==0:print(json.dumps({'stage':'coverage','done':k,'total':len(src)}),flush=True)
  if time.perf_counter()-start>20*60:raise RuntimeError('P34 G2 total cap exceeded')
 rep={'experiment':'P34_VCLS24','gate':'G2','baseline_coverage':float(np.mean(base)),'selected_coverage':float(np.mean(sel)),'gain_pp':100*(float(np.mean(sel))-float(np.mean(base))),'invalid':int(sum(not x['valid'] for x in rows)),'rows':rows,'labels_used':True,'targets_opened':False,'p8_imported':False};rep['passes_G2']=bool(rep['gain_pp']>=2.0 and rep['invalid']==0);(a.work/'p34_g2_report.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep),flush=True)
 if not rep['passes_G2']:raise RuntimeError('P34 G2 rejected')
if __name__=='__main__':main()
