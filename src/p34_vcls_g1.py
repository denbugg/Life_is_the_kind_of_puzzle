"""P34 VCLS-24 G1: input-only vectorized consensus-loop candidate validation."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from p34_vcls import N,RIGHT,DOWN,LEFT,UP,witness_masks
from p29_dpcg import load_tiles,desc,model,topm
import p13_component_pose as p13

SRC=('img_000025.png','img_000098.png','img_000155.png','img_000168.png','img_000172.png','img_000215.png','img_000254.png','img_000266.png','img_000425.png','img_000437.png','img_000449.png','img_000486.png','img_000538.png','img_000653.png','img_000664.png','img_000667.png')

def bad(*x):
 if 'p8' in '\n'.join(map(lambda q:str(q).lower(),x)):raise RuntimeError('P8 prohibited')
def one(m,inputs,scores,n,dev):
 c,v,_=p13.load_score_cache(scores,n);z=desc(m,load_tiles(inputs,n),dev);b=torch.stack([z[:,:,-1,:],z[:,-1,:,:],z[:,:,0,:],z[:,0,:,:]]);dn=[topm(b[d],b[(d+2)%4],64)[0] for d in range(4)];mask=np.zeros((4,N,N),dtype=bool);ii=np.arange(N)
 for d in range(4):
  rows=[]
  for i in range(N):
   u=[]
   for q in list(map(int,dn[d][i]))+list(map(int,c[i,v[i]])):
    if q!=i and q not in u:u.append(q)
    if len(u)==128:break
   rows.append(u)
   if u:mask[d,i,np.asarray(u,np.int32)]=True
 rw,dw=witness_masks(mask);sr=rw & mask[LEFT].T;sd=dw & mask[UP].T
 return {'source':n,'right_candidates':int(mask[RIGHT].sum()),'down_candidates':int(mask[DOWN].sum()),'right_selected':int(sr.sum()),'down_selected':int(sd.sum()),'valid':bool(np.isfinite(mask).all() and sr.shape==(N,N) and sd.shape==(N,N))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,default=Path(r'E:\pazzle_data\train\inputs'));p.add_argument('--scores',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache'));p.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P34_vcls'));p.add_argument('--sources',nargs='*',default=SRC);a=p.parse_args();bad(a.inputs,a.scores,a.work);a.work.mkdir(parents=True,exist_ok=True);dev=torch.device('cuda');m=model(dev);rows=[]
 for k,n in enumerate(a.sources,1):
  st=time.perf_counter();r=one(m,a.inputs,a.scores,n,dev);r['seconds']=time.perf_counter()-st;r['valid'] &= r['seconds']<=90;rows.append(r)
  if k%4==0:print(json.dumps({'stage':'g1','done':k,'total':len(a.sources)}),flush=True)
 rep={'experiment':'P34_VCLS24','gate':'G1','rows':rows,'labels_used':False,'targets_opened':False,'p8_imported':False,'passes_G1':bool(all(x['valid'] for x in rows))};(a.work/'p34_g1_report.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep),flush=True)
 if not rep['passes_G1']:raise RuntimeError('P34 G1 rejected')
if __name__=='__main__':main()
