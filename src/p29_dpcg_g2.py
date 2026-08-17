"""P29 G2 dense descriptor candidate coverage only after G1."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parent));from p29_dpcg import load_tiles,desc,model,topm,N

def p13():import p13_component_pose as p;return p
def labs(d,n):
 with np.load(d/(Path(n).stem+'.npz'),allow_pickle=False) as z:po=z['target_tile_to_slot'].astype(np.int32);src=str(z['source'])
 if src!=n or np.unique(po).size!=N:raise RuntimeError('bad labels')
 iv=np.empty(N,np.int32);iv[po]=np.arange(N);return po,iv
def nb(po,iv,i,d):
 r,c=divmod(int(iv[i]),24);r+=d==1;r-=d==3;c+=d==0;c-=d==2
 return -1 if r<0 or r>=24 or c<0 or c>=24 else int(po[r*24+c])
def dense(z):
 b=torch.stack([z[:,:,-1,:],z[:,-1,:,:],z[:,:,0,:],z[:,0,:,:]]);out=[]
 for d in range(4):ii,_=topm(b[d],b[(d+2)%4],64);out.append(ii)
 return np.stack(out)
def main():
 q=argparse.ArgumentParser();q.add_argument('--inputs',type=Path,default=Path(r'E:\pazzle_data\train\inputs'));q.add_argument('--label-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache'));q.add_argument('--score-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache'));q.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));q.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P29_dpcg'));a=q.parse_args()
 if 'p8' in '\n'.join(map(str,[a.inputs,a.label_dir,a.score_dir,a.manifest,a.work])).lower():raise RuntimeError('P8 prohibited')
 dev=torch.device('cuda');m=model(dev);p=p13();fit,_=p.source_lists(a.manifest);names=sorted(fit)[:128];cnt={16:[0,0],32:[0,0],64:[0,0]}
 for k,n in enumerate(names):
  z=desc(m,load_tiles(a.inputs,n),dev);dn=dense(z);po,iv=labs(a.label_dir,n);c,v,s=p.load_score_cache(a.score_dir,n)
  for d in range(4):
   for i in range(N):
    qid=nb(po,iv,i,d)
    if qid<0:continue
    fr=c[i,np.flatnonzero(v[i])]
    for M in cnt:
     u=[]
     for x in list(dn[d,i,:M])+list(fr):
      if int(x)!=i and int(x) not in u:u.append(int(x))
      if len(u)==128:break
     cnt[M][0]+=int(qid in u);cnt[M][1]+=int(qid in fr)
  if (k+1)%16==0:print(json.dumps({'stage':'coverage','done':k+1,'total':128}),flush=True)
 rows=[]
 for M,(h,b) in cnt.items():rows.append({'M':M,'union_coverage':h/(128*4*552),'frozen_coverage':b/(128*4*552),'gain_pp':100*(h-b)/(128*4*552)})
 best=max(rows,key=lambda x:x['gain_pp']);rep={'experiment':'P29_DPCG24','gate':'G2','rows':rows,'selected':best,'labels_used':True,'targets_opened':False,'p8_imported':False,'passes_G2':bool(best['gain_pp']>=2)};a.work.mkdir(parents=True,exist_ok=True);(a.work/'p29_g2_report.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep),flush=True)
 if not rep['passes_G2']:raise RuntimeError('P29 G2 coverage reject')
if __name__=='__main__':main()
