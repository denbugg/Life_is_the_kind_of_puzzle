"""P33 CVA-24 G1: deterministic input-only frozen+DINO candidate union validation."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from p29_dpcg import N,load_tiles,desc,model,topm
import p13_component_pose as p13

def bad(*x):
 if 'p8' in '\n'.join(map(lambda q:str(q).lower(),x)):raise RuntimeError('P8 prohibited')
def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,default=Path(r'E:\pazzle_data\train\inputs'));p.add_argument('--scores',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache'));p.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P33_cva'));p.add_argument('--sources',nargs='*',default=('img_000002.png','img_000025.png','img_000098.png','img_000168.png','img_000172.png','img_000194.png','img_000223.png','img_000243.png','img_000267.png','img_000304.png','img_000344.png','img_000384.png','img_000426.png','img_000457.png','img_000480.png','img_000513.png'));a=p.parse_args();bad(a.inputs,a.scores,a.work);a.work.mkdir(parents=True,exist_ok=True);dev=torch.device('cuda');m=model(dev);rows=[]
 for k,n in enumerate(a.sources,1):
  st=time.perf_counter();c,v,_=p13.load_score_cache(a.scores,n);z=desc(m,load_tiles(a.inputs,n),dev);b=torch.stack([z[:,:,:,-1,:],z[:,-1,:,:],z[:,:,0,:],z[:,0,:,:]]);dn=[topm(b[d],b[(d+2)%4],64)[0] for d in range(4)];length=[];valid=True
  for d in range(4):
   for i in range(N):
    u=[]
    for q in list(dn[d][i])+list(c[i,v[i]]):
     q=int(q)
     if q!=i and q not in u:u.append(q)
     if len(u)==128:break
    valid &= 1<=len(u)<=128 and len(u)==len(set(u));length.append(len(u))
  sec=time.perf_counter()-st;rows.append({'source':n,'seconds':sec,'min_union':int(min(length)),'max_union':int(max(length)),'valid':bool(valid and sec<=90)})
  if k%4==0:print(json.dumps({'stage':'g1','done':k,'total':len(a.sources)}),flush=True)
 rep={'experiment':'P33_CVA24','gate':'G1','rows':rows,'labels_used':False,'targets_opened':False,'p8_imported':False,'passes_G1':bool(all(x['valid'] for x in rows))};(a.work/'p33_g1_report.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep),flush=True)
 if not rep['passes_G1']:raise RuntimeError('P33 G1 rejected')
if __name__=='__main__':main()
