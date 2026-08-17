"""P25 SCXR-24 G2a: bounded streamed P23 candidate-pool preparation after G0/G1."""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
N=576;T=20
def rss_gb():
 import ctypes
 class PMC(ctypes.Structure):_fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong),('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t)]
 p=PMC();p.cb=ctypes.sizeof(p);ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(p),p.cb);return p.WorkingSetSize/2**30
def p13():sys.path.insert(0,str(Path(__file__).resolve().parent));import p13_component_pose as x;return x
def load_tiles(ld,n,dev):
 with np.load(ld/(Path(n).stem+'.npz'),allow_pickle=False) as z:x=z['tiles_uint8'].copy();src=str(z['source'])
 if src!=n or x.shape!=(N,T,T,3):raise RuntimeError('bad approved label cache')
 return torch.from_numpy(x.transpose(0,3,1,2).copy()).to(dev,dtype=torch.float32).div_(255.)
def pool_one(n,a,retr,r23,dev):
 t0=time.perf_counter();p=p13();c,v,s=p.load_score_cache(a.score_dir,n);b=load_tiles(a.label_dir,n,dev);ids=torch.arange(N,device=dev);out=np.full((4,N,128),-1,np.int32)
 with torch.no_grad():
  for d in range(4):
   src=retr(r23.role_gpu(b[None],torch.zeros(N,device=dev,dtype=torch.long),ids,d,True));can=retr(r23.role_gpu(b[None],torch.zeros(N,device=dev,dtype=torch.long),ids,d,False));top=(src@can.T).fill_diagonal_(-1e9).topk(64,dim=1).indices.cpu().numpy()
   for i in range(N):
    seen=set();u=[]
    for z in np.r_[top[i],c[i,np.flatnonzero(v[i])]]:
     z=int(z)
     if z!=i and z not in seen:seen.add(z);u.append(z)
     if len(u)==128:break
    out[d,i,:len(u)]=u
 sec=time.perf_counter()-t0;gb=rss_gb()
 if sec>a.per_source_seconds:raise RuntimeError(f'per-source cap {sec:.2f}s')
 if gb>a.max_rss_gb:raise RuntimeError(f'RSS cap {gb:.2f}GB')
 return out,sec,gb
def main():
 q=argparse.ArgumentParser();q.add_argument('--mode',choices=['one','all'],required=True);q.add_argument('--source');q.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P25_scxr'));q.add_argument('--score-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache'));q.add_argument('--label-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache'));q.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));q.add_argument('--checkpoint',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P23_dctr\p23_g2_retriever_fp32.pt'));q.add_argument('--per-source-seconds',type=float,default=45);q.add_argument('--max-rss-gb',type=float,default=8);a=q.parse_args();
 if 'p8' in '\n'.join(map(lambda x:str(x).lower(),[a.work,a.score_dir,a.label_dir,a.checkpoint])):raise RuntimeError('P8 prohibited')
 if not torch.cuda.is_available():raise RuntimeError('interactive CUDA required')
 import p23_dctr as r23;dev=torch.device('cuda');retr=r23.Enc().to(dev);retr.load_state_dict(torch.load(a.checkpoint,map_location=dev,weights_only=True)['state']);retr.eval();p=p13();fit,_=p.source_lists(a.manifest);names=[a.source] if a.mode=='one' else sorted(fit)[:128];a.work.mkdir(parents=True,exist_ok=True);rows=[];tall=time.perf_counter()
 for k,n in enumerate(names):
  out,sec,gb=pool_one(n,a,retr,r23,dev);np.savez_compressed(a.work/(Path(n).stem+'.npz'),pool=out,source=n);rows.append({'source':n,'seconds':sec,'rss_gb':gb});print(json.dumps({'stage':'pool','done':k+1,'total':len(names),'source':n,'seconds':sec,'rss_gb':gb}),flush=True)
  if time.perf_counter()-tall>180*max(1,(len(names)+31)//32):raise RuntimeError('split total preparation cap')
 rep={'experiment':'P25_SCXR24','mode':a.mode,'rows':rows,'labels_used':True,'targets_opened':False,'p8_imported':False,'passes_G2a':True};(a.work/'p25_g2a_report.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep),flush=True)
if __name__=='__main__':main()
