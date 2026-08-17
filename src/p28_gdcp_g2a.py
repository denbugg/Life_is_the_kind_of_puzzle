"""P28 GDCP-24 G2a: bounded two-board edge-conditioned coordinate denoiser."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parent));from p28_gdcp import N

def p13():import p13_component_pose as p;return p
def coords(ld,n):
 with np.load(ld/(Path(n).stem+'.npz'),allow_pickle=False) as z:po=z['target_tile_to_slot'].astype(np.int32);src=str(z['source'])
 if src!=n or np.unique(po).size!=N:raise RuntimeError('bad label cache')
 iv=np.empty(N,np.int32);iv[po]=np.arange(N);r,c=np.divmod(iv,24);return np.stack([c/23.,r/23.],1).astype(np.float32)
def edges(sd,n):
 p=p13();c,v,s=p.load_score_cache(sd,n);a=[];b=[];d=[];w=[]
 for di in range(4):
  for i in range(N):
   j=np.flatnonzero(v[i]);j=j[np.argsort(-s[di,i,j])[:8]];a.extend([i]*len(j));b.extend(c[i,j]);d.extend([di]*len(j));w.extend(s[di,i,j])
 return tuple(torch.tensor(x) for x in [a,b,d,w])
class Net(torch.nn.Module):
 def __init__(self):super().__init__();self.m=torch.nn.Sequential(torch.nn.Linear(2+2+4+1,64),torch.nn.SiLU(),torch.nn.Linear(64,2));self.o=torch.nn.Sequential(torch.nn.Linear(4,64),torch.nn.SiLU(),torch.nn.Linear(64,2))
 def forward(self,x,e):
  a,b,d,w=e;a=a.long();b=b.long();h=torch.cat([x[a],x[b]-x[a],torch.nn.functional.one_hot(d.long(),4).float(),w[:,None]],1);z=self.m(h);agg=torch.zeros_like(x);agg.index_add_(0,a,z);cnt=torch.bincount(a,minlength=N).float().to(x.device)[:,None];return torch.sigmoid(self.o(torch.cat([x,agg/cnt.clamp_min(1)],1)))
def main():
 q=argparse.ArgumentParser();q.add_argument('--label-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache'));q.add_argument('--score-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache'));q.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));q.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P28_gdcp'));a=q.parse_args()
 if 'p8' in '\n'.join(map(str,[a.label_dir,a.score_dir,a.manifest,a.work])).lower():raise RuntimeError('P8 prohibited')
 if not torch.cuda.is_available():raise RuntimeError('interactive CUDA required')
 p=p13();fit,_=p.source_lists(a.manifest);ns=sorted(fit)[:2];dev=torch.device('cuda');ys=[torch.tensor(coords(a.label_dir,n),device=dev) for n in ns];es=[tuple(x.to(dev) for x in edges(a.score_dir,n)) for n in ns];torch.manual_seed(20260817);m=Net().to(dev);op=torch.optim.AdamW(m.parameters(),lr=2e-3);rng=torch.Generator(device=dev).manual_seed(20260817)
 for st in range(600):
  k=st%2;x=torch.rand((N,2),device=dev,generator=rng);pr=m(x,es[k]);loss=((pr-ys[k])**2).mean();op.zero_grad(set_to_none=True);loss.backward();op.step()
  if (st+1)%100==0:print(json.dumps({'stage':'capacity','step':st+1,'loss':float(loss)}),flush=True)
 rms=[]
 with torch.no_grad():
  for y,e in zip(ys,es):rms.append(float(torch.sqrt(((m(torch.full((N,2),.5,device=dev),e)-y)**2).mean())))
 rand=float(torch.sqrt(((torch.full((N,2),.5,device=dev)-ys[0])**2).mean()));rep={'experiment':'P28_GDCP24','gate':'G2a','sources':ns,'rmse':rms,'random_rmse':rand,'axis_orientation':True,'labels_used':True,'targets_opened':False,'p8_imported':False,'passes_G2a':bool(max(rms)<=.5*rand)};a.work.mkdir(parents=True,exist_ok=True);(a.work/'p28_g2a_report.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep),flush=True)
 if not rep['passes_G2a']:raise RuntimeError('P28 G2a reject')
if __name__=='__main__':main()
