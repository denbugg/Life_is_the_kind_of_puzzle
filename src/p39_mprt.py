"""P39 MPRT-24 image-only masked RGB pretraining; labels/targets never referenced."""
from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from torch.nn import functional as F
N=576

def tiles(inp,source):
 im=cv2.imread(str(inp/source),cv2.IMREAD_COLOR)
 if im is None or im.shape!=(480,480,3): raise RuntimeError('invalid input')
 im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
 a=im.reshape(24,20,24,20,3).transpose(0,2,4,1,3).reshape(N,3,20,20)
 return torch.from_numpy(a.copy()).float()/255.
class MAE(nn.Module):
 def __init__(self):
  super().__init__();self.e=nn.Sequential(nn.Conv2d(3,64,3,2,1),nn.GELU(),nn.Conv2d(64,128,3,2,1),nn.GELU(),nn.Conv2d(128,192,5),nn.GELU());self.d=nn.Sequential(nn.ConvTranspose2d(192,128,5),nn.GELU(),nn.ConvTranspose2d(128,64,4,2,1),nn.GELU(),nn.ConvTranspose2d(64,3,4,2,1),nn.Sigmoid())
 def forward(self,x): return self.d(self.e(x))
def mask(x):
 y=x.clone();m=torch.zeros_like(x[:,0:1]);
 for i in range(x.shape[0]):
  for _ in range(3):
   r=random.randrange(0,17,4);c=random.randrange(0,17,4);y[i,:,r:r+4,c:c+4]=0;m[i,:,r:r+4,c:c+4]=1
 return y,m
def g0(a):
 x=torch.rand(8,3,20,20);m=MAE();y=m(x);return {'experiment':'P39_MPRT24','gate':'G0','shape_ok':list(y.shape)==[8,3,20,20],'finite':bool(torch.isfinite(y).all()),'targets_opened':False,'p8_imported':False,'passes_G0':bool(torch.isfinite(y).all())}
def g1(a):
 split=json.loads(a.split.read_text());names=list(split['splits']['fit']);
 if len(names)!=5360 or len(set(names))!=5360:raise RuntimeError('pinned FIT list invalid')
 device=torch.device('cuda');random.seed(20260817);torch.manual_seed(20260817);net=MAE().to(device);opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.01);losses=[];t=time.perf_counter()
 for i,s in enumerate(names,1):
  x=tiles(a.inputs,s);idx=torch.randperm(N)[:96];x=x[idx].to(device);z,m=mask(x);pred=net(z);loss=((pred-x).abs()*m).sum()/m.sum().clamp_min(1);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach().cpu()))
  if i%200==0:print(json.dumps({'stage':'pretrain','done':i,'total':5360,'loss':float(np.mean(losses[-200:]))}),flush=True)
 first=float(np.mean(losses[:200]));last=float(np.mean(losses[-200:]));sec=time.perf_counter()-t;a.ckpt.parent.mkdir(parents=True,exist_ok=True);torch.save({'encoder':net.e.state_dict(),'fit_count':len(names)},a.ckpt)
 return {'experiment':'P39_MPRT24','gate':'G1','fit_sources':len(names),'first200_loss':first,'last200_loss':last,'decrease_fraction':(first-last)/first,'seconds':sec,'invalid':0,'targets_opened':False,'labels_opened':False,'p8_imported':False,'passes_G1':bool(np.isfinite(last) and last<=.9*first and sec<=2100)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['g0','g1'],required=True);p.add_argument('--inputs',type=Path);p.add_argument('--split',type=Path);p.add_argument('--ckpt',type=Path);p.add_argument('--report',type=Path,required=True);a=p.parse_args();r={'g0':g0,'g1':g1}[a.mode](a);a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
 if not r['passes_'+a.mode.upper()]:raise RuntimeError('P39 rejected')
if __name__=='__main__':main()
