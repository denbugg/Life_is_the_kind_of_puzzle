"""P39 G2: raw-only relational fine-tuning from image-only MAE encoder."""
import argparse,json,random,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from p37_rit import N,raw_tiles,labels,splits,top20_recall
from p39_mprt import MAE
class Rel(nn.Module):
 def __init__(self,enc):
  super().__init__();self.e=enc;self.p=nn.Linear(192,256);b=nn.TransformerEncoderLayer(256,8,1024,batch_first=True,activation='gelu',norm_first=True);self.t=nn.TransformerEncoder(b,4);self.rq=nn.Linear(256,256,False);self.rk=nn.Linear(256,256,False);self.dq=nn.Linear(256,256,False);self.dk=nn.Linear(256,256,False)
 def forward(self,x):
  b,n=x.shape[:2];z=self.e(x.reshape(b*n,3,20,20)).flatten(1).reshape(b,n,192);z=self.t(self.p(z));r=self.rq(z)@self.rk(z).transpose(1,2)/16;d=self.dq(z)@self.dk(z).transpose(1,2)/16;e=torch.eye(n,device=x.device,dtype=torch.bool)[None];return r.masked_fill(e,-1e4),d.masked_fill(e,-1e4)
def remap(r,d,o,dev):
 inv=torch.empty_like(o);inv[o]=torch.arange(N);r,d=r[o],d[o];vr,vd=r>=0,d>=0;return torch.where(vr,inv[r.clamp_min(0)],r).to(dev),torch.where(vd,inv[d.clamp_min(0)],d).to(dev),vr.to(dev),vd.to(dev)
def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--encoder',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();torch.manual_seed(20260817);random.seed(20260817);train,_,_=splits(a.manifest);dev=torch.device('cuda');m=MAE();m.e.load_state_dict(torch.load(a.encoder,map_location='cpu',weights_only=True)['encoder']);net=Rel(m.e).to(dev);opt=torch.optim.AdamW(net.parameters(),2e-4,weight_decay=.02);data=[(s,raw_tiles(a.inputs,s),*labels(a.labels,s)) for s in train];losses=[];t=time.perf_counter()
 for ep in range(30):
  order=list(range(96));random.shuffle(order);z=[]
  for k in order:
   _,x,r,d=data[k];o=torch.randperm(N);r,d,vr,vd=remap(r,d,o,dev);sr,sd=net(x[o][None].to(dev));loss=F.cross_entropy(sr[0,vr],r[vr])+F.cross_entropy(sd[0,vd],d[vd]);opt.zero_grad();loss.backward();opt.step();z.append(float(loss.detach().cpu()))
  print(json.dumps({'epoch':ep+1,'loss':float(np.mean(z))}),flush=True);losses.append(float(np.mean(z)))
 net.eval();rows=[]
 with torch.no_grad():
  for s,x,r,d in data:
   sr,sd=net(x[None].to(dev));rows.append(top20_recall(sr[0],sd[0],r,d))
 rec=float(np.mean(rows));report={'experiment':'P39_MPRT24','gate':'G2','sources':96,'top20_recall':rec,'terminal_loss':losses[-1],'seconds':time.perf_counter()-t,'invalid':0,'targets_opened':False,'p8_imported':False,'selection_opened':False,'passes_G2':bool(rec>=.15)};a.out.parent.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':net.state_dict()},a.out);a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
 if not report['passes_G2']:raise RuntimeError('P39 G2 rejected')
if __name__=='__main__':main()
