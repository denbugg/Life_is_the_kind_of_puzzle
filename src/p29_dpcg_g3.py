"""P29 G3: FIT-only dense/rank fusion after G2 candidate-coverage pass."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parent));from p29_dpcg import load_tiles,desc,model,N

def p13():import p13_component_pose as p;return p
def labs(d,n):
 with np.load(d/(Path(n).stem+'.npz'),allow_pickle=False) as z:po=z['target_tile_to_slot'].astype(np.int32);src=str(z['source'])
 if src!=n or np.unique(po).size!=N:raise RuntimeError('bad labels')
 iv=np.empty(N,np.int32);iv[po]=np.arange(N);return po,iv
def nb(po,iv,i,d):
 r,c=divmod(int(iv[i]),24);r+=d==1;r-=d==3;c+=d==0;c-=d==2
 return -1 if r<0 or r>=24 or c<0 or c>=24 else int(po[r*24+c])
def board(m,dev,inputs,sd,n):
 z=desc(m,load_tiles(inputs,n),dev);b=torch.stack([z[:,:,-1,:],z[:,-1,:,:],z[:,:,0,:],z[:,0,:,:]]);p=p13();c,v,s=p.load_score_cache(sd,n);out=[]
 with torch.no_grad():
  for d in range(4):
   ds=(torch.einsum('idc,jdc->ij',b[d],b[(d+2)%4])/16).cpu().numpy();dn=np.argsort(-ds,axis=1)[:,:64]
   for i in range(N):
    u=[]
    for x in list(dn[i])+list(c[i,np.flatnonzero(v[i])]):
     if int(x)!=i and int(x) not in u:u.append(int(x))
     if len(u)==128:break
    u=np.asarray(u,np.int32);fs=np.full(len(u),float(s[d,i,np.flatnonzero(v[i])].min()-1),np.float32);mp={int(x):float(y) for x,y in zip(c[i,np.flatnonzero(v[i])],s[d,i,np.flatnonzero(v[i])])}
    for k,x in enumerate(u):fs[k]=mp.get(int(x),fs[k])
    fn=(fs-fs.mean())/(fs.std()+1e-6);dv=ds[i,u];dnorm=(dv-dv.mean())/(dv.std()+1e-6);out.append((d,i,u,fn.astype(np.float32),dnorm.astype(np.float32)))
 return out
def fit(rows,ld,names):
 X=[];Y=[];rng=np.random.default_rng(20260817)
 for n in names:
  po,iv=labs(ld,n)
  for d,i,u,f,ds in rows[n]:
   q=nb(po,iv,i,d);h=np.flatnonzero(u==q)
   if q<0 or not len(h):continue
   neg=np.flatnonzero(u!=q);k=rng.choice(neg,min(12,len(neg)),False);X.append([f[h[0]],ds[h[0]]]);Y.append(1.);X.extend(np.stack([f[k],ds[k]],1));Y.extend([0.]*len(k))
 x=torch.tensor(np.asarray(X),device='cuda');y=torch.tensor(Y,device='cuda');w=torch.zeros(2,device='cuda',requires_grad=True);b=torch.zeros(1,device='cuda',requires_grad=True);o=torch.optim.Adam([w,b],lr=.05)
 for _ in range(400):
  loss=torch.nn.functional.binary_cross_entropy_with_logits(x@w+b,y)+1e-4*(w*w).sum();o.zero_grad();loss.backward();o.step()
 return w.detach().cpu().numpy(),float(b.detach().cpu()),len(Y)
def eval(rows,ld,names,wb):
 w,b=wb;out=[]
 for a in [0.,.05,.1,.2,.4, .8]:
  h=ba=tot=0
  for n in names:
   po,iv=labs(ld,n)
   for d,i,u,f,ds in rows[n]:
    q=nb(po,iv,i,d)
    if q<0:continue
    lg=f*w[0]+ds*w[1]+b;ln=(lg-lg.mean())/(lg.std()+1e-6);h+=int(q in u[np.argsort(-(f+a*ln))[:20]]);ba+=int(q in u[np.argsort(-f)[:20]]);tot+=1
  out.append({'alpha':a,'recall20':h/tot,'baseline_recall20':ba/tot,'gain_pp':100*(h-ba)/tot})
 return out
def main():
 q=argparse.ArgumentParser();q.add_argument('--inputs',type=Path,default=Path(r'E:\pazzle_data\train\inputs'));q.add_argument('--label-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache'));q.add_argument('--score-dir',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache'));q.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));q.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P29_dpcg'));a=q.parse_args()
 if 'p8' in '\n'.join(map(str,[a.inputs,a.label_dir,a.score_dir,a.manifest,a.work])).lower():raise RuntimeError('P8 prohibited')
 p=p13();fitn,_=p.source_lists(a.manifest);ns=sorted(fitn)[:128];tr,se=ns[:96],ns[96:];m=model(torch.device('cuda'));rows={}
 for k,n in enumerate(ns):
  rows[n]=board(m,torch.device('cuda'),a.inputs,a.score_dir,n)
  if (k+1)%16==0:print(json.dumps({'stage':'features','done':k+1,'total':128}),flush=True)
 w,b,cnt=fit(rows,a.label_dir,tr);res=eval(rows,a.label_dir,se,(w,b));best=max(res,key=lambda z:z['recall20']);rep={'experiment':'P29_DPCG24','gate':'G3','train_rows':cnt,'weights':w.tolist(),'bias':b,'scores':res,'selected':best,'labels_used':True,'targets_opened':False,'p8_imported':False,'passes_G3':bool(best['gain_pp']>=1)};a.work.mkdir(parents=True,exist_ok=True);(a.work/'p29_g3_report.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep),flush=True)
 if not rep['passes_G3']:raise RuntimeError('P29 G3 reject')
if __name__=='__main__':main()
