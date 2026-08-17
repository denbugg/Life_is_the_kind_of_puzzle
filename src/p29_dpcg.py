"""P29 DPCG-24 input-only contracts, frozen DINOv2 dense descriptors."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
N,T,G=576,20,24
def sha(x):x=np.ascontiguousarray(x);return hashlib.sha256(str(x.dtype).encode()+repr(x.shape).encode()+x.tobytes()).hexdigest()
def topm(q,k,m):
 s=torch.einsum('idc,jdc->ij',q,k)/q.shape[1];s.fill_diagonal_(-float('inf'));return torch.topk(s,m,dim=1).indices.cpu().numpy(),s.cpu().numpy()
def g0(a):
 torch.manual_seed(20260817);x=torch.randn(5,16,8);y=x.clone();r,_=topm(x,y,2);xt=x.transpose(1,2).contiguous().transpose(1,2);rt,_=topm(xt,xt,2);perm=torch.tensor([2,4,0,3,1]);rp,_=topm(x[perm],y[perm],2);out={'experiment':'P29_DPCG24','gate':'G0','finite':bool(torch.isfinite(x).all()),'transpose_consistent':bool(np.array_equal(r,rt)),'candidate_permutation_invariant':bool(np.array_equal(perm[rp],r[perm])),'directional_boundary_shape':list(x.shape),'labels_used':False,'targets_opened':False,'p8_imported':False};out['passes_G0']=all(out[k] for k in ['finite','transpose_consistent','candidate_permutation_invariant']);a.work.mkdir(parents=True,exist_ok=True);(a.work/'p29_g0_report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
 if not out['passes_G0']:raise RuntimeError('P29 G0 failed')
def load_tiles(root,n):
 z=np.asarray(Image.open(root/n).convert('RGB'),np.float32)/255.;return torch.from_numpy(z.reshape(G,T,G,T,3).transpose(0,2,4,1,3).reshape(N,3,T,T).copy())
def model(dev):
 m=torch.hub.load('facebookresearch/dinov2','dinov2_vits14',trust_repo=True).to(dev).eval();return m
def desc(m,t,dev):
 x=torch.nn.functional.interpolate(t.to(dev),size=(224,224),mode='bicubic',align_corners=False);x=(x-torch.tensor([.485,.456,.406],device=dev)[None,:,None,None])/torch.tensor([.229,.224,.225],device=dev)[None,:,None,None]
 o=[]
 with torch.no_grad():
  for b in x.split(64):o.append(m.forward_features(b)['x_norm_patchtokens'].reshape(-1,16,16,384))
 return torch.cat(o)
def g1(a):
 dev=torch.device('cuda');m=model(dev);rows=[]
 for n in a.sources:
  st=time.perf_counter();z=desc(m,load_tiles(a.inputs,n),dev);bands=torch.stack([z[:,:,-1,:],z[:,-1,:,:],z[:,:,0,:],z[:,0,:,:]]);inds=[]
  for d in range(4):
   q=bands[d];k=bands[(d+2)%4];ii,_=topm(q,k,64);inds.append(ii)
  arr=np.stack(inds);rows.append({'source':n,'seconds':time.perf_counter()-st,'descriptor_sha256':sha(z.cpu().numpy()),'topm_sha256':sha(arr),'finite':bool(torch.isfinite(z).all())})
 out={'experiment':'P29_DPCG24','gate':'G1','rows':rows,'labels_used':False,'targets_opened':False,'p8_imported':False};out['passes_G1']=bool(all(r['finite'] and r['seconds']<=120 for r in rows));a.work.mkdir(parents=True,exist_ok=True);(a.work/'p29_g1_report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
 if not out['passes_G1']:raise RuntimeError('P29 G1 failed')
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['g0','g1']);p.add_argument('--inputs',type=Path,default=Path(r'E:\pazzle_data\train\inputs'));p.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P29_dpcg'));p.add_argument('--sources',nargs=4,default=['img_000025.png','img_000098.png','img_000168.png','img_000172.png']);a=p.parse_args();
 if 'p8' in '\n'.join(map(str,[a.inputs,a.work])).lower():raise RuntimeError('P8 prohibited')
 {'g0':g0,'g1':g1}[a.mode](a)
if __name__=='__main__':main()
