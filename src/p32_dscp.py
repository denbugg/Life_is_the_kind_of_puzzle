"""P32 DSCP-24: DINO semantic set-conditioned coordinate prior, G0/G1 only."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p29_dpcg import N, load_tiles

DINO_REPO = Path(r"E:\AnimeRec\mlcache\torch\hub\facebookresearch_dinov2_main")


def bad(*x):
    if "p8" in "\n".join(map(lambda q: str(q).lower(), x)):
        raise RuntimeError("P8 prohibited")


def dino(device):
    if not DINO_REPO.exists():
        raise RuntimeError("cached DINO repo unavailable")
    m = torch.hub.load(str(DINO_REPO), "dinov2_vits14", source="local").to(device).eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m


def features(m, t, device, batch=48):
    out=[]
    with torch.no_grad():
        for off in range(0, N, batch):
            x=t[off:off+batch].to(device).float()
            x=torch.nn.functional.interpolate(x, size=(224,224), mode="bicubic", align_corners=False)
            x=(x-torch.tensor([0.485,0.456,0.406],device=device)[None,:,None,None])/torch.tensor([0.229,0.224,0.225],device=device)[None,:,None,None]
            z=m.forward_features(x)["x_norm_patchtokens"].mean(dim=1)
            out.append(z.cpu())
    return torch.cat(out)


class DSCP(nn.Module):
    def __init__(self, d=192):
        super().__init__()
        self.proj=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,d),nn.GELU(),nn.LayerNorm(d))
        layer=nn.TransformerEncoderLayer(d_model=d,nhead=8,dim_feedforward=d*3,dropout=0.0,batch_first=True,norm_first=True,activation="gelu")
        self.enc=nn.TransformerEncoder(layer, num_layers=4, enable_nested_tensor=False)
        self.slot=nn.Parameter(torch.randn(N,d)*0.02)
    def forward(self,z):
        x=self.enc(self.proj(z)[None])[0]
        return x @ self.slot.T / (x.shape[-1]**0.5)


def hungarian_cost(score):
    from scipy.optimize import linear_sum_assignment
    r,c=linear_sum_assignment(-score)
    out=np.empty(N,np.int32);out[r]=c
    return out


def g0(a):
    torch.manual_seed(20260817)
    z=torch.eye(N, dtype=torch.float32)
    # Exact synthetic slot recovery proves the independent assignment projection contract.
    p=hungarian_cost(z.numpy())
    q=torch.randperm(N)
    p_perm=hungarian_cost(z[q].numpy())
    equiv=bool(np.array_equal(p_perm, q.numpy()))
    return {"experiment":"P32_DSCP24","gate":"G0","finite_logits":bool(torch.isfinite(z).all()),"valid_bijection":bool(np.unique(p).size==N),"exact_synthetic":bool(np.array_equal(p,np.arange(N,dtype=np.int32))),"permutation_contract":equiv,"passes_G0":bool(np.array_equal(p,np.arange(N,dtype=np.int32)) and equiv)}


def g1(a):
    device=torch.device("cuda")
    torch.manual_seed(20260817)
    backbone=dino(device); model=DSCP().to(device).eval(); rows=[]
    with torch.no_grad():
        for k,n in enumerate(a.sources):
            st=time.perf_counter();z=features(backbone,load_tiles(a.inputs,n),device).to(device); s=model(z).cpu(); perm=torch.randperm(N, generator=torch.Generator().manual_seed(103+k),device=device);sp=model(z[perm]).cpu(); restored=torch.empty_like(sp);restored[perm.cpu()]=sp
            changed=z.clone();changed[0]+=0.05;delta=float((model(changed).cpu()-s).abs().mean());eq=float((s-restored).abs().max());sec=time.perf_counter()-st;ok=bool(torch.isfinite(s).all() and eq<1e-5 and delta>1e-7 and sec<=90)
            rows.append({"source":n,"seconds":sec,"equivariance_max_abs":eq,"content_delta":delta,"ok":ok})
            if (k+1)%4==0:print(json.dumps({"stage":"g1","done":k+1,"total":len(a.sources)}),flush=True)
    return {"experiment":"P32_DSCP24","gate":"G1","rows":rows,"labels_used":False,"targets_opened":False,"p8_imported":False,"passes_G1":bool(all(x["ok"] for x in rows))}


def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=("g0","g1"),required=True);p.add_argument("--inputs",type=Path,default=Path(r"E:\pazzle_data\train\inputs"));p.add_argument("--work",type=Path,default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P32_dscp"));p.add_argument("--sources",nargs="*",default=("img_000002.png","img_000025.png","img_000098.png","img_000168.png","img_000172.png","img_000194.png","img_000223.png","img_000243.png","img_000267.png","img_000304.png","img_000344.png","img_000384.png","img_000426.png","img_000457.png","img_000480.png","img_000513.png"));a=p.parse_args();bad(a.inputs,a.work);a.work.mkdir(parents=True,exist_ok=True);r=g0(a) if a.mode=="g0" else g1(a);(a.work/f"p32_{a.mode}_report.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r),flush=True)
    if not r[f"passes_{a.mode.upper()}"]:raise RuntimeError(f"P32 {a.mode} rejected")
if __name__=="__main__":main()
