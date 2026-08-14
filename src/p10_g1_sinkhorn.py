"""ORBIT-24 P10 G1 locked FIT-only Sinkhorn refiner harness."""
from __future__ import annotations
import argparse, hashlib, json, multiprocessing as mp, os, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch import nn

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from p9_rank96_loop_g1 import SEED as P9_SEED, canonical_dense_rd, distort_frags, to_frags
from solve_buddies import solve_buddies_from_scores
from p10_sinkhorn_contracts import GRID, N_TILES, SINKHORN_ITERS, log_sinkhorn

TILE, TRAIN_N, HELD_N, EPOCHS, LR, WD, SEED = 20, 128, 32, 12, 2e-4, 1e-4, 20260814


def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def fourier(slots: torch.Tensor) -> torch.Tensor:
    r=torch.div(slots.long(),GRID,rounding_mode='floor').float()/(GRID-1); c=torch.remainder(slots.long(),GRID).float()/(GRID-1)
    phase=torch.stack((r,c),-1).unsqueeze(-1)*(2.0**torch.arange(4,device=slots.device,dtype=torch.float32))*torch.pi
    return torch.cat((torch.sin(phase),torch.cos(phase)),-1).flatten(1)

def decode(logits: torch.Tensor) -> np.ndarray:
    a=logits.detach().float().cpu().numpy(); rows,cols=linear_sum_assignment(-a)
    if not np.array_equal(rows,np.arange(N_TILES)): raise RuntimeError('assignment missed tile rows')
    out=np.empty(N_TILES,np.int64); out[rows]=cols
    if np.unique(out).size!=N_TILES: raise RuntimeError('assignment is not bijective')
    return out

def target_rgb(path: Path) -> np.ndarray:
    x=np.asarray(Image.open(path).convert('RGB'),dtype=np.uint8)
    if x.shape!=(480,480,3): raise RuntimeError(f'bad FIT target shape {x.shape}')
    return x

def tile_to_slot(board: np.ndarray) -> np.ndarray:
    flat=np.asarray(board,np.int64).reshape(-1)
    if flat.shape!=(N_TILES,) or np.unique(flat).size!=N_TILES: raise RuntimeError('rank96 board is not bijective')
    out=np.empty(N_TILES,np.int64); out[flat]=np.arange(N_TILES); return out

def edge_stats(scores: np.ndarray) -> np.ndarray:
    if scores.shape[:2]!=(4,N_TILES): raise RuntimeError(f'bad score shape {scores.shape}')
    top=np.sort(scores,axis=-1)[...,-8:]
    f=np.stack((top.mean(-1),top.std(-1),top.max(-1)),axis=-1) # [4,N,3]
    return np.transpose(f,(1,0,2)).reshape(N_TILES,12).astype(np.float32)

def build_one(item: tuple[str,int,str,str,str]) -> dict[str,object]:
    source,index,p9cache,targets,outdir=item; dst=Path(outdir)/(source.replace('.png','.npz'))
    if dst.is_file(): return {'source':source,'index':index,'cache':str(dst),'cached':True}
    with np.load(p9cache,allow_pickle=False) as z:
        need={'candidates','scores','permutation','source'}
        if not need.issubset(z.files): raise RuntimeError(f'missing P9 keys in {p9cache}')
        cand=z['candidates'].copy(); scores=z['scores'].copy(); perm=z['permutation'].astype(np.int64); cached_source=str(z['source'].item())
    if cached_source!=source or perm.shape!=(N_TILES,) or np.unique(perm).size!=N_TILES: raise RuntimeError('P9 cache identity/permutation violation')
    right,down=canonical_dense_rd(cand,scores); board,obj=solve_buddies_from_scores(right,down,max_edges=96); initial=tile_to_slot(board)
    frags=distort_frags(to_frags(target_rgb(Path(targets)/source)),np.random.default_rng(P9_SEED*1009+index)); tiles=frags[perm]
    if tiles.shape!=(N_TILES,TILE,TILE,3): raise RuntimeError('corrupted FIT tile shape mismatch')
    dst.parent.mkdir(parents=True,exist_ok=True); tmp=dst.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,tiles_uint8=tiles,target_tile_to_slot=perm.astype(np.int32),initial_tile_to_slot=initial.astype(np.int32),edge_stats=edge_stats(scores),source=np.array(source),source_index=np.array(index),p9_cache_sha256=np.array(sha256(Path(p9cache))),initial_objective=np.array(obj,np.float64))
    os.replace(tmp,dst); return {'source':source,'index':index,'cache':str(dst),'cached':False}

def source_rows(report: Path) -> list[dict[str,object]]:
    rows=json.loads(report.read_text(encoding='utf-8'))['source_rows']
    if len(rows)!=TRAIN_N+HELD_N: raise RuntimeError('locked P10 G1 requires exactly 160 P9 sources')
    return rows

def prepare(args: argparse.Namespace) -> None:
    rows=source_rows(args.p9_prepare_report); items=[]
    for i,row in enumerate(rows):
        source=str(row['source']); p9cache=Path(str(row['cache']))
        if not p9cache.is_file(): raise FileNotFoundError(p9cache)
        items.append((source,i,str(p9cache),str(args.targets),str(args.cache_dir)))
    with mp.Pool(args.workers) as pool: built=list(pool.imap(build_one,items))
    manifest={'experiment':'P10_sinkhorn_refiner','gate':'G1_prepare_FIT_only','source_rows':built,'train_sources':[r['source'] for r in built[:TRAIN_N]],'held_sources':[r['source'] for r in built[TRAIN_N:]],'targets_opened':'FIT_only','cal_target_opened':False,'dev_targets_opened':False,'test_accessed':False,'p8_labels_imported':False,'rank96_mining_invoked':False,'rank96_ranker_invoked':False}
    args.work.mkdir(parents=True,exist_ok=True); (args.work/'p10_g1_prepare_report.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8'); print(json.dumps({'prepared':len(built),'new':sum(not r['cached'] for r in built)},indent=2))

class Refiner(nn.Module):
    def __init__(self,w=64):
        super().__init__(); self.enc=nn.Sequential(nn.Conv2d(3,32,3,1,1),nn.GELU(),nn.Conv2d(32,w,3,1,1),nn.GELU(),nn.AdaptiveAvgPool2d(1),nn.Flatten()); self.edge=nn.Sequential(nn.LayerNorm(12),nn.Linear(12,w),nn.GELU()); self.mix=nn.Sequential(nn.LayerNorm(w*2+16),nn.Linear(w*2+16,w),nn.GELU(),nn.LayerNorm(w)); layer=nn.TransformerEncoderLayer(w,4,w*4,batch_first=True,activation='gelu',norm_first=True); self.ctx=nn.TransformerEncoder(layer,2); self.slot=nn.Sequential(nn.Linear(16,w),nn.GELU(),nn.LayerNorm(w)); self.bias=nn.Parameter(torch.zeros(N_TILES,w))
    def forward(self,tiles,edge,initial):
        t=self.enc(tiles); x=self.mix(torch.cat((t,self.edge(edge),fourier(initial)),1)); x=self.ctx(x.unsqueeze(0)).squeeze(0); slots=self.slot(fourier(torch.arange(N_TILES,device=tiles.device)))+self.bias; return x@slots.T/(x.shape[1]**.5)

def load_cache(cache_dir: Path,source: str,device: torch.device):
    with np.load(cache_dir/(source.replace('.png','.npz')),allow_pickle=False) as z:
        tiles=torch.from_numpy(z['tiles_uint8']).permute(0,3,1,2).float().div_(255).to(device); edge=torch.from_numpy(z['edge_stats']).float().to(device); init=torch.from_numpy(z['initial_tile_to_slot']).long().to(device); target=z['target_tile_to_slot'].astype(np.int64)
    if torch.unique(init).numel()!=N_TILES or np.unique(target).size!=N_TILES: raise RuntimeError('cache bijection contract failed')
    return tiles,edge,init,target

def train_eval(args: argparse.Namespace) -> None:
    prep=json.loads((args.work/'p10_g1_prepare_report.json').read_text(encoding='utf-8')); train=prep['train_sources']; held=prep['held_sources']
    if len(train)!=TRAIN_N or len(held)!=HELD_N: raise RuntimeError('locked split absent')
    torch.manual_seed(SEED); np.random.seed(SEED); device=torch.device('cuda'); model=Refiner().to(device); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD); history=[]
    for epoch in range(EPOCHS):
        model.train(); losses=[]
        for source in train:
            tiles,edge,init,target=load_cache(args.cache_dir,source,device); logits=model(tiles,edge,init); p=log_sinkhorn(logits,SINKHORN_ITERS); idx=torch.arange(N_TILES,device=device); loss=-torch.log(p[idx,torch.from_numpy(target).to(device)].clamp_min(1e-12)).mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses))}); print(json.dumps(history[-1]),flush=True)
    torch.save({'state_dict':model.state_dict(),'seed':SEED,'epochs':EPOCHS},args.work/'p10_g1_final.pt'); model.eval(); baseline=[]; refined=[]; invalid=0
    with torch.no_grad():
        for source in held:
            tiles,edge,init,target=load_cache(args.cache_dir,source,device); logits=model(tiles,edge,init); pred=decode(logits); baseline.append(float(np.mean(init.cpu().numpy()==target))); refined.append(float(np.mean(pred==target))); invalid+=int(np.unique(pred).size!=N_TILES)
    b=float(np.mean(baseline)); r=float(np.mean(refined)); delta=(r-b)*100; report={'experiment':'P10_sinkhorn_refiner','gate':'G1_train128_held32','selected_by':'fixed final epoch 12; held never inspected during training','baseline_held_accuracy':b,'refined_held_accuracy':r,'held_delta_pp_vs_rank96':delta,'invalid_decodes':invalid,'passes_G1':bool(delta>=5.0 and invalid==0),'decision':'PASS_open_CAL' if delta>=5.0 and invalid==0 else 'REJECT_before_CAL','train_history':history,'targets_opened':'FIT_only','cal_target_opened':False,'dev_targets_opened':False,'test_accessed':False,'p8_labels_imported':False,'rank96_mining_invoked':False,'rank96_ranker_invoked':False,'amp_used':False}
    (args.work/'p10_g1_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8'); print(json.dumps(report,indent=2))

def main():
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=('prepare','train_eval')); p.add_argument('--work',type=Path,required=True); p.add_argument('--cache-dir',type=Path,required=True); p.add_argument('--p9-prepare-report',type=Path); p.add_argument('--targets',type=Path); p.add_argument('--workers',type=int,default=4); a=p.parse_args()
    if a.phase=='prepare':
        if a.p9_prepare_report is None or a.targets is None: raise ValueError('prepare requires P9 report and FIT targets')
        prepare(a)
    else: train_eval(a)
if __name__=='__main__': main()
