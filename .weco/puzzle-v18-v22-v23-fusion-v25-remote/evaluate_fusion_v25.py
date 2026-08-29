"""Direct held-out fusion of Kaggle V18/V22 with remote V23 boundary models."""
from __future__ import annotations

import importlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path("/home/kva/pazzle_v18_v22_v23_fusion_v25")
ASSETS = Path("/home/kva/pazzle_kaggle_winners/assets")
sys.path[:0] = [str(ROOT), str(ASSETS), "/home/kva/pazzle_boundary_biencoder_v23_xl"]
import train_puzzle_transformer_v10 as v10
import train_boundary_v22 as v22
import evaluate_ensemble_v23 as e23
import train_boundary_biencoder_v23 as v23

V18_PATH = Path("/home/kva/pazzle_kaggle_winners/v18/best.pt")
V22_PATH = Path("/home/kva/pazzle_kaggle_winners/v22/boundary_best.pt")
DATA = Path("/home/kva/pazzle_directional_transformer/data/real/restored_target_order")
OUT = ROOT / "outputs"
CALIBRATION = range(6700, 6720)
HOLDOUT = range(6957, 6973)
ALPHAS = tuple(float(x) for x in np.linspace(0, 1, 21))


def load_winner(device):
    sys.path.insert(0, str(ASSETS))
    winner_contour = importlib.import_module("contour_model")
    winner_core = importlib.import_module("solver_core")
    winner_pair = importlib.import_module("big_pair")
    contour_checkpoint = torch.load(ASSETS / "contour_model.pt", map_location="cpu", weights_only=True)
    contour_net = winner_contour.build_model(contour_checkpoint["seed"])
    contour_net.load_state_dict(contour_checkpoint["model"], strict=True)
    contour_net = contour_net.to(device).eval()
    pair_checkpoint = torch.load(ASSETS / "big_pair_scorer.pt", map_location="cpu", weights_only=True)
    pair_model = winner_pair.build_model(v10.SEED)
    pair_model.load_state_dict(pair_checkpoint["model"], strict=True)
    pair_model = pair_model.to(device).eval()
    return winner_contour, winner_core, winner_pair, contour_net, float(contour_checkpoint["threshold"]), pair_model


def load_models(device):
    state18 = torch.load(V18_PATH, map_location="cpu", weights_only=True)
    model18 = v10.DensePuzzleTransformer(v10.ModelConfig(**state18["model_config"]), use_checkpoint=False)
    model18.load_state_dict(state18["model"], strict=True); model18 = model18.to(device).eval()
    state22 = torch.load(V22_PATH, map_location="cpu", weights_only=True)
    model22 = v22.BoundaryReranker(v22.ModelConfig(**state22["model_config"]))
    model22.load_state_dict(state22["reranker"], strict=True); model22 = model22.to(device).eval()
    small, _ = e23.load_model(e23.SMALL, device); xl, _ = e23.load_model(e23.XL, device)
    return model18, model22, small, xl, state18, state22


def row_z(matrix):
    result=matrix.copy();np.fill_diagonal(result,np.nan)
    result=(result-np.nanmean(result,1,keepdims=True))/(np.nanstd(result,1,keepdims=True)+1e-6)
    np.fill_diagonal(result,-1e4);return result


def metrics(matrices):
    values=[v23.retrieval(matrix.copy(),v23.GRID,direction)
            for matrix,direction in zip(matrices,("right","down"))]
    return {key:float(np.mean([x[key] for x in values])) for key in values[0]}


def aggregate(rows):return {key:float(np.mean([x[key] for x in rows])) for key in rows[0]}


def objective(x):return .40*x["top1"]+.25*x["top5"]+.20*x["mrr"]+.15*x["top32"]


def blend(a,b,alpha):return [row_z(alpha*a[d]+(1-alpha)*b[d]) for d in range(2)]


def union_recall(a,b,k,direction):
    grid=np.arange(v23.GRID**2).reshape(v23.GRID,v23.GRID)
    if direction=="right":sources=grid[:,:-1].reshape(-1);targets=grid[:,1:].reshape(-1)
    else:sources=grid[:-1].reshape(-1);targets=grid[1:].reshape(-1)
    aa=np.argpartition(-a[sources],k-1,axis=1)[:,:k];bb=np.argpartition(-b[sources],k-1,axis=1)[:,:k]
    return float(((aa==targets[:,None]).any(1)|(bb==targets[:,None]).any(1)).mean())


@torch.inference_mode()
def score_scene(scene,model18,model22,winner,small,xl,device):
    board=v23.load_board(DATA/f"img_{scene:06d}.png")
    tiles=board.reshape(v23.GRID**2,3,v23.TILE,v23.TILE).to(device)
    x=tiles.unsqueeze(0)
    base22,refined22=v22.refine(model22,model18,winner,x)
    score_sets=[]
    for model in (small,xl):
        emb=model(tiles)
        score_sets.append([row_z((emb["right"]@emb["left"].t()).float().cpu().numpy()),
                           row_z((emb["bottom"]@emb["top"].t()).float().cpu().numpy())])
    seam=[]
    for source_side,target_side in (("right","left"),("bottom","top")):
        source=small.side_features(tiles,source_side).flatten(1);target=small.side_features(tiles,target_side).flatten(1)
        source=torch.nn.functional.normalize(source-source.mean(1,keepdim=True),dim=1)
        target=torch.nn.functional.normalize(target-target.mean(1,keepdim=True),dim=1)
        seam.append(row_z((source@target.t()).float().cpu().numpy()))
    blend23=[row_z(.25*score_sets[0][d]+.75*score_sets[1][d]+.50*seam[d]) for d in range(2)]
    return [list(map(row_z,base22)),list(map(row_z,refined22)),blend23]


def main():
    device=torch.device("cuda");torch.backends.cuda.matmul.allow_tf32=True
    model18,model22,small,xl,state18,state22=load_models(device);winner=load_winner(device)
    started=time.perf_counter();calibration=[]
    for i,scene in enumerate(CALIBRATION,1):
        calibration.append(score_scene(scene,model18,model22,winner,small,xl,device))
        print(json.dumps({"event":"calibration_scene","scene":scene,"of":len(CALIBRATION),"seconds":time.perf_counter()-started}),flush=True)
    trials={"v18_v23":[],"v22_v23":[]}
    for key,left_index in (("v18_v23",0),("v22_v23",1)):
        for alpha in ALPHAS:
            value=aggregate([metrics(blend(x[left_index],x[2],alpha)) for x in calibration])
            trials[key].append({"alpha":alpha,**value,"objective":objective(value)})
    selected={key:max(values,key=lambda x:x["objective"]) for key,values in trials.items()}
    rows={"v18":[],"v22":[],"v23":[],"v18_v23":[],"v22_v23":[]};union32=[];union64=[]
    for i,scene in enumerate(HOLDOUT,1):
        scores=score_scene(scene,model18,model22,winner,small,xl,device)
        rows["v18"].append(metrics(scores[0]));rows["v22"].append(metrics(scores[1]));rows["v23"].append(metrics(scores[2]))
        rows["v18_v23"].append(metrics(blend(scores[0],scores[2],selected["v18_v23"]["alpha"])))
        rows["v22_v23"].append(metrics(blend(scores[1],scores[2],selected["v22_v23"]["alpha"])))
        union32.append(np.mean([union_recall(scores[1][d],scores[2][d],16,direction) for d,direction in enumerate(("right","down"))]))
        union64.append(np.mean([union_recall(scores[1][d],scores[2][d],32,direction) for d,direction in enumerate(("right","down"))]))
        print(json.dumps({"event":"holdout_scene","scene":scene,"of":len(HOLDOUT),"seconds":time.perf_counter()-started}),flush=True)
    report={"schema":"puzzle-v18-v22-v23-fusion-v25","v18_step":state18["step"],"v22_step":state22["step"],
            "calibration_scenes":[min(CALIBRATION),max(CALIBRATION)],"holdout_scenes":[min(HOLDOUT),max(HOLDOUT)],
            "selected":selected,"holdout":{key:aggregate(value) for key,value in rows.items()},
            "v22_v23_union_recall_at_32_budget":float(np.mean(union32)),
            "v22_v23_union_recall_at_64_budget":float(np.mean(union64)),"trials":trials,
            "seconds":time.perf_counter()-started}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"event":"complete","report":report}),flush=True)


if __name__=="__main__":main()
