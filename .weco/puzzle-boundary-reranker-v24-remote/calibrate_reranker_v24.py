"""Calibrate V24 residual strength independently of reranker training."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/kva/pazzle_boundary_reranker_v24")
import train_reranker_v24 as v24
import train_boundary_biencoder_v23 as v23


CHECKPOINT = Path("/home/kva/pazzle_boundary_reranker_v24/outputs/reranker_latest.pt")
OUT = Path("/home/kva/pazzle_boundary_reranker_v24/calibration")
ALPHAS = (-0.20, -0.10, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.10, 0.14, 0.18, 0.24)


def load_reranker(device):
    state=torch.load(CHECKPOINT,map_location="cpu",weights_only=True)
    model=v24.BoundaryReranker(v24.ModelConfig(**state["model_config"]))
    model.load_state_dict(state["model"],strict=True)
    return model.to(device).eval(),state


@torch.inference_mode()
def components(reranker,small,xl,scene,device):
    board=v23.load_board(v24.DATA_DIR/f"img_{scene:06d}.png")
    tiles=board.reshape(v23.GRID**2,3,v23.TILE,v23.TILE).to(device)
    base=v24.candidate_scores(small,xl,tiles)
    candidates,source,target,direction=v24.candidates_from(base,device)
    tokens=reranker.encode_sides(tiles)
    residual=reranker.score_edges(tokens,source,target,direction).reshape(2,len(tiles),v24.TOPK).float().cpu().numpy()
    return base,candidates,residual


def matrices(parts,alpha):
    base,candidates,residual=parts;output=[]
    for d in range(2):
        matrix=base[d].copy();values=np.take_along_axis(matrix,candidates[d],1)
        matrix[np.arange(len(matrix))[:,None],candidates[d]]=values+alpha*residual[d]
        np.fill_diagonal(matrix,-1e4);output.append(matrix)
    return output


def metrics(parts,alpha):
    result=[]
    for matrix,direction in zip(matrices(parts,alpha),("right","down")):
        result.append(v23.retrieval(matrix,v23.GRID,direction))
    return {key:float(np.mean([x[key] for x in result])) for key in result[0]}


def aggregate(rows):return {key:float(np.mean([x[key] for x in rows])) for key in rows[0]}


def objective(x):return .45*x["top1"]+.25*x["top5"]+.20*x["mrr"]+.10*x["top32"]


def main():
    device=torch.device("cuda");small,_=v24.load_frozen(v24.SMALL_PATH,device);xl,_=v24.load_frozen(v24.XL_PATH,device)
    reranker,state=load_reranker(device)
    calibration=[components(reranker,small,xl,s,device) for s in range(6748,6756)]
    trials=[]
    for alpha in ALPHAS:
        value=aggregate([metrics(x,alpha) for x in calibration]);trials.append({"alpha":alpha,**value,"objective":objective(value)})
    selected=max(trials,key=lambda x:x["objective"])
    holdout=[components(reranker,small,xl,s,device) for s in range(6957,6973)]
    result={"schema":"puzzle-boundary-reranker-v24-calibration","checkpoint_step":state["step"],
            "selected":selected,"holdout_base":aggregate([metrics(x,0.) for x in holdout]),
            "holdout_selected":aggregate([metrics(x,selected["alpha"]) for x in holdout]),
            "trials":trials}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
