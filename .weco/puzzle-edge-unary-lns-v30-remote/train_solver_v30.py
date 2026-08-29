"""V30: learned edge calibration, graph coordinate heads, and unary-aware LNS."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path("/home/kva/pazzle_edge_unary_lns_v30")
V25_ROOT = Path("/home/kva/pazzle_v18_v22_v23_fusion_v25")
V26_ROOT = Path("/home/kva/pazzle_union_reranker_v26")
V27_ROOT = Path("/home/kva/pazzle_set_transformer_v27")
V28_ROOT = Path("/home/kva/pazzle_multimodal_boundary_v28")
V29_ROOT = Path("/home/kva/pazzle_global_soft_v29")
sys.path[:0] = [str(ROOT), str(V29_ROOT), str(V28_ROOT), str(V27_ROOT),
                str(V26_ROOT), str(V25_ROOT)]
import global_solver
import evaluate_fusion_v25 as v25
import evaluate_union_reranker_v26 as v26
import evaluate_set_transformer_v27 as v27
import evaluate_soft_solver_v29 as v29

OUT = ROOT / "outputs"
CACHE = V26_ROOT / "cache"
SIDE = 24
N = SIDE * SIDE
SEED = 300826
TRAIN_SCENES = tuple(range(6700, 6728)) + tuple(range(6957, 6981))
VALID_SCENES = tuple(range(6981, 6989))
EVAL_SCENES = (6732, 6733, 6734, 6735) + tuple(range(6989, 7000))
ASSEMBLY_SCENE = 6989
METHODS = ("baseline", "unfreeze", "packed1", "packed1_unfreeze", "packed2", "packed4")


def log(**payload):
    print(json.dumps(payload), flush=True)


def row_z(matrix):
    value = np.asarray(matrix, np.float32).copy()
    np.fill_diagonal(value, np.nan)
    value = (value - np.nanmean(value, axis=1, keepdims=True)) / (
        np.nanstd(value, axis=1, keepdims=True) + 1e-6)
    np.fill_diagonal(value, -12.)
    return value


class EdgeCalibrator(nn.Module):
    def __init__(self, features=9, width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, width), nn.SiLU(), nn.LayerNorm(width),
                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DirectionalCoordinateGNN(nn.Module):
    def __init__(self, input_features=20, width=96, steps=12):
        super().__init__(); self.width = width; self.steps = steps
        self.input = nn.Sequential(nn.Linear(input_features, width), nn.GELU(), nn.LayerNorm(width))
        self.direction = nn.Parameter(torch.randn(4, width) * .02)
        self.message = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, width))
        self.update = nn.GRUCell(width * 4, width)
        self.norm = nn.LayerNorm(width)
        self.row_head = nn.Linear(width, SIDE)
        self.col_head = nn.Linear(width, SIDE)
        self.border_head = nn.Linear(width, 4)

    def forward(self, node_features, neighbours, weights):
        h = self.input(node_features)
        for _ in range(self.steps):
            messages = []
            for direction in range(4):
                gathered = h[neighbours[direction]]
                directional = self.direction[direction].view(1, 1, -1).expand_as(gathered)
                encoded = self.message(torch.cat((gathered, directional), dim=-1))
                messages.append((encoded * weights[direction].unsqueeze(-1)).sum(1))
            h = self.update(torch.cat(messages, dim=-1), h)
        h = self.norm(h)
        return self.row_head(h), self.col_head(h), self.border_head(h)


def load_v27(scene, model, device):
    with np.load(CACHE / f"scene_{scene:06d}.npz") as data:
        old = (data["v22"].astype(np.float32), data["v23"].astype(np.float32))
    return v27.rerank_scene(model, old, 1.35, device)


def load_eval(scene, model, device):
    base = load_v27(scene, model, device)
    with np.load(V28_ROOT / "score_cache" / f"scene_{scene:06d}.npz") as data:
        extra = data["scores"].astype(np.float32)
    return [row_z(.30 * base[d] + .70 * extra[d]) for d in range(2)]


def matrix_statistics(matrix):
    x = np.asarray(matrix, np.float32)
    safe = x.copy(); np.fill_diagonal(safe, -1e6)
    row_order = np.argsort(-safe, axis=1)
    col_order = np.argsort(-safe, axis=0)
    row_rank = np.empty_like(row_order); row_rank[np.arange(N)[:, None], row_order] = np.arange(N)[None]
    col_rank = np.empty_like(col_order); col_rank[col_order, np.arange(N)[None, :]] = np.arange(N)[:, None]
    return safe, safe.max(1), safe.mean(1), safe.std(1) + 1e-6, safe.max(0), safe.mean(0), safe.std(0) + 1e-6, row_rank, col_rank


def edge_features(stats, sources, targets):
    x, rmax, rmean, rstd, cmax, cmean, cstd, rrank, crank = stats
    score = x[sources, targets]
    return np.stack((score, (score-rmean[sources])/rstd[sources], score-rmax[sources],
                     (score-cmean[targets])/cstd[targets], score-cmax[targets],
                     rrank[sources, targets]/32., crank[sources, targets]/32.,
                     (rmax-rmean)[sources]/rstd[sources],
                     (cmax-cmean)[targets]/cstd[targets]), axis=1).astype(np.float32)


def edge_dataset(bundles, negatives=15):
    features, labels = [], []
    grid = np.arange(N).reshape(SIDE, SIDE)
    true_pairs = ((grid[:, :-1].reshape(-1), grid[:, 1:].reshape(-1)),
                  (grid[:-1].reshape(-1), grid[1:].reshape(-1)))
    for scene, matrices in bundles.items():
        for direction, matrix in enumerate(matrices):
            stats = matrix_statistics(matrix); safe = stats[0]
            sources, targets = true_pairs[direction]
            order = np.argpartition(-safe[sources], negatives, axis=1)[:, :negatives+1]
            for source, target, candidates in zip(sources, targets, order):
                hard = candidates[candidates != target][:negatives]
                chosen = np.concatenate(([target], hard))
                features.append(edge_features(stats, np.full(len(chosen), source), chosen))
                labels.append(np.concatenate(([1.], np.zeros(len(hard), np.float32))))
    return np.concatenate(features), np.concatenate(labels)


def train_edge(model, train_bundles, valid_bundles, device):
    x, y = edge_dataset(train_bundles); vx, vy = edge_dataset(valid_bundles)
    tx, ty = torch.from_numpy(x), torch.from_numpy(y); generator = torch.Generator().manual_seed(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    history=[]; best=None; best_ap=-1.
    for epoch in range(12):
        model.train(); order=torch.randperm(len(tx),generator=generator)
        losses=[]
        for offset in range(0,len(order),8192):
            ids=order[offset:offset+8192]; bx=tx[ids].to(device); by=ty[ids].to(device)
            logits=model(bx); loss=F.binary_cross_entropy_with_logits(logits,by,pos_weight=torch.tensor(15.,device=device))
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode(): pred=torch.sigmoid(model(torch.from_numpy(vx).to(device))).cpu().numpy()
        # Average precision without sklearn.
        order_np=np.argsort(-pred); positives=vy[order_np]; precision=np.cumsum(positives)/(np.arange(len(positives))+1)
        ap=float((precision*positives).sum()/positives.sum())
        history.append({"epoch":epoch+1,"loss":float(np.mean(losses)),"valid_ap":ap})
        log(event="edge_epoch",**history[-1])
        if ap>best_ap: best_ap=ap;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best); return history,best_ap


@torch.inference_mode()
def calibrate_matrix(model, matrix, gamma, device):
    # The model is trained on hard negatives. Restrict inference to the original
    # top-64 so unseen easy negatives can never be promoted into the candidate set.
    stats=matrix_statistics(matrix);safe=stats[0];topk=64
    candidates=np.argpartition(-safe,topk-1,axis=1)[:,:topk]
    sources=np.repeat(np.arange(N,dtype=np.int32),topk);targets=candidates.reshape(-1)
    features=edge_features(stats,sources,targets);predictions=[]
    for start in range(0,len(features),32768):
        predictions.append(model(torch.from_numpy(features[start:start+32768]).to(device)).float().cpu().numpy())
    logits=np.full((N,N),-20.,np.float32);logits[sources,targets]=np.concatenate(predictions)
    return row_z((1-gamma)*row_z(matrix)+gamma*row_z(logits))


def retrieval_metrics(matrices):
    ranks=[]; grid=np.arange(N).reshape(SIDE,SIDE)
    pairs=((grid[:,:-1].reshape(-1),grid[:,1:].reshape(-1)),
           (grid[:-1].reshape(-1),grid[1:].reshape(-1)))
    for matrix,(sources,targets) in zip(matrices,pairs):
        order=np.argsort(-matrix[sources],axis=1)
        ranks.extend((np.argmax(order==targets[:,None],axis=1)+1).tolist())
    ranks=np.asarray(ranks)
    return {"top1":float(np.mean(ranks<=1)),"top5":float(np.mean(ranks<=5)),
            "top32":float(np.mean(ranks<=32)),"mrr":float(np.mean(1/ranks))}


def graph_inputs(matrices, topk=8):
    relations=(matrices[0],matrices[0].T,matrices[1],matrices[1].T)
    neighbours=[];weights=[];features=[]
    for relation in relations:
        x=np.asarray(relation,np.float32).copy();np.fill_diagonal(x,-1e6)
        ids=np.argpartition(-x,topk-1,axis=1)[:,:topk]
        values=np.take_along_axis(x,ids,axis=1)
        order=np.argsort(-values,axis=1);ids=np.take_along_axis(ids,order,axis=1);values=np.take_along_axis(values,order,axis=1)
        shifted=values-values.max(1,keepdims=True);prob=np.exp(shifted);prob/=prob.sum(1,keepdims=True)
        entropy=-(prob*np.log(prob+1e-8)).sum(1)/math.log(topk)
        features.extend((values[:,0],values[:,0]-values[:,1],values.mean(1),values.std(1),entropy))
        neighbours.append(ids.astype(np.int64));weights.append(prob.astype(np.float32))
    node=np.stack(features,axis=1).astype(np.float32)
    node=(node-node.mean(0,keepdims=True))/(node.std(0,keepdims=True)+1e-5)
    return node,np.stack(neighbours),np.stack(weights)


def targets(device):
    positions=torch.arange(N,device=device);row=positions//SIDE;col=positions%SIDE
    border=torch.stack((row==0,row==SIDE-1,col==0,col==SIDE-1),dim=1).float()
    return row,col,border


@torch.inference_mode()
def head_metrics(model,bundles,device):
    model.eval(); rows=[]; row_t,col_t,border_t=targets(device)
    for matrices in bundles.values():
        node,nbr,w=graph_inputs(matrices)
        logits=model(torch.from_numpy(node).to(device),torch.from_numpy(nbr).to(device),torch.from_numpy(w).to(device))
        row,col,border=logits
        pred_border=border>0;tp=(pred_border&border_t.bool()).sum().item()
        rows.append({"row":float((row.argmax(1)==row_t).float().mean()),
                     "column":float((col.argmax(1)==col_t).float().mean()),
                     "border_f1":float(2*tp/max(1,pred_border.sum().item()+border_t.sum().item()))})
    return {key:float(np.mean([r[key] for r in rows])) for key in rows[0]}


def train_heads(model,train_bundles,valid_bundles,device):
    prepared={scene:graph_inputs(matrices) for scene,matrices in {**train_bundles,**valid_bundles}.items()}
    row_t,col_t,border_t=targets(device);optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-3)
    rng=np.random.default_rng(SEED);history=[];best=None;best_score=-1.
    for step in range(1,801):
        scene=int(rng.choice(list(train_bundles)));node,nbr,w=prepared[scene]
        logits=model(torch.from_numpy(node).to(device),torch.from_numpy(nbr).to(device),torch.from_numpy(w).to(device))
        row,col,border=logits
        loss=F.cross_entropy(row,row_t,label_smoothing=.03)+F.cross_entropy(col,col_t,label_smoothing=.03)
        loss=loss+.35*F.binary_cross_entropy_with_logits(border,border_t,pos_weight=torch.full((4,),12.,device=device))
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step()
        if step%100==0:
            metrics=head_metrics(model,valid_bundles,device);score=metrics["row"]+metrics["column"]+.5*metrics["border_f1"]
            row_log={"step":step,"loss":float(loss.detach()),**metrics,"score":score};history.append(row_log);log(event="head_validation",**row_log)
            if score>best_score:best_score=score;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best);return history,best_score


@torch.inference_mode()
def unary_from_heads(model,matrices,device):
    node,nbr,w=graph_inputs(matrices)
    row,col,border=model(torch.from_numpy(node).to(device),torch.from_numpy(nbr).to(device),torch.from_numpy(w).to(device))
    row=row.log_softmax(1).cpu().numpy();col=col.log_softmax(1).cpu().numpy();border=border.cpu().numpy()
    cells=np.arange(N);rr,cc=cells//SIDE,cells%SIDE
    unary=row[:,rr]+col[:,cc]
    border_targets=np.stack((rr==0,rr==SIDE-1,cc==0,cc==SIDE-1),axis=1).astype(np.float32)
    unary += .25*(border[:,None,:]*(2*border_targets[None]-1)).sum(2)
    unary=(unary-unary.mean(1,keepdims=True))/(unary.std(1,keepdims=True)+1e-6)
    return unary.astype(np.float32)


def total_objective(board,right,down,unary,unary_weight):
    return global_solver.board_objective(board,right,down,SIDE)+unary_weight*float(unary[board,np.arange(N)].sum())


def lns_refine(board,right,down,unary,unary_weight,seed,rounds=18,width=96):
    rnorm,dnorm=global_solver._normalise(right),global_solver._normalise(down);current=board.copy()
    best_score=total_objective(current,rnorm,dnorm,unary,unary_weight);rng=np.random.default_rng(seed)
    for iteration in range(rounds):
        grid=current.reshape(SIDE,SIDE);local=np.zeros(N,np.float32)
        local.reshape(SIDE,SIDE)[:,:-1]+=rnorm[grid[:,:-1],grid[:,1:]]
        local.reshape(SIDE,SIDE)[:,1:]+=rnorm[grid[:,:-1],grid[:,1:]]
        local.reshape(SIDE,SIDE)[:-1]+=dnorm[grid[:-1],grid[1:]]
        local.reshape(SIDE,SIDE)[1:]+=dnorm[grid[:-1],grid[1:]]
        local+=unary_weight*unary[current,np.arange(N)]
        worst=np.argpartition(local,width//2)[:width//2]
        center=int(rng.integers(N));cr,cc=divmod(center,SIDE);radius=4
        region=[r*SIDE+c for r in range(max(0,cr-radius),min(SIDE,cr+radius))
                for c in range(max(0,cc-radius),min(SIDE,cc+radius))]
        cells=np.unique(np.concatenate((worst,np.asarray(region,np.int32))))[:width]
        tiles=current[cells].copy();scores=global_solver._cell_scores(current,tiles,cells,rnorm,dnorm,SIDE)
        scores+=unary_weight*unary[np.ix_(tiles,cells)]
        tile_rows,cell_cols=linear_sum_assignment(-scores.astype(np.float64))
        candidate=current.copy();candidate[cells[cell_cols]]=tiles[tile_rows]
        score=total_objective(candidate,rnorm,dnorm,unary,unary_weight)
        if score>best_score+1e-6:current,best_score=candidate,score
    return current,best_score


def candidate_portfolio(right,down,seed):
    anchor,_=v25.v10.assemble_components(right,down,SIDE)
    baseline=global_solver.solve_complete(right,down,SIDE,anchor,seed=seed,beam_width=4,hungarian_rounds=5,swap_proposals=12000)
    boards={"baseline":baseline.board}
    boards["unfreeze"]=v29.unfreeze_refine(right,down,baseline.board,seed+7)[0]
    for topk in (1,2,4):boards[f"packed{topk}"]=v29.packed_refine(right,down,anchor,seed+topk,topk)[0]
    boards["packed1_unfreeze"]=v29.unfreeze_refine(right,down,boards["packed1"],seed+101)[0]
    return boards


def solve_v30(matrices,heads,unary_weight,device,scene):
    right,down=matrices;unary=unary_from_heads(heads,matrices,device);portfolio=candidate_portfolio(right,down,SEED+scene)
    refined={};objectives={}
    for index,(name,board) in enumerate(portfolio.items()):
        refined[name],objectives[name]=lns_refine(board,right,down,unary,unary_weight,SEED+scene+index*97)
    selected=max(refined,key=objectives.get)
    return refined[selected],selected,objectives,portfolio["baseline"]


def placement_metrics(board):
    return global_solver.placement_metrics(board,np.arange(N,dtype=np.int32),SIDE)


def render_board(tiles,board):
    grid=tiles[np.asarray(board).reshape(SIDE,SIDE)]
    return grid.transpose(0,2,1,3,4).reshape(480,480,3)


def labelled(image,text):
    canvas=np.full((525,480,3),255,np.uint8);canvas[45:]=image
    cv2.putText(canvas,text,(10,29),cv2.FONT_HERSHEY_SIMPLEX,.62,(15,15,15),2,cv2.LINE_AA);return canvas


def smoke():
    OUT.mkdir(parents=True,exist_ok=True);device=torch.device("cuda")
    state=torch.load(V27_ROOT/"outputs/set_reranker_best.pt",map_location=device,weights_only=True)
    reranker=v27.SetReranker().to(device);reranker.load_state_dict(state["model"]);reranker.eval()
    matrices=load_v27(6700,reranker,device);edge=EdgeCalibrator().to(device)
    x,y=edge_dataset({6700:matrices},negatives=3);sample=torch.from_numpy(x[:4096]).to(device)
    loss=F.binary_cross_entropy_with_logits(edge(sample),torch.from_numpy(y[:4096]).to(device));loss.backward()
    calibrated=[calibrate_matrix(edge,m,.2,device) for m in matrices]
    node,nbr,w=graph_inputs(calibrated);heads=DirectionalCoordinateGNN().to(device)
    logits=heads(torch.from_numpy(node).to(device),torch.from_numpy(nbr).to(device),torch.from_numpy(w).to(device))
    unary=unary_from_heads(heads,calibrated,device);portfolio=candidate_portfolio(*calibrated,SEED)
    board,objective=lns_refine(portfolio["baseline"],*calibrated,unary,.1,SEED,rounds=2)
    log(event="smoke",edge_examples=len(x),edge_loss=float(loss.detach()),
        head_shapes=[list(value.shape) for value in logits],unique_tiles=int(len(np.unique(board))),
        objective=objective,metrics=placement_metrics(board))


def main():
    torch.manual_seed(SEED);np.random.seed(SEED);OUT.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda");torch.backends.cuda.matmul.allow_tf32=True
    state=torch.load(V27_ROOT/"outputs/set_reranker_best.pt",map_location=device,weights_only=True)
    reranker=v27.SetReranker().to(device);reranker.load_state_dict(state["model"]);reranker.eval()
    started=time.perf_counter();all_support=TRAIN_SCENES+VALID_SCENES
    support={scene:load_v27(scene,reranker,device) for scene in all_support}
    train={scene:support[scene] for scene in TRAIN_SCENES};valid={scene:support[scene] for scene in VALID_SCENES}
    edge=EdgeCalibrator().to(device);edge_history,edge_ap=train_edge(edge,train,valid,device)
    gamma_trials=[]
    for gamma in (0.,.20,.40,.60,.80):
        calibrated={scene:[calibrate_matrix(edge,m,gamma,device) for m in matrices] for scene,matrices in valid.items()}
        values=[retrieval_metrics(x) for x in calibrated.values()]
        mean={key:float(np.mean([v[key] for v in values])) for key in values[0]}
        # Assembly is driven by the very top edges; top-32 remains a guardrail,
        # not a dominant selection term for the calibrated solver objective.
        mean["gamma"]=gamma;mean["score"]=.60*mean["top1"]+.25*mean["top5"]+.15*mean["mrr"]
        gamma_trials.append(mean);log(event="gamma",**mean)
    gamma=max(gamma_trials,key=lambda x:x["score"])["gamma"]
    # The calibrator is a measured ablation. It is not injected into the final
    # fused V28 solver until a same-domain cross-validated model passes its gate.
    solver_gamma=0.
    calibrated_train=train
    calibrated_valid=valid
    heads=DirectionalCoordinateGNN().to(device);head_history,head_score=train_heads(heads,calibrated_train,calibrated_valid,device)
    # Tune unary strength on four support-validation scenes, never on final CV scenes.
    weight_trials=[]
    for weight in (0.,.10,.20,.35,.50):
        rows=[]
        for scene in VALID_SCENES[:4]:
            board,selected,_,baseline=solve_v30(calibrated_valid[scene],heads,weight,device,scene)
            rows.append({"v30":placement_metrics(board),"baseline":placement_metrics(baseline),"selected":selected})
        score=float(np.mean([r["v30"]["adjacency"]+.25*r["v30"]["translation_aligned_placement"] for r in rows]))
        trial={"weight":weight,"score":score,"rows":rows};weight_trials.append(trial);log(event="unary_weight",weight=weight,score=score)
    unary_weight=max(weight_trials,key=lambda x:x["score"])["weight"]
    eval_rows=[];example=None;maps=np.load(v25.MAP_FILE)["inv"]
    for scene in EVAL_SCENES:
        raw=load_eval(scene,reranker,device);calibrated=raw
        board,selected,objectives,baseline=solve_v30(calibrated,heads,unary_weight,device,scene)
        row={"scene":scene,"baseline":placement_metrics(baseline),"v30":placement_metrics(board),
             "selected":selected,"candidate_objectives":objectives,"retrieval_raw":retrieval_metrics(raw),
             "retrieval_calibrated":retrieval_metrics(calibrated)}
        eval_rows.append(row);log(event="eval",**row)
        if scene==ASSEMBLY_SCENE:example=(baseline.copy(),board.copy())
    aggregate={name:{key:float(np.mean([row[name][key] for row in eval_rows]))
                     for key in ("coverage","direct_placement","translation_aligned_placement","adjacency")}
               for name in ("baseline","v30")}
    tiles=v25.load_raw_target_order(ASSEMBLY_SCENE,maps).permute(0,2,3,1).mul(255).byte().numpy()
    target=v25.v10.load_rgb(v25.RAW_INPUTS.parent/"targets"/f"img_{ASSEMBLY_SCENE:06d}.png")
    montage=np.hstack((labelled(render_board(tiles,example[0]),"V29-style calibrated baseline"),
                       labelled(render_board(tiles,example[1]),"V30 edge + unary + LNS"),
                       labelled(target,"Clean target (reference)")))
    image=OUT/f"assembly_scene_{ASSEMBLY_SCENE}.png";cv2.imwrite(str(image),cv2.cvtColor(montage,cv2.COLOR_RGB2BGR))
    checkpoint={"edge":edge.state_dict(),"heads":heads.state_dict(),"gamma":gamma,
                "solver_gamma":solver_gamma,"unary_weight":unary_weight,
                "edge_features":9,"head_width":heads.width,"head_steps":heads.steps}
    torch.save(checkpoint,OUT/"solver_v30.pt")
    report={"schema":"puzzle-edge-unary-lns-v30","train_scenes":list(TRAIN_SCENES),
            "validation_scenes":list(VALID_SCENES),"evaluation_scenes":list(EVAL_SCENES),
            "edge_history":edge_history,"edge_valid_ap":edge_ap,"gamma_trials":gamma_trials,"gamma":gamma,
            "solver_gamma":solver_gamma,
            "head_history":head_history,"head_score":head_score,"head_valid":head_metrics(heads,calibrated_valid,device),
            "weight_trials":weight_trials,"unary_weight":unary_weight,"aggregate":aggregate,"rows":eval_rows,
            "parameters":{"edge":sum(p.numel() for p in edge.parameters()),"heads":sum(p.numel() for p in heads.parameters())},
            "seconds":time.perf_counter()-started,"visualization":str(image)}
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n");log(event="complete",report=report)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--smoke",action="store_true");args=parser.parse_args()
    smoke() if args.smoke else main()
