"""V29 global assignment experiments over frozen V28 compatibility matrices."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path("/home/kva/pazzle_global_soft_v29")
V25_ROOT = Path("/home/kva/pazzle_v18_v22_v23_fusion_v25")
V26_ROOT = Path("/home/kva/pazzle_union_reranker_v26")
V27_ROOT = Path("/home/kva/pazzle_set_transformer_v27")
V28_ROOT = Path("/home/kva/pazzle_multimodal_boundary_v28")
sys.path[:0] = [str(ROOT), str(V28_ROOT), str(V27_ROOT), str(V26_ROOT), str(V25_ROOT)]
import global_solver
import evaluate_fusion_v25 as v25
import evaluate_union_reranker_v26 as v26
import evaluate_set_transformer_v27 as v27

OUT = ROOT / "outputs"
SCENES = (6732, 6733, 6734, 6735, 6989, 6990, 6991, 6992, 6993, 6994,
          6995, 6996, 6997, 6998, 6999)
SIDE = 24
SEED = 290826
METHODS = ("baseline", "unfreeze", "packed1", "packed1_unfreeze", "packed2", "packed4")
METRICS = ("coverage", "direct_placement", "translation_aligned_placement", "adjacency", "objective")


def sinkhorn(logits, tau, iterations):
    value = logits / tau
    for _ in range(iterations):
        value = value - torch.logsumexp(value, dim=1, keepdim=True)
        value = value - torch.logsumexp(value, dim=0, keepdim=True)
    return value.exp()


def normalize(matrix):
    value = np.asarray(matrix, np.float32).copy(); np.fill_diagonal(value, np.nan)
    value = (value - np.nanmean(value)) / (np.nanstd(value) + 1e-6)
    np.fill_diagonal(value, -12.); return value


def soft_assign(right, down, init, *, steps, lr, tau_lo, seed, device):
    n = SIDE * SIDE; r = torch.from_numpy(normalize(right)).to(device)
    d = torch.from_numpy(normalize(down)).to(device)
    positions = torch.arange(n, device=device)
    hp = positions[positions % SIDE != SIDE - 1]; vp = positions[positions < n - SIDE]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn((n, n), generator=generator).to(device).mul_(.01)
    board = torch.as_tensor(init, dtype=torch.long, device=device)
    logits[board, positions] += 1.8
    logits.requires_grad_(True); optimizer = torch.optim.AdamW([logits], lr=lr, weight_decay=0.)
    history = []
    for step in range(steps):
        progress = step / max(1, steps - 1)
        tau = 1.0 * (tau_lo / 1.0) ** progress
        assignment = sinkhorn(logits, tau, 12)
        score = ((assignment[:, hp] * (r @ assignment[:, hp + 1])).sum()
                 + (assignment[:, vp] * (d @ assignment[:, vp + SIDE])).sum())
        # Confidence is delayed until the continuous assignment has moved globally.
        entropy = -(assignment.clamp_min(1e-8) * assignment.clamp_min(1e-8).log()).sum()
        loss = -score / (2 * SIDE * (SIDE - 1)) + (.0002 * progress) * entropy
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([logits], 5.); optimizer.step()
        if (step + 1) % 25 == 0:
            history.append({"step":step+1,"tau":tau,"soft_score":float(score.detach()),
                            "entropy":float(entropy.detach())})
    with torch.inference_mode(): assignment = sinkhorn(logits.detach(), tau_lo, 30)
    tiles, cells = linear_sum_assignment(-assignment.float().cpu().numpy())
    result = np.empty(n, np.int32); result[cells] = tiles
    rnorm, dnorm = global_solver._normalise(right), global_solver._normalise(down)
    polished, objective = global_solver._swap_polish(
        result, rnorm, dnorm, SIDE, 18000, np.random.default_rng(seed + 91))
    return polished, objective, history


def unfreeze_refine(right, down, init, seed):
    """Let the final global rounds repair mistakes inside the formerly rigid anchor."""
    rnorm, dnorm = global_solver._normalise(right), global_solver._normalise(down)
    board, _ = global_solver._hungarian_refine(init, rnorm, dnorm, SIDE, 8, None)
    board, objective = global_solver._swap_polish(
        board, rnorm, dnorm, SIDE, 30000, np.random.default_rng(seed), None)
    return board, objective


def packed_refine(right, down, anchor, seed, topk):
    """Pack secondary coordinate-consistent fragments before global refinement."""
    rnorm, dnorm = global_solver._normalise(right), global_solver._normalise(down)
    anchor_tiles = np.fromiter(anchor.keys(), dtype=np.int32)
    remaining = np.setdiff1d(np.arange(SIDE * SIDE, dtype=np.int32), anchor_tiles)
    components = global_solver._extract_components(rnorm, dnorm, remaining, SIDE, topk=topk)
    components = [component for component in components if len(component) >= 2][:32]
    starts = global_solver._pack_components(anchor, components, rnorm, dnorm, SIDE, beam_width=8)
    rng = np.random.default_rng(seed); candidates = []
    for start in starts:
        board = global_solver._frontier_complete(start, rnorm, dnorm, SIDE, rng)
        # Preserve only the primary high-precision anchor; secondary fragments seed the search.
        movable = np.flatnonzero(~np.isin(board, anchor_tiles)).astype(np.int32)
        board, _ = global_solver._hungarian_refine(board, rnorm, dnorm, SIDE, 6, movable)
        board, objective = global_solver._swap_polish(board, rnorm, dnorm, SIDE, 18000, rng, movable)
        candidates.append((objective, board))
    return max(candidates, key=lambda item: item[0])[1], max(x[0] for x in candidates), [len(x) for x in components]


def load_fused(scene, model27, device):
    with np.load(v26.CACHE / f"scene_{scene:06d}.npz") as data:
        old = (data["v22"].astype(np.float32), data["v23"].astype(np.float32))
    base = v27.rerank_scene(model27, old, 1.35, device)
    with np.load(V28_ROOT / "score_cache" / f"scene_{scene:06d}.npz") as data:
        extra = data["scores"].astype(np.float32)
    return [v25.row_z(.30 * base[d] + .70 * extra[d]) for d in range(2)]


def render_board(tiles, board):
    grid = tiles[np.asarray(board).reshape(SIDE, SIDE)]
    return grid.transpose(0, 2, 1, 3, 4).reshape(480, 480, 3)


def labelled(image, text):
    canvas = np.full((525, 480, 3), 255, np.uint8); canvas[45:] = image
    cv2.putText(canvas, text, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, .64,
                (15, 15, 15), 2, cv2.LINE_AA)
    return canvas


def aggregate(rows, method):
    return {key: float(np.mean([row[method][key] for row in rows])) for key in METRICS}


def selection_score(metrics):
    # Adjacency is the primary assembly metric; placement breaks close ties.
    return metrics["adjacency"] + .25 * metrics["translation_aligned_placement"]


def edge_pairs(board):
    grid = np.asarray(board).reshape(SIDE, SIDE)
    return (grid[:, :-1].reshape(-1), grid[:, 1:].reshape(-1),
            grid[:-1].reshape(-1), grid[1:].reshape(-1))


def candidate_features(board, baseline, right, down, anchor_size, component_sizes, method):
    rnorm, dnorm = global_solver._normalise(right), global_solver._normalise(down)
    rs, rt, ds, dt = edge_pairs(board)
    brs, brt, bds, bdt = edge_pairs(baseline)
    values = np.concatenate((rnorm[rs, rt], dnorm[ds, dt]))
    deficits = np.concatenate((rnorm.max(1)[rs] - rnorm[rs, rt],
                               dnorm.max(1)[ds] - dnorm[ds, dt]))
    candidate_edges = set(zip(rs.tolist(), rt.tolist(), [0] * len(rs))) | \
                      set(zip(ds.tolist(), dt.tolist(), [1] * len(ds)))
    baseline_edges = set(zip(brs.tolist(), brt.tolist(), [0] * len(brs))) | \
                     set(zip(bds.tolist(), bdt.tolist(), [1] * len(bds)))
    sizes = component_sizes or []
    numeric = [
        float(values.mean()), float(values.std()), float(np.quantile(values, .10)),
        float(np.quantile(values, .50)), float(deficits.mean()),
        float(np.quantile(deficits, .90)), float((deficits < .25).mean()),
        float(np.mean(np.asarray(board) == np.asarray(baseline))),
        float(len(candidate_edges & baseline_edges) / len(candidate_edges)),
        float(anchor_size / (SIDE * SIDE)), float(len(sizes) / 32),
        float((max(sizes) if sizes else anchor_size) / (SIDE * SIDE)),
    ]
    numeric.extend(float(method == candidate) for candidate in METHODS)
    return numeric


def ridge_selector(rows, alpha=8.):
    folds = [rows[offset::3] for offset in range(3)]; results = []
    for fold_index, heldout in enumerate(folds):
        train = [row for index, fold in enumerate(folds) if index != fold_index for row in fold]
        x = np.asarray([row[method]["selector_features"] for row in train for method in METHODS], np.float64)
        y = np.asarray([selection_score(row[method]) for row in train for method in METHODS], np.float64)
        mean, scale = x.mean(0), x.std(0) + 1e-6
        z = (x - mean) / scale
        design = np.column_stack((np.ones(len(z)), z))
        penalty = np.eye(design.shape[1]); penalty[0, 0] = 0
        weights = np.linalg.solve(design.T @ design + alpha * penalty, design.T @ y)
        selected_rows = []
        for row in heldout:
            candidates = np.asarray([row[method]["selector_features"] for method in METHODS], np.float64)
            predictions = np.column_stack((np.ones(len(METHODS)), (candidates - mean) / scale)) @ weights
            selected = METHODS[int(np.argmax(predictions))]
            selected_rows.append(row[selected])
        results.append({"fold": fold_index, "heldout_scenes": [row["scene"] for row in heldout],
                        "selected": [METHODS[int(np.argmax(np.column_stack((np.ones(len(METHODS)),
                            (np.asarray([row[m]["selector_features"] for m in METHODS]) - mean) / scale)) @ weights))]
                                     for row in heldout],
                        "metrics": {key: float(np.mean([item[key] for item in selected_rows])) for key in METRICS}})
    weighted = {key: float(sum(result["metrics"][key] * len(result["heldout_scenes"]) for result in results)
                           / len(rows)) for key in METRICS}
    return results, weighted


def cross_validate(rows):
    folds = [rows[offset::3] for offset in range(3)]
    results = []
    for fold_index, heldout in enumerate(folds):
        train = [row for index, fold in enumerate(folds) if index != fold_index for row in fold]
        train_metrics = {method: aggregate(train, method) for method in METHODS}
        selected = max(METHODS, key=lambda method: selection_score(train_metrics[method]))
        results.append({
            "fold": fold_index,
            "train_scenes": [row["scene"] for row in train],
            "heldout_scenes": [row["scene"] for row in heldout],
            "selected": selected,
            "train_metrics": train_metrics[selected],
            "heldout_metrics": aggregate(heldout, selected),
            "heldout_baseline": aggregate(heldout, "baseline"),
        })
    weighted = {}
    for key in METRICS:
        weighted[key] = float(sum(result["heldout_metrics"][key] * len(result["heldout_scenes"])
                                  for result in results) / len(rows))
    return results, weighted


def main():
    OUT.mkdir(parents=True, exist_ok=True); device = torch.device("cuda")
    state = torch.load(V27_ROOT / "outputs/set_reranker_best.pt", map_location=device, weights_only=True)
    model27 = v27.SetReranker().to(device); model27.load_state_dict(state["model"]); model27.eval()
    rows=[]; started=time.perf_counter(); truth=np.arange(SIDE*SIDE,dtype=np.int32)
    example_boards = {}
    for scene in SCENES:
        right,down=load_fused(scene,model27,device)
        anchor,_=v25.v10.assemble_components(right,down,SIDE)
        baseline=global_solver.solve_complete(right,down,SIDE,anchor,seed=SEED+scene,
            beam_width=4,hungarian_rounds=5,swap_proposals=12000)
        variants={}; boards={"baseline": baseline.board.copy()}; component_sizes={"baseline": []}
        if scene == 6989: example_boards["baseline"] = baseline.board.copy()
        unfreeze, objective = unfreeze_refine(right, down, baseline.board, SEED + scene + 7)
        boards["unfreeze"] = unfreeze; component_sizes["unfreeze"] = []
        variants["unfreeze"]={**global_solver.placement_metrics(unfreeze,truth,SIDE),"objective":objective}
        for topk in (1,2,4):
            board,objective,sizes=packed_refine(right,down,anchor,SEED+scene+topk,topk)
            boards[f"packed{topk}"] = board; component_sizes[f"packed{topk}"] = sizes
            variants[f"packed{topk}"]={**global_solver.placement_metrics(board,truth,SIDE),
                                      "objective":objective,"component_sizes":sizes}
            if scene == 6989 and topk == 1: example_boards["packed1"] = board.copy()
            if topk == 1:
                free_board, free_objective = unfreeze_refine(right, down, board, SEED + scene + 101)
                boards["packed1_unfreeze"] = free_board
                component_sizes["packed1_unfreeze"] = sizes
                variants["packed1_unfreeze"]={**global_solver.placement_metrics(free_board,truth,SIDE),
                                               "objective":free_objective}
        row={"scene":scene,"baseline":{**global_solver.placement_metrics(baseline.board,truth,SIDE),
             "objective":baseline.objective,"anchor_size":baseline.anchor_size},
             **variants,"seconds":time.perf_counter()-started}
        for method in METHODS:
            row[method]["selector_features"] = candidate_features(
                boards[method], baseline.board, right, down, baseline.anchor_size,
                component_sizes[method], method)
        rows.append(row);print(json.dumps({"event":"scene",**row}),flush=True)
    summaries = {method: aggregate(rows, method) for method in METHODS}
    folds, cv_selected = cross_validate(rows)
    selector_folds, selector_cv = ridge_selector(rows)
    maps = np.load(v25.MAP_FILE)["inv"]
    tiles = v25.load_raw_target_order(6989, maps).permute(0, 2, 3, 1).mul(255).byte().numpy()
    target = v25.v10.load_rgb(v25.RAW_INPUTS.parent / "targets" / "img_006989.png")
    montage = np.hstack((
        labelled(render_board(tiles, example_boards["baseline"]), "V28 + baseline global solver"),
        labelled(render_board(tiles, example_boards["packed1"]), "V29 packed-component solver"),
        labelled(target, "Clean target (reference)"),
    ))
    image_path = OUT / "assembly_scene_6989.png"
    cv2.imwrite(str(image_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    report={"schema":"puzzle-global-discrete-v29-cross-validation","scenes":list(SCENES),
            "selection_rule":"adjacency + 0.25 * translation_aligned_placement",
            "summaries":summaries,"folds":folds,"cv_selected":cv_selected,
            "selector":{"type":"ridge candidate ranker","alpha":8.,
                        "features":["edge_mean","edge_std","edge_p10","edge_median",
                                    "deficit_mean","deficit_p90","near_best_fraction",
                                    "baseline_position_agreement","baseline_edge_agreement",
                                    "anchor_fraction","component_count","largest_component_fraction",
                                    "method_one_hot"],
                        "folds":selector_folds,"cv":selector_cv},
            "visualization":str(image_path),"rows":rows,"seconds":time.perf_counter()-started}
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"event":"complete","report":report}),flush=True)


if __name__=="__main__":main()
