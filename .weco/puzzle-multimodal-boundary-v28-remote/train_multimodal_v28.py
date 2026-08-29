"""V28 multimodal boundary encoder: raw RGB + denoised gray + learned contours."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path("/home/kva/pazzle_multimodal_boundary_v28")
V25_ROOT = Path("/home/kva/pazzle_v18_v22_v23_fusion_v25")
V26_ROOT = Path("/home/kva/pazzle_union_reranker_v26")
V27_ROOT = Path("/home/kva/pazzle_set_transformer_v27")
sys.path[:0] = [str(ROOT), str(V27_ROOT), str(V26_ROOT), str(V25_ROOT),
                "/home/kva/pazzle_boundary_biencoder_v23_xl"]
import tile_denoiser
import evaluate_fusion_v25 as v25
import evaluate_union_reranker_v26 as v26
import evaluate_set_transformer_v27 as v27
import train_boundary_biencoder_v23 as v23

OUT = ROOT / "outputs"
CACHE = ROOT / "score_cache"
DENOISER_CKPT = Path("/home/kva/pazzle_cnn_gnn_xl_20260825/models/real_fragment_restorer_best.pt")
V23_XL_CKPT = Path("/home/kva/pazzle_boundary_biencoder_v23_xl/outputs/boundary_biencoder_best.pt")
V27_CKPT = V27_ROOT / "outputs/set_reranker_best.pt"
SELECT_SCENES = range(6728, 6732)
CALIBRATION_SCENES = range(6732, 6736)
TEST_SCENES = range(6989, 7000)
ASSEMBLY_SCENE = 6989
STEPS = 900
SEED = 280826


@dataclass(frozen=True)
class LossConfig:
    hard_weight: float = .12
    hard_margin: float = .38


class MultimodalBoundary(v23.BoundaryBiEncoder):
    """V23-XL backbone whose 36 side channels come from four explicit modalities."""
    def side_features(self, views, side):
        features = []
        for width in self.config.widths:
            if side == "right": strip = views[:, :, :, -width:]
            elif side == "left": strip = views[:, :, :, :width]
            elif side == "bottom": strip = views[:, :, -width:, :].transpose(-2, -1)
            else: strip = views[:, :, :width, :].transpose(-2, -1)
            features.extend((strip.mean(-1), strip.std(-1, unbiased=False)))
        return torch.cat(features, 1)


def crop_indices(side, rng):
    row = int(rng.integers(0, 24 - side + 1)); col = int(rng.integers(0, 24 - side + 1))
    grid = np.arange(576).reshape(24, 24)
    return grid[row:row+side, col:col+side].reshape(-1)


@torch.inference_mode()
def modality_views(raw, denoiser, contour_module, contour_net, threshold):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        denoised = denoiser(raw).float().clamp(0, 1)
    den_u8 = denoised.mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    robust = torch.from_numpy(contour_module._robust_features_np(den_u8)).to(raw.device)
    soft = contour_net(robust).sigmoid().unsqueeze(1)
    binary = (soft >= threshold).float()
    gray = .299 * denoised[:, :1] + .587 * denoised[:, 1:2] + .114 * denoised[:, 2:3]
    return torch.cat((raw, gray, soft, binary), 1)


def training_loss(model, raw, stable, side, generator):
    raw_a = v23.augment_tiles(raw, generator, strong=True)
    raw_b = v23.augment_tiles(raw, generator, strong=True)
    a = torch.cat((raw_a, stable[:, 3:]), 1)
    b = torch.cat((raw_b, stable[:, 3:]), 1)
    ea, eb = model(a), model(b); config = LossConfig()
    terms = (
        v23.direction_loss(ea["right"], eb["left"], side, "right", model.scale(), config),
        v23.direction_loss(ea["bottom"], eb["top"], side, "down", model.scale(), config),
        v23.direction_loss(eb["right"], ea["left"], side, "right", model.scale(), config),
        v23.direction_loss(eb["bottom"], ea["top"], side, "down", model.scale(), config),
    )
    return torch.stack([term[0] for term in terms]).mean()


@torch.inference_mode()
def score_scene(model, scene, denoiser, contour_module, contour_net, threshold, maps, device):
    raw = v25.load_raw_target_order(scene, maps).to(device)
    views = modality_views(raw, denoiser, contour_module, contour_net, threshold)
    with torch.autocast("cuda", dtype=torch.bfloat16): embeddings = model(views)
    scale = float(model.scale())
    return [v25.row_z((scale * embeddings["right"] @ embeddings["left"].t()).float().cpu().numpy()),
            v25.row_z((scale * embeddings["bottom"] @ embeddings["top"].t()).float().cpu().numpy())]


def evaluate_standalone(model, scenes, denoiser, contour_module, contour_net, threshold, maps, device):
    return v25.aggregate([v25.metrics(score_scene(
        model, scene, denoiser, contour_module, contour_net, threshold, maps, device)) for scene in scenes])


def train(model, denoiser, contour_module, contour_net, threshold, maps, device):
    rng = np.random.default_rng(SEED + 2)
    generator = torch.Generator(device=device).manual_seed(SEED + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-4, weight_decay=.035, betas=(.9, .95))
    scaler = torch.amp.GradScaler("cuda", init_scale=128.)
    best, best_metric, history, started = None, -1., [], time.perf_counter()
    for step in range(1, STEPS + 1):
        side = 12 if step <= 450 else 16
        scene = int(rng.integers(0, 6700)); indices = crop_indices(side, rng)
        raw = v25.load_raw_target_order(scene, maps)[indices].to(device)
        stable = modality_views(raw, denoiser, contour_module, contour_net, threshold)
        model.train(); optimizer.zero_grad(set_to_none=True)
        progress = step / STEPS
        lr = 1.0e-5 + .5 * (1.8e-4 - 1.0e-5) * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups: group["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = training_loss(model, raw, stable, side, generator)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        grad = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        scaler.step(optimizer); scaler.update()
        if step % 25 == 0:
            print(json.dumps({"event":"train","step":step,"loss":float(loss.detach()),
                              "lr":lr,"grad":grad,"seconds":time.perf_counter()-started}), flush=True)
        if step % 300 == 0:
            model.eval()
            value = evaluate_standalone(model, SELECT_SCENES, denoiser, contour_module,
                                        contour_net, threshold, maps, device)
            metric = v25.objective(value)
            history.append({"step":step, **value, "objective":metric})
            print(json.dumps({"event":"selection",**history[-1]}), flush=True)
            if metric > best_metric:
                best_metric, best = metric, copy.deepcopy(model.state_dict())
                torch.save({"model":best,"step":step,"metric":metric,
                            "model_config":vars(model.config)}, OUT / "multimodal_best.pt")
    model.load_state_dict(best)
    return model.eval(), history, best_metric


def load_or_score_v28(model, scene, denoiser, contour_module, contour_net, threshold, maps, device):
    path = CACHE / f"scene_{scene:06d}.npz"
    if not path.exists():
        scores = score_scene(model, scene, denoiser, contour_module, contour_net, threshold, maps, device)
        np.savez_compressed(path, scores=np.asarray(scores, np.float16))
    with np.load(path) as data: return data["scores"].astype(np.float32)


def blend(base, extra, alpha):
    return [v25.row_z((1-alpha)*base[d] + alpha*extra[d]) for d in range(2)]


def render_board(tiles, board):
    grid = tiles[np.asarray(board).reshape(24,24)]
    return grid.transpose(0,2,1,3,4).reshape(480,480,3)


def labelled(image, text):
    canvas=np.full((525,480,3),255,np.uint8);canvas[45:]=image
    cv2.putText(canvas,text,(10,29),cv2.FONT_HERSHEY_SIMPLEX,.67,(15,15,15),2,cv2.LINE_AA)
    return canvas


def assemble(scene, base, fused, maps):
    truth=np.arange(576,dtype=np.int32); rows={}; boards={}
    for name,matrices in (("v27",base),("v28",fused)):
        anchor,_=v25.v10.assemble_components(matrices[0],matrices[1],24)
        result=v27.global_solver.solve_complete(matrices[0],matrices[1],24,anchor,
            seed=SEED+scene,beam_width=4,hungarian_rounds=5,swap_proposals=12000)
        boards[name]=result.board
        rows[name]={**v27.global_solver.placement_metrics(result.board,truth,24),
                    "objective":result.objective,"anchor_size":result.anchor_size}
    tiles=v25.load_raw_target_order(scene,maps).permute(0,2,3,1).mul(255).byte().numpy()
    target=v25.v10.load_rgb(v25.RAW_INPUTS.parent/"targets"/f"img_{scene:06d}.png")
    montage=np.hstack((labelled(render_board(tiles,boards["v27"]),"V27 + global solver"),
                       labelled(render_board(tiles,boards["v28"]),"V28 multimodal + solver"),
                       labelled(target,"Clean target (reference)")))
    path=OUT/f"assembly_scene_{scene}.png"
    cv2.imwrite(str(path),cv2.cvtColor(montage,cv2.COLOR_RGB2BGR));return rows,path


def main():
    torch.manual_seed(SEED);np.random.seed(SEED);OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda");torch.backends.cuda.matmul.allow_tf32=True
    maps=np.load(v25.MAP_FILE)["inv"]
    denoiser,_=tile_denoiser.load_denoiser(DENOISER_CKPT,device)
    contour_module,_,_,contour_net,threshold,_=v25.load_winner(device)
    state=torch.load(V23_XL_CKPT,map_location="cpu",weights_only=True)
    model=MultimodalBoundary(v23.ModelConfig(**state["model_config"])).to(device)
    model.load_state_dict(state["model"],strict=True)
    parameters=sum(p.numel() for p in model.parameters())
    model,history,best_metric=train(model,denoiser,contour_module,contour_net,threshold,maps,device)

    loaded=v25.load_models(device); frozen=loaded[:4]; winner=v25.load_winner(device)
    state27=torch.load(V27_CKPT,map_location=device,weights_only=True)
    model27=v27.SetReranker().to(device);model27.load_state_dict(state27["model"]);model27.eval()
    all_scenes=list(CALIBRATION_SCENES)+list(TEST_SCENES); bundles={};started=time.perf_counter()
    for index,scene in enumerate(all_scenes,1):
        old=v26.cache_scene(scene,frozen,winner,device,maps)
        base=v27.rerank_scene(model27,old,float(state27["beta"]),device)
        extra=load_or_score_v28(model,scene,denoiser,contour_module,contour_net,threshold,maps,device)
        bundles[scene]=(base,extra)
        print(json.dumps({"event":"score","scene":scene,"of":len(all_scenes),"seconds":time.perf_counter()-started}),flush=True)
    trials=[]
    for alpha in (0.,.05,.10,.15,.20,.30,.40,.55,.70):
        value=v25.aggregate([v25.metrics(blend(*bundles[s],alpha)) for s in CALIBRATION_SCENES])
        trials.append({"alpha":alpha,**value,"objective":v25.objective(value)})
    selected=max(trials,key=lambda x:x["objective"])
    base_test=v25.aggregate([v25.metrics(bundles[s][0]) for s in TEST_SCENES])
    extra_test=v25.aggregate([v25.metrics(list(bundles[s][1])) for s in TEST_SCENES])
    fused_test=v25.aggregate([v25.metrics(blend(*bundles[s],selected["alpha"])) for s in TEST_SCENES])
    assembly,image=assemble(ASSEMBLY_SCENE,bundles[ASSEMBLY_SCENE][0],
                            blend(*bundles[ASSEMBLY_SCENE],selected["alpha"]),maps)
    report={"schema":"puzzle-multimodal-boundary-v28","parameters":parameters,
            "modalities":["raw_rgb","unet_denoised_grayscale","soft_learned_contour","binary_learned_contour"],
            "train_scenes":[0,6699],"selection_scenes":[min(SELECT_SCENES),max(SELECT_SCENES)],
            "calibration_scenes":[min(CALIBRATION_SCENES),max(CALIBRATION_SCENES)],
            "test_scenes":[min(TEST_SCENES),max(TEST_SCENES)],"history":history,
            "selection_best_objective":best_metric,"selected":selected,"test_v27":base_test,
            "test_v28_standalone":extra_test,"test_v28_fused":fused_test,
            "assembly_scene":ASSEMBLY_SCENE,"assembly":assembly,"assembly_image":str(image)}
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"event":"complete","report":report}),flush=True)


if __name__=="__main__":main()
