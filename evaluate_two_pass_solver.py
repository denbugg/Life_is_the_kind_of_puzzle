"""Honest no-oracle ablation for tile diffusion -> RL -> context -> RL."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

os.environ.setdefault("EDGE_TOPK", "8")
os.environ.setdefault("SWAP_STEPS", "30000")
os.environ.setdefault("RL_STEPS", "1200")
os.environ.setdefault("RL_PROPOSALS", "64")
os.environ.setdefault("RL_STAGNATION", "180")
os.environ.setdefault("POSITION_WEIGHT", "0.01")

import kaggle_solve_puzzles as solver
from kaggle_ddpm_denoise_fragments import TinyCondUNet
from train_rl_swap_actor_critic_v2 import SwapActorCritic as TrainedSwapActorCritic

ROOT = Path(os.getenv("MODEL_ROOT", "."))
SOURCE = os.getenv("SOURCE", "img_002127")
VARIANT = int(os.getenv("VARIANT", "0"))
SEED = int(os.getenv("SEED", "20260730"))


def load_model(cls, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = cls().to(solver.DEVICE)
    model.load_state_dict(ck["model"])
    return model.eval()


def true_layout_for_perm(perm):
    layout = np.empty(solver.N, dtype=np.int32)
    layout[perm] = np.arange(solver.N, dtype=np.int32)
    return layout


def exact_accuracy(layout, truth):
    return float(np.mean(np.asarray(layout) == truth))


def adjacency_accuracy(layout, perm):
    original = perm[np.asarray(layout)].reshape(solver.GRID, solver.GRID)
    right = original[:, 1:] == original[:, :-1] + 1
    down = original[1:, :] == original[:-1, :] + solver.GRID
    return float((right.sum() + down.sum()) / (right.size + down.size))


def psnr(image, target):
    mse = np.mean((image.astype(np.float32)-target.astype(np.float32))**2)/(255**2)
    return float(-10*np.log10(max(mse,1e-12)))


@torch.inference_mode()
def context_refine(model, image, batch_size=48):
    tensor=torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))).float()/127.5-1
    patches=[]; coords=[]
    for r in range(22):
        for c in range(22):
            patches.append(tensor[:,r*20:r*20+60,c*20:c*20+60]); coords.append((r,c))
    accum=torch.zeros((3,480,480),dtype=torch.float32)
    weight=torch.zeros((1,480,480),dtype=torch.float32)
    gen=torch.Generator(device=solver.DEVICE).manual_seed(SEED+700)
    for start in range(0,len(patches),batch_size):
        cond=torch.stack(patches[start:start+batch_size]).to(solver.DEVICE)
        noise=torch.randn(cond.shape,generator=gen,device=solver.DEVICE,dtype=cond.dtype)
        t=torch.full((len(cond),),199,device=solver.DEVICE,dtype=torch.long)
        pred=(cond+model(noise,cond,t).clamp(-1,1)).clamp(-1,1).cpu()
        for patch,(r,c) in zip(pred,coords[start:start+batch_size]):
            y,x=r*20,c*20; accum[:,y:y+60,x:x+60]+=patch; weight[:,y:y+60,x:x+60]+=1
    result=(accum/weight.clamp_min(1))
    return ((result.permute(1,2,0).numpy()+1)*127.5).round().clip(0,255).astype(np.uint8)


def tiles_from_image(image):
    return image.reshape(24,20,24,20,3).transpose(0,2,1,3,4).reshape(576,20,20,3)


def make_montage(items, path):
    size=320; margin=14; title=34
    canvas=Image.new("RGB",(len(items)*size+(len(items)+1)*margin,size+title+2*margin),"white")
    draw=ImageDraw.Draw(canvas)
    for i,(name,img) in enumerate(items):
        x=margin+i*(size+margin); draw.text((x,margin+8),name,fill="black")
        canvas.paste(Image.fromarray(img).resize((size,size)),(x,margin+title))
    canvas.save(path)


def main():
    edge=load_model(solver.EdgeMatcher,ROOT/"edge_matcher_diffusion_v3_epoch4.pt")
    pos=load_model(solver.PositionPrior,ROOT/"position_prior_diffusion_v3_epoch4.pt")
    rl=load_model(TrainedSwapActorCritic,ROOT/"rl_swap_actor_critic_diffusion_v3_epoch14.pt")
    ctx_ck=torch.load(ROOT/"context_diffusion_v3_best.pt",map_location="cpu",weights_only=False)
    context=TinyCondUNet(base=64).to(solver.DEVICE)
    context.load_state_dict(ctx_ck["model"]); context.eval()

    clean=np.asarray(Image.open(ROOT/"clean_targets"/f"{SOURCE}.png").convert("RGB").resize((480,480)),np.uint8)
    restored=np.asarray(Image.open(ROOT/"restored_rl_targets"/f"{SOURCE}_v{VARIANT:02d}.png").convert("RGB"),np.uint8)
    original_tiles=tiles_from_image(restored)
    rng=np.random.default_rng(SEED)
    perm=rng.permutation(solver.N)
    tiles=original_tiles[perm]
    truth=true_layout_for_perm(perm)
    tiles_t=solver.tiles_tensor(tiles)

    pos_score,row_logp,col_logp=solver.position_scores(pos,tiles_t)
    right=solver.compatibility_scores(edge,tiles,tiles_t,0)
    down=solver.compatibility_scores(edge,tiles,tiles_t,1)
    greedy,graph_stats=solver.greedy_graph_layout(right,down,pos_score)
    optimized=solver.optimize_layout(greedy,right,down,pos_score,SEED)
    rl1,_,steps1=solver.refine_layout_rl(rl,tiles,row_logp,col_logp,optimized,SEED+1)
    rl1,_,_,accepted1=solver.select_layout_candidate(optimized,rl1,right,down,pos_score)
    assembled1=solver.assemble(tiles,rl1)

    context_image=context_refine(context,assembled1)
    positional=tiles_from_image(context_image)
    refined=np.empty_like(positional)
    for position,tile_id in enumerate(rl1):
        refined[int(tile_id)]=positional[position]
    refined_t=solver.tiles_tensor(refined)
    pos2,row2,col2=solver.position_scores(pos,refined_t)
    right2=solver.compatibility_scores(edge,refined,refined_t,0)
    down2=solver.compatibility_scores(edge,refined,refined_t,1)
    rl2,_,steps2=solver.refine_layout_rl(rl,refined,row2,col2,rl1,SEED+2)
    final,_,_,accepted2=solver.select_layout_candidate(rl1,rl2,right2,down2,pos2)
    final_image=solver.assemble(refined,final)

    stages={
        "greedy":{"exact":exact_accuracy(greedy,truth),"adjacency":adjacency_accuracy(greedy,perm)},
        "optimized":{"exact":exact_accuracy(optimized,truth),"adjacency":adjacency_accuracy(optimized,perm)},
        "rl_first":{"exact":exact_accuracy(rl1,truth),"adjacency":adjacency_accuracy(rl1,perm),
                    "steps":steps1,"accepted":bool(accepted1),"psnr":psnr(assembled1,clean)},
        "context":{"psnr":psnr(context_image,clean)},
        "rl_second":{"exact":exact_accuracy(final,truth),"adjacency":adjacency_accuracy(final,perm),
                     "steps":steps2,"accepted":bool(accepted2),"psnr":psnr(final_image,clean)},
        "graph":graph_stats,
    }
    Path("two_pass_metrics.json").write_text(json.dumps(stages,indent=2))
    shuffled=solver.assemble(tiles,np.arange(solver.N))
    make_montage([
        ("Shuffled",shuffled),("Greedy",solver.assemble(tiles,greedy)),
        ("RL pass 1",assembled1),("Context",context_image),
        ("RL pass 2",final_image),("Clean target",clean),
    ],Path("two_pass_example.png"))
    print(json.dumps(stages,indent=2))


if __name__=="__main__":
    main()
