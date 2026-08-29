"""Scene-group CV training for V32 spatial board reranking."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import spatial_critic_v32 as spatial

ROOT = Path("/home/kva/pazzle_global_autoresearch_v32_noise")
CACHE = ROOT / "spatial_cache"
OUT = ROOT / "outputs"
TRAIN = tuple(range(6700, 6728)) + tuple(range(6957, 6981))
VALID = tuple(range(6981, 6989))


def load_scene(scene):
    with np.load(CACHE / f"scene_{scene:06d}.npz", allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def normalizer(data, scenes):
    values = np.concatenate([data[s]["x_clean"].astype(np.float32) for s in scenes])
    mean = values.mean((0, 2, 3), keepdims=True)
    std = values.std((0, 2, 3), keepdims=True) + 1e-4
    return mean.astype(np.float32), std.astype(np.float32)


def update_ema(teacher, student, decay=.995):
    with torch.no_grad():
        for target, source in zip(teacher.parameters(), student.parameters()):
            target.mul_(decay).add_(source, alpha=1 - decay)


def sample_batch(data, scenes, rng, boards_per_scene=4):
    chosen_scenes = rng.sample(list(scenes), min(4, len(scenes)))
    clean = []; noisy = []; local = []; target = []; groups = []
    for group, scene in enumerate(chosen_scenes):
        row = data[scene]; real = np.flatnonzero(~row["synthetic"]); synth = np.flatnonzero(row["synthetic"])
        pool = np.concatenate((rng.choices(real.tolist(), k=boards_per_scene // 2),
                               rng.choices(synth.tolist() or real.tolist(), k=boards_per_scene - boards_per_scene // 2)))
        for index in pool:
            replica = rng.randrange(row["x_noise"].shape[1])
            clean.append(row["x_clean"][index]); noisy.append(row["x_noise"][index, replica])
            local.append(row["local"][index]); target.append(row["global_y"][index]); groups.append(group)
    return tuple(torch.from_numpy(np.asarray(value, np.float32)) for value in (clean, noisy, local, target)), groups


def fit(data, scenes, mean, std, config, seed, device):
    torch.manual_seed(seed); rng = random.Random(seed)
    model = spatial.SpatialBoardCritic(config[0], config[1]).to(device)
    teacher = copy.deepcopy(model).eval(); teacher.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=3e-3)
    mean_t, std_t = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    masks = torch.ones(2, 24, 24, device=device); masks[0, :, -1] = 0; masks[1, -1] = 0
    scaler = torch.amp.GradScaler("cuda")
    for step in range(1, config[2] + 1):
        (clean, noisy, local_y, global_y), groups = sample_batch(data, scenes, rng)
        clean, noisy, local_y, global_y = (value.to(device) for value in (clean, noisy, local_y, global_y))
        noisy = (noisy - mean_t) / std_t; clean = (clean - mean_t) / std_t
        model.train(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            score, local = model(noisy)
            with torch.no_grad(): teacher_score, teacher_local = teacher(clean)
            regression = F.huber_loss(torch.sigmoid(score), global_y)
            edge = F.binary_cross_entropy_with_logits(local[:, :2], local_y[:, :2], reduction="none")
            edge = (edge * masks).sum() / (masks.sum() * len(local))
            cell = F.huber_loss(torch.sigmoid(local[:, 2]), local_y[:, 2])
            pair_terms = []
            for group in set(groups):
                indices = [i for i, value in enumerate(groups) if value == group]
                for left, right in zip(indices[::2], indices[1::2]):
                    delta = global_y[left] - global_y[right]
                    if abs(float(delta)) > 1e-7:
                        pair_terms.append(F.softplus(-torch.sign(delta) * (score[left] - score[right])) *
                                          (torch.sqrt(abs(delta)) + .02))
            rank = torch.stack(pair_terms).mean() if pair_terms else score.sum() * 0
            ramp = min(1.0, step / max(1, config[2] * .15))
            consistency = F.huber_loss(score, teacher_score) + .25 * F.binary_cross_entropy_with_logits(
                local, torch.sigmoid(teacher_local))
            loss = (rank + .30 * regression + config[4] * edge + (config[4] / 3) * cell +
                    config[3] * ramp * consistency)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 2.0); scaler.step(optimizer); scaler.update()
        update_ema(teacher, model)
        if step % 250 == 0:
            print(json.dumps({"event":"train","step":step,"loss":float(loss.detach()),
                              "rank":float(rank.detach()),"edge":float(edge.detach())}), flush=True)
    return teacher.eval()


@torch.inference_mode()
def evaluate(model, data, scenes, mean, std, device):
    selected=[]; baseline=[]; oracle=[]; noisy_agreement=[]; rows=[]
    mean_t, std_t = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    for scene in scenes:
        row=data[scene]; real=np.flatnonzero(~row["synthetic"])
        clean=torch.from_numpy(row["x_clean"].astype(np.float32)).to(device)
        noisy=torch.from_numpy(row["x_noise"][:,0].astype(np.float32)).to(device)
        clean_score=model((clean-mean_t)/std_t)[0].cpu().numpy()
        noisy_score=model((noisy-mean_t)/std_t)[0].cpu().numpy()
        picked=int(real[np.argmax(noisy_score[real])]); clean_pick=int(real[np.argmax(clean_score[real])])
        base=int(row["baseline_index"]); best=int(real[np.argmax(row["global_y"][real])])
        selected.append(row["global_y"][picked]);baseline.append(row["global_y"][base]);oracle.append(row["global_y"][best])
        noisy_agreement.append(picked==clean_pick)
        rows.append({"scene":scene,"picked":str(row["names"][picked]),"selected":float(row["global_y"][picked]),
                     "baseline":float(row["global_y"][base]),"oracle":float(row["global_y"][best])})
    return {"selected":float(np.mean(selected)),"baseline":float(np.mean(baseline)),
            "oracle":float(np.mean(oracle)),"clean_noisy_agreement":float(np.mean(noisy_agreement)),"rows":rows}


def combine(rows):
    keys=("selected","baseline","oracle","clean_noisy_agreement")
    return {key:float(np.mean([row[key] for row in rows])) for key in keys}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--variant",choices=("s1","s2","s3","s4"),default="s3")
    parser.add_argument("--steps",type=int,default=3000);parser.add_argument("--cv-steps",type=int,default=1500)
    parser.add_argument("--smoke",action="store_true");args=parser.parse_args()
    available=tuple(scene for scene in TRAIN+VALID if (CACHE/f"scene_{scene:06d}.npz").exists())
    if len(available)<4: raise RuntimeError(f"need >=4 cached scenes, found {len(available)}")
    train=tuple(scene for scene in TRAIN if scene in available); valid=tuple(scene for scene in VALID if scene in available)
    if args.smoke: train=available[:3];valid=available[3:4];args.steps=min(args.steps,50)
    data={scene:load_scene(scene) for scene in train+valid}; mean,std=normalizer(data,train)
    configs={"s1":(64,96,args.steps,0.0,0.0),"s2":(72,104,args.steps,0.0,.45),
             "s3":(72,104,args.steps,.20,.45),"s4":(80,112,args.steps,.20,.45)}
    device=torch.device("cuda"); config=configs[args.variant]
    oof=[]
    if not args.smoke:
        folds=tuple(tuple(train[index::4]) for index in range(4))
        for fold,heldout in enumerate(folds):
            fitting=tuple(scene for scene in train if scene not in heldout)
            cv_config=(config[0],config[1],args.cv_steps,config[3],config[4])
            cv_model=fit(data,fitting,mean,std,cv_config,320826+fold*101,device)
            value=evaluate(cv_model,data,heldout,mean,std,device);oof.append(value)
            print(json.dumps({"event":"fold","fold":fold,**value}),flush=True)
    model=fit(data,train,mean,std,config,321826,device)
    report={"variant":args.variant,"parameters":spatial.parameter_count(model),"train_scenes":list(train),
            "validation_scenes":list(valid),"oof":combine(oof) if oof else None,
            "validation":evaluate(model,data,valid,mean,std,device)}
    OUT.mkdir(parents=True,exist_ok=True);torch.save({"model":model.state_dict(),"mean":mean,"std":std,
        "config":configs[args.variant][:2],"report":report},OUT/f"spatial_critic_{args.variant}.pt")
    (OUT/f"spatial_critic_{args.variant}.json").write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=="__main__":main()
