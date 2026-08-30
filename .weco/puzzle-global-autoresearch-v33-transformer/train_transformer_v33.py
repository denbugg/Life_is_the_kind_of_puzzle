"""Group-CV training of V33 transformer board rerankers."""
from __future__ import annotations
import argparse,copy,json,random,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path("/home/kva/pazzle_global_autoresearch_v33_transformer")
V32=Path("/home/kva/pazzle_global_autoresearch_v32_noise")
sys.path.insert(0,str(ROOT))
import board_transformer_v33 as architecture
CACHE=V32/"spatial_cache";OUT=ROOT/"outputs"
TRAIN=tuple(range(6700,6728))+tuple(range(6957,6981));VALID=tuple(range(6981,6989))

def load_scene(scene):
    with np.load(CACHE/f"scene_{scene:06d}.npz",allow_pickle=False) as z:return {key:z[key] for key in z.files}

def normalizer(data,scenes):
    values=np.concatenate([data[s]["x_clean"].astype(np.float32) for s in scenes])
    return values.mean((0,2,3),keepdims=True).astype("f4"),(values.std((0,2,3),keepdims=True)+1e-4).astype("f4")

def sample_batch(data,scenes,rng,boards_per_scene=4):
    chosen=rng.sample(list(scenes),min(2,len(scenes)));clean=[];noisy=[];local=[];target=[];groups=[];synthetic=[]
    for group,scene in enumerate(chosen):
        row=data[scene];real=np.flatnonzero(~row["synthetic"]);near=np.flatnonzero(row["synthetic"])
        indices=rng.choices(real.tolist(),k=boards_per_scene//2)+rng.choices(near.tolist() or real.tolist(),k=boards_per_scene-boards_per_scene//2)
        for index in indices:
            replica=rng.randrange(row["x_noise"].shape[1]);clean.append(row["x_clean"][index]);noisy.append(row["x_noise"][index,replica])
            local.append(row["local"][index]);target.append(row["global_y"][index]);groups.append(group);synthetic.append(bool(row["synthetic"][index]))
    tensors=tuple(torch.from_numpy(np.asarray(value,np.float32)) for value in (clean,noisy,local,target))
    return tensors,groups,synthetic

@torch.no_grad()
def update_ema(teacher,student,decay=.995):
    for target,source in zip(teacher.parameters(),student.parameters()):target.mul_(decay).add_(source,alpha=1-decay)

def fit(data,scenes,mean,std,variant,steps,seed,device,consistency):
    torch.manual_seed(seed);rng=random.Random(seed);model,_,_=architecture.make_variant(variant,steps,consistency);model=model.to(device)
    teacher=copy.deepcopy(model).eval();teacher.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.05,betas=(.9,.95));scaler=torch.amp.GradScaler("cuda")
    mean_t,std_t=torch.from_numpy(mean).to(device),torch.from_numpy(std).to(device)
    masks=torch.ones(2,24,24,device=device);masks[0,:,-1]=0;masks[1,-1]=0
    for step in range(1,steps+1):
        (clean,noisy,local_y,global_y),groups,synthetic=sample_batch(data,scenes,rng)
        clean,noisy,local_y,global_y=(value.to(device) for value in (clean,noisy,local_y,global_y));clean=(clean-mean_t)/std_t;noisy=(noisy-mean_t)/std_t
        if model.training and rng.random()<.8:
            token_mask=torch.rand((noisy.shape[0],1,24,24),device=device)<.06;noisy=noisy.masked_fill(token_mask,0)
        model.train();optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            score,local=model(noisy)
            with torch.no_grad():teacher_score,teacher_local=teacher(clean)
            regression=F.huber_loss(torch.sigmoid(score),global_y)
            edge=(F.binary_cross_entropy_with_logits(local[:,:2],local_y[:,:2],reduction="none")*masks).sum()/(masks.sum()*len(local))
            cell=F.huber_loss(torch.sigmoid(local[:,2]),local_y[:,2]);pairs=[];monotonic=[]
            for group in set(groups):
                indices=[i for i,value in enumerate(groups) if value==group]
                for left in indices:
                    for right in indices:
                        delta=global_y[left]-global_y[right]
                        if float(delta)>1e-7:
                            term=F.softplus(-(score[left]-score[right]))*(torch.sqrt(delta)+.02);pairs.append(term)
                            if not synthetic[left] and synthetic[right]:monotonic.append(term)
            rank=torch.stack(pairs).mean() if pairs else score.sum()*0
            mono=torch.stack(monotonic).mean() if monotonic else score.sum()*0
            ramp=min(1.,step/max(1,steps*.15));consistent=F.huber_loss(score,teacher_score)+.25*F.binary_cross_entropy_with_logits(local,torch.sigmoid(teacher_local))
            loss=rank+.30*regression+.45*edge+.15*cell+.10*mono+consistency*ramp*consistent
        scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(optimizer);scaler.update();update_ema(teacher,model)
        if step%250==0:print(json.dumps({"event":"train","step":step,"loss":float(loss.detach()),"rank":float(rank.detach()),"edge":float(edge.detach())}),flush=True)
    return teacher.eval()

@torch.inference_mode()
def predict(model,row,mean_t,std_t,device,view="noisy"):
    values=row["x_noise"][:,0] if view=="noisy" else row["x_clean"]
    x=torch.from_numpy(values.astype(np.float32)).to(device);scores=[]
    for start in range(0,len(x),8):scores.append(model((x[start:start+8]-mean_t)/std_t)[0].float().cpu())
    return torch.cat(scores).numpy()

def evaluation_rows(model,data,scenes,mean,std,device):
    mean_t,std_t=torch.from_numpy(mean).to(device),torch.from_numpy(std).to(device);rows=[]
    for scene in scenes:
        row=data[scene];real=np.flatnonzero(~row["synthetic"]);score=predict(model,row,mean_t,std_t,device);clean=predict(model,row,mean_t,std_t,device,"clean")
        picked=int(real[np.argmax(score[real])]);clean_pick=int(real[np.argmax(clean[real])]);base=int(row["baseline_index"]);best=int(real[np.argmax(row["global_y"][real])])
        rows.append({"scene":scene,"picked":picked,"base":base,"best":best,"margin":float(score[picked]-score[base]),"agree":picked==clean_pick,
                     "selected":float(row["global_y"][picked]),"baseline":float(row["global_y"][base]),"oracle":float(row["global_y"][best])})
    return rows

def aggregate(rows,threshold=None):
    if threshold is None: selected=[r["selected"] for r in rows]
    else:selected=[r["selected"] if r["margin"]>threshold else r["baseline"] for r in rows]
    return {"selected":float(np.mean(selected)),"baseline":float(np.mean([r["baseline"] for r in rows])),"oracle":float(np.mean([r["oracle"] for r in rows])),"agreement":float(np.mean([r["agree"] for r in rows]))}

def main():
    p=argparse.ArgumentParser();p.add_argument("--variant",choices=("ts","tm","tmc"),default="ts");p.add_argument("--steps",type=int,default=3000);p.add_argument("--cv-steps",type=int,default=1200);p.add_argument("--smoke",action="store_true");a=p.parse_args()
    data={scene:load_scene(scene) for scene in TRAIN+VALID};train=TRAIN;valid=VALID
    if a.smoke:train=TRAIN[:3];valid=TRAIN[3:4];a.steps=20;a.cv_steps=0
    mean,std=normalizer(data,train);device=torch.device("cuda");_,_,consistency=architecture.make_variant(a.variant,a.steps)
    oof=[]
    if not a.smoke:
        folds=tuple(tuple(train[index::4]) for index in range(4))
        for fold,heldout in enumerate(folds):
            fitting=tuple(scene for scene in train if scene not in heldout)
            model=fit(data,fitting,mean,std,a.variant,a.cv_steps,330826+fold*101,device,consistency);rows=evaluation_rows(model,data,heldout,mean,std,device);oof+=rows
            print(json.dumps({"event":"fold","fold":fold,**aggregate(rows)}),flush=True)
    thresholds=[-1e9] if not oof else sorted(set([-1e9,0.0]+[r["margin"] for r in oof]))
    threshold=max(thresholds,key=lambda value:aggregate(oof,value)["selected"]) if oof else -1e9
    final=fit(data,train,mean,std,a.variant,a.steps,331826,device,consistency);validation_rows=evaluation_rows(final,data,valid,mean,std,device)
    report={"variant":a.variant,"parameters":architecture.parameter_count(final),"threshold":threshold,"oof":aggregate(oof,threshold) if oof else None,
            "validation":aggregate(validation_rows,threshold),"validation_rows":validation_rows}
    OUT.mkdir(parents=True,exist_ok=True);torch.save({"model":final.state_dict(),"mean":mean,"std":std,"variant":a.variant,"report":report},OUT/f"transformer_{a.variant}.pt")
    (OUT/f"transformer_{a.variant}.json").write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=="__main__":main()
