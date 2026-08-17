"""P38 SRIT-24: scaled raw-RGB transformer; FP32 only, no cached visual features."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p37_rit import N, RawRelationalTransformer, raw_tiles, splits, labels, equivariance, top20_recall


def seed(value=20260817):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def model(device):
    return RawRelationalTransformer(dim=512, depth=10, heads=8).to(device)


def gate_g0(args):
    seed()
    small = RawRelationalTransformer(dim=64, depth=2, heads=4).eval()
    pixels = torch.rand(N, 3, 20, 20)
    error = equivariance(small, pixels, torch.device("cpu"))
    return {"experiment":"P38_SRIT24","gate":"G0","equivariance_error":error,"invalid":0,"targets_opened":False,"p8_imported":False,"passes_G0":bool(error < 1e-5)}


def gate_g1(args):
    seed()
    device_ = torch.device("cuda")
    network = model(device_).eval()
    rows=[]
    for source in args.sources:
        error = equivariance(network, raw_tiles(args.inputs, source), device_)
        rows.append({"source":source,"equivariance_error":error,"finite":bool(np.isfinite(error))})
    maximum=max(row["equivariance_error"] for row in rows)
    invalid=sum(not row["finite"] for row in rows)
    return {"experiment":"P38_SRIT24","gate":"G1","sources":len(rows),"max_equivariance_error":maximum,"invalid":invalid,"targets_opened":False,"p8_imported":False,"passes_G1":bool(maximum < 1e-5 and invalid == 0),"rows":rows}


def remap(right, down, order, device):
    inverse=torch.empty_like(order)
    inverse[order]=torch.arange(N)
    r, d = right[order], down[order]
    vr, vd = r >= 0, d >= 0
    r=torch.where(vr, inverse[r.clamp_min(0)], r).to(device)
    d=torch.where(vd, inverse[d.clamp_min(0)], d).to(device)
    return r, d, vr.to(device), vd.to(device)


def gate_g2(args):
    seed()
    train, _, _ = splits(args.manifest)
    if len(args.sources) != 96 or set(args.sources) != set(train):
        raise RuntimeError("G2 must use exactly the locked 96 FIT-train sources")
    device_ = torch.device("cuda")
    network = model(device_)
    optimizer=torch.optim.AdamW(network.parameters(),lr=2e-4,weight_decay=0.05,betas=(0.9,0.95))
    epochs=80
    steps=epochs*len(args.sources)
    warmup=5*len(args.sources)
    schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step: (step+1)/warmup if step<warmup else 0.5*(1.0+math.cos(math.pi*(step-warmup)/max(1,steps-warmup))))
    data=[(source,raw_tiles(args.inputs,source),*labels(args.labels,source)) for source in args.sources]
    started=time.perf_counter(); global_step=0; terminal=float("inf")
    network.train()
    for epoch in range(epochs):
        order=list(range(len(data))); random.Random(20260817+epoch).shuffle(order); losses=[]
        for index in order:
            _, pixels, right, down=data[index]
            permutation=torch.randperm(N)
            pixels=pixels[permutation].unsqueeze(0).to(device_)
            right, down, vr, vd=remap(right,down,permutation,device_)
            optimizer.zero_grad(set_to_none=True)
            sr, sd=network(pixels)
            loss=F.cross_entropy(sr[0,vr],right[vr])+F.cross_entropy(sd[0,vd],down[vd])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),1.0)
            optimizer.step(); schedule.step(); global_step+=1
            losses.append(float(loss.detach().cpu()))
        terminal=float(np.mean(losses))
        print(json.dumps({"stage":"train","epoch":epoch+1,"epochs":epochs,"loss":terminal,"lr":float(schedule.get_last_lr()[0])}),flush=True)
    network.eval(); rows=[]
    with torch.no_grad():
        for source,pixels,right,down in data:
            sr,sd=network(pixels.unsqueeze(0).to(device_))
            value=top20_recall(sr[0],sd[0],right,down)
            rows.append({"source":source,"top20_recall":value,"valid":bool(np.isfinite(value))})
    recall=float(np.mean([row["top20_recall"] for row in rows])); invalid=sum(not row["valid"] for row in rows); seconds=time.perf_counter()-started
    args.model_out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":network.state_dict(),"sources":args.sources,"epochs":epochs},args.model_out)
    return {"experiment":"P38_SRIT24","gate":"G2","sources":len(rows),"top20_recall":recall,"terminal_loss":terminal,"invalid":invalid,"seconds":seconds,"targets_opened":False,"p8_imported":False,"selection_opened":False,"held_opened":False,"passes_G2":bool(recall >= .20 and terminal < 10.0 and invalid == 0 and seconds <= 1800),"rows":rows}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("g0","g1","g2"),required=True)
    parser.add_argument("--inputs",type=Path)
    parser.add_argument("--labels",type=Path)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--sources",nargs="*",default=[])
    parser.add_argument("--model-out",type=Path)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args()
    report={"g0":gate_g0,"g1":gate_g1,"g2":gate_g2}[args.mode](args)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)
    if not report[f"passes_{args.mode.upper()}"]:
        raise RuntimeError(f"P38 {args.mode.upper()} rejected")

if __name__=="__main__": main()
