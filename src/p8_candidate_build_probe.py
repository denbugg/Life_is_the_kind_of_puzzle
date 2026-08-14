"""One FIT-source P8 candidate-cache diagnostic; no CAL/DEV/test access."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import torch
import infer_rank96 as rank96
import p3_g1_cdcs_capacity as p3
import p8_context_candidate_graph as p8

p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args()
c=SimpleNamespace(targets=p8.FIT_TARGETS,split=p8.SPLIT,p7=p8.P7_CKPT,work=a.work,device='cuda',seed=20260820,train_sources=128,eval_sources=32,steps=4000,batch_queries=12,eval_queries=256,phase='g0')
train,_=p8.sources(c)
print('before_model',flush=True); models=rank96.load_models(p3.config(),torch.device('cuda')); print('before_build',flush=True)
print(p8.build_one(c,train[0],0,models,torch.device('cuda')),flush=True)
print('after_build',flush=True)
