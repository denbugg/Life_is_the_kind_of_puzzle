"""Minimal FIT-safe P8 rank96 model-load diagnostic; no image or target access."""
import torch
import p3_g1_cdcs_capacity as p3
import infer_rank96 as rank96

print("before_load",flush=True)
models=rank96.load_models(p3.config(),torch.device("cuda"))
print({"after_load":True,"ranker":type(models.ranker).__name__,"affinity":type(models.affinity_primary).__name__},flush=True)
