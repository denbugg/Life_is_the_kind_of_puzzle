import json
import random
import time
import torch
from canvas_data import CanvasDataset
from train_r8_holistic_pair import HolisticPairNet, backward_sampled_loss

names = json.loads(open(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json", encoding="utf-8").read())["splits"]["fit"]
device = torch.device("cuda")
torch.manual_seed(20260814)
rng = random.Random(20260814)
dataset = CanvasDataset(names[:5360], real_prob=0.0, seed=20260825)
row = dataset[0]
tiles, perm = row["tiles"].to(device), row["perm"].to(device)
model = HolisticPairNet(width=96, blocks=5).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
opt.zero_grad(set_to_none=True)
started = time.time()
loss, stats = backward_sampled_loss(model, tiles, perm, anchors_per_board=96, negatives=15, row_microbatch=24, rng=rng)
grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).item())
opt.step()
torch.cuda.synchronize()
print(json.dumps({"loss": loss, "stats": stats, "grad_norm": grad, "elapsed_s": time.time()-started, "cuda_allocated": torch.cuda.memory_allocated(), "cuda_reserved": torch.cuda.memory_reserved()}))
