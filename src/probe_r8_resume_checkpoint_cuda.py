import json
import random
import time
from pathlib import Path
import torch
from canvas_data import CanvasDataset
from train_r8_holistic_pair import HolisticPairNet, backward_sampled_loss

ckpt_path = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R8_holistic_full_pair\g1_capacity_retry1_microbatch\r8_last.pt")
split_path = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json")
device = torch.device("cuda")
payload = torch.load(ckpt_path, map_location=device, weights_only=False)
model = HolisticPairNet(**payload["architecture"]).to(device)
model.load_state_dict(payload["model"])
model.train()
names = json.loads(split_path.read_text(encoding="utf-8"))["splits"]["fit"]
dataset = CanvasDataset(names[:5360], real_prob=0.0, seed=20260825)
row = dataset[0]
tiles, perm = row["tiles"].to(device), row["perm"].to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
opt.zero_grad(set_to_none=True)
started = time.time()
loss, stats = backward_sampled_loss(model, tiles, perm, anchors_per_board=96, negatives=15, row_microbatch=24, rng=random.Random(20260814))
grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).item())
opt.step()
torch.cuda.synchronize()
print(json.dumps({"checkpoint_step": payload["step"], "loss": loss, "stats": stats, "grad_norm": grad, "elapsed_s": time.time()-started, "cuda_allocated": torch.cuda.memory_allocated(), "cuda_reserved": torch.cuda.memory_reserved()}))
