"""Quick smoke test: shapes, forward passes, one train step for each model."""
import time, numpy as np, torch
from config import FS, NFRAG
from imgio import train_val_split
from datasets import RestoreDataset, CompatDataset
from models import RestoreNet, CompatNet, restore_loss, count_params
from train_compat import grid_targets, losses_and_acc
DEV = "cuda"

trn, val = train_val_split()
print("train/val:", len(trn), len(val))

# ---- Restore ----
rds = RestoreDataset(trn, crop_frags=12, real_prob=0.5)
t = time.time(); di, cl = rds[0]; print("restore item", di.shape, cl.shape, f"{time.time()-t:.3f}s")
m = RestoreNet(base=48).to(DEV)
print("RestoreNet params", f"{count_params(m):,}")
x = di.unsqueeze(0).to(DEV)
with torch.autocast("cuda", dtype=torch.float16):
    y = m(x); loss = restore_loss(y.float(), cl.unsqueeze(0).to(DEV).float())
print("restore fwd", y.shape, "loss", float(loss), "mem", f"{torch.cuda.max_memory_allocated()/1e9:.2f}GB")

# batched fwd/bwd timing at bs=16
xb = torch.rand(16, 3, 240, 240, device=DEV)
opt = torch.optim.AdamW(m.parameters(), 1e-3); sc = torch.amp.GradScaler("cuda")
torch.cuda.synchronize(); t = time.time()
for _ in range(5):
    with torch.autocast("cuda", dtype=torch.float16):
        yb = m(xb); lb = restore_loss(yb.float(), torch.rand_like(yb).float())
    opt.zero_grad(); sc.scale(lb).backward(); sc.step(opt); sc.update()
torch.cuda.synchronize(); print("restore bs16 step", f"{(time.time()-t)/5:.3f}s/it",
                                "mem", f"{torch.cuda.max_memory_allocated()/1e9:.2f}GB")
del m, xb, yb, opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

# ---- Compat ----
cds = CompatDataset(trn, real_prob=0.5)
t = time.time(); fr = cds[0]; print("compat item", fr.shape, f"{time.time()-t:.3f}s")
cm = CompatNet().to(DEV); print("CompatNet params", f"{count_params(cm):,}")
ha, ht, va, vt = grid_targets()
fb = torch.stack([cds[i] for i in range(6)])
opt = torch.optim.AdamW(cm.parameters(), 1e-3); sc = torch.amp.GradScaler("cuda")
torch.cuda.synchronize(); t = time.time()
for _ in range(3):
    with torch.autocast("cuda", dtype=torch.float16):
        loss, h, v = losses_and_acc(cm, fb, ha, ht, va, vt)
    opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update()
torch.cuda.synchronize()
print("compat bs6 step", f"{(time.time()-t)/3:.3f}s/it", "loss", float(loss),
      "H@1", f"{h:.3f}", "V@1", f"{v:.3f}", "mem", f"{torch.cuda.max_memory_allocated()/1e9:.2f}GB")
print("SMOKE OK")
