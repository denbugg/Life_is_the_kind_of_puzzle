"""Train the rendered assignment, on a size curriculum, read as PLACEMENT.

The only number reported is how many fragments land in their own cell. SSIM is
deliberately absent: the project decision is that assembly moves first and
restoration is not touched until it does.

The curriculum is the one thing in the global family that already works. The
discrete assembler reaches 0.2222 at 6x6 against a chance of 0.0278 and 0.0625
at 12x12 against 0.0069 -- eight to nine times chance at both -- and collapses
at 24x24. Cells here are queries carrying coordinates normalised by the board
and tiles are keys, so one model runs at every size and the small boards are
pretraining rather than a separate experiment.

THE GATE. Evaluation feeds PURE NOISE as the picture and the true bag as the
conditioning, so `x_t` carries no information about the arrangement and the only
route to a correct cell is the bag. That is G2's causal requirement built into
the measurement instead of bolted beside it: a model that has learned an
unconditional photograph prior scores chance here by construction.

Bars, all held out: chance 1/576 = 0.0017, the MSE regression control 0.0026,
the shipping seam pipeline 0.0251, G3's first useful rung 0.10, its target 0.30.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import CACHE_DIR, CKPT_DIR, GRID as G

N_BOARD = G * G
from field_diffusion import Schedule
from render_assign import SUB, RenderAssign, decode

HELD = 300


def parse_stages(items):
    out = []
    for it in items:
        side, ep = it.split(":")
        out.append((int(side), int(ep)))
    return out


def crop_cells(side, r0, c0):
    """Board-cell indices of an s x s crop, in the crop's own reading order."""
    r = np.arange(side)[:, None] + r0
    c = np.arange(side)[None, :] + c0
    return (r * G + c).reshape(-1)


class Batches:
    """Crops of boards, with the bag shuffled so its order carries nothing."""

    def __init__(self, bag8, bag4, stats, pic, rng, sidx=None, sval=None):
        self.bag8, self.bag4, self.stats, self.pic = bag8, bag4, stats, pic
        self.rng = rng
        self.sidx, self.sval = sidx, sval

    def seam(self, idx, cells_list, perms, side, dev, floor=-12.0):
        """Dense right/down seam logits for a batch, restricted to the crop.

        The candidates are board-level, so a candidate outside the crop is
        dropped; at side 24 nothing is dropped. Rows are indexed by the tile's
        position in the SHUFFLED bag, matching the model's own token order.
        """
        m = side * side
        out = np.full((2, len(idx), m, m), floor, np.float32)
        for n, b in enumerate(idx):
            cells, perm = cells_list[n], perms[n]
            slot = np.full(N_BOARD, -1, np.int64)
            slot[cells] = np.arange(m)          # board cell -> crop cell
            inv = np.empty(m, np.int64)
            inv[perm] = np.arange(m)            # crop cell -> bag slot
            src = cells[perm]                   # board cell of each bag slot
            rows = np.repeat(np.arange(m)[:, None], self.sidx.shape[3], 1)
            for ax in range(2):
                cand = np.asarray(self.sidx[b, ax], np.int64)[src]
                val = np.asarray(self.sval[b, ax], np.float32)[src]
                cell = slot[cand]
                ok = cell >= 0
                col = inv[np.clip(cell, 0, m - 1)]
                out[ax, n][rows[ok], col[ok]] = val[ok]
        return (torch.from_numpy(out[0]).to(dev),
                torch.from_numpy(out[1]).to(dev))

    def draw(self, idx, side, dev, fixed=False):
        """One batch of crops.

        Tile ``j`` of the model's input is board cell ``cells[perm[j]]``, so the
        cell that must receive tile ``j`` is ``perm[j]`` and the target for cell
        ``k`` is the inverse permutation. Both are returned: the inverse scores
        placement, the forward one says which true cell each assignment landed
        on, which is what adjacency counts.
        """
        view, stat, cells4, maps, tgt, fwd = [], [], [], [], [], []
        e_r, e_d, cell_list = [], [], []
        for b in idx:
            rng = (np.random.default_rng(1000 + int(b)) if fixed else self.rng)
            if fixed:
                r0 = c0 = (G - side) // 2
            else:
                r0 = int(rng.integers(0, G - side + 1))
                c0 = int(rng.integers(0, G - side + 1))
            cells = crop_cells(side, r0, c0)
            cell_list.append(cells)
            perm = rng.permutation(side * side)
            view.append(self.bag8[b][cells[perm]])
            stat.append(self.stats[b][cells[perm]])
            cells4.append(self.bag4[b][cells[perm]].reshape(side * side, -1))
            inv = np.empty(side * side, np.int64)
            inv[perm] = np.arange(side * side)
            tgt.append(inv)
            fwd.append(perm)
            # the true right/down partner of every TILE, as a tile index, with
            # -100 where the crop has no such neighbour
            er = np.full(side * side, -100, np.int64)
            ed = np.full(side * side, -100, np.int64)
            ok = (perm % side) < side - 1
            er[ok] = inv[perm[ok] + 1]
            ok = (perm // side) < side - 1
            ed[ok] = inv[perm[ok] + side]
            e_r.append(er)
            e_d.append(ed)
            maps.append(self.pic[b][r0 * SUB:(r0 + side) * SUB,
                                    c0 * SUB:(c0 + side) * SUB])

        def f(arr):
            return torch.from_numpy(np.asarray(arr)).to(dev, torch.float32)

        def i64(arr):
            return torch.from_numpy(np.asarray(arr)).to(dev)

        return (f(view), f(stat), f(cells4) / 127.5 - 1.0,
                f(maps) / 127.5 - 1.0, i64(tgt), np.asarray(fwd),
                i64(e_r), i64(e_d), np.asarray(cell_list))


def adjacency(order, side):
    """Fraction of realised true neighbour relations, the project's second read."""
    grid = np.asarray(order).reshape(side, side)
    ok = tot = 0
    for dy, dx in ((0, 1), (1, 0)):
        a = grid[:side - dy, :side - dx]
        b = grid[dy:, dx:]
        ok += int(((b - a) == (dy * side + dx)).sum())
        tot += a.size
    return ok / max(tot, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="field_cache.npz")
    ap.add_argument("--stages", nargs="+", default=["6:6", "12:8", "24:30"],
                    help="side:epochs, in order")
    ap.add_argument("--batch", type=int, default=8, help="at side 24")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--bag-layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--sink-iters", type=int, default=10)
    ap.add_argument("--perm-weight", type=float, default=1.0)
    ap.add_argument("--edge-weight", type=float, default=1.0,
                    help="supervision on the tile-to-tile relation "
                         "head; 0 reduces the model to absolute only")
    ap.add_argument("--assign-rounds", type=int, default=2,
                    help="times the current assignment is fed back "
                         "as context inside one forward pass")
    ap.add_argument("--rel-rounds", type=int, default=2,
                    help="rounds of neighbour message passing into "
                         "the cell logits; 0 is the baseline")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--high-noise", type=float, default=0.7,
                    help="fraction of samples drawn at the top of the schedule, "
                         "where x_t cannot reveal the arrangement and the bag is "
                         "the only route -- G2's failure was a model reading the "
                         "picture instead of the bag")
    ap.add_argument("--eval-boards", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--eval-steps", type=int, default=16,
                    help="denoising rounds at evaluation; one "
                         "round is the non-iterative control")
    ap.add_argument("--train-boards", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--msg-damp", type=float, default=0.5)
    ap.add_argument("--grad-rounds", type=int, default=2)
    ap.add_argument("--rel-gain-init", type=float, default=0.0,
                    help="weight of the neighbour messages at "
                         "initialisation. Zero starts the model at "
                         "the absolute-only baseline, which is "
                         "right when the relation head is learned "
                         "from a pooled view and wrong when it is "
                         "frozen seam evidence already known to "
                         "decode to adjacency 0.205 (M475)")
    ap.add_argument("--seam-cache", default="",
                    help="verify_top5_v2; when set the relation "
                         "head is BASED on frozen seam evidence "
                         "and only learns a residual on it")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--out", default="render_assign.pt")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    z = np.load(Path(CACHE_DIR) / a.cache)
    bag8, bag4, stats, pic = z["bag8"], z["bag4"], z["stats"], z["pic"]
    n = len(pic)
    n_train = n - HELD
    if a.train_boards:
        n_train = min(n_train, a.train_boards)
    tr = np.arange(n_train)
    ev = np.arange(n - HELD, n)[:a.eval_boards]
    sidx = sval = None
    if a.seam_cache:
        sd = Path(CACHE_DIR) / a.seam_cache
        sidx = np.load(sd / "idx.npy", mmap_mode="r")
        sval = np.load(sd / "val.npy", mmap_mode="r")
        names_a = np.load(Path(CACHE_DIR) / a.cache)["names"]
        names_b = np.load(sd / "names.npy")
        if not (names_a[:len(names_b)] == names_b).all():
            raise SystemExit("the two caches are not in the same board order")
        print("relation head based on frozen seam evidence", flush=True)
    data = Batches(bag8, bag4, stats, pic, rng, sidx, sval)

    dev = (("cuda" if torch.cuda.is_available() else "cpu")
           if a.device == "auto" else a.device)
    model = RenderAssign(a.d, a.bag_layers, a.blocks, a.heads).to(dev)
    if a.resume:
        model.load_state_dict(torch.load(Path(CKPT_DIR) / a.resume,
                                         map_location=dev)["model"])
        print(f"resumed from {a.resume}", flush=True)
    with torch.no_grad():
        model.rel_gain.fill_(a.rel_gain_init)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters "
          f"on {dev}; {len(tr)} train boards, {len(ev)} held out", flush=True)
    sched = Schedule(a.steps, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler(dev)
    stages = parse_stages(a.stages)

    def batch_for(side):
        return int(min(64, max(2, a.batch * (G // max(side, 1)) ** 2)))

    @torch.no_grad()
    def evaluate(side=G, steps=None):
        """Pure noise as the picture: only the bag can put a tile in its cell.

        The picture is denoised over several rounds rather than read once. Each
        round re-renders the current estimate from the bag and hands it back as
        context, which is the same self-conditioning loop the recurrent discrete
        assembler runs over its feature canvas -- and a single forward pass at
        the top of the schedule measures the model with that loop switched off.
        Both are reported so the value of the iteration is visible.
        """
        steps = a.eval_steps if steps is None else steps
        model.eval()
        one, itr, ad = [], [], []
        for i in ev:
            v, s_, cel, _m, tgt, fwd, _er, _ed, cl = data.draw(
                [i], side, dev, fixed=True)
            sm = (data.seam(np.array([i]), cl, fwd, side, dev)
                  if a.seam_cache else None)
            truth = tgt[0].cpu().numpy()
            gen = torch.Generator(device=dev).manual_seed(a.seed + int(i))
            x = torch.randn(1, side * SUB, side * SUB, 3, device=dev,
                            generator=gen)
            ts = torch.linspace(sched.T - 1, 0, steps).long().to(dev)
            for k, t in enumerate(ts):
                x0, logp, _e = model(x, t.expand(1), v, s_, cel, side,
                                     a.sink_iters,
                                     rel_rounds=a.rel_rounds,
                                     assign_rounds=a.assign_rounds,
                                     seam=sm, damp=a.msg_damp,
                                     grad_rounds=a.grad_rounds)
                if k == 0:
                    one.append(float((decode(logp[0]) == truth).mean()))
                if k + 1 < len(ts):
                    ab, an = sched.abar[t], sched.abar[ts[k + 1]]
                    e = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
                    x = an.sqrt() * x0 + (1 - an).sqrt() * e
            got = decode(logp[0])
            itr.append(float((got == truth).mean()))
            ad.append(adjacency(fwd[0][got], side))
        model.train()
        return (float(np.mean(itr)), float(np.mean(ad)), float(np.mean(one)))

    p0, a0, o0 = evaluate()
    print(f"[init] placed {p0:.4f} (one pass {o0:.4f})  adjacency {a0:.4f}  "
          f"(chance {1/(G*G):.4f}, regression control 0.0026, "
          f"seam pipeline 0.0251)", flush=True)
    best = p0
    ep_all = 0
    for side, epochs in stages:
        bs = batch_for(side)
        print(f"--- stage side {side}, batch {bs}, {epochs} epochs "
              f"(chance {1/(side*side):.4f})", flush=True)
        for _ in range(epochs):
            perm = rng.permutation(tr)
            nb = len(perm) // bs
            if a.max_batches:
                nb = min(nb, a.max_batches)
            run, t0 = [], time.time()
            for k in range(nb):
                idx = perm[k * bs:(k + 1) * bs]
                v, s, cel, m, tgt, _fwd, er, ed, cl = data.draw(
                    idx, side, dev)
                sm = (data.seam(idx, cl, _fwd, side, dev)
                      if a.seam_cache else None)
                # high noise most of the time: the arrangement must come from
                # the bag, not be read off a lightly corrupted picture
                u = torch.rand(len(idx), device=dev)
                hi = u < a.high_noise
                t = torch.where(
                    hi,
                    (sched.T - 1 - (torch.rand(len(idx), device=dev)
                                    * 0.15 * sched.T)).long(),
                    (torch.rand(len(idx), device=dev) * sched.T).long()
                ).clamp(0, sched.T - 1)
                eps = torch.randn_like(m)
                xt = (sched.sqrt_abar[t][:, None, None, None] * m
                      + sched.sqrt_one_minus[t][:, None, None, None] * eps)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(dev, dtype=torch.float16):
                    x0, logp, (e_rl, e_dl) = model(
                        xt, t, v, s, cel, side, a.sink_iters,
                        rel_rounds=a.rel_rounds,
                        assign_rounds=a.assign_rounds, seam=sm,
                        damp=a.msg_damp, grad_rounds=a.grad_rounds)
                    pix = F.mse_loss(x0, m)
                    ce = F.nll_loss(logp.reshape(-1, logp.shape[-1]),
                                    tgt.reshape(-1))
                    edge = 0.5 * (
                        F.cross_entropy(e_rl.reshape(-1, e_rl.shape[-1]),
                                        er.reshape(-1), ignore_index=-100)
                        + F.cross_entropy(e_dl.reshape(-1, e_dl.shape[-1]),
                                          ed.reshape(-1), ignore_index=-100))
                    loss = pix + a.perm_weight * ce + a.edge_weight * edge
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                run.append((float(pix.detach()), float(ce.detach()),
                            float(edge.detach())))
            r = np.mean(run, axis=0)
            msg = (f"[side {side} ep {ep_all}] pixel {r[0]:.4f} "
                   f"perm {r[1]:.3f} edge {r[2]:.3f} "
                   f"{(time.time()-t0)/60:.1f} min")
            if ep_all % a.eval_every == 0:
                p, ad, o = evaluate()
                msg += f"  placed24 {p:.4f} (1pass {o:.4f}) adj {ad:.4f}"
                if side != G:
                    # the stage's own size says whether it is learning at all;
                    # 24x24 says whether that transfers, and only it gates
                    ps, ads, os_ = evaluate(side)
                    msg += (f"  [side {side}: {ps:.4f} (1pass {os_:.4f}) "
                            f"adj {ads:.4f}]")
                ck = {"model": model.state_dict(),
                      "args": {k: getattr(a, k) for k in
                               ("d", "blocks", "bag_layers", "heads", "steps",
                                "sink_iters")},
                      "placed": p}
                torch.save(ck, Path(CKPT_DIR) / a.out)
                if p > best:
                    best = p
                    torch.save(ck, Path(CKPT_DIR) / (a.out[:-3] + "_best.pt"))
                    msg += "  *"
            print(msg, flush=True)
            ep_all += 1
    print(f"best held-out placement at 24x24: {best:.4f}  "
          f"(chance 0.0017, regression 0.0026, seam pipeline 0.0251, "
          f"G3 rung 0.10)")


if __name__ == "__main__":
    main()
