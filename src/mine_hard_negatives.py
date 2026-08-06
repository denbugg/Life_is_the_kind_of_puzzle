"""Mine hard false neighbours for PairwiseNet training."""
import os
import argparse
import numpy as np
from config import GRID, NFRAG, TRAIN_TGT, CACHE_DIR
from imgio import load, to_frags, train_val_split
from distort import distort_frags
from datasets import real_recon
from pipeline import load_pair
from solve import pairwise_scores_full
from match_preprocess import load_match_denoiser, preprocess_frags_np

DEV = "cuda"


def top_false(M, offset, K):
    out = -np.ones((NFRAG, K), np.int16)
    for a in range(NFRAG):
        b = a + offset
        if b < 0 or b >= NFRAG:
            continue
        if offset == 1 and a % GRID == GRID - 1:
            continue
        if offset == GRID and a // GRID == GRID - 1:
            continue
        row = M[a].copy()
        row[a] = -1e30
        row[b] = -1e30
        kk = min(K, NFRAG - 2)
        idx = np.argpartition(-row, kk)[:kk]
        idx = idx[np.argsort(-row[idx])]
        out[a, :len(idx)] = idx.astype(np.int16)
    return out


def make_frags(name, real_prob, rng):
    if real_prob > 0 and rng.random() < real_prob:
        rr = real_recon(name)
        if rr is not None:
            return rr[0]
    clean = load(os.path.join(TRAIN_TGT, name))
    return distort_frags(to_frags(clean), rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--split", choices=("train", "val"), default="train")
    ap.add_argument("--real_prob", type=float, default=0.6)
    ap.add_argument("--preprocess", choices=("raw", "norm", "denoise", "denoise_norm"), default="raw")
    ap.add_argument("--denoise_tag", default="matchden")
    ap.add_argument("--pair_tag", default="pair")
    ap.add_argument("--bs_score", type=int, default=4096)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pair, pck = load_pair(args.pair_tag)
    if pair is None:
        raise FileNotFoundError("no pair checkpoint found")
    print(f"pair step={pck.get('step')} val={pck.get('val')}", flush=True)

    denoiser = None
    if args.preprocess in ("denoise", "denoise_norm"):
        denoiser, _ = load_match_denoiser(args.denoise_tag, device=DEV)
        if denoiser is None:
            raise FileNotFoundError("no matching denoiser checkpoint found")

    trn, val = train_val_split()
    names = (trn if args.split == "train" else val)[:args.n]
    out = args.out or os.path.join(CACHE_DIR, f"hardneg_{args.split}_{args.preprocess}_K{args.K}.npz")
    rng = np.random.default_rng(1234)
    right = np.empty((len(names), NFRAG, args.K), np.int16)
    down = np.empty((len(names), NFRAG, args.K), np.int16)

    for i, nm in enumerate(names):
        frags = make_frags(nm, args.real_prob, rng)
        sf = preprocess_frags_np(frags, args.preprocess, denoiser, DEV)
        R, D = pairwise_scores_full(pair, sf, DEV, bs=args.bs_score)
        right[i] = top_false(R, 1, args.K)
        down[i] = top_false(D, GRID, args.K)
        if (i + 1) % 25 == 0 or i == len(names) - 1:
            print(f"mined {i+1}/{len(names)} -> {out}", flush=True)
    np.savez_compressed(out, names=np.array(names), right=right, down=down,
                        K=args.K, preprocess=args.preprocess)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()



