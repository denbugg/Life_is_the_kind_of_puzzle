"""P34 VCLS-24: vectorized 2x2 consensus-loop contracts (G0)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

N = 576
RIGHT, DOWN, LEFT, UP = 0, 1, 2, 3
OPP = {RIGHT: LEFT, DOWN: UP, LEFT: RIGHT, UP: DOWN}


def witness_masks(cand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return right/down candidates with a closed 2x2 witness.
    cand[d,i,j] is directional candidate adjacency. Operations are boolean-matrix products.
    """
    # right i->j witness i-down->a, a-right->b, j-down->b
    right_paths = (cand[DOWN].astype(np.int16) @ cand[RIGHT].astype(np.int16)) > 0
    right_w = cand[RIGHT] & ((right_paths[:, None, :] & cand[DOWN][None, :, :]).any(axis=2))
    # down i->j witness i-right->a, a-down->b, j-right->b
    down_paths = (cand[RIGHT].astype(np.int16) @ cand[DOWN].astype(np.int16)) > 0
    down_w = cand[DOWN] & ((down_paths[:, None, :] & cand[RIGHT][None, :, :]).any(axis=2))
    return right_w, down_w


def g0(a):
    # Four tiles form 2x2: 0 right 1, 0 down 2, 2 right 3, 1 down 3.
    cand = np.zeros((4, 4, 4), dtype=bool)
    cand[RIGHT,0,1] = cand[DOWN,0,2] = cand[RIGHT,2,3] = cand[DOWN,1,3] = True
    # inverse directions, plus a broken distracting right edge 0->3 without closure.
    cand[LEFT,1,0] = cand[UP,2,0] = cand[LEFT,3,2] = cand[UP,3,1] = True
    cand[RIGHT,0,3] = True
    rw, dw = witness_masks(cand)
    valid = bool(rw[0,1] and dw[0,2])
    reject = bool(not rw[0,3])
    # Exact reciprocal check on selected positive directions.
    reciprocal = bool(cand[LEFT,1,0] and cand[UP,2,0] and cand[LEFT,3,2] and cand[UP,3,1])
    rep = {'experiment':'P34_VCLS24','gate':'G0','right_witnesses':int(rw.sum()),'down_witnesses':int(dw.sum()),'valid_2x2':valid,'rejects_broken':reject,'reciprocal_contract':reciprocal,'passes_G0':bool(valid and reject and reciprocal)}
    return rep


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('g0',),required=True);p.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P34_vcls'));a=p.parse_args();a.work.mkdir(parents=True,exist_ok=True);r=g0(a);(a.work/'p34_g0_report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
    if not r['passes_G0']:raise RuntimeError('P34 G0 rejected')
if __name__=='__main__':main()
