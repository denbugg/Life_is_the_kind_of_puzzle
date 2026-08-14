"""P8 FIT-only cache-schema probe; no training/evaluation targets are read."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args()
    paths=sorted((a.work/'cache').glob('*.npz'))
    if not paths: raise FileNotFoundError('no P3 FIT cache files')
    with np.load(paths[0],allow_pickle=False) as x:
        print({'cache':str(paths[0]),'fields':{k:{'shape':list(x[k].shape),'dtype':str(x[k].dtype)} for k in x.files}},flush=True)
if __name__=='__main__':main()
