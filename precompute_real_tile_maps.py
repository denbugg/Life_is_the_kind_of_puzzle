"""Precompute target-position -> real noisy tile mappings for all train pairs."""
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

GRID, TILE, N = 24, 20, 576
DATA_ROOT = Path(os.getenv("DATA_ROOT", "data/real/train"))
OUT = Path(os.getenv("MAP_FILE", "real_tile_maps.npz"))
WORKERS = int(os.getenv("WORKERS", "8"))


def split(path):
    x=np.asarray(Image.open(path).convert("RGB").resize((480,480)),np.uint8)
    return x.reshape(GRID,TILE,GRID,TILE,3).transpose(0,2,1,3,4).reshape(N,TILE,TILE,3)


def features(tiles):
    x=tiles.astype(np.float32)/255
    low=x.reshape(N,5,4,5,4,3).mean((2,4)); gray=low.mean(3)
    gray=(gray-gray.mean((1,2),keepdims=True))/(gray.std((1,2),keepdims=True)+1e-5)
    dx=np.diff(gray,axis=2,append=gray[:,:,-1:]); dy=np.diff(gray,axis=1,append=gray[:,-1:,:])
    color=x.mean((1,2))
    f=np.concatenate([gray.reshape(N,-1),.35*dx.reshape(N,-1),.35*dy.reshape(N,-1),.2*color],1)
    return f/(np.linalg.norm(f,axis=1,keepdims=True)+1e-6)


def process(stem):
    noisy=split(DATA_ROOT/"inputs"/(stem+".png")); clean=split(DATA_ROOT/"targets"/(stem+".png"))
    cost=2-2*np.clip(features(noisy)@features(clean).T,-1,1)
    noisy_i,pos_i=linear_sum_assignment(cost)
    inverse=np.empty(N,np.uint16); inverse[pos_i]=noisy_i
    return stem,inverse,float(cost[noisy_i,pos_i].mean())


def main():
    stems=sorted(p.stem for p in (DATA_ROOT/"targets").glob("*.png") if (DATA_ROOT/"inputs"/p.name).exists())
    maps=np.empty((len(stems),N),np.uint16); costs=np.empty(len(stems),np.float32)
    with Pool(WORKERS) as pool:
        for i,(stem,mapping,cost) in enumerate(pool.imap(process,stems,chunksize=4)):
            maps[i]=mapping; costs[i]=cost
            if (i+1)%250==0: print(json.dumps({"done":i+1,"total":len(stems),"mean_cost":float(costs[:i+1].mean())}),flush=True)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT,stems=np.asarray(stems),maps=maps,costs=costs)
    print(json.dumps({"pairs":len(stems),"mean_cost":float(costs.mean()),"p95_cost":float(np.quantile(costs,.95)),"output":str(OUT)}),flush=True)


if __name__=="__main__": main()
