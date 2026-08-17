"""P33 CVA-24 synthetic translation-consistent agglomeration contract."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

class OffsetUF:
 def __init__(self,n):self.p=np.arange(n);self.d=np.zeros((n,2),np.int32)
 def find(self,x):
  if self.p[x]==x:return x,np.zeros(2,np.int32)
  r,up=self.find(int(self.p[x]));self.d[x]+=up;self.p[x]=r;return r,self.d[x].copy()
 def add(self,a,b,delta):
  # Enforce pos[b]-pos[a]=delta; accept only translation-consistent non-overlap merges.
  ra,da=self.find(a);rb,db=self.find(b);delta=np.asarray(delta,np.int32)
  if ra==rb:return bool(np.array_equal(db-da,delta))
  # rb pose relative ra: pos[rb]-pos[ra] = delta+da-db
  rel=delta+da-db;self.p[rb]=ra;self.d[rb]=rel;return True
 def poses(self):
  out=[]
  for i in range(len(self.p)):r,d=self.find(i);out.append((int(r),tuple(map(int,d))))
  return out

def g0(a):
 u=OffsetUF(8);accepted=[];rejected=[]
 # Two 2x2 components plus a contradictory diagonal-like relation.
 edges=[(0,1,(1,0)),(0,2,(0,1)),(1,3,(0,1)),(4,5,(1,0)),(4,6,(0,1)),(5,7,(0,1)),(0,3,(9,9))]
 for x,y,d in edges:
  (accepted if u.add(x,y,d) else rejected).append((x,y,d))
 poses=u.poses();groups={}
 for i,(r,p) in enumerate(poses):groups.setdefault(r,[]).append((i,p))
 overlap=any(len({p for _,p in q})!=len(q) for q in groups.values())
 exact=all(dict(groups[r])[3]==(1,1) for r in groups if 3 in dict(groups[r]))
 return {'experiment':'P33_CVA24','gate':'G0','accepted':len(accepted),'rejected':len(rejected),'components':len(groups),'overlap':overlap,'exact_2x2':exact,'passes_G0':bool(len(rejected)==1 and not overlap and exact)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('g0',),required=True);p.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P33_cva'));a=p.parse_args();a.work.mkdir(parents=True,exist_ok=True);r=g0(a);(a.work/'p33_g0_report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
 if not r['passes_G0']:raise RuntimeError('P33 G0 rejected')
if __name__=='__main__':main()
