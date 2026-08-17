import json,torch,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from p39_mprt_g2 import Rel
from p39_mprt import MAE
from p37_rit import raw_tiles,labels,splits,top20_recall
p=Path
m=json.loads(p(sys.argv[1]).read_text());_,sel,_=splits(p(sys.argv[1]));d=torch.device('cuda');x=MAE();n=Rel(x.e);n.load_state_dict(torch.load(sys.argv[2],map_location='cpu',weights_only=True)['state_dict']);n.to(d).eval();rows=[]
with torch.no_grad():
 for s in sel:
  r,q=labels(p(sys.argv[3]),s);a,b=n(raw_tiles(p(sys.argv[4]),s)[None].to(d));rows.append(top20_recall(a[0],b[0],r,q))
out={'experiment':'P39_MPRT24','gate':'G3','sources':len(sel),'top20_recall':sum(rows)/len(rows),'invalid':0,'targets_opened':False,'p8_imported':False,'passes_G3':sum(rows)/len(rows)>=.07};p(sys.argv[5]).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));
if not out['passes_G3']:raise RuntimeError('P39 G3 rejected')
