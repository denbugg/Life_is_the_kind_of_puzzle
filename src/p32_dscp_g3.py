"""P32 DSCP-24 G3: frozen checkpoint on the locked FIT-selection sources only."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from p32_dscp import DSCP,bad
from p32_dscp_g2 import cache_features,load_label,evaluate
import p13_component_pose as p13

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,default=Path(r'E:\pazzle_data\train\inputs'));p.add_argument('--labels',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache'));p.add_argument('--manifest',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json'));p.add_argument('--work',type=Path,default=Path(r'E:\pazzle_work\pazzle_fixed_orientation_20260813\P32_dscp'));a=p.parse_args();a.cache=a.work/'feature_cache';bad(a.inputs,a.labels,a.manifest,a.work);train,_=p13.source_lists(a.manifest);sources=sorted(train)[96:128]
 if len(sources)!=32:raise RuntimeError('locked selection split unavailable')
 ck=torch.load(a.work/'p32_g2_checkpoint.pt',map_location='cpu',weights_only=False);m=DSCP();m.load_state_dict(ck['state_dict']);m.eval();lab={n:load_label(a.labels,n) for n in sources};feat=cache_features(a,sources);top,placement,invalid=evaluate(m,feat,lab,sources);rep={'experiment':'P32_DSCP24','gate':'G3','checkpoint_epochs':ck['fit']['epochs_run'],'selection_top20':top,'selection_placement':placement,'invalid':invalid,'labels_used':True,'targets_opened':False,'p8_imported':False,'passes_G3':bool(top>=0.05 and placement>=0.005 and invalid==0)};(a.work/'p32_g3_report.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep),flush=True)
 if not rep['passes_G3']:raise RuntimeError('P32 G3 rejected')
if __name__=='__main__':main()
