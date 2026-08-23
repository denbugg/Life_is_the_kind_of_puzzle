import inspect
import json
from pathlib import Path
import torch
import direct_pose

report_path = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R7_full_contrastive_retriever\g1_capacity\r7_g1_report.json")
report = json.loads(report_path.read_text(encoding="utf-8"))
print("R7_GATE", report["gate"])
print("R7_PARAMETERS", report["parameters"])
print("R7_BEST_CAL_R20", report["best_cal_recall_at_20"])
for row in report["history"]:
    cal = row["cal"]
    print("R7_ROW", row["step"], row["train_loss"], cal["recall_at_1"], cal["recall_at_5"], cal["recall_at_20"], cal["recall_at_96"], cal["recall_at_128"])
print("DIRECT_POSE_PUBLIC", [name for name in dir(direct_pose) if not name.startswith("_")])
for name in ["DirectionalSiamese", "direct_scores", "score_pairs"]:
    if hasattr(direct_pose, name):
        obj = getattr(direct_pose, name)
        print("DIRECT_POSE", name, inspect.signature(obj))
        if inspect.isclass(obj):
            print(inspect.getsource(obj.forward))
checkpoint = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R2L_siamese\best.pt")
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
print("R2L_CKPT_KEYS", sorted(payload.keys()))
print("R2L_CKPT_ARGS", payload.get("args"))
print("R2L_CKPT_METRICS", payload.get("metrics"))
print("R2L_STATE_SAMPLE", list(payload["model"].keys())[:25])
