from __future__ import annotations
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--report", type=Path, required=True)
args = parser.parse_args()
report = json.loads(args.report.read_text(encoding="utf-8"))
if report.get("schema") != "pazzle-e26-fp16-preflight-v1":
    raise SystemExit("unexpected preflight schema")
if report.get("mode") != "FP16_PREFLIGHT_ONLY":
    raise SystemExit("unexpected preflight mode")
if report.get("status") != "PASS":
    raise SystemExit("preflight did not pass")
if report.get("not_production_equivalent") is not True:
    raise SystemExit("preflight report must mark non-production equivalence")
if report.get("precision") not in {"float16_autocast_with_gradscaler", "bfloat16_autocast"}:
    raise SystemExit("unexpected precision")
print("e26 fp16 preflight verifier PASS")
