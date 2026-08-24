#!/bin/bash
# One external-CLI (e.g. codex) mutation in the shared AIDE ledger. The SECOND engine.
# Usage: cli_fleet_mutant.sh <task-id> <gen> <m> "<hypothesis>"
# Journals to the SAME ledger.jsonl the Claude workflow uses, node prefix cx_ (or_ for OpenRouter).
set -u
WD="__WORKSPACE__/task$1"; TID="$1"; GEN="$2"; M="$3"; IDEA="$4"
PY="__GRADER_PARITY_PYTHON__"           # score in the grader-parity env, never a drifting local one
CAND="cand_cx_g${GEN}m${M}.py"
BEST=$(cat "$WD/best_cost.txt" 2>/dev/null || echo 999999999)
PROMPT="ONE mutation golfing artifact for task $TID. Budget: implement ONE idea, eval <=3 times, persist, stop.
WORKSPACE $WD  PYTHON $PY  PARENT $WD/solution.py  best=$BEST  HYPOTHESIS: $IDEA
1. read $WD/instructions.md (metric model + the ground-truth spec — read it).
2. MD=/tmp/cx_${TID}_g${GEN}m${M}; mkdir -p \$MD; cp $WD/baseline.* $WD/evaluate.py $WD/baseline_freshfail.txt \$MD/ 2>/dev/null; cp $WD/solution.py \$MD/solution.py
3. rewrite \$MD/solution.py for the hypothesis (respect all hard constraints in instructions.md).
4. $PY \$MD/evaluate.py  (max 3 evals)
5. ALWAYS: cp \$MD/solution.py $WD/$CAND
6. append ONE line to $WD/ledger.jsonl: {\"node\":\"cx_g${GEN}m${M}\",\"tid\":$TID,\"gen\":$GEN,\"status\":\"<WIN|CORRECT_NOT_CHEAPER|LESS_GENERAL|WRONG|CRASH>\",\"cost\":<int|null>,\"cand\":\"$CAND\",\"note\":\"<one line>\"}
7. if WIN: $PY $WD/$CAND --out $WD/best_cx_g${GEN}m${M}.onnx"
cd "$WD" && codex e "$PROMPT" --yolo --skip-git-repo-check > "$WD/cx_g${GEN}m${M}.log" 2>&1
# safety-net: if the CLI wrote a candidate but forgot the ledger line, score it ourselves and journal it
if [ -f "$WD/$CAND" ] && ! grep -qa "cx_g${GEN}m${M}" "$WD/ledger.jsonl"; then
  "$PY" "__SCORE_AND_LEDGER_HELPER__" "$WD" "$CAND" "$TID" "cx_g${GEN}m${M}"
fi
