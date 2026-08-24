// AIDE-tree Workflow template — fill the >>> FILL <<< sections, then pass as the Workflow
// tool's `script`. Encodes the battle-tested search: per target, sequential generations of
// 2 parallel timeboxed mutation agents with diverse ideas + failure-history feedback,
// then a finalizer that re-verifies the winner. Targets run concurrently (pipeline).
export const meta = {
  name: 'aide-tree',
  description: 'AIDE tree-search: generations of timeboxed mutation subagents vs a scored eval',
  phases: [{ title: 'Tree', detail: 'per-target generations of mutation agents' }],
}

// >>> FILL: interpreter/runtime used to run builders and evals <<<
const PY = "/usr/bin/env python3"
// >>> FILL: absolute path holding work/<target>/ workspaces <<<
const W = "/abs/path/to/work"
// >>> FILL: one entry per optimization target: id + baseline metric (lower=better here).
// If your metric is maximize, flip the comparisons marked [GOAL] below. <<<
const TASKS = [
  {tid: "targetA", base: 100000},
  {tid: "targetB", base: 50000},
]
const GENS = 2          // 2-3; returns diminish after that — prefer fresh waves
const BUDGET = 12       // hard tool-call budget per mutant (see SKILL.md for why)

const MUT_SCHEMA = {
  type: "object",
  properties: {
    tid: {type: "string"}, gen: {type: "integer"}, idea: {type: "string"},
    status: {type: "string", enum: ["WIN","CORRECT_NOT_CHEAPER","LESS_GENERAL","WRONG","CRASH"]},
    cost: {type: ["number","null"]}, cand: {type: "string"}, note: {type: "string"},
  },
  required: ["tid","gen","idea","status","cost","cand","note"],
  additionalProperties: false,
}
const FIN_SCHEMA = {
  type: "object",
  properties: { tid: {type: "string"}, final_cost: {type: "number"}, improved: {type: "boolean"}, note: {type: "string"} },
  required: ["tid","final_cost","improved","note"], additionalProperties: false,
}

function mutPrompt(t, gen, m, idea, parentCand, bestCost, history) {
  const wd = `${W}/${t.tid}`
  const cand = `cand_g${gen}_m${m}.py`
  return `ONE MUTATION in an AIDE tree-search optimizing target ${t.tid}.

HARD BUDGET: at most ${BUDGET} tool calls. You are ONE fast mutation, not a researcher.
Implement YOUR idea, evaluate at most 3 times, persist, return. If the budget runs out:
persist your best attempt with an honest status and return. Speed over polish.

WORKSPACE: ${wd}   RUNTIME: ${PY}
PARENT code (start from it): ${wd}/${parentCand}
Best metric so far: ${bestCost}. Baseline: ${t.base}.
WIN = evaluate prints the fully-valid status AND metric < ${bestCost}.   // [GOAL]
${history ? `TREE HISTORY (do not repeat failures):\n${history}\n` : ""}
YOUR ASSIGNED IDEA: ${idea}

PROTOCOL (exactly this — siblings run in parallel, avoid races):
1. ONE read of ${wd}/instructions.md (metric model, hard constraints, spec path — read the
   spec/reference, it defines the required behavior).
2. MD=/tmp/${t.tid}_g${gen}m${m}; mkdir -p $MD; cp ${wd}/baseline.* ${wd}/evaluate.py $MD/ 2>/dev/null; cp ${wd}/${parentCand} $MD/solution.py
3. Rewrite $MD/solution.py implementing YOUR idea. Respect every hard constraint in instructions.md.
4. ${PY} $MD/evaluate.py — at most 3 evals with quick fixes between.
5. ALWAYS persist, even on failure: cp $MD/solution.py ${wd}/${cand}
6. Append ONE single-line JSON to the ledger:
   echo '{"node":"g${gen}m${m}","tid":"${t.tid}","gen":${gen},"idea":"<5 words>","status":"<WIN|CORRECT_NOT_CHEAPER|LESS_GENERAL|WRONG|CRASH>","cost":<number|null>,"cand":"${cand}","note":"<one line: what you did / why it failed>"}' >> ${wd}/ledger.jsonl

Return the JSON: tid="${t.tid}", gen=${gen}, idea, status, cost (metric when fully valid,
else null), cand="${cand}", note.`
}

async function runTree(t) {
  const wd = `${W}/${t.tid}`
  let bestCost = t.base
  let bestCand = "solution.py"
  let history = ""
  let attempts = 0
  for (let gen = 1; gen <= GENS; gen++) {
    // >>> FILL: adapt the idea menus to your domain — keep the ideas ORTHOGONAL <<<
    const ideas = gen === 1 ? [
      "REBUILD FROM SPEC: ignore the baseline's structure; implement the ground-truth rule/reference directly with a minimal program.",
      "SURGERY ON THE BASELINE: inspect the artifact, find its most expensive components, and attack them in place while keeping behavior identical.",
    ] : [
      `REFINE the current best (${bestCand}, metric ${bestCost}): micro-optimizations, dead-code removal, cheaper datatypes/representations.`,
      `NEW ANGLE from history: the boldest structural idea not yet tried (read what failed and why before choosing).`,
    ]
    const recs = await parallel(ideas.map((idea, m) => () =>
      agent(mutPrompt(t, gen, m, idea, bestCand, bestCost, history),
            {label: `${t.tid} g${gen}m${m}`, phase: 'Tree', schema: MUT_SCHEMA, effort: 'medium'})))
    for (const r of recs.filter(Boolean)) {
      attempts++
      history += `[${r.idea}] -> ${r.status}${r.cost ? ` cost=${r.cost}` : ``}; ${r.note}\n`
      if (r.status === "WIN" && r.cost && r.cost < bestCost) { bestCost = r.cost; bestCand = r.cand }  // [GOAL]
    }
    log(`${t.tid} gen${gen}: best ${bestCost} (base ${t.base})`)
    if (history.length > 2000) history = history.slice(-2000)
  }
  if (bestCand === "solution.py") return { tid: t.tid, base: t.base, final: bestCost, improved: false, attempts }
  // Finalizer: never trust a mutant's own report — rebuild and re-verify, fall back on failure.
  const fin = await agent(
`Finalize target ${t.tid} (max 8 tool calls). Workspace ${wd}, runtime ${PY}.
Winning candidate: ${wd}/${bestCand} (metric ${bestCost}, baseline ${t.base}).
1. cp ${wd}/${bestCand} ${wd}/solution.py
2. Build the final artifact from solution.py (see instructions.md for the build command).
3. ${PY} ${wd}/evaluate.py must print the fully-valid status with metric <= ${bestCost}.
   If it does NOT: restore the baseline loader into solution.py, rebuild, report improved=false.
Return JSON: tid="${t.tid}", final_cost, improved (final_cost beats ${t.base} AND confirmed), note.`,
    {label: `${t.tid} finalize`, phase: 'Tree', schema: FIN_SCHEMA, effort: 'medium'})
  return { tid: t.tid, base: t.base, final: fin?.final_cost ?? bestCost, improved: !!(fin?.improved), attempts }
}

phase('Tree')
log(`AIDE tree: ${TASKS.length} targets x ${GENS} generations x 2 mutations`)
const results = await pipeline(TASKS, runTree)
const ok = results.filter(Boolean)
const wins = ok.filter(r => r.improved)
log(`done: ${wins.length} improved of ${ok.length}`)
return { wins, all: ok }
