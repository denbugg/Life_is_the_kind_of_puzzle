# Weco Observe — solver step 143

Parent: confirmed selective+unique-fullres six-arm fusion, step `102`.

Before any new inference or scoring, an exact source16 x draw2 organizer-train
roster and an exclusion snapshot covering 2,054 explicit prior sources were
SHA-signed.  The frozen local32+held32 relation classifier, all six independent
post-tail layouts, 6×1,104 relation features, expected scores and final whole-arm
choice were frozen before exact references were reconstructed.

Formal result:

- pairs `332.21875 -> 338.06250`, delta **`+5.84375`**, source CI95
  **`[+3.00000,+9.12578]`**;
- case W/T/L `13/19/0`, source W/T/L `11/5/0`;
- exact `1.21875 -> 1.06250`, delta `-0.15625`;
- strict original upright permutations `32/32`.

The preregistered gate (pair mean at least `+1`, source-CI lower at least zero,
exact at least `-1`, all strict) passed.  Competition test, production,
postprocessing and submission were untouched.

Artifact: `outputs/taska-relation-truth-selector/formal-confirmation-v1/report.json`,
SHA-256 `d260872251077e1515251b6c7afc316af25df75045c8119112dff4f36c68ea23`.
