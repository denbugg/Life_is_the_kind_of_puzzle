# Weco Observe — solver step 141

Parent: confirmed selective+unique-fullres six-arm fusion, step `102`.

One fixed `HistGradientBoostingClassifier` was trained on every realised seam
of all six independently post-tail-polished local32 layouts.  Features are
target-free local raw ranks/margins, focal/provenance, six-arm relation support,
and fixed arm/control identity.  A layout score is the sum of its 1,104
predicted truth probabilities; the selector returns exactly one whole arm.

On source-disjoint held32, control `345.31250/1.90625` pairs/exact became
`350.28125/1.53125`: pair delta **`+4.96875`**, source-bootstrap CI95
`[+1.65625,+8.78125]`, W/T/L `6/26/0`; exact delta `-0.375`.  Edge ROC-AUC was
`0.9650`.  The preregistered held signal gate passed, so the same parameters
were fitted once on local32+held32 before the already-opened fresh development
panel.

Artifact: `outputs/taska-relation-truth-selector/fixed-v1/report.json`.

