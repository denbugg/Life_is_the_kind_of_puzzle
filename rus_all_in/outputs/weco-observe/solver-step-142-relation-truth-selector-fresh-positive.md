# Weco Observe — solver step 142

Parent: confirmed selective+unique-fullres six-arm fusion, step `102`.

The frozen local32+held32 relation classifier was evaluated once on the
already-opened, historically model-selection-exposed fresh32 development
panel.  Control `355.62500/0.93750` pairs/exact became
`358.78125/0.81250`: pair delta **`+3.15625`**, source-bootstrap CI95
`[+0.18750,+6.34453]`, W/T/L `10/16/6`; exact delta `-0.125`.  Edge ROC-AUC
was `0.9691`.

The signed development gate (`pairs >= +0.5`, `exact >= -1`) passed.  This is
not formal confirmation.  A new source16 x draw2 roster and complete exclusion
digest were signed before any new target-free generation or scoring for Weco
step `143`.

Artifact: `outputs/taska-relation-truth-selector/fixed-v1/report.json`.
