# Data

- Source: Kaggle `pasha883/vsos-ai-initiative-pazzle`.
- Remote root and alignment maps already exist under `/home/kva`.
- Training/support scenes: `6700–6727`, `6957–6980` where cached V27 matrices exist.
- Hyperparameter validation: `6981–6988`.
- Final comparison: the same fixed 15-scene V30 evaluation set
  (`6732–6735`, `6989–6999`).
- V31 must not select a hypothesis or parameter from final-set labels. Fresh-seed
  verification uses the same fixed scenes but a different stochastic solver seed;
  results are explicitly development CV, not a new untouched test.
