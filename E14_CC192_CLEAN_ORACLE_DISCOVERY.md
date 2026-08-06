# E14: fixed CC192 clean-oracle discovery

E14 is a separate CPU-only diagnostic on the already-open E12 calibration IDs
10–17. It does not modify E11, E12, E13, Rank96 production, or generic solver
files.

## Fixed arms

- `RR96` is replayed exactly from the raw E12 candidates/scores with buddies
  `max_edges=96`, `min_margin=0`, `repair_passes=0`. All eight board hashes,
  solved-canvas hashes, mean solve SSIM `0.094607964147414`, and mean final SSIM
  `0.15930445310452002` must reproduce.
- `CC192` uses only the existing byte-pinned E12 clean-oracle candidates and
  clean-oracle scores. It uses buddies `max_edges=192`, `min_margin=0`,
  `repair_passes=0`.

There is no sweep, alternate budget, rank/energy transplant, model scoring,
GPU path, rotation, or reflection.

## Structural reproducibility gate

The exact chain is:

```text
E12 cc_candidates/cc_scores
  -> E12 dense_from_graph
  -> solve_buddies._candidate_edges(..., max_edges=192, min_margin=0)
  -> solve_buddies.build_buddies_components(..., 192, 0)
```

The selected prefix must contain exactly 192 edges for every scene. These are
192 attempted mutual-argmax claims; rejected or redundant Builder edges are
not backfilled.

- `selected_edge_precision`: coordinate-safe true directional edges divided by
  the 192 selected claims. It is not accepted-edge or component-internal
  precision.
- `component_coverage`: unique tiles present in the resulting production
  components divided by 576. It measures reach, not component purity.

End-to-end metrics run only if both inclusive thresholds pass across the eight
scenes:

- mean selected-edge precision `>= 0.95`;
- mean component coverage `>= 0.45`.

## End-to-end decision

CC192 assembles only original corrupted upright tiles and applies OpenCV NLM
with `h=10` exactly once per scene. It is a go only if every comparison with
RR96 passes:

- mean solve-only SSIM delta `>= +0.010`;
- mean final SSIM delta `>= +0.015`;
- strict final-SSIM wins `>= 6/8`;
- worst final-SSIM delta `>= -0.020`.

CC192 is an oracle diagnostic, not a deployable or submission arm.

## Files and execution boundary

- Fixed core: `src/e14_cc192_oracle.py`
- Diagnostic CLI: `src/eval_e14_cc192_discovery.py`
- Tests: `tests/test_e14_cc192_discovery.py`
- Default report:
  `E:/pazzle_work/cc192_oracle_e14/cc192_clean_oracle_discovery_v1.json`

The report is updated atomically after each completed scene. Score caches and
outputs are restricted to `E:`. E12 report, cache, checkpoint, current source,
Python, NumPy, scikit-image, OpenCV build, and Torch provenance are pinned.

Run later only when intentionally authorized:

```powershell
python src/eval_e14_cc192_discovery.py
```

