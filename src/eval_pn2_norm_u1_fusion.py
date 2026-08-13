"""PN2 matched normalized U1 fusion wrapper.

Imports the established U1 fusion evaluator and replaces only its local
PairwiseNet scorer with a per-tile photometrically normalized variant. Candidate
retrieval and DirectPoseNet remain raw, so this is matched to PN1 training.
"""
from __future__ import annotations

from match_preprocess import photometric_normalize_tensor
import eval_u1_union_fusion as base


_original_pair_scorer = base.score_pairwise_directions


def normalized_pair_scorer(models, tiles, candidates, valid, *, pair_batch, device):
    return _original_pair_scorer(
        models,
        photometric_normalize_tensor(tiles),
        candidates,
        valid,
        pair_batch=pair_batch,
        device=device,
    )


base.score_pairwise_directions = normalized_pair_scorer
base.main()
