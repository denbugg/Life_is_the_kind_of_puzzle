"""Selector-aligned BasinCycle Stage-B successor (unsigned, data blocked).

This module keeps the reviewed 6x6 short-cycle proposal generator and strict
candidate materialisation from :mod:`aiijc_puzzle.basincycle_stage_b`.  It
replaces the failed quantile/risk inference conjunction with two quantities
that are supervised and consumed directly:

* a candidate-vs-KEEP safe-improvement logit; and
* a candidate-vs-KEEP true-pair gain score.

KEEP has analytical gain zero and is never passed through either prediction
head.  The module contains no data loader, runner, checkpoint, source roster,
DEV/test path, threshold sweep, or pixel-output path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from aiijc_puzzle.basincycle_stage_b import (
    GRID_SIZE,
    ProposalBank,
    _strict_layouts,
    materialize_candidate_layouts,
    model_static_ledger,
)
from aiijc_puzzle.basincycle_stage_b_mps_reductions_v3 import (
    BasinCycleStageBMPSReductionsV3,
)
from aiijc_puzzle.basincycle_stage_b_mps_transfer_v2 import (
    build_target_free_proposal_bank_mps_transfer_v2,
)

PAIR_GAIN_INDEX = 0
COMPOSITIONAL_FEATURE_NAMES = (
    "expected_changed_pair_gain",
    "probability_any_removed_contact_is_true",
)


@dataclass(frozen=True)
class SelectorAlignedOutput:
    """Target-free model output with KEEP fixed analytically at index zero."""

    pair_logits: torch.Tensor
    boundary_prediction: torch.Tensor
    proposal_bank: ProposalBank
    candidate_layouts: torch.Tensor
    safe_improvement_logits: torch.Tensor
    pair_gain_scores: torch.Tensor
    compositional_features: torch.Tensor


@dataclass(frozen=True)
class SelectorAlignedLabels:
    """FIT-only labels attached after target-free proposal identities freeze."""

    safe_improvement: torch.Tensor
    pair_delta: torch.Tensor
    loses_existing_true_pair: torch.Tensor
    edge_targets: torch.Tensor
    clean_boundary_targets: torch.Tensor


def changed_contact_mask(candidate_layouts: torch.Tensor, *, grid_size: int) -> torch.Tensor:
    """Return bonds changed relative to proposal-zero KEEP, shape ``B x P x E``."""

    if candidate_layouts.ndim != 3:
        raise ValueError("candidate_layouts must have shape B x P x N")
    batch_size, proposal_count, tile_count = candidate_layouts.shape
    if tile_count != grid_size * grid_size:
        raise ValueError("candidate layout size differs from grid")
    board = candidate_layouts.reshape(batch_size, proposal_count, grid_size, grid_size)
    keep = board[:, :1]
    horizontal = (
        (board[:, :, :, :-1] != keep[:, :, :, :-1]) | (board[:, :, :, 1:] != keep[:, :, :, 1:])
    ).reshape(batch_size, proposal_count, -1)
    vertical = (
        (board[:, :, :-1, :] != keep[:, :, :-1, :]) | (board[:, :, 1:, :] != keep[:, :, 1:, :])
    ).reshape(batch_size, proposal_count, -1)
    return torch.cat((horizontal, vertical), dim=-1)


def compositional_changed_edge_features(
    pair_logits: torch.Tensor,
    candidate_layouts: torch.Tensor,
    *,
    grid_size: int,
) -> torch.Tensor:
    """Compose target-free candidate-vs-KEEP evidence over changed bonds.

    Directional logits are converted to row probabilities using the same
    candidate set used by directional cross-entropy.  For every changed board
    bond, the expected gain is ``p(new)-p(old)``.  The second feature is the
    independent-edge probability that at least one removed incumbent contact
    is true.  These are model-visible features, not oracle labels.
    """

    if pair_logits.ndim != 4 or pair_logits.shape[1] != 2:
        raise ValueError("pair_logits must have shape B x 2 x N x N")
    batch_size, proposal_count, tile_count = candidate_layouts.shape
    if pair_logits.shape != (batch_size, 2, tile_count, tile_count):
        raise ValueError("pair logits and candidate layouts are misaligned")
    if tile_count != grid_size * grid_size:
        raise ValueError("candidate layout size differs from grid")

    probabilities = torch.softmax(pair_logits, dim=-1)
    board = candidate_layouts.reshape(batch_size, proposal_count, grid_size, grid_size)
    batch = torch.arange(batch_size, device=pair_logits.device)[:, None, None, None]
    horizontal = probabilities[
        batch,
        0,
        board[:, :, :, :-1],
        board[:, :, :, 1:],
    ].reshape(batch_size, proposal_count, -1)
    vertical = probabilities[
        batch,
        1,
        board[:, :, :-1, :],
        board[:, :, 1:, :],
    ].reshape(batch_size, proposal_count, -1)
    contact_probability = torch.cat((horizontal, vertical), dim=-1)
    keep_probability = contact_probability[:, :1]
    changed = changed_contact_mask(candidate_layouts, grid_size=grid_size)

    expected_gain = ((contact_probability - keep_probability) * changed).sum(dim=-1)
    epsilon = torch.finfo(contact_probability.dtype).eps
    removed_probability = keep_probability.clamp(min=0.0, max=1.0 - epsilon)
    log_no_loss = (torch.log1p(-removed_probability) * changed).sum(dim=-1)
    loss_risk = -torch.expm1(log_no_loss)
    result = torch.stack((expected_gain, loss_risk), dim=-1)
    if not torch.equal(result[:, 0], torch.zeros_like(result[:, 0])):
        raise AssertionError("KEEP compositional delta/risk must be exact zero")
    return result


class BasinCycleSelectorAlignedV2(BasinCycleStageBMPSReductionsV3):
    """6x6 short-cycle model whose inference rule uses its trained heads."""

    def __init__(
        self,
        *,
        grid_size: int = GRID_SIZE,
        feature_channels: int = 48,
        retrieval_dim: int = 64,
        state_dim: int = 96,
        encoder_blocks: int = 4,
        state_blocks: int = 3,
        proposal_top_k: int = 8,
        proposal_candidate_cap: int = 6,
        proposal_seed_count: int = 18,
        max_cycle_length: int = 3,
        proposal_cap: int = 256,
    ) -> None:
        super().__init__(
            grid_size=grid_size,
            feature_channels=feature_channels,
            retrieval_dim=retrieval_dim,
            state_dim=state_dim,
            encoder_blocks=encoder_blocks,
            state_blocks=state_blocks,
            proposal_top_k=proposal_top_k,
            proposal_candidate_cap=proposal_candidate_cap,
            proposal_seed_count=proposal_seed_count,
            max_cycle_length=max_cycle_length,
            proposal_cap=proposal_cap,
        )
        del self.action_head
        action_input_dim = 5 * state_dim + self.scalar_feature_count
        self.selector_head = nn.Sequential(
            nn.Linear(action_input_dim + len(COMPOSITIONAL_FEATURE_NAMES), 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

    def forward(self, tiles: torch.Tensor, layouts: torch.Tensor) -> SelectorAlignedOutput:
        if layouts.shape != (len(tiles), self.tile_count) or not _strict_layouts(
            layouts,
            tile_count=self.tile_count,
        ):
            raise ValueError("layouts must be strict B x 36 permutations")
        preprocessed = self._preprocess(tiles)
        features = self.image_stem(preprocessed)
        for block in self.image_blocks:
            features = block(features)
        sides = self._side_sequences(features)
        pair_logits = self._pair_logits(sides)
        boundary_prediction = self.boundary_head(sides)
        pooled = features.mean(dim=(2, 3)).reshape(len(tiles), self.tile_count, -1)
        tile_embeddings = self.tile_projection(pooled)
        context, slot_embeddings = self._state_context(tile_embeddings, pair_logits, layouts)

        bank = build_target_free_proposal_bank_mps_transfer_v2(
            pair_logits,
            layouts,
            grid_size=self.grid_size,
            top_k=self.proposal_top_k,
            candidate_cap=self.proposal_candidate_cap,
            seed_count=self.proposal_seed_count,
            max_cycle_length=self.max_cycle_length,
            proposal_cap=self.proposal_cap,
        )
        candidates = materialize_candidate_layouts(layouts, bank)
        action_features = self._action_features(
            context,
            slot_embeddings,
            tile_embeddings,
            pair_logits,
            candidates,
            bank,
        )
        compositional = compositional_changed_edge_features(
            pair_logits,
            candidates,
            grid_size=self.grid_size,
        )
        raw = self.selector_head(torch.cat((action_features, compositional), dim=-1))
        safe_logits = raw[..., 0]
        pair_gain = raw[..., 1]

        # KEEP is the analytical comparator, never a learned estimate.  Padding
        # is unselectable and contributes to no loss term.
        safe_logits = safe_logits.masked_fill(~bank.valid, -torch.inf)
        pair_gain = pair_gain.masked_fill(~bank.valid, -torch.inf)
        safe_logits = safe_logits.clone()
        pair_gain = pair_gain.clone()
        safe_logits[:, 0] = -torch.inf
        pair_gain[:, 0] = 0.0
        return SelectorAlignedOutput(
            pair_logits=pair_logits,
            boundary_prediction=boundary_prediction,
            proposal_bank=bank,
            candidate_layouts=candidates,
            safe_improvement_logits=safe_logits,
            pair_gain_scores=pair_gain,
            compositional_features=compositional,
        )


def selector_aligned_static_ledger(
    model: BasinCycleSelectorAlignedV2,
    *,
    batch_size: int,
) -> dict[str, object]:
    """Return the v1-comparable parameter/MAC ledger with the revised head."""

    ledger = model_static_ledger(model, batch_size=batch_size)
    old_action_macs = int(ledger["per_board"]["action_head"])
    input_dim = 5 * model.state_dim + model.scalar_feature_count
    new_action_macs = model.proposal_cap * (
        (input_dim + len(COMPOSITIONAL_FEATURE_NAMES)) * 128 + 128 * 64 + 64 * 2
    )
    per_board = dict(ledger["per_board"])
    per_board["selector_head"] = new_action_macs
    del per_board["action_head"]
    forward = int(ledger["forward_learned_macs_per_board"]) - old_action_macs + new_action_macs
    ledger.update(
        {
            "trainable_parameters": model.trainable_parameter_count(),
            "forward_learned_macs_per_board": forward,
            "forward_learned_macs_per_batch": forward * batch_size,
            "per_board": per_board,
        }
    )
    return ledger


def vectorized_pair_delta_and_loss(
    candidate_layouts: np.ndarray,
    truth_layouts: np.ndarray,
    valid: np.ndarray,
    *,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Attach pair-gain and incumbent-pair-loss labels in one NumPy batch.

    This avoids one Python set construction per proposal.  It is intended for
    CPU label workers after proposal identities and layouts have been frozen.
    Invalid padding receives delta zero and loss false.
    """

    candidates = np.asarray(candidate_layouts)
    truths = np.asarray(truth_layouts)
    valid_values = np.asarray(valid, dtype=bool)
    if candidates.ndim != 3:
        raise ValueError("candidate_layouts must have shape B x P x N")
    batch_size, proposal_count, tile_count = candidates.shape
    if truths.shape != (batch_size, tile_count):
        raise ValueError("truth_layouts must have shape B x N")
    if valid_values.shape != (batch_size, proposal_count):
        raise ValueError("valid mask differs from candidate layouts")
    if tile_count != grid_size * grid_size:
        raise ValueError("candidate layout size differs from grid")
    expected = np.arange(tile_count)
    if not np.all(np.sort(candidates, axis=-1) == expected):
        raise ValueError("all candidates must be strict permutations")
    if not np.all(np.sort(truths, axis=-1) == expected):
        raise ValueError("all truths must be strict permutations")
    if not np.all(valid_values[:, 0]):
        raise ValueError("proposal zero must be valid KEEP")

    truth_board = truths.reshape(batch_size, grid_size, grid_size)
    right_of = np.full((batch_size, tile_count), -1, dtype=np.int64)
    down_of = np.full((batch_size, tile_count), -1, dtype=np.int64)
    batch = np.arange(batch_size)[:, None]
    right_of[batch, truth_board[:, :, :-1].reshape(batch_size, -1)] = truth_board[:, :, 1:].reshape(
        batch_size, -1
    )
    down_of[batch, truth_board[:, :-1, :].reshape(batch_size, -1)] = truth_board[:, 1:, :].reshape(
        batch_size, -1
    )

    board = candidates.reshape(batch_size, proposal_count, grid_size, grid_size)
    horizontal_source = board[:, :, :, :-1].reshape(batch_size, proposal_count, -1)
    horizontal_target = board[:, :, :, 1:].reshape(batch_size, proposal_count, -1)
    vertical_source = board[:, :, :-1, :].reshape(batch_size, proposal_count, -1)
    vertical_target = board[:, :, 1:, :].reshape(batch_size, proposal_count, -1)
    batch3 = np.arange(batch_size)[:, None, None]
    horizontal_true = right_of[batch3, horizontal_source] == horizontal_target
    vertical_true = down_of[batch3, vertical_source] == vertical_target

    # Pair preservation is about directed pair identity, not the raster bond
    # where the pair happened to occur.  A closed cycle may move an intact
    # true pair to another board bond.  Index realised pairs by (axis, source
    # tile identity), which uniquely determines the true target tile.
    realised = np.zeros((batch_size, proposal_count, 2, tile_count), dtype=bool)
    proposal3 = np.arange(proposal_count)[None, :, None]
    realised[batch3, proposal3, 0, horizontal_source] = horizontal_true
    realised[batch3, proposal3, 1, vertical_source] = vertical_true
    baseline = realised[:, :1]
    delta = realised.sum(axis=(2, 3), dtype=np.int16) - baseline.sum(
        axis=(2, 3),
        dtype=np.int16,
    )
    loses = np.any(baseline & ~realised, axis=(2, 3))
    delta = np.where(valid_values, delta, 0).astype(np.int16, copy=False)
    loses = np.where(valid_values, loses, False)
    if not np.all(delta[:, 0] == 0) or np.any(loses[:, 0]):
        raise AssertionError("KEEP labels must be analytical zero/no-loss")
    return delta, loses


def selector_aligned_targets(
    pair_delta: np.ndarray,
    loses_existing_true_pair: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Return the exact action target consumed by the abstention head."""

    delta = np.asarray(pair_delta)
    loses = np.asarray(loses_existing_true_pair, dtype=bool)
    valid_values = np.asarray(valid, dtype=bool)
    if delta.shape != loses.shape or delta.shape != valid_values.shape or delta.ndim != 2:
        raise ValueError("pair delta, loss and valid tensors must share B x P shape")
    safe = valid_values & (delta > 0) & ~loses
    safe[:, 0] = False
    return safe


def _masked_state_means(
    values: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = mask.sum(dim=1)
    means = (values * mask).sum(dim=1) / counts.clamp_min(1)
    return means, counts > 0


def _fixed_topk_mask(scores: torch.Tensor, mask: torch.Tensor, *, k: int) -> torch.Tensor:
    if k <= 0:
        raise ValueError("hard-negative k must be positive")
    take = min(k, scores.shape[1])
    ranked = scores.detach().masked_fill(~mask, -torch.inf)
    values, indices = torch.topk(ranked, k=take, dim=1, largest=True, sorted=True)
    selected = torch.zeros_like(mask)
    selected.scatter_(1, indices, torch.isfinite(values))
    return selected


def selector_aligned_loss(
    output: SelectorAlignedOutput,
    labels: SelectorAlignedLabels,
    *,
    hard_negatives_per_stratum: int = 32,
    listwise_weight: float = 0.75,
    gain_weight: float = 0.25,
    edge_weight: float = 0.25,
    restore_weight: float = 0.15,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Balanced safe-BCE plus the exact inference gain ranking objective."""

    valid = output.proposal_bank.valid
    nonkeep = valid.clone()
    nonkeep[:, 0] = False
    if labels.pair_delta.shape != valid.shape:
        raise ValueError("pair delta labels must match proposal bank")
    if labels.loses_existing_true_pair.shape != valid.shape:
        raise ValueError("pair-loss labels must match proposal bank")
    if labels.safe_improvement.shape != valid.shape:
        raise ValueError("safe labels must match proposal bank")
    expected_safe = nonkeep & (labels.pair_delta > 0) & ~labels.loses_existing_true_pair
    if not torch.equal(labels.safe_improvement, expected_safe):
        raise ValueError("safe labels must equal pair_delta>0 AND no incumbent-pair loss")
    if torch.any(labels.safe_improvement & ~valid):
        raise ValueError("a safe-improvement label points to padding")

    logits = output.safe_improvement_logits
    finite_logits = torch.where(nonkeep, logits, torch.zeros_like(logits))
    bce = functional.binary_cross_entropy_with_logits(
        finite_logits,
        labels.safe_improvement.to(finite_logits.dtype),
        reduction="none",
    )
    positive = labels.safe_improvement
    no_gain_no_loss = nonkeep & (labels.pair_delta <= 0) & ~labels.loses_existing_true_pair
    gain_but_loss = nonkeep & (labels.pair_delta > 0) & labels.loses_existing_true_pair
    no_gain_and_loss = nonkeep & (labels.pair_delta <= 0) & labels.loses_existing_true_pair
    negative_strata = (no_gain_no_loss, gain_but_loss, no_gain_and_loss)
    hard_masks = tuple(
        _fixed_topk_mask(
            output.compositional_features[..., PAIR_GAIN_INDEX],
            mask,
            k=hard_negatives_per_stratum,
        )
        for mask in negative_strata
    )
    group_means: list[torch.Tensor] = []
    for mask in (positive, *hard_masks):
        means, present = _masked_state_means(bce, mask)
        if torch.any(present):
            group_means.append(means[present].mean())
    if not group_means:
        raise ValueError("batch contains no valid non-KEEP training actions")
    safe_bce = torch.stack(group_means).mean()

    # KEEP is a fixed zero score.  If a safe action exists, all safe actions
    # attaining the largest true pair gain form the listwise target set;
    # otherwise KEEP is the sole target.  Unsafe candidates remain competitors.
    pair_delta = labels.pair_delta.to(output.pair_gain_scores.dtype)
    safe_gain = pair_delta.masked_fill(~positive, -torch.inf)
    best_gain = safe_gain.max(dim=1).values
    has_safe = positive.any(dim=1)
    best_actions = positive & (pair_delta == best_gain[:, None])
    best_actions[:, 0] = ~has_safe
    ranking_scores = output.pair_gain_scores.masked_fill(~valid, -torch.inf).clone()
    ranking_scores[:, 0] = 0.0
    log_probabilities = functional.log_softmax(ranking_scores, dim=1)
    listwise = -torch.logsumexp(
        log_probabilities.masked_fill(~best_actions, -torch.inf),
        dim=1,
    ).mean()

    regression_mask = positive.clone()
    for mask in hard_masks:
        regression_mask |= mask
    gain_errors = functional.smooth_l1_loss(
        torch.where(regression_mask, output.pair_gain_scores, torch.zeros_like(pair_delta)),
        torch.where(regression_mask, pair_delta, torch.zeros_like(pair_delta)),
        reduction="none",
        beta=1.0,
    )
    gain_means, gain_present = _masked_state_means(gain_errors, regression_mask)
    if not torch.any(gain_present):
        raise ValueError("batch contains no gain-regression actions")
    gain = gain_means[gain_present].mean()

    if labels.edge_targets.shape != output.pair_logits.shape[:3]:
        raise ValueError("edge targets must have shape B x two axes x N")
    edge = functional.cross_entropy(
        output.pair_logits.reshape(-1, output.pair_logits.shape[-1]),
        labels.edge_targets.reshape(-1),
        ignore_index=-1,
    )
    if labels.clean_boundary_targets.shape != output.boundary_prediction.shape:
        raise ValueError("clean boundary targets differ from prediction shape")
    restore_error = output.boundary_prediction - labels.clean_boundary_targets
    restore = torch.sqrt(restore_error.square() + 1e-6).mean()

    total = (
        safe_bce
        + listwise_weight * listwise
        + gain_weight * gain
        + edge_weight * edge
        + restore_weight * restore
    )
    return total, {
        "safe_bce": safe_bce,
        "listwise": listwise,
        "gain": gain,
        "edge": edge,
        "restore": restore,
    }


def select_selector_aligned_action(
    output: SelectorAlignedOutput,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the trained safe/gain heads or analytically abstain to KEEP.

    The two zero thresholds are architectural, not fitted on a scored panel:
    balanced BCE uses zero safe logit as its decision boundary and the gain
    head is trained against a fixed zero-gain KEEP comparator.
    """

    bank = output.proposal_bank
    batch_size = len(output.candidate_layouts)
    # Stage whole batches once.  Calling ``.item()`` for every MPS proposal
    # would introduce hundreds of device synchronisations per board.
    scores = (
        torch.stack(
            (output.safe_improvement_logits, output.pair_gain_scores),
            dim=-1,
        )
        .detach()
        .to(device="cpu")
        .numpy()
    )
    valid = bank.valid.detach().cpu().numpy()
    lengths = bank.lengths.detach().cpu().numpy()
    positions = bank.positions.detach().cpu().numpy()
    selected_values = np.zeros(batch_size, dtype=np.int64)
    for batch_index in range(batch_size):
        eligible = np.flatnonzero(
            valid[batch_index]
            & (np.arange(valid.shape[1]) > 0)
            & (scores[batch_index, :, 0] >= 0.0)
            & (scores[batch_index, :, 1] > 0.0)
        )
        if len(eligible):
            selected_values[batch_index] = min(
                (int(index) for index in eligible),
                key=lambda index: (
                    -float(scores[batch_index, index, 1]),
                    -float(scores[batch_index, index, 0]),
                    int(lengths[batch_index, index]),
                    tuple(int(value) for value in positions[batch_index, index]),
                ),
            )
    selected = torch.from_numpy(selected_values).to(output.pair_gain_scores.device)
    batch = torch.arange(batch_size, device=selected.device)
    layouts = output.candidate_layouts[batch, selected]
    if not _strict_layouts(layouts, tile_count=layouts.shape[-1]):
        raise AssertionError("selected action violated strict permutation legality")
    return selected, layouts


__all__ = [
    "COMPOSITIONAL_FEATURE_NAMES",
    "PAIR_GAIN_INDEX",
    "BasinCycleSelectorAlignedV2",
    "SelectorAlignedLabels",
    "SelectorAlignedOutput",
    "changed_contact_mask",
    "compositional_changed_edge_features",
    "select_selector_aligned_action",
    "selector_aligned_static_ledger",
    "selector_aligned_loss",
    "selector_aligned_targets",
    "vectorized_pair_delta_and_loss",
]
