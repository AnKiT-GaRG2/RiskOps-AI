"""Phase 10 — selective labels.

Every HOLD (and, arguably, every STEPUP that ends in abandonment)
produces no outcome label: we never learn whether it would have been
fraud. Training data is censored by the policy's own past decisions —
the model progressively goes blind in exactly the region it blocks,
while measured precision on the shrinking "let-through" population can
improve even as real performance decays.

Mitigation: a randomized exploration slice — let a small, fixed
fraction of would-be-holds through anyway, purely to keep recovering
ground truth in the blind region. This module only computes the
budget's rupee cost and expected label recovery; it does not implement
the randomization itself (that belongs in the live-scoring path, out of
scope for this offline evaluation harness).
"""
from dataclasses import dataclass

from src.policy.costs import CostModel, DEFAULT_COSTS

DEFAULT_EXPLORATION_RATE = 0.02


@dataclass
class ExplorationBudget:
    hold_volume: int
    exploration_rate: float
    fraud_rate_in_holds: float
    expected_cost: float
    labels_recovered_per_period: float


def compute_exploration_budget(
    hold_volume: int,
    fraud_rate_in_holds: float,
    exploration_rate: float = DEFAULT_EXPLORATION_RATE,
    costs: CostModel = DEFAULT_COSTS,
) -> ExplorationBudget:
    """hold_volume: number of transactions the policy sent to HOLD over
    the period being reported (e.g. the test-slice window).
    fraud_rate_in_holds: observed isFraud rate among held transactions
    on the (labelled) test slice — the best available estimate of what
    the exploration slice would let through.
    """
    exploration_volume = hold_volume * exploration_rate
    expected_cost = exploration_volume * fraud_rate_in_holds * costs.fn_cost
    labels_recovered = exploration_volume  # every explored txn eventually resolves to a real label
    return ExplorationBudget(
        hold_volume=hold_volume,
        exploration_rate=exploration_rate,
        fraud_rate_in_holds=fraud_rate_in_holds,
        expected_cost=float(expected_cost),
        labels_recovered_per_period=float(labels_recovered),
    )
