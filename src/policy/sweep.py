"""Phase 5 — threshold selection by net rupees, never F1.

sweep_bands: grids T_HIGH, and for each T_HIGH grids T_LOW <= T_HIGH,
keeping the best T_LOW per T_HIGH. That collapses to a single curve
(best net vs T_HIGH) for the centrepiece chart, with the true 2-D
optimum, the naive 0.5 cutoff, and the F1-optimal cutoff all
identifiable as points on or under it.

NO ML in this module — it operates on already-scored (y_true, scores).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.policy.costs import CostModel, DEFAULT_COSTS
from src.policy.evaluate import net_rupees

DEFAULT_GRID = np.round(np.linspace(0.01, 0.99, 99), 4)


def sweep_bands(y_true, scores, costs: CostModel = DEFAULT_COSTS, grid: np.ndarray = DEFAULT_GRID) -> pd.DataFrame:
    rows = []
    for t_high in grid:
        best = None
        for t_low in grid[grid <= t_high]:
            m = net_rupees(y_true, scores, t_low, t_high, costs)
            if best is None or m["net"] > best["net"]:
                best = {"t_low": t_low, "t_high": t_high, **m}
        rows.append(best)
    return pd.DataFrame(rows)


def global_optimum(sweep_df: pd.DataFrame) -> pd.Series:
    return sweep_df.loc[sweep_df["net"].idxmax()]


def f1_optimal_threshold(y_true, scores, grid: np.ndarray = DEFAULT_GRID) -> float:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        preds = (scores >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def net_at_single_threshold(y_true, scores, t: float, costs: CostModel = DEFAULT_COSTS) -> dict:
    """Evaluate a naive single-cutoff policy (no step-up band): t_low = t_high = t."""
    return net_rupees(y_true, scores, t_low=t, t_high=t, costs=costs)


def sensitivity_sweep(y_true, scores, base_costs: CostModel = DEFAULT_COSTS, perturbation: float = 0.30,
                       grid: np.ndarray = DEFAULT_GRID) -> pd.DataFrame:
    """Perturb each named base cost input by +/-perturbation, re-run the
    full 2-D sweep, and record where the optimum lands. If the optimum
    barely moves across all of these, that's the finding: the
    recommendation survives being wrong about the inputs."""
    base_inputs = [
        "avg_fraud_amount", "dispute_fee", "handling",
        "avg_legit_margin", "support_contact", "p_churn_wrong_block", "customer_ltv",
        "p_abandon_stepup", "p_stepup_stops_fraud", "review_cost", "retry_rate",
    ]
    rows = []
    for param in base_inputs:
        base_value = getattr(base_costs, param)
        for direction, factor in [("low", 1 - perturbation), ("high", 1 + perturbation)]:
            perturbed_costs = base_costs.perturbed(**{param: base_value * factor})
            sweep_df = sweep_bands(y_true, scores, perturbed_costs, grid)
            opt = global_optimum(sweep_df)
            rows.append({
                "param": param, "direction": direction, "perturbed_value": base_value * factor,
                "opt_t_low": opt["t_low"], "opt_t_high": opt["t_high"], "opt_net": opt["net"],
            })
    return pd.DataFrame(rows)
