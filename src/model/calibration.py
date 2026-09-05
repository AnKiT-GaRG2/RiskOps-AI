"""Reliability diagram, Brier score, expected calibration error.

The isotonic calibrator itself must be fit on the calibration slice
only — enforced by the caller (src/model/train.py), not here.
"""
import numpy as np
import pandas as pd


def reliability_table(y_true, probs, n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bin_edges, right=True) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        mean_pred = float(probs[mask].mean()) if count else float((bin_edges[b] + bin_edges[b + 1]) / 2)
        observed = float(y_true[mask].mean()) if count else np.nan
        rows.append({
            "bin_lower": bin_edges[b], "bin_upper": bin_edges[b + 1],
            "mean_predicted": mean_pred, "observed_frequency": observed, "count": count,
        })
    return pd.DataFrame(rows)


def brier_score(y_true, probs) -> float:
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    return float(np.mean((probs - y_true) ** 2))


def expected_calibration_error(y_true, probs, n_bins: int = 10) -> float:
    table = reliability_table(y_true, probs, n_bins)
    n = table["count"].sum()
    valid = table.dropna(subset=["observed_frequency"])
    ece = np.sum(valid["count"] / n * np.abs(valid["observed_frequency"] - valid["mean_predicted"]))
    return float(ece)
