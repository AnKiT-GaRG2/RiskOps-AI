"""Feature attributions — INTERNAL ANALYST ONLY (non-negotiable #7).
Nothing under src/serving/ or a customer-facing template may import
this module; customer_view() doesn't even accept a parameter this data
could travel through.

Uses LightGBM's built-in TreeSHAP (pred_contrib=True) — exact, not an
approximation, and free from an already-trained booster.
"""
import numpy as np
import pandas as pd

from src.model.features import FEATURE_COLS


def compute_contributions(model, df: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame same-indexed as df, one column per feature,
    holding that feature's SHAP contribution to the raw model output
    for that row (last column, 'expected_value', is the base value)."""
    booster = model.booster_
    contrib = booster.predict(df[FEATURE_COLS], pred_contrib=True)
    cols = FEATURE_COLS + ["expected_value"]
    return pd.DataFrame(contrib, columns=cols, index=df.index)


def top_k_attributions(df: pd.DataFrame, contrib_df: pd.DataFrame, row_idx, k: int = 3) -> list:
    row_values = df.loc[row_idx, FEATURE_COLS]
    row_contrib = contrib_df.loc[row_idx, FEATURE_COLS]
    order = row_contrib.abs().sort_values(ascending=False).index[:k]
    return [
        {"name": feat, "value": _to_native(row_values[feat]), "contribution": float(row_contrib[feat])}
        for feat in order
    ]


def _to_native(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if pd.isna(value):
        return None
    return value
