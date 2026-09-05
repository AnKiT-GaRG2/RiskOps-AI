"""Phase 7 spike monitor. A monitoring heuristic over rolling windows of
scorer output — NOT a separate model, and it must be labelled that way
on screen. We deliberately do not report precision/recall for spike
detection: a demo produces a handful of spike events, and a metric
computed on a handful of events is unfalsifiable. Declining to report a
meaningless metric is itself the point of this track.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

BUCKET_SECONDS = 5 * 60
BASELINE_WINDOW_BUCKETS = 7 * 24 * 60 // 5  # 7 days of 5-minute buckets
ALERT_Z_THRESHOLD = 3.0


def bucket_scored_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """df needs: TransactionDT, calibrated_score, band, distinct_cards_per_device."""
    d = df.copy()
    d["bucket"] = d["TransactionDT"] // BUCKET_SECONDS
    grouped = d.groupby("bucket").agg(
        mean_score=("calibrated_score", "mean"),
        hold_rate=("band", lambda s: (s == "HOLD").mean()),
        n_txns=("calibrated_score", "size"),
        max_distinct_cards_per_device=("distinct_cards_per_device", "max"),
    ).reset_index()
    grouped["bucket_start_dt"] = grouped["bucket"] * BUCKET_SECONDS
    return grouped


@dataclass
class SpikeAlert:
    bucket: int
    metric: str
    value: float
    baseline_mean: float
    baseline_std: float
    z_score: float


def detect_spikes(bucketed: pd.DataFrame, metrics=("mean_score", "hold_rate", "max_distinct_cards_per_device"),
                   z_threshold: float = ALERT_Z_THRESHOLD) -> list:
    alerts = []
    b = bucketed.sort_values("bucket").reset_index(drop=True)
    for metric in metrics:
        series = b[metric].astype(float)
        baseline_mean = series.rolling(BASELINE_WINDOW_BUCKETS, min_periods=10).mean().shift(1)
        baseline_std = series.rolling(BASELINE_WINDOW_BUCKETS, min_periods=10).std(ddof=0).shift(1)
        z = (series - baseline_mean) / baseline_std.replace(0, np.nan)
        for idx in z[z.abs() > z_threshold].index:
            alerts.append(SpikeAlert(
                bucket=int(b.loc[idx, "bucket"]), metric=metric, value=float(series[idx]),
                baseline_mean=float(baseline_mean[idx]), baseline_std=float(baseline_std[idx]),
                z_score=float(z[idx]),
            ))
    return alerts
