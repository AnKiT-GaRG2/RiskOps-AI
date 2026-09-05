"""Behavioural aggregates. Every function here takes an already
as_of-filtered `history` — never the raw table — and must not be called
except from `compute_features` (single-transaction) or
`build_behavioral_matrix` (the vectorized batch builder proven
equivalent to it by eval/test_leakage.py).

`card_entity` here is the customer proxy (see src/data/entities.py).
"velocity"/"distinct devices"/"amount z-score"/"decline rate" below are
all computed against that proxy's history, not a true customer id.

`historical_decline_rate` is a stated approximation: IEEE-CIS carries no
issuer decline codes, so we use the card entity's own historical
isFraud rate as the nearest available proxy. In production this would
leak label latency (chargebacks resolve weeks after the transaction);
here it's still as_of-safe because it only ever reads history strictly
before the scored transaction's own timestamp.
"""
import math
from collections import deque

import pandas as pd

WINDOWS = {
    "velocity_10min": 600, "velocity_1h": 3600, "velocity_24h": 86400, "velocity_7d": 7 * 86400,
}


def compute_behavioral_features(txn: pd.Series, history: pd.DataFrame, as_of) -> dict:
    """Canonical, single-transaction reference implementation. `history`
    must already be filtered to as_of (see src/features/as_of.py)."""
    card_hist = history[history["card_entity"] == txn["card_entity"]]

    device_entity = txn.get("device_entity")
    has_device = pd.notna(device_entity)
    device_hist = history[history["device_entity"] == device_entity] if has_device else history.iloc[0:0]

    out = {}
    for name, window_secs in WINDOWS.items():
        out[name] = int((card_hist["TransactionDT"] > as_of - window_secs).sum())

    out["distinct_devices_per_card"] = int(card_hist["device_entity"].dropna().nunique())
    out["distinct_cards_per_device"] = (
        int(device_hist["card_entity"].nunique()) if has_device else None
    )

    history_count = len(card_hist)
    out["history_count"] = history_count

    amounts = card_hist["TransactionAmt"]
    if history_count >= 2 and amounts.std(ddof=0) > 1e-6:
        out["amount_zscore"] = float((txn["TransactionAmt"] - amounts.mean()) / amounts.std(ddof=0))
    else:
        out["amount_zscore"] = 0.0

    if history_count > 0:
        out["time_since_prev_txn"] = float(as_of - card_hist["TransactionDT"].max())
        out["is_first_txn"] = 0
        out["historical_decline_rate"] = float(card_hist["isFraud"].mean())
    else:
        out["time_since_prev_txn"] = -1.0
        out["is_first_txn"] = 1
        out["historical_decline_rate"] = None

    return out


class _CardState:
    __slots__ = ("timestamps", "amt_sum", "amt_sumsq", "amt_count", "last_ts", "fraud_sum", "devices_seen")

    def __init__(self):
        self.timestamps = deque()
        self.amt_sum = 0.0
        self.amt_sumsq = 0.0
        self.amt_count = 0
        self.last_ts = None
        self.fraud_sum = 0
        self.devices_seen = set()


def build_behavioral_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized-in-spirit (single forward pass, O(n)) batch builder.
    df must be pre-sorted by TransactionDT ascending and carry
    card_entity/device_entity. Ties on TransactionDT are grouped into
    one "tick": every row in a tick is scored off state as of strictly
    before the tick, then the tick's rows are folded into state
    together — so simultaneous transactions never see each other,
    matching the strict `<` as_of contract in as_of.py.
    """
    card_states: dict[str, _CardState] = {}
    device_cards_seen: dict[str, set] = {}

    n = len(df)
    results = {
        "velocity_10min": [0] * n, "velocity_1h": [0] * n, "velocity_24h": [0] * n, "velocity_7d": [0] * n,
        "distinct_devices_per_card": [0] * n, "distinct_cards_per_device": [None] * n,
        "history_count": [0] * n, "amount_zscore": [0.0] * n,
        "time_since_prev_txn": [-1.0] * n, "is_first_txn": [1] * n,
        "historical_decline_rate": [None] * n,
    }

    dts = df["TransactionDT"].to_numpy()
    card_entities = df["card_entity"].to_numpy()
    device_entities = df["device_entity"].to_numpy()
    amounts = df["TransactionAmt"].to_numpy()
    frauds = df["isFraud"].to_numpy()

    i = 0
    while i < n:
        j = i
        tick_dt = dts[i]
        while j < n and dts[j] == tick_dt:
            j += 1
        tick_idx = range(i, j)

        for k in tick_idx:
            ce = card_entities[k]
            de = device_entities[k]
            cs = card_states.get(ce)

            if cs is not None:
                while cs.timestamps and cs.timestamps[0] <= tick_dt - WINDOWS["velocity_7d"]:
                    cs.timestamps.popleft()
                results["velocity_7d"][k] = len(cs.timestamps)
                results["velocity_24h"][k] = sum(1 for t in cs.timestamps if t > tick_dt - WINDOWS["velocity_24h"])
                results["velocity_1h"][k] = sum(1 for t in cs.timestamps if t > tick_dt - WINDOWS["velocity_1h"])
                results["velocity_10min"][k] = sum(1 for t in cs.timestamps if t > tick_dt - WINDOWS["velocity_10min"])
                results["distinct_devices_per_card"][k] = len(cs.devices_seen)
                hc = cs.amt_count
                results["history_count"][k] = hc
                if hc >= 2:
                    mean = cs.amt_sum / hc
                    var = max(cs.amt_sumsq / hc - mean * mean, 0.0)
                    std = math.sqrt(var)
                    results["amount_zscore"][k] = (amounts[k] - mean) / std if std > 1e-6 else 0.0
                results["time_since_prev_txn"][k] = float(tick_dt - cs.last_ts)
                results["is_first_txn"][k] = 0
                results["historical_decline_rate"][k] = cs.fraud_sum / hc if hc > 0 else None

            if pd.notna(de):
                results["distinct_cards_per_device"][k] = len(device_cards_seen.get(de, ()))

        for k in tick_idx:
            ce = card_entities[k]
            de = device_entities[k]
            cs = card_states.setdefault(ce, _CardState())
            cs.timestamps.append(tick_dt)
            cs.amt_sum += amounts[k]
            cs.amt_sumsq += amounts[k] * amounts[k]
            cs.amt_count += 1
            cs.last_ts = tick_dt
            cs.fraud_sum += frauds[k]
            if pd.notna(de):
                cs.devices_seen.add(de)
                device_cards_seen.setdefault(de, set()).add(ce)

        i = j

    return pd.DataFrame(results, index=df.index)
