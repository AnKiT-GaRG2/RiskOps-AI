"""Phase 9 — synthetic attack burst. Demo only. NEVER used for reported
metrics: nothing in eval/ imports this module, and every row this
module produces carries is_synthetic=True so it can't be silently
folded into a metric anywhere downstream.

Pattern: one shared device across many distinct cards, a velocity
spike, anomalous amounts, and a geo mismatch (large dist1) — the
textbook "card testing from one compromised device" shape.
"""
import numpy as np
import pandas as pd

SHARED_DEVICE_INFO = "SYN-DEVICE-BURST"
SHARED_DEVICE_TYPE = "mobile"
SHARED_BROWSER = "synthetic-injected"
SHARED_SCREEN = "0x0"


def generate_attack_burst(start_dt: int, n_cards: int = 12, seconds_between: int = 15,
                           seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_cards):
        dt = start_dt + i * seconds_between
        rows.append({
            "TransactionID": f"SYN-{start_dt}-{i}",
            "TransactionDT": dt,
            "TransactionAmt": float(rng.uniform(2500, 9000)),  # anomalously high vs typical order
            "ProductCD": "W",
            "card1": int(rng.integers(100000, 999999)),  # a fresh, never-seen card each time
            "card2": float(rng.integers(100, 599)),
            "card3": 150.0,
            "card4": "visa",
            "card5": 226.0,
            "card6": "credit",
            "addr1": float(rng.integers(100, 199)),
            "addr2": 87.0,
            "dist1": float(rng.uniform(3000, 9000)),  # geo mismatch: billing/shipping far apart
            "dist2": None,
            "D1": 0.0,
            "P_emaildomain": "gmail.com",
            "R_emaildomain": None,
            "DeviceType": SHARED_DEVICE_TYPE,
            "DeviceInfo": SHARED_DEVICE_INFO,
            "id_30": "Android",
            "id_31": SHARED_BROWSER,
            "id_33": SHARED_SCREEN,
            "isFraud": 1,          # ground truth for the injector's own bookkeeping only
            "is_synthetic": True,   # never eligible for reported metrics
        })
    return pd.DataFrame(rows)


def label_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    """Defensive helper: guarantee the flag exists and is never silently
    False for anything produced by this module."""
    df = df.copy()
    df["is_synthetic"] = True
    return df
