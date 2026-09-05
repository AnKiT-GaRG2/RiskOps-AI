"""Entity-key derivation for card and device disjointness.

IEEE-CIS has no literal `card_id` or `device_id` column, and this is not
a cosmetic gap: `card1..card6` are BIN/network/issuer/type attributes
shared by thousands of distinct physical cards (card6 alone is just
"credit"/"debit"). Measured on this dataset, the full card1-6 combo has
~14.9k unique values and 94% of all rows belong to an entity that
recurs across more than one temporal slice — enforcing disjointness on
it would collapse calibration and test to a few hundred rows each,
well under the positives-in-the-thousands floor Phase 1 calls for.

We instead use the proxy the IEEE-CIS/Kaggle community converged on for
approximating a persistent client: `card1 + addr1 + D1n`, where
`D1n = floor(TransactionDT / 86400) - D1` is an approximate account-open
day that stays constant for the same underlying account even as `D1`
("days since account open") grows with later transactions. This raises
distinct entities to ~218k and cuts multi-slice-touching rows to ~41% —
still substantial (this dataset is genuinely dominated by repeat
clients), but a defensible proxy rather than an accident of low
cardinality. It is a heuristic, not ground truth — documented here so
the assumption is visible, not buried in a merge.

Device proxy: DeviceType + DeviceInfo + id_31 (browser) + id_33 (screen
resolution) — a fingerprint, present only for the ~24% of rows with
identity data. Rows without any identity signal get a null device
entity and are exempt from device-disjointness (there is nothing to
collide on).
"""
import numpy as np
import pandas as pd

DEVICE_COLS = ["DeviceType", "DeviceInfo", "id_31", "id_33"]


def _row_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    parts = [df[c].astype("string").fillna("NA") for c in present]
    key = parts[0]
    for p in parts[1:]:
        key = key + "|" + p
    return key


def add_entity_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    d1n = (df["TransactionDT"] // 86400 - df["D1"]).round(0)
    card1 = df["card1"].astype("string").fillna("NA")
    addr1 = df["addr1"].astype("string").fillna("NA")
    df["card_entity"] = card1 + "|" + addr1 + "|" + d1n.astype("string").fillna("NA")

    device_cols_present = [c for c in DEVICE_COLS if c in df.columns]
    has_device = df[device_cols_present].notna().any(axis=1)
    device_key = _row_key(df, DEVICE_COLS)
    df["device_entity"] = np.where(has_device, device_key, pd.NA)
    return df
