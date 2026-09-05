"""The as_of regression test. Two independent checks:

1. FUTURE-BLINDNESS: for a sample of transactions, compute_features()
   with the full dataset available must equal compute_features() with
   every row at-or-after the transaction's own timestamp deleted first.
   If these differ, a feature is reading the future.

2. BATCH EQUIVALENCE: the fast vectorized build_feature_matrix() used
   to build the real train/calib/test matrices must agree with the
   safe row-by-row compute_features() reference on the same sample.
   This is what makes the fast path trustworthy.

Run: python -m eval.test_leakage
"""
import math
import sys

import numpy as np
import pandas as pd

from src.data.entities import add_entity_keys
from src.data.split import load_raw
from src.features.batch import build_feature_matrix
from src.features.pipeline import compute_features

SAMPLE_SIZE = 200
RNG_SEED = 42


def _load_sorted_with_entities() -> pd.DataFrame:
    df = load_raw()
    df = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    df = add_entity_keys(df)
    return df


def _is_missing(x) -> bool:
    """None, NaN, and NaT are all the same "missing" for this purpose —
    a static field's None (from a dict default) and a NaN (from a raw
    pandas cell) both mean 'not present', regardless of which of the two
    representations a given code path happened to produce."""
    if isinstance(x, str):
        return False
    try:
        result = pd.isna(x)
        return bool(result) if not hasattr(result, "__len__") else False
    except (TypeError, ValueError):
        return False


def _values_equal(a, b) -> bool:
    a_missing, b_missing = _is_missing(a), _is_missing(b)
    if a_missing or b_missing:
        return a_missing and b_missing
    if isinstance(a, float) or isinstance(b, float):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return a == b
        return math.isclose(fa, fb, rel_tol=1e-6, abs_tol=1e-6)
    return a == b


def check_future_blindness(df: pd.DataFrame, sample_idx: np.ndarray) -> int:
    # Slim once, outside the loop: compute_features only ever needs these
    # columns, and repeatedly filtering the full ~394-column raw table
    # (rather than this 5-column slice) is the dominant cost at 590k rows.
    df_slim = df[["TransactionDT", "TransactionAmt", "isFraud", "card_entity", "device_entity"]]

    mismatches = 0
    for idx in sample_idx:
        txn = df.loc[idx]
        as_of = txn["TransactionDT"]

        truncated_history = df_slim[df_slim["TransactionDT"] < as_of]  # rows at/after deleted upfront

        out_full = compute_features(txn, df_slim, as_of)
        out_truncated = compute_features(txn, truncated_history, as_of)

        for key in out_full:
            if not _values_equal(out_full[key], out_truncated[key]):
                mismatches += 1
                print(f"  [FUTURE LEAK] TransactionID={txn['TransactionID']} field={key} "
                      f"full={out_full[key]!r} truncated={out_truncated[key]!r}")
    return mismatches


def check_batch_equivalence(df: pd.DataFrame, sample_idx: np.ndarray) -> int:
    batch_matrix = build_feature_matrix(df)
    mismatches = 0
    for idx in sample_idx:
        txn = df.loc[idx]
        as_of = txn["TransactionDT"]
        reference = compute_features(txn, df, as_of)
        batch_row = batch_matrix.loc[idx]

        for key, ref_val in reference.items():
            if key not in batch_row.index:
                continue
            batch_val = batch_row[key]
            if isinstance(batch_val, (np.floating, np.integer)):
                batch_val = batch_val.item()
            if pd.isna(batch_val):
                batch_val = None
            if not _values_equal(ref_val, batch_val):
                mismatches += 1
                print(f"  [BATCH MISMATCH] TransactionID={txn['TransactionID']} field={key} "
                      f"reference={ref_val!r} batch={batch_val!r}")
    return mismatches


def run() -> bool:
    print("Loading and sorting full dataset...")
    df = _load_sorted_with_entities()
    n = len(df)

    rng = np.random.default_rng(RNG_SEED)
    eligible = df.index[df["TransactionDT"] > df["TransactionDT"].quantile(0.05)].to_numpy()
    sample_idx = rng.choice(eligible, size=min(SAMPLE_SIZE, len(eligible)), replace=False)

    print(f"Sampled {len(sample_idx)} transactions out of {n} rows.")

    print("\n[1/2] Future-blindness check (full dataset vs future-deleted)...")
    fb_mismatches = check_future_blindness(df, sample_idx)
    print(f"  -> {fb_mismatches} mismatches" if fb_mismatches else "  -> PASS, 0 mismatches")

    print("\n[2/2] Batch-vs-reference equivalence check...")
    batch_mismatches = check_batch_equivalence(df, sample_idx)
    print(f"  -> {batch_mismatches} mismatches" if batch_mismatches else "  -> PASS, 0 mismatches")

    ok = fb_mismatches == 0 and batch_mismatches == 0
    print("\n" + ("LEAKAGE TEST: PASS" if ok else "LEAKAGE TEST: FAIL"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
