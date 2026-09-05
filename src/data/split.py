"""Phase 1 — temporal, entity-disjoint three-way split.

Order matters:
  1. Sort by TransactionDT (time, not row order).
  2. Cut at ~60% / ~15% / ~25% -> train / calibration / test.
  3. Enforce entity-disjointness: any card_entity or device_entity that
     appears in more than one slice is kept only in its earliest slice;
     its rows in later slices are dropped.

TransactionDT is seconds-since-a-reference-point, not a Unix timestamp.
We anchor it to 2017-12-01 purely for human-readable reporting (a
convention used across the IEEE-CIS community, not an official value) —
no modelling logic depends on the anchor being correct.
"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.entities import add_entity_keys

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
REFERENCE_DATE = pd.Timestamp("2017-12-01")

TRAIN_FRAC = 0.60
CALIB_FRAC = 0.15
# remainder (~0.25) -> test


@dataclass
class SliceStats:
    name: str
    rows: int
    positives: int
    base_rate: float
    date_start: str
    date_end: str


_RAW_CACHE = PROCESSED_DIR / "raw_merged.parquet"


def load_raw() -> pd.DataFrame:
    """CSV parsing across ~394 columns / 590k rows is the slow part
    (~2 minutes); cache the merged result as parquet so repeated runs
    (split, leakage test, ad-hoc profiling) don't pay it every time."""
    if _RAW_CACHE.exists():
        return pd.read_parquet(_RAW_CACHE)

    txn = pd.read_csv(RAW_DIR / "train_transaction.csv")
    ident = pd.read_csv(RAW_DIR / "train_identity.csv")
    df = txn.merge(ident, on="TransactionID", how="left")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_RAW_CACHE, index=False)
    return df


def assign_temporal_slices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    n = len(df)
    cut1 = int(n * TRAIN_FRAC)
    cut2 = int(n * (TRAIN_FRAC + CALIB_FRAC))
    slice_arr = np.empty(n, dtype=object)
    slice_arr[:cut1] = "train"
    slice_arr[cut1:cut2] = "calib"
    slice_arr[cut2:] = "test"
    df["slice"] = slice_arr
    return df


_SLICE_ORDER = {"train": 0, "calib": 1, "test": 2}


def enforce_entity_disjoint(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = add_entity_keys(df)
    df["_slice_ord"] = df["slice"].map(_SLICE_ORDER)

    drop_mask = pd.Series(False, index=df.index)
    dropped_by = {}

    for entity_col in ["card_entity", "device_entity"]:
        sub = df.loc[df[entity_col].notna(), [entity_col, "_slice_ord"]]
        earliest = sub.groupby(entity_col)["_slice_ord"].min()
        row_earliest = df[entity_col].map(earliest)
        conflict = df[entity_col].notna() & (df["_slice_ord"] > row_earliest)
        dropped_by[entity_col] = int(conflict.sum())
        drop_mask |= conflict

    kept = df.loc[~drop_mask].drop(columns=["_slice_ord"]).reset_index(drop=True)
    report = {
        "rows_dropped_total": int(drop_mask.sum()),
        "rows_dropped_by_entity": dropped_by,
    }
    return kept, report


def summarize(df: pd.DataFrame) -> list[SliceStats]:
    stats = []
    for name in ["train", "calib", "test"]:
        s = df[df["slice"] == name]
        dt_start = REFERENCE_DATE + pd.to_timedelta(s["TransactionDT"].min(), unit="s")
        dt_end = REFERENCE_DATE + pd.to_timedelta(s["TransactionDT"].max(), unit="s")
        stats.append(
            SliceStats(
                name=name,
                rows=len(s),
                positives=int(s["isFraud"].sum()),
                base_rate=float(s["isFraud"].mean()),
                date_start=str(dt_start.date()),
                date_end=str(dt_end.date()),
            )
        )
    return stats


def assert_no_overlap(df: pd.DataFrame) -> None:
    for entity_col in ["card_entity", "device_entity"]:
        sub = df.loc[df[entity_col].notna(), [entity_col, "slice"]]
        slices_per_entity = sub.groupby(entity_col)["slice"].nunique()
        n_overlap = int((slices_per_entity > 1).sum())
        assert n_overlap == 0, f"{entity_col} has {n_overlap} entities spanning multiple slices"


def assert_temporal_order(df: pd.DataFrame) -> None:
    train_max = df.loc[df["slice"] == "train", "TransactionDT"].max()
    calib_min = df.loc[df["slice"] == "calib", "TransactionDT"].min()
    calib_max = df.loc[df["slice"] == "calib", "TransactionDT"].max()
    test_min = df.loc[df["slice"] == "test", "TransactionDT"].min()
    assert train_max <= calib_min, "train overlaps calib in time"
    assert calib_max <= test_min, "calib overlaps test in time"


def run() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    df = assign_temporal_slices(df)
    df, drop_report = enforce_entity_disjoint(df)

    assert_no_overlap(df)
    assert_temporal_order(df)

    stats = summarize(df)

    print("=" * 72)
    print("PHASE 1 — SPLIT REPORT")
    print("=" * 72)
    header = f"{'slice':<8}{'rows':>10}{'positives':>12}{'base_rate':>12}{'date_start':>14}{'date_end':>14}"
    print(header)
    for s in stats:
        print(f"{s.name:<8}{s.rows:>10}{s.positives:>12}{s.base_rate:>12.4%}{s.date_start:>14}{s.date_end:>14}")
    print("-" * 72)
    print(f"Rows dropped to entity-disjointness: {drop_report['rows_dropped_total']}")
    print(f"  by card_entity:   {drop_report['rows_dropped_by_entity']['card_entity']}")
    print(f"  by device_entity: {drop_report['rows_dropped_by_entity']['device_entity']}")
    print("=" * 72)

    for name in ["train", "calib", "test"]:
        out = df[df["slice"] == name].drop(columns=["slice"])
        out.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)

    report = {
        "slices": [asdict(s) for s in stats],
        "drop_report": drop_report,
        "reference_date_anchor": str(REFERENCE_DATE.date()),
        "train_frac": TRAIN_FRAC,
        "calib_frac": CALIB_FRAC,
    }
    with open(PROCESSED_DIR / "split_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote train/calib/test parquet + split_report.json to {PROCESSED_DIR}")


if __name__ == "__main__":
    run()
