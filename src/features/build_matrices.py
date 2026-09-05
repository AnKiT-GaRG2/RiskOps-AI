"""Builds train/calib/test feature matrices from the Phase 1 split
output. Entity-disjointness (Phase 1) confines every card_entity and
device_entity to a single slice, so each slice's behavioural history is
self-contained — no cross-slice history is needed or used here.
"""
from pathlib import Path

import pandas as pd

from src.data.split import PROCESSED_DIR
from src.features.batch import build_feature_matrix


def run() -> None:
    for name in ["train", "calib", "test"]:
        df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
        df = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
        matrix = build_feature_matrix(df)
        out_path = PROCESSED_DIR / f"{name}_features.parquet"
        matrix.to_parquet(out_path, index=False)
        print(f"{name}: {matrix.shape[0]} rows, {matrix.shape[1]} cols -> {out_path}")


if __name__ == "__main__":
    run()
