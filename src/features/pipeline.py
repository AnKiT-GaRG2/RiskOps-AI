"""The one entry point (non-negotiable #3). Every call site — live
scoring, the attack injector, the leakage test — passes txn's own
timestamp as as_of and gets `history_df` filtered here, structurally,
before any aggregate touches it. No other function computes a
behavioural aggregate from unfiltered history.
"""
from datetime import datetime

import pandas as pd

from src.features.as_of import filter_history
from src.features.behavioral import compute_behavioral_features
from src.features.static import compute_static_features

# compute_behavioral_features only ever touches these columns. Selecting
# them BEFORE filtering by as_of (rather than after) is what makes this
# usable on a wide raw table: filtering is a row-copy proportional to
# (rows kept x columns), and history_df can carry ~400 raw columns a
# single scored transaction never needs.
_NEEDED_HISTORY_COLS = ["TransactionDT", "TransactionAmt", "isFraud", "card_entity", "device_entity"]


def compute_features(txn: pd.Series, history_df: pd.DataFrame, as_of) -> dict:
    history = filter_history(history_df[_NEEDED_HISTORY_COLS], as_of)
    static = compute_static_features(txn)
    behavioral = compute_behavioral_features(txn, history, as_of)
    return {**static, **behavioral}
