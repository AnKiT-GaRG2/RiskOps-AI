"""The as_of cutoff. Used only from within compute_features."""
import pandas as pd

TIMESTAMP_COL = "TransactionDT"


def filter_history(history_df: pd.DataFrame, as_of) -> pd.DataFrame:
    """Rows strictly before as_of. `<`, never `<=` — a row sharing the
    exact same timestamp as the scored transaction is "at", not "before",
    and must be excluded even if it sorts earlier in the table."""
    return history_df[history_df[TIMESTAMP_COL] < as_of]
