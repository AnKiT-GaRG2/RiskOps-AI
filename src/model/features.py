"""Feature column contracts shared by training, scoring, and the
policy/dashboard layers that read a scored feature matrix."""
import pandas as pd

CATEGORICAL_COLS = [
    "product_cd", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "p_email_domain", "r_email_domain",
    "device_type", "device_info", "os", "browser", "screen_res",
]

NUMERIC_COLS = [
    "amount", "hour_of_day", "day_of_week", "dist1", "dist2",
    "has_dist1", "has_dist2", "email_domain_match",
    "velocity_10min", "velocity_1h", "velocity_24h", "velocity_7d",
    "distinct_devices_per_card", "distinct_cards_per_device", "history_count",
    "amount_zscore", "time_since_prev_txn", "is_first_txn", "historical_decline_rate",
]

FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

NON_FEATURE_COLS = ["TransactionID", "isFraud", "TransactionDT"]

# Feature attributions are internal-analyst-only (non-negotiable #7).
# Anything under src/analyst or app/ serving a customer-facing view must
# never read this list's *output* (contributions), only decisions.
TOP_K_ATTRIBUTIONS = 3


def prep_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CATEGORICAL_COLS:
        out[col] = out[col].astype("string").astype("category")
    for col in NUMERIC_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out
