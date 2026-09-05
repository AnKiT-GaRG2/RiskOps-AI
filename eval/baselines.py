"""Three hand-written rule baselines, evaluated through the same
net_rupees cost function as the model. If LightGBM doesn't clear these
by a wide margin, that's the finding, not a bug to hide.

Rule 3 as specified needs `failed_attempts_10min` — a decline-attempt
counter IEEE-CIS does not carry (no acquirer response codes in this
dataset). We substitute the nearest available signal, `velocity_10min`
(all attempts, successful or not, by the same card in 10 minutes), and
say so here rather than fabricate a field the data can't support.

`customer_median` in rule 1 is the median velocity_1h over the TRAIN
slice only (a global reference a rule-writer would calibrate once from
history), never from calib/test, to avoid leaking the split.
"""
import numpy as np
import pandas as pd

from src.policy.costs import DEFAULT_COSTS
from src.policy.evaluate import net_rupees

RULE1_DISTINCT_CARDS_MIN = 3
RULE1_VELOCITY_MULTIPLIER = 5
RULE2_ZSCORE_THRESHOLD = 4
RULE3_ATTEMPTS_THRESHOLD = 5


def fit_reference_stats(train_matrix: pd.DataFrame) -> dict:
    return {"customer_median_velocity_1h": float(train_matrix["velocity_1h"].median())}


def rule_1_velocity_fanout(df: pd.DataFrame, ref: dict) -> np.ndarray:
    threshold = RULE1_VELOCITY_MULTIPLIER * max(ref["customer_median_velocity_1h"], 1e-9)
    return ((df["velocity_1h"] > threshold) & (df["distinct_cards_per_device"].fillna(0) >= RULE1_DISTINCT_CARDS_MIN)).to_numpy()


def rule_2_amount_outlier(df: pd.DataFrame, ref: dict) -> np.ndarray:
    return (df["amount_zscore"] > RULE2_ZSCORE_THRESHOLD).to_numpy()


def rule_3_rapid_attempts(df: pd.DataFrame, ref: dict) -> np.ndarray:
    return (df["velocity_10min"] >= RULE3_ATTEMPTS_THRESHOLD).to_numpy()


RULES = {
    "rule_1_velocity_fanout": rule_1_velocity_fanout,
    "rule_2_amount_outlier": rule_2_amount_outlier,
    "rule_3_rapid_attempts": rule_3_rapid_attempts,
}


def evaluate_baselines(train_matrix: pd.DataFrame, test_matrix: pd.DataFrame, costs=DEFAULT_COSTS) -> dict:
    ref = fit_reference_stats(train_matrix)
    y_true = test_matrix["isFraud"].to_numpy()

    results = {}
    combined_flag = np.zeros(len(test_matrix), dtype=bool)

    for name, fn in RULES.items():
        flagged = fn(test_matrix, ref)
        combined_flag |= flagged
        scores = flagged.astype(float)
        # t_low = t_high = 0.5: score 1.0 -> HOLD, score 0.0 -> ALLOW. No step-up band for a binary rule.
        metrics = net_rupees(y_true, scores, t_low=0.5, t_high=0.5, costs=costs)
        precision = float((flagged & (y_true == 1)).sum() / flagged.sum()) if flagged.sum() else 0.0
        recall = float((flagged & (y_true == 1)).sum() / (y_true == 1).sum()) if (y_true == 1).sum() else 0.0
        results[name] = {"precision": precision, "recall": recall, "flagged_rate": float(flagged.mean()), **metrics}

    combined_scores = combined_flag.astype(float)
    combined_metrics = net_rupees(y_true, combined_scores, t_low=0.5, t_high=0.5, costs=costs)
    precision = float((combined_flag & (y_true == 1)).sum() / combined_flag.sum()) if combined_flag.sum() else 0.0
    recall = float((combined_flag & (y_true == 1)).sum() / (y_true == 1).sum()) if (y_true == 1).sum() else 0.0
    results["combined_any_rule"] = {
        "precision": precision, "recall": recall, "flagged_rate": float(combined_flag.mean()), **combined_metrics,
    }
    results["_reference_stats"] = ref
    return results


def run() -> None:
    from src.data.split import PROCESSED_DIR

    train_matrix = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    test_matrix = pd.read_parquet(PROCESSED_DIR / "test_features.parquet")

    results = evaluate_baselines(train_matrix, test_matrix)

    print("=" * 80)
    print("BASELINE RULES — evaluated on test slice through net_rupees")
    print("=" * 80)
    for name, m in results.items():
        if name == "_reference_stats":
            continue
        print(f"\n{name}")
        print(f"  precision={m['precision']:.4f}  recall={m['recall']:.4f}  flagged_rate={m['flagged_rate']:.4%}")
        print(f"  net=Rs.{m['net']:,.0f}  prevented=Rs.{m['prevented']:,.0f}  "
              f"fn_loss=Rs.{m['fn_loss']:,.0f}  fp_loss=Rs.{m['fp_loss']:,.0f}  review_loss=Rs.{m['review_loss']:,.0f}")


if __name__ == "__main__":
    run()
