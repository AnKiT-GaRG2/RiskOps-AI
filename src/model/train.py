"""Phase 4 — LightGBM + isotonic calibration.

Imbalance handled with scale_pos_weight, not SMOTE: synthetic
oversampling on entity-linked fraud data fabricates transactions that
never existed and interacts badly with the temporal, entity-disjoint
split (a synthetic neighbour of a train-slice fraud could easily land
arbitrarily close to a real calib/test-slice transaction).

Calibration is fit on the calibration slice ONLY — never on train.
"""
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.split import PROCESSED_DIR
from src.model.calibration import brier_score, expected_calibration_error, reliability_table
from src.model.features import FEATURE_COLS, CATEGORICAL_COLS, prep_features

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_VERSION = "lgbm_v1"

LGBM_PARAMS = dict(
    n_estimators=400,
    num_leaves=63,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=30,
    random_state=42,
)


def load_matrix(name: str) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}_features.parquet")
    return prep_features(df)


def train_model(train_df: pd.DataFrame) -> lgb.LGBMClassifier:
    n_pos = int(train_df["isFraud"].sum())
    n_neg = int(len(train_df) - n_pos)
    scale_pos_weight = n_neg / n_pos

    model = lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight, verbosity=-1, **LGBM_PARAMS)
    model.fit(
        train_df[FEATURE_COLS], train_df["isFraud"],
        categorical_feature=CATEGORICAL_COLS,
    )
    return model


def score(model: lgb.LGBMClassifier, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURE_COLS])[:, 1]


def run() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading feature matrices...")
    train_df = load_matrix("train")
    calib_df = load_matrix("calib")
    test_df = load_matrix("test")

    print(f"train={len(train_df)} calib={len(calib_df)} test={len(test_df)}")

    print("Training LightGBM...")
    model = train_model(train_df)

    raw_calib = score(model, calib_df)
    raw_test = score(model, test_df)

    pr_auc_test = average_precision_score(test_df["isFraud"], raw_test)
    roc_auc_test = roc_auc_score(test_df["isFraud"], raw_test)

    print("Fitting isotonic calibration on CALIB slice only...")
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_calib, calib_df["isFraud"])

    cal_calib = calibrator.predict(raw_calib)
    cal_test = calibrator.predict(raw_test)

    brier_before = brier_score(test_df["isFraud"], raw_test)
    brier_after = brier_score(test_df["isFraud"], cal_test)
    ece_before = expected_calibration_error(test_df["isFraud"], raw_test)
    ece_after = expected_calibration_error(test_df["isFraud"], cal_test)

    reliability_before = reliability_table(test_df["isFraud"], raw_test)
    reliability_after = reliability_table(test_df["isFraud"], cal_test)

    print("=" * 72)
    print(f"PR-AUC (primary, test):  {pr_auc_test:.4f}")
    print(f"ROC-AUC (secondary, test): {roc_auc_test:.4f}")
    print(f"  PR-AUC leads because ROC-AUC is flattered by the huge true-negative")
    print(f"  mass at this base rate ({test_df['isFraud'].mean():.2%}).")
    print(f"Brier score:  before={brier_before:.4f}  after={brier_after:.4f}")
    print(f"ECE:          before={ece_before:.4f}  after={ece_after:.4f}")
    print("=" * 72)

    joblib.dump(model, MODELS_DIR / f"{MODEL_VERSION}_model.joblib")
    joblib.dump(calibrator, MODELS_DIR / f"{MODEL_VERSION}_calibrator.joblib")

    metrics = {
        "model_version": MODEL_VERSION,
        "pr_auc_test": pr_auc_test,
        "roc_auc_test": roc_auc_test,
        "brier_before": brier_before,
        "brier_after": brier_after,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "n_train": len(train_df), "n_calib": len(calib_df), "n_test": len(test_df),
        "base_rate_test": float(test_df["isFraud"].mean()),
    }
    with open(MODELS_DIR / f"{MODEL_VERSION}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    reliability_before.to_json(MODELS_DIR / f"{MODEL_VERSION}_reliability_before.json", orient="records")
    reliability_after.to_json(MODELS_DIR / f"{MODEL_VERSION}_reliability_after.json", orient="records")

    scored_test = test_df[["TransactionID", "isFraud", "TransactionDT"]].copy()
    scored_test["raw_score"] = raw_test
    scored_test["calibrated_score"] = cal_test
    scored_test.to_parquet(PROCESSED_DIR / "test_scored.parquet", index=False)

    scored_calib = calib_df[["TransactionID", "isFraud", "TransactionDT"]].copy()
    scored_calib["raw_score"] = raw_calib
    scored_calib["calibrated_score"] = cal_calib
    scored_calib.to_parquet(PROCESSED_DIR / "calib_scored.parquet", index=False)

    print(f"Saved model, calibrator, metrics to {MODELS_DIR}")


if __name__ == "__main__":
    run()
