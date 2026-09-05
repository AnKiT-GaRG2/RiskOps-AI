"""Phase 5 runner — produces every artifact the dashboard's sweep,
sensitivity, and case-detail screens read. NO ML training here: it
consumes the scores src/model/train.py already produced.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.split import PROCESSED_DIR
from src.model.attributions import compute_contributions, top_k_attributions
from src.model.features import prep_features
from src.model.train import MODELS_DIR, MODEL_VERSION
from src.policy.costs import DEFAULT_COSTS
from src.policy.sweep import f1_optimal_threshold, global_optimum, net_at_single_threshold, sensitivity_sweep, sweep_bands
from src.policy.selective_labels import compute_exploration_budget

RESULTS_DIR = MODELS_DIR / "phase5"
N_CASE_SAMPLES = 30
REVIEW_MINUTES_PER_CASE = 4  # matches REVIEW_COST's stated basis in src/policy/costs.py


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scored = pd.read_parquet(PROCESSED_DIR / "test_scored.parquet")
    features = pd.read_parquet(PROCESSED_DIR / "test_features.parquet")
    y_true = scored["isFraud"].to_numpy()
    scores = scored["calibrated_score"].to_numpy()

    print("Sweeping T_HIGH x T_LOW over net rupees...")
    sweep_df = sweep_bands(y_true, scores)
    sweep_df.to_json(RESULTS_DIR / "sweep.json", orient="records")

    opt = global_optimum(sweep_df)
    naive_05 = net_at_single_threshold(y_true, scores, 0.5)
    f1_t = f1_optimal_threshold(y_true, scores)
    f1_result = net_at_single_threshold(y_true, scores, f1_t)
    # "Do nothing" (allow everyone, t=1.0 puts every real score < 1.0 into
    # ALLOW): the honest floor every other number here should be read
    # against, since net at the optimum can still be negative in absolute
    # terms — the model doesn't eliminate fraud loss, it reduces it.
    do_nothing = net_at_single_threshold(y_true, scores, 1.0)

    n_total_test = int(opt["counts"]["n_total"])
    analyst_hours_per_1000 = opt["review_rate"] * 1000 * REVIEW_MINUTES_PER_CASE / 60
    net_per_1000 = opt["net"] / n_total_test * 1000

    summary = {
        "optimum": {
            "t_low": float(opt["t_low"]), "t_high": float(opt["t_high"]), "net": float(opt["net"]),
            "precision": float(opt["precision"]), "recall": float(opt["recall"]),
            "review_rate": float(opt["review_rate"]),
            "analyst_hours_per_1000_txns": float(analyst_hours_per_1000),
            "net_per_1000_txns": float(net_per_1000),
        },
        "naive_0.5": {"t_low": 0.5, "t_high": 0.5, "net": naive_05["net"],
                      "precision": naive_05["precision"], "recall": naive_05["recall"]},
        "gap_vs_0.5_rupees": float(opt["net"] - naive_05["net"]),
        "f1_optimal": {"t": float(f1_t), "net": f1_result["net"],
                       "precision": f1_result["precision"], "recall": f1_result["recall"]},
        "gap_vs_f1_rupees": float(opt["net"] - f1_result["net"]),
        "do_nothing": {"net": do_nothing["net"], "recall": do_nothing["recall"]},
        "improvement_vs_do_nothing_rupees": float(opt["net"] - do_nothing["net"]),
    }
    print(json.dumps(summary, indent=2))
    with open(RESULTS_DIR / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Sensitivity sweep (+/-30% per cost input)...")
    sens_df = sensitivity_sweep(y_true, scores, DEFAULT_COSTS, perturbation=0.30)
    sens_df.to_json(RESULTS_DIR / "sensitivity.json", orient="records")
    t_high_band = (sens_df["opt_t_high"].min(), sens_df["opt_t_high"].max())
    print(f"Optimal T_HIGH ranges [{t_high_band[0]:.3f}, {t_high_band[1]:.3f}] across all +/-30% perturbations "
          f"(unperturbed optimum: {opt['t_high']:.3f})")

    print("Baselines...")
    from eval.baselines import evaluate_baselines
    train_features = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    baseline_results = evaluate_baselines(train_features, features)
    with open(RESULTS_DIR / "baselines.json", "w") as f:
        json.dump(baseline_results, f, indent=2, default=str)

    print("Model attributions for case-detail sample...")
    model = joblib.load(MODELS_DIR / f"{MODEL_VERSION}_model.joblib")
    features_prepped = prep_features(features)
    contrib_df = compute_contributions(model, features_prepped)

    rng = np.random.default_rng(0)
    band_labels = np.where(scores >= opt["t_high"], "HOLD", np.where(scores >= opt["t_low"], "STEPUP", "ALLOW"))
    sample_idx = rng.choice(features.index, size=min(N_CASE_SAMPLES, len(features)), replace=False)

    cases = []
    for idx in sample_idx:
        pos = features.index.get_loc(idx)
        cases.append({
            "TransactionID": str(features.loc[idx, "TransactionID"]),
            "isFraud": int(y_true[pos]),
            "raw_score": float(scored.loc[idx, "raw_score"]),
            "calibrated_score": float(scores[pos]),
            "band": band_labels[pos],
            "top_attributions": top_k_attributions(features_prepped, contrib_df, idx, k=3),
        })
    with open(RESULTS_DIR / "case_samples.json", "w") as f:
        json.dump(cases, f, indent=2, default=str)

    print("Selective-labels budget on test-slice volumes...")
    n_hold = int((band_labels == "HOLD").sum())
    fraud_rate_in_holds = float(y_true[band_labels == "HOLD"].mean()) if n_hold else 0.0
    budget = compute_exploration_budget(hold_volume=n_hold, fraud_rate_in_holds=fraud_rate_in_holds)
    with open(RESULTS_DIR / "selective_labels.json", "w") as f:
        json.dump(vars(budget), f, indent=2)

    print(f"All Phase 5 artifacts written to {RESULTS_DIR}")


if __name__ == "__main__":
    run()
