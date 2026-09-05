"""Seeds the audit chain with real (test-slice) decisions so the
dashboard's audit-trail screen has content beyond the dev demo. Reads
Phase 5's case_samples.json (already scored + attributed) and the
chosen optimal thresholds.
"""
import json

from src.audit.chain import DEFAULT_DB_PATH, append_decision, append_threshold_change, connect
from src.model.train import MODEL_VERSION, MODELS_DIR

RESULTS_DIR = MODELS_DIR / "phase5"


def run() -> None:
    with open(RESULTS_DIR / "sweep_summary.json") as f:
        summary = json.load(f)
    with open(RESULTS_DIR / "case_samples.json") as f:
        cases = json.load(f)

    t_low, t_high = summary["optimum"]["t_low"], summary["optimum"]["t_high"]

    if DEFAULT_DB_PATH.exists():
        DEFAULT_DB_PATH.unlink()
    conn = connect()

    append_threshold_change(
        conn, model_version=MODEL_VERSION, t_low=t_low, t_high=t_high,
        reason="Phase 5 net-rupee sweep optimum on test slice",
    )
    for case in cases:
        append_decision(
            conn, transaction_id=case["TransactionID"], model_version=MODEL_VERSION,
            raw_score=case["raw_score"], calibrated_score=case["calibrated_score"],
            band=case["band"], t_low=t_low, t_high=t_high,
            top_attributions=case["top_attributions"], action=case["band"],
        )
    conn.close()
    print(f"Seeded {len(cases) + 1} audit events to {DEFAULT_DB_PATH}")


if __name__ == "__main__":
    run()
