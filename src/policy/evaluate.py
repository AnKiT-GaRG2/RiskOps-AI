"""The net-rupee function. NO ML, NO LLM — pure arithmetic over
(y_true, scores, thresholds, costs). This is what every threshold sweep,
baseline comparison, and sensitivity band in Phase 5 calls.

Accounting, per transaction:
  ALLOW & fraud   -> fn_loss    += FN_COST            (fraud got through)
  ALLOW & legit   -> 0
  STEPUP & legit  -> stepup_loss += STEPUP_COST         (expected abandonment cost)
  STEPUP & fraud  -> a P_STEPUP_STOPS_FRAUD fraction is prevented (same
                     discounted credit as a HOLD catch); the rest still
                     gets through and costs FN_COST. HOLD, not STEPUP, is
                     the band that genuinely stops a transaction (a human
                     reviews it); OTP/3DS is a friction step a fraudster
                     may or may not clear, and IEEE-CIS carries no
                     completion signal to measure how often. An earlier
                     version of this function credited STEPUP with the
                     same 100% catch certainty as HOLD at a fraction of
                     the cost, which made "step up everyone regardless of
                     score" accounting-optimal — see P_STEPUP_STOPS_FRAUD
                     in costs.py for why that was wrong.
  HOLD & legit    -> fp_loss    += FP_COST; review_loss += REVIEW_COST
  HOLD & fraud    -> prevented  += AVG_FRAUD_AMOUNT * (1 - RETRY_RATE); review_loss += REVIEW_COST

net = prevented - fn_loss - fp_loss - stepup_loss - review_loss

"prevented" is discounted by RETRY_RATE because a blocked fraudster
typically retries on another card — blocking one transaction does not
recover the full fraud amount as realized value.
"""
import numpy as np

from src.policy.bands import ALLOW, HOLD, STEPUP, decide
from src.policy.costs import CostModel, DEFAULT_COSTS


def net_rupees(y_true, scores, t_low: float, t_high: float, costs: CostModel = DEFAULT_COSTS) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    decisions = decide(scores, t_low, t_high)

    is_fraud = y_true == 1
    is_legit = ~is_fraud

    n_allow_fraud = int(np.sum((decisions == ALLOW) & is_fraud))
    n_allow_legit = int(np.sum((decisions == ALLOW) & is_legit))
    n_stepup_fraud = int(np.sum((decisions == STEPUP) & is_fraud))
    n_stepup_legit = int(np.sum((decisions == STEPUP) & is_legit))
    n_hold_fraud = int(np.sum((decisions == HOLD) & is_fraud))
    n_hold_legit = int(np.sum((decisions == HOLD) & is_legit))

    stepup_fraud_stopped = n_stepup_fraud * costs.p_stepup_stops_fraud
    stepup_fraud_through = n_stepup_fraud * (1 - costs.p_stepup_stops_fraud)

    fn_loss = (n_allow_fraud + stepup_fraud_through) * costs.fn_cost
    fp_loss = n_hold_legit * costs.fp_cost
    stepup_loss = n_stepup_legit * costs.stepup_cost
    n_reviewed = n_hold_fraud + n_hold_legit
    review_loss = n_reviewed * costs.review_cost
    caught_fraud = stepup_fraud_stopped + n_hold_fraud
    prevented = caught_fraud * costs.avg_fraud_amount * (1 - costs.retry_rate)

    net = prevented - fn_loss - fp_loss - stepup_loss - review_loss

    n_total = len(y_true)
    n_fraud_total = int(is_fraud.sum())

    recall = caught_fraud / n_fraud_total if n_fraud_total else 0.0
    n_flagged = n_stepup_fraud + n_stepup_legit + n_hold_fraud + n_hold_legit
    precision = caught_fraud / n_flagged if n_flagged else 0.0

    return {
        "net": float(net),
        "fn_loss": float(fn_loss),
        "fp_loss": float(fp_loss),
        "stepup_loss": float(stepup_loss),
        "review_loss": float(review_loss),
        "prevented": float(prevented),
        "counts": {
            "n_total": n_total,
            "n_allow_fraud": n_allow_fraud, "n_allow_legit": n_allow_legit,
            "n_stepup_fraud": n_stepup_fraud, "n_stepup_legit": n_stepup_legit,
            "n_hold_fraud": n_hold_fraud, "n_hold_legit": n_hold_legit,
            "n_fraud_total": n_fraud_total,
        },
        "review_rate": n_reviewed / n_total if n_total else 0.0,
        "recall": recall,
        "precision": precision,
    }
