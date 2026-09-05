"""Phase 5 — deterministic policy assertions. Pure arithmetic, no
randomness: every assertion here should be reproducible by hand.

Run: python -m eval.test_policy
"""
import numpy as np

from src.policy.bands import ALLOW, HOLD, STEPUP, decide
from src.policy.costs import CostModel
from src.policy.evaluate import net_rupees

FAILURES = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def test_band_boundaries():
    print("\nband boundaries exactly at T_LOW and T_HIGH")
    scores = [0.0, 0.2999, 0.3, 0.3001, 0.6999, 0.7, 0.7001, 1.0]
    decisions = decide(scores, t_low=0.3, t_high=0.7)
    expected = [ALLOW, ALLOW, STEPUP, STEPUP, STEPUP, HOLD, HOLD, HOLD]
    check("t_low is inclusive of STEPUP, t_high is inclusive of HOLD",
          list(decisions) == expected, f"got {list(decisions)}")


def test_extreme_scores():
    print("\nscores of 0.0 and 1.0")
    decisions = decide([0.0, 1.0], t_low=0.3, t_high=0.7)
    check("score 0.0 -> ALLOW", decisions[0] == ALLOW)
    check("score 1.0 -> HOLD", decisions[1] == HOLD)


def test_empty_input():
    print("\nempty input")
    result = net_rupees([], [], t_low=0.3, t_high=0.7)
    check("net is 0.0 on empty input", result["net"] == 0.0, str(result["net"]))
    check("review_rate is 0.0 on empty input", result["review_rate"] == 0.0)


def test_all_fraud():
    print("\nall-fraud input")
    costs = CostModel()
    n = 20
    y_true = np.ones(n, dtype=int)
    scores = np.full(n, 0.9)  # all HOLD
    result = net_rupees(y_true, scores, t_low=0.3, t_high=0.7, costs=costs)
    expected_prevented = n * costs.avg_fraud_amount * (1 - costs.retry_rate)
    expected_review = n * costs.review_cost
    expected_net = expected_prevented - expected_review
    check("all-fraud, all-held: net matches hand computation",
          np.isclose(result["net"], expected_net), f"{result['net']} vs {expected_net}")
    check("recall is 1.0", np.isclose(result["recall"], 1.0))


def test_all_legit():
    print("\nall-legit input")
    costs = CostModel()
    n = 20
    y_true = np.zeros(n, dtype=int)
    scores = np.full(n, 0.9)  # all HOLD -> all false positives
    result = net_rupees(y_true, scores, t_low=0.3, t_high=0.7, costs=costs)
    expected_net = -(n * costs.fp_cost) - (n * costs.review_cost)
    check("all-legit, all-held: net matches hand computation",
          np.isclose(result["net"], expected_net), f"{result['net']} vs {expected_net}")
    check("precision is 0.0", np.isclose(result["precision"], 0.0))


def test_ten_row_fixture():
    print("\n10-row fixture reconciled by hand")
    costs = CostModel()
    #            fraud?  score    decision (t_low=0.3, t_high=0.7)
    y_true = [0, 0, 0, 1, 1, 0, 1, 0, 1, 0]
    scores = [0.1, 0.2, 0.35, 0.9, 0.5, 0.05, 0.75, 0.6, 0.29, 0.99]
    #        ALLOW ALLOW STEPUP HOLD STEPUP ALLOW HOLD STEPUP ALLOW HOLD
    # ALLOW & fraud (FN): idx 8 (fraud=1, score=0.29) -> 1 FN
    # STEPUP & legit: idx 2 (0.35), idx 7 (0.6) -> 2
    # STEPUP & fraud: idx 4 (0.5) -> 1, split by p_stepup_stops_fraud
    # HOLD & legit: idx 9 (0.99) -> 1
    # HOLD & fraud (caught): idx 3 (0.9), idx 6 (0.75) -> 2
    stepup_fraud_stopped = 1 * costs.p_stepup_stops_fraud
    stepup_fraud_through = 1 * (1 - costs.p_stepup_stops_fraud)
    fn_loss = (1 + stepup_fraud_through) * costs.fn_cost
    fp_loss = 1 * costs.fp_cost
    stepup_loss = 2 * costs.stepup_cost
    caught = stepup_fraud_stopped + 2
    prevented = caught * costs.avg_fraud_amount * (1 - costs.retry_rate)
    review_loss = (1 + 2) * costs.review_cost
    expected_net = prevented - fn_loss - fp_loss - stepup_loss - review_loss

    result = net_rupees(y_true, scores, t_low=0.3, t_high=0.7, costs=costs)
    check("net", np.isclose(result["net"], expected_net), f"{result['net']} vs {expected_net}")
    check("fn_loss", np.isclose(result["fn_loss"], fn_loss))
    check("fp_loss", np.isclose(result["fp_loss"], fp_loss))
    check("stepup_loss", np.isclose(result["stepup_loss"], stepup_loss))
    check("prevented", np.isclose(result["prevented"], prevented))
    check("review_loss", np.isclose(result["review_loss"], review_loss))
    check("review_rate", np.isclose(result["review_rate"], 3 / 10))


def run() -> bool:
    test_band_boundaries()
    test_extreme_scores()
    test_empty_input()
    test_all_fraud()
    test_all_legit()
    test_ten_row_fixture()

    print("\n" + ("POLICY TESTS: PASS" if not FAILURES else f"POLICY TESTS: FAIL ({len(FAILURES)} failures)"))
    return not FAILURES


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
