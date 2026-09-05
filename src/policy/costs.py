"""The cost model. Every parameter is named and sourced or flagged as an
estimate — this file is the only place rupee assumptions live, and every
downstream number (net rupees, threshold choice, sensitivity band)
traces back to it.

NO ML, NO LLM in this module or anywhere under src/policy/.
"""
from dataclasses import dataclass

# ---------------------------------------------------------------------
# False negative: a fraudulent transaction we let through.
# ---------------------------------------------------------------------
AVG_FRAUD_AMOUNT = 4500      # given by the brief; median disputed-txn amount is a reasonable proxy
DISPUTE_FEE = 400            # ESTIMATE: Razorpay publishes a Rs 200-600 dispute-fee range per
                              # chargeback (razorpay.com/docs/payments/disputes/); midpoint used here.
HANDLING = 300                # ESTIMATE: ~20 analyst-minutes fully loaded to process a chargeback case
FN_COST = AVG_FRAUD_AMOUNT + DISPUTE_FEE + HANDLING

# ---------------------------------------------------------------------
# False positive: a legitimate transaction we wrongly block.
# ---------------------------------------------------------------------
AVG_LEGIT_MARGIN = 240        # given by the brief: 12% margin on a Rs 2,000 order
SUPPORT_CONTACT = 150         # given by the brief: cost of the resulting support ticket
P_CHURN_WRONG_BLOCK = 0.05    # given by the brief: probability a wrongly-blocked customer never returns
CUSTOMER_LTV = 8000           # given by the brief
# Churn dominates: 0.05 * 8000 = 400, nearly double the margin itself.
# Pricing FP at "the fee we didn't collect" (a few hundred rupees) is
# roughly an order of magnitude low once churn is counted.
FP_COST = AVG_LEGIT_MARGIN + SUPPORT_CONTACT + (P_CHURN_WRONG_BLOCK * CUSTOMER_LTV)

# ---------------------------------------------------------------------
# Step-up: OTP / 3DS. Not free — it's a partial decline.
# ---------------------------------------------------------------------
P_ABANDON_STEPUP = 0.09       # given by the brief: legitimate customers who abandon at OTP/3DS
STEPUP_COST = P_ABANDON_STEPUP * (AVG_LEGIT_MARGIN + P_CHURN_WRONG_BLOCK * CUSTOMER_LTV)

P_STEPUP_STOPS_FRAUD = 0.35   # ASSUMPTION, not measured on this dataset (no step-up completion/
                              # outcome field in IEEE-CIS). OTP/3DS stops a fraud attempt only if
                              # the fraudster lacks the second factor; some do (SIM-swap, phishing),
                              # most don't. Deliberately set BELOW HOLD's implicit ~100% catch rate
                              # (a human reviewer who holds a transaction genuinely stops it) — an
                              # early version of this model credited step-up with the same certainty
                              # as HOLD at a fraction of the cost, which made "step up everyone"
                              # accounting-optimal regardless of score. See sensitivity sweep for how
                              # much this one number moves the recommended threshold.

# ---------------------------------------------------------------------
# Review: a human looks at it.
# ---------------------------------------------------------------------
REVIEW_COST = 60               # given by the brief: ~4 analyst-minutes fully loaded

# ---------------------------------------------------------------------
# Prevented fraud is not fully recovered value: blocked fraudsters retry.
# ---------------------------------------------------------------------
RETRY_RATE = 0.40              # given by the brief: ASSUMPTION, not measured on this dataset


@dataclass(frozen=True)
class CostModel:
    avg_fraud_amount: float = AVG_FRAUD_AMOUNT
    dispute_fee: float = DISPUTE_FEE
    handling: float = HANDLING
    fn_cost: float = FN_COST

    avg_legit_margin: float = AVG_LEGIT_MARGIN
    support_contact: float = SUPPORT_CONTACT
    p_churn_wrong_block: float = P_CHURN_WRONG_BLOCK
    customer_ltv: float = CUSTOMER_LTV
    fp_cost: float = FP_COST

    p_abandon_stepup: float = P_ABANDON_STEPUP
    stepup_cost: float = STEPUP_COST
    p_stepup_stops_fraud: float = P_STEPUP_STOPS_FRAUD

    review_cost: float = REVIEW_COST
    retry_rate: float = RETRY_RATE

    @property
    def fn_fp_ratio(self) -> float:
        return self.fn_cost / self.fp_cost

    def perturbed(self, **overrides: float) -> "CostModel":
        """Return a new CostModel with named base inputs overridden,
        re-deriving the composite costs (fn_cost, fp_cost, stepup_cost)
        from them. Used by the sensitivity sweep in Phase 5."""
        base = {
            "avg_fraud_amount": self.avg_fraud_amount,
            "dispute_fee": self.dispute_fee,
            "handling": self.handling,
            "avg_legit_margin": self.avg_legit_margin,
            "support_contact": self.support_contact,
            "p_churn_wrong_block": self.p_churn_wrong_block,
            "customer_ltv": self.customer_ltv,
            "p_abandon_stepup": self.p_abandon_stepup,
            "p_stepup_stops_fraud": self.p_stepup_stops_fraud,
            "review_cost": self.review_cost,
            "retry_rate": self.retry_rate,
        }
        base.update(overrides)
        fn_cost = base["avg_fraud_amount"] + base["dispute_fee"] + base["handling"]
        fp_cost = base["avg_legit_margin"] + base["support_contact"] + (
            base["p_churn_wrong_block"] * base["customer_ltv"]
        )
        stepup_cost = base["p_abandon_stepup"] * (
            base["avg_legit_margin"] + base["p_churn_wrong_block"] * base["customer_ltv"]
        )
        return CostModel(
            avg_fraud_amount=base["avg_fraud_amount"], dispute_fee=base["dispute_fee"],
            handling=base["handling"], fn_cost=fn_cost,
            avg_legit_margin=base["avg_legit_margin"], support_contact=base["support_contact"],
            p_churn_wrong_block=base["p_churn_wrong_block"], customer_ltv=base["customer_ltv"],
            fp_cost=fp_cost, p_abandon_stepup=base["p_abandon_stepup"], stepup_cost=stepup_cost,
            p_stepup_stops_fraud=base["p_stepup_stops_fraud"],
            review_cost=base["review_cost"], retry_rate=base["retry_rate"],
        )


DEFAULT_COSTS = CostModel()
