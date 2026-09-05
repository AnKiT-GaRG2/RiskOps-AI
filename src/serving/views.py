"""Phase 7 dual view, enforced at the data layer rather than the
template: customer_view()'s function signature physically cannot accept
attributions, feature values, thresholds, or model version, so the
detailed payload cannot reach a customer surface even by a rendering
mistake. If a screen needs to show something to a customer, it must
call customer_view(); nothing else in this module returns a dict a
customer template could accidentally over-render.
"""
from dataclasses import dataclass

CUSTOMER_MESSAGE = "This transaction needs additional verification."


@dataclass(frozen=True)
class CustomerView:
    status: str    # "approved" | "additional_verification_required"
    message: str   # None when approved


@dataclass(frozen=True)
class AnalystView:
    transaction_id: str
    model_version: str
    raw_score: float
    calibrated_score: float
    band: str
    t_low: float
    t_high: float
    top_attributions: list


def customer_view(band: str) -> CustomerView:
    """The ONLY thing a customer-facing surface may call. Takes a bare
    band string — there is no code path by which it could see scores,
    attributions, or thresholds, because they are never passed in."""
    if band == "ALLOW":
        return CustomerView(status="approved", message=None)
    return CustomerView(status="additional_verification_required", message=CUSTOMER_MESSAGE)


def analyst_view(decision_record: dict) -> AnalystView:
    """Internal only. Full detail for the fraud analyst / audit UI."""
    return AnalystView(
        transaction_id=decision_record["transaction_id"],
        model_version=decision_record["model_version"],
        raw_score=decision_record["raw_score"],
        calibrated_score=decision_record["calibrated_score"],
        band=decision_record["band"],
        t_low=decision_record["t_low"],
        t_high=decision_record["t_high"],
        top_attributions=decision_record.get("top_attributions", []),
    )
