"""Phase 8 entry point. The model scores, the policy decides — this
module only explains, to an analyst, after both have already happened.

generate_analyst_note(evidence: dict) -> dict
  {"refused": True, "reason": "..."} on missing evidence, or
  {"refused": False, "narrative": str, "source": "llm"|"template", ...}
"""
from src.analyst.generator import generate_narrative
from src.analyst.schema import AnalystEvidence


def generate_analyst_note(evidence: dict) -> dict:
    parsed = AnalystEvidence(
        risk_score=evidence.get("risk_score"),
        top_features=evidence.get("top_features"),
        decision=evidence.get("decision"),
        transaction_id=evidence.get("transaction_id"),
    )
    missing = parsed.missing_fields()
    if missing:
        return {
            "refused": True,
            "reason": f"Missing required evidence: {', '.join(missing)}. "
                      "Case routed to a human analyst without a generated rationale.",
        }

    result = generate_narrative(evidence)
    result["refused"] = False
    return result
