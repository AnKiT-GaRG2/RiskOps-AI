"""Extractive-only enforcement: every numeric claim in a generated
narrative must trace to a value actually present in the evidence
payload. Anything that doesn't is grounds for refusal.

Ranks 1-3 ("the top 3 factors", "the first signal") are structural
narration, not data claims, and are allowed unconditionally.
"""
import re

NUMBER_RE = re.compile(r"-?\d+\.?\d*")
STRUCTURAL_RANKS = {1.0, 2.0, 3.0}
REL_TOL = 0.02
ABS_TOL = 0.05


def _allowed_numbers(payload: dict) -> set:
    allowed = set()
    risk_score = payload.get("risk_score")
    if risk_score is not None:
        allowed.add(round(float(risk_score), 6))
        allowed.add(round(float(risk_score) * 100, 6))

    for feat in payload.get("top_features", []):
        value = feat.get("value")
        contribution = feat.get("contribution")
        if value is not None:
            allowed.add(round(float(value), 6))
        if contribution is not None:
            allowed.add(round(float(contribution), 6))
            allowed.add(round(float(contribution) * 100, 6))
    return allowed


def _is_grounded(number: float, allowed: set) -> bool:
    if any(abs(number - a) < ABS_TOL for a in STRUCTURAL_RANKS):
        return True
    for a in allowed:
        if abs(number - a) <= max(ABS_TOL, abs(a) * REL_TOL):
            return True
    return False


def find_ungrounded_numbers(narrative: str, payload: dict) -> list:
    allowed = _allowed_numbers(payload)
    ungrounded = []
    for match in NUMBER_RE.finditer(narrative):
        number = float(match.group())
        if not _is_grounded(number, allowed):
            ungrounded.append(number)
    return ungrounded
