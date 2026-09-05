"""The three-band decision rule. Deterministic, zero ML, zero LLM.

The middle band is not "medium risk" — it's an abstention mechanism:
the system declines to decide and asks for stronger evidence (OTP/3DS)
instead of guessing under uncertainty.
"""
import numpy as np

ALLOW, STEPUP, HOLD = "ALLOW", "STEPUP", "HOLD"


def decide(scores, t_low: float, t_high: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    decisions = np.full(scores.shape, ALLOW, dtype=object)
    decisions[(scores >= t_low) & (scores < t_high)] = STEPUP
    decisions[scores >= t_high] = HOLD
    return decisions
