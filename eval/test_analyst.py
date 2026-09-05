"""Phase 8 checkpoint: refusal on missing evidence; every number in a
normal narrative traces back to the input payload.

Run: python -m eval.test_analyst
"""
from src.analyst.llm_analyst import generate_analyst_note
from src.analyst.validate import find_ungrounded_numbers

NORMAL_PAYLOAD = {
    "risk_score": 0.91,
    "top_features": [
        {"name": "distinct_cards_per_device", "value": 7, "contribution": 0.31},
        {"name": "velocity_1h_ratio", "value": 6.2, "contribution": 0.24},
        {"name": "amount_zscore", "value": 4.1, "contribution": 0.18},
    ],
    "decision": "HOLD",
}

NULL_PAYLOAD = {
    "risk_score": 0.91,
    "top_features": [
        {"name": "distinct_cards_per_device", "value": None, "contribution": 0.31},
    ],
    "decision": "HOLD",
}


def run() -> bool:
    ok = True

    print("[1/2] Missing-evidence payload -> expect refusal")
    result = generate_analyst_note(NULL_PAYLOAD)
    print(f"  refused={result['refused']}  reason={result.get('reason')}")
    if not result["refused"]:
        print("  FAIL: should have refused")
        ok = False
    else:
        print("  PASS")

    print("\n[2/2] Normal payload -> expect grounded narrative")
    result = generate_analyst_note(NORMAL_PAYLOAD)
    print(f"  refused={result['refused']}  source={result.get('source')}")
    print(f"  narrative:\n{result.get('narrative')}")
    ungrounded = find_ungrounded_numbers(result.get("narrative", ""), NORMAL_PAYLOAD)
    if result["refused"] or ungrounded:
        print(f"  FAIL: refused={result['refused']} ungrounded={ungrounded}")
        ok = False
    else:
        print("  PASS: every number traces to the input payload")

    print("\n" + ("ANALYST TEST: PASS" if ok else "ANALYST TEST: FAIL"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
