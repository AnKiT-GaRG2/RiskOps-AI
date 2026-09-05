"""Runs every deterministic/fast check in one shot: the leakage test,
the policy tests, and the analyst refusal/grounding test. Does NOT
train the model or run the full sweep (those are separate, slower
pipeline steps — see README.md).

Run: python -m eval.run_all
"""
import sys

from eval import test_analyst, test_leakage, test_policy


def run() -> bool:
    results = {}
    print("\n" + "#" * 72)
    print("# LEAKAGE TEST")
    print("#" * 72)
    results["leakage"] = test_leakage.run()

    print("\n" + "#" * 72)
    print("# POLICY TESTS")
    print("#" * 72)
    results["policy"] = test_policy.run()

    print("\n" + "#" * 72)
    print("# ANALYST TEST")
    print("#" * 72)
    results["analyst"] = test_analyst.run()

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    ok = True
    for name, passed in results.items():
        print(f"  {name:10s}: {'PASS' if passed else 'FAIL'}")
        ok = ok and passed
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
