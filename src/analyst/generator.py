"""Two narrative backends. The template backend is grounded by
construction (it only ever prints numbers it read from the payload) and
is always available — the demo must work with no API key configured.
The LLM backend is used when ANTHROPIC_API_KEY is set, and its output
still goes through validate.find_ungrounded_numbers before it's trusted.
"""
import os


def generate_template_narrative(payload: dict) -> str:
    risk_score = payload["risk_score"]
    decision = payload["decision"]
    features = payload["top_features"][:3]

    lines = [
        f"Risk score {risk_score:.2f} ({risk_score * 100:.0f}%) led to decision: {decision}.",
        "Top contributing signals:",
    ]
    for i, feat in enumerate(features, start=1):
        lines.append(
            f"  {i}. {feat['name']} = {feat['value']} (contribution {feat['contribution']:.2f}, "
            f"{feat['contribution'] * 100:.0f}% of the score)"
        )
    return "\n".join(lines)


def _try_llm_narrative(payload: dict) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    system_prompt = (
        "You write a two-sentence internal fraud-analyst note from a JSON evidence "
        "payload. Rules: cite ONLY numbers that appear in the payload (risk_score, "
        "feature values, contributions, or their plain percentage form). Never invent "
        "a number. Never mention thresholds, policy, or raw transaction data — you "
        "were not given them. Do not decide anything; the decision field is already final."
    )
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=200,
            system=system_prompt,
            messages=[{"role": "user", "content": str(payload)}],
        )
        return response.content[0].text
    except Exception:
        return None


def generate_narrative(payload: dict) -> dict:
    """Returns {"narrative": str, "source": "llm"|"template", "llm_rejected": bool}."""
    llm_output = _try_llm_narrative(payload)
    if llm_output is None:
        return {"narrative": generate_template_narrative(payload), "source": "template", "llm_rejected": False}

    from src.analyst.validate import find_ungrounded_numbers
    ungrounded = find_ungrounded_numbers(llm_output, payload)
    if ungrounded:
        return {
            "narrative": generate_template_narrative(payload),
            "source": "template", "llm_rejected": True, "rejected_numbers": ungrounded,
        }
    return {"narrative": llm_output, "source": "llm", "llm_rejected": False}
