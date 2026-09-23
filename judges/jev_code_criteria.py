"""Jev criteria-based verifier for generated code -- closed, checkable
Choice questions, judging code against spec post-execution."""

import time

from typesafe_sdk import Choice, TypeSafeClient

from config import settings
from judges.base import VerifierResult
from state import Spec

client = TypeSafeClient(api_key=settings.jev_api_key)

YES_NO = {"yes": "Criterion is met.", "no": "Criterion is not met."}

CODE_CRITERIA_QUESTIONS = {
    "matches_interface": Choice(
        instructions=(
            "Does the code define the function using the exact name, module, "
            "and signature (parameter names/types, return type) given in the spec?"
        ),
        criteria=YES_NO,
    ),
    "implements_described_behavior": Choice(
        instructions=(
            "Does the code's logic actually implement what the spec's description "
            "asks for -- not just pass some tests narrowly, but genuinely do what "
            "is described, in a way that would generalize to inputs beyond any "
            "specific test case?"
        ),
        criteria=YES_NO,
    ),
    "avoids_unsafe_patterns": Choice(
        instructions=(
            "Does the code avoid unsafe patterns: eval/exec on non-constant input, "
            "subprocess calls with shell=True, hardcoded credentials or secrets, "
            "unsafe deserialization (e.g. pickle.loads on untrusted data), or "
            "string-concatenated SQL/command construction from input?"
        ),
        criteria=YES_NO,
    ),
    "handles_valid_domain_edge_cases": Choice(
        instructions=(
            "Does the code correctly handle boundary values that are still valid "
            "under the function's declared parameter types (e.g. zero, empty "
            "input, a single element, very large or very small values) -- NOT "
            "inputs of the wrong type, since the function has no obligation to "
            "validate caller types. If the valid domain has no meaningful edge "
            "cases beyond normal input, answer 'yes'."
        ),
        criteria=YES_NO,
    ),
}

# Severity if this criterion is the one that fails, used for escalation bucketing.
CODE_CRITERIA_SEVERITY = {
    "matches_interface": "critical",
    "implements_described_behavior": "critical",
    "avoids_unsafe_patterns": "critical",  # security issues are always critical, regardless of order
    "handles_valid_domain_edge_cases": "medium",
}

# order matters: most fundamental checks first, so `issue` reflects the most
# important thing wrong, not just the first thing checked
CODE_CRITERIA_ORDER = [
    "matches_interface", "avoids_unsafe_patterns",
    "implements_described_behavior", "handles_valid_domain_edge_cases",
]

CONFIDENCE_FLOOR = 0.6

def jev_verify_code_criteria(content: str, spec: Spec) -> VerifierResult:
    """Ask Jev a checklist of closed yes/no questions about generated code.
    passed = all criteria met AND every criterion cleared CONFIDENCE_FLOOR.
    issue = the first unmet criterion, or the weakest-confidence one if all
    technically passed but one was borderline.
    """
    start = time.perf_counter()
    response = client.system_one(
        state={"specification": spec.as_prompt(), "content": content},
        questions=CODE_CRITERIA_QUESTIONS,
    )
    elapsed = time.perf_counter() - start

    results = {name: response.answers[name].choice == "yes" for name in CODE_CRITERIA_QUESTIONS}
    confidences = {name: response.answers[name].confidence for name in CODE_CRITERIA_QUESTIONS}

    all_yes = all(results.values())
    weakest = min(confidences, key=confidences.get)
    passed = all_yes and confidences[weakest] >= CONFIDENCE_FLOOR

    if not all_yes:
        issue = next(name for name in CODE_CRITERIA_ORDER if not results[name])
    elif not passed:
        issue = weakest
    else:
        issue = "none"

    reported_confidence = confidences[issue] if issue != "none" else min(confidences.values())

    return VerifierResult(
        passed=passed,
        confidence=reported_confidence,
        issue=issue,
        latency_seconds=elapsed,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


if __name__ == "__main__":
    from state import Spec

    spec = Spec(
        description="Return the sum of two integers.",
        function_name="add_two",
        signature="def add_two(a: int, b: int) -> int",
    )

    correct = "def add_two(a: int, b: int) -> int:\n    return a + b\n"

    narrow_hack = (
        "def add_two(a: int, b: int) -> int:\n"
        "    if a == 2 and b == 3:\n"
        "        return 5\n"
        "    return a + b + 1\n"
    )

    unsafe = "def add_two(a: int, b: int) -> int:\n    return eval(f'{a} + {b}')\n"

    for label, code in [("correct", correct), ("narrow_hack", narrow_hack), ("unsafe", unsafe)]:
        print(f"--- {label} ---")
        print(jev_verify_code_criteria(code, spec))
        print()