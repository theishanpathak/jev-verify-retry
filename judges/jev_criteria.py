"""Jev criteria-based verifier -- closed, checkable Choice questions."""

import time

from typesafe_sdk import Choice, TypeSafeClient

from config import settings
from judges.base import VerifierResult
from state import Spec

client = TypeSafeClient(api_key=settings.jev_api_key)

YES_NO = {"yes": "Criterion is met.", "no": "Criterion is not met."}

CRITERIA_QUESTIONS = {
    "matches_interface": Choice(
        instructions=(
            "Does the test content import and call the function using the exact "
            "name, module, and argument pattern given in the specification's signature?"
        ),
        criteria=YES_NO,
    ),
    "has_real_assertions": Choice(
        instructions=(
            "Do the tests make actual assertions that check the function's return "
            "value against an expected result, rather than just calling it, and "
            "avoid trivial/tautological checks like `assert True`?"
        ),
        criteria=YES_NO,
    ),
    "covers_normal_case": Choice(
        instructions=(
            "Does the test content include at least one test using typical, "
            "non-edge-case input for the function described in the spec?"
        ),
        criteria=YES_NO,
    ),
    "covers_edge_cases": Choice(
        instructions=(
            "Does the test content test boundary conditions implied by the "
            "function's parameter types or the spec's description (e.g. empty "
            "input, None, a single element, or a boundary value)?"
        ),
        criteria=YES_NO,
    ),
    "covers_adversarial_input": Choice(
        instructions=(
            "Does the test suite cover boundary or unusual inputs that are still "
            "valid under the function's declared parameter types (e.g. zero, "
            "negative numbers, very large or very small values, empty collections, "
            "or a single-element case) -- NOT inputs of the wrong type, since "
            "the function is not required to validate its callers' types."
        ),
        criteria=YES_NO,
    ),
    "deterministic": Choice(
        instructions=(
            "Do the tests avoid relying on unseeded randomness, wall-clock time, "
            "network access, or filesystem state in a way that could make them "
            "flaky or non-reproducible?"
        ),
        criteria=YES_NO,
    ),
}

# Severity if this criterion is the one that fails, used for escalation bucketing.
# "critical"/"high" block trust outright; "medium"/"low" are coverage gaps, not correctness risks.
CRITERIA_SEVERITY = {
    "matches_interface": "critical",
    "has_real_assertions": "critical",
    "deterministic": "high",
    "covers_normal_case": "high",
    "covers_adversarial_input": "medium",
    "covers_edge_cases": "medium",
}

CRITERIA_ORDER = [
    "matches_interface", "has_real_assertions", "deterministic",
    "covers_normal_case", "covers_adversarial_input", "covers_edge_cases",
]

CONFIDENCE_FLOOR = 0.6  # below this, even a "yes" answer is flagged as worth improving


def jev_verify_criteria(content: str, spec: Spec) -> VerifierResult:
    """Ask Jev a checklist of closed yes/no questions instead of one open
    'does this satisfy the spec' question. passed = all criteria met AND
    every criterion cleared CONFIDENCE_FLOOR. issue = the first unmet
    criterion, or the weakest-confidence one if all technically passed
    but one was borderline.
    """
    start = time.perf_counter()
    response = client.system_one(
        state={"specification": spec.as_prompt(), "content": content},
        questions=CRITERIA_QUESTIONS,
    )
    elapsed = time.perf_counter() - start

    results = {name: response.answers[name].choice == "yes" for name in CRITERIA_QUESTIONS}
    confidences = {name: response.answers[name].confidence for name in CRITERIA_QUESTIONS}

    all_yes = all(results.values())
    weakest = min(confidences, key=confidences.get)
    passed = all_yes and confidences[weakest] >= CONFIDENCE_FLOOR

    if not all_yes:
        issue = next(name for name in CRITERIA_ORDER if not results[name])
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