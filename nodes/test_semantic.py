"""Test-semantic gate: judges candidate tests against the spec."""

from collections.abc import Callable

from judges.base import VerifierResult
from nodes.semantic_check import semantic_check
from state import PipelineState, Spec

ISSUE_HINTS = {
    "matches_interface": "the test doesn't call the function using the exact name/signature/module from the spec",
    "has_real_assertions": "the tests don't actually check return values, or use trivial assertions",
    "covers_normal_case": "the tests don't include a typical, non-edge-case input",
    "covers_edge_cases": "the tests don't cover boundary conditions implied by the spec",
    "covers_adversarial_input": "the tests don't probe unexpected or malformed input",
    "deterministic": "the tests rely on randomness, time, or external state that could make them flaky",
}

Verifier = Callable[[str, Spec], VerifierResult]


def make_test_semantic_node(verifier: Verifier, verifier_name: str) -> Callable[[PipelineState], dict[str, object]]:
    """Build a test_semantic node bound to one verifier -- this is the swap point
    for running the same pipeline once per judge in the two-way comparison."""

    def test_semantic_node(state: PipelineState) -> dict[str, object]:
        branch = state.test_branch
        result = semantic_check(
            verifier=verifier, verifier_name=verifier_name,
            checkpoint="test_semantic", branch="test",
            content=branch.content, spec=state.spec,
            attempt=branch.semantic_attempt + 1, model_used=branch.active_model,
        )
        issue = result.issue if result.issue and result.issue != "none" else None
        updated_branch = branch.model_copy(update={
            "semantic_attempt": result.attempt,
            "feedback": None if result.passed else (
                f"Semantic check failed: {ISSUE_HINTS.get(issue, 'the verifier rejected this without a specific reason')}"
            ),
            "frozen": result.passed,
        })
        return {"test_branch": updated_branch, "gate_results": [result]}

    return test_semantic_node