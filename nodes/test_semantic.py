"""Test-semantic gate: judges candidate tests against the spec."""

from collections.abc import Callable

from judges.base import VerifierResult
from nodes.semantic_check import semantic_check
from state import PipelineState, Spec

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
        updated_branch = branch.model_copy(update={
            "semantic_attempt": result.attempt,
            "feedback": None if result.passed else (
                result.issue if result.issue and result.issue != "none" else "semantic check failed"
            ),
            "frozen": result.passed,
        })
        return {"test_branch": updated_branch, "gate_results": [result]}

    return test_semantic_node
