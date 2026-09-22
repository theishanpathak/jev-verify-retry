"""Code-semantic gate: judges generated code against the spec, post-execution."""

from collections.abc import Callable

from judges.base import VerifierResult
from nodes.semantic_check import semantic_check
from state import PipelineState, Spec

ISSUE_HINTS = {
    "matches_interface": "the code doesn't define the function with the exact name/signature/module from the spec",
    "avoids_unsafe_patterns": "the code uses an unsafe pattern (eval/exec, shell=True, hardcoded secrets, unsafe deserialization, or string-built SQL/commands)",
    "implements_described_behavior": "the code's logic doesn't genuinely implement the spec's described behavior -- it may only work for specific cases rather than generally",
    "handles_valid_domain_edge_cases": "the code doesn't correctly handle boundary values within its valid input domain",
}

Verifier = Callable[[str, Spec], VerifierResult]


def make_code_semantic_node(verifier: Verifier, verifier_name: str) -> Callable[[PipelineState], dict[str, object]]:
    """Build a code_semantic node bound to one verifier -- the swap point for
    running the same pipeline once per judge in the two-way comparison.
    Runs only after execution has already passed."""

    def code_semantic_node(state: PipelineState) -> dict[str, object]:
        branch = state.code_branch
        result = semantic_check(
            verifier=verifier, verifier_name=verifier_name,
            checkpoint="code_semantic", branch="code",
            content=branch.content, spec=state.spec,
            attempt=branch.semantic_attempt + 1, model_used=branch.active_model,
        )
        issue = result.issue if result.issue and result.issue != "none" else None
        updated_branch = branch.model_copy(update={
            "semantic_attempt": result.attempt,
            "feedback": None if result.passed else (
                f"Semantic check failed: {ISSUE_HINTS.get(issue, 'the verifier rejected this without a specific reason')}"
            ),
        })
        return {"code_branch": updated_branch, "gate_results": [result]}

    return code_semantic_node