"""Semantic gate: run one verifier, map its result onto a GateResult."""

from collections.abc import Callable

from judges.base import VerifierResult
from state import Branch, Checkpoint, GateResult, Spec

Verifier = Callable[[str, Spec], VerifierResult]


def semantic_check(
    verifier: Verifier, verifier_name: str, checkpoint: Checkpoint, branch: Branch,
    content: str, spec: Spec, attempt: int, model_used: str,
) -> GateResult:
    """Call `verifier` once and log its verdict. No escalation logic."""
    r = verifier(content, spec)
    return GateResult(
        checkpoint=checkpoint, branch=branch, attempt=attempt,
        model_used=model_used, verifier_used=verifier_name,
        passed=r.passed, confidence=r.confidence, issue=r.issue,
        latency_seconds=r.latency_seconds,
        input_tokens=r.input_tokens, output_tokens=r.output_tokens,
    )