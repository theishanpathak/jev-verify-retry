"""Shared verifier result shape, so any verifier is directly comparable."""

from pydantic import BaseModel


class VerifierResult(BaseModel):
    """Uniform result from any verifier."""

    passed: bool
    confidence: float | None = None
    issue: str | None = None
    latency_seconds: float
    input_tokens: int
    output_tokens: int