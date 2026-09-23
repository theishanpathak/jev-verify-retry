"""Graph state definitions for the jev-verify-retry pipeline.

Defines the Pydantic models that flow through the LangGraph state graph:
per-branch state for the independent test-gen and code-gen branches,
gate results that accumulate across the whole run, and the terminal
outcome tags used to distinguish where in the pipeline a spec finally
succeeded or failed.
"""

import operator
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

Checkpoint = Literal[
    "test_lint",
    "test_semantic",
    "code_lint",
    "execution",
    "code_semantic",
]

Branch = Literal["test", "code"]

TerminalState = Literal[
    "test_lint_exhausted",
    "code_lint_exhausted",
    "test_semantic_exhausted",
    "execution_exhausted",
    "code_semantic_rejected",
    "human_escalated",
    "passed",
]

class EscalationSeverity(str, Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"


class EscalationRecord(BaseModel):
    """Built when a checkpoint exhausts its retry budget -- buckets the failure
    by its worst-seen severity so a human reviewer can triage by urgency
    instead of reading every raw failure individually."""
    checkpoint: Checkpoint
    branch: Branch | None
    spec_description: str
    severity: EscalationSeverity
    bucket: str  # the criterion/issue name that drove the severity -- the "why", for grouping
    attempts_seen: list[str]  # issue per attempt, in order, for the human reviewer's context


class Spec(BaseModel):
    """Spec description plus its resolved interface contract."""
    description: str
    function_name: str
    signature: str
    module_name: str = "solution"

    def as_prompt(self) -> str:
        return f"{self.description}\n\nSignature: {self.signature}\nModule: {self.module_name}"


class BranchState(BaseModel):
    """State for one of the two independent generation branches.

    Used for both branches. `semantic_attempt` increments on every
    semantic check the branch goes through -- multiple times on the
    test branch (retries before freezing), currently just once on the
    code branch, since code_semantic has no retry loop yet.
    """

    content: str = ""
    lint_attempt: int = 0
    semantic_attempt: int = 0
    active_model: str
    frozen: bool = False
    feedback: str | None = None


class GateResult(BaseModel):
    """A single gate/checkpoint outcome, logged for every attempt.

    One of these is appended to `PipelineState.gate_results` every time
    a lint check, semantic check, or execution attempt runs -- including
    failed attempts -- so the full retry history survives for later
    logging and dashboard aggregation, not just the final outcome.
    """

    checkpoint: Checkpoint
    branch: Branch | None  # None for "execution": a merge-point gate, not branch-specific
    attempt: int
    model_used: str
    verifier_used: str | None = None
    passed: bool
    confidence: float | None = None  # populated only at semantic checkpoints
    detail: str = ""

    issue: str | None = None  # verifier's issue category (wrong_logic, incomplete, ...)
    latency_seconds: float | None = None  # verifier call time; None for lint gates
    input_tokens: int | None = None
    output_tokens: int | None = None


class PipelineState(BaseModel):
    """Top-level graph state for one spec's run through the pipeline."""

    spec: Spec
    test_branch: BranchState
    code_branch: BranchState
    execution_attempt: int = 0
    last_traceback: str | None = None
    gate_results: Annotated[list[GateResult], operator.add] = Field(default_factory=list)
    terminal_state: TerminalState | None = None
    escalation: EscalationRecord | None = None

    def final_verdict(self) -> GateResult | None:
        """Return the run's final code_semantic GateResult, if one exists.

        Deliberately not a separate stored field: keeping the final
        verdict as a derived read over `gate_results` means there is
        only one place a verdict is ever written, so a dashboard reading
        this can never see a value that's out of sync with the gate
        history itself.
        """
        for result in reversed(self.gate_results):
            if result.checkpoint == "code_semantic":
                return result
        return None