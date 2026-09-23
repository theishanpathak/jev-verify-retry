"""LangGraph port of main.py -- same pipeline, nodes, and retry logic, as a StateGraph."""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from config import settings
from judges.jev_code_criteria import CODE_CRITERIA_SEVERITY, jev_verify_code_criteria
from judges.jev_criteria import CRITERIA_SEVERITY, jev_verify_criteria
from judges.llm_judge import llm_judge_verify_code_criteria, llm_judge_verify_test_criteria
from nodes.codegen import codegen_node
from nodes.code_semantic import make_code_semantic_node
from nodes.execution import execution_node
from nodes.lint_check import code_lint_node, test_lint_node
from nodes.test_semantic import make_test_semantic_node
from nodes.testgen import test_gen_node
from state import BranchState, EscalationRecord, EscalationSeverity, PipelineState, Spec

SEVERITY_ORDER = ["critical", "high", "medium", "low"]

Judge = Literal["jev", "llm"]


def worst_severity(issues_seen: list[str]) -> tuple[EscalationSeverity, str]:
    ranked = sorted(
        ((CRITERIA_SEVERITY.get(i, "low"), i)
         for i in issues_seen if i and i != "none"),
        key=lambda pair: SEVERITY_ORDER.index(pair[0]),
    )
    return (EscalationSeverity(ranked[0][0]), ranked[0][1]) if ranked else (EscalationSeverity.low, "unknown")


def reset_test_lint(state: PipelineState) -> dict[str, object]:
    return {"test_branch": state.test_branch.model_copy(update={"lint_attempt": 0})}


def reset_code_lint(state: PipelineState) -> dict[str, object]:
    return {
        "code_branch": state.code_branch.model_copy(update={
            "lint_attempt": 0, "feedback": state.last_traceback,
        })
    }


def mark_test_lint_exhausted(state: PipelineState) -> dict[str, object]:
    return {"terminal_state": "test_lint_exhausted"}


def mark_code_lint_exhausted(state: PipelineState) -> dict[str, object]:
    return {"terminal_state": "code_lint_exhausted"}


def mark_execution_exhausted(state: PipelineState) -> dict[str, object]:
    return {"terminal_state": "execution_exhausted"}


def mark_test_semantic_exhausted(state: PipelineState) -> dict[str, object]:
    issues = [r.issue for r in state.gate_results if r.checkpoint == "test_semantic"]
    severity, bucket = worst_severity(issues)
    escalation = EscalationRecord(
        checkpoint="test_semantic", branch="test",
        spec_description=state.spec.description,
        severity=severity, bucket=bucket, attempts_seen=issues,
    )
    print("ESCALATED:", escalation)
    return {"terminal_state": "test_semantic_exhausted"}


def mark_passed(state: PipelineState) -> dict[str, object]:
    return {"terminal_state": "passed"}


def mark_code_semantic_rejected(state: PipelineState) -> dict[str, object]:
    result = state.gate_results[-1]
    severity = CODE_CRITERIA_SEVERITY.get(result.issue, "low")
    escalation = EscalationRecord(
        checkpoint="code_semantic", branch="code",
        spec_description=state.spec.description,
        severity=EscalationSeverity(severity),
        bucket=result.issue, attempts_seen=[result.issue],
    )
    print("ESCALATED:", escalation)
    return {"terminal_state": "code_semantic_rejected"}


def route_test_lint(state: PipelineState) -> str:
    if state.test_branch.feedback is None:
        return "semantic"
    if state.test_branch.lint_attempt >= settings.max_lint_attempts:
        return "exhausted"
    return "retry"


def route_test_semantic(state: PipelineState) -> str:
    if state.test_branch.frozen:
        return "freeze"
    if state.test_branch.semantic_attempt >= settings.max_semantic_attempts:
        return "exhausted"
    return "retry"


def route_code_lint(state: PipelineState) -> str:
    if state.code_branch.feedback is None:
        return "next"
    if state.code_branch.lint_attempt >= settings.max_lint_attempts:
        return "exhausted"
    return "retry"


def route_execution(state: PipelineState) -> str:
    if state.code_branch.frozen:
        return "code_semantic"
    if state.execution_attempt >= settings.max_execution_attempts:
        return "exhausted"
    return "retry"


def route_code_semantic(state: PipelineState) -> str:
    return "passed" if state.gate_results[-1].passed else "rejected"


def build_graph(judge: Judge = "jev"):
    """Build the pipeline graph, bound to either judge. The judge choice is
    fixed at graph-construction time, not per-invoke -- LangGraph compiles
    node functions into the graph structure once, so switching judges means
    building (and compiling) a new graph, not passing an argument to invoke().
    """
    if judge == "jev":
        test_semantic_node = make_test_semantic_node(
            jev_verify_criteria, "jev-criteria")
        code_semantic_node = make_code_semantic_node(
            jev_verify_code_criteria, "jev-criteria")
    elif judge == "llm":
        test_semantic_node = make_test_semantic_node(
            llm_judge_verify_test_criteria, settings.default_judge_model)
        code_semantic_node = make_code_semantic_node(
            llm_judge_verify_code_criteria, settings.default_judge_model)
    else:
        raise ValueError(f"unknown judge: {judge!r} (expected 'jev' or 'llm')")

    builder = StateGraph(PipelineState)

    builder.add_node("test_gen", test_gen_node)
    builder.add_node("test_lint", test_lint_node)
    builder.add_node("test_semantic", test_semantic_node)
    builder.add_node("reset_test_lint", reset_test_lint)
    builder.add_node("codegen", codegen_node)
    builder.add_node("code_lint", code_lint_node)
    builder.add_node("execution", execution_node)
    builder.add_node("reset_code_lint", reset_code_lint)
    builder.add_node("code_semantic", code_semantic_node)

    builder.add_node("mark_test_lint_exhausted", mark_test_lint_exhausted)
    builder.add_node("mark_test_semantic_exhausted",
                     mark_test_semantic_exhausted)
    builder.add_node("mark_code_lint_exhausted", mark_code_lint_exhausted)
    builder.add_node("mark_execution_exhausted", mark_execution_exhausted)
    builder.add_node("mark_passed", mark_passed)
    builder.add_node("mark_code_semantic_rejected",
                     mark_code_semantic_rejected)

    builder.add_edge(START, "test_gen")
    builder.add_edge("test_gen", "test_lint")
    builder.add_conditional_edges("test_lint", route_test_lint, {
        "semantic": "test_semantic",
        "retry": "test_gen",
        "exhausted": "mark_test_lint_exhausted",
    })
    builder.add_conditional_edges("test_semantic", route_test_semantic, {
        "freeze": "codegen",
        "retry": "reset_test_lint",
        "exhausted": "mark_test_semantic_exhausted",
    })
    builder.add_edge("reset_test_lint", "test_gen")

    builder.add_edge("codegen", "code_lint")
    builder.add_conditional_edges("code_lint", route_code_lint, {
        "next": "execution",
        "retry": "codegen",
        "exhausted": "mark_code_lint_exhausted",
    })
    builder.add_conditional_edges("execution", route_execution, {
        "code_semantic": "code_semantic",
        "retry": "reset_code_lint",
        "exhausted": "mark_execution_exhausted",
    })
    builder.add_edge("reset_code_lint", "codegen")

    builder.add_conditional_edges("code_semantic", route_code_semantic, {
        "passed": "mark_passed",
        "rejected": "mark_code_semantic_rejected",
    })

    for terminal_node in [
        "mark_test_lint_exhausted", "mark_test_semantic_exhausted",
        "mark_code_lint_exhausted", "mark_execution_exhausted",
        "mark_passed", "mark_code_semantic_rejected",
    ]:
        builder.add_edge(terminal_node, END)

    return builder.compile()