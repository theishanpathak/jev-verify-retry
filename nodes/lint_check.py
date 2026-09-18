"""Lint gate: syntax + ruff check, shared by both branches."""

import ast
import json
import subprocess
import tempfile
from pathlib import Path

from state import Branch, BranchState, GateResult, PipelineState

PROJECT_ROOT = Path(__file__).parent


def lint_check(branch_name: Branch, branch: BranchState) -> tuple[GateResult, str]:
    """ast.parse + ruff --fix on branch.content. Returns (result, fixed_content)."""
    checkpoint = "test_lint" if branch_name == "test" else "code_lint"

    try:
        ast.parse(branch.content)
    except SyntaxError as e:
        result = GateResult(
            checkpoint=checkpoint, branch=branch_name, attempt=branch.lint_attempt + 1,
            model_used=branch.active_model, passed=False, detail=f"syntax error: {e}",
        )
        return result, branch.content

    with tempfile.NamedTemporaryFile(dir=PROJECT_ROOT, suffix=".py", mode="w", delete=False) as f:
        f.write(branch.content)
        tmp_path = Path(f.name)

    try:
        proc = subprocess.run(
            ["ruff", "check", "--fix", str(tmp_path), "--output-format=json"],
            capture_output=True, text=True,
        )
        issues = json.loads(proc.stdout or "[]")
        fixed_content = tmp_path.read_text()
    finally:
        tmp_path.unlink()

    result = GateResult(
        checkpoint=checkpoint, branch=branch_name, attempt=branch.lint_attempt + 1,
        model_used=branch.active_model, passed=not issues,
        detail="; ".join(
            f"{i['code']}: {i['message']}" for i in issues) if issues else "",
    )
    return result, fixed_content


def _lint_node(state: PipelineState, branch_name: Branch) -> dict[str, object]:
    branch = state.test_branch if branch_name == "test" else state.code_branch
    result, fixed_content = lint_check(branch_name, branch)

    updated_branch = branch.model_copy(update={"lint_attempt": result.attempt, "content": fixed_content})
    field = "test_branch" if branch_name == "test" else "code_branch"
    return {field: updated_branch, "gate_results": [result]}


def test_lint_node(state: PipelineState) -> dict[str, object]:
    """LangGraph node: lint the test branch."""
    return _lint_node(state, "test")


def code_lint_node(state: PipelineState) -> dict[str, object]:
    """LangGraph node: lint the code branch."""
    return _lint_node(state, "code")

