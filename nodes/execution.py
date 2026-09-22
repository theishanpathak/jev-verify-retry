"""Execution sandbox: run frozen code against frozen tests via a pytest subprocess."""

import subprocess
import sys
import tempfile
from pathlib import Path

from state import GateResult, PipelineState, Spec


def run_execution(spec: Spec, code: str, tests: str, timeout: int = 10) -> tuple[bool, str]:
    """Write code+tests into a fresh temp dir and run pytest against them.
    Returns (passed, output) -- output is the captured stdout+stderr on failure,
    empty string on pass.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / f"{spec.module_name}.py").write_text(code)
        test_file = tmp_path / "test_generated.py"
        test_file.write_text(tests)

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
                cwd=tmp_path, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, f"execution timed out after {timeout}s (possible infinite loop)"

        passed = proc.returncode == 0
        return passed, "" if passed else (proc.stdout + proc.stderr)


def execution_node(state: PipelineState) -> dict[str, object]:
    """LangGraph node: run frozen code against frozen tests, log the result."""
    passed, output = run_execution(state.spec, state.code_branch.content, state.test_branch.content)
    attempt = state.execution_attempt + 1

    result = GateResult(
        checkpoint="execution", branch=None, attempt=attempt,
        model_used=state.code_branch.active_model, passed=passed,
        detail=output[:2000],  # cap what lands in the log; full text still goes to last_traceback
    )
    updated_code_branch = state.code_branch.model_copy(update={"frozen": passed})

    return {
        "code_branch": updated_code_branch,
        "execution_attempt": attempt,
        "last_traceback": None if passed else output,
        "gate_results": [result],
    }