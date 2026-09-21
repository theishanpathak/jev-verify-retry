"""Test-gen node: turns a spec into a candidate pytest suite."""

from openai import OpenAI
from config import settings
from state import PipelineState
from utils import strip_code_fences, build_messages

client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

TESTGEN_SYSTEM_PROMPT = (
    "You are a Python test generator. Given a spec description and an exact "
    "function signature, write pytest tests that import that function from "
    "the given module name and verify the described behavior. Return only "
    "the code, no explanation, no markdown fences."
)


def test_gen_node(state: PipelineState) -> dict[str, object]:
    """Generate tests for `state.spec` and update the test branch."""
    response = client.chat.completions.create(
        model=state.test_branch.active_model,
        messages=build_messages(TESTGEN_SYSTEM_PROMPT, state.spec, state.test_branch),
    )
    tests = strip_code_fences(response.choices[0].message.content or "")

    updated_branch = state.test_branch.model_copy(update={"content": tests})
    return {"test_branch": updated_branch}