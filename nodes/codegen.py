"""Code-gen node: turns a spec into a candidate implementation."""

from openai import OpenAI
from config import settings
from state import PipelineState
from utils import strip_code_fences

client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

CODEGEN_SYSTEM_PROMPT = (
    "You are a Python code generator. Given a specification, write a single, "
    "complete Python implementation that satisfies it. Return only the code, "
    "no explanation, no markdown fences."
)


def codegen_node(state: PipelineState) -> dict[str, object]:
    """Generate an implementation for `state.spec` and update the code branch.

    Returns a partial state update: `code_branch.content` is overwritten
    with the freshly generated code. `code_branch.lint_attempt` is left
    untouched here -- that counter belongs to the lint node's retry
    bookkeeping, not to generation.
    """
    response = client.chat.completions.create(
        model=state.code_branch.active_model,
        messages=[
            {"role": "system", "content": CODEGEN_SYSTEM_PROMPT},
            {"role": "user", "content": state.spec.as_prompt()},
        ],
    )
    code = strip_code_fences(response.choices[0].message.content or "")

    updated_branch = state.code_branch.model_copy(update={"content": code})
    return {"code_branch": updated_branch}