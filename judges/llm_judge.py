"""LLM-as-judge verifier -- structured JSON output, same shape as Jev."""

import json
import time

from openai import OpenAI
from pydantic import BaseModel

from config import settings
from judges.base import VerifierResult
from state import Spec

client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

JUDGE_SYSTEM_PROMPT = (
    "You are a code reviewer. Given a specification and content, judge whether "
    "the content correctly and completely satisfies the spec. Respond with JSON only: "
    '{"passed": bool, "confidence": float 0-1, "issue": one of '
    '"none", "wrong_logic", "incomplete", "wrong_interface", "other"}. '
    'Use "none" for issue only when passed is true.'
)


class _JudgeAnswer(BaseModel):
    passed: bool
    confidence: float
    issue: str


def llm_judge_verify(content: str, spec: Spec) -> VerifierResult:
    """Ask the LLM-as-judge model for a structured verdict."""
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=settings.default_code_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"{spec.as_prompt()}\n\nContent:\n{content}"},
        ],
    )
    elapsed = time.perf_counter() - start

    raw = response.choices[0].message.content or "{}"
    answer = _JudgeAnswer.model_validate(json.loads(raw))

    return VerifierResult(
        passed=answer.passed,
        confidence=answer.confidence,
        issue=answer.issue,
        latency_seconds=elapsed,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
    )