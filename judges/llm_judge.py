"""LLM-as-judge, criteria-based -- reuses the exact same Choice questions as
jev_verify_criteria / jev_verify_code_criteria, so Jev vs LLM-judge is a fair
same-questions comparison instead of checklist vs. single open-ended question.
"""

import json
import time
from collections.abc import Callable

from openai import OpenAI

from config import settings
from judges.base import VerifierResult
from judges.jev_code_criteria import CODE_CRITERIA_ORDER, CODE_CRITERIA_QUESTIONS
from judges.jev_criteria import CRITERIA_ORDER, CRITERIA_QUESTIONS
from state import Spec

client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


def _build_system_prompt(questions: dict) -> str:
    """Build a prompt asking every criterion in one call, reusing Jev's exact
    instruction text so both judges are evaluated on identical wording."""
    lines = [
        "You are a strict, literal code reviewer. Answer each question below "
        "about whether the given content meets the criterion, judged against "
        "the specification. Do not soften or hedge -- answer exactly what is asked.",
    ]
    for name, q in questions.items():
        lines.append(f"- {name}: {q.instructions}")
    lines.append(
        'Respond with JSON only, no explanation: '
        '{"answers": {"<criterion_name>": {"choice": "yes"|"no", "confidence": <float 0-1>}, ...}}'
    )
    return "\n".join(lines)


def _make_verifier(questions: dict, order: list[str]) -> Callable[[str, Spec], VerifierResult]:
    """Factory mirroring jev_verify_criteria's shape: one call, all criteria,
    passed = all yes, issue = first unmet criterion in `order`."""
    system_prompt = _build_system_prompt(questions)
    names = list(questions.keys())

    def verify(content: str, spec: Spec) -> VerifierResult:
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=settings.default_judge_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{spec.as_prompt()}\n\nContent:\n{content}"},
            ],
        )
        elapsed = time.perf_counter() - start

        raw = response.choices[0].message.content or "{}"
        answers = json.loads(raw).get("answers", {})

        results = {name: str(answers.get(name, {}).get("choice", "no")).lower() == "yes" for name in names}
        confidences = {name: float(answers.get(name, {}).get("confidence", 0.0)) for name in names}

        passed = all(results.values())
        issue = next((name for name in order if not results[name]), None) if not passed else "none"
        reported_confidence = confidences[issue] if issue and issue != "none" else min(confidences.values())

        return VerifierResult(
            passed=passed,
            confidence=reported_confidence,
            issue=issue,
            latency_seconds=elapsed,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

    return verify


llm_judge_verify_test_criteria = _make_verifier(CRITERIA_QUESTIONS, CRITERIA_ORDER)
llm_judge_verify_code_criteria = _make_verifier(CODE_CRITERIA_QUESTIONS, CODE_CRITERIA_ORDER)


