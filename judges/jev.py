"""Jev client -- real TypeSafe System One API, Noul + Choice."""

import time
from typesafe_sdk import Choice, Noul, TypeSafeClient

from config import settings
from state import Spec
from judges.base import VerifierResult

client = TypeSafeClient(api_key=settings.jev_api_key)

JEV_QUESTIONS = {
    "satisfies_spec": Noul(
        instructions="Does the content correctly and completely satisfy the specification?"
    ),
    "issue_category": Choice(
        instructions=(
            "If the content does not satisfy the spec, what's the primary reason? "
            "If it does satisfy the spec, choose 'none'."
        ),
        criteria={
            "none": "The content correctly satisfies the spec.",
            "wrong_logic": "Behavior or logic doesn't match the spec.",
            "incomplete": "Misses part of what the spec asks for.",
            "wrong_interface": "Doesn't match the required name/signature/module.",
            "other": "Some other issue.",
        },
    ),
}


def jev_verify(content: str, spec: Spec) -> VerifierResult:
    """Ask Jev whether content satisfies spec."""
    start = time.perf_counter()
    response = client.system_one(
        state={"specification": spec.as_prompt(), "content": content},
        questions=JEV_QUESTIONS,
    )
    elapsed = time.perf_counter() - start

    noul = response.answers["satisfies_spec"].noul
    return VerifierResult(
        passed=noul >= 0.5,
        confidence=abs(noul - 0.5) * 2,  # distance from the uncertain midpoint
        issue=response.answers["issue_category"].choice,
        latency_seconds=elapsed,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )

if __name__ == "__main__":
    from state import Spec

    spec = Spec(
        description="Return the sum of two integers.",
        function_name="add_two",
        signature="def add_two(a: int, b: int) -> int",
    )
    correct = "def add_two(a: int, b: int) -> int:\n    return a + b\n"
    wrong = "def add_two(a: int, b: int) -> int:\n    return a - b\n"
    edge = "def add_two(a: int, b: str) -> int:\n    return a + int(b)\n"

    for label, code in [("correct", correct), ("wrong", wrong), ("edge", edge)]:
        print(label, jev_verify(code, spec))