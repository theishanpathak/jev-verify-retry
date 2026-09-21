"""Jev client -- real TypeSafe System One API, Noul + Choice."""

from typesafe_sdk import Choice, Noul, TypeSafeClient

from config import settings
from state import Spec

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


def jev_verify(content: str, spec: Spec) -> tuple[bool, float, str]:
    """Ask Jev whether content satisfies spec. Returns (passed, confidence, issue_category)."""
    response = client.system_one(
        state={"specification": spec.as_prompt(), "content": content},
        questions=JEV_QUESTIONS,
    )
    noul = response.answers["satisfies_spec"].noul
    passed = noul >= 0.5
    confidence = abs(noul - 0.5) * 2  # distance from the uncertain midpoint
    issue = response.answers["issue_category"].choice
    return passed, confidence, issue
