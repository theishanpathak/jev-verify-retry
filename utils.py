import re
from state import BranchState, Spec

_FENCE_RE = re.compile(r"^```(?:\w+)?\n(.*)\n```$", re.DOTALL)

def strip_code_fences(text: str) -> str:
    """Strip a wrapping ```lang ... ``` fence if present."""
    match = _FENCE_RE.match(text.strip())
    return match.group(1) if match else text.strip()

def build_messages(system_prompt: str, spec: Spec, branch: BranchState) -> list[dict[str, str]]:
    """Base prompt, plus the prior attempt and its failure if this is a retry."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": spec.as_prompt()},
    ]
    if branch.feedback:
        messages += [
            {"role": "assistant", "content": branch.content},
            {
                "role": "user",
                "content": f"That attempt failed:\n{branch.feedback}\n\n"
                "Fix it. Return the full corrected code only.",
            },
        ]
    return messages