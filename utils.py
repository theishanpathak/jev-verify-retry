import re

_FENCE_RE = re.compile(r"^```(?:\w+)?\n(.*)\n```$", re.DOTALL)

def strip_code_fences(text: str) -> str:
    """Strip a wrapping ```lang ... ``` fence if present."""
    match = _FENCE_RE.match(text.strip())
    return match.group(1) if match else text.strip()