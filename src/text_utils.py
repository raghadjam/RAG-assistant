"""Small text helpers for the evaluation scripts.

The corpus still contains LaTeX from the arXiv abstracts (for example 50\\%,
\\emph{converts} and $10 \\times$). A person asking a question would not type that,
so the evaluation removes it from the queries it builds. The stored corpus is not changed.
"""

from __future__ import annotations

import re

MATH_RE = re.compile(r"\$\$.*?\$\$|\$.*?\$", re.DOTALL)
WRAPPED_COMMAND_RE = re.compile(r"\\[A-Za-z]+\*?\{([^{}]*)\}")     # \emph{word} -> word
BARE_COMMAND_RE = re.compile(r"\\[A-Za-z]+\*?")                     # \times -> (nothing)


def strip_latex(text: str) -> str:
    """Remove math, LaTeX commands and braces, keeping the words inside commands."""
    if not text:
        return ""
    text = MATH_RE.sub(" ", text)
    text = text.replace("\\%", "%")
    for _ in range(3):                      # a few passes handle nested commands
        text = WRAPPED_COMMAND_RE.sub(r"\1", text)
    text = BARE_COMMAND_RE.sub(" ", text)
    text = text.replace("\\", " ").replace("{", " ").replace("}", " ")
    return re.sub(r"[ \t]+", " ", text).strip()
