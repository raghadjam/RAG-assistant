"""
scripts/verify_env.py

Sanity-checks the project environment before any pipeline work begins:
  1. Confirms core third-party packages import cleanly.
  2. Reports CUDA/GPU availability (warns, does not hard-fail, if absent —
     CPU-only dev boxes are fine for this stage; just slower for embeddings).
  3. Prints package versions so environment drift is easy to spot later.

Run with:
    python scripts/verify_env.py

Exit code 0  -> all required imports succeeded
Exit code 1  -> one or more required imports failed
"""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass, field


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    warning: bool = False


@dataclass
class Report:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    @property
    def has_hard_failures(self) -> bool:
        return any(not r.ok and not r.warning for r in self.results)


REQUIRED_MODULES = [
    "pandas",
    "numpy",
    "sentence_transformers",
    "chromadb",
    "spacy",
    "bs4",       # beautifulsoup4
    "ftfy",
    "langdetect",
]


def check_import(module_name: str) -> CheckResult:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "unknown")
        return CheckResult(name=module_name, ok=True, detail=f"v{version}")
    except ImportError as exc:
        return CheckResult(name=module_name, ok=False, detail=str(exc))


def check_torch_cuda() -> CheckResult:
    try:
        import torch
    except ImportError as exc:
        return CheckResult(
            name="torch",
            ok=False,
            detail=f"torch not importable: {exc}",
        )

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        detail = f"CUDA available — GPU 0: {device_name} (torch {torch.__version__})"
        return CheckResult(name="CUDA/GPU", ok=True, detail=detail)

    detail = (
        f"CUDA NOT available (torch {torch.__version__}). "
        "This project requires a GPU for embedding generation."
    )
    return CheckResult(name="CUDA/GPU", ok=False, detail=detail)


def check_spacy_model() -> CheckResult:
    try:
        import spacy

        try:
            spacy.load("en_core_web_sm")
            return CheckResult(name="spaCy model (en_core_web_sm)", ok=True, detail="loaded")
        except OSError:
            return CheckResult(
                name="spaCy model (en_core_web_sm)",
                ok=False,
                warning=True,
                detail="Not downloaded yet. Run: python -m spacy download en_core_web_sm",
            )
    except ImportError as exc:
        return CheckResult(name="spaCy model (en_core_web_sm)", ok=False, detail=str(exc))


def main() -> int:
    report = Report()

    print("=" * 70)
    print("RAG Assistant — Environment Verification")
    print("=" * 70)
    print(f"Python       : {sys.version.split()[0]} ({platform.platform()})")
    print("-" * 70)

    for module_name in REQUIRED_MODULES:
        result = check_import(module_name)
        report.add(result)

    report.add(check_torch_cuda())
    report.add(check_spacy_model())

    for result in report.results:
        if result.ok and not result.warning:
            status = "OK   "
        elif result.warning:
            status = "WARN "
        else:
            status = "FAIL "
        print(f"[{status}] {result.name:35s} {result.detail}")

    print("-" * 70)
    if report.has_hard_failures:
        print("Environment check FAILED. Install missing packages:")
        print("    pip install -r requirements.txt")
        print("    python -m spacy download en_core_web_sm")
        return 1

    print("Environment check PASSED (see WARN lines above for non-blocking notes).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
