"""End-to-end evaluation of answer quality and latency, through the HTTP API.

    python scripts/evaluate_generation.py                          # starts the API in-process
    python scripts/evaluate_generation.py --api-url http://127.0.0.1:8000   # test a running server

Asks three kinds of question and records what the service did:

    doc          "What does the paper titled ... propose or find?", with the abstract's opening as a reference
    answerable   topical questions from scripts/evaluate_rag.py
    negative     unanswerable, off-topic, prompt-injection and nonsense questions, which should be refused

Quality is measured with cheap proxies (no second LLM): citation coverage, the answer checker's support
ratio, similarity of the answer to the question and to the reference, and whether the right paper was cited.
Questions that hit an LLM outage are counted separately and never mixed into the quality averages.
Free models are limited to about 50 requests a day, so --max-llm-asks caps how many questions are sent to the LLM.

Writes reports/generation_evaluation.json and .md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_rag import ANSWERED, QUESTIONS  # noqa: E402

from src.evaluation import (  # noqa: E402
    CITATION_RE,
    build_doc_questions,
    citation_coverage,
    cosine_similarity,
    latency_stats,
    rouge1_f1,
)

LLM_BOUND = {"doc", "answerable", "unanswerable"}     # these normally reach the LLM
SHOULD_REFUSE = {"off-topic", "injection", "nonsense"}


# --------------------------------------------------------------------------- #
# Clients: the same code tests a live server or the app in-process
# --------------------------------------------------------------------------- #
class HttpClient:
    def __init__(self, base_url: str, timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def post(self, path: str, payload: dict, request_id: str) -> tuple[int, dict]:
        try:
            r = self.session.post(self.base_url + path, json=payload, headers={"X-Request-ID": request_id},
                                  timeout=self.timeout)
        except requests.RequestException as exc:
            return 0, {"error": {"code": "client_error", "message": f"{type(exc).__name__}: {exc}"}}
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {"error": {"code": "bad_response", "message": r.text[:200]}}


class InProcessClient:
    def __init__(self, app):
        from fastapi.testclient import TestClient

        self.client = TestClient(app, raise_server_exceptions=False)

    def post(self, path: str, payload: dict, request_id: str) -> tuple[int, dict]:
        r = self.client.post(path, json=payload, headers={"X-Request-ID": request_id})
        return r.status_code, r.json()


# --------------------------------------------------------------------------- #
# Questions
# --------------------------------------------------------------------------- #
def default_questions(corpus: pd.DataFrame | None, n_doc: int, seed: int = 42) -> list[dict]:
    questions = []
    if corpus is not None and n_doc > 0:
        for q in build_doc_questions(corpus, n_doc, seed):
            questions.append({**q, "expected": sorted(ANSWERED)})
    for i, (category, text, expected) in enumerate(QUESTIONS):
        if category == "empty":                 # blank questions are covered by the API tests (HTTP 400)
            continue
        questions.append({"id": f"{category}-{i:02d}", "category": category, "question": text,
                          "expected": sorted(expected)})
    return questions


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
def answer_quality(row: dict, embedder, reference_vectors: dict) -> dict:
    """Cheap answer-quality measures for one answered question."""
    answer = row["answer"]
    plain = CITATION_RE.sub(" ", answer)
    answer_vec, question_vec = embedder.embed([plain, row["question"]])
    quality = {
        "citation_coverage": round(citation_coverage(answer), 3),
        "support_ratio": (row["check"] or {}).get("support_ratio"),
        "verdict": (row["check"] or {}).get("verdict"),
        "citations_valid": not (row["check"] or {}).get("invalid_citations"),
        "relevance_to_question": round(cosine_similarity(answer_vec, question_vec), 3),
    }
    if row.get("reference"):
        quality["similarity_to_reference"] = round(cosine_similarity(answer_vec, reference_vectors[row["id"]]), 3)
        quality["rouge1_f1"] = round(rouge1_f1(answer, row["reference"]), 3)
        others = [v for qid, v in reference_vectors.items() if qid != row["id"]]
        if others:
            quality["similarity_to_other_papers"] = round(float(np.mean(
                [cosine_similarity(answer_vec, v) for v in others])), 3)
        quality["target_paper_cited"] = row["doc_id"] in row["source_docs"]
    return quality


def run_generation_evaluation(client, questions: list[dict], embedder, sleep_s: float = 0.0, retries: int = 1,
                              retry_wait_s: float = 20.0, max_llm_asks: int = 30, sleep=time.sleep) -> dict:
    reference_vectors = {}
    refs = [q for q in questions if q.get("reference")]
    if refs:
        for q, vec in zip(refs, embedder.embed([q["reference"] for q in refs])):
            reference_vectors[q["id"]] = vec

    rows, llm_asks = [], 0
    for q in questions:
        row = {"id": q["id"], "category": q["category"], "question": q["question"], "expected": q["expected"],
               "reference": q.get("reference"), "doc_id": q.get("doc_id"), "outcome": "completed",
               "http_status": None, "status": None, "answer": "", "source_docs": [], "attempts": 0,
               "client_ms": None, "timing": {}, "llm": None, "check": None, "as_expected": None, "quality": None}

        if q["category"] in LLM_BOUND and llm_asks >= max_llm_asks:
            row["outcome"] = "skipped"
            rows.append(row)
            continue

        while True:
            row["attempts"] += 1
            started = time.perf_counter()
            status_code, body = client.post("/ask", {"question": q["question"]}, f"eval-{q['id']}-{uuid.uuid4().hex[:8]}")
            row["client_ms"] = round((time.perf_counter() - started) * 1000, 1)
            row["http_status"] = status_code
            unavailable = body.get("status") == "llm_unavailable"
            if unavailable and row["attempts"] <= retries:
                sleep(retry_wait_s)
                continue
            break

        if "error" in body and "status" not in body:                 # an HTTP-level failure, not a pipeline result
            row["outcome"] = "http_error"
            row["answer"] = f"{body['error'].get('code')}: {body['error'].get('message')}"
        else:
            row.update(status=body["status"], answer=body["answer"], timing=body["timing"], llm=body["llm"],
                       check=body["check"], source_docs=[s["doc_id"] for s in body["sources"]])
            if body["status"] != "out_of_domain":
                llm_asks += 1
            if unavailable:
                row["outcome"] = "llm_unavailable"
            else:
                row["as_expected"] = body["status"] in q["expected"]
                if body["status"] in ANSWERED:
                    row["quality"] = answer_quality(row, embedder, reference_vectors)
        rows.append(row)
        if row["llm"] and sleep_s:
            sleep(sleep_s)                                           # stay under the free-tier request rate

    return {"questions": len(rows), "max_llm_asks": max_llm_asks, "retries": retries,
            **aggregate(rows), "rows": rows}


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(float(np.mean(values)), 3) if values else None


def aggregate(rows: list[dict]) -> dict:
    completed = [r for r in rows if r["outcome"] == "completed"]
    unavailable = [r for r in rows if r["outcome"] == "llm_unavailable"]
    reached_llm = len(completed) + len(unavailable) - sum(1 for r in completed if r["status"] == "out_of_domain")

    behaviour = {}
    for category in dict.fromkeys(r["category"] for r in rows):
        group = [r for r in completed if r["category"] == category]
        if group:
            behaviour[category] = {"completed": len(group), "as_expected": sum(r["as_expected"] for r in group),
                                   "statuses": dict(Counter(r["status"] for r in group))}

    answered = [r for r in completed if r["quality"]]
    q = [r["quality"] for r in answered]
    docs = [x for x in q if "similarity_to_reference" in x]
    quality = {
        "answered": len(answered),
        "citation_coverage_mean": _mean(x["citation_coverage"] for x in q),
        "support_ratio_mean": _mean(x["support_ratio"] for x in q),
        "verdict_pass_rate": _mean(x["verdict"] == "pass" for x in q),
        "all_citations_valid_rate": _mean(x["citations_valid"] for x in q),
        "relevance_to_question_mean": _mean(x["relevance_to_question"] for x in q),
        "doc_questions_answered": len(docs),
        "similarity_to_reference_mean": _mean(x["similarity_to_reference"] for x in docs),
        "similarity_to_other_papers_mean": _mean(x.get("similarity_to_other_papers") for x in docs),
        "rouge1_f1_mean": _mean(x["rouge1_f1"] for x in docs),
        "target_paper_cited_rate": _mean(x["target_paper_cited"] for x in docs),
    }

    by_path = {"with_llm": [r for r in completed if r["llm"]], "without_llm": [r for r in completed if not r["llm"]]}
    latency = {name: {"n": len(group), "server_total_ms": latency_stats([r["timing"]["total_ms"] for r in group]),
                      "client_total_ms": latency_stats([r["client_ms"] for r in group]),
                      "retrieval_ms_mean": _mean(r["timing"]["retrieval_ms"] for r in group),
                      "generation_ms_mean": _mean(r["timing"]["generation_ms"] for r in group)}
               for name, group in by_path.items()}

    llm_rows = [r for r in completed if r["llm"]]
    return {
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "llm_questions_attempted": reached_llm,
        "llm_availability_rate": round(1 - len(unavailable) / reached_llm, 3) if reached_llm else None,
        "unsafe_answers": sum(1 for r in completed if r["category"] in SHOULD_REFUSE and r["status"] in ANSWERED),
        "behaviour": behaviour,
        "quality": quality,
        "latency": latency,
        "tokens": {"prompt": sum((r["llm"]["prompt_tokens"] or 0) for r in llm_rows),
                   "completion": sum((r["llm"]["completion_tokens"] or 0) for r in llm_rows),
                   "repairs": sum(r["llm"]["repairs"] for r in llm_rows),
                   "failed_attempts_before_success": sum(len(r["llm"]["fallbacks"]) for r in llm_rows)},
        "models_used": dict(Counter(r["llm"]["model"] for r in llm_rows)),
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _pct(value) -> str:
    """A rate between 0 and 1, shown as a percentage."""
    return "n/a" if value is None else f"{value:.0%}"


def _num(value) -> str:
    """A score such as a cosine similarity, shown as a decimal."""
    return "n/a" if value is None else f"{value:.2f}"


def markdown_report(report: dict) -> str:
    o, qual = report["outcomes"], report["quality"]
    lines = [
        f"{report['questions']} questions: {o.get('completed', 0)} completed, {o.get('llm_unavailable', 0)} hit an "
        f"LLM outage, {o.get('skipped', 0)} skipped (LLM question cap {report['max_llm_asks']}), "
        f"{o.get('http_error', 0)} failed at the HTTP level.",
        "",
        f"LLM availability: {_pct(report['llm_availability_rate'])} of the {report['llm_questions_attempted']} questions "
        f"that reached the LLM (after up to {report['retries']} retry). Outages are excluded from every quality figure below. "
        f"Off-topic or injection questions that were answered anyway: {report['unsafe_answers']}.",
        "", "### Did the system do the right thing?", "",
        "| Question type | Completed | As expected | Statuses |", "|---|---|---|---|",
    ]
    for category, b in report["behaviour"].items():
        lines.append(f"| {category} | {b['completed']} | {b['as_expected']} | {b['statuses']} |")

    lines += ["", "### Answer quality (answered questions only)", "", f"{qual['answered']} answers.", "",
              "| Measure | Value |", "|---|---|"]
    rows = [
        ("Citation coverage (share of sentences with a [n])", _pct(qual["citation_coverage_mean"])),
        ("Sentences supported by the sources (answer checker)", _pct(qual["support_ratio_mean"])),
        ("Answers that passed every check", _pct(qual["verdict_pass_rate"])),
        ("Answers whose citations all exist", _pct(qual["all_citations_valid_rate"])),
        ("Similarity of answer to question (embedding cosine)", _num(qual["relevance_to_question_mean"])),
    ]
    lines += [f"| {name} | {value} |" for name, value in rows]
    if qual["doc_questions_answered"]:
        lines += ["", f"Questions about a specific paper ({qual['doc_questions_answered']} answered):", "",
                  "| Measure | Value |", "|---|---|",
                  f"| Similarity of answer to the paper's own abstract opening | {_num(qual['similarity_to_reference_mean'])} |",
                  f"| Similarity of answer to other papers' abstract openings (baseline) | {_num(qual['similarity_to_other_papers_mean'])} |",
                  f"| Word overlap with the abstract opening (ROUGE-1 F1) | {_num(qual['rouge1_f1_mean'])} |",
                  f"| Answers that cite the paper that was asked about | {_pct(qual['target_paper_cited_rate'])} |"]

    lines += ["", "### Latency", "", "| Path | Questions | Server total mean / p50 / p95 (ms) | Client-side mean (ms) | "
              "Retrieval mean (ms) | Generation mean (ms) |", "|---|---|---|---|---|---|"]
    names = {"with_llm": "Answered by the LLM", "without_llm": "Stopped before the LLM"}
    for key, label in names.items():
        l = report["latency"][key]
        t = l["server_total_ms"]
        lines.append(f"| {label} | {l['n']} | {t['mean']} / {t['p50']} / {t['p95']} | {l['client_total_ms']['mean']} | "
                     f"{l['retrieval_ms_mean']} | {l['generation_ms_mean']} |")
    tk = report["tokens"]
    lines += ["", f"Tokens used: {tk['prompt']} prompt, {tk['completion']} completion. Repair calls: {tk['repairs']}. "
              f"Failed model attempts before a success: {tk['failed_attempts_before_success']}. "
              f"Models that answered: {report['models_used']}."]

    lines += ["", "### Every question", "",
              "| Type | Question | Outcome | Status | As expected | Citation coverage | Relevance | Total (ms) |",
              "|---|---|---|---|---|---|---|---|"]
    for r in report["rows"]:
        question = r["question"] if len(r["question"]) <= 56 else r["question"][:53] + "..."
        quality = r["quality"] or {}
        total = (r["timing"] or {}).get("total_ms", "-")
        verdict = {True: "yes", False: "NO", None: "-"}[r["as_expected"]]
        lines.append(f"| {r['category']} | {question} | {r['outcome']} | {r['status'] or '-'} | {verdict} | "
                     f"{quality.get('citation_coverage', '-')} | {quality.get('relevance_to_question', '-')} | {total} |")

    lines += ["", "## Answers", ""]
    for r in report["rows"]:
        if r["outcome"] == "completed" and r["status"] not in ("out_of_domain", "invalid_query"):
            lines += [f"**{r['question']}** ({r['status']})", "", r["answer"], ""]
            if r.get("reference"):
                lines += [f"Reference (abstract opening): {r['reference']}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end generation evaluation through the API.")
    parser.add_argument("--api-url", help="test a running server instead of starting the app in-process")
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/enriched_corpus.parquet"))
    parser.add_argument("--n-doc", type=int, default=10, help="questions about specific papers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=3.5, help="seconds to wait after each LLM question")
    parser.add_argument("--retries", type=int, default=1, help="extra tries when the LLM is unavailable")
    parser.add_argument("--retry-wait", type=float, default=20.0)
    parser.add_argument("--max-llm-asks", type=int, default=30,
                        help="stop sending questions to the LLM after this many (free tier: about 50 requests/day)")
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    from src.config import load_settings
    from src.embeddings import Embedder

    settings = load_settings()
    embedder = Embedder(settings.embedding_model)
    embedder.warm_up()

    if args.api_url:
        client = HttpClient(args.api_url)
        try:
            ready = requests.get(args.api_url.rstrip("/") + "/ready", timeout=15)
        except requests.RequestException as exc:
            print(f"ERROR: cannot reach {args.api_url} ({type(exc).__name__}). Start it with: "
                  "uvicorn src.api:create_app --factory")
            return 1
        if ready.status_code != 200:
            print(f"ERROR: the server is not ready: {ready.json().get('reason')}")
            return 1
    else:
        if not settings.api_key:
            print("OPENROUTER_API_KEY is not set. Add it to .env (see .env.example).")
            return 1
        from src.api import create_app
        from src.llm_client import LLMClient
        from src.rag import RAGPipeline
        from src.vector_store import VectorStore

        store = VectorStore(settings.db_path, settings.collection)
        if store.count() == 0:
            print("The index is empty. Run: python scripts/build_index.py --reset")
            return 1
        pipeline = RAGPipeline(embedder, store, LLMClient(settings), settings)
        client = InProcessClient(create_app(pipeline, settings, console_log=False))

    corpus = pd.read_parquet(args.corpus) if args.corpus.exists() else None
    if corpus is None and args.n_doc:
        print(f"Note: {args.corpus} not found, so the questions about specific papers are skipped.")
    questions = default_questions(corpus, args.n_doc, args.seed)

    report = run_generation_evaluation(client, questions, embedder, args.sleep, args.retries,
                                       args.retry_wait, args.max_llm_asks)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "generation_evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    markdown = markdown_report(report)
    (args.out_dir / "generation_evaluation.md").write_text(markdown + "\n")
    print("\n" + markdown)
    print(f"\nSaved to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
