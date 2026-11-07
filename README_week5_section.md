
---

# Week 5: Evaluation & API (FastAPI)

This week wraps the pipeline in an HTTP service and measures how good it is:

1. A FastAPI application (`src/api.py`) with search, question answering, topic listing and health endpoints
2. Proper error responses (400, 422, 500, 503) in one consistent JSON shape
3. Structured request logging: one JSON line per request, with a request id that ties everything together
4. A retrieval evaluation with Precision@K, Recall@K and MRR on a labelled test set
5. An end-to-end evaluation of answer quality and latency, run through the API itself
6. 204 new tests for the endpoints, the metrics and the evaluation scripts

New files this week:

```
src/api.py                       # the FastAPI app, endpoints, error handlers, request logging
src/api_schemas.py               # request and response models (these also generate /docs)
src/api_logging.py               # JSON-lines request log
src/evaluation.py                # ranking metrics, test-set builder, answer-quality helpers
src/text_utils.py                # LaTeX removal for building realistic queries
scripts/build_retrieval_testset.py
scripts/evaluate_retrieval.py    # Precision@K, Recall@K, MRR
scripts/evaluate_generation.py   # answer quality and latency, end to end through the API
data/eval/retrieval_testset.jsonl
tests/test_api.py  tests/test_evaluation.py  tests/test_week5_scripts.py
```

Small changes to existing files: `src/config.py` gained settings for the index location and the API log (optional environment variables `CHROMA_DB_PATH`, `CHROMA_COLLECTION`, `EMBEDDING_MODEL` and `API_LOG_PATH`; the defaults match the earlier weeks), and `RAGPipeline.ask` accepts an optional `top_k`.

## The API

Start it (the first start loads the embedding model, which takes a few seconds):

```bash
uvicorn src.api:create_app --factory --port 8000
```

Interactive documentation, generated from the code, is at http://127.0.0.1:8000/docs.

| Endpoint | What it does |
|---|---|
| `POST /ask` | Retrieve passages, have the LLM answer from them with citations, and check the answer |
| `POST /search` | Return the matching passages only. No LLM call, so it is fast and uses no quota |
| `GET /topics` | The topic ids and sentiment labels that can be used as filters, with their sizes |
| `GET /health` | Liveness: always 200 while the process runs |
| `GET /ready` | Readiness: 200 when the index is loaded, otherwise 503 with the reason |

Both `/ask` and `/search` accept optional `filters` (`topic_id`, `sentiment`) and `top_k`:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "How can hallucination in language models be reduced?", "filters": {"topic_id": 2}}'

curl -X POST http://127.0.0.1:8000/search -H "Content-Type: application/json" \
  -d '{"query": "federated learning privacy", "top_k": 3, "filters": {"sentiment": "positive"}}'
```

A response to `/ask` contains the `answer`, its `status`, the cited `sources` (with topic and sentiment), the best retrieval `top_score`, a `timing` breakdown (embedding, search, retrieval, generation, total), the `llm` details (model, tokens, retries), the answer `check` result, and a `request_id`.

### Design decisions

- **App factory with an injected pipeline.** `create_app(pipeline)` lets the tests run the real service against a fake LLM and an in-memory index, with no model download, network or key. With no argument it builds the real pipeline at startup.
- **Plain `def` endpoints.** Embedding, ChromaDB and the LLM call all block. FastAPI runs plain functions in a thread pool, so one slow answer does not freeze the other endpoints.
- **The service survives a failed startup.** If the model or index cannot be loaded, `/health` still answers and `/ready` and `/ask` return 503 with the reason, instead of the process dying with a traceback.
- **A refusal is a normal answer.** An off-topic question returns 200 with `status: "out_of_domain"`; the `status` field says what the pipeline did. Only a real outage returns an error status.
- **Strict inputs.** `top_k: "3"` or `topic_id: true` are rejected rather than silently converted. Unknown fields are rejected too, so a typo such as `"questions"` is caught.
- **The rejected draft never leaves the server.** If an answer fails the grounding checks the client sees only a refusal message, not the unverified text.

### Error responses

Every error uses the same body, so clients need one parser:

```json
{"error": {"code": "unknown_topic", "message": "topic_id 42 does not exist. Valid ids: [0, 1, 2, 3, 4, 5, 6, 7].",
           "details": null, "request_id": "5f0c1e..."}}
```

| HTTP status | When | `error.code` |
|---|---|---|
| 200 | The request was handled. The `status` field says what the pipeline did | n/a |
| 400 | The request is well-formed but cannot be processed | `invalid_json`, `blank_text`, `unknown_topic`, `filters_unavailable` |
| 422 | The body breaks the schema: missing or unknown field, wrong type, value out of range (`top_k` above 20, question over 1,000 characters) | `validation_error`, with a `details` list naming each field |
| 404 / 405 | Unknown route or wrong method | `not_found`, `method_not_allowed` |
| 500 | An unexpected bug. The traceback is logged on the server, never sent to the client | `internal_error` |
| 503 | The index is not loaded or empty, or the LLM could not be reached. Includes a `Retry-After` header | `service_unavailable`, `index_empty`, or (for an LLM outage) a normal `/ask` body with `status: "llm_unavailable"` and the retrieved `sources` |

A 400 means "fix the request"; a 422 means "the request does not match the schema". Validation messages never echo the submitted text back.

### Request logging

Every request writes one JSON line to the console and to `logs/api_requests.jsonl` (git-ignored). Each response also carries `X-Request-ID` and `X-Response-Time-ms` headers. A client can supply its own `X-Request-ID` (8 to 64 letters, digits, `.`, `_`, `-`); otherwise one is generated.

```json
{"time": "2026-11-03T10:15:02.113+00:00", "request_id": "5f0c1e9a...", "method": "POST", "path": "/ask",
 "status_code": 200, "duration_ms": 1432.1, "client": "127.0.0.1", "question_chars": 52,
 "question_preview": "How can hallucination in language models be reduc...", "top_k": null,
 "filters": {"topic_id": 2}, "rag_status": "answered", "top_score": 0.71, "embed_ms": 14.2,
 "search_ms": 9.1, "retrieval_ms": 23.3, "generation_ms": 1408.8, "total_ms": 1432.1,
 "model": "openrouter/free", "llm_calls": 1, "prompt_tokens": 842, "completion_tokens": 96,
 "check_verdict": "pass"}
```

Error lines carry an `error_code` instead. Only the first 80 characters of a question are logged, as a preview. The evaluation scripts send ids like `eval-doc-03-1a2b3c4d`, so a slow evaluation question can be found in the server log. The log file can be read with `jq` or loaded into pandas (`pd.read_json(path, lines=True)`).

## Retrieval evaluation

**Test set.** `scripts/build_retrieval_testset.py` builds labelled queries from the corpus itself, so no manual labelling is needed and the right answers are known. Three kinds, with different ground truth:

| Type | Query | Relevant papers |
|---|---|---|
| title | A paper's title (40 queries, spread evenly over the 8 topics) | That paper |
| sentence | One sentence from the middle of a different paper's abstract (40 queries) | That paper |
| concept | A phrase such as "federated learning" or "diffusion model" (those that at least 10 papers mention) | Every paper whose abstract mentions the phrase, tolerating plurals and hyphens |

The test set is saved to `data/eval/retrieval_testset.jsonl` and committed, so the evaluation can be rerun on exactly the same queries. The script refuses to overwrite it without `--force`.

**Metrics.** K is the number of chunks returned, and a chunk is relevant when its paper is in the query's relevant set.

- **Precision@K**: relevant chunks among the top K, divided by K.
- **Recall@K**: distinct relevant papers found in the top K, divided by min(number of relevant papers, K). The cap makes 1.0 reachable for a concept with hundreds of relevant papers and only K slots. For a query with a single relevant paper this is the hit rate: was the paper found in the top K?
- **MRR**: the mean of 1 / (rank of the first relevant chunk).

Results are reported by query type at K = 1, 3, 5 and 10. For the known-paper queries the script also searches a second time with a filter on that paper's own topic, to show what the metadata filter costs or gains when the topic is known.

## End-to-end evaluation of generation

`scripts/evaluate_generation.py` asks questions through the API (in-process by default, or against a running server with `--api-url`) and records what the service did. There are three kinds of question:

- **doc**: "What does the paper titled ... propose or find?" for sampled papers, with the opening of the abstract as a reference answer
- **answerable**: the topical questions from the Week 3 evaluation
- **negative**: unanswerable, off-topic, prompt-injection and nonsense questions, which should be refused

Quality is measured with cheap proxies, with no second LLM acting as judge:

| Measure | What it shows |
|---|---|
| Behaved as expected | Answered what could be answered, and refused what should be refused |
| Off-topic questions answered | Should be 0 |
| Citation coverage | Share of answer sentences carrying a `[n]` |
| Support ratio, share passing every check | The Week 3 answer checker: citations exist, numbers appear in the sources, wording overlaps the sources |
| Similarity to the question | Embedding cosine between answer and question |
| Similarity to the reference, and to other papers' references | Does the answer resemble the right paper's abstract more than unrelated ones? The second number is the baseline that gives the first one meaning |
| ROUGE-1 F1 | Word overlap with the reference |
| Target paper cited | For doc questions: did the answer cite the paper that was asked about? |

Latency is reported as server-side time (embedding, search, retrieval, generation, total) and client-observed time, separately for questions the LLM answered and for those stopped before it.

**LLM outages are counted, never averaged in.** In the Week 3 evaluation most answerable questions hit an LLM outage on the free tier. A question that reaches an outage is retried once after a wait; if it still fails it is recorded as `llm_unavailable` and excluded from every quality figure, and the report states the availability rate. Free models allow about 50 requests per day, so `--max-llm-asks` (default 30) caps how many questions are sent to the LLM; the remaining ones are marked `skipped`.

## How to run Week 5

```bash
# 1. Make sure the LLM models in .env work (an outage burns quota)
python scripts/check_llm.py

# 2. Build the labelled retrieval test set (once) and evaluate retrieval. No API quota used
python scripts/build_retrieval_testset.py
python scripts/evaluate_retrieval.py

# 3. Run the API and look at /docs
uvicorn src.api:create_app --factory --port 8000

# 4. Evaluate answer quality and latency (about 15 to 25 LLM requests)
python scripts/evaluate_generation.py
#    or against the running server, from a second terminal:
python scripts/evaluate_generation.py --api-url http://127.0.0.1:8000

# 5. Run all tests
python -m pytest tests/ -q
```

## Results

> TODO: paste the contents of `reports/retrieval_evaluation.md` here (the definitions, the table by query type, the topic-filter comparison, and the list of queries that were not found).

> TODO: paste the first parts of `reports/generation_evaluation.md` here (the outcome summary, "Did the system do the right thing?", answer quality, and latency tables).

> TODO: paste one real `/ask` response from your running server (a `curl` call, shortened if needed) and two lines from `logs/api_requests.jsonl`.

> TODO: write 5 to 8 sentences of your own on what you saw: which query type was hardest for retrieval and why, whether the topic filter helped or hurt, how often the LLM was available, whether any off-topic question was answered, how the answers compared with the reference, and where the time goes in a full answer.

## Testing

The Week 5 tests add 204 tests (395 in total). They need no network, no model download and no API key.

- `tests/test_api.py` (109 tests) runs the real service against a real in-memory ChromaDB, a hashing embedder and a scripted LLM. It covers every endpoint and every status code: 400 (invalid JSON, blank text, unknown topic, filters on an index without metadata), 422 (missing, unknown, wrongly typed and out-of-range fields), 500 (a bug, with a check that nothing internal leaks), 503 (failed startup, empty index, LLM outage). It also checks filtered retrieval, request ids, the log lines, and the generated OpenAPI documentation.
- `tests/test_evaluation.py` (61 tests) checks Precision@K, Recall@K and MRR against hand-computed examples, the test-set builder (reproducibility, topic coverage, correct relevance sets) and the answer-quality helpers.
- `tests/test_week5_scripts.py` (34 tests) covers the three scripts, including the retry on LLM outage, the question cap, HTTP-level failures, and the HTTP client against a real local server.

I also started the real `uvicorn` server and exercised it over TCP with `curl` and the evaluation client, in addition to the in-process tests.

## Limitations

- **The retrieval test set is synthetic.** Titles and abstract sentences are easier queries than real questions, because the answer text is in the index. Treat the known-paper scores as an upper bound. For concept queries a paper that is relevant but never uses the exact phrase counts as irrelevant, which understates precision.
- **The generation measures are proxies.** Embedding similarity and word overlap are rough stand-ins for human judgement of correctness, and the question set is small (about 25 questions), so differences of a few points are noise. The free LLM can give different answers to the same question on different runs.
- **Free-tier limits shape the evaluation.** About 50 requests a day, and an outage still costs requests, so the generation evaluation cannot be rerun many times in one day.
- **Filters cover what Week 4 stored.** Only `topic_id` and `sentiment` can be filtered. The topic and sentiment catalog is read once when first needed, so restart the service after rebuilding the index.
- **Single process, no authentication.** There is no API key check, rate limiting or CORS configuration, and the embedding model is loaded once per process, so run one worker. Question previews (80 characters) appear in the request log.
