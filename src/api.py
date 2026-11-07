"""FastAPI service for the RAG pipeline.

Run it:

    uvicorn src.api:create_app --factory --port 8000

then open http://127.0.0.1:8000/docs for the interactive documentation.

HTTP status codes used:

    200  the request was handled (the "status" field says what the pipeline did,
         for example answered, out_of_domain or no_answer)
    400  the request is well-formed but cannot be processed: invalid JSON, blank text,
         an unknown topic id, filters on an index that has no metadata
    422  the request body breaks the schema: missing or unknown fields, wrong types, values out of range
    500  an unexpected bug (details are logged, never sent to the client)
    503  the service cannot answer right now: the index is not loaded, or the LLM is unavailable
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api_logging import configure_api_logging, log_request, preview
from src.api_schemas import (
    AnswerCheck,
    AskRequest,
    AskResponse,
    ErrorResponse,
    Filters,
    HealthResponse,
    LlmInfo,
    ReadyResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    Source,
    Timing,
    TopicInfo,
    TopicsResponse,
)
from src.config import Settings, load_settings
from src.search import build_metadata_filter, semantic_search

API_VERSION = "0.5.0"
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
STATUS_TO_HTTP = {"llm_unavailable": 503, "invalid_query": 400}   # every other pipeline status is a 200
RETRY_AFTER_S = "30"

logger = logging.getLogger("rag.api")
router = APIRouter()


class ApiError(Exception):
    """An error with an HTTP status and a stable code, turned into the JSON error body."""

    def __init__(self, status_code: int, code: str, message: str, details=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details
        self.headers = headers


ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Well-formed request that cannot be processed (blank text, invalid JSON, unknown topic)."},
    422: {"model": ErrorResponse, "description": "The request body breaks the schema (missing or unknown field, wrong type, out of range)."},
    500: {"model": ErrorResponse, "description": "Unexpected server error. Quote the request_id when reporting it."},
    503: {"model": ErrorResponse, "description": "The index or the language model is not available."},
}


# --------------------------------------------------------------------------- #
# Building the real pipeline
# --------------------------------------------------------------------------- #
def build_pipeline(settings: Settings):
    """Load the embedding model, open the index and connect the LLM client."""
    from src.embeddings import Embedder
    from src.llm_client import LLMClient
    from src.rag import RAGPipeline
    from src.vector_store import VectorStore

    embedder = Embedder(settings.embedding_model)
    embedder.warm_up()
    store = VectorStore(settings.db_path, settings.collection)
    return RAGPipeline(embedder, store, LLMClient(settings), settings)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _error(request: Request, status: int, code: str, message: str, details=None, headers=None) -> JSONResponse:
    """The one JSON shape every error uses."""
    request_id = getattr(request.state, "request_id", None)
    if hasattr(request.state, "log_extra"):
        request.state.log_extra["error_code"] = code
    body = {"error": {"code": code, "message": message, "details": details, "request_id": request_id}}
    return JSONResponse(status_code=status, content=body, headers=headers)


def _require_pipeline(request: Request):
    pipeline = request.app.state.pipeline
    if pipeline is None:
        reason = request.app.state.startup_error or "the pipeline is not loaded"
        raise ApiError(503, "service_unavailable", f"The service is not ready: {reason}",
                       headers={"Retry-After": RETRY_AFTER_S})
    if pipeline.store.count() == 0:
        raise ApiError(503, "index_empty", "The index is empty. Run scripts/build_index.py first.",
                       headers={"Retry-After": RETRY_AFTER_S})
    return pipeline


def _require_text(text: str, field: str) -> str:
    if not text.strip():
        raise ApiError(400, "blank_text", f"{field} must contain text, not only whitespace.")
    return text.strip()


def _load_catalog(store) -> dict:
    """Count chunks per topic and sentiment by reading the metadata page by page."""
    topics: dict[int, dict] = {}
    sentiments: Counter = Counter()
    chunks, enriched, offset = 0, False, 0
    while True:
        page = store.collection.get(limit=5000, offset=offset, include=["metadatas"])
        if not page["ids"]:
            break
        for meta in page["metadatas"] or []:
            meta = meta or {}
            chunks += 1
            if "topic_id" in meta:
                enriched = True
                entry = topics.setdefault(int(meta["topic_id"]), {"name": meta.get("topic_name"),
                                                                   "chunks": 0, "docs": set()})
                entry["chunks"] += 1
                entry["docs"].add(meta.get("doc_id"))
            if meta.get("sentiment_label"):
                sentiments[meta["sentiment_label"]] += 1
        offset += len(page["ids"])
    return {"chunks": chunks, "enriched": enriched, "topics": topics, "sentiments": dict(sentiments)}


def _catalog(app: FastAPI, store) -> dict:
    """The topic catalog, computed once. The index does not change while the service runs."""
    with app.state.catalog_lock:
        if app.state.catalog is None:
            app.state.catalog = _load_catalog(store)
        return app.state.catalog


def _resolve_filters(request: Request, pipeline, filters: Filters) -> dict | None:
    """Turn the request's filters into a Chroma filter, rejecting ones that cannot work."""
    if filters.topic_id is None and filters.sentiment is None:
        return None
    catalog = _catalog(request.app, pipeline.store)
    if not catalog["enriched"]:
        raise ApiError(400, "filters_unavailable",
                       "This index has no topic or sentiment metadata. "
                       "Run scripts/enrich_corpus.py, then scripts/build_index.py --reset.")
    if filters.topic_id is not None and filters.topic_id not in catalog["topics"]:
        raise ApiError(400, "unknown_topic",
                       f"topic_id {filters.topic_id} does not exist. Valid ids: {sorted(catalog['topics'])}.")
    return build_metadata_filter(topic_id=filters.topic_id, sentiment=filters.sentiment)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.get("/", include_in_schema=False)
def root():
    return {"name": "arXiv RAG Assistant API", "version": API_VERSION, "docs": "/docs"}


@router.get("/health", response_model=HealthResponse, tags=["service"], summary="Liveness check")
def health(request: Request):
    """Always 200 while the process is running. Does not touch the index or the LLM."""
    return HealthResponse(status="ok", version=API_VERSION,
                          uptime_s=round(time.time() - request.app.state.started, 1))


@router.get("/ready", response_model=ReadyResponse, tags=["service"], summary="Readiness check",
            responses={503: {"model": ReadyResponse, "description": "Index missing or empty."}})
def ready(request: Request, response: Response):
    """200 when the index is loaded and has chunks, otherwise 503 with the reason."""
    app = request.app
    pipeline = app.state.pipeline
    settings = app.state.settings
    llm = {"llm_configured": bool(settings.api_key), "models": list(settings.models)}
    if pipeline is None:
        response.status_code = 503
        return ReadyResponse(status="not_ready", version=API_VERSION, **llm,
                             reason=app.state.startup_error or "the pipeline is not loaded")
    chunks = pipeline.store.count()
    if chunks == 0:
        response.status_code = 503
        return ReadyResponse(status="not_ready", version=API_VERSION, **llm,
                             reason="The index is empty. Run scripts/build_index.py first.")
    sample = pipeline.store.collection.get(limit=1, include=["metadatas"])["metadatas"] or [{}]
    return ReadyResponse(status="ready", version=API_VERSION, chunks=chunks,
                         enriched="topic_id" in (sample[0] or {}), **llm)


@router.get("/topics", response_model=TopicsResponse, tags=["rag"], responses=ERROR_RESPONSES,
            summary="Topics and sentiment counts available as filters")
def topics(request: Request):
    """The topic ids to use in `filters.topic_id`, with their labels and sizes."""
    pipeline = _require_pipeline(request)
    catalog = _catalog(request.app, pipeline.store)
    listing = [TopicInfo(topic_id=tid, topic_name=entry["name"], chunks=entry["chunks"],
                         documents=len(entry["docs"]))
               for tid, entry in sorted(catalog["topics"].items())]
    return TopicsResponse(enriched=catalog["enriched"], total_chunks=catalog["chunks"],
                          topics=listing, sentiments=catalog["sentiments"])


@router.post("/search", response_model=SearchResponse, tags=["rag"], responses=ERROR_RESPONSES,
             summary="Semantic search (no LLM)")
def search(body: SearchRequest, request: Request):
    """Return the chunks most similar to the query, optionally restricted by topic or sentiment.

    Fast and free: it uses no LLM call. An empty `results` list is a valid answer.
    """
    pipeline = _require_pipeline(request)
    query = _require_text(body.query, "query")
    where = _resolve_filters(request, pipeline, body.filters)

    out = semantic_search(query, pipeline.embedder, pipeline.store, k=body.top_k, where=where)
    hits = []
    for hit in out["results"]:
        meta = hit.get("metadata") or {}
        hits.append(SearchHit(
            chunk_id=hit["chunk_id"], doc_id=meta.get("doc_id"), title=meta.get("title"),
            text=hit["text"], score=round(hit["score"], 4), chunk_index=meta.get("chunk_index"),
            topic_id=meta.get("topic_id"), topic_name=meta.get("topic_name"),
            sentiment_label=meta.get("sentiment_label"), sentiment_score=meta.get("sentiment_score"),
        ))
    t = out["timing"]
    timing = Timing(embed_ms=round(t["embed_ms"], 1), search_ms=round(t["search_ms"], 1),
                    retrieval_ms=round(t["total_ms"], 1), total_ms=round(t["total_ms"], 1))

    request.state.log_extra.update({
        "query_chars": len(query), "query_preview": preview(query), "top_k": body.top_k,
        "filters": body.filters.model_dump(exclude_none=True), "n_results": len(hits),
        "top_score": hits[0].score if hits else None,
        "embed_ms": timing.embed_ms, "search_ms": timing.search_ms,
    })
    return SearchResponse(request_id=request.state.request_id, query=query, filters=body.filters,
                          results=hits, timing=timing)


@router.post("/ask", response_model=AskResponse, tags=["rag"], responses=ERROR_RESPONSES,
             summary="Ask a question (retrieval + LLM answer)")
def ask(body: AskRequest, request: Request, response: Response):
    """Retrieve the relevant chunks, have the LLM answer from them, and check the answer.

    The `status` field says what happened. A refusal (`out_of_domain`, `no_answer`) is a normal
    200 answer. If the LLM cannot be reached the status is 503 and `sources` still lists the
    documents that were found.
    """
    pipeline = _require_pipeline(request)
    question = _require_text(body.question, "question")
    where = _resolve_filters(request, pipeline, body.filters)

    result = pipeline.ask(question, where=where, top_k=body.top_k)
    data = result.to_dict()

    http_status = STATUS_TO_HTTP.get(result.status, 200)
    response.status_code = http_status
    if http_status == 503:
        response.headers["Retry-After"] = RETRY_AFTER_S

    llm = data.get("llm")
    check = data.get("check")
    request.state.log_extra.update({
        "question_chars": len(question), "question_preview": preview(question), "top_k": body.top_k,
        "filters": body.filters.model_dump(exclude_none=True), "rag_status": result.status,
        "top_score": result.top_score, **{k: v for k, v in data["timing"].items()},
        "model": (llm or {}).get("model"), "llm_calls": (llm or {}).get("calls"),
        "prompt_tokens": (llm or {}).get("prompt_tokens"),
        "completion_tokens": (llm or {}).get("completion_tokens"),
        "check_verdict": (check or {}).get("verdict"),
    })
    return AskResponse(
        request_id=request.state.request_id, question=question, status=result.status, answer=result.answer,
        sources=[Source(**s) for s in data["sources"]], top_score=result.top_score, filters=body.filters,
        timing=Timing(**data["timing"]), llm=LlmInfo(**llm) if llm else None,
        check=AnswerCheck(**check) if check else None, note=data.get("error"),
    )


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
DESCRIPTION = """
Question answering over a corpus of arXiv abstracts (AI, machine learning, statistics).

* `POST /ask` retrieves relevant passages and has an LLM answer from them, with citations.
* `POST /search` returns the matching passages only (no LLM).
* `GET /topics` lists the topic ids and sentiment labels usable as filters.
* `GET /health` and `GET /ready` are for monitoring.

**Errors** always use one JSON shape: `{"error": {"code", "message", "details", "request_id"}}`.
Every response carries an `X-Request-ID` header that also appears in the server's request log.
"""

TAGS = [
    {"name": "rag", "description": "Retrieval and question answering."},
    {"name": "service", "description": "Health and readiness."},
]

_USE_SETTINGS = object()


def create_app(pipeline=None, settings: Settings | None = None, log_path=_USE_SETTINGS,
               console_log: bool = True) -> FastAPI:
    """Create the app.

    With no pipeline the real one is built at startup (embedding model, index, LLM client).
    Tests pass a ready-made pipeline. If startup fails the app still runs, answers
    /health, and returns 503 elsewhere with the reason.
    """
    settings = settings or load_settings()
    access_logger = configure_api_logging(
        settings.api_log_path if log_path is _USE_SETTINGS else log_path, console=console_log)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.pipeline is None:
            try:
                app.state.pipeline = build_pipeline(settings)
                logger.info("pipeline ready: %s chunks", app.state.pipeline.store.count())
            except Exception as exc:    # keep the process alive so /health and /ready can explain
                logger.exception("could not build the pipeline")
                app.state.startup_error = f"{type(exc).__name__}: {exc}"
        yield

    app = FastAPI(title="arXiv RAG Assistant API", version=API_VERSION, description=DESCRIPTION,
                  openapi_tags=TAGS, lifespan=lifespan)
    app.state.settings = settings
    app.state.pipeline = pipeline
    app.state.startup_error = None
    app.state.catalog = None
    app.state.catalog_lock = threading.Lock()
    app.state.started = time.time()
    app.include_router(router)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        request.state.request_id = incoming if REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
        request.state.log_extra = {}
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("unhandled error, request_id=%s", request.state.request_id)
            response = _error(request, 500, "internal_error",
                              "The server hit an unexpected error. If it keeps happening, report the request_id.")
        duration_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Response-Time-ms"] = f"{duration_ms:.1f}"
        log_request(access_logger, {
            "request_id": request.state.request_id, "method": request.method, "path": request.url.path,
            "status_code": response.status_code, "duration_ms": round(duration_ms, 1),
            "client": request.client.host if request.client else None, **request.state.log_extra,
        })
        return response

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError):
        return _error(request, exc.status_code, exc.code, exc.message, exc.details, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        if any(e.get("type") == "json_invalid" for e in errors):
            return _error(request, 400, "invalid_json", "The request body is not valid JSON.")
        details = [{"field": ".".join(str(p) for p in e["loc"] if p != "body") or "body",
                    "message": e["msg"], "type": e["type"]} for e in errors]
        return _error(request, 422, "validation_error", "The request body failed validation.", details)

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _error(request, exc.status_code, code, str(exc.detail), headers=exc.headers)

    return app
