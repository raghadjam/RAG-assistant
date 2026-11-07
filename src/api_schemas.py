"""Request and response models for the HTTP API.

The request models decide what counts as a structurally invalid request (HTTP 422).
They also produce the documentation shown at /docs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.rag import MAX_QUESTION_CHARS

Sentiment = Literal["negative", "neutral", "positive"]


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class Filters(BaseModel):
    """Restrict retrieval to chunks with this metadata (from the Week 4 enrichment)."""

    model_config = ConfigDict(extra="forbid")

    topic_id: int | None = Field(
        None, ge=-1, strict=True, description="LDA topic id. See GET /topics. -1 means 'unassigned'."
    )
    sentiment: Sentiment | None = Field(None, description="VADER sentiment label of the paper.")


class SearchRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"query": "federated learning privacy", "top_k": 3,
                                         "filters": {"sentiment": "positive"}}]},
    )

    query: str = Field(..., max_length=MAX_QUESTION_CHARS, description="Text to search for.")
    top_k: int = Field(5, ge=1, le=20, strict=True, description="Number of chunks to return.")
    filters: Filters = Field(default_factory=Filters)


class AskRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"question": "How can hallucination in language models be reduced?",
                                         "top_k": 5, "filters": {"topic_id": 2}}]},
    )

    question: str = Field(..., max_length=MAX_QUESTION_CHARS, description="Question about the paper collection.")
    top_k: int | None = Field(None, ge=1, le=10, strict=True, description="Chunks to retrieve (default from settings).")
    filters: Filters = Field(default_factory=Filters)


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class Timing(BaseModel):
    embed_ms: float = 0.0
    search_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class Source(BaseModel):
    n: int = Field(..., description="The [n] used in the answer.")
    doc_id: str | None = None
    title: str | None = None
    chunk_index: int | None = None
    topic_id: int | None = None
    topic_name: str | None = None
    sentiment_label: str | None = None
    score: float


class LlmInfo(BaseModel):
    model: str | None = None
    calls: int = 0
    http_attempts: int = 0
    repairs: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    truncated: bool = False
    fallbacks: list[str] = Field(default_factory=list, description="Failed attempts before this model answered.")


class AnswerCheck(BaseModel):
    verdict: str
    citations: list[int] = Field(default_factory=list)
    invalid_citations: list[int] = Field(default_factory=list)
    unsupported_numbers: list[str] = Field(default_factory=list)
    support_ratio: float = 1.0
    hard_failures: list[str] = Field(default_factory=list)
    soft_failures: list[str] = Field(default_factory=list)


class AskResponse(BaseModel):
    request_id: str
    question: str
    status: str = Field(
        ...,
        description=("answered, answered_with_warnings, no_answer, out_of_domain, rejected_ungrounded, "
                     "llm_unavailable or invalid_query"),
    )
    answer: str
    sources: list[Source] = Field(default_factory=list)
    top_score: float | None = Field(None, description="Cosine similarity of the best retrieved chunk.")
    filters: Filters = Field(default_factory=Filters)
    timing: Timing
    llm: LlmInfo | None = None
    check: AnswerCheck | None = None
    note: str | None = Field(None, description="Why no answer was produced, when relevant.")


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str | None = None
    title: str | None = None
    text: str
    score: float
    chunk_index: int | None = None
    topic_id: int | None = None
    topic_name: str | None = None
    sentiment_label: str | None = None
    sentiment_score: float | None = None


class SearchResponse(BaseModel):
    request_id: str
    query: str
    filters: Filters
    results: list[SearchHit]
    timing: Timing


class TopicInfo(BaseModel):
    topic_id: int
    topic_name: str | None = None
    chunks: int
    documents: int


class TopicsResponse(BaseModel):
    enriched: bool = Field(..., description="False if the index was built without NLP metadata.")
    total_chunks: int
    topics: list[TopicInfo]
    sentiments: dict[str, int]


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_s: float


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    version: str
    chunks: int = 0
    enriched: bool = False
    llm_configured: bool = False
    models: list[str] = Field(default_factory=list)
    reason: str | None = None


class ErrorBody(BaseModel):
    code: str = Field(..., description="Stable machine-readable error code.")
    message: str
    details: list[dict] | None = Field(None, description="Per-field problems for validation errors.")
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
