# RAG Assistant

An end-to-end Retrieval-Augmented Generation (RAG) project over arXiv abstracts: clean the corpus, chunk and embed it into ChromaDB, retrieve context for questions, generate grounded answers with an LLM, and serve it all through FastAPI.

| Week | Topic | Status |
|---|---|---|
| 1 | Environment, data ingestion, text cleaning | Done (see Week 1 below) |
| 2 | Chunking, embeddings, ChromaDB, retrieval benchmark | Done (see Week 2 below) |
| 3-6 | LLM integration, NLP analysis, FastAPI, evaluation | Upcoming |

All work lives in this single repository. Week 2 builds directly on the clean corpus produced in Week 1.

---

# Week 1: Environment, Data Ingestion & Preprocessing

This week covers the foundation:

1. A reproducible environment (git, venv, locked dependencies) with a verification script
2. Acquisition of a real-world corpus of 5,000+ documents
3. A modular, unit-tested text cleaning pipeline
4. A validated clean dataset saved as Parquet

## Results at a glance

| Item | Result |
|---|---|
| Raw documents acquired | 5,500 (arXiv abstracts) |
| Documents after cleaning and validation | **5,499** |
| Rejected | 1 (shorter than 200 characters) |
| Language of clean corpus | 100% English |
| Document length (title + abstract) | mean about 1,550 chars, min 320, max 2,930 |
| Unit tests | 32 passed |
| Environment | Python 3.12.3, WSL2 (Ubuntu), NVIDIA GeForce MX570 A detected, torch 2.4.1+cu121 |

Output of `verify_env.py` is saved in `verify_env_output.txt`.

## Approach

### Data source: arXiv
I used the public arXiv API because it needs no account or API key, returns clean structured metadata, and its abstracts are dense, technical English that works well for a RAG assistant. Papers come from five categories (cs.AI, cs.CL, cs.LG, stat.ML, cs.IR) and are de-duplicated by arXiv ID, because one paper can carry several category tags. Each page of 100 results is followed by a 3-second pause, as arXiv requests. The data is a snapshot pulled in October 2026, newest papers first.

### Cleaning pipeline
Cleaning lives in `src/preprocessing.py` as small, separate functions so each can be tested on its own:

| Function | What it does |
|---|---|
| `strip_html` | Removes HTML tags and the contents of `<script>`/`<style>` using BeautifulSoup, which tolerates broken markup |
| `normalize_unicode` | Repairs mojibake (for example `â€™`) with `ftfy`, normalizes to NFC form, and removes control characters while keeping tabs and newlines |
| `normalize_whitespace` | Collapses repeated spaces and tabs, limits blank lines to one paragraph break, trims the ends |
| `detect_language` | Returns an ISO language code with `langdetect` (seeded for repeatable results); returns `None` for text under 20 characters |
| `is_valid_document` | Rejects empty text, text under the minimum length (default 200 chars), and text outside the allowed languages (default English only), and reports the reason |
| `clean_text` / `clean_document` | Run the stages in order, then validate |

Cleaning and validation are separate steps, so every rejected document has a recorded reason. `build_corpus.py` prints these counts.

Title and abstract are combined into one `text` field so both are searchable later.

### Environment verification
`scripts/verify_env.py` imports pandas, numpy, sentence-transformers, ChromaDB, spaCy, BeautifulSoup, ftfy and langdetect, loads the spaCy English model, and **asserts CUDA is available**. If no GPU is detected it prints FAIL and exits with code 1. Library versions are printed so environment drift is easy to spot.

## Project structure

```
RAG-assistant/
├── scripts/
│   ├── verify_env.py       # environment, GPU and import checks
│   ├── acquire_data.py     # downloads raw documents from the arXiv API
│   ├── build_corpus.py     # cleans and validates raw data into Parquet
│   ├── build_index.py      # (Week 2) chunks, embeds and indexes the corpus
│   └── benchmark_retrieval.py  # (Week 2) retrieval latency benchmark
├── src/
│   ├── preprocessing.py    # cleaning functions
│   ├── chunking.py         # (Week 2) recursive text splitter
│   ├── embeddings.py       # (Week 2) sentence-transformers wrapper with timing
│   ├── vector_store.py     # (Week 2) ChromaDB wrapper
│   └── search.py           # (Week 2) semantic search
├── tests/
│   ├── test_preprocessing.py
│   ├── test_chunking.py        # (Week 2)
│   └── test_vector_store.py    # (Week 2)
├── data/
│   ├── raw/                # git-ignored
│   └── processed/          # git-ignored
├── chroma_db/              # (Week 2) vector database, git-ignored
├── reports/                # (Week 2) timing and benchmark results
├── requirements.txt        # locked with pip freeze
├── verify_env_output.txt
└── README.md
```

`data/raw/*`, `data/processed/*`, `chroma_db/`, `.env` and `venv/` are listed in `.gitignore`.

## How to run

All commands were run in WSL2 (Ubuntu) with Python 3.12.

```bash
# 1. Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install locked dependencies and the spaCy model
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Verify the environment (fails if no GPU is found)
python scripts/verify_env.py

# 4. Acquire raw data into data/raw/arxiv_raw.jsonl (about 5 minutes)
python scripts/acquire_data.py --target 5500

# 5. Clean and validate into data/processed/clean_corpus.parquet
python scripts/build_corpus.py

# 6. Run the unit tests
python -m pytest tests/ -v
```

GPU note: `requirements.txt` pins `torch==2.4.1`. For CUDA 12.1, install it with `pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121`.

## Output dataset

`data/processed/clean_corpus.parquet` has 5,499 rows and these columns:

| Column | Description |
|---|---|
| `doc_id` | arXiv URL/ID |
| `source` | `arxiv` |
| `title` | paper title |
| `text` | cleaned title and abstract |
| `language` | detected language code |
| `char_count` | length of `text` |
| `categories` | arXiv categories, comma-separated |
| `published` | publication timestamp |

## Testing

`tests/test_preprocessing.py` contains 32 tests covering every cleaning function: normal input, empty input, malformed HTML, mojibake, control characters, NFC composition, whitespace edge cases, language detection (English, Spanish, too short), and the validation reasons. The tests use small made-up strings, so they run in under a second and need no data or network.

---

# Week 2: Chunking, Embeddings & Vector Database

This week turns the 5,499 clean documents into a searchable ChromaDB index:

1. A recursive chunking strategy, unit-tested
2. Embeddings with sentence-transformers, with timing logged
3. A persistent ChromaDB collection and a semantic search function
4. A benchmark script that logs retrieval latency for different kinds of queries
5. Handling for ChromaDB edge cases, each covered by a test

## How it works

```
clean_corpus.parquet -> chunk_document -> Embedder -> VectorStore (ChromaDB) -> semantic_search
                        (src/chunking)   (src/embeddings)  (src/vector_store)    (src/search)
```

## Chunking strategy

**Why chunk at all.** Abstracts average about 1,550 characters. One vector for a whole abstract blurs several ideas together, while a smaller chunk gives a more focused match and a tighter passage to hand to the LLM in Week 3. The embedding model also truncates anything past 256 tokens, so smaller chunks make sure no text is silently dropped.

**Method: recursive splitting** (`src/chunking.py`, written from scratch, no framework). The splitter tries the largest natural boundary first and only falls back to smaller ones for pieces that are still too long:

1. paragraph break (`\n\n`)
2. line break (`\n`)
3. sentence end (`. `)
4. space (between words)
5. single character (last resort)

Small pieces are packed together until the next one would exceed the limit. Compared with cutting every N characters, this avoids slicing sentences in half in most cases.

| Parameter | Value | Reason |
|---|---|---|
| `chunk_size` | 600 characters (about 120-150 tokens) | Well under the model's 256-token limit; 4.4 chunks per abstract on average (measured) |
| `chunk_overlap` | 150 characters | Repeats context across a boundary so an idea split by a cut is still findable |
| Unit | characters, not tokens | Simple and needs no tokenizer; close enough for this model |

**Overlap is made of whole sentences.** The tail carried into the next chunk is only kept if it fits within 150 characters. When the last sentence before a cut is longer than that, the next chunk starts with no overlap. I chose to keep sentences intact over guaranteeing overlap.

**Chunk records.** Each chunk gets the id `<doc_id>::<chunk_index>` plus metadata (`doc_id`, `chunk_index`, `title`, `source`, `categories`, `published`). Stable ids mean re-running the indexer overwrites chunks instead of duplicating them.

**Known limitation.** The title is joined to the abstract in `text`, so it is only part of the first chunk's embedded text. Later chunks carry the title in their metadata, but it is not embedded with them.

## Embeddings

Model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions). It is small and fast enough for a laptop GPU and works well for semantic search on English text. Vectors are L2-normalized and the collection uses cosine distance, so `score = 1 - distance` is cosine similarity. Embeddings are computed outside ChromaDB and passed in, so indexing and querying always use the same model. `Embedder.embed_timed` logs total time, chunks per second and milliseconds per chunk, after a warm-up call so one-time GPU start-up cost is not counted.

## ChromaDB edge cases handled

Every row has a test in `tests/test_vector_store.py` or `tests/test_chunking.py`.

| Edge case | Handling |
|---|---|
| Empty or whitespace-only document | Produces zero chunks |
| Text with no separators (one long string) | Falls back to character splitting; no chunk exceeds the limit |
| Invalid `chunk_size` / `chunk_overlap` | Raises `ValueError` |
| Empty-text chunk | Skipped with a warning, not embedded |
| Duplicate chunk ids in one batch | Chroma rejects these, so the last one wins |
| Re-running the indexer | `upsert` makes it idempotent; the count does not grow |
| Empty metadata dict | Chroma 0.5.20 rejects `{}` (found by a failing test); stored as no metadata |
| `None` or list values in metadata | `None` dropped, other non-primitive values converted to strings |
| NaN or infinite embeddings | Rejected with `ValueError` |
| Chunk and embedding counts differ | Rejected with `ValueError` |
| Large collections | Written in batches within Chroma's maximum batch size |
| `k` larger than the collection | Clipped to the number of stored chunks |
| Query on an empty collection | Returns an empty list |
| Empty query or `k <= 0` | Raises `ValueError` |
| Metadata filter with no matches | Returns an empty list |
| Invalid metadata filter | Re-raised as a clear `ValueError` |
| Restart | Data persists on disk; verified by reopening the database in a test |
| Collection names | Chroma requires 3-63 characters; its own error message is clear, so it is not wrapped |

## How to run Week 2

Requires the Week 1 setup and `data/processed/clean_corpus.parquet`. The first run downloads the embedding model (about 90 MB), so it needs internet access.

```bash
# 1. Build the index (chunk, embed, store). Use --limit 200 for a quick trial first
python scripts/build_index.py --reset

# 2. Benchmark retrieval latency
python scripts/benchmark_retrieval.py

# 3. Run all tests (Week 1 + Week 2)
python -m pytest tests/ -v
```

Outputs: `chroma_db/` (the vector database, git-ignored because of its size), `reports/embedding_timing.json`, `reports/retrieval_benchmark.json` and `reports/retrieval_benchmark.md`. Logs go to `logs/`. Anyone can rebuild the database with step 1.

## Results

Run on the full corpus, in WSL2 with an NVIDIA GeForce MX570 A (CUDA). Raw numbers are in `reports/`.

### Indexing

| Metric | Value |
|---|---|
| Documents | 5,499 |
| Chunks | 24,368 (4.43 per document) |
| Chunk size / overlap | 600 / 150 characters |
| Chunk length (mean, min, max) | 371.5, 3, 600 |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 (384 dimensions) |
| Device | cuda |
| Chunking time | 0.32 s |
| Embedding time | 51.7 s (471.4 chunks/s, 2.12 ms/chunk) |
| Ingestion time | 31.1 s |
| Chunks in ChromaDB | 24,368 |

### Retrieval benchmark

Index size: 24,368 chunks, top-5 search, 5 repeats per query (60 searches), after one warm-up query.

Overall latency: mean 20.31 ms, p50 20.08 ms, p95 28.73 ms, max 31.82 ms.

| Type | Query | Top-1 score | Embed (ms) | Search (ms) | Total (ms) |
|---|---|---|---|---|---|
| topic | transformer attention mechanism | 0.691 | 14.19 | 8.74 | 22.93 |
| topic | retrieval augmented generation for question answering | 0.722 | 14.73 | 8.82 | 23.55 |
| topic | reinforcement learning from human feedback | 0.622 | 14.43 | 10.15 | 24.58 |
| topic | graph neural networks for node classification | 0.687 | 14.05 | 9.16 | 23.21 |
| topic | diffusion models for image generation | 0.719 | 12.48 | 7.26 | 19.74 |
| topic | federated learning with differential privacy | 0.781 | 9.14 | 7.09 | 16.23 |
| question | How can large language models be made to hallucinate... | 0.737 | 10.08 | 6.72 | 16.8 |
| question | What methods exist for evaluating the reasoning abil... | 0.68 | 8.79 | 7.23 | 16.02 |
| long | We study how fine-tuning a pretrained language model... | 0.728 | 12.38 | 8.1 | 20.49 |
| short | LLM | 0.644 | 9.77 | 7.97 | 17.75 |
| off-topic | best pizza recipe with fresh basil | 0.19 | 11.77 | 9.58 | 21.35 |
| nonsense | qzxv blorf wibble | 0.378 | 11.57 | 9.5 | 21.07 |

The score is cosine similarity between the query and the best-matching chunk (1.0 would be identical meaning).

### What the results show

**Relevance.** Every on-topic query, whether a short phrase, a full question or a long paragraph, scored between 0.62 and 0.78. The off-topic pizza query scored 0.19 and the nonsense query 0.38. So on these 12 queries, a cutoff around 0.5 would cleanly separate "the corpus has something relevant" from "it does not". That is useful for Week 3, where the system must refuse questions it cannot answer from the sources. Two cautions: 12 queries is a small sample, and the nonsense string scored higher than the pizza query, so a score cutoff alone will not catch everything. The one-word query "LLM" scored lowest of the on-topic group (0.644), which suggests very short queries carry less signal.

**Latency.** A full search took about 20 ms on average, and the slowest of the 60 searches took under 32 ms. Roughly 60% of that time goes to embedding the query (about 12 ms on average) and 40% to the ChromaDB lookup (about 8 ms). Query length and type made little difference to the total, and even the long paragraph query took 20 ms.

**Effect of index size.** I also ran the benchmark on a 200-document trial index of 896 chunks. Mean latency there was 15.1 ms against 20.3 ms on the full index, and the ChromaDB lookup took 3.5 to 5.9 ms against 7 to 10 ms. Scores on the same queries were also lower on the small index (for example 0.487 against 0.691 for "transformer attention mechanism") because there were fewer documents to match. Embedding the query costs the same either way, so lookup time is the part that grows with the corpus.

**Indexing cost.** Embedding all 24,368 chunks took under a minute on the GPU, and writing them to ChromaDB took about 31 seconds, so rebuilding the index is cheap.

## Testing

77 tests pass (32 from Week 1, 45 new). The Week 2 tests cover the chunker, the vector store, search, persistence, and the index and benchmark pipelines. The vector-store tests use a small hashing embedder in place of the real model, so they run in about a second with no download. The real model is exercised by `build_index.py` and `benchmark_retrieval.py`.

## Limitations

- LaTeX markup from the abstracts (for example `50\%`, `$10 \times$`) is still in the text and therefore in the chunks and embeddings. A LaTeX-stripping step would be a good addition.
- Chunk length is measured in characters, so the token count per chunk varies a little with the text.
- A few chunks are very short. The smallest is 3 characters (mean length is 371.5), left over when a split ends on a tiny fragment. A minimum chunk size, or merging a tiny final piece into its neighbour, would remove them.
- A similarity cutoff does not perfectly detect off-topic queries: the nonsense query scored 0.378, higher than the off-topic pizza query at 0.19.
- ChromaDB 0.5.20 prints `Failed to send telemetry event` errors in the logs. They come from its anonymous telemetry client and do not affect results. Set `ANONYMIZED_TELEMETRY=False` to silence them.
- Retrieval is pure semantic search; there is no keyword matching or reranking yet.
