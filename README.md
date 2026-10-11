# RAG Assistant: Week 1 (Environment, Data Ingestion & Preprocessing)

This is the first stage of an end-to-end Retrieval-Augmented Generation (RAG) project. The full system will chunk and embed a document corpus into ChromaDB, retrieve context for user questions, generate grounded answers with an LLM, and serve everything through FastAPI.

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
│   └── build_corpus.py     # cleans and validates raw data into Parquet
├── src/
│   └── preprocessing.py    # cleaning functions
├── tests/
│   └── test_preprocessing.py
├── data/
│   ├── raw/                # git-ignored
│   └── processed/          # git-ignored
├── requirements.txt        # locked with pip freeze
├── verify_env_output.txt
└── README.md
```

`data/raw/*`, `data/processed/*`, `.env` and `venv/` are listed in `.gitignore`.

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

## Known limitations

- **LaTeX is not removed.** Abstracts contain markup such as `50\%`, `\emph{...}` and `$10 \times$`. The task asked for HTML, Unicode, whitespace and language cleaning, so this stays in the text. A LaTeX-stripping step is a good addition before embedding.
- **The language filter rejected nothing here.** arXiv computer science abstracts are almost all English. The filter is verified by unit tests (a Spanish text is rejected), not by this dataset.
- **The download script retries forever on network errors.** If arXiv is unreachable, `acquire_data.py` keeps retrying every 10 seconds instead of giving up. A retry limit should be added.
- **Short, mixed-language text can fool `langdetect`.** Documents under 20 characters are treated as undetectable.

## Next steps

Chunking strategies, embedding with sentence-transformers, and indexing into ChromaDB.
