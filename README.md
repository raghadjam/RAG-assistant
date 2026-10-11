# RAG Assistant — Week 1: Environment & Data Cleaning Pipeline

Enterprise RAG system (ArXiv corpus → ChromaDB → DeepSeek/OpenRouter LLM →
FastAPI). This week covers environment setup, data acquisition, and a
validated, unit-tested text-cleaning pipeline.

## 1. Setup

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Lock the exact resolved versions for your machine
pip freeze > requirements.lock.txt
```

Create a `.env` file (git-ignored) for later weeks' API keys:
```
OPENROUTER_API_KEY=your_key_here
```

## 2. Verify the environment

```bash
python scripts/verify_env.py
```
Checks CUDA/GPU availability and that pandas, sentence-transformers,
ChromaDB, spaCy, and other core packages import correctly. GPU absence is
a warning, not a hard failure — CPU works fine for this stage.

## 3. Acquire raw data (5,000+ documents)

Requires real internet access to `export.arxiv.org` (not available in
sandboxed dev containers — run this on your local machine or CI runner):

```bash
python scripts/acquire_data.py --target 5500 --categories cs.AI cs.CL cs.LG stat.ML cs.IR
```
Writes raw, uncleaned records to `data/raw/arxiv_raw.jsonl` (git-ignored).

## 4. Build the clean, validated corpus

```bash
python scripts/build_corpus.py \
    --input data/raw/arxiv_raw.jsonl \
    --output data/processed/clean_corpus.parquet \
    --min-chars 200
```
Runs every document through `src/preprocessing.py`
(HTML stripping → Unicode normalization → whitespace cleanup → language
filtering), drops invalid/too-short/non-English documents, and writes the
validated result to `data/processed/clean_corpus.parquet`.

## 5. Run tests

```bash
pytest tests/ -v
pytest --cov=src --cov-report=term-missing   # with coverage
```

## Project layout

```
rag-assistant/
├── scripts/
│   ├── verify_env.py       # environment / GPU / import checks
│   ├── acquire_data.py     # pulls raw documents from arXiv API
│   └── build_corpus.py     # cleans + validates raw → data/processed/
├── src/
│   └── preprocessing.py    # modular cleaning functions (unit-tested)
├── tests/
│   └── test_preprocessing.py
├── data/
│   ├── raw/                # git-ignored
│   └── processed/          # git-ignored
├── requirements.txt
└── .gitignore
```

## Design notes

- **Cleaning pipeline is staged and composable**: `strip_html` →
  `normalize_unicode` → `normalize_whitespace` are independent, pure
  functions, each unit-tested in isolation, then composed in `clean_text`.
- **Language filtering** uses `langdetect` with a fixed seed for
  deterministic results; documents under 20 chars are treated as
  undetectable rather than guessed.
- **Validation is separate from cleaning**: `is_valid_document` gatekeeps
  on length/language *after* cleaning, so rejection reasons are logged
  and auditable (see `build_corpus.py` output).
- **ArXiv chosen as the data source** over Reddit/Wikipedia dumps: no
  auth required, small clean payloads, and abstracts are dense
  domain-specific prose that stresses the pipeline in realistic ways
  (LaTeX artifacts, HTML entities, inconsistent whitespace).
