"""
scripts/build_corpus.py

Reads raw documents from data/raw/*.jsonl, runs them through the cleaning
pipeline in src/preprocessing.py, drops invalid documents (empty, too
short, wrong language), and writes the validated result to
data/processed/clean_corpus.parquet (JSONL fallback also supported).

Run:
    python scripts/build_corpus.py \
        --input data/raw/arxiv_raw.jsonl \
        --output data/processed/clean_corpus.parquet \
        --min-chars 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing import clean_document  # noqa: E402


def load_raw_records(input_path: Path) -> list[dict]:
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_corpus(
    input_path: Path,
    output_path: Path,
    min_chars: int,
    allowed_languages: tuple[str, ...],
    text_field: str = "abstract",
) -> pd.DataFrame:
    raw_records = load_raw_records(input_path)
    print(f"Loaded {len(raw_records)} raw records from {input_path}")

    rows = []
    rejection_counts: dict[str, int] = {}

    for rec in tqdm(raw_records, desc="Cleaning documents"):
        raw_text = rec.get(text_field, "") or ""
        title = rec.get("title", "") or ""
        # Combine title + body so both are cleaned and searchable together.
        combined_raw = f"{title}\n\n{raw_text}" if title else raw_text

        cleaned_text, validation = clean_document(
            combined_raw, min_chars=min_chars, allowed_languages=allowed_languages
        )

        if not validation.is_valid:
            rejection_counts[validation.reason] = rejection_counts.get(validation.reason, 0) + 1
            continue

        rows.append({
            "doc_id": rec.get("id", ""),
            "source": rec.get("source", "unknown"),
            "title": title,
            "text": cleaned_text,
            "language": validation.language,
            "char_count": len(cleaned_text),
            "categories": ",".join(rec.get("categories", [])) if rec.get("categories") else "",
            "published": rec.get("published", ""),
        })

    df = pd.DataFrame(rows)

    print(f"\nValidated documents : {len(df)} / {len(raw_records)}")
    if rejection_counts:
        print("Rejections by reason:")
        for reason, count in sorted(rejection_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason:30s} {count}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    elif output_path.suffix in (".jsonl", ".json"):
        df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"Unsupported output extension: {output_path.suffix}")

    print(f"Wrote clean corpus to {output_path}")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the validated, cleaned corpus from raw documents.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/arxiv_raw.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/clean_corpus.parquet"))
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--languages", nargs="+", default=["en"])
    parser.add_argument("--text-field", default="abstract",
                         help="Key in the raw JSON record holding the main body text.")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}")
        print("Run scripts/acquire_data.py first to produce raw data.")
        return 1

    df = build_corpus(
        input_path=args.input,
        output_path=args.output,
        min_chars=args.min_chars,
        allowed_languages=tuple(args.languages),
        text_field=args.text_field,
    )

    if len(df) < 5000:
        print(f"\nWARNING: clean corpus has only {len(df)} documents (< 5000 target).")
        print("Acquire more raw documents or lower --min-chars.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
