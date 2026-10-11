"""
scripts/acquire_data.py

Acquires a real-world text corpus from the arXiv public API (no API key
required) and saves the RAW, un-cleaned records to /data/raw/.

Why arXiv: full-text abstracts are freely available in bulk via a simple
REST API, license permits reuse for research/education, and results are
naturally rich, domain-specific English prose — good RAG material.

Run locally (this needs internet access to export.arxiv.org):
    python scripts/acquire_data.py --target 5500 --categories cs.AI cs.CL cs.LG stat.ML

Output:
    data/raw/arxiv_raw.jsonl   (one JSON object per line, one per paper)

Each record contains: id, title, abstract, authors, categories,
published, updated, raw_html_summary (kept as-is, uncleaned — cleaning
happens downstream in scripts/build_corpus.py using src/preprocessing.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

DEFAULT_CATEGORIES = ["cs.AI", "cs.CL", "cs.LG", "stat.ML", "cs.IR"]
PAGE_SIZE = 100          # arXiv API max per request is 2000, but be a good citizen
REQUEST_DELAY_SEC = 3.0  # arXiv asks for >=3s between requests


def fetch_page(category: str, start: int, page_size: int) -> str:
    query = f"search_query=cat:{category}&start={start}&max_results={page_size}"
    query += "&sortBy=submittedDate&sortOrder=descending"
    url = f"{ARXIV_API_URL}?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "rag-assistant-coursework/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def parse_entries(atom_xml: str) -> list[dict]:
    root = ET.fromstring(atom_xml)
    records = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        arxiv_id = entry.findtext(f"{ATOM_NS}id", default="").strip()
        title = (entry.findtext(f"{ATOM_NS}title", default="") or "").strip()
        summary = (entry.findtext(f"{ATOM_NS}summary", default="") or "").strip()
        published = entry.findtext(f"{ATOM_NS}published", default="")
        updated = entry.findtext(f"{ATOM_NS}updated", default="")

        authors = [
            (a.findtext(f"{ATOM_NS}name", default="") or "").strip()
            for a in entry.findall(f"{ATOM_NS}author")
        ]
        categories = [
            c.attrib.get("term", "")
            for c in entry.findall(f"{ATOM_NS}category")
        ]

        if not arxiv_id or not summary:
            continue

        records.append({
            "id": arxiv_id,
            "source": "arxiv",
            "title": title,
            "abstract": summary,   # raw, uncleaned text (may contain LaTeX, whitespace noise)
            "authors": authors,
            "categories": categories,
            "published": published,
            "updated": updated,
        })
    return records


def acquire(target: int, categories: list[str], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    written = 0

    with out_path.open("w", encoding="utf-8") as f:
        for category in categories:
            start = 0
            print(f"\n== Fetching category: {category} ==")
            while written < target:
                try:
                    xml_data = fetch_page(category, start, PAGE_SIZE)
                except (urllib.error.URLError, urllib.error.HTTPError) as exc:
                    print(f"  Request failed at start={start}: {exc}. Backing off 10s...")
                    time.sleep(10)
                    continue

                records = parse_entries(xml_data)
                if not records:
                    print(f"  No more results for {category} (stopped at start={start}).")
                    break

                new_count = 0
                for rec in records:
                    if rec["id"] in seen_ids:
                        continue
                    seen_ids.add(rec["id"])
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                    new_count += 1
                    if written >= target:
                        break

                print(f"  start={start:5d}  fetched={len(records):3d}  new={new_count:3d}  total_written={written}")
                start += PAGE_SIZE
                time.sleep(REQUEST_DELAY_SEC)

                if written >= target:
                    break
            if written >= target:
                break

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire raw arXiv documents for the RAG corpus.")
    parser.add_argument("--target", type=int, default=5500,
                         help="Minimum number of documents to collect (default: 5500, i.e. 5000+ with margin).")
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES,
                         help=f"arXiv category codes to pull from (default: {DEFAULT_CATEGORIES}).")
    parser.add_argument("--out", type=Path, default=Path("data/raw/arxiv_raw.jsonl"),
                         help="Output JSONL path (default: data/raw/arxiv_raw.jsonl).")
    args = parser.parse_args()

    print(f"Target documents : {args.target}")
    print(f"Categories       : {args.categories}")
    print(f"Output           : {args.out}")

    total = acquire(args.target, args.categories, args.out)

    print(f"\nDone. Wrote {total} raw documents to {args.out}")
    if total < 5000:
        print("WARNING: collected fewer than 5000 documents. Add more --categories or raise --target.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
