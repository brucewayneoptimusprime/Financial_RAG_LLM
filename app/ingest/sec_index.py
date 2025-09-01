# app/ingest/sec_index.py
"""
Index parsed SEC filings into processed/chunks.jsonl.

Inputs (from previous steps):
  data/raw/sec/<TICKER>/<ACCESSION>/meta.json
  data/interim/sec/<TICKER>/<ACCESSION>/parsed.txt
  data/interim/sec/<TICKER>/<ACCESSION>/page_map.json   # may be empty for HTML/TXT

Outputs:
  processed/chunks.jsonl (appended)
  # Then run your existing: python app/build_index.py  (to refresh FAISS)

Usage:
  # index a specific accession
  python -m app.ingest.sec_index --ticker AAPL --accession 0000320193-24-000123

  # index the most recent parsed filing for a ticker
  python -m app.ingest.sec_index --ticker AAPL --latest
"""

from __future__ import annotations
from pathlib import Path
import json
import argparse
import re
from typing import List, Dict, Tuple, Optional

# project root: app/ingest/sec_index.py -> parents[2]
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "sec"
INTERIM_DIR = ROOT / "data" / "interim" / "sec"
PROCESSED_DIR = ROOT / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

def _most_recent_accession(ticker: str) -> str:
    base = INTERIM_DIR / ticker.upper()
    if not base.exists():
        raise SystemExit(f"No interim filings for ticker {ticker}")
    dirs = [d for d in base.iterdir() if d.is_dir()]
    if not dirs:
        raise SystemExit(f"No accession folders under {base}")
    latest = max(dirs, key=lambda p: p.stat().st_mtime)
    return latest.name

def _read_meta(ticker: str, accession: str) -> dict:
    meta_p = RAW_DIR / ticker.upper() / accession / "meta.json"
    if not meta_p.exists():
        raise SystemExit(f"Missing meta.json: {meta_p}")
    return json.loads(meta_p.read_text(encoding="utf-8"))

def _read_parsed(ticker: str, accession: str) -> str:
    p = INTERIM_DIR / ticker.upper() / accession / "parsed.txt"
    if not p.exists():
        raise SystemExit(f"Missing parsed.txt: {p}")
    return p.read_text(encoding="utf-8", errors="replace")

def _read_page_map(ticker: str, accession: str) -> List[dict]:
    p = INTERIM_DIR / ticker.upper() / accession / "page_map.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []

def _normalize_ws(text: str) -> str:
    t = text.replace("\xa0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def _chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    """Simple char-based splitter with sentence-friendly boundaries."""
    text = _normalize_ws(text)
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)

        # try to break at a sentence end near the end window
        window = text[start:end]
        break_pos = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
        if break_pos >= 0 and break_pos > max_chars * 0.6:
            end = start + break_pos + 2  # include the punctuation+space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(0, end - overlap)

    return chunks

def _derive_year(meta: dict) -> Optional[int]:
    # filingDate like "2023-11-03"
    d = meta.get("filingDate")
    if not d:
        return None
    try:
        return int(d.split("-")[0])
    except Exception:
        return None

def _make_doc_name(meta: dict) -> str:
    # e.g., "AAPL_10-K_2023.html" (even if actual source was .txt)
    issuer = meta.get("ticker", "UNKNOWN")
    form = meta.get("form", "UNK")
    year = _derive_year(meta) or 0
    return f"{issuer}_{form}_{year}.html"

# inside app/ingest/sec_index.py

def _load_existing_ids(path: Path) -> set:
    ids = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                _id = obj.get("id")
                if _id:
                    ids.add(_id)
            except Exception:
                continue
    return ids

def _append_chunks_dedup(chunks: List[dict]) -> int:
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing_ids(CHUNKS_PATH)
    to_write = [rec for rec in chunks if rec.get("id") not in existing]
    if not to_write:
        return 0
    with CHUNKS_PATH.open("a", encoding="utf-8") as f:
        for rec in to_write:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(to_write)

def index_one(ticker: str, accession: str) -> int:
    meta = _read_meta(ticker, accession)
    text = _read_parsed(ticker, accession)
    page_map = _read_page_map(ticker, accession)  # unused for HTML/TXT

    issuer = meta.get("ticker", ticker.upper())
    form = meta.get("form", "UNK")
    year = _derive_year(meta)
    accession_id = meta.get("accession", accession)
    doc_name = _make_doc_name(meta)

    raw_chunks = _chunk_text(text, max_chars=1200, overlap=150)

    out: List[dict] = []
    for i, chunk in enumerate(raw_chunks, start=1):
        rec = {
            "id": f"{issuer}_{form}_{year}_{accession_id}_c{i}",
            "doc": doc_name,
            "issuer": issuer,
            "form": form,
            "year": year,
            "page_start": 1,
            "page_end": 1,
            "text": chunk,
            "accession": accession_id,
        }
        out.append(rec)

    written = _append_chunks_dedup(out)
    print(f"✅ Appended {written} new chunks to {CHUNKS_PATH}")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, help="e.g., AAPL")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--accession", help="e.g., 0000320193-24-000123")
    g.add_argument("--latest", action="store_true", help="use most recent parsed filing")
    args = ap.parse_args()

    accession = args.accession or _most_recent_accession(args.ticker)
    index_one(args.ticker, accession)
    print("👉 Now rebuild the vector index:\n    python app/build_index.py")

if __name__ == "__main__":
    main()
