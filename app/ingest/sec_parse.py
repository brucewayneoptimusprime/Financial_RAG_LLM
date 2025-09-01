# app/ingest/sec_parse.py
"""
Parse a downloaded SEC filing (HTML/TXT/PDF) into clean text.

Inputs (created by sec_fetch.py):
  data/raw/sec/<TICKER>/<ACCESSION>/source.(html|txt|pdf)

Outputs:
  data/interim/sec/<TICKER>/<ACCESSION>/parsed.txt
  data/interim/sec/<TICKER>/<ACCESSION>/page_map.json   # page offsets for PDFs; [] for HTML/TXT

Usage:
  # parse a specific accession
  python -m app.ingest.sec_parse --ticker AAPL --accession 0000320193-24-000123

  # parse the most recent downloaded filing for a ticker
  python -m app.ingest.sec_parse --ticker AAPL --latest
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import json
import argparse
import re

# project root: app/ingest/sec_parse.py -> parents[2]
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "sec"
INTERIM_DIR = ROOT / "data" / "interim" / "sec"

# Optional HTML parser
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None

# PDF parser
try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None


def _clean_whitespace(text: str) -> str:
    # normalize whitespace, keep paragraph breaks
    t = text.replace("\xa0", " ")
    # collapse runs of spaces/tabs
    t = re.sub(r"[ \t]+", " ", t)
    # collapse 3+ newlines to at most 2
    t = re.sub(r"\n{3,}", "\n\n", t)
    # strip trailing spaces per line
    t = "\n".join(line.rstrip() for line in t.splitlines())
    return t.strip()


def _extract_text_html(html: str) -> str:
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        # remove script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text("\n")
    else:
        # very crude fallback: strip tags
        text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
        text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
    return _clean_whitespace(text)


def _extract_text_txt(txt: str) -> str:
    return _clean_whitespace(txt)


def _extract_text_pdf(pdf_path: Path) -> Tuple[str, List[Dict]]:
    if PdfReader is None:
        raise SystemExit("pypdf not installed. Please `pip install pypdf`.")

    reader = PdfReader(str(pdf_path))
    pages = []
    out_lines = []
    offset = 0

    for i, page in enumerate(reader.pages, start=1):
        try:
            ptxt = page.extract_text() or ""
        except Exception:
            ptxt = ""
        ptxt = _clean_whitespace(ptxt)
        if ptxt:
            out_lines.append(ptxt)
            start = offset
            offset += len(ptxt) + 1  # +1 for the newline we’ll join with
            pages.append({"page": i, "start": start, "end": offset})
        else:
            # even if page blank, record a small span so mapping stays monotonic
            pages.append({"page": i, "start": offset, "end": offset})

    full_text = "\n".join(out_lines).strip()
    return full_text, pages


def parse_one(ticker: str, accession: str) -> Path:
    """
    Parse one filing folder to interim outputs. Returns the interim folder path.
    """
    raw_dir = RAW_DIR / ticker.upper() / accession
    if not raw_dir.exists():
        raise SystemExit(f"Raw folder not found: {raw_dir}")

    # detect source file
    src = None
    for ext in (".html", ".htm", ".txt", ".pdf"):
        cand = raw_dir / f"source{ext}"
        if cand.exists():
            src = cand
            break
    if src is None:
        # try to find any primary doc by name
        for p in raw_dir.glob("*"):
            if p.suffix.lower() in {".html", ".htm", ".txt", ".pdf"}:
                src = p
                break
    if src is None:
        raise SystemExit(f"No source file found in {raw_dir}")

    interim_dir = INTERIM_DIR / ticker.upper() / accession
    interim_dir.mkdir(parents=True, exist_ok=True)

    page_map: List[Dict] = []
    if src.suffix.lower() in {".html", ".htm"}:
        text = _extract_text_html(src.read_text(encoding="utf-8", errors="replace"))
    elif src.suffix.lower() == ".txt":
        text = _extract_text_txt(src.read_text(encoding="utf-8", errors="replace"))
    elif src.suffix.lower() == ".pdf":
        text, page_map = _extract_text_pdf(src)
    else:
        raise SystemExit(f"Unsupported file type: {src.suffix}")

    # write outputs
    parsed_path = interim_dir / "parsed.txt"
    parsed_path.write_text(text, encoding="utf-8")

    page_map_path = interim_dir / "page_map.json"
    page_map_path.write_text(json.dumps(page_map, indent=2), encoding="utf-8")

    print(f"✅ Parsed → {parsed_path}")
    print(f"🗂  Page map → {page_map_path} ({len(page_map)} pages)" if page_map else f"🗂  Page map → {page_map_path} (empty)")
    return interim_dir


def _most_recent_accession(ticker: str) -> str:
    base = RAW_DIR / ticker.upper()
    if not base.exists():
        raise SystemExit(f"No raw filings for ticker {ticker}")
    # pick the folder with the newest modified time
    dirs = [d for d in base.iterdir() if d.is_dir()]
    if not dirs:
        raise SystemExit(f"No accession folders under {base}")
    latest = max(dirs, key=lambda p: p.stat().st_mtime)
    return latest.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, help="e.g., AAPL")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--accession", help="e.g., 0000320193-24-000123")
    g.add_argument("--latest", action="store_true", help="parse the most recent downloaded filing for this ticker")
    args = ap.parse_args()

    accession = args.accession or _most_recent_accession(args.ticker)
    parse_one(args.ticker, accession)


if __name__ == "__main__":
    main()
