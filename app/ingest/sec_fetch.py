# app/ingest/sec_fetch.py
"""
Download latest SEC filing(s) for a ticker and form type.

Usage:
  python -m app.ingest.sec_fetch AAPL 10-K --limit 1

Outputs under:
  data/raw/sec/<TICKER>/<ACCESSION>/
    - source.html|.txt|.pdf       (primary document)
    - meta.json                   (cik, ticker, form, filingDate, accession, urls)
    - filing_index.json           (list of documents in the filing)
"""

from __future__ import annotations
import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import urllib.request
import urllib.error

from app.ingest.sec_map import ticker_to_cik, ROOT  # reuse ROOT from sec_map

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

RAW_DIR = ROOT / "data" / "raw" / "sec"
USER_AGENT = os.getenv("SEC_USER_AGENT", "FinancialDocAssistant (contact@example.com)")
REQ_DELAY = float(os.getenv("SEC_REQUEST_DELAY", "0.4"))

def _http_get(url: str, accept: str = "application/json") -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": accept},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        return resp.read()

def _http_get_json(url: str) -> Any:
    data = _http_get(url, accept="application/json")
    return json.loads(data.decode("utf-8", errors="replace"))

def _http_get_text(url: str) -> str:
    data = _http_get(url, accept="text/html, text/plain, */*")
    return data.decode("utf-8", errors="replace")

def _save_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)

def _save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def fetch_latest(ticker: str, form: str = "10-K", limit: int = 1) -> List[Path]:
    """
    Fetch latest `limit` filings of `form` for `ticker`.
    Returns list of directories written: data/raw/sec/<TICKER>/<ACCESSION>/
    """
    cik = ticker_to_cik(ticker)
    if not cik:
        raise SystemExit(f"Unknown ticker: {ticker}")

    # 1) submissions API
    subs_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    subs = _http_get_json(subs_url)
    time.sleep(REQ_DELAY)

    # 2) filter recent filings
    recent = subs.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accns = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    out_dirs: List[Path] = []
    picked = 0

    for form_i, accn, date, pdoc in zip(forms, accns, dates, primary_docs):
        if form_i != form:
            continue
        accession = accn.replace("/", "")  # e.g., 0000320193-23-000106
        # 3) build archive base
        # NOTE: SEC uses folders: data/<CIK without leading zeros>/<ACCESSION no dashes>/
        cik_no_zeros = str(int(cik))
        accn_nodash = accession.replace("-", "")
        base = f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{accn_nodash}/"

        # 4) download index page to list docs
        index_url = base  # listing directory renders an HTML index
        index_html = _http_get_text(index_url)
        time.sleep(REQ_DELAY)

        # best effort find all hrefs (simple scrape)
        hrefs = []
        for line in index_html.splitlines():
            line = line.strip()
            # crude anchor extraction
            if 'href="' in line.lower():
                start = line.lower().find('href="') + 6
                end = line.find('"', start)
                if end > start:
                    href = line[start:end]
                    if href and not href.startswith("?") and not href.startswith("/"):
                        hrefs.append(href)

        # 5) choose primary document (from submissions metadata)
        primary_doc_url = base + pdoc
        # fetch primary document bytes
        try:
            primary_bytes = _http_get(primary_doc_url, accept="*/*")
        except Exception:
            # as a fallback, try first HTML/TXT in hrefs
            cand = next((h for h in hrefs if h.lower().endswith((".htm", ".html", ".txt", ".pdf"))), None)
            if not cand:
                print(f"⚠️ No downloadable doc found for {ticker} {form} {accession}")
                continue
            primary_doc_url = base + cand
            primary_bytes = _http_get(primary_doc_url, accept="*/*")

        # 6) write to disk
        out_dir = RAW_DIR / ticker.upper() / accession
        out_dir.mkdir(parents=True, exist_ok=True)

        # source filename based on extension
        ext = ".html"
        low = primary_doc_url.lower()
        if low.endswith(".txt"):
            ext = ".txt"
        elif low.endswith(".pdf"):
            ext = ".pdf"

        _save_bytes(out_dir / f"source{ext}", primary_bytes)
        _save_text(out_dir / "filing_index.html", index_html)
        _save_json(out_dir / "filing_index.json", {"hrefs": hrefs})
        _save_json(out_dir / "meta.json", {
            "ticker": ticker.upper(),
            "cik": cik,
            "form": form,
            "filingDate": date,
            "accession": accession,
            "archiveBaseUrl": base,
            "primaryDocument": primary_doc_url,
            "downloadedAt": time.time(),
            "userAgent": USER_AGENT,
        })

        out_dirs.append(out_dir)
        picked += 1
        if picked >= limit:
            break

    return out_dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", help="e.g., AAPL, MSFT")
    ap.add_argument("form", nargs="?", default="10-K", help="e.g., 10-K, 10-Q")
    ap.add_argument("--limit", type=int, default=1, help="how many to download (default 1)")
    args = ap.parse_args()

    dirs = fetch_latest(args.ticker, args.form, args.limit)
    if not dirs:
        print("No filings downloaded.")
    else:
        for d in dirs:
            print(f"✅ Downloaded → {d}")

if __name__ == "__main__":
    main()
