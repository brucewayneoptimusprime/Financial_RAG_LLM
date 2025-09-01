# app/ingest/sec_map.py
"""
SEC ticker→CIK utilities.

- Downloads official mapping from:
    https://www.sec.gov/files/company_tickers.json

- Caches to: data/cache/sec/ticker_cik_map.json
- Provides: ticker_to_cik("AAPL") -> "0000320193" (10-digit, zero-padded)

Usage:
  python -m app.ingest.sec_map                 # refresh cache & report count
  python -m app.ingest.sec_map AAPL MSFT TSLA  # print CIKs for tickers
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
import urllib.request
import urllib.error

try:
    # optional: load .env if present (for SEC_USER_AGENT)
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


# ---- Paths ----
# This file lives at app/ingest/sec_map.py → project root is two levels up
ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache" / "sec"
CACHE_FILE = CACHE_DIR / "ticker_cik_map.json"

# ---- SEC endpoint ----
SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"

# ---- Headers ----
DEFAULT_UA = os.getenv("SEC_USER_AGENT", "FinancialDocAssistant (contact@example.com)")


def _ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _http_get_json(url: str, user_agent: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read().decode("utf-8", errors="replace")
        return json.loads(data)


def _build_ticker_map(sec_obj: dict) -> Dict[str, str]:
    """
    SEC JSON is indexed by numeric keys:
      {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    Return: { "AAPL": "0000320193", ... }
    """
    out: Dict[str, str] = {}
    for _, rec in sec_obj.items():
        try:
            ticker = str(rec["ticker"]).upper().strip()
            cik_str = str(rec["cik_str"]).strip()
            cik10 = cik_str.zfill(10)
            if ticker and cik10.isdigit():
                out[ticker] = cik10
        except Exception:
            continue
    return out


def refresh_cache(user_agent: Optional[str] = None) -> Dict[str, str]:
    """
    Download the SEC mapping and write to CACHE_FILE.
    Returns the mapping dict.
    """
    ua = user_agent or DEFAULT_UA
    if ua == "FinancialDocAssistant (contact@example.com)":
        # gentle reminder if the user forgot to set a proper UA
        print("⚠️  SEC_USER_AGENT not set in .env; using placeholder UA. "
              "Set SEC_USER_AGENT='Project Name (you@example.com)' for best results.")
    _ensure_dirs()
    sec_obj = _http_get_json(SEC_TICKER_URL, ua)
    mapping = _build_ticker_map(sec_obj)
    payload = {
        "source_url": SEC_TICKER_URL,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "entries": len(mapping),
        "map": mapping,
    }
    CACHE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return mapping


def _load_cache() -> Optional[Dict[str, str]]:
    if not CACHE_FILE.exists():
        return None
    try:
        obj = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        return obj.get("map") or {}
    except Exception:
        return None


def ticker_to_cik(ticker: str, auto_refresh: bool = True) -> Optional[str]:
    """
    Return 10-digit CIK for a given ticker, or None if not found.
    If the cache is missing and auto_refresh=True, it will download it.
    """
    ticker_up = ticker.strip().upper()
    mapping = _load_cache()
    if mapping is None and auto_refresh:
        mapping = refresh_cache()
    if not mapping:
        return None
    return mapping.get(ticker_up)


def main(argv: list[str]) -> None:
    if len(argv) <= 1:
        # no tickers passed → just refresh and report
        mapping = refresh_cache()
        print(f"✅ Refreshed ticker→CIK map: {len(mapping)} entries cached at {CACHE_FILE}")
        return

    # tickers provided → look them up (refresh if needed)
    mapping = _load_cache() or refresh_cache()
    for t in argv[1:]:
        cik = mapping.get(t.upper())
        if cik:
            print(f"{t.upper()}: {cik}")
        else:
            print(f"{t.upper()}: (not found)")

if __name__ == "__main__":
    main(sys.argv)
