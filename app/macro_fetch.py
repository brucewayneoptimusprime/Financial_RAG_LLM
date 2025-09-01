# app/macro_fetch.py
from pathlib import Path
import os
from dotenv import load_dotenv
from fredapi import Fred
import pandas as pd

# load .env
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("FRED_API_KEY")
if not API_KEY:
    raise ValueError("❌ Missing FRED_API_KEY in .env file")

fred = Fred(api_key=API_KEY)

OUT_DIR = ROOT / "data" / "macro"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_series(series_id: str, out_name: str):
    """Fetch a FRED series, save CSV, return DataFrame."""
    df = fred.get_series(series_id)
    df = df.to_frame(name="value")
    df.index.name = "date"
    out_path = OUT_DIR / f"{out_name}.csv"
    df.to_csv(out_path)
    print(f"✅ Saved {series_id} → {out_path}")
    return df

def main():
    # CPI All Urban Consumers (index)
    fetch_series("CPIAUCSL", "cpi")

    # Unemployment rate
    fetch_series("UNRATE", "unemployment")

if __name__ == "__main__":
    main()
