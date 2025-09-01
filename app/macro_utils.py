# app/macro_utils.py
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MACRO_DIR = ROOT / "data" / "macro"

def latest_value(series: str):
    """Return latest date + value from a macro CSV (raw level)."""
    path = MACRO_DIR / f"{series}.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna()
    latest = df.iloc[-1]
    return latest["date"].strftime("%Y-%m-%d"), latest["value"]

def latest_yoy(series: str):
    """Return latest date + YoY % change."""
    path = MACRO_DIR / f"{series}.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna()
    df["yoy"] = df["value"].pct_change(periods=12) * 100
    latest = df.iloc[-1]
    return latest["date"].strftime("%Y-%m-%d"), latest["yoy"]
