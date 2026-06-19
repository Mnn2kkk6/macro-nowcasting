"""
data_loader.py
--------------
Pulls macro series from FRED API, caches locally as CSV.
Run directly:  python -m src.data_loader
"""

import os
import logging
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from fredapi import Fred

# ── Setup ──────────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def get_fred_client() -> Fred:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY not found. Add it to your .env file:\n"
            "  FRED_API_KEY=your_key_here"
        )
    return Fred(api_key=api_key)


def fetch_series(fred: Fred, series_map: dict, start: str, end: str | None) -> pd.DataFrame:
    """
    Fetch multiple FRED series and return as a single DataFrame.
    Each series keeps its original frequency (will be aligned later in features.py).
    """
    frames = {}
    for name, series_id in series_map.items():
        log.info(f"  Fetching {name} ({series_id}) ...")
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            s.name = name
            frames[name] = s
        except Exception as e:
            log.warning(f"  Failed to fetch {series_id}: {e}")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def main():
    cfg = load_config()
    fred_cfg = cfg["fred"]
    fred = get_fred_client()

    # Combine targets + leading indicators into one map
    all_series = {**fred_cfg["target_series"], **fred_cfg["leading_indicators"]}

    log.info(f"Fetching {len(all_series)} series from FRED (start={fred_cfg['start_date']}) ...")
    df = fetch_series(fred, all_series, fred_cfg["start_date"], fred_cfg["end_date"])

    # Save raw (daily index — each column has NaN where its frequency doesn't apply)
    out_path = RAW_DIR / "macro_raw.csv"
    df.to_csv(out_path)
    log.info(f"Saved → {out_path}  ({df.shape[0]} rows × {df.shape[1]} cols)")

    # Quick summary
    print("\n── Missing value summary ──────────────────────────")
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(1)
    summary = pd.DataFrame({"missing": missing, "pct_%": pct, "dtype": df.dtypes})
    print(summary.to_string())
    print(f"\nDate range: {df.index.min().date()}  →  {df.index.max().date()}")


if __name__ == "__main__":
    main()
