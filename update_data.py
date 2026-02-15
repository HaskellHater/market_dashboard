"""
Daily ETL pipeline: download financial series from Yahoo Finance,
normalize, and store locally with metadata and logs.

Usage: python update_data.py [config_path]
Default config: etl_config.json
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REQUIRED_SERIES_KEYS = {"id", "name", "ticker", "source", "field"}


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    for i, s in enumerate(cfg.get("series", [])):
        missing = REQUIRED_SERIES_KEYS - s.keys()
        if missing:
            raise ValueError(f"Series index {i} ({s.get('id','?')}) missing keys: {missing}")
    return cfg


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str) -> logging.Logger:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / f"etl_{datetime.now().strftime('%Y%m%d')}.log"

    logger = logging.getLogger("etl")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ---------------------------------------------------------------------------
# Download (with retries)
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30), reraise=True)
def download_one(ticker: str, start_date: str, auto_adjust: bool) -> pd.DataFrame:
    """Download OHLCV for a single ticker. Raises on empty result."""
    df = yf.download(
        ticker,
        start=start_date,
        auto_adjust=auto_adjust,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"Empty data returned for {ticker}")
    # yfinance returns MultiIndex columns when downloading a single ticker
    # with certain versions; flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_series(series_cfg: dict, raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normalize raw download into standardized schema. Returns (df, warnings)."""
    warnings = []
    field = series_cfg.get("field", "close")

    # Pick price column
    # With auto_adjust=True, yfinance puts adjusted prices in "Close" (no "Adj Close").
    # With auto_adjust=False, "Adj Close" exists separately.
    if field == "adj_close":
        if "Adj Close" in raw_df.columns:
            price_col = "Adj Close"
        elif "Close" in raw_df.columns:
            warnings.append("Adj Close not available, falling back to Close")
            price_col = "Close"
        else:
            raise ValueError("Neither Adj Close nor Close found in data")
    else:
        if "Close" in raw_df.columns:
            price_col = "Close"
        else:
            raise ValueError("Close column not found in data")

    # Build output dataframe
    out = pd.DataFrame()
    out["date"] = raw_df.index.tz_localize(None).date if raw_df.index.tz else raw_df.index.date
    out["price"] = raw_df[price_col].values
    if "Volume" in raw_df.columns:
        out["volume"] = raw_df["Volume"].values

    # Forward-fill None/NaN values with the last valid observation
    out["price"] = out["price"].ffill()
    if "volume" in out.columns:
        out["volume"] = out["volume"].ffill()

    # Log return: ln(P_t / P_{t-1})
    out["return"] = np.log(out["price"] / out["price"].shift(1))

    # Clean: sort, drop duplicate dates (keep last)
    out = out.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)

    # Check min history
    min_years = series_cfg.get("min_history_years")
    if min_years and len(out) > 0:
        actual_days = (out["date"].iloc[-1] - out["date"].iloc[0]).days
        actual_years = actual_days / 365.25
        if actual_years < min_years:
            warnings.append(
                f"History shorter than requested {min_years} years (got {actual_years:.1f} years)"
            )

    return out, warnings


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def save_raw(df: pd.DataFrame, series_id: str, output_dir: str):
    path = Path(output_dir) / "raw" / f"{series_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def save_processed(df: pd.DataFrame, series_id: str, output_dir: str):
    path = Path(output_dir) / "processed" / f"{series_id}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Main ETL
# ---------------------------------------------------------------------------

def run_etl(config_path: str):
    cfg = load_config(config_path)
    settings = cfg.get("settings", {})
    output_dir = settings.get("output_dir", "data")
    log_dir = settings.get("log_dir", "logs")
    start_date = settings.get("start_date", "2001-01-01")

    log = setup_logging(log_dir)
    log.info(f"ETL started — config: {config_path}, {len(cfg['series'])} series")

    results = {}

    for s in cfg["series"]:
        sid = s["id"]
        ticker = s["ticker"]
        field = s.get("field", "close")
        auto_adjust = field != "adj_close"  # adj_close needs auto_adjust=False to get the column

        log.info(f"[{sid}] Downloading {ticker} (field={field}, auto_adjust={auto_adjust})")

        try:
            raw_df = download_one(ticker, start_date, auto_adjust)
        except (RetryError, Exception) as e:
            log.error(f"[{sid}] Download failed after retries: {e}")
            results[sid] = {
                "status": "failed",
                "last_available_date": None,
                "row_count": 0,
                "warnings": [],
                "error_message": str(e),
            }
            continue

        save_raw(raw_df, sid, output_dir)
        log.info(f"[{sid}] Raw saved ({len(raw_df)} rows)")

        try:
            processed_df, warns = process_series(s, raw_df)
        except Exception as e:
            log.error(f"[{sid}] Processing failed: {e}")
            results[sid] = {
                "status": "failed",
                "last_available_date": None,
                "row_count": 0,
                "warnings": [],
                "error_message": str(e),
            }
            continue

        save_processed(processed_df, sid, output_dir)

        last_date = str(processed_df["date"].iloc[-1]) if len(processed_df) > 0 else None
        status = "warning" if warns else "success"

        for w in warns:
            log.warning(f"[{sid}] {w}")

        log.info(f"[{sid}] Processed: {len(processed_df)} rows, last date: {last_date}")

        results[sid] = {
            "status": status,
            "last_available_date": last_date,
            "row_count": len(processed_df),
            "warnings": warns,
            "error_message": None,
        }

    # Write metadata
    metadata = {
        "last_update": datetime.now(timezone.utc).isoformat(),
        "series": results,
    }
    meta_path = Path(output_dir) / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    log.info(f"Metadata written to {meta_path}")

    # Final summary
    print("\n" + "=" * 70)
    print(f"{'ID':<15} {'STATUS':<10} {'ROWS':>7}  WARNINGS")
    print("-" * 70)
    for sid, r in results.items():
        w = "; ".join(r["warnings"]) if r["warnings"] else r.get("error_message") or ""
        print(f"{sid:<15} {r['status']:<10} {r['row_count']:>7}  {w}")
    print("=" * 70)

    failed = sum(1 for r in results.values() if r["status"] == "failed")
    if failed:
        log.warning(f"ETL completed with {failed} failure(s)")
    else:
        log.info("ETL completed successfully")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "etl_config.json"
    run_etl(config)
