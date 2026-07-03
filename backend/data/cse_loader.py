"""
data/cse_loader.py
Loads and serves CSE (Colombo Stock Exchange) static data from the bundled CSV.
The CSV is manually maintained and updated periodically.

Usage:
    from data.cse_loader import get_cse_stock, list_cse_stocks
"""

import logging
import os
from functools import lru_cache
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Absolute path to the CSV file (adjacent to this module)
_CSV_PATH = os.path.join(os.path.dirname(__file__), "cse_stocks.csv")


@lru_cache(maxsize=1)
def _load_dataframe() -> pd.DataFrame:
    """Load and cache the CSE CSV. Called once at first access."""
    if not os.path.exists(_CSV_PATH):
        logger.error("CSE CSV not found at %s", _CSV_PATH)
        return pd.DataFrame()

    df = pd.read_csv(_CSV_PATH)
    # Normalise ticker column to uppercase
    df["ticker"] = df["ticker"].str.upper().str.strip()
    logger.info("Loaded %d CSE stocks from CSV", len(df))
    return df


def get_cse_stock(ticker: str) -> Optional[dict]:
    """
    Look up a CSE stock by ticker.

    Args:
        ticker: e.g. 'JKH.N' or 'jkh.n'

    Returns:
        dict with stock data, or None if not found.
    """
    df = _load_dataframe()
    if df.empty:
        return None

    ticker_upper = ticker.strip().upper()
    row = df[df["ticker"] == ticker_upper]

    if row.empty:
        logger.warning("CSE ticker not found: %s", ticker_upper)
        return None

    record = row.iloc[0].to_dict()

    # Convert NaN → None for JSON serialisation
    return {
        k: (None if (isinstance(v, float) and pd.isna(v)) else v)
        for k, v in record.items()
    }


def list_cse_stocks(sector: Optional[str] = None) -> list[dict]:
    """
    Return all CSE stocks, optionally filtered by sector.

    Args:
        sector: Optional sector name to filter by (case-insensitive).

    Returns:
        List of stock dicts.
    """
    df = _load_dataframe()
    if df.empty:
        return []

    if sector:
        df = df[df["sector"].str.lower() == sector.lower()]

    records = df.to_dict(orient="records")

    # Convert NaN → None
    return [
        {k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in r.items()}
        for r in records
    ]


def get_cse_sectors() -> list[str]:
    """Return a sorted list of unique sectors in the CSE dataset."""
    df = _load_dataframe()
    if df.empty:
        return []
    return sorted(df["sector"].dropna().unique().tolist())
