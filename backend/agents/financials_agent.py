"""
agents/financials_agent.py
Phase 2: Stock Data Pipeline

Fetches stock fundamental data from two sources:
  1. yfinance   — for all supported exchanges (USA, Japan, UK, Germany, etc.)
  2. CSE CSV    — for Colombo Stock Exchange tickers (.N / .X / .CSE suffix)

Then calls Groq LLM to generate a concise valuation commentary paragraph.

Public API:
    run_financials_agent(ticker, exchange="AUTO") -> FinancialsAgentOutput
"""

import asyncio
import logging
from typing import Any, Optional

import yfinance as yf

from core.config import get_settings
from core.exchange_detector import detect_exchange
from core.llm_client import chat_completion
from core.models import FinancialsAgentOutput
from data.cse_loader import get_cse_stock

logger = logging.getLogger(__name__)
settings = get_settings()


# ─── LLM Prompt ───────────────────────────────────────────────────────────────

_VALUATION_SYSTEM_PROMPT = """
You are a seasoned equity research analyst. Given the following financial
metrics for a publicly listed company, write a concise 2-3 sentence valuation
commentary for a retail investor. Focus on:
  • Whether the stock appears cheap, fairly valued, or expensive relative to
    its reported earnings (use the PE ratio as a guide).
  • Any notable strengths or red flags in the data.
  • Keep the language clear and jargon-free.

Do NOT give a buy/sell recommendation. Do NOT include numbers that are not
in the data provided. If most fields are N/A, say the data is limited.
"""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_float(value: Any) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        f = float(value)
        return None if (f != f) else round(f, 4)   # NaN check
    except (TypeError, ValueError):
        return None


def _fmt_metrics(data: dict) -> str:
    """Format a metrics dict into a readable string for the LLM prompt."""
    lines = []
    labels = {
        "company_name":     "Company",
        "exchange":         "Exchange",
        "current_price":    "Current Price",
        "market_cap":       "Market Cap",
        "pe_ratio":         "P/E Ratio",
        "eps":              "EPS",
        "revenue":          "Revenue (TTM)",
        "debt_to_equity":   "Debt/Equity",
        "dividend_yield":   "Dividend Yield (%)",
        "fifty_two_week_high": "52-Week High",
        "fifty_two_week_low":  "52-Week Low",
    }
    for key, label in labels.items():
        val = data.get(key)
        lines.append(f"  {label}: {val if val is not None else 'N/A'}")
    return "\n".join(lines)


# ─── yfinance fetcher (runs in thread pool to avoid blocking async loop) ───────

def _fetch_yfinance_sync(ticker: str) -> dict:
    """
    Fetch fundamentals from yfinance (synchronous — called via asyncio executor).
    Returns a dict of raw values; None where data is missing.
    """
    info: dict = {}
    try:
        stock = yf.Ticker(ticker)
        raw = stock.info or {}
        info = {
            "company_name":       raw.get("longName") or raw.get("shortName") or ticker,
            "current_price":      _safe_float(raw.get("currentPrice") or raw.get("regularMarketPrice")),
            "market_cap":         _safe_float(raw.get("marketCap")),
            "pe_ratio":           _safe_float(raw.get("trailingPE") or raw.get("forwardPE")),
            "eps":                _safe_float(raw.get("trailingEps")),
            "revenue":            _safe_float(raw.get("totalRevenue")),
            "debt_to_equity":     _safe_float(raw.get("debtToEquity")),
            "dividend_yield":     _safe_float(
                                    (raw.get("dividendYield") or 0) * 100
                                  ),
            "fifty_two_week_high":_safe_float(raw.get("fiftyTwoWeekHigh")),
            "fifty_two_week_low": _safe_float(raw.get("fiftyTwoWeekLow")),
        }
    except Exception as exc:
        logger.warning("yfinance error for %s: %s", ticker, exc)
    return info


# ─── CSE CSV fetcher ──────────────────────────────────────────────────────────

def _fetch_cse_data(ticker: str) -> dict:
    """
    Fetch data from the bundled CSE CSV.
    Returns a dict of values matching the yfinance schema keys.
    """
    row = get_cse_stock(ticker)
    if not row:
        return {}

    # Map CSV columns → internal schema
    return {
        "company_name":       row.get("company") or ticker,
        "current_price":      None,   # CSE CSV does not include live price
        "market_cap":         _safe_float(row.get("market_cap_lkr_mn")),
        "pe_ratio":           _safe_float(row.get("pe_ratio")),
        "eps":                _safe_float(row.get("eps")),
        "revenue":            _safe_float(row.get("revenue_lkr_mn")),
        "debt_to_equity":     _safe_float(row.get("debt_to_equity")),
        "dividend_yield":     _safe_float(row.get("dividend_yield")),
        "fifty_two_week_high": None,
        "fifty_two_week_low":  None,
    }


# ─── Valuation commentary via Groq ────────────────────────────────────────────

async def _generate_valuation_commentary(metrics: dict, exchange_name: str) -> str:
    """Call Groq to write a 2-3 sentence valuation commentary."""
    if not settings.groq_api_key:
        return "Valuation commentary unavailable: Groq API key not configured."

    user_msg = (
        f"Here are the financial metrics for {metrics.get('company_name', 'this company')} "
        f"listed on the {exchange_name}:\n\n"
        f"{_fmt_metrics(metrics)}\n\n"
        "Please write a 2-3 sentence valuation commentary."
    )
    try:
        commentary = await chat_completion(
            _VALUATION_SYSTEM_PROMPT,
            user_msg,
            temperature=0.3,
            max_tokens=256,
        )
        return commentary.strip()
    except Exception as exc:
        logger.warning("Valuation commentary failed: %s", exc)
        return "Valuation commentary temporarily unavailable."


# ─── Public entry point ────────────────────────────────────────────────────────

async def run_financials_agent(ticker: str, exchange: str = "AUTO") -> FinancialsAgentOutput:
    """
    Fetch stock fundamentals and generate valuation commentary.

    Args:
        ticker:   Stock ticker symbol (e.g. 'AAPL', '7203.T', 'JKH.N')
        exchange: Exchange hint (default 'AUTO' = detect from suffix)

    Returns:
        FinancialsAgentOutput with all available metrics + LLM commentary.
    """
    ticker = ticker.strip().upper()
    exchange_info = detect_exchange(ticker)
    exchange_name = exchange_info.name

    logger.info(
        "Financials Agent: fetching %s from %s (CSE=%s)",
        ticker, exchange_name, exchange_info.is_cse
    )

    # ── Fetch raw fundamentals ────────────────────────────────────────────────
    if exchange_info.is_cse:
        raw = _fetch_cse_data(ticker)
        data_source = "cse_csv"
        if not raw:
            # CSE ticker not in CSV — return minimal output
            return FinancialsAgentOutput(
                ticker=ticker,
                company_name=ticker,
                exchange=exchange_name,
                data_source="cse_csv",
                valuation_commentary=(
                    f"{ticker} is not yet in the CSE dataset. "
                    "CSE data is manually maintained and may be incomplete."
                ),
            )
    else:
        # Run blocking yfinance call in a thread pool
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, _fetch_yfinance_sync, ticker)
        data_source = "yfinance"

    # ── Generate valuation commentary ─────────────────────────────────────────
    metrics_for_llm = {**raw, "company_name": raw.get("company_name", ticker), "exchange": exchange_name}
    commentary = await _generate_valuation_commentary(metrics_for_llm, exchange_name)

    return FinancialsAgentOutput(
        ticker=ticker,
        company_name=raw.get("company_name") or ticker,
        exchange=exchange_name,
        pe_ratio=raw.get("pe_ratio"),
        eps=raw.get("eps"),
        revenue=raw.get("revenue"),
        market_cap=raw.get("market_cap"),
        debt_to_equity=raw.get("debt_to_equity"),
        dividend_yield=raw.get("dividend_yield"),
        fifty_two_week_high=raw.get("fifty_two_week_high"),
        fifty_two_week_low=raw.get("fifty_two_week_low"),
        current_price=raw.get("current_price"),
        valuation_commentary=commentary,
        data_source=data_source,
    )
