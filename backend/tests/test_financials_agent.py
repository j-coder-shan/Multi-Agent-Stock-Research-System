"""
tests/test_financials_agent.py
Integration tests for agents/financials_agent.py

Uses mocking to avoid real network calls — tests:
  - yfinance path (mock yf.Ticker)
  - CSE CSV path (real CSV, no network)
  - Missing ticker / invalid ticker graceful fallback
  - N/A handling when yfinance fields are absent
"""
import pytest
import sys, os
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.financials_agent import run_financials_agent, _safe_float, _fetch_cse_data
from core.models import FinancialsAgentOutput


# ─── _safe_float helper ───────────────────────────────────────────────────────
def test_safe_float_none():
    assert _safe_float(None) is None

def test_safe_float_nan():
    import math
    result = _safe_float(float("nan"))
    assert result is None

def test_safe_float_valid():
    assert _safe_float("18.5") == 18.5
    assert _safe_float(0) == 0.0

def test_safe_float_invalid_string():
    assert _safe_float("N/A") is None


# ─── CSE path (no network — uses real CSV) ────────────────────────────────────
@pytest.mark.asyncio
async def test_cse_ticker_found():
    """JKH.N is in the CSV — should return FinancialsAgentOutput without network calls."""
    with patch("agents.financials_agent.settings") as mock_settings, \
         patch("agents.financials_agent.chat_completion", new_callable=AsyncMock) as mock_llm:
        mock_settings.groq_api_key = "fake-key"
        mock_llm.return_value = "Mocked valuation commentary."
        result = await run_financials_agent("JKH.N")

    assert isinstance(result, FinancialsAgentOutput)
    assert result.ticker == "JKH.N"
    assert result.data_source == "cse_csv"
    assert result.valuation_commentary == "Mocked valuation commentary."
    assert "Colombo" in result.exchange


@pytest.mark.asyncio
async def test_cse_ticker_not_in_csv():
    """A CSE ticker not in the CSV should return graceful output, not raise."""
    with patch("agents.financials_agent.chat_completion", new_callable=AsyncMock):
        result = await run_financials_agent("UNKNOWN.N")

    assert isinstance(result, FinancialsAgentOutput)
    assert result.ticker == "UNKNOWN.N"
    assert result.data_source == "cse_csv"
    assert "not yet in the CSE dataset" in result.valuation_commentary


# ─── yfinance path (mocked) ───────────────────────────────────────────────────
_MOCK_YFINANCE_INFO = {
    "longName": "Apple Inc.",
    "currentPrice": 195.50,
    "marketCap": 3_050_000_000_000,
    "trailingPE": 32.1,
    "trailingEps": 6.08,
    "totalRevenue": 385_000_000_000,
    "debtToEquity": 175.0,
    "dividendYield": 0.005,
    "fiftyTwoWeekHigh": 237.23,
    "fiftyTwoWeekLow": 164.08,
    "regularMarketPrice": 195.50,
}


@pytest.mark.asyncio
async def test_yfinance_usa_ticker():
    """AAPL should go through yfinance path and produce valid output."""
    mock_ticker = MagicMock()
    mock_ticker.info = _MOCK_YFINANCE_INFO

    with patch("agents.financials_agent.yf.Ticker", return_value=mock_ticker), \
         patch("agents.financials_agent.settings") as mock_settings, \
         patch("agents.financials_agent.chat_completion", new_callable=AsyncMock) as mock_llm:
        mock_settings.groq_api_key = "fake-key"
        mock_settings.groq_model = "llama3-8b-8192"
        mock_llm.return_value = "Apple looks fairly valued."
        result = await run_financials_agent("AAPL")

    assert isinstance(result, FinancialsAgentOutput)
    assert result.ticker == "AAPL"
    assert result.company_name == "Apple Inc."
    assert result.data_source == "yfinance"
    assert result.pe_ratio == 32.1
    assert result.eps == 6.08
    assert result.current_price == 195.5
    assert result.market_cap == 3_050_000_000_000
    assert result.valuation_commentary == "Apple looks fairly valued."


@pytest.mark.asyncio
async def test_yfinance_japan_ticker():
    """Toyota (7203.T) — verify exchange detection + yfinance path."""
    mock_info = {**_MOCK_YFINANCE_INFO, "longName": "Toyota Motor Corporation"}
    mock_ticker = MagicMock()
    mock_ticker.info = mock_info

    with patch("agents.financials_agent.yf.Ticker", return_value=mock_ticker), \
         patch("agents.financials_agent.chat_completion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = "Toyota looks stable."
        result = await run_financials_agent("7203.T")

    assert result.ticker == "7203.T"
    assert "Tokyo" in result.exchange
    assert result.company_name == "Toyota Motor Corporation"
    assert result.data_source == "yfinance"


@pytest.mark.asyncio
async def test_missing_fields_return_none():
    """When yfinance returns an empty info dict, all optional fields should be None."""
    mock_ticker = MagicMock()
    mock_ticker.info = {}  # Empty — simulates missing data

    with patch("agents.financials_agent.yf.Ticker", return_value=mock_ticker), \
         patch("agents.financials_agent.chat_completion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = "Limited data available."
        result = await run_financials_agent("AAPL")

    assert result.pe_ratio is None
    assert result.eps is None
    assert result.revenue is None
    assert result.market_cap is None


@pytest.mark.asyncio
async def test_yfinance_exception_is_handled():
    """If yfinance raises, the agent should return output with None fields, not crash."""
    with patch("agents.financials_agent.yf.Ticker", side_effect=Exception("Network error")), \
         patch("agents.financials_agent.chat_completion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = "No data available."
        result = await run_financials_agent("BADTICKER")

    assert isinstance(result, FinancialsAgentOutput)
    assert result.pe_ratio is None


@pytest.mark.asyncio
async def test_llm_failure_does_not_crash_agent():
    """If Groq LLM fails, valuation_commentary should be a fallback string, not an exception."""
    mock_ticker = MagicMock()
    mock_ticker.info = _MOCK_YFINANCE_INFO

    with patch("agents.financials_agent.yf.Ticker", return_value=mock_ticker), \
         patch("agents.financials_agent.chat_completion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = RuntimeError("Groq API unavailable")
        result = await run_financials_agent("AAPL")

    assert isinstance(result, FinancialsAgentOutput)
    assert "unavailable" in result.valuation_commentary.lower()
