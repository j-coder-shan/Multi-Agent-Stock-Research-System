"""
tests/test_orchestrator.py
Tests for orchestrator.py — verifies parallel execution, error isolation,
and full pipeline output shape.
"""
import pytest
import sys, os
from unittest.mock import patch, AsyncMock, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator import run_research_pipeline
from core.models import (
    ResearchRequest,
    ResearchReport,
    Exchange,
    Verdict,
    Confidence,
    Sentiment,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_request(ticker="AAPL", exchange=Exchange.AUTO) -> ResearchRequest:
    return ResearchRequest(ticker=ticker, exchange=exchange)


def _make_mock_news():
    from core.models import NewsAgentOutput
    return NewsAgentOutput(
        ticker="AAPL",
        headlines=[],
        overall_sentiment=Sentiment.POSITIVE,
        key_events=["Strong earnings reported"],
    )


def _make_mock_financials():
    from core.models import FinancialsAgentOutput
    return FinancialsAgentOutput(
        ticker="AAPL",
        company_name="Apple Inc.",
        exchange="NYSE / NASDAQ",
        pe_ratio=32.1,
        current_price=195.5,
        data_source="yfinance",
    )


def _make_mock_synthesis():
    from core.models import SynthesisAgentOutput
    return SynthesisAgentOutput(
        verdict=Verdict.BUY,
        confidence=Confidence.HIGH,
        reasoning="Strong fundamentals and positive news sentiment.",
        risks=["Risk A", "Risk B", "Risk C"],
    )


# ─── Happy path ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_returns_research_report():
    """Full pipeline with all agents mocked should return a valid ResearchReport."""
    mock_news = _make_mock_news()
    mock_fin  = _make_mock_financials()
    mock_syn  = _make_mock_synthesis()

    with patch("orchestrator.run_news_agent", new_callable=AsyncMock, return_value=mock_news), \
         patch("orchestrator.run_financials_agent", new_callable=AsyncMock, return_value=mock_fin), \
         patch("orchestrator.run_synthesis_agent", new_callable=AsyncMock, return_value=mock_syn):
        result = await run_research_pipeline(_make_request("AAPL"))

    assert isinstance(result, ResearchReport)
    assert result.ticker == "AAPL"
    assert result.news is not None
    assert result.financials is not None
    assert result.synthesis is not None
    assert result.synthesis.verdict == Verdict.BUY
    assert result.id is not None
    assert result.processing_time_seconds is not None

@pytest.mark.asyncio
async def test_pipeline_ticker_normalised_to_uppercase():
    mock_news = _make_mock_news()
    mock_fin  = _make_mock_financials()
    mock_syn  = _make_mock_synthesis()

    with patch("orchestrator.run_news_agent", new_callable=AsyncMock, return_value=mock_news), \
         patch("orchestrator.run_financials_agent", new_callable=AsyncMock, return_value=mock_fin), \
         patch("orchestrator.run_synthesis_agent", new_callable=AsyncMock, return_value=mock_syn):
        result = await run_research_pipeline(_make_request("aapl"))

    assert result.ticker == "AAPL"

@pytest.mark.asyncio
async def test_pipeline_unique_report_ids():
    """Each pipeline run must produce a unique report ID."""
    mock_news = _make_mock_news()
    mock_fin  = _make_mock_financials()
    mock_syn  = _make_mock_synthesis()

    with patch("orchestrator.run_news_agent", new_callable=AsyncMock, return_value=mock_news), \
         patch("orchestrator.run_financials_agent", new_callable=AsyncMock, return_value=mock_fin), \
         patch("orchestrator.run_synthesis_agent", new_callable=AsyncMock, return_value=mock_syn):
        r1 = await run_research_pipeline(_make_request("AAPL"))
        r2 = await run_research_pipeline(_make_request("AAPL"))

    assert r1.id != r2.id


# ─── Error isolation ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_news_agent_failure_pipeline_continues():
    """News Agent crash must NOT abort the pipeline — Synthesis still runs."""
    mock_fin = _make_mock_financials()
    mock_syn = _make_mock_synthesis()

    with patch("orchestrator.run_news_agent", new_callable=AsyncMock, side_effect=Exception("GNews down")), \
         patch("orchestrator.run_financials_agent", new_callable=AsyncMock, return_value=mock_fin), \
         patch("orchestrator.run_synthesis_agent", new_callable=AsyncMock, return_value=mock_syn) as mock_syn_call:
        result = await run_research_pipeline(_make_request("AAPL"))

    assert result is not None
    assert result.news is None          # news failed
    assert result.financials is not None
    assert result.synthesis is not None
    # Synthesis was called with news=None
    call_args = mock_syn_call.call_args
    assert call_args[0][0] is None

@pytest.mark.asyncio
async def test_financials_agent_failure_pipeline_continues():
    """Financials Agent crash must NOT abort the pipeline."""
    mock_news = _make_mock_news()
    mock_syn  = _make_mock_synthesis()

    with patch("orchestrator.run_news_agent", new_callable=AsyncMock, return_value=mock_news), \
         patch("orchestrator.run_financials_agent", new_callable=AsyncMock, side_effect=Exception("yfinance down")), \
         patch("orchestrator.run_synthesis_agent", new_callable=AsyncMock, return_value=mock_syn):
        result = await run_research_pipeline(_make_request("AAPL"))

    assert result.news is not None
    assert result.financials is None    # financials failed
    assert result.synthesis is not None

@pytest.mark.asyncio
async def test_both_agents_fail_synthesis_still_runs():
    """Both News + Financials failing — Synthesis receives (None, None) and produces fallback."""
    mock_syn = _make_mock_synthesis()

    with patch("orchestrator.run_news_agent", new_callable=AsyncMock, side_effect=Exception("crash")), \
         patch("orchestrator.run_financials_agent", new_callable=AsyncMock, side_effect=Exception("crash")), \
         patch("orchestrator.run_synthesis_agent", new_callable=AsyncMock, return_value=mock_syn):
        result = await run_research_pipeline(_make_request("AAPL"))

    assert result.news is None
    assert result.financials is None
    assert result.synthesis is not None


# ─── Processing time ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_processing_time_recorded():
    """processing_time_seconds must be a positive float."""
    mock_news = _make_mock_news()
    mock_fin  = _make_mock_financials()
    mock_syn  = _make_mock_synthesis()

    with patch("orchestrator.run_news_agent", new_callable=AsyncMock, return_value=mock_news), \
         patch("orchestrator.run_financials_agent", new_callable=AsyncMock, return_value=mock_fin), \
         patch("orchestrator.run_synthesis_agent", new_callable=AsyncMock, return_value=mock_syn):
        result = await run_research_pipeline(_make_request("AAPL"))

    assert isinstance(result.processing_time_seconds, float)
    assert result.processing_time_seconds >= 0
