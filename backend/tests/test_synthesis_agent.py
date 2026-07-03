"""
tests/test_synthesis_agent.py
Unit tests for agents/synthesis_agent.py

All Groq LLM calls are mocked — no real API keys required.
"""
import pytest
import sys, os
from unittest.mock import patch, AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.synthesis_agent import (
    run_synthesis_agent,
    _parse_verdict_json,
    _format_news_section,
    _format_financials_section,
    _DISCLAIMER,
)
from core.models import (
    Confidence,
    FinancialsAgentOutput,
    NewsAgentOutput,
    NewsItem,
    Sentiment,
    SynthesisAgentOutput,
    Verdict,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_news(ticker="AAPL") -> NewsAgentOutput:
    return NewsAgentOutput(
        ticker=ticker,
        headlines=[
            NewsItem(
                title="Apple beats earnings",
                url="https://example.com/1",
                published_at="2026-07-01T10:00:00Z",
                sentiment=Sentiment.POSITIVE,
                source="Reuters",
            )
        ],
        overall_sentiment=Sentiment.POSITIVE,
        key_events=["Record earnings beat analyst estimates"],
    )


def make_financials(ticker="AAPL") -> FinancialsAgentOutput:
    return FinancialsAgentOutput(
        ticker=ticker,
        company_name="Apple Inc.",
        exchange="NYSE / NASDAQ",
        pe_ratio=32.1,
        eps=6.08,
        revenue=385_000_000_000.0,
        market_cap=3_050_000_000_000.0,
        debt_to_equity=175.0,
        dividend_yield=0.5,
        fifty_two_week_high=237.23,
        fifty_two_week_low=164.08,
        current_price=195.5,
        valuation_commentary="Apple appears fairly valued at current PE.",
        data_source="yfinance",
    )


_VALID_LLM_RESPONSE = """{
  "verdict": "BUY",
  "confidence": "High",
  "reasoning": "Apple shows strong earnings momentum. The PE ratio is within historical norms. News sentiment is broadly positive.",
  "risks": [
    "Antitrust regulatory pressure in EU and US",
    "Slowing iPhone upgrade cycle",
    "USD strength may weigh on international revenues"
  ]
}"""


# ─── _parse_verdict_json ──────────────────────────────────────────────────────

def test_parse_valid_json():
    result = _parse_verdict_json(_VALID_LLM_RESPONSE)
    assert result is not None
    assert result["verdict"] == "BUY"
    assert result["confidence"] == "High"
    assert isinstance(result["risks"], list)
    assert len(result["risks"]) == 3

def test_parse_strips_markdown_fences():
    fenced = "```json\n" + _VALID_LLM_RESPONSE + "\n```"
    result = _parse_verdict_json(fenced)
    assert result is not None
    assert result["verdict"] == "BUY"

def test_parse_invalid_verdict_returns_none():
    bad = '{"verdict": "STRONG_BUY", "confidence": "High", "reasoning": "...", "risks": ["r1"]}'
    assert _parse_verdict_json(bad) is None

def test_parse_invalid_confidence_returns_none():
    bad = '{"verdict": "BUY", "confidence": "Very High", "reasoning": "...", "risks": ["r1"]}'
    assert _parse_verdict_json(bad) is None

def test_parse_missing_key_returns_none():
    bad = '{"verdict": "BUY", "confidence": "High", "reasoning": "..."}'
    assert _parse_verdict_json(bad) is None

def test_parse_empty_risks_returns_none():
    bad = '{"verdict": "BUY", "confidence": "High", "reasoning": "...", "risks": []}'
    assert _parse_verdict_json(bad) is None

def test_parse_garbage_returns_none():
    assert _parse_verdict_json("not json at all") is None

def test_parse_empty_string_returns_none():
    assert _parse_verdict_json("") is None


# ─── _format_news_section ─────────────────────────────────────────────────────

def test_format_news_not_none():
    news = make_news()
    text = _format_news_section(news)
    assert "AAPL" in text
    assert "Positive" in text
    assert "Record earnings" in text

def test_format_news_none():
    text = _format_news_section(None)
    assert "Not available" in text

def test_format_news_empty_headlines():
    news = NewsAgentOutput(
        ticker="AAPL",
        headlines=[],
        overall_sentiment=Sentiment.NEUTRAL,
        key_events=[],
    )
    text = _format_news_section(news)
    assert "AAPL" in text


# ─── _format_financials_section ───────────────────────────────────────────────

def test_format_financials_not_none():
    fin = make_financials()
    text = _format_financials_section(fin)
    assert "AAPL" in text
    assert "Apple Inc." in text
    assert "32.20" in text or "32.10" in text  # PE ratio

def test_format_financials_none_fields():
    fin = FinancialsAgentOutput(
        ticker="AAPL", company_name="Apple", exchange="NASDAQ",
        pe_ratio=None, eps=None, revenue=None, market_cap=None,
        debt_to_equity=None, dividend_yield=None,
        data_source="yfinance",
    )
    text = _format_financials_section(fin)
    assert "N/A" in text

def test_format_financials_none():
    text = _format_financials_section(None)
    assert "Not available" in text


# ─── run_synthesis_agent — happy path ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_happy_path_returns_verdict():
    news = make_news()
    fin = make_financials()

    with patch("agents.synthesis_agent.settings") as mock_settings, \
         patch("agents.synthesis_agent.chat_completion", new_callable=AsyncMock) as mock_llm:
        mock_settings.groq_api_key = "fake-key"
        mock_settings.groq_model = "llama3-8b-8192"
        mock_llm.return_value = _VALID_LLM_RESPONSE

        result = await run_synthesis_agent(news, fin)

    assert isinstance(result, SynthesisAgentOutput)
    assert result.verdict == Verdict.BUY
    assert result.confidence == Confidence.HIGH
    assert len(result.reasoning) > 10
    assert len(result.risks) == 3
    assert _DISCLAIMER in result.disclaimer

@pytest.mark.asyncio
async def test_disclaimer_always_injected():
    """Disclaimer must be present on every output regardless of LLM response."""
    with patch("agents.synthesis_agent.settings") as mock_settings, \
         patch("agents.synthesis_agent.chat_completion", new_callable=AsyncMock) as mock_llm:
        mock_settings.groq_api_key = "fake-key"
        mock_llm.return_value = _VALID_LLM_RESPONSE
        result = await run_synthesis_agent(make_news(), make_financials())

    assert result.disclaimer == _DISCLAIMER

@pytest.mark.asyncio
async def test_all_verdict_values_parsed():
    """BUY, HOLD, SELL must all round-trip correctly."""
    for v in ("BUY", "HOLD", "SELL"):
        response = _VALID_LLM_RESPONSE.replace('"BUY"', f'"{v}"')
        with patch("agents.synthesis_agent.settings") as mock_settings, \
             patch("agents.synthesis_agent.chat_completion", new_callable=AsyncMock) as mock_llm:
            mock_settings.groq_api_key = "fake-key"
            mock_llm.return_value = response
            result = await run_synthesis_agent(make_news(), make_financials())
        assert result.verdict == Verdict(v)

@pytest.mark.asyncio
async def test_all_confidence_values_parsed():
    for c in ("Low", "Medium", "Medium-High", "High"):
        response = _VALID_LLM_RESPONSE.replace('"High"', f'"{c}"')
        with patch("agents.synthesis_agent.settings") as mock_settings, \
             patch("agents.synthesis_agent.chat_completion", new_callable=AsyncMock) as mock_llm:
            mock_settings.groq_api_key = "fake-key"
            mock_llm.return_value = response
            result = await run_synthesis_agent(make_news(), make_financials())
        assert result.confidence == Confidence(c)


# ─── Retry logic ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retries_on_invalid_json():
    """First two calls return garbage; third returns valid JSON."""
    responses = ["not json", "also not json", _VALID_LLM_RESPONSE]
    idx = 0
    async def side_effect(*args, **kwargs):
        nonlocal idx
        r = responses[min(idx, len(responses) - 1)]
        idx += 1
        return r

    with patch("agents.synthesis_agent.settings") as mock_settings, \
         patch("agents.synthesis_agent.chat_completion", side_effect=side_effect):
        mock_settings.groq_api_key = "fake-key"
        result = await run_synthesis_agent(make_news(), make_financials())

    assert result.verdict == Verdict.BUY
    assert idx == 3   # 3 attempts were made

@pytest.mark.asyncio
async def test_all_retries_exhausted_returns_hold_fallback():
    """If all 3 attempts fail, return a safe HOLD fallback."""
    with patch("agents.synthesis_agent.settings") as mock_settings, \
         patch("agents.synthesis_agent.chat_completion", new_callable=AsyncMock) as mock_llm:
        mock_settings.groq_api_key = "fake-key"
        mock_llm.return_value = "invalid json every time"
        result = await run_synthesis_agent(make_news(), make_financials())

    assert result.verdict == Verdict.HOLD
    assert result.confidence == Confidence.LOW
    assert "Unable to generate" in result.reasoning


# ─── None inputs ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_none_news_does_not_crash():
    """Synthesis must complete even if news is None."""
    with patch("agents.synthesis_agent.settings") as mock_settings, \
         patch("agents.synthesis_agent.chat_completion", new_callable=AsyncMock) as mock_llm:
        mock_settings.groq_api_key = "fake-key"
        mock_llm.return_value = _VALID_LLM_RESPONSE
        result = await run_synthesis_agent(None, make_financials())

    assert isinstance(result, SynthesisAgentOutput)

@pytest.mark.asyncio
async def test_none_financials_does_not_crash():
    """Synthesis must complete even if financials is None."""
    with patch("agents.synthesis_agent.settings") as mock_settings, \
         patch("agents.synthesis_agent.chat_completion", new_callable=AsyncMock) as mock_llm:
        mock_settings.groq_api_key = "fake-key"
        mock_llm.return_value = _VALID_LLM_RESPONSE
        result = await run_synthesis_agent(make_news(), None)

    assert isinstance(result, SynthesisAgentOutput)

@pytest.mark.asyncio
async def test_both_none_returns_fallback():
    """Both None → HOLD fallback with clear message."""
    with patch("agents.synthesis_agent.settings") as mock_settings, \
         patch("agents.synthesis_agent.chat_completion", new_callable=AsyncMock) as mock_llm:
        mock_settings.groq_api_key = "fake-key"
        mock_llm.return_value = _VALID_LLM_RESPONSE
        result = await run_synthesis_agent(None, None)

    assert isinstance(result, SynthesisAgentOutput)


# ─── No API key ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_api_key_returns_hold_fallback():
    with patch("agents.synthesis_agent.settings") as mock_settings:
        mock_settings.groq_api_key = ""
        result = await run_synthesis_agent(make_news(), make_financials())

    assert result.verdict == Verdict.HOLD
    assert result.confidence == Confidence.LOW
    assert "not configured" in result.reasoning
