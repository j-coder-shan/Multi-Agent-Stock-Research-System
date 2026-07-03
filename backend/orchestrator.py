"""
orchestrator.py
Phase 5 (wired here in Phase 4 for completeness): Multi-Agent Orchestrator

Runs the News Agent and Financials Agent in parallel using asyncio.gather(),
then feeds both outputs to the Synthesis Agent. Each agent is error-isolated —
a failure in one agent does not abort the pipeline; the Synthesis Agent handles
partial data gracefully.

Public API:
    run_research_pipeline(request: ResearchRequest) -> ResearchReport
"""

import asyncio
import logging
import time
import uuid

from agents.financials_agent import run_financials_agent
from agents.news_agent import run_news_agent
from agents.synthesis_agent import run_synthesis_agent
from core.models import (
    FinancialsAgentOutput,
    NewsAgentOutput,
    ResearchReport,
    ResearchRequest,
)

logger = logging.getLogger(__name__)


async def _safe_news_agent(ticker: str) -> NewsAgentOutput | None:
    """Run News Agent with full error isolation."""
    try:
        return await run_news_agent(ticker)
    except Exception as exc:
        logger.error("News Agent failed for %s: %s", ticker, exc)
        return None


async def _safe_financials_agent(ticker: str, exchange: str) -> FinancialsAgentOutput | None:
    """Run Financials Agent with full error isolation."""
    try:
        return await run_financials_agent(ticker, exchange)
    except Exception as exc:
        logger.error("Financials Agent failed for %s: %s", ticker, exc)
        return None


async def run_research_pipeline(request: ResearchRequest) -> ResearchReport:
    """
    Execute the full 3-agent research pipeline for a given ticker.

    Flow:
      1. News Agent + Financials Agent run in PARALLEL (asyncio.gather)
      2. Synthesis Agent runs SEQUENTIALLY after both complete
      3. Each agent is error-isolated — partial data is passed to Synthesis
      4. Total pipeline time is recorded in the report

    Args:
        request: ResearchRequest with ticker and optional exchange hint

    Returns:
        ResearchReport with all agent outputs and metadata
    """
    ticker = request.ticker.strip().upper()
    exchange = request.exchange.value
    report_id = str(uuid.uuid4())

    logger.info("Pipeline START | id=%s | ticker=%s | exchange=%s", report_id, ticker, exchange)
    t_start = time.perf_counter()

    # ── Stage 1: Parallel agent execution ─────────────────────────────────────
    logger.info("Stage 1: running News + Financials agents in parallel")
    news_result, financials_result = await asyncio.gather(
        _safe_news_agent(ticker),
        _safe_financials_agent(ticker, exchange),
    )

    t_parallel = time.perf_counter() - t_start
    logger.info(
        "Stage 1 complete in %.2fs | news=%s | financials=%s",
        t_parallel,
        "OK" if news_result else "FAILED",
        "OK" if financials_result else "FAILED",
    )

    # ── Stage 2: Synthesis ─────────────────────────────────────────────────────
    logger.info("Stage 2: running Synthesis Agent")
    synthesis_result = await run_synthesis_agent(news_result, financials_result)

    t_total = time.perf_counter() - t_start
    logger.info("Pipeline COMPLETE | id=%s | total=%.2fs", report_id, t_total)

    return ResearchReport(
        id=report_id,
        ticker=ticker,
        exchange=financials_result.exchange if financials_result else exchange,
        news=news_result,
        financials=financials_result,
        synthesis=synthesis_result,
        processing_time_seconds=round(t_total, 2),
    )
