"""
main.py
FastAPI application — fully wired from Phase 4 onward.

Endpoints:
  GET  /             — service info
  GET  /health       — health check
  POST /api/research — run full 3-agent research pipeline
  GET  /api/history  — retrieve past reports (DB stub until Phase 6)
  GET  /api/cse      — list CSE stocks
  GET  /api/cse/{ticker} — get single CSE stock
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import get_settings
from core.models import (
    ErrorResponse,
    HealthResponse,
    HistoryResponse,
    ResearchReport,
    ResearchRequest,
)
from data.cse_loader import get_cse_stock, list_cse_stocks, get_cse_sectors
from orchestrator import run_research_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()

# In-memory report store (replaced by real DB in Phase 6)
_report_store: list[dict] = []


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Multi-Agent Stock Research System — starting up")
    logger.info("   Groq model  : %s", settings.groq_model)
    logger.info("   CORS origins: %s", settings.cors_origins_list)
    logger.info("   GNews key   : %s", "configured" if settings.gnews_api_key else "NOT SET")
    yield
    logger.info("🛑 Shutting down")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multi-Agent Stock Research System",
    description=(
        "AI-powered financial research platform. "
        "Submit a ticker to receive a structured BUY/HOLD/SELL research report "
        "produced by three coordinated AI agents."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Timing middleware ────────────────────────────────────────────────────────
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.perf_counter() - t0:.3f}s"
    return response


# ─── System endpoints ─────────────────────────────────────────────────────────
@app.get("/", tags=["System"])
async def root():
    return {
        "service": "Multi-Agent Stock Research System",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Returns service health status."""
    return HealthResponse(status="ok", version="1.0.0")


# ─── Research endpoint ────────────────────────────────────────────────────────
@app.post(
    "/api/research",
    response_model=ResearchReport,
    tags=["Research"],
    summary="Run full stock research pipeline",
    responses={
        200: {"description": "Full research report with BUY/HOLD/SELL verdict"},
        400: {"model": ErrorResponse, "description": "Invalid ticker"},
        500: {"model": ErrorResponse, "description": "Pipeline error"},
    },
)
async def research(request: ResearchRequest):
    """
    Submit a stock ticker to run the full multi-agent research pipeline.

    - **ticker**: Stock symbol (e.g. `AAPL`, `7203.T`, `JKH.N`)
    - **exchange**: Optional exchange hint (`AUTO` detects from ticker suffix)

    Returns a complete research report within ~30-45 seconds.
    """
    ticker = request.ticker.strip().upper()

    # Basic input validation
    if not ticker or len(ticker) > 20:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ticker: '{request.ticker}'. Must be 1-20 characters.",
        )

    # Block obviously invalid characters
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.")
    if not all(c in allowed for c in ticker):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ticker: '{ticker}'. Only letters, digits, and dots allowed.",
        )

    logger.info("POST /api/research | ticker=%s | exchange=%s", ticker, request.exchange)

    try:
        report = await run_research_pipeline(request)
    except Exception as exc:
        logger.exception("Pipeline error for ticker %s: %s", ticker, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Research pipeline failed: {exc}",
        )

    # Store in memory (Phase 6 will replace with DB write)
    _report_store.append(report.model_dump(mode="json"))

    return report


# ─── History endpoint ─────────────────────────────────────────────────────────
@app.get(
    "/api/history",
    response_model=HistoryResponse,
    tags=["History"],
    summary="Retrieve past research reports",
)
async def history(
    ticker: Optional[str] = Query(None, description="Filter by ticker symbol"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=50, description="Reports per page"),
):
    """
    Returns paginated list of previously generated research reports.

    Phase 6 will replace the in-memory store with Supabase PostgreSQL.
    """
    reports = list(reversed(_report_store))  # newest first

    if ticker:
        ticker_up = ticker.strip().upper()
        reports = [r for r in reports if r.get("ticker") == ticker_up]

    total = len(reports)
    start = (page - 1) * page_size
    paginated = reports[start: start + page_size]

    return HistoryResponse(
        reports=paginated,
        total=total,
        page=page,
        page_size=page_size,
    )


# ─── CSE endpoints ────────────────────────────────────────────────────────────
@app.get(
    "/api/cse",
    tags=["CSE"],
    summary="List all Colombo Stock Exchange stocks",
)
async def cse_list(
    sector: Optional[str] = Query(None, description="Filter by sector"),
):
    """
    Returns the full CSE stock dataset (50 stocks).
    Optionally filtered by sector.
    """
    stocks = list_cse_stocks(sector=sector)
    sectors = get_cse_sectors()
    return {
        "stocks": stocks,
        "total": len(stocks),
        "sectors": sectors,
        "note": "CSE data is manually maintained and may not reflect real-time prices.",
    }


@app.get(
    "/api/cse/{ticker}",
    tags=["CSE"],
    summary="Get a single CSE stock by ticker",
)
async def cse_ticker(ticker: str):
    """
    Returns static fundamental data for a single CSE ticker (e.g. `JKH.N`).
    Returns 404 if the ticker is not in the dataset.
    """
    stock = get_cse_stock(ticker)
    if not stock:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker.upper()}' not found in CSE dataset.",
        )
    return stock


# ─── Dev runner ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
    )
