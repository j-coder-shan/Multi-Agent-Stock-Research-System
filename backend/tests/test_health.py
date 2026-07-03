"""
tests/test_health.py
Phase 1 smoke test — verifies the FastAPI app starts and /health responds.
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_root_returns_service_info():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Multi-Agent Stock Research System"
    assert data["status"] == "running"


def test_research_endpoint_exists():
    """Research endpoint is wired up (Phase 4+) — returns 200 or 422 for empty body."""
    # Sending no body → Pydantic validation error (422), proving endpoint exists
    response = client.post("/api/research", json={})
    assert response.status_code in (200, 400, 422)


def test_history_endpoint_exists():
    """History endpoint is wired up (Phase 4+) — returns 200 with empty list."""
    response = client.get("/api/history")
    assert response.status_code == 200
    data = response.json()
    assert "reports" in data
    assert "total" in data
