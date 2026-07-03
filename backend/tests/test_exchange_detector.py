"""
tests/test_exchange_detector.py
Unit tests for core/exchange_detector.py
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.exchange_detector import detect_exchange


# ─── USA ──────────────────────────────────────────────────────────────────────
def test_usa_no_suffix():
    info = detect_exchange("AAPL")
    assert info.name == "NYSE / NASDAQ"
    assert info.country == "US"
    assert info.currency == "USD"
    assert info.yf_ticker == "AAPL"
    assert not info.is_cse

def test_usa_lowercase_normalised():
    info = detect_exchange("tsla")
    assert info.yf_ticker == "TSLA"
    assert not info.is_cse

# ─── Japan ────────────────────────────────────────────────────────────────────
def test_japan_suffix():
    info = detect_exchange("7203.T")
    assert "Tokyo" in info.name
    assert info.country == "JP"
    assert info.currency == "JPY"
    assert info.yf_ticker == "7203.T"
    assert not info.is_cse

def test_japan_lowercase_suffix():
    info = detect_exchange("7203.t")
    assert "Tokyo" in info.name
    assert info.yf_ticker == "7203.T"

# ─── UK ───────────────────────────────────────────────────────────────────────
def test_uk_suffix():
    info = detect_exchange("VOD.L")
    assert "London" in info.name
    assert info.country == "GB"
    assert info.currency == "GBP"
    assert not info.is_cse

# ─── Germany ──────────────────────────────────────────────────────────────────
def test_germany_suffix():
    info = detect_exchange("SAP.DE")
    assert "XETRA" in info.name or "Frankfurt" in info.name
    assert info.country == "DE"
    assert info.currency == "EUR"
    assert not info.is_cse

# ─── CSE ──────────────────────────────────────────────────────────────────────
def test_cse_n_suffix():
    info = detect_exchange("JKH.N")
    assert "Colombo" in info.name
    assert info.country == "LK"
    assert info.currency == "LKR"
    assert info.is_cse

def test_cse_x_suffix():
    info = detect_exchange("TEST.X")
    assert info.is_cse
    assert info.country == "LK"

def test_cse_cse_suffix():
    info = detect_exchange("TEST.CSE")
    assert info.is_cse

# ─── Unknown suffix ───────────────────────────────────────────────────────────
def test_unknown_suffix_treated_as_usa():
    info = detect_exchange("XYZ.ZZ")
    assert info.country == "US"
    assert not info.is_cse

# ─── Whitespace stripping ─────────────────────────────────────────────────────
def test_whitespace_stripped():
    info = detect_exchange("  MSFT  ")
    assert info.yf_ticker == "MSFT"
