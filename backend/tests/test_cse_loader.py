"""
tests/test_cse_loader.py
Unit tests for data/cse_loader.py
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.cse_loader import get_cse_stock, list_cse_stocks, get_cse_sectors


# ─── get_cse_stock ────────────────────────────────────────────────────────────
def test_get_known_ticker():
    stock = get_cse_stock("JKH.N")
    assert stock is not None
    assert stock["ticker"] == "JKH.N"
    assert "company" in stock
    assert "sector" in stock

def test_get_known_ticker_lowercase():
    stock = get_cse_stock("jkh.n")
    assert stock is not None
    assert stock["ticker"] == "JKH.N"

def test_get_unknown_ticker_returns_none():
    stock = get_cse_stock("XXXXX.N")
    assert stock is None

def test_numeric_fields_are_float_or_none():
    stock = get_cse_stock("COMB.N")
    assert stock is not None
    # pe_ratio should be a float or None (not a string or NaN)
    pe = stock.get("pe_ratio")
    assert pe is None or isinstance(pe, float)

def test_no_nan_values_in_result():
    """All None-able fields must be Python None, never float NaN."""
    import math
    stock = get_cse_stock("JKH.N")
    assert stock is not None
    for key, val in stock.items():
        if isinstance(val, float):
            assert not math.isnan(val), f"Field {key} is NaN"


# ─── list_cse_stocks ──────────────────────────────────────────────────────────
def test_list_all_returns_nonempty():
    stocks = list_cse_stocks()
    assert len(stocks) > 0

def test_list_all_has_expected_minimum():
    stocks = list_cse_stocks()
    assert len(stocks) >= 30

def test_list_filter_by_sector():
    banking = list_cse_stocks(sector="Banking & Finance")
    assert len(banking) > 0
    for stock in banking:
        assert stock["sector"] == "Banking & Finance"

def test_list_filter_unknown_sector_returns_empty():
    result = list_cse_stocks(sector="Nonexistent Sector XYZ")
    assert result == []


# ─── get_cse_sectors ──────────────────────────────────────────────────────────
def test_get_sectors_returns_list():
    sectors = get_cse_sectors()
    assert isinstance(sectors, list)
    assert len(sectors) > 0

def test_banking_sector_present():
    sectors = get_cse_sectors()
    assert "Banking & Finance" in sectors

def test_sectors_are_sorted():
    sectors = get_cse_sectors()
    assert sectors == sorted(sectors)
