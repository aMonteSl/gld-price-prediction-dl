"""Tests for TradeLogStore — JSONL persistence."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from gldpred.storage.trade_log import TradeLogEntry, TradeLogStore


@pytest.fixture()
def store(tmp_path):
    """Provide a TradeLogStore backed by a temp directory."""
    return TradeLogStore(base_dir=str(tmp_path / "log"))


def _entry(id_: str = "t1", asset: str = "GLD", signal: str = "BUY") -> TradeLogEntry:
    return TradeLogEntry(
        id=id_,
        asset=asset,
        signal=signal,
        confidence=75.0,
        entry_price=195.0,
        expected_return_pct=2.2,
        stop_loss_pct=3.0,
        take_profit_pct=5.0,
        investment=10_000.0,
        horizon=20,
        model_id="model_abc",
        model_label="TCN v1",
    )


class TestTradeLogStore:

    def test_append_and_load(self, store):
        entry = _entry()
        store.append(entry)
        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0].id == "t1"
        assert loaded[0].asset == "GLD"
        assert loaded[0].signal == "BUY"

    def test_multiple_entries(self, store):
        store.append(_entry("t1"))
        store.append(_entry("t2", asset="SLV"))
        store.append(_entry("t3", asset="BTC-USD", signal="AVOID"))
        assert store.count == 3

    def test_load_by_asset(self, store):
        store.append(_entry("t1", asset="GLD"))
        store.append(_entry("t2", asset="SLV"))
        store.append(_entry("t3", asset="GLD", signal="HOLD"))
        gld = store.load_by_asset("GLD")
        assert len(gld) == 2
        slv = store.load_by_asset("SLV")
        assert len(slv) == 1

    def test_load_open(self, store):
        e1 = _entry("t1")
        e2 = _entry("t2")
        e2.status = "closed"
        store.append(e1)
        store.append(e2)
        open_trades = store.load_open()
        assert len(open_trades) == 1
        assert open_trades[0].id == "t1"

    def test_close_trade(self, store):
        store.append(_entry("t1"))
        ok = store.close_trade("t1", actual_return_pct=3.5, actual_exit_price=201.0)
        assert ok
        loaded = store.load_all()
        assert loaded[0].status == "closed"
        assert loaded[0].actual_return_pct == 3.5
        assert loaded[0].actual_exit_price == 201.0
        assert loaded[0].outcome_date is not None

    def test_close_nonexistent_trade(self, store):
        store.append(_entry("t1"))
        ok = store.close_trade("nonexist", 0.0, 0.0)
        assert not ok

    def test_summary_stats_empty(self, store):
        stats = store.summary_stats()
        assert stats["total"] == 0
        assert stats["win_rate"] == 0.0

    def test_summary_stats_with_closed(self, store):
        e1 = _entry("t1")
        e1.status = "closed"
        e1.actual_return_pct = 5.0
        e2 = _entry("t2")
        e2.status = "closed"
        e2.actual_return_pct = -2.0
        e3 = _entry("t3")  # open
        store.append(e1)
        store.append(e2)
        store.append(e3)
        stats = store.summary_stats()
        assert stats["total"] == 3
        assert stats["open"] == 1
        assert stats["closed"] == 2
        assert stats["win_rate"] == 50.0
        assert stats["avg_return"] == 1.5

    def test_clear(self, store):
        store.append(_entry("t1"))
        store.clear()
        assert store.count == 0

    def test_load_empty_file(self, store):
        assert store.load_all() == []

    def test_entry_fields_roundtrip(self, store):
        entry = TradeLogEntry(
            id="test123",
            asset="BTC-USD",
            signal="HOLD",
            confidence=60.0,
            entry_price=43000.0,
            expected_return_pct=1.5,
            stop_loss_pct=5.0,
            take_profit_pct=8.0,
            investment=5000.0,
            horizon=10,
            model_id="m_btc",
            model_label="LSTM BTC v2",
            notes="test note",
        )
        store.append(entry)
        loaded = store.load_all()[0]
        assert loaded.id == "test123"
        assert loaded.asset == "BTC-USD"
        assert loaded.signal == "HOLD"
        assert loaded.notes == "test note"
        assert loaded.status == "open"
        assert loaded.actual_return_pct is None

    def test_persistence_across_instances(self, tmp_path):
        """Two instances sharing the same dir see same data."""
        base = str(tmp_path / "shared")
        s1 = TradeLogStore(base_dir=base)
        s1.append(_entry("t1"))
        s2 = TradeLogStore(base_dir=base)
        assert s2.count == 1

    def test_close_preserves_other_entries(self, store):
        store.append(_entry("t1"))
        store.append(_entry("t2", asset="SLV"))
        store.close_trade("t1", 3.0, 200.0)
        all_entries = store.load_all()
        assert len(all_entries) == 2
        assert all_entries[0].status == "closed"
        assert all_entries[1].status == "open"
