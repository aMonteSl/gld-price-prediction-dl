"""Tests for HealthService and ModelHealthReport.

Covers staleness verdicts, trade-log accuracy, training quality extraction,
recommendation generation, and the report_all_* methods.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from gldpred.services.health_service import (
    AGING_MAX_DAYS,
    FRESH_MAX_DAYS,
    STALE_MAX_DAYS,
    HealthService,
    ModelHealthReport,
    staleness_verdict,
)
from gldpred.registry.store import ModelRegistry
from gldpred.registry.assignments import ModelAssignments
from gldpred.storage.trade_log import TradeLogEntry, TradeLogStore


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_dir(tmp_path):
    """Create a temp directory tree for registry + assignments + trade log."""
    reg_dir = tmp_path / "registry"
    state_dir = tmp_path / "state"
    log_dir = tmp_path / "trade_log"
    reg_dir.mkdir()
    state_dir.mkdir()
    log_dir.mkdir()
    return reg_dir, state_dir, log_dir


def _make_model_dir(reg_dir: Path, model_id: str, **kwargs) -> None:
    """Create a minimal model entry in the registry directory."""
    d = reg_dir / model_id
    d.mkdir(parents=True, exist_ok=True)
    created_at = kwargs.get("created_at", datetime.now().isoformat())
    meta = {
        "model_id": model_id,
        "label": kwargs.get("label", f"test_{model_id}"),
        "asset": kwargs.get("asset", "GLD"),
        "architecture": kwargs.get("architecture", "TCN"),
        "created_at": created_at,
        "config": kwargs.get("config", {}),
        "feature_names": kwargs.get("feature_names", ["f1", "f2"]),
        "training_summary": kwargs.get("training_summary", {}),
        "evaluation_summary": kwargs.get("evaluation_summary", {}),
    }
    (d / "metadata.json").write_text(json.dumps(meta, indent=2))
    # dummy weights + scaler (won't load, but list_models only reads metadata)


def _make_trade(
    model_id: str,
    asset: str = "GLD",
    signal: str = "BUY",
    expected: float = 2.0,
    actual: float | None = None,
    status: str = "open",
) -> TradeLogEntry:
    return TradeLogEntry(
        id=f"t_{model_id}_{expected}",
        asset=asset,
        signal=signal,
        confidence=0.7,
        entry_price=100.0,
        expected_return_pct=expected,
        stop_loss_pct=-3.0,
        take_profit_pct=5.0,
        investment=10000,
        horizon=20,
        model_id=model_id,
        model_label=f"label_{model_id}",
        actual_return_pct=actual,
        status=status,
    )


# ── Staleness verdict (standalone) ──────────────────────────────────────

class TestStalenessVerdict:
    def test_fresh(self):
        assert staleness_verdict(0) == ("fresh", "green")
        assert staleness_verdict(FRESH_MAX_DAYS) == ("fresh", "green")

    def test_aging(self):
        assert staleness_verdict(FRESH_MAX_DAYS + 1) == ("aging", "orange")
        assert staleness_verdict(AGING_MAX_DAYS) == ("aging", "orange")

    def test_stale(self):
        assert staleness_verdict(AGING_MAX_DAYS + 1) == ("stale", "red")
        assert staleness_verdict(STALE_MAX_DAYS) == ("stale", "red")

    def test_expired(self):
        assert staleness_verdict(STALE_MAX_DAYS + 1) == ("expired", "gray")
        assert staleness_verdict(365) == ("expired", "gray")


# ── HealthService.report_for_model ──────────────────────────────────────

class TestReportForModel:
    def test_basic_report(self, tmp_dir):
        reg_dir, state_dir, log_dir = tmp_dir
        now = datetime(2025, 3, 1, 12, 0, 0)
        created = (now - timedelta(days=5)).isoformat()
        _make_model_dir(reg_dir, "m1", created_at=created, asset="GLD")

        svc = HealthService(
            registry=ModelRegistry(base_dir=reg_dir),
            assignments=ModelAssignments(state_dir=state_dir),
            trade_log=TradeLogStore(base_dir=str(log_dir)),
        )
        report = svc.report_for_model("m1", now=now)

        assert isinstance(report, ModelHealthReport)
        assert report.model_id == "m1"
        assert report.asset == "GLD"
        assert report.age_days == 5
        assert report.staleness == "fresh"
        assert report.staleness_color == "green"
        assert report.is_primary is False

    def test_primary_flag(self, tmp_dir):
        reg_dir, state_dir, log_dir = tmp_dir
        _make_model_dir(reg_dir, "m1", asset="GLD")
        assignments = ModelAssignments(state_dir=state_dir)
        assignments.assign("GLD", "m1")

        svc = HealthService(
            registry=ModelRegistry(base_dir=reg_dir),
            assignments=assignments,
            trade_log=TradeLogStore(base_dir=str(log_dir)),
        )
        report = svc.report_for_model("m1")
        assert report.is_primary is True

    def test_stale_model(self, tmp_dir):
        reg_dir, state_dir, log_dir = tmp_dir
        now = datetime(2025, 3, 1)
        created = (now - timedelta(days=25)).isoformat()
        _make_model_dir(reg_dir, "m1", created_at=created)

        svc = HealthService(
            registry=ModelRegistry(base_dir=reg_dir),
            assignments=ModelAssignments(state_dir=state_dir),
            trade_log=TradeLogStore(base_dir=str(log_dir)),
        )
        report = svc.report_for_model("m1", now=now)
        assert report.staleness == "stale"
        assert report.staleness_color == "red"

    def test_expired_model(self, tmp_dir):
        reg_dir, state_dir, log_dir = tmp_dir
        now = datetime(2025, 6, 1)
        created = (now - timedelta(days=60)).isoformat()
        _make_model_dir(reg_dir, "m1", created_at=created)

        svc = HealthService(
            registry=ModelRegistry(base_dir=reg_dir),
            assignments=ModelAssignments(state_dir=state_dir),
            trade_log=TradeLogStore(base_dir=str(log_dir)),
        )
        report = svc.report_for_model("m1", now=now)
        assert report.staleness == "expired"
        assert report.staleness_color == "gray"

    def test_model_not_found(self, tmp_dir):
        reg_dir, state_dir, log_dir = tmp_dir
        svc = HealthService(
            registry=ModelRegistry(base_dir=reg_dir),
            assignments=ModelAssignments(state_dir=state_dir),
            trade_log=TradeLogStore(base_dir=str(log_dir)),
        )
        with pytest.raises(FileNotFoundError):
            svc.report_for_model("nonexistent")

    def test_trade_accuracy(self, tmp_dir):
        """Closed trades should produce accuracy metrics."""
        reg_dir, state_dir, log_dir = tmp_dir
        _make_model_dir(reg_dir, "m1", asset="GLD")

        store = TradeLogStore(base_dir=str(log_dir))
        # 2 winning trades, 1 losing
        store.append(_make_trade("m1", expected=3.0, actual=2.5, status="closed"))
        store.append(_make_trade("m1", expected=2.0, actual=1.5, status="closed"))
        store.append(_make_trade("m1", expected=1.0, actual=-0.5, status="closed"))

        svc = HealthService(
            registry=ModelRegistry(base_dir=reg_dir),
            assignments=ModelAssignments(state_dir=state_dir),
            trade_log=store,
        )
        report = svc.report_for_model("m1")

        assert report.total_trades == 3
        assert report.closed_trades == 3
        assert report.open_trades == 0
        assert abs(report.win_rate - 66.67) < 1  # 2/3
        assert abs(report.avg_predicted_return - 2.0) < 0.01  # (3+2+1)/3
        assert abs(report.avg_actual_return - (2.5 + 1.5 - 0.5) / 3) < 0.01
        assert report.prediction_bias > 0  # model was over-optimistic

    def test_open_trades_counted(self, tmp_dir):
        reg_dir, state_dir, log_dir = tmp_dir
        _make_model_dir(reg_dir, "m1")
        store = TradeLogStore(base_dir=str(log_dir))
        store.append(_make_trade("m1", status="open"))
        store.append(_make_trade("m1", expected=1.0, actual=0.5, status="closed"))

        svc = HealthService(
            registry=ModelRegistry(base_dir=reg_dir),
            assignments=ModelAssignments(state_dir=state_dir),
            trade_log=store,
        )
        report = svc.report_for_model("m1")
        assert report.total_trades == 2
        assert report.open_trades == 1
        assert report.closed_trades == 1

    def test_training_verdict_extraction(self, tmp_dir):
        reg_dir, state_dir, log_dir = tmp_dir
        training_summary = {
            "diagnostics": {"verdict": "overfitting"},
            "val_losses": [0.5, 0.4, 0.35, 0.3, 0.32, 0.34],
        }
        _make_model_dir(reg_dir, "m1", training_summary=training_summary)

        svc = HealthService(
            registry=ModelRegistry(base_dir=reg_dir),
            assignments=ModelAssignments(state_dir=state_dir),
            trade_log=TradeLogStore(base_dir=str(log_dir)),
        )
        report = svc.report_for_model("m1")
        assert report.training_verdict == "overfitting"
        assert report.best_val_loss == 0.3
        assert report.total_epochs == 6

    def test_unknown_training_verdict(self, tmp_dir):
        reg_dir, state_dir, log_dir = tmp_dir
        _make_model_dir(reg_dir, "m1", training_summary={})

        svc = HealthService(
            registry=ModelRegistry(base_dir=reg_dir),
            assignments=ModelAssignments(state_dir=state_dir),
            trade_log=TradeLogStore(base_dir=str(log_dir)),
        )
        report = svc.report_for_model("m1")
        assert report.training_verdict == "unknown"

    def test_missing_created_at(self, tmp_dir):
        reg_dir, state_dir, log_dir = tmp_dir
        _make_model_dir(reg_dir, "m1", created_at="")

        svc = HealthService(
            registry=ModelRegistry(base_dir=reg_dir),
            assignments=ModelAssignments(state_dir=state_dir),
            trade_log=TradeLogStore(base_dir=str(log_dir)),
        )
        report = svc.report_for_model("m1")
        assert report.staleness == "expired"  # unknown age → expired


# ── Recommendations ─────────────────────────────────────────────────────

class TestRecommendations:
    def test_fresh_no_trades(self, tmp_dir):
        reg_dir, state_dir, log_dir = tmp_dir
        now = datetime(2025, 3, 1)
        created = (now - timedelta(days=2)).isoformat()
        _make_model_dir(reg_dir, "m1", created_at=created)

        svc = HealthService(
            registry=ModelRegistry(base_dir=reg_dir),
            assignments=ModelAssignments(state_dir=state_dir),
            trade_log=TradeLogStore(base_dir=str(log_dir)),
        )
        report = svc.report_for_model("m1", now=now)
        recs = report.recommendations
        assert any("no recalibration" in r.lower() for r in recs)
        assert any("no closed trades" in r.lower() for r in recs)

    def test_stale_low_winrate(self, tmp_dir):
        reg_dir, state_dir, log_dir = tmp_dir
        now = datetime(2025, 3, 1)
        created = (now - timedelta(days=25)).isoformat()
        _make_model_dir(reg_dir, "m1", created_at=created)

        store = TradeLogStore(base_dir=str(log_dir))
        # 1 win, 4 losses = 20% win rate
        store.append(_make_trade("m1", expected=2.0, actual=1.0, status="closed"))
        for i in range(4):
            store.append(_make_trade("m1", expected=2.0, actual=-1.0, status="closed"))

        svc = HealthService(
            registry=ModelRegistry(base_dir=reg_dir),
            assignments=ModelAssignments(state_dir=state_dir),
            trade_log=store,
        )
        report = svc.report_for_model("m1", now=now)
        recs_text = " ".join(report.recommendations).lower()
        assert "stale" in recs_text or "recalibration" in recs_text
        assert "win rate" in recs_text

    def test_overfitting_recommendation(self, tmp_dir):
        reg_dir, state_dir, log_dir = tmp_dir
        training_summary = {"diagnostics": {"verdict": "overfitting"}}
        _make_model_dir(reg_dir, "m1", training_summary=training_summary)

        svc = HealthService(
            registry=ModelRegistry(base_dir=reg_dir),
            assignments=ModelAssignments(state_dir=state_dir),
            trade_log=TradeLogStore(base_dir=str(log_dir)),
        )
        report = svc.report_for_model("m1")
        recs_text = " ".join(report.recommendations).lower()
        assert "overfitting" in recs_text

    def test_prediction_bias_warning(self, tmp_dir):
        reg_dir, state_dir, log_dir = tmp_dir
        _make_model_dir(reg_dir, "m1")
        store = TradeLogStore(base_dir=str(log_dir))
        # Predicted 5%, actual 1% → bias +4pp (over-optimistic)
        for _ in range(3):
            store.append(_make_trade("m1", expected=5.0, actual=1.0, status="closed"))

        svc = HealthService(
            registry=ModelRegistry(base_dir=reg_dir),
            assignments=ModelAssignments(state_dir=state_dir),
            trade_log=store,
        )
        report = svc.report_for_model("m1")
        recs_text = " ".join(report.recommendations).lower()
        assert "over-optimistic" in recs_text


# ── report_all_* ────────────────────────────────────────────────────────

class TestReportAll:
    def test_report_all_assigned(self, tmp_dir):
        reg_dir, state_dir, log_dir = tmp_dir
        _make_model_dir(reg_dir, "m_gld", asset="GLD")
        _make_model_dir(reg_dir, "m_slv", asset="SLV")
        _make_model_dir(reg_dir, "m_btc", asset="BTC-USD")

        assignments = ModelAssignments(state_dir=state_dir)
        assignments.assign("GLD", "m_gld")
        assignments.assign("SLV", "m_slv")

        svc = HealthService(
            registry=ModelRegistry(base_dir=reg_dir),
            assignments=assignments,
            trade_log=TradeLogStore(base_dir=str(log_dir)),
        )
        reports = svc.report_all_assigned()
        assert len(reports) == 2
        assets = {r.asset for r in reports}
        assert assets == {"GLD", "SLV"}

    def test_report_all_models(self, tmp_dir):
        reg_dir, state_dir, log_dir = tmp_dir
        _make_model_dir(reg_dir, "m1", asset="GLD")
        _make_model_dir(reg_dir, "m2", asset="SLV")

        svc = HealthService(
            registry=ModelRegistry(base_dir=reg_dir),
            assignments=ModelAssignments(state_dir=state_dir),
            trade_log=TradeLogStore(base_dir=str(log_dir)),
        )
        reports = svc.report_all_models()
        assert len(reports) == 2

    def test_report_all_models_filter_asset(self, tmp_dir):
        reg_dir, state_dir, log_dir = tmp_dir
        _make_model_dir(reg_dir, "m1", asset="GLD")
        _make_model_dir(reg_dir, "m2", asset="SLV")

        svc = HealthService(
            registry=ModelRegistry(base_dir=reg_dir),
            assignments=ModelAssignments(state_dir=state_dir),
            trade_log=TradeLogStore(base_dir=str(log_dir)),
        )
        reports = svc.report_all_models(asset="GLD")
        assert len(reports) == 1
        assert reports[0].asset == "GLD"

    def test_report_all_empty_registry(self, tmp_dir):
        reg_dir, state_dir, log_dir = tmp_dir
        svc = HealthService(
            registry=ModelRegistry(base_dir=reg_dir),
            assignments=ModelAssignments(state_dir=state_dir),
            trade_log=TradeLogStore(base_dir=str(log_dir)),
        )
        assert svc.report_all_assigned() == []
        assert svc.report_all_models() == []

    def test_assigned_missing_model_skipped(self, tmp_dir):
        """If an assigned model doesn't exist in registry, it's skipped."""
        reg_dir, state_dir, log_dir = tmp_dir
        assignments = ModelAssignments(state_dir=state_dir)
        assignments.assign("GLD", "nonexistent_model")

        svc = HealthService(
            registry=ModelRegistry(base_dir=reg_dir),
            assignments=assignments,
            trade_log=TradeLogStore(base_dir=str(log_dir)),
        )
        reports = svc.report_all_assigned()
        assert len(reports) == 0  # skipped, no error
