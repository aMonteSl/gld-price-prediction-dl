"""Tests for ForecastCache."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ── Patch st.session_state before import ───────────────────────────────
_FAKE_SESSION: dict = {}

@pytest.fixture(autouse=True)
def _mock_session_state(monkeypatch):
    """Provide a fresh dict-like st.session_state for each test."""
    _FAKE_SESSION.clear()
    mock_st = MagicMock()
    mock_st.session_state = _FAKE_SESSION
    monkeypatch.setattr("gldpred.app.components.forecast_cache.st", mock_st)


def _make_df(rows: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2024-01-01", periods=rows)
    return pd.DataFrame(
        {"Close": 100 + rng.standard_normal(rows).cumsum()},
        index=idx,
    )


class TestForecastCache:

    def test_put_and_get(self):
        from gldpred.app.components.forecast_cache import ForecastCache

        cache = ForecastCache(ttl=300)
        df = _make_df()
        cache.put("GLD", "m1", df, "forecast_obj")
        assert cache.get("GLD", "m1", df) == "forecast_obj"

    def test_miss_returns_none(self):
        from gldpred.app.components.forecast_cache import ForecastCache

        cache = ForecastCache()
        assert cache.get("GLD", "m1") is None

    def test_data_change_invalidates(self):
        from gldpred.app.components.forecast_cache import ForecastCache

        cache = ForecastCache()
        df1 = _make_df(50)
        df2 = _make_df(60)
        cache.put("GLD", "m1", df1, "old")
        # Different data → miss
        assert cache.get("GLD", "m1", df2) is None

    def test_ttl_expiry(self):
        from gldpred.app.components.forecast_cache import ForecastCache

        cache = ForecastCache(ttl=0)  # expire immediately
        df = _make_df()
        cache.put("GLD", "m1", df, "val")
        time.sleep(0.01)
        assert cache.get("GLD", "m1", df) is None

    def test_invalidate_single_asset(self):
        from gldpred.app.components.forecast_cache import ForecastCache

        cache = ForecastCache()
        df = _make_df()
        cache.put("GLD", "m1", df, "g1")
        cache.put("SLV", "m2", df, "s1")
        cache.invalidate("GLD")
        assert cache.get("GLD", "m1", df) is None
        assert cache.get("SLV", "m2", df) == "s1"

    def test_invalidate_all(self):
        from gldpred.app.components.forecast_cache import ForecastCache

        cache = ForecastCache()
        df = _make_df()
        cache.put("GLD", "m1", df, "g1")
        cache.put("SLV", "m2", df, "s1")
        cache.invalidate()
        assert cache.size == 0

    def test_has(self):
        from gldpred.app.components.forecast_cache import ForecastCache

        cache = ForecastCache()
        df = _make_df()
        assert not cache.has("GLD", "m1")
        cache.put("GLD", "m1", df, "val")
        assert cache.has("GLD", "m1")

    def test_different_models_different_entries(self):
        from gldpred.app.components.forecast_cache import ForecastCache

        cache = ForecastCache()
        df = _make_df()
        cache.put("GLD", "m1", df, "forecast_a")
        cache.put("GLD", "m2", df, "forecast_b")
        assert cache.get("GLD", "m1", df) == "forecast_a"
        assert cache.get("GLD", "m2", df) == "forecast_b"
        assert cache.size == 2

    def test_overwrite_same_key(self):
        from gldpred.app.components.forecast_cache import ForecastCache

        cache = ForecastCache()
        df = _make_df()
        cache.put("GLD", "m1", df, "old_val")
        cache.put("GLD", "m1", df, "new_val")
        assert cache.get("GLD", "m1", df) == "new_val"
        assert cache.size == 1

    def test_get_without_df_skips_hash_check(self):
        from gldpred.app.components.forecast_cache import ForecastCache

        cache = ForecastCache()
        df = _make_df()
        cache.put("GLD", "m1", df, "val")
        # Get without df → should still return (no hash check)
        assert cache.get("GLD", "m1") == "val"

    def test_singleton_via_session_state(self):
        from gldpred.app.components.forecast_cache import ForecastCache

        c1 = ForecastCache.get_instance()
        c2 = ForecastCache.get_instance()
        assert c1 is c2

    def test_hash_df_none(self):
        from gldpred.app.components.forecast_cache import ForecastCache

        assert ForecastCache._hash_df(None) == "none"

    def test_hash_df_consistent(self):
        from gldpred.app.components.forecast_cache import ForecastCache

        df = _make_df()
        h1 = ForecastCache._hash_df(df)
        h2 = ForecastCache._hash_df(df)
        assert h1 == h2
