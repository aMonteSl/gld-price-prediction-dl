"""Tests for the asset catalog."""
from __future__ import annotations

import pytest

from gldpred.config.assets import (
    ASSET_CATALOG,
    ASSET_CATEGORIES,
    ASSET_ROLES,
    AssetInfo,
    BENCHMARK_ASSET,
    INVESTMENT_HORIZONS,
    RISK_LEVELS,
    VOLATILITY_PROFILES,
    assets_by_category,
    assets_by_risk,
    assets_by_role,
    get_asset_info,
    get_benchmark,
    supported_tickers,
)
from gldpred.config import SUPPORTED_ASSETS


class TestAssetCatalog:
    def test_all_supported_assets_in_catalog(self):
        for ticker in SUPPORTED_ASSETS:
            assert ticker in ASSET_CATALOG

    def test_catalog_entries_are_assetinfo(self):
        for info in ASSET_CATALOG.values():
            assert isinstance(info, AssetInfo)

    def test_get_asset_info(self):
        info = get_asset_info("GLD")
        assert info.name == "SPDR Gold Shares"
        assert info.asset_type == "ETF"
        assert info.currency == "USD"

    def test_get_asset_info_crypto(self):
        info = get_asset_info("BTC-USD")
        assert info.is_crypto is True
        assert info.max_volatility > 0.02  # BTC has higher threshold

    def test_get_asset_info_missing(self):
        with pytest.raises(KeyError):
            get_asset_info("DOESNT-EXIST")

    def test_supported_tickers(self):
        tickers = supported_tickers()
        assert "GLD" in tickers
        assert "BTC-USD" in tickers
        assert len(tickers) == len(ASSET_CATALOG)

    def test_asset_info_frozen(self):
        info = get_asset_info("GLD")
        with pytest.raises(AttributeError):
            info.name = "Modified"  # type: ignore

    def test_max_volatility_varies_by_asset(self):
        gld = get_asset_info("GLD")
        btc = get_asset_info("BTC-USD")
        assert btc.max_volatility > gld.max_volatility


class TestAssetClassification:
    """Tests for the new risk-tier classification fields."""

    def test_all_assets_have_valid_risk_level(self):
        for ticker, info in ASSET_CATALOG.items():
            assert info.risk_level in RISK_LEVELS, (
                f"{ticker} has invalid risk_level: {info.risk_level}"
            )

    def test_all_assets_have_valid_horizons(self):
        for ticker, info in ASSET_CATALOG.items():
            assert len(info.investment_horizon) > 0
            for h in info.investment_horizon:
                assert h in INVESTMENT_HORIZONS, (
                    f"{ticker} has invalid horizon: {h}"
                )

    def test_all_assets_have_valid_volatility_profile(self):
        for ticker, info in ASSET_CATALOG.items():
            assert info.volatility_profile in VOLATILITY_PROFILES, (
                f"{ticker} has invalid volatility_profile: {info.volatility_profile}"
            )

    def test_all_assets_have_valid_role(self):
        for ticker, info in ASSET_CATALOG.items():
            assert info.role in ASSET_ROLES, (
                f"{ticker} has invalid role: {info.role}"
            )

    def test_all_assets_have_valid_category(self):
        for ticker, info in ASSET_CATALOG.items():
            assert info.category in ASSET_CATEGORIES, (
                f"{ticker} has invalid category: {info.category}"
            )

    def test_benchmark_exists_and_is_spy(self):
        assert BENCHMARK_ASSET == "SPY"
        assert BENCHMARK_ASSET in ASSET_CATALOG

    def test_get_benchmark(self):
        bench = get_benchmark()
        assert bench.ticker == "SPY"
        assert bench.role == "benchmark"
        assert bench.risk_level == "low"

    def test_assets_by_risk(self):
        low = assets_by_risk("low")
        assert len(low) >= 3
        assert all(a.risk_level == "low" for a in low)

        high = assets_by_risk("high")
        assert len(high) >= 3
        assert all(a.risk_level == "high" for a in high)

    def test_assets_by_category(self):
        crypto = assets_by_category("crypto")
        assert len(crypto) >= 2
        assert all(a.is_crypto for a in crypto)

    def test_assets_by_role(self):
        benchmarks = assets_by_role("benchmark")
        assert len(benchmarks) >= 1
        assert benchmarks[0].ticker == "SPY"

    def test_asset_count_at_least_17(self):
        """The expanded catalog has 17+ assets."""
        assert len(ASSET_CATALOG) >= 17

    def test_crypto_are_high_risk(self):
        for info in ASSET_CATALOG.values():
            if info.is_crypto:
                assert info.risk_level == "high"
                assert info.volatility_profile == "volatile"

    def test_bond_etfs_are_low_or_medium_risk(self):
        bonds = assets_by_category("bond_etf")
        for b in bonds:
            assert b.risk_level in ("low", "medium")

    def test_all_have_bilingual_descriptions(self):
        for ticker, info in ASSET_CATALOG.items():
            assert info.description_en, f"{ticker} missing EN description"
            assert info.description_es, f"{ticker} missing ES description"
