"""Tests for the asset catalog."""
from __future__ import annotations

import pytest

from gldpred.config.assets import (
    ASSET_CATALOG,
    AssetInfo,
    get_asset_info,
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
