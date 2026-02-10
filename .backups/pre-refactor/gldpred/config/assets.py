"""Centralised asset catalog with metadata for each supported ticker.

Every supported asset is described by an :class:`AssetInfo` dataclass that
holds display name, type, volatility thresholds, trading schedule, and
currency. The module-level constant :data:`ASSET_CATALOG` maps tickers to
their metadata.

Usage::

    from gldpred.config.assets import ASSET_CATALOG, AssetInfo

    info = ASSET_CATALOG["GLD"]
    print(info.name, info.asset_type, info.currency)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class AssetInfo:
    """Immutable metadata for a single financial asset.

    Attributes:
        ticker: Yahoo Finance ticker symbol.
        name: Human-readable asset name.
        asset_type: Category (ETF, Crypto, …).
        currency: ISO 4217 code of the asset's denomination.
        max_volatility: ATR-% threshold for the decision engine.
        is_crypto: Whether the asset trades 24/7.
        description_en: Short English description.
        description_es: Short Spanish description.
    """

    ticker: str
    name: str
    asset_type: str
    currency: str = "USD"
    max_volatility: float = 0.02
    is_crypto: bool = False
    description_en: str = ""
    description_es: str = ""


ASSET_CATALOG: Dict[str, AssetInfo] = {
    "GLD": AssetInfo(
        ticker="GLD",
        name="SPDR Gold Shares",
        asset_type="ETF",
        currency="USD",
        max_volatility=0.02,
        is_crypto=False,
        description_en="Gold ETF tracking the spot price of gold bullion.",
        description_es="ETF de oro que sigue el precio spot del lingote de oro.",
    ),
    "SLV": AssetInfo(
        ticker="SLV",
        name="iShares Silver Trust",
        asset_type="ETF",
        currency="USD",
        max_volatility=0.025,
        is_crypto=False,
        description_en="Silver ETF tracking the spot price of silver.",
        description_es="ETF de plata que sigue el precio spot de la plata.",
    ),
    "BTC-USD": AssetInfo(
        ticker="BTC-USD",
        name="Bitcoin",
        asset_type="Crypto",
        currency="USD",
        max_volatility=0.05,
        is_crypto=True,
        description_en="Decentralised cryptocurrency — trades 24/7.",
        description_es="Criptomoneda descentralizada — cotiza 24/7.",
    ),
    "PALL": AssetInfo(
        ticker="PALL",
        name="Aberdeen Physical Palladium",
        asset_type="ETF",
        currency="USD",
        max_volatility=0.03,
        is_crypto=False,
        description_en="Palladium ETF backed by physical palladium.",
        description_es="ETF de paladio respaldado por paladio físico.",
    ),
}


def get_asset_info(ticker: str) -> AssetInfo:
    """Look up asset metadata by ticker.

    Raises:
        KeyError: If ticker is not in the catalog.
    """
    return ASSET_CATALOG[ticker]


def supported_tickers() -> list[str]:
    """Return the list of all supported ticker symbols."""
    return list(ASSET_CATALOG.keys())
