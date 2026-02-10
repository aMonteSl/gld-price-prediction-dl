"""Centralised asset catalog with metadata for each supported ticker.

Every supported asset is described by an :class:`AssetInfo` dataclass that
holds display name, type, risk classification, volatility thresholds,
investment horizon, trading schedule, and currency.  The module-level
constant :data:`ASSET_CATALOG` maps tickers to their metadata.

Assets are classified into **three risk tiers** (``low``, ``medium``,
``high``) and tagged with an investment role (``benchmark``,
``diversifier``, ``tactical``, ``speculative``).  The :data:`BENCHMARK_ASSET`
constant identifies the S&P 500 proxy (SPY) used as comparison baseline
throughout the application.

Usage::

    from gldpred.config.assets import ASSET_CATALOG, AssetInfo

    info = ASSET_CATALOG["GLD"]
    print(info.name, info.risk_level, info.role)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


# ── Risk tiers, horizons, volatility profiles, roles ─────────────────

RISK_LEVELS: Tuple[str, ...] = ("low", "medium", "high")
INVESTMENT_HORIZONS: Tuple[str, ...] = ("short", "medium", "long")
VOLATILITY_PROFILES: Tuple[str, ...] = ("stable", "moderate", "volatile")
ASSET_ROLES: Tuple[str, ...] = ("benchmark", "diversifier", "tactical", "speculative")
ASSET_CATEGORIES: Tuple[str, ...] = (
    "equity_etf", "bond_etf", "commodity_etf", "crypto", "cash_proxy",
)

BENCHMARK_ASSET: str = "SPY"


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
        risk_level: ``"low"``, ``"medium"``, or ``"high"``.
        investment_horizon: Suggested horizons (``"short"``, ``"medium"``, ``"long"``).
        volatility_profile: ``"stable"``, ``"moderate"``, or ``"volatile"``.
        role: ``"benchmark"``, ``"diversifier"``, ``"tactical"``, or ``"speculative"``.
        category: ``"equity_etf"``, ``"bond_etf"``, ``"commodity_etf"``, ``"crypto"``, or ``"cash_proxy"``.
    """

    ticker: str
    name: str
    asset_type: str
    currency: str = "USD"
    max_volatility: float = 0.02
    is_crypto: bool = False
    description_en: str = ""
    description_es: str = ""
    risk_level: str = "medium"
    investment_horizon: Tuple[str, ...] = ("medium",)
    volatility_profile: str = "moderate"
    role: str = "diversifier"
    category: str = "equity_etf"


# ── Full asset catalog ───────────────────────────────────────────────

ASSET_CATALOG: Dict[str, AssetInfo] = {
    # ── LOW RISK ─────────────────────────────────────────────────────
    "SPY": AssetInfo(
        ticker="SPY",
        name="SPDR S&P 500 ETF",
        asset_type="ETF",
        max_volatility=0.018,
        description_en="S&P 500 ETF — broad US equity benchmark.",
        description_es="ETF del S&P 500 — referencia de renta variable US.",
        risk_level="low",
        investment_horizon=("medium", "long"),
        volatility_profile="moderate",
        role="benchmark",
        category="equity_etf",
    ),
    "VT": AssetInfo(
        ticker="VT",
        name="Vanguard Total World Stock",
        asset_type="ETF",
        max_volatility=0.018,
        description_en="Global equity ETF covering developed + emerging markets.",
        description_es="ETF de renta variable global: mercados desarrollados y emergentes.",
        risk_level="low",
        investment_horizon=("long",),
        volatility_profile="moderate",
        role="diversifier",
        category="equity_etf",
    ),
    "TLT": AssetInfo(
        ticker="TLT",
        name="iShares 20+ Year Treasury Bond",
        asset_type="ETF",
        max_volatility=0.015,
        description_en="Long-term US Treasury bond ETF — interest-rate sensitive.",
        description_es="ETF de bonos del Tesoro US a largo plazo — sensible a tipos de interés.",
        risk_level="low",
        investment_horizon=("medium", "long"),
        volatility_profile="moderate",
        role="diversifier",
        category="bond_etf",
    ),
    "IEF": AssetInfo(
        ticker="IEF",
        name="iShares 7-10 Year Treasury Bond",
        asset_type="ETF",
        max_volatility=0.012,
        description_en="Intermediate-term US Treasury bonds — lower duration risk.",
        description_es="Bonos del Tesoro US a plazo intermedio — menor riesgo de duración.",
        risk_level="low",
        investment_horizon=("medium",),
        volatility_profile="stable",
        role="diversifier",
        category="bond_etf",
    ),
    "SHV": AssetInfo(
        ticker="SHV",
        name="iShares Short Treasury Bond",
        asset_type="ETF",
        max_volatility=0.005,
        description_en="Short-term Treasury bonds — near-cash safety.",
        description_es="Bonos del Tesoro a corto plazo — seguridad cuasi-liquidez.",
        risk_level="low",
        investment_horizon=("short",),
        volatility_profile="stable",
        role="diversifier",
        category="cash_proxy",
    ),
    # ── MEDIUM RISK ──────────────────────────────────────────────────
    "QQQ": AssetInfo(
        ticker="QQQ",
        name="Invesco QQQ Trust (Nasdaq-100)",
        asset_type="ETF",
        max_volatility=0.022,
        description_en="Nasdaq-100 ETF — tech-heavy large-cap growth.",
        description_es="ETF del Nasdaq-100 — crecimiento de gran capitalización tecnológica.",
        risk_level="medium",
        investment_horizon=("medium", "long"),
        volatility_profile="moderate",
        role="tactical",
        category="equity_etf",
    ),
    "GLD": AssetInfo(
        ticker="GLD",
        name="SPDR Gold Shares",
        asset_type="ETF",
        max_volatility=0.02,
        description_en="Gold ETF tracking the spot price of gold bullion.",
        description_es="ETF de oro que sigue el precio spot del lingote de oro.",
        risk_level="medium",
        investment_horizon=("medium", "long"),
        volatility_profile="moderate",
        role="diversifier",
        category="commodity_etf",
    ),
    "SLV": AssetInfo(
        ticker="SLV",
        name="iShares Silver Trust",
        asset_type="ETF",
        max_volatility=0.025,
        description_en="Silver ETF tracking the spot price of silver.",
        description_es="ETF de plata que sigue el precio spot de la plata.",
        risk_level="medium",
        investment_horizon=("medium",),
        volatility_profile="moderate",
        role="diversifier",
        category="commodity_etf",
    ),
    "VNQ": AssetInfo(
        ticker="VNQ",
        name="Vanguard Real Estate ETF",
        asset_type="ETF",
        max_volatility=0.022,
        description_en="US real estate investment trust ETF.",
        description_es="ETF de fondos de inversión inmobiliaria (REIT) de EE.UU.",
        risk_level="medium",
        investment_horizon=("medium", "long"),
        volatility_profile="moderate",
        role="diversifier",
        category="equity_etf",
    ),
    "TIP": AssetInfo(
        ticker="TIP",
        name="iShares TIPS Bond ETF",
        asset_type="ETF",
        max_volatility=0.012,
        description_en="Inflation-protected US Treasury bonds.",
        description_es="Bonos del Tesoro US protegidos contra la inflación.",
        risk_level="medium",
        investment_horizon=("medium", "long"),
        volatility_profile="stable",
        role="diversifier",
        category="bond_etf",
    ),
    "PALL": AssetInfo(
        ticker="PALL",
        name="Aberdeen Physical Palladium",
        asset_type="ETF",
        max_volatility=0.03,
        description_en="Palladium ETF backed by physical palladium.",
        description_es="ETF de paladio respaldado por paladio físico.",
        risk_level="medium",
        investment_horizon=("medium",),
        volatility_profile="moderate",
        role="tactical",
        category="commodity_etf",
    ),
    # ── HIGH RISK ────────────────────────────────────────────────────
    "BTC-USD": AssetInfo(
        ticker="BTC-USD",
        name="Bitcoin",
        asset_type="Crypto",
        max_volatility=0.05,
        is_crypto=True,
        description_en="Decentralised cryptocurrency — trades 24/7.",
        description_es="Criptomoneda descentralizada — cotiza 24/7.",
        risk_level="high",
        investment_horizon=("long",),
        volatility_profile="volatile",
        role="speculative",
        category="crypto",
    ),
    "ETH-USD": AssetInfo(
        ticker="ETH-USD",
        name="Ethereum",
        asset_type="Crypto",
        max_volatility=0.06,
        is_crypto=True,
        description_en="Smart-contract blockchain — trades 24/7.",
        description_es="Blockchain de contratos inteligentes — cotiza 24/7.",
        risk_level="high",
        investment_horizon=("long",),
        volatility_profile="volatile",
        role="speculative",
        category="crypto",
    ),
    "USO": AssetInfo(
        ticker="USO",
        name="United States Oil Fund",
        asset_type="ETF",
        max_volatility=0.035,
        description_en="Crude oil futures ETF — high commodity exposure.",
        description_es="ETF de futuros de petróleo crudo — alta exposición a materias primas.",
        risk_level="high",
        investment_horizon=("short", "medium"),
        volatility_profile="volatile",
        role="tactical",
        category="commodity_etf",
    ),
    "COPX": AssetInfo(
        ticker="COPX",
        name="Global X Copper Miners ETF",
        asset_type="ETF",
        max_volatility=0.035,
        description_en="Copper mining equities — leveraged industrial metals exposure.",
        description_es="Acciones mineras de cobre — exposición apalancada a metales industriales.",
        risk_level="high",
        investment_horizon=("medium",),
        volatility_profile="volatile",
        role="tactical",
        category="commodity_etf",
    ),
    "ARKK": AssetInfo(
        ticker="ARKK",
        name="ARK Innovation ETF",
        asset_type="ETF",
        max_volatility=0.04,
        description_en="Actively managed disruptive innovation ETF.",
        description_es="ETF de innovación disruptiva gestionado activamente.",
        risk_level="high",
        investment_horizon=("long",),
        volatility_profile="volatile",
        role="speculative",
        category="equity_etf",
    ),
    "EEM": AssetInfo(
        ticker="EEM",
        name="iShares MSCI Emerging Markets",
        asset_type="ETF",
        max_volatility=0.025,
        description_en="Emerging markets equity ETF — geopolitical & currency risk.",
        description_es="ETF de renta variable de mercados emergentes — riesgo geopolítico y cambiario.",
        risk_level="high",
        investment_horizon=("medium", "long"),
        volatility_profile="moderate",
        role="tactical",
        category="equity_etf",
    ),
}


# ── Lookup helpers ───────────────────────────────────────────────────

def get_asset_info(ticker: str) -> AssetInfo:
    """Look up asset metadata by ticker.

    Raises:
        KeyError: If ticker is not in the catalog.
    """
    return ASSET_CATALOG[ticker]


def supported_tickers() -> list[str]:
    """Return the list of all supported ticker symbols."""
    return list(ASSET_CATALOG.keys())


def assets_by_risk(level: str) -> list[AssetInfo]:
    """Return assets matching a given risk level.

    Args:
        level: One of ``"low"``, ``"medium"``, ``"high"``.
    """
    return [a for a in ASSET_CATALOG.values() if a.risk_level == level]


def assets_by_category(category: str) -> list[AssetInfo]:
    """Return assets matching a given category.

    Args:
        category: One of :data:`ASSET_CATEGORIES`.
    """
    return [a for a in ASSET_CATALOG.values() if a.category == category]


def assets_by_role(role: str) -> list[AssetInfo]:
    """Return assets matching a given role.

    Args:
        role: One of :data:`ASSET_ROLES`.
    """
    return [a for a in ASSET_CATALOG.values() if a.role == role]


def get_benchmark() -> AssetInfo:
    """Return the benchmark asset (SPY)."""
    return ASSET_CATALOG[BENCHMARK_ASSET]
