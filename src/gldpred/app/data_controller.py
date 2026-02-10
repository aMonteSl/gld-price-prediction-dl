"""Cached data loading and feature engineering controller.

Centralises the data-fetch + feature-engineering pipeline behind
``@st.cache_data`` so the Streamlit app never re-downloads unless the
asset ticker changes or the user explicitly refreshes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import streamlit as st

from gldpred.data import AssetDataLoader
from gldpred.features import FeatureEngineering


@dataclass(frozen=True)
class LoadedData:
    """Immutable container returned by ``fetch_asset_data``."""

    df: pd.DataFrame
    daily_returns: pd.Series
    feature_names: List[str]


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_asset_data(ticker: str) -> LoadedData:
    """Download OHLCV data + compute features (cached 1 h).

    Args:
        ticker: Asset ticker, e.g. ``"GLD"``.

    Returns:
        A ``LoadedData`` bundle with the enriched DataFrame,
        daily-return series, and feature-name list.
    """
    loader = AssetDataLoader(ticker=ticker)
    df = loader.load_data()
    daily_ret = loader.daily_returns()

    eng = FeatureEngineering()
    df = eng.add_technical_indicators(df)
    feat_df = eng.select_features(df)
    feature_names = feat_df.columns.tolist()

    return LoadedData(df=df, daily_returns=daily_ret, feature_names=feature_names)


def invalidate_cache() -> None:
    """Force the next call to ``fetch_asset_data`` to re-download."""
    fetch_asset_data.clear()
