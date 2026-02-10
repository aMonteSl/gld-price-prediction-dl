"""In-memory forecast cache keyed by (asset, model_id).

Avoids re-running inference when the user switches between tabs
or comes back to the forecast view without changing model or data.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import streamlit as st


@dataclass
class _CacheEntry:
    """Single cached forecast result."""
    forecast: Any
    timestamp: float
    data_hash: str


class ForecastCache:
    """Session-scoped forecast cache.

    Keyed by ``(asset, model_id)`` so each pair stores at most one
    forecast.  A data hash is kept to invalidate when new data arrives.

    Usage::

        cache = ForecastCache.get_instance()
        hit = cache.get("GLD", "model_abc", df)
        if hit is None:
            forecast = run_forecast(...)
            cache.put("GLD", "model_abc", df, forecast)

    The cache lives inside ``st.session_state`` so it persists across
    Streamlit reruns within the same browser session.
    """

    _SESSION_KEY = "_forecast_cache"
    _DEFAULT_TTL = 3600  # seconds

    def __init__(self, ttl: int = _DEFAULT_TTL) -> None:
        self._store: Dict[Tuple[str, str], _CacheEntry] = {}
        self._ttl = ttl

    # ------------------------------------------------------------------
    # Singleton accessor (session-state backed)
    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls, ttl: int = _DEFAULT_TTL) -> "ForecastCache":
        """Return the session-scoped singleton, creating it if needed."""
        if cls._SESSION_KEY not in st.session_state:
            st.session_state[cls._SESSION_KEY] = cls(ttl=ttl)
        return st.session_state[cls._SESSION_KEY]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(
        self, asset: str, model_id: str, df: Any = None,
    ) -> Optional[Any]:
        """Retrieve a cached forecast, or *None* on miss / stale.

        If *df* is provided its hash is compared to detect data changes.
        """
        key = (asset, model_id)
        entry = self._store.get(key)
        if entry is None:
            return None
        # TTL check
        if time.time() - entry.timestamp > self._ttl:
            del self._store[key]
            return None
        # Data-hash check
        if df is not None:
            current_hash = self._hash_df(df)
            if current_hash != entry.data_hash:
                del self._store[key]
                return None
        return entry.forecast

    def put(
        self, asset: str, model_id: str, df: Any, forecast: Any,
    ) -> None:
        """Store a forecast in the cache."""
        key = (asset, model_id)
        self._store[key] = _CacheEntry(
            forecast=forecast,
            timestamp=time.time(),
            data_hash=self._hash_df(df),
        )

    def invalidate(self, asset: Optional[str] = None) -> None:
        """Remove cached entries.

        If *asset* is given, only entries for that asset are removed.
        Otherwise the entire cache is cleared.
        """
        if asset is None:
            self._store.clear()
            return
        keys = [k for k in self._store if k[0] == asset]
        for k in keys:
            del self._store[k]

    def has(self, asset: str, model_id: str) -> bool:
        """Return *True* if a (possibly stale) entry exists."""
        return (asset, model_id) in self._store

    @property
    def size(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _hash_df(df: Any) -> str:
        """Fast hash of a DataFrame (shape + last few rows)."""
        if df is None:
            return "none"
        try:
            tail = str(df.tail(3).values.tobytes())
            shape = str(df.shape)
            return hashlib.md5((shape + tail).encode()).hexdigest()
        except Exception:
            return str(id(df))
