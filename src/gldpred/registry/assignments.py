"""Asset–model assignment: persist which registry model is the
"primary" model for each supported asset.

The mapping is stored as a JSON file in ``data/app_state/asset_model_map.json``
and is loaded / saved atomically.  The Streamlit app uses this to
automatically load the right model when the user switches assets or when
the portfolio comparison tab needs forecasts for several assets at once.

Usage::

    from gldpred.registry.assignments import ModelAssignments

    assignments = ModelAssignments()
    assignments.assign("GLD", "20250101_120000_abcd1234")
    model_id = assignments.get("GLD")
    assignments.unassign("GLD")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

DEFAULT_STATE_DIR = Path("data") / "app_state"
MAP_FILENAME = "asset_model_map.json"


class ModelAssignments:
    """Manage the primary model assignment for each asset."""

    def __init__(self, state_dir: str | Path | None = None) -> None:
        self._dir = Path(state_dir) if state_dir else DEFAULT_STATE_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / MAP_FILENAME
        self._map: Dict[str, str] = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self) -> Dict[str, str]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save(self) -> None:
        self._path.write_text(
            json.dumps(self._map, indent=2), encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def assign(self, ticker: str, model_id: str) -> None:
        """Set *model_id* as the primary model for *ticker*."""
        self._map[ticker] = model_id
        self._save()

    def unassign(self, ticker: str) -> None:
        """Remove the primary model assignment for *ticker*."""
        self._map.pop(ticker, None)
        self._save()

    def get(self, ticker: str) -> Optional[str]:
        """Return the assigned model ID, or ``None``."""
        return self._map.get(ticker)

    def get_all(self) -> Dict[str, str]:
        """Return a copy of the full assignment mapping."""
        return dict(self._map)

    def reset(self) -> None:
        """Clear all assignments."""
        self._map.clear()
        self._save()

    def has(self, ticker: str) -> bool:
        """Return whether *ticker* has a primary model assigned."""
        return ticker in self._map
