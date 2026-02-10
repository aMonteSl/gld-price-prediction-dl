"""JSONL-based trade log storage.

Persists investment decisions (action plans) to disk so users can track
their decisions over time and compare predicted vs actual outcomes.

Storage format: one JSON object per line in ``data/trade_log/trades.jsonl``.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TradeLogEntry:
    """One recorded trading decision."""

    id: str
    asset: str
    signal: str                       # BUY / HOLD / AVOID / SELL
    confidence: float
    entry_price: float
    expected_return_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    investment: float
    horizon: int
    model_id: str
    model_label: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""
    # Outcome tracking (filled in later)
    actual_return_pct: Optional[float] = None
    actual_exit_price: Optional[float] = None
    outcome_date: Optional[str] = None
    status: str = "open"  # open / closed / expired


class TradeLogStore:
    """Append-only JSONL trade log.

    Args:
        base_dir: Directory for the log file.  Defaults to
            ``data/trade_log``.
    """

    def __init__(self, base_dir: str = "data/trade_log") -> None:
        self._dir = Path(base_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "trades.jsonl"

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def append(self, entry: TradeLogEntry) -> None:
        """Append a trade entry to the log."""
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(entry), default=str) + "\n")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def load_all(self) -> List[TradeLogEntry]:
        """Load all entries from the log file."""
        if not self._path.exists():
            return []
        entries: List[TradeLogEntry] = []
        with open(self._path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                entries.append(TradeLogEntry(**data))
        return entries

    def load_by_asset(self, asset: str) -> List[TradeLogEntry]:
        """Load entries filtered by asset ticker."""
        return [e for e in self.load_all() if e.asset == asset]

    def load_open(self) -> List[TradeLogEntry]:
        """Load only open (not yet resolved) trades."""
        return [e for e in self.load_all() if e.status == "open"]

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def close_trade(
        self,
        trade_id: str,
        actual_return_pct: float,
        actual_exit_price: float,
    ) -> bool:
        """Mark a trade as closed with actual outcome.

        Rewrites the entire file (trades are few, so this is fine).
        Returns True if the trade was found and updated.
        """
        entries = self.load_all()
        found = False
        for entry in entries:
            if entry.id == trade_id:
                entry.actual_return_pct = actual_return_pct
                entry.actual_exit_price = actual_exit_price
                entry.outcome_date = datetime.now().isoformat()
                entry.status = "closed"
                found = True
                break
        if found:
            self._rewrite(entries)
        return found

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def summary_stats(self) -> Dict[str, Any]:
        """Compute basic stats from closed trades."""
        entries = self.load_all()
        closed = [e for e in entries if e.status == "closed"]
        if not closed:
            return {
                "total": len(entries),
                "open": len(entries),
                "closed": 0,
                "win_rate": 0.0,
                "avg_return": 0.0,
            }
        wins = [e for e in closed if (e.actual_return_pct or 0) > 0]
        returns = [e.actual_return_pct or 0 for e in closed]
        return {
            "total": len(entries),
            "open": len(entries) - len(closed),
            "closed": len(closed),
            "win_rate": len(wins) / len(closed) * 100 if closed else 0.0,
            "avg_return": sum(returns) / len(returns) if returns else 0.0,
        }

    @property
    def count(self) -> int:
        return len(self.load_all())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _rewrite(self, entries: List[TradeLogEntry]) -> None:
        """Overwrite the JSONL file with updated entries."""
        with open(self._path, "w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(asdict(entry), default=str) + "\n")

    def clear(self) -> None:
        """Remove all entries (for testing)."""
        if self._path.exists():
            self._path.unlink()
