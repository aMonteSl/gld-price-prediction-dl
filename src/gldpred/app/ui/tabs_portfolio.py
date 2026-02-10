"""Portfolio / Trade Log tab — track investment decisions over time."""
from __future__ import annotations

import traceback
import uuid
from datetime import datetime
from typing import Dict

import pandas as pd
import streamlit as st

from gldpred.app import state
from gldpred.storage.trade_log import TradeLogEntry, TradeLogStore


def render(t: Dict[str, str], lang: str) -> None:
    """Render the Portfolio tracking tab."""
    st.header(t.get("portfolio_header", "Portfolio & Trade Log"))
    st.caption(t.get("portfolio_subtitle",
        "Track your investment decisions and compare predictions vs outcomes."
    ))

    store = TradeLogStore()
    entries = store.load_all()

    # ── Stats overview ───────────────────────────────────────────
    stats = store.summary_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        t.get("portfolio_total", "Total Trades"),
        stats["total"],
    )
    c2.metric(
        t.get("portfolio_open", "Open"),
        stats["open"],
    )
    c3.metric(
        t.get("portfolio_closed", "Closed"),
        stats["closed"],
    )
    c4.metric(
        t.get("portfolio_win_rate", "Win Rate"),
        f"{stats['win_rate']:.1f}%",
    )

    st.divider()

    # ── Archive current action plan ──────────────────────────────
    st.subheader(t.get("portfolio_archive", "Archive Current Plan"))
    plan = st.session_state.get("_action_plan")
    forecast = state.get(state.KEY_FORECAST)
    asset = state.get(state.KEY_ASSET, "GLD")

    if plan is not None and forecast is not None:
        notes = st.text_input(
            t.get("portfolio_notes", "Notes (optional)"),
            key="trade_notes",
        )
        if st.button(
            t.get("portfolio_save_trade", "💾 Save to Trade Log"),
            type="primary",
        ):
            entry = TradeLogEntry(
                id=str(uuid.uuid4())[:8],
                asset=asset,
                signal=plan.overall_signal,
                confidence=plan.overall_confidence,
                entry_price=plan.entry_price,
                expected_return_pct=getattr(
                    plan.scenarios.base, "return_pct", 0.0,
                ),
                stop_loss_pct=plan.params.get("stop_loss_pct", 3.0),
                take_profit_pct=plan.params.get("take_profit_pct", 5.0),
                investment=plan.params.get("investment", 10_000.0),
                horizon=plan.params.get("horizon", 20),
                model_id=plan.params.get("model_id", ""),
                model_label=plan.params.get("model_id", ""),
                notes=notes,
            )
            store.append(entry)
            st.success(
                t.get("portfolio_saved", "✅ Trade saved to log!")
            )
            st.rerun()
    else:
        st.info(t.get("portfolio_no_plan",
            "Generate an action plan in the Recommendation tab first."
        ))

    # ── Trade log table ──────────────────────────────────────────
    st.subheader(t.get("portfolio_log_header", "Trade Log"))

    if not entries:
        st.info(t.get("portfolio_empty",
            "No trades recorded yet. Generate a recommendation and "
            "archive it to start tracking."
        ))
        return

    rows = []
    for e in reversed(entries):  # newest first
        rows.append({
            "ID": e.id,
            t.get("portfolio_col_date", "Date"): e.timestamp[:10],
            t.get("portfolio_col_asset", "Asset"): e.asset,
            t.get("portfolio_col_signal", "Signal"): e.signal,
            t.get("portfolio_col_conf", "Conf."): f"{e.confidence:.0f}%",
            t.get("portfolio_col_expected", "Expected"): f"{e.expected_return_pct:+.2f}%",
            t.get("portfolio_col_actual", "Actual"): (
                f"{e.actual_return_pct:+.2f}%" if e.actual_return_pct is not None
                else "—"
            ),
            t.get("portfolio_col_status", "Status"): e.status.upper(),
            t.get("portfolio_col_investment", "Investment"): f"${e.investment:,.0f}",
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    # ── Close trade form ─────────────────────────────────────────
    open_trades = store.load_open()
    if open_trades:
        st.subheader(t.get("portfolio_close_header", "Close a Trade"))
        trade_ids = [f"{e.id} ({e.asset} {e.signal})" for e in open_trades]
        selected = st.selectbox(
            t.get("portfolio_select_trade", "Select trade to close"),
            trade_ids,
            key="close_trade_select",
        )
        idx = trade_ids.index(selected)
        trade = open_trades[idx]

        cl1, cl2 = st.columns(2)
        actual_ret = cl1.number_input(
            t.get("portfolio_actual_return", "Actual Return (%)"),
            value=0.0,
            step=0.1,
            key="close_ret",
        )
        actual_price = cl2.number_input(
            t.get("portfolio_actual_price", "Exit Price ($)"),
            value=float(trade.entry_price),
            step=0.01,
            key="close_price",
        )

        if st.button(
            t.get("portfolio_close_btn", "Close Trade"),
            key="btn_close_trade",
        ):
            store.close_trade(trade.id, actual_ret, actual_price)
            st.success(
                t.get("portfolio_closed_msg", "Trade closed!")
            )
            st.rerun()
