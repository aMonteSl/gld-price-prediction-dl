"""Data Hub tab — transparency & control over all persisted application data.

Subsections:
1. 📊 Market Data — loaded assets, date ranges, cache management
2. 🧠 Models — registry listing, metadata export, actions
3. 🔮 Forecasts — cached forecasts, reuse/delete/export
4. 💼 Trade Log — recorded investments with P&L, CSV export
5. 📈 Model Performance — historical prediction accuracy
6. ⚙️  Global actions — export all, reset
"""
from __future__ import annotations

import csv
import io
import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from gldpred.app import state
from gldpred.app.components.forecast_cache import ForecastCache
from gldpred.app.data_controller import fetch_asset_data, invalidate_cache
from gldpred.config import SUPPORTED_ASSETS
from gldpred.registry import ModelAssignments, ModelRegistry
from gldpred.storage import TradeLogStore


def render(t: Dict[str, str], lang: str) -> None:
    """Render the Data Hub tab."""
    st.header(t["hub_header"])
    st.caption(t["hub_subtitle"])

    # Sub-sections as expanders for clean layout
    _render_market_data(t, lang)
    _render_models_section(t, lang)
    _render_forecasts_section(t, lang)
    _render_trade_log_section(t, lang)
    _render_performance_section(t, lang)
    st.divider()
    _render_global_actions(t, lang)


# =====================================================================
# 1. Market Data
# =====================================================================

def _render_market_data(t: Dict[str, str], lang: str) -> None:
    with st.expander(t["hub_market_header"], expanded=True):
        # Show loaded data for each asset
        df = state.get(state.KEY_RAW_DF)
        loaded_asset = state.get(state.KEY_DATA_LOADED_ASSET)

        if df is not None and loaded_asset:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(t["hub_market_asset"], loaded_asset)
            c2.metric(t["hub_market_records"], f"{len(df):,}")
            start = pd.Timestamp(df.index[0]).strftime("%Y-%m-%d")
            end = pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d")
            c3.metric(t["hub_market_range"], f"{start} → {end}")
            # Estimate cache size
            size_kb = df.memory_usage(deep=True).sum() / 1024
            c4.metric(t["hub_market_cache_size"], f"{size_kb:.0f} KB")

            # Show last 5 rows
            st.dataframe(df.tail(5), use_container_width=True)

            bc1, bc2 = st.columns(2)
            if bc1.button(t["hub_market_refresh"], key="hub_refresh_data"):
                invalidate_cache()
                state.clear_data_state()
                st.rerun()
            
            # Export market data as CSV
            csv_data = df.to_csv()
            bc2.download_button(
                t["hub_market_export_csv"],
                csv_data,
                file_name=f"{loaded_asset}_data.csv",
                mime="text/csv",
                key="hub_export_market",
            )
        else:
            st.info(t["hub_market_no_data"])


# =====================================================================
# 2. Models
# =====================================================================

def _render_models_section(t: Dict[str, str], lang: str) -> None:
    with st.expander(t["hub_models_header"], expanded=True):
        registry = ModelRegistry()
        assignments = ModelAssignments()
        all_models = registry.list_models()

        if not all_models:
            st.info(t["hub_models_empty"])
            return

        st.metric(t["hub_models_total"], len(all_models))

        for meta in all_models:
            mid = meta["model_id"]
            label = meta.get("label", mid)
            asset = meta.get("asset", "?")
            arch = meta.get("architecture", "?")
            created = meta.get("created_at", "?")
            primary_id = assignments.get(asset)
            is_primary = mid == primary_id

            badge = "⭐ " if is_primary else ""
            header = f"{badge}{label} ({asset} / {arch}) — {created[:10] if len(created) > 10 else created}"

            with st.container():
                st.markdown(f"**{header}**")
                mc1, mc2, mc3, mc4 = st.columns(4)

                # Training info
                ts = meta.get("training_summary", {})
                mc1.caption(f"{t['hub_models_epochs']}: {ts.get('epochs_trained', '?')}")
                mc2.caption(f"{t['hub_models_verdict']}: {ts.get('diagnostics_verdict', '?')}")

                # Evaluation info
                ev = meta.get("evaluation_summary", {})
                if ev:
                    mc3.caption(f"Dir. Acc: {ev.get('directional_accuracy', '?')}")

                mc4.caption(f"ID: {mid[:16]}...")

                # Actions
                ac1, ac2, ac3 = st.columns(3)
                
                # Export metadata JSON
                meta_json = json.dumps(meta, indent=2, default=str)
                ac1.download_button(
                    t["hub_models_export_meta"],
                    meta_json,
                    file_name=f"{label}_metadata.json",
                    mime="application/json",
                    key=f"hub_export_{mid}",
                )

                # Set as primary
                if not is_primary:
                    if ac2.button(
                        t["hub_models_set_primary"],
                        key=f"hub_primary_{mid}",
                    ):
                        assignments.assign(asset, mid)
                        st.success(t["hub_models_primary_done"].format(asset=asset, label=label))
                        st.rerun()
                else:
                    ac2.caption(t["hub_models_is_primary"])

                # Delete model
                if ac3.button(
                    t["hub_models_delete"],
                    key=f"hub_del_{mid}",
                ):
                    st.session_state[f"_hub_confirm_del_{mid}"] = True

                if st.session_state.get(f"_hub_confirm_del_{mid}"):
                    confirm = st.text_input(
                        t["hub_models_confirm_delete"],
                        key=f"hub_confirm_input_{mid}",
                    )
                    if confirm == "DELETE":
                        try:
                            registry.delete_model(mid)
                            if is_primary:
                                assignments.unassign(asset)
                            st.success(t["hub_models_deleted"])
                            st.session_state.pop(f"_hub_confirm_del_{mid}", None)
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

                st.divider()


# =====================================================================
# 3. Forecasts
# =====================================================================

def _render_forecasts_section(t: Dict[str, str], lang: str) -> None:
    with st.expander(t["hub_forecasts_header"], expanded=False):
        forecast = state.get(state.KEY_FORECAST)
        if forecast is None:
            st.info(t["hub_forecasts_empty"])
            return

        # Display cached forecast info
        asset = state.get(state.KEY_DATA_LOADED_ASSET, "?")
        model_id = state.get(state.KEY_ACTIVE_MODEL_ID, "?")
        st.caption(f"{t['hub_forecasts_asset']}: {asset} | {t['hub_forecasts_model']}: {model_id}")

        # Build DataFrame from forecast for export
        if hasattr(forecast, "prices_p50") and forecast.prices_p50 is not None:
            rows = []
            for i, (p10, p50, p90) in enumerate(zip(
                forecast.prices_p10, forecast.prices_p50, forecast.prices_p90
            )):
                rows.append({
                    t["forecast_col_day"]: i + 1,
                    "P10": round(float(p10), 4),
                    "P50": round(float(p50), 4),
                    "P90": round(float(p90), 4),
                })
            df_fc = pd.DataFrame(rows)
            st.dataframe(df_fc, use_container_width=True, hide_index=True)

            # Export forecast CSV
            csv_data = df_fc.to_csv(index=False)
            st.download_button(
                t["hub_forecasts_export"],
                csv_data,
                file_name=f"forecast_{asset}.csv",
                mime="text/csv",
                key="hub_export_forecast",
            )

        # Clear forecast
        if st.button(t["hub_forecasts_clear"], key="hub_clear_forecast"):
            state.put(state.KEY_FORECAST, None)
            st.success(t["hub_forecasts_cleared"])
            st.rerun()


# =====================================================================
# 4. Trade Log
# =====================================================================

def _render_trade_log_section(t: Dict[str, str], lang: str) -> None:
    with st.expander(t["hub_trades_header"], expanded=False):
        store = TradeLogStore()
        entries = store.load_all()

        if not entries:
            st.info(t["hub_trades_empty"])
            return

        stats = store.summary_stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t["portfolio_total"], stats["total"])
        c2.metric(t["portfolio_open"], stats["open"])
        c3.metric(t["portfolio_closed"], stats["closed"])
        wr = stats["win_rate"]
        c4.metric(t["portfolio_win_rate"], f"{wr:.1f}%")

        # Trade log table
        rows = []
        for e in entries:
            rows.append({
                t["portfolio_col_date"]: e.timestamp[:10],
                t["portfolio_col_asset"]: e.asset,
                t["portfolio_col_signal"]: e.signal,
                t["portfolio_col_conf"]: f"{e.confidence:.0f}%",
                t["portfolio_col_expected"]: f"{e.expected_return_pct:+.2f}%",
                t["portfolio_col_actual"]: f"{e.actual_return_pct:+.2f}%" if e.actual_return_pct is not None else "—",
                t["portfolio_col_status"]: e.status,
                t["portfolio_col_investment"]: f"${e.investment:,.2f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Export CSV
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        st.download_button(
            t["hub_trades_export"],
            buf.getvalue(),
            file_name="trade_log.csv",
            mime="text/csv",
            key="hub_export_trades",
        )


# =====================================================================
# 5. Model Performance History
# =====================================================================

def _render_performance_section(t: Dict[str, str], lang: str) -> None:
    with st.expander(t["hub_performance_header"], expanded=False):
        store = TradeLogStore()
        closed = [e for e in store.load_all() if e.status == "closed"]

        if not closed:
            st.info(t["hub_performance_empty"])
            return

        # Build a performance summary by model
        model_stats: Dict[str, Dict[str, Any]] = {}
        for e in closed:
            key = e.model_label or e.model_id
            if key not in model_stats:
                model_stats[key] = {
                    "trades": 0, "wins": 0,
                    "pred_returns": [], "actual_returns": [],
                    "errors": [],
                }
            s = model_stats[key]
            s["trades"] += 1
            actual = e.actual_return_pct or 0
            predicted = e.expected_return_pct or 0
            if actual > 0:
                s["wins"] += 1
            s["pred_returns"].append(predicted)
            s["actual_returns"].append(actual)
            s["errors"].append(abs(actual - predicted))

        rows = []
        for model_label, s in model_stats.items():
            win_rate = (s["wins"] / s["trades"] * 100) if s["trades"] else 0
            avg_err = sum(s["errors"]) / len(s["errors"]) if s["errors"] else 0
            avg_pred = sum(s["pred_returns"]) / len(s["pred_returns"]) if s["pred_returns"] else 0
            avg_actual = sum(s["actual_returns"]) / len(s["actual_returns"]) if s["actual_returns"] else 0
            bias = avg_pred - avg_actual
            rows.append({
                t["hub_perf_model"]: model_label,
                t["hub_perf_trades"]: s["trades"],
                t["hub_perf_win_rate"]: f"{win_rate:.1f}%",
                t["hub_perf_mae"]: f"{avg_err:.2f}%",
                t["hub_perf_bias"]: f"{bias:+.2f}%",
                t["hub_perf_avg_pred"]: f"{avg_pred:+.2f}%",
                t["hub_perf_avg_actual"]: f"{avg_actual:+.2f}%",
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Degradation flags
        for model_label, s in model_stats.items():
            win_rate = (s["wins"] / s["trades"] * 100) if s["trades"] else 0
            if win_rate < 40 and s["trades"] >= 3:
                st.warning(t["hub_perf_degradation"].format(model=model_label, wr=f"{win_rate:.0f}"))


# =====================================================================
# Global Actions
# =====================================================================

def _render_global_actions(t: Dict[str, str], lang: str) -> None:
    st.subheader(t["hub_global_header"])

    c1, c2 = st.columns(2)

    # Export all data as ZIP
    if c1.button(t["hub_global_export_all"], key="hub_export_all", type="primary"):
        zip_buf = _create_export_zip()
        if zip_buf:
            st.download_button(
                t["hub_global_download_zip"],
                zip_buf.getvalue(),
                file_name=f"gldpred_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                key="hub_download_zip",
            )
        else:
            st.info(t["hub_global_nothing_to_export"])

    # Reset all application data
    if c2.button(t["hub_global_reset"], key="hub_reset_btn"):
        st.session_state["_hub_confirm_reset"] = True

    if st.session_state.get("_hub_confirm_reset"):
        st.warning(t["hub_global_reset_warning"])
        confirm = st.text_input(
            t["hub_global_reset_confirm"],
            key="hub_reset_confirm_input",
        )
        if confirm == "RESET":
            _reset_all_data()
            st.session_state.pop("_hub_confirm_reset", None)
            st.success(t["hub_global_reset_done"])
            st.rerun()


def _create_export_zip() -> io.BytesIO | None:
    """Create a ZIP file with all exportable data."""
    buf = io.BytesIO()
    has_data = False

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Market data
        df = state.get(state.KEY_RAW_DF)
        asset = state.get(state.KEY_DATA_LOADED_ASSET, "unknown")
        if df is not None:
            zf.writestr(f"market_data/{asset}.csv", df.to_csv())
            has_data = True

        # Model metadata
        registry = ModelRegistry()
        for meta in registry.list_models():
            mid = meta["model_id"]
            zf.writestr(
                f"models/{mid}_metadata.json",
                json.dumps(meta, indent=2, default=str),
            )
            has_data = True

        # Trade log
        trade_path = Path("data/trade_log/trades.jsonl")
        if trade_path.exists():
            zf.write(trade_path, "trade_log/trades.jsonl")
            has_data = True

        # Forecast
        forecast = state.get(state.KEY_FORECAST)
        if forecast and hasattr(forecast, "prices_p50") and forecast.prices_p50 is not None:
            rows = []
            for i, (p10, p50, p90) in enumerate(zip(
                forecast.prices_p10, forecast.prices_p50, forecast.prices_p90
            )):
                rows.append({"day": i + 1, "P10": float(p10), "P50": float(p50), "P90": float(p90)})
            zf.writestr(
                f"forecasts/{asset}_forecast.json",
                json.dumps(rows, indent=2),
            )
            has_data = True

    if not has_data:
        return None
    buf.seek(0)
    return buf


def _reset_all_data() -> None:
    """Clear all persisted application data."""
    # Clear trade log
    trade_path = Path("data/trade_log/trades.jsonl")
    if trade_path.exists():
        trade_path.unlink()

    # Clear forecast cache
    state.put(state.KEY_FORECAST, None)

    # Clear session data
    state.clear_data_state()
    state.clear_training_state()

    # Clear model registry (preserve directory)
    registry_dir = Path("data/model_registry")
    if registry_dir.exists():
        for entry in registry_dir.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)

    # Clear assignments
    assignments_path = Path("data/model_registry/assignments.json")
    if assignments_path.exists():
        assignments_path.unlink()
