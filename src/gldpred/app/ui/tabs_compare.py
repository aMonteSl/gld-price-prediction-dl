"""Compare tab rendering."""
from __future__ import annotations

import traceback
from typing import Dict

import pandas as pd
import streamlit as st

from gldpred.app import state
from gldpred.app.compare_controller import CompareRow, run_comparison
from gldpred.config import SUPPORTED_ASSETS
from gldpred.registry import ModelAssignments, ModelRegistry


def render(t: Dict[str, str]) -> None:
    """Render the Compare tab."""
    st.header(t["compare_header"])
    st.info(t["compare_info"])

    registry = ModelRegistry()
    assignments = ModelAssignments()

    # -- Build comparison rows --
    all_assignments = assignments.get_all()

    if "compare_rows" not in st.session_state:
        st.session_state["compare_rows"] = []
        for ticker, model_id in all_assignments.items():
            st.session_state["compare_rows"].append(
                {"ticker": ticker, "model_id": model_id},
            )
        # If no assignments, seed with first supported asset
        if not st.session_state["compare_rows"]:
            st.session_state["compare_rows"].append(
                {"ticker": SUPPORTED_ASSETS[0], "model_id": ""},
            )

    rows = st.session_state["compare_rows"]

    # -- Render each row --
    for idx, row in enumerate(rows):
        prefix = (
            t["compare_base_label"] if idx == 0 else t["compare_vs_label"]
        )
        rc1, rc2, rc3 = st.columns([1, 3, 3])
        rc1.markdown(f"**{prefix}**")

        # Asset picker
        ticker_list = list(SUPPORTED_ASSETS)
        default_ticker_idx = (
            ticker_list.index(row["ticker"])
            if row["ticker"] in ticker_list else 0
        )
        chosen_ticker = rc2.selectbox(
            t["compare_select_asset"],
            ticker_list,
            index=default_ticker_idx,
            key=f"cmp_asset_{idx}",
            label_visibility="collapsed",
        )
        rows[idx]["ticker"] = chosen_ticker

        # Model picker
        asset_models = registry.list_models(asset=chosen_ticker)
        if not asset_models:
            rc3.warning(
                t["compare_no_models_for_asset"].format(asset=chosen_ticker),
            )
            rows[idx]["model_id"] = ""
        else:
            model_labels = [
                f"{m.get('label', m['model_id'])} "
                f"({m.get('architecture', '?')})"
                for m in asset_models
            ]
            # Pre-select primary model if exists
            primary_id = all_assignments.get(chosen_ticker)
            default_model_idx = 0
            for mi, m in enumerate(asset_models):
                if m["model_id"] == primary_id:
                    default_model_idx = mi
                    break
            chosen_model_label = rc3.selectbox(
                t["compare_select_model"],
                model_labels,
                index=default_model_idx,
                key=f"cmp_model_{idx}",
                label_visibility="collapsed",
            )
            rows[idx]["model_id"] = asset_models[
                model_labels.index(chosen_model_label)
            ]["model_id"]

    # Add / remove row
    bc1, bc2 = st.columns(2)
    if bc1.button(t["compare_add_row"], key="btn_cmp_add"):
        rows.append({"ticker": SUPPORTED_ASSETS[0], "model_id": ""})
        st.rerun()
    if len(rows) > 1 and bc2.button(
        t["compare_remove_row"], key="btn_cmp_remove",
    ):
        rows.pop()
        st.rerun()

    # Controls
    cc1, cc2 = st.columns(2)
    investment = cc1.number_input(
        t["compare_investment"],
        min_value=10.0, max_value=1_000_000.0,
        value=1000.0, step=10.0, key="compare_investment",
    )
    horizon = cc2.slider(
        t["compare_horizon"], 1, 60, 5, key="compare_horizon",
    )

    # -- Run --
    if not st.button(t["compare_btn"], key="btn_compare"):
        prev = state.get(state.KEY_COMPARE_RESULT)
        if prev:
            _render_comparison(t, prev)
        return

    # Validate rows
    valid_rows = [
        CompareRow(ticker=r["ticker"], model_id=r["model_id"])
        for r in rows
        if r["model_id"]
    ]
    if not valid_rows:
        st.warning(t["compare_no_models"])
        return

    try:
        with st.spinner(t["compare_spinner"]):
            result = run_comparison(valid_rows, investment, horizon)
            state.put(state.KEY_COMPARE_RESULT, result)
        _render_comparison(t, result)
    except Exception as exc:
        st.error(t["compare_error"].format(err=exc))
        st.code(traceback.format_exc())


def _render_comparison(t: Dict[str, str], result) -> None:
    if not result.outcomes:
        st.warning(t["compare_no_models"])
        return

    action_label = {
        "BUY": t["reco_buy"],
        "HOLD": t["reco_hold"],
        "AVOID": t["reco_avoid"],
    }

    best = result.outcomes[0]
    st.markdown(
        f"### {t['compare_best_asset']}: **{best.ticker}** -- "
        f"{action_label.get(best.recommendation.action, best.recommendation.action)} "
        f"({best.recommendation.confidence:.0f}/100)"
    )

    st.subheader(t["compare_leaderboard"])
    rows = []
    for o in result.outcomes:
        rows.append({
            t["compare_rank"]: o.rank,
            t["compare_asset"]: o.ticker,
            t["compare_action"]: action_label.get(
                o.recommendation.action, o.recommendation.action,
            ),
            t["compare_confidence"]: f"{o.recommendation.confidence:.0f}",
            t["compare_pnl_p50"]: f"${o.pnl_p50:+,.2f}",
            t["compare_pnl_pct"]: f"{o.pnl_pct_p50:+.2f}%",
            t["compare_value_p10"]: f"${o.projected_value_p10:,.2f}",
            t["compare_value_p50"]: f"${o.projected_value_p50:,.2f}",
            t["compare_value_p90"]: f"${o.projected_value_p90:,.2f}",
        })
    st.dataframe(
        pd.DataFrame(rows), use_container_width=True, hide_index=True,
    )

    for o in result.outcomes:
        with st.expander(
            t["compare_outcome_header"].format(asset=o.ticker),
        ):
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric(
                t["compare_current_price"], f"${o.current_price:,.2f}",
            )
            cc2.metric(t["compare_shares"], f"{o.shares:.4f}")
            cc3.metric(t["compare_pnl_p50"], f"${o.pnl_p50:+,.2f}")

            risk = o.recommendation.risk
            rr1, rr2, rr3 = st.columns(3)
            rr1.metric(
                t["risk_stop_loss"], f"{risk.stop_loss_pct:+.2f}%",
            )
            rr2.metric(
                t["risk_take_profit"], f"{risk.take_profit_pct:+.2f}%",
            )
            rr3.metric(
                t["risk_reward_ratio"], f"{risk.risk_reward_ratio:.2f}",
            )

            if o.recommendation.rationale:
                for r in o.recommendation.rationale[:3]:
                    st.markdown(f"- {r}")
