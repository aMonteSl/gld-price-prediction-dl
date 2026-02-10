"""Models tab rendering."""
from __future__ import annotations

from typing import Dict

import streamlit as st

from gldpred.app.controllers import models_controller
from gldpred.config import SUPPORTED_ASSETS


def render(t: Dict[str, str]) -> None:
    """Render the Models tab."""
    st.header(t["models_header"])
    st.info(t["models_info"])

    # Filter by asset
    filter_opts = [t["models_all_assets"]] + list(SUPPORTED_ASSETS)
    chosen_filter = st.selectbox(
        t["models_asset_filter"], filter_opts, key="models_filter",
    )
    filter_asset = (
        None if chosen_filter == t["models_all_assets"] else chosen_filter
    )

    models = models_controller.list_models(asset=filter_asset)
    if not models:
        st.info(t["models_no_models"])
        return

    all_assignments = models_controller.get_assignments()

    for meta in models:
        model_id = meta["model_id"]
        label = meta.get("label", model_id)
        asset = meta.get("asset", "?")
        arch = meta.get("architecture", "?")
        created = meta.get("created_at", "?")[:16]
        is_primary = all_assignments.get(asset) == model_id

        badge = "* " if is_primary else ""
        with st.expander(
            f"{badge}{label}  --  {asset} / {arch}  --  {created}",
            expanded=False,
        ):
            # Details
            dc1, dc2, dc3, dc4 = st.columns(4)
            dc1.metric(t["models_col_asset"], asset)
            dc2.metric(t["models_col_arch"], arch)
            dc3.metric(t["models_col_created"], created)
            dc4.metric(
                t["models_col_primary"],
                t["models_primary_badge"] if is_primary else "--",
            )

            # Config info
            cfg = meta.get("config", {})
            ts = meta.get("training_summary", {})
            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("Hidden", cfg.get("hidden_size", "?"))
            ec2.metric("Layers", cfg.get("num_layers", "?"))
            ec3.metric("Epochs", ts.get("epochs", "?"))

            st.divider()

            # -- Actions --
            ac1, ac2, ac3 = st.columns(3)

            # Rename
            with ac1:
                new_label = st.text_input(
                    t["models_rename_label"],
                    value=label,
                    key=f"rename_{model_id}",
                    max_chars=60,
                )
                if st.button(
                    t["models_rename_btn"], key=f"btn_rename_{model_id}",
                ):
                    try:
                        models_controller.update_label(model_id, new_label)
                        st.success(
                            t["models_rename_success"].format(label=new_label),
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(t["models_rename_error"].format(err=e))

            # Primary assignment
            with ac2:
                if is_primary:
                    if st.button(
                        t["models_unset_primary_btn"],
                        key=f"btn_unprimary_{model_id}",
                    ):
                        models_controller.unset_primary(asset)
                        st.success(
                            t["models_primary_removed"].format(asset=asset),
                        )
                        st.rerun()
                else:
                    if st.button(
                        t["models_set_primary_btn"],
                        key=f"btn_primary_{model_id}",
                    ):
                        models_controller.set_primary(asset, model_id)
                        st.success(
                            t["models_primary_set"].format(
                                asset=asset, label=label,
                            ),
                        )
                        st.rerun()

            # Delete
            with ac3:
                confirm = st.text_input(
                    t["models_delete_confirm"],
                    key=f"del_confirm_{model_id}",
                    placeholder="DELETE",
                )
                if st.button(
                    t["models_delete_btn"], key=f"btn_del_{model_id}",
                ):
                    if confirm.strip() == "DELETE":
                        try:
                            models_controller.delete_model(model_id, asset, is_primary)
                            st.success(t["models_delete_success"])
                            st.rerun()
                        except Exception as e:
                            st.error(
                                t["models_delete_error"].format(err=e),
                            )
                    else:
                        st.warning("Type DELETE to confirm.")

    # -- Bulk delete --
    st.divider()
    with st.expander(t["models_bulk_delete_header"]):
        count = len(models)
        st.warning(t["models_bulk_confirm"].format(count=count))
        bulk_confirm = st.text_input(
            t["registry_confirm_input"],
            key="bulk_del_confirm",
            placeholder="DELETE ALL",
        )
        if st.button(t["models_bulk_delete_btn"], key="btn_bulk_del"):
            if bulk_confirm.strip() == "DELETE ALL":
                try:
                    deleted = models_controller.bulk_delete(models, filter_asset)
                    st.success(
                        t["registry_delete_success"].format(count=deleted),
                    )
                    st.rerun()
                except Exception as e:
                    st.error(t["registry_delete_error"].format(err=e))
            else:
                st.warning("Type DELETE ALL to confirm.")
