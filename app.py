"""Thin entrypoint for the Streamlit GUI.

Run with::

    streamlit run app.py

All application logic lives in ``src/gldpred/app/streamlit_app.py``.
"""
import sys
import os

# Ensure ``import gldpred`` works regardless of working directory.
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Import triggers Streamlit execution at module level.
import gldpred.app.streamlit_app as _app  # noqa: F401, E402
