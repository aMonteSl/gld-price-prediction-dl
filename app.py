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

# Launch the Streamlit application.
from gldpred.app.streamlit_app import main  # noqa: E402

main()
