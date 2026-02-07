"""Minimal test to check if streamlit works."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from gldpred.app.i18n import STRINGS, LANGUAGES

st.title("Test App")
st.write("Basic content works")

# Test language loading
st.write(f"LANGUAGES: {LANGUAGES}")
st.write(f"STRINGS keys: {list(STRINGS.keys())}")

# Test getting a translation
t = STRINGS["en"]
st.write(f"Sample key 'app_title': {t.get('app_title', 'NOT FOUND')}")
