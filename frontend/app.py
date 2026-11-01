"""Streamlit entry point.  Run:  streamlit run frontend/app.py"""

from __future__ import annotations

import streamlit as st
from ui import PAGES, load_css

st.set_page_config(
    page_title="AI Resume-Job Matcher",
    page_icon=":material/work:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
load_css()

navigation = st.navigation(
    [
        st.Page(PAGES["home"], title="Home", icon=":material/home:", default=True),
        st.Page(PAGES["analyze"], title="Analyze resume", icon=":material/upload_file:"),
        st.Page(PAGES["results"], title="Results", icon=":material/insights:"),
        st.Page(PAGES["compare"], title="Compare resumes", icon=":material/compare_arrows:"),
        st.Page(PAGES["about"], title="About & methodology", icon=":material/info:"),
    ],
    position="top",
)
navigation.run()
