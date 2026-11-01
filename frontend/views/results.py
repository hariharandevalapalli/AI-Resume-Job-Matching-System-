"""Results dashboard for the latest single-resume analysis."""

from __future__ import annotations

import json

import streamlit as st
from ui import PAGES, disclaimer, markdown_report, metric_tiles, page_header, result_detail, score_header

result = st.session_state.get("result")

if not result:
    page_header("Step 2 of 2", "Results", "No analysis yet. Run one to see the score, evidence and recommendations.")
    if st.button("Analyze a resume", type="primary", icon=":material/upload_file:"):
        st.switch_page(PAGES["analyze"])
    st.stop()

page_header("Match results", result["resume_name"])
score_header(result)
st.write("")
metric_tiles(result)
st.write("")
result_detail(result, key_prefix="single_")

st.divider()
col1, col2, col3, _ = st.columns([1, 1, 1, 1.5])
stem = result["resume_name"].rsplit(".", 1)[0]
with col1:
    st.download_button(
        "Download report (.md)",
        markdown_report(result),
        file_name=f"{stem}_match_report.md",
        mime="text/markdown",
        icon=":material/description:",
        width="stretch",
    )
with col2:
    st.download_button(
        "Download data (.json)",
        json.dumps(result, indent=2),
        file_name=f"{stem}_match.json",
        mime="application/json",
        icon=":material/data_object:",
        width="stretch",
    )
with col3:
    if st.button("Analyze another", icon=":material/refresh:", width="stretch"):
        st.switch_page(PAGES["analyze"])

info = result["model_info"]
st.caption(f"Model {info['model_name']} v{info['model_version']} · config {info['config_fingerprint']}")
disclaimer(result["disclaimer"])
