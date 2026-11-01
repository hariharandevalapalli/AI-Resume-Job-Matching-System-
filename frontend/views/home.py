"""Home dashboard."""

from __future__ import annotations

from html import escape

import streamlit as st
from ui import PAGES, band_class, fetch_status, html, page_header

page_header(
    "Explainable resume screening",
    "AI Resume-Job Matcher",
    "Upload a resume and a job description to see how well they match, which skills are covered or missing, "
    "the evidence behind every number, and what to improve next.",
)

is_up, info, error = fetch_status()
if is_up and info:
    html(
        f'<div class="rm-status"><span class="rm-dot ok"></span>Analysis service connected · '
        f"model v{escape(info['model_version'])} · {info['skills_in_taxonomy']} skills in taxonomy</div>"
    )
else:
    html('<div class="rm-status"><span class="rm-dot down"></span>Analysis service unavailable</div>')
    st.warning(
        f"{error} Start it with `uvicorn resume_matcher.api.main:app --port 8000` and refresh this page.",
        icon=":material/cloud_off:",
    )

st.write("")
col1, col2, _ = st.columns([1, 1, 2])
with col1:
    if st.button("Analyze a resume", type="primary", icon=":material/upload_file:", width="stretch"):
        st.switch_page(PAGES["analyze"])
with col2:
    if st.button("Compare resumes", icon=":material/compare_arrows:", width="stretch"):
        st.switch_page(PAGES["compare"])

last = st.session_state.get("result")
if last:
    st.write("")
    html(
        f'<div class="rm-card"><div class="rm-step">Your latest analysis</div>'
        f"<h4>{escape(last['resume_name'])} · {last['overall_score']:.0f}/100 "
        f'<span class="rm-pill {band_class(last["overall_score"])}">{escape(last["band"])}</span></h4>'
        f"<p>{escape(last['summary'])}</p></div>"
    )
    if st.button("Open results", icon=":material/insights:"):
        st.switch_page(PAGES["results"])

st.write("")
st.subheader("How it works")
steps = [
    ("1. Provide", "Upload a PDF resume and paste the job description. Sample files are included to try it out."),
    (
        "2. Analyze",
        "Skills are extracted with a configurable taxonomy, wording is compared with TF-IDF, and "
        "experience and education are checked against the role.",
    ),
    (
        "3. Understand",
        "See a 0-100 score broken into weighted components, with the exact resume lines used "
        "as evidence for each skill.",
    ),
    ("4. Improve", "Get prioritised resume suggestions and a phased learning roadmap for missing skills."),
]
for col, (title, text) in zip(st.columns(4), steps, strict=True):
    with col:
        html(f'<div class="rm-card"><div class="rm-step">{escape(title)}</div><p>{escape(text)}</p></div>')

st.write("")
st.subheader("Use it responsibly")
html(
    '<div class="rm-note">Scores are indicative estimates of text and skill overlap, not judgements of a person\'s '
    "ability. They can miss skills described in unusual words and cannot see context such as career breaks or "
    "potential. Use them to support, never replace, human review. Resumes are processed in memory and are not "
    "stored by this application.</div>"
)
