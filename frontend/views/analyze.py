"""Single resume analysis: upload a PDF, provide a job description, run the match."""

from __future__ import annotations

import streamlit as st
from api_client import ApiError
from ui import PAGES, api_error, get_client, html, page_header, sample_jobs, sample_resumes

page_header(
    "Step 1 of 2",
    "Analyze a resume",
    "Upload one PDF resume and paste the job description. The analysis takes a few seconds.",
)

samples = sample_resumes()
jobs = sample_jobs()

left, right = st.columns([1, 1.25], gap="large")

with left:
    st.markdown("#### Resume")
    source = st.radio(
        "Resume source", ["Upload a PDF", "Use a sample resume"], horizontal=True, label_visibility="collapsed"
    )
    resume_name: str | None = None
    resume_bytes: bytes | None = None
    if source == "Upload a PDF":
        upload = st.file_uploader("Resume (PDF, up to 5 MB)", type=["pdf"], accept_multiple_files=False)
        if upload is not None:
            resume_name, resume_bytes = upload.name, upload.getvalue()
    elif samples:
        choice = st.selectbox("Sample resume (fictional candidates)", list(samples))
        path = samples[choice]
        resume_name, resume_bytes = path.name, path.read_bytes()
        st.caption(f"Using `{path.name}`.")
    else:
        st.info("No sample PDFs found. Run `python scripts/generate_sample_pdfs.py` to create them.")
    html(
        '<div class="rm-note">Only text-based PDFs are supported (no scanned images). Files are analysed in '
        "memory and are not saved.</div>"
    )

with right:
    st.markdown("#### Job description")
    if jobs:
        options = ["(paste my own)", *jobs]
        picked = st.selectbox("Start from a sample job description", options)
        if picked != "(paste my own)" and st.session_state.get("_jd_sample") != picked:
            st.session_state["jd_text"] = jobs[picked]
            st.session_state["_jd_sample"] = picked
        elif picked == "(paste my own)" and st.session_state.get("_jd_sample"):
            st.session_state["_jd_sample"] = None
    jd_text = st.text_area(
        "Job description",
        key="jd_text",
        height=320,
        placeholder="Paste the full job description, including requirements and nice-to-have skills...",
        label_visibility="collapsed",
    )
    st.caption(
        f"{len(jd_text.strip())} characters · tip: keep the 'Requirements' and 'Nice to have' headings, "
        "they help separate required from preferred skills."
    )

st.write("")
ready = resume_bytes is not None and len(jd_text.strip()) > 0
if st.button("Run analysis", type="primary", icon=":material/play_arrow:", disabled=not ready):
    with st.spinner("Reading the resume and scoring the match..."):
        try:
            result = get_client().match(resume_name or "resume.pdf", resume_bytes or b"", jd_text)
        except ApiError as exc:
            api_error(str(exc))
        else:
            st.session_state["result"] = result
            st.switch_page(PAGES["results"])
if not ready:
    st.caption("Add a resume and a job description to enable the analysis.")
