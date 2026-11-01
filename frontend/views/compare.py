"""Compare and rank several resumes against one job description."""

from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from api_client import ApiError
from ui import (
    ACCENT,
    INK,
    api_error,
    band_class,
    disclaimer,
    get_client,
    html,
    page_header,
    result_detail,
    sample_jobs,
    sample_resumes,
    score_header,
)

MATCHED_FILL = "#BFE3CD"
PARTIAL_FILL = "#F6DFAE"
MISSING_FILL = "#F4C7C1"
COMPONENT_COLORS = {
    "required_skills": ACCENT,
    "preferred_skills": "#6C8EBF",
    "text_similarity": "#A9BCD6",
    "experience": "#7A8B99",
    "education": "#C5CDD6",
}
COMPONENT_LABELS = {
    "required_skills": "Required skills",
    "preferred_skills": "Preferred skills",
    "text_similarity": "Content similarity",
    "experience": "Experience",
    "education": "Education",
}

page_header(
    "Shortlist support",
    "Compare resumes",
    "Rank two or more resumes against the same job description and see where each one is strong or missing skills.",
)

samples = sample_resumes()
jobs = sample_jobs()

left, right = st.columns([1, 1.25], gap="large")
with left:
    st.markdown("#### Resumes")
    uploads = st.file_uploader("Resumes (PDF, 2-10 files, up to 5 MB each)", type=["pdf"], accept_multiple_files=True)
    use_samples = st.toggle("Add sample resumes", value=False, disabled=not samples)
    chosen_samples: list[str] = []
    if use_samples:
        chosen_samples = st.multiselect("Sample resumes (fictional)", list(samples), default=list(samples)[:5])

with right:
    st.markdown("#### Job description")
    if jobs:
        picked = st.selectbox("Start from a sample job description", ["(paste my own)", *jobs], key="cmp_pick")
        if picked != "(paste my own)" and st.session_state.get("_cmp_sample") != picked:
            st.session_state["cmp_jd"] = jobs[picked]
            st.session_state["_cmp_sample"] = picked
    jd_text = st.text_area(
        "Job description",
        key="cmp_jd",
        height=260,
        label_visibility="collapsed",
        placeholder="Paste the job description...",
    )

documents: list[tuple[str, bytes]] = [(u.name, u.getvalue()) for u in uploads or []]
documents += [(samples[name].name, samples[name].read_bytes()) for name in chosen_samples]
ready = len(documents) >= 2 and len(jd_text.strip()) > 0

if st.button("Compare resumes", type="primary", icon=":material/compare_arrows:", disabled=not ready):
    with st.spinner(f"Analysing {len(documents)} resumes..."):
        try:
            st.session_state["comparison"] = get_client().compare(documents, jd_text)
        except ApiError as exc:
            api_error(str(exc))
if not ready:
    st.caption(f"{len(documents)} resume(s) selected. Add at least two resumes and a job description.")

comparison = st.session_state.get("comparison")
if not comparison:
    st.stop()

st.divider()
st.subheader("Ranking")
for failure in comparison["failed"]:
    st.warning(f"Skipped {failure['resume_name']}: {failure['error']}", icon=":material/warning:")

rankings = comparison["rankings"]
table = pd.DataFrame(
    [
        {
            "Rank": r["rank"],
            "Resume": r["resume_name"],
            "Score": r["overall_score"],
            "Band": r["band"],
            "Required skills": f"{r['required_matched']} / {r['required_total']}" if r["required_total"] else "n/a",
            "Missing required": ", ".join(r["missing_required"]) or "-",
            "Shown in experience": ", ".join(r["strengths"]) or "-",
        }
        for r in rankings
    ]
)
st.dataframe(
    table,
    hide_index=True,
    column_config={
        "Rank": st.column_config.NumberColumn(width="small"),
        "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
        "Missing required": st.column_config.TextColumn(width="medium"),
        "Shown in experience": st.column_config.TextColumn(width="medium"),
    },
)

# Stacked contributions show *why* each resume ranks where it does.
st.markdown("#### What each score is made of")
names = [r["resume_name"] for r in rankings][::-1]
fig = go.Figure()
keys = [k for k in COMPONENT_LABELS if any(k in r["component_scores"] for r in rankings)]
for key in keys:
    fig.add_bar(
        y=names,
        x=[r["component_scores"].get(key, 0) for r in rankings][::-1],
        orientation="h",
        name=COMPONENT_LABELS[key],
        marker_color=COMPONENT_COLORS[key],
        hovertemplate="%{y}<br>" + COMPONENT_LABELS[key] + ": %{x:.1f} points<extra></extra>",
    )
fig.update_layout(
    barmode="stack",
    height=90 + 44 * len(names),
    margin=dict(l=10, r=20, t=10, b=10),
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color=INK, size=13),
    xaxis=dict(title="Points (0-100)", range=[0, 100], gridcolor="#F0F2F5"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(fig, config={"displayModeBar": False})

matrix = comparison["skill_matrix"]
job = comparison["job_analysis"]
skills = job["required_skills"] + job["preferred_skills"]
if skills:
    st.markdown("#### Skill coverage matrix")
    st.caption("✓ matched · ~ partial credit via a related skill · ✗ missing. Preferred skills are marked with *.")
    status_value = {"missing": 0, "partial": 1, "matched": 2}
    symbols = {"missing": "✗", "partial": "~", "matched": "✓"}
    resume_names = [r["resume_name"] for r in rankings]
    statuses = [[matrix[name].get(skill, "missing") for skill in skills] for name in resume_names]
    x_labels = [s if s in job["required_skills"] else f"{s}*" for s in skills]
    heat = go.Figure(
        go.Heatmap(
            z=[[status_value[s] for s in row] for row in statuses],
            x=x_labels,
            y=resume_names,
            text=[[symbols[s] for s in row] for row in statuses],
            customdata=statuses,
            texttemplate="%{text}",
            textfont=dict(color=INK, size=14),
            zmin=0,
            zmax=2,
            showscale=False,
            xgap=2,
            ygap=2,
            colorscale=[
                [0, MISSING_FILL],
                [0.33, MISSING_FILL],
                [0.34, PARTIAL_FILL],
                [0.66, PARTIAL_FILL],
                [0.67, MATCHED_FILL],
                [1, MATCHED_FILL],
            ],
            hovertemplate="%{y}<br>%{x}: %{customdata}<extra></extra>",
        )
    )
    heat.update_layout(
        height=120 + 38 * len(resume_names),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=INK, size=12),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(side="top", tickangle=-35),
    )
    st.plotly_chart(heat, config={"displayModeBar": False})

st.download_button(
    "Download ranking (.csv)",
    table.to_csv(index=False),
    file_name="resume_ranking.csv",
    mime="text/csv",
    icon=":material/download:",
)

st.divider()
st.subheader("Inspect a resume")
by_name = {r["resume_name"]: r for r in comparison["results"]}
selected = st.selectbox("Resume", list(by_name), label_visibility="collapsed")
chosen = by_name[selected]
html(
    f'<span class="rm-pill {band_class(chosen["overall_score"])}">Rank '
    f"{next(r['rank'] for r in rankings if r['resume_name'] == selected)} of {len(rankings)}</span> "
    f'<span style="color:#5F6B7A">{escape(selected)}</span>'
)
st.write("")
score_header(chosen)
st.write("")
result_detail(chosen, key_prefix="cmp_")
st.write("")
disclaimer(comparison["disclaimer"])
