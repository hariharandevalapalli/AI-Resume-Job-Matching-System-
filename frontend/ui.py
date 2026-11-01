"""Shared Streamlit UI helpers: styling, API access and result rendering.

All user- or document-derived text is HTML-escaped before it is rendered with
unsafe_allow_html.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st
from api_client import ApiClient, ApiError

FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_DIR.parent
SAMPLE_RESUME_DIR = PROJECT_ROOT / "samples" / "resumes"
SAMPLE_JD_DIR = PROJECT_ROOT / "data" / "job_descriptions"

ACCENT = "#1F4E79"
GOOD = "#1F7A4D"
WARN = "#C98A1B"
BAD = "#B42318"
TRACK = "#E3E6EB"
INK = "#1A202C"
MUTED = "#5F6B7A"

STATUS_LABELS = {"matched": ("Matched", "good"), "partial": ("Partial", "warn"), "missing": ("Missing", "bad")}
STRENGTH_LABELS = {
    "demonstrated": "Shown in experience or projects",
    "listed": "Listed only (skills or summary)",
    "related": "Related skill",
    "none": "Not found",
}
SECTION_LABELS = {
    "general": "Header / untitled",
    "summary": "Summary",
    "experience": "Experience",
    "projects": "Projects",
    "skills": "Skills",
    "education": "Education",
    "certifications": "Certifications",
    "achievements": "Achievements",
    "publications": "Publications",
}
EDUCATION_LABELS = {
    "high_school": "High school",
    "diploma": "Diploma",
    "bachelor": "Bachelor's",
    "master": "Master's",
    "phd": "PhD",
}

PAGES = {
    "home": "views/home.py",
    "analyze": "views/analyze.py",
    "results": "views/results.py",
    "compare": "views/compare.py",
    "about": "views/about.py",
}


# ------------------------------------------------------------------ setup


def load_css() -> None:
    css = (FRONTEND_DIR / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient()


@st.cache_data(ttl=20, show_spinner=False)
def fetch_status() -> tuple[bool, dict[str, Any] | None, str | None]:
    """(is_up, model_info, error_message), cached briefly to keep pages fast."""
    client = get_client()
    try:
        client.health()
        return True, client.model_info(), None
    except ApiError as exc:
        return False, None, str(exc)


def html(content: str) -> None:
    st.markdown(content, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, lead: str | None = None) -> None:
    html(f'<div class="rm-eyebrow">{escape(eyebrow)}</div>')
    st.title(title)
    if lead:
        html(f'<p class="rm-lead">{escape(lead)}</p>')


def band_class(score: float) -> str:
    if score >= 75:
        return "good"
    if score >= 60:
        return "neutral"
    if score >= 40:
        return "warn"
    return "bad"


def disclaimer(text: str) -> None:
    html(f'<div class="rm-note">{escape(text)}</div>')


# ------------------------------------------------------------------ samples


_ACRONYMS = {"Ml": "ML", "Ai": "AI", "Devops": "DevOps", "Fullstack": "Full-stack"}


def _display_name(stem: str) -> str:
    return " ".join(_ACRONYMS.get(w, w) for w in stem.replace("_", " ").title().split())


def sample_resumes() -> dict[str, Path]:
    return {_display_name(p.stem): p for p in sorted(SAMPLE_RESUME_DIR.glob("*.pdf"))}


def sample_jobs() -> dict[str, str]:
    return {_display_name(p.stem): p.read_text(encoding="utf-8") for p in sorted(SAMPLE_JD_DIR.glob("*.txt"))}


# ------------------------------------------------------------------ result pieces


def score_header(result: dict[str, Any]) -> None:
    score = result["overall_score"]
    html(
        f"""
        <div class="rm-score-wrap">
          <div class="rm-score">{score:.0f}<small>/100</small></div>
          <div class="rm-score-meta">
            <span class="rm-pill {band_class(score)}">{escape(result["band"])}</span>
            <p>{escape(result["summary"])}</p>
          </div>
        </div>
        """
    )


def _tile(label: str, value: str, hint: str = "") -> str:
    return (
        f'<div class="rm-tile"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div><div class="hint">{escape(hint)}</div></div>'
    )


def metric_tiles(result: dict[str, Any]) -> None:
    req = result["required_skills"]
    pref = result["preferred_skills"]
    req_matched = sum(1 for m in req if m["status"] == "matched")
    req_partial = sum(1 for m in req if m["status"] == "partial")
    pref_matched = sum(1 for m in pref if m["status"] == "matched")
    sim = result["text_similarity"]
    profile = result["resume_profile"]
    job = result["job_analysis"]
    years = profile["years_experience"]
    need = job["min_years_experience"]
    tiles = [
        _tile(
            "Required skills",
            f"{req_matched} / {len(req)}" if req else "n/a",
            f"+{req_partial} partial" if req_partial else "matched",
        ),
        _tile("Preferred skills", f"{pref_matched} / {len(pref)}" if pref else "n/a", "matched"),
        _tile("Content similarity", f"{sim['score']:.0%}", f"cosine {sim['cosine']:.2f}"),
        _tile(
            "Experience",
            f"{years:g} yrs" if years is not None else "Not found",
            f"{need:g}+ yrs requested" if need else "no requirement stated",
        ),
    ]
    for col, tile in zip(st.columns(4), tiles, strict=True):
        with col:
            html(tile)


def contribution_chart(components: list[dict[str, Any]]) -> go.Figure:
    included = [c for c in components if c["included"]]
    labels = [c["label"] for c in included][::-1]
    max_points = [c["effective_weight"] * 100 for c in included][::-1]
    earned = [c["contribution"] for c in included][::-1]
    fig = go.Figure()
    fig.add_bar(
        y=labels,
        x=max_points,
        orientation="h",
        marker_color=TRACK,
        name="Maximum possible",
        hovertemplate="%{y}: up to %{x:.1f} points<extra></extra>",
    )
    fig.add_bar(
        y=labels,
        x=earned,
        orientation="h",
        marker_color=ACCENT,
        name="Earned",
        text=[f"{e:.1f} / {m:.1f}" for e, m in zip(earned, max_points, strict=True)],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x:.1f} points<extra></extra>",
    )
    fig.update_layout(
        barmode="overlay",
        height=60 + 52 * len(included),
        margin=dict(l=10, r=70, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=INK, size=13),
        xaxis=dict(
            title="Points toward the overall score", gridcolor="#F0F2F5", range=[0, max(max_points + [1]) * 1.15]
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        bargap=0.35,
    )
    return fig


def terms_chart(terms: list[dict[str, Any]]) -> go.Figure:
    top = terms[:10][::-1]
    fig = go.Figure(
        go.Bar(
            y=[t["term"] for t in top],
            x=[t["share"] * 100 for t in top],
            orientation="h",
            marker_color=ACCENT,
            hovertemplate="%{y}: %{x:.1f}% of similarity<extra></extra>",
        )
    )
    fig.update_layout(
        height=60 + 28 * len(top),
        margin=dict(l=10, r=20, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=INK, size=12),
        xaxis=dict(title="Share of content similarity (%)", gridcolor="#F0F2F5"),
    )
    return fig


def components_table(components: list[dict[str, Any]]) -> None:
    rows = []
    for c in components:
        if c["included"]:
            weight = f"{c['effective_weight']:.0%}"
            score = f"{c['raw_score']:.0%}"
            points = f"{c['contribution']:.1f}"
        else:
            weight, score, points = "not used", "-", "-"
        rows.append(
            f"<tr><td class='skill'>{escape(c['label'])}</td><td>{score}</td><td>{weight}</td>"
            f"<td>{points}</td><td class='evidence'>{escape(c['explanation'])}</td></tr>"
        )
    html(
        "<div class='rm-scroll'><table class='rm-table'><thead><tr>"
        "<th>Component</th><th>Score</th><th>Weight</th><th>Points</th>"
        f"<th>Why</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def skills_table(matches: list[dict[str, Any]], empty_message: str) -> None:
    if not matches:
        html(f'<div class="rm-note">{escape(empty_message)}</div>')
        return
    rows = []
    order = {"missing": 0, "partial": 1, "matched": 2}
    for m in sorted(matches, key=lambda x: (order[x["status"]], x["name"])):
        label, css = STATUS_LABELS[m["status"]]
        strength = STRENGTH_LABELS[m["evidence_strength"]]
        if m.get("matched_via"):
            kind = "Accepted alternative" if m["status"] == "matched" else "Related skill"
            strength = f"{kind}: {m['matched_via']}"
        evidence_parts = [
            f"<b>{escape(SECTION_LABELS.get(e['section'], e['section']))}:</b> {escape(e['snippet'])}"
            for e in m["evidence"][:2]
        ]
        evidence = "<br>".join(evidence_parts) if evidence_parts else "No mention found in the resume."
        rows.append(
            f"<tr><td class='skill'>{escape(m['name'])}</td>"
            f"<td><span class='rm-pill {css}'>{label}</span></td>"
            f"<td class='evidence'>{escape(strength)}<br>credit {m['credit']:.2f}</td>"
            f"<td class='evidence'>{evidence}</td></tr>"
        )
    html(
        "<div class='rm-scroll'><table class='rm-table'><thead><tr><th>Skill</th><th>Status</th><th>Evidence type</th>"
        f"<th>Evidence from resume</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def chips(items: list[str]) -> str:
    return '<div class="rm-chips">' + "".join(f"<span>{escape(i)}</span>" for i in items) + "</div>"


def recommendations_list(recs: list[dict[str, Any]]) -> None:
    if not recs:
        html('<div class="rm-note">No specific recommendations: the resume already aligns well with this role.</div>')
        return
    pill = {"high": "bad", "medium": "warn", "low": "neutral"}
    for r in recs:
        html(
            f'<div class="rm-rec {r["priority"]}">'
            f'<span class="rm-pill {pill[r["priority"]]}">'
            f"{r['priority'].title()} priority</span>"
            f'<h4 style="margin-top:0.45rem">{escape(r["title"])}</h4>'
            f'<p>{escape(r["detail"])}</p><p class="action"><b>Next step:</b> {escape(r["action"])}</p></div>'
        )


def roadmap_view(steps: list[dict[str, Any]]) -> None:
    if not steps:
        html('<div class="rm-note">No learning gaps detected for this job description.</div>')
        return
    phases: dict[int, list[dict[str, Any]]] = {}
    for step in steps:
        phases.setdefault(step["phase"], []).append(step)
    total = sum(s["estimated_weeks"] for s in steps)
    st.caption(
        f"Estimated total effort: about {total:g} weeks of part-time study. Estimates are rough guides from the "
        "skill taxonomy, not guarantees."
    )
    for phase in sorted(phases):
        items = phases[phase]
        weeks = sum(s["estimated_weeks"] for s in items)
        html(
            f'<div class="rm-phase">Phase {phase}: {escape(items[0]["phase_title"])}<span>~{weeks:g} weeks</span></div>'
        )
        for s in items:
            with st.expander(f"{s['skill']}  ·  {s['category_label']}  ·  ~{s['estimated_weeks']:g} weeks"):
                st.write(s["reason"])
                st.markdown(f"**Practice project:** {s['practice_project']}")
                if s["resources"]:
                    st.markdown("**Resources:** " + " · ".join(f"[{r['title']}]({r['url']})" for r in s["resources"]))
                else:
                    st.markdown("**Resources:** the official documentation and a small hands-on project.")


def plain_language_sections(result: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """Turn match details into concise, actionable guidance for resume owners."""
    required = result["required_skills"]
    preferred = result["preferred_skills"]
    demonstrated = [
        match["name"]
        for match in required + preferred
        if match["status"] == "matched" and match["evidence_strength"] == "demonstrated"
    ]
    missing_required = [match["name"] for match in required if match["status"] == "missing"]
    partial_required = [match for match in required if match["status"] == "partial"]
    listed = [
        match["name"]
        for match in required + preferred
        if match["status"] == "matched" and match["evidence_strength"] == "listed"
    ]
    missing_preferred = [match["name"] for match in preferred if match["status"] == "missing"]

    sections = [
        (
            "How to read this result",
            [
                f"This resume shows a {result['band'].lower()} match based on the information we could identify. "
                "The score measures written overlap; it cannot tell whether you have experience that is not "
                "included in the resume."
            ],
        )
    ]
    if demonstrated:
        sections.append(
            (
                "What already comes through",
                ["Clearly supported by the resume text: " + ", ".join(demonstrated) + "."],
            )
        )

    next_steps: list[str] = []
    if missing_required:
        next_steps.append(
            "Required skills not found in the resume: "
            + ", ".join(missing_required)
            + ". If you have used them, add a real example under Experience or Projects describing what you did "
            "and the result. If not, choose a skill from the learning roadmap and build a small practice project "
            "before listing it."
        )
    if partial_required:
        explanations = [
            f"{match['name']} (related experience with {match['matched_via']})" for match in partial_required
        ]
        next_steps.append(
            "Related experience may transfer to these required skills: "
            + ", ".join(explanations)
            + ". Explain the connection in your resume, and consider a small project using the exact tool."
        )
    if listed:
        next_steps.append(
            "These skills are mentioned but not demonstrated with a work or project example: "
            + ", ".join(listed)
            + ". Add a bullet showing how you used each one and what changed as a result."
        )
    if not next_steps:
        next_steps.append(
            "No missing required skills were detected. Make sure your experience bullets clearly show how you "
            "used the matched skills, and review the optional requirements below."
        )
    sections.append(("What to work on first", next_steps))

    if missing_preferred:
        sections.append(
            (
                "Optional skills",
                [
                    "The job also mentions "
                    + ", ".join(missing_preferred)
                    + ". These are nice-to-have, not required. Focus on required skills first, and include these "
                    "only if you genuinely have the experience."
                ],
            )
        )

    other_recommendations = [
        recommendation
        for recommendation in result["recommendations"]
        if recommendation["category"] not in {"skills", "evidence"}
    ]
    if other_recommendations:
        sections.append(
            (
                "Other useful improvements",
                [f"{recommendation['title']}: {recommendation['action']}" for recommendation in other_recommendations],
            )
        )
    return sections


def plain_language_summary(result: dict[str, Any]) -> None:
    """Show the most useful interpretation before the technical result tabs."""
    with st.container(border=True):
        st.subheader("Your results, in plain language")
        for heading, paragraphs in plain_language_sections(result):
            st.markdown(f"**{heading}**")
            for paragraph in paragraphs:
                st.write(paragraph)


def profile_view(result: dict[str, Any]) -> None:
    profile = result["resume_profile"]
    left, right = st.columns(2)
    with left:
        st.markdown("**Detected sections**")
        sections = [SECTION_LABELS.get(s, s) for s in profile["sections_found"]]
        html(chips(sections) if sections else '<div class="rm-note">No standard section headings detected.</div>')
        st.markdown("**Experience**")
        if profile["years_experience"] is not None:
            st.write(f"About {profile['years_experience']:g} years ({profile['years_experience_source']}).")
        else:
            st.write("No employment dates or experience statements detected.")
        st.markdown("**Highest degree**")
        st.write(EDUCATION_LABELS.get(profile["education_level"], "Not detected"))
    with right:
        st.markdown("**Resume quality signals**")
        st.write(
            f"- Words: {profile['word_count']}"
            + (f" · Pages: {profile['page_count']}" if profile.get("page_count") else "")
            + f"\n- Lines with measurable results: {profile['quantified_lines']}"
            + f"\n- Contact email present: {'yes' if profile['has_email'] else 'no'}"
            + f"\n- Contact phone present: {'yes' if profile['has_phone'] else 'no'}"
        )
        st.markdown("**Other skills found (not in this job description)**")
        extra = result["additional_skills"]
        html(chips(extra) if extra else '<div class="rm-note">None detected.</div>')
    with st.expander("Extracted text preview (personal details redacted)"):
        html(f'<pre class="rm-preview">{escape(profile["preview"])}</pre>')


def result_detail(result: dict[str, Any], key_prefix: str = "") -> None:
    """Full explainable breakdown for one resume (used on Results and Compare pages)."""
    plain_language_summary(result)
    tabs = st.tabs(["Score breakdown", "Skills & evidence", "Recommendations", "Learning roadmap", "Resume profile"])
    with tabs[0]:
        st.plotly_chart(
            contribution_chart(result["components"]), config={"displayModeBar": False}, key=f"{key_prefix}contrib"
        )
        components_table(result["components"])
        sim = result["text_similarity"]
        st.markdown("#### What drives content similarity")
        col1, col2 = st.columns([3, 2])
        with col1:
            if sim["matched_terms"]:
                st.plotly_chart(
                    terms_chart(sim["matched_terms"]), config={"displayModeBar": False}, key=f"{key_prefix}terms"
                )
            else:
                html('<div class="rm-note">The resume and job description share no meaningful terms.</div>')
        with col2:
            st.markdown("**Important job terms missing from the resume**")
            html(chips(sim["missing_terms"]) if sim["missing_terms"] else '<div class="rm-note">None.</div>')
    with tabs[1]:
        st.markdown("#### Required skills")
        skills_table(
            result["required_skills"], "No required skills from the taxonomy were found in the job description."
        )
        st.markdown("#### Preferred skills")
        skills_table(result["preferred_skills"], "The job description lists no separate preferred skills.")
    with tabs[2]:
        recommendations_list(result["recommendations"])
    with tabs[3]:
        roadmap_view(result["roadmap"])
    with tabs[4]:
        profile_view(result)


def markdown_report(result: dict[str, Any]) -> str:
    """Plain Markdown summary for download/sharing."""
    lines = [
        f"# Resume match report: {result['resume_name']}",
        "",
        f"**Score:** {result['overall_score']:.0f}/100 ({result['band']})",
        "",
        result["summary"],
        "",
        "## Your results, in plain language",
    ]
    for heading, paragraphs in plain_language_sections(result):
        lines += ["", f"### {heading}"]
        lines.extend(f"- {paragraph}" for paragraph in paragraphs)
    lines += [
        "",
        "## Score breakdown",
        "| Component | Score | Weight | Points | Explanation |",
        "|---|---|---|---|---|",
    ]
    for c in result["components"]:
        if c["included"]:
            lines.append(
                f"| {c['label']} | {c['raw_score']:.0%} | {c['effective_weight']:.0%} | {c['contribution']:.1f} | "
                f"{c['explanation']} |"
            )
        else:
            lines.append(f"| {c['label']} | - | not used | - | {c['explanation']} |")
    for title, key in (("Required skills", "required_skills"), ("Preferred skills", "preferred_skills")):
        lines += ["", f"## {title}"]
        for m in result[key]:
            via = f" (via {m['matched_via']})" if m.get("matched_via") else ""
            lines.append(f"- **{m['name']}**: {m['status']}{via}, {STRENGTH_LABELS[m['evidence_strength']].lower()}")
    lines += ["", "## Recommendations"]
    for r in result["recommendations"]:
        lines.append(f"- **[{r['priority']}] {r['title']}**: {r['detail']} Next step: {r['action']}")
    lines += ["", "## Learning roadmap"]
    for s in result["roadmap"]:
        lines.append(f"- Phase {s['phase']} - {s['skill']} (~{s['estimated_weeks']:g} weeks): {s['reason']}")
    info = result["model_info"]
    lines += [
        "",
        "---",
        f"Model {info['model_name']} v{info['model_version']} (config {info['config_fingerprint']}).",
        "",
        f"_{result['disclaimer']}_",
    ]
    return "\n".join(lines)


def api_error(message: str) -> None:
    st.error(message, icon=":material/error:")
