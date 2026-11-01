"""About & methodology: how scores are computed, live from the backend configuration."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from ui import fetch_status, html, page_header

COMPONENT_TEXT = {
    "required_skills": ("Required skills", "Weighted share of the job's required skills found in the resume."),
    "preferred_skills": ("Preferred skills", "Same calculation for nice-to-have skills."),
    "text_similarity": ("Content similarity", "TF-IDF cosine similarity between resume and job description wording."),
    "experience": ("Experience", "Years detected from employment dates, compared with the years requested."),
    "education": ("Education", "Highest degree detected, compared with the degree requested."),
}

page_header(
    "How it works",
    "About & methodology",
    "Every score is built from a small number of transparent components. All weights and rules live in "
    "configuration files and are versioned, so results are reproducible and auditable.",
)

is_up, info, error = fetch_status()

st.subheader("1. Reading the documents")
st.markdown(
    "- **PDF text extraction** with PyMuPDF, in memory. Scanned (image-only) PDFs are rejected because OCR is "
    "not included.\n"
    "- **Section detection** finds headings such as *Experience*, *Projects*, *Skills* and *Education*.\n"
    "- **Job description parsing** separates *required* from *preferred* skills using headings "
    '("Requirements", "Nice to have") and phrases such as "is a plus". Sections like *Benefits* are ignored.'
)

st.subheader("2. Finding skills")
st.markdown(
    "Skills come from a curated taxonomy with aliases (for example *k8s* → Kubernetes, *sklearn* → scikit-learn). "
    "Matching is technology-aware: *Java* is not found inside *JavaScript*, and *C* is not found inside *C++*. "
    "Each skill is classified by the strength of its evidence:"
)
credits = info["skill_credits"] if info else {"demonstrated": 1.0, "listed_only": 0.85, "related": 0.5}
st.dataframe(
    pd.DataFrame(
        [
            {"Evidence": "Shown in experience, projects or achievements", "Credit": credits["demonstrated"]},
            {"Evidence": "Listed only (skills list or summary)", "Credit": credits["listed_only"]},
            {
                "Evidence": "Missing, but a closely related skill is present (e.g. TensorFlow for PyTorch)",
                "Credit": credits["related"],
            },
            {"Evidence": "Not found", "Credit": 0.0},
        ]
    ),
    hide_index=True,
)
if info and info["category_weights"]:
    weights_text = ", ".join(f"{k} count ×{v:g}" for k, v in info["category_weights"].items())
    st.caption(f"Category weights when averaging coverage: {weights_text}.")
st.markdown(
    "**Alternatives.** When a job lists options such as *PyTorch or TensorFlow* or *Tableau / Power BI*, the "
    "group counts as one requirement: having either skill satisfies it, and the group's weight is shared so "
    "a candidate is not rewarded twice for one skill."
)

st.subheader("3. Scoring")
st.latex(
    r"\text{score} = 100 \times \sum_{c \,\in\, \text{included}} \frac{w_c}{\sum_{k \in \text{included}} w_k} \, s_c"
)
st.markdown(
    "Each component score *s* is between 0 and 1. A component is **excluded** (and the remaining weights are "
    "re-normalised) when the job description gives nothing to compare against, for example when no degree is "
    "requested. Content similarity is computed on the resume/job pair only, so a resume receives the same "
    "score whether analysed alone or in a comparison."
)
if info:
    rows = [
        {"Component": COMPONENT_TEXT[k][0], "Configured weight": v, "What it measures": COMPONENT_TEXT[k][1]}
        for k, v in info["weights"].items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(
        f"Content similarity: cosine ÷ {info['tfidf_saturation']:g}, capped at 1 (raw resume/job cosines rarely "
        "exceed this even for strong matches). Score bands: "
        + ", ".join(f"{b['label']} ≥ {b['min_score']:g}" for b in info["bands"])
        + "."
    )
else:
    st.warning(f"Live configuration unavailable: {error}")

st.subheader("4. Recommendations and roadmap")
st.markdown(
    "Suggestions are generated from the analysis: missing required skills, skills that are only listed without "
    "evidence, related-skill gaps, weak wording overlap, missing dates, unquantified achievements and unclear "
    "section headings. The learning roadmap orders gaps into phases (required → related → preferred → evidence) "
    "with effort estimates and official learning resources from the taxonomy."
)

st.subheader("5. Quality and versioning")
st.markdown(
    "- An evaluation script ranks a labelled set of resumes for several job descriptions and reports NDCG@k, "
    "MAP, MRR, precision@3 and Spearman correlation, alongside TF-IDF-only and skills-only baselines.\n"
    "- Every result includes the model version and a fingerprint of the configuration; evaluation runs are "
    "recorded in a model registry file.\n"
    "- Automated tests cover parsing, extraction, scoring, the API and edge cases."
)
if info:
    st.caption(
        f"Running: {info['model_name']} v{info['model_version']} · taxonomy v{info['taxonomy_version']} "
        f"({info['skills_in_taxonomy']} skills) · config fingerprint {info['config_fingerprint']} · "
        f"app v{info['app_version']}"
    )

st.subheader("Limitations and responsible use")
html(
    '<div class="rm-note">'
    "<b>Scores are indicative, not hiring decisions.</b> The system measures overlap between documents. It can "
    "miss skills phrased unusually or absent from the taxonomy, cannot judge quality or seniority of work, does "
    "not read images or scanned PDFs, and is tuned for English technical roles. Experience is estimated from "
    "dates and may count overlapping or part-time roles imperfectly. Keyword overlap can be gamed, so always "
    "review the evidence. Do not use these scores as the sole basis for any decision about a person.<br><br>"
    "<b>Privacy.</b> Resumes are processed in memory and are not stored by the application. Logs record only "
    "sizes, counts, scores and timings, never resume content. Previews redact names, emails, phone numbers and "
    "links."
    "</div>"
)
