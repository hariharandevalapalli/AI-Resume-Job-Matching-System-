"""Transparent, configuration-driven scoring.

Overall score (0-100) = sum over included components of
    effective_weight * raw_score * 100
where effective weights are the configured weights re-normalised over the
components that can be computed for this job description.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from resume_matcher.config import ScoringConfig, Taxonomy
from resume_matcher.schemas import (
    ExtractedSkill,
    JobAnalysis,
    ResumeProfile,
    ScoreComponent,
    SkillMatch,
    TermContribution,
    TextSimilarity,
)

COMPONENT_LABELS = {
    "required_skills": "Required skills",
    "preferred_skills": "Preferred skills",
    "text_similarity": "Content similarity",
    "experience": "Experience",
    "education": "Education",
}

EDUCATION_LABELS = {
    "high_school": "High school",
    "diploma": "Diploma / associate degree",
    "bachelor": "Bachelor's degree",
    "master": "Master's degree",
    "phd": "PhD",
}


# ------------------------------------------------------------------ text similarity


def compute_text_similarity(jd_text: str, resume_text: str, config: ScoringConfig) -> TextSimilarity:
    """TF-IDF cosine similarity between the JD and the resume, with per-term contributions.

    The vectorizer is fitted on the (JD, resume) pair only, so a resume gets the same
    score whether it is analysed alone or as part of a comparison.
    """
    cfg = config.tfidf
    vectorizer = TfidfVectorizer(
        stop_words=sorted(ENGLISH_STOP_WORDS | {w.lower() for w in cfg.extra_stop_words}),
        ngram_range=tuple(cfg.ngram_range),
        sublinear_tf=cfg.sublinear_tf,
        max_features=cfg.max_features,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b",
        lowercase=True,
    )
    try:
        matrix = vectorizer.fit_transform([jd_text, resume_text])
    except ValueError:  # empty vocabulary, e.g. only stop words
        return TextSimilarity(cosine=0.0, score=0.0, matched_terms=[], missing_terms=[])

    dense = matrix.toarray()
    jd_vec, res_vec = dense[0], dense[1]
    products = jd_vec * res_vec
    cosine = float(products.sum())  # rows are L2-normalised, so the dot product is the cosine
    terms = vectorizer.get_feature_names_out()

    matched: list[TermContribution] = []
    if cosine > 0:
        for idx in np.argsort(products)[::-1][: cfg.top_terms]:
            if products[idx] <= 0:
                break
            matched.append(TermContribution(term=str(terms[idx]), share=round(float(products[idx] / cosine), 4)))

    # Single words only: missing bigrams such as "build operate" are mostly noise.
    missing_idx = [i for i in np.argsort(jd_vec)[::-1] if jd_vec[i] > 0 and res_vec[i] == 0 and " " not in terms[i]]
    missing = [str(terms[i]) for i in missing_idx[: cfg.top_terms]]

    score = min(1.0, cosine / cfg.saturation)
    return TextSimilarity(cosine=round(cosine, 4), score=round(score, 4), matched_terms=matched, missing_terms=missing)


# ------------------------------------------------------------------ skills


class SkillScorer:
    def __init__(self, config: ScoringConfig, taxonomy: Taxonomy) -> None:
        self.config = config
        self.taxonomy = taxonomy
        self._categories = {s.name: s.category for s in taxonomy.skills}
        self._related = taxonomy.related_map()

    def _weight(self, skill: str) -> float:
        return self.config.skill_matching.category_weights.get(self._categories[skill], 1.0)

    def match_skills(
        self,
        job_skills: list[str],
        requirement: str,
        resume_skills: list[ExtractedSkill],
        alternatives: dict[str, list[str]] | None = None,
    ) -> list[SkillMatch]:
        """Classify each job skill as matched (directly or via an accepted alternative),
        partial (a related skill is present) or missing."""
        cfg = self.config.skill_matching
        by_name = {s.name: s for s in resume_skills}
        alternatives = alternatives or {}
        listed = set(job_skills)
        matches: list[SkillMatch] = []
        for skill in job_skills:
            category = self._categories[skill]
            label = self.taxonomy.categories[category].label
            # An "A or B" group is one requirement: its members share the weight.
            group_size = 1 + sum(1 for a in alternatives.get(skill, []) if a in listed)
            weight = self._weight(skill) / group_size
            # Use the strongest evidence among the skill itself and its accepted alternatives,
            # preferring the exact skill when the evidence is equally strong.
            options = [by_name[s] for s in [skill, *alternatives.get(skill, [])] if s in by_name]
            found = max(options, key=lambda s: (s.demonstrated, s.name == skill, s.mentions)) if options else None
            via = found.name if found is not None and found.name != skill else None
            if found is not None:
                demonstrated = found.demonstrated
                matches.append(
                    SkillMatch(
                        name=skill,
                        category=category,
                        category_label=label,
                        requirement=requirement,
                        status="matched",
                        evidence_strength="demonstrated" if demonstrated else "listed",
                        credit=cfg.demonstrated_credit if demonstrated else cfg.listed_only_credit,
                        weight=weight,
                        matched_via=via,
                        evidence=found.evidence,
                    )
                )
                continue
            related = [by_name[r] for r in sorted(self._related.get(skill, set())) if r in by_name]
            if related:
                best = max(related, key=lambda s: (s.demonstrated, s.mentions))
                matches.append(
                    SkillMatch(
                        name=skill,
                        category=category,
                        category_label=label,
                        requirement=requirement,
                        status="partial",
                        evidence_strength="related",
                        credit=cfg.related_skill_credit,
                        weight=weight,
                        matched_via=best.name,
                        evidence=best.evidence[:1],
                    )
                )
                continue
            matches.append(
                SkillMatch(
                    name=skill,
                    category=category,
                    category_label=label,
                    requirement=requirement,
                    status="missing",
                    evidence_strength="none",
                    credit=0.0,
                    weight=weight,
                )
            )
        return matches


def skill_coverage(matches: list[SkillMatch]) -> float:
    total = sum(m.weight for m in matches)
    if total == 0:
        return 0.0
    return sum(m.weight * m.credit for m in matches) / total


def describe_skill_coverage(matches: list[SkillMatch], kind: str) -> str:
    matched = [m for m in matches if m.status == "matched"]
    demonstrated = sum(1 for m in matched if m.evidence_strength == "demonstrated")
    partial = [m for m in matches if m.status == "partial"]
    missing = [m.name for m in matches if m.status == "missing"]
    parts = [f"Covers {len(matched)} of {len(matches)} {kind} skills"]
    if matched:
        parts[0] += f" ({demonstrated} shown in experience or projects, {len(matched) - demonstrated} only listed)"
    alternative = [m for m in matched if m.matched_via]
    if alternative:
        via = ", ".join(f"{m.name} satisfied by {m.matched_via}" for m in alternative)
        parts.append(f"accepted alternatives: {via}")
    if partial:
        via = ", ".join(f"{m.name} via {m.matched_via}" for m in partial)
        parts.append(f"partial credit for related skills: {via}")
    if missing:
        shown = ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else "")
        parts.append(f"missing: {shown}")
    return "; ".join(parts) + "."


# ------------------------------------------------------------------ experience & education


def score_experience(job: JobAnalysis, profile: ResumeProfile) -> tuple[bool, float, str]:
    required = job.min_years_experience
    if required is None or required <= 0:
        return False, 0.0, "The job description does not state a years-of-experience requirement."
    years = profile.years_experience
    if years is None:
        return (
            True,
            0.0,
            (
                f"The role asks for {required:g}+ years, but no employment dates or experience "
                "statements were detected in the resume."
            ),
        )
    score = min(1.0, years / required)
    verdict = "meets" if score >= 1 else "is below"
    return (
        True,
        score,
        (
            f"About {years:g} years detected ({profile.years_experience_source}), which {verdict} "
            f"the {required:g}+ years requested."
        ),
    )


def score_education(job: JobAnalysis, profile: ResumeProfile, config: ScoringConfig) -> tuple[bool, float, str]:
    required = job.education_requirement
    if required is None:
        return False, 0.0, "The job description does not state a degree requirement."
    levels = config.education.levels
    required_label = EDUCATION_LABELS.get(required, required)
    have = profile.education_level
    if have is None:
        return True, 0.0, f"The role asks for a {required_label}; no degree was detected in the resume."
    have_label = EDUCATION_LABELS.get(have, have)
    gap = levels.get(required, 0) - levels.get(have, 0)
    if gap <= 0:
        return True, 1.0, f"Highest degree detected ({have_label}) meets the {required_label} requirement."
    if gap <= 1:
        credit = config.education.one_level_below_credit
        return (
            True,
            credit,
            f"Highest degree detected ({have_label}) is one level below the {required_label} requirement.",
        )
    return True, 0.0, f"Highest degree detected ({have_label}) is below the {required_label} requirement."


# ------------------------------------------------------------------ aggregation


def build_components(
    raw: dict[str, tuple[bool, float, str]], config: ScoringConfig
) -> tuple[list[ScoreComponent], float]:
    """Combine raw component scores into weighted contributions and an overall 0-100 score."""
    weights = config.weights.model_dump()
    included_total = sum(weights[key] for key, (included, _, _) in raw.items() if included)
    components: list[ScoreComponent] = []
    overall = 0.0
    for key in COMPONENT_LABELS:
        included, score, explanation = raw[key]
        effective = weights[key] / included_total if included and included_total > 0 else 0.0
        contribution = effective * score * 100
        overall += contribution
        components.append(
            ScoreComponent(
                key=key,
                label=COMPONENT_LABELS[key],
                included=included,
                raw_score=round(max(0.0, min(1.0, score)), 4),
                configured_weight=weights[key],
                effective_weight=round(effective, 4),
                contribution=round(contribution, 2),
                explanation=explanation,
            )
        )
    if included_total == 0:
        # Every included component has zero weight: fall back to content similarity alone.
        overall = raw["text_similarity"][1] * 100
    return components, round(max(0.0, min(100.0, overall)), 1)
