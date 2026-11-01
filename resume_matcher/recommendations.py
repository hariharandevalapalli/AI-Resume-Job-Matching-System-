"""Personalised resume suggestions and a learning roadmap derived from the match analysis."""

from __future__ import annotations

from resume_matcher.config import RecommendationConfig, Taxonomy
from resume_matcher.schemas import (
    JobAnalysis,
    Recommendation,
    Resource,
    ResumeProfile,
    RoadmapStep,
    SkillMatch,
    TextSimilarity,
)

_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _join(items: list[str], limit: int = 5) -> str:
    shown = items[:limit]
    text = ", ".join(shown)
    return f"{text} and {len(items) - limit} more" if len(items) > limit else text


class RecommendationEngine:
    def __init__(self, taxonomy: Taxonomy, config: RecommendationConfig) -> None:
        self.taxonomy = taxonomy
        self.config = config
        self._skills = taxonomy.by_name()

    # ------------------------------------------------------------------ suggestions

    def recommend(
        self,
        required: list[SkillMatch],
        preferred: list[SkillMatch],
        similarity: TextSimilarity,
        profile: ResumeProfile,
        job: JobAnalysis,
    ) -> list[Recommendation]:
        recs: list[Recommendation] = []

        missing_required = [m.name for m in required if m.status == "missing"]
        if missing_required:
            recs.append(
                Recommendation(
                    priority="high",
                    category="skills",
                    title="Address missing required skills",
                    detail=f"The job lists {_join(missing_required)} as required, "
                    "but the resume does not mention them.",
                    action="If you have used these skills, add a bullet under Experience or Projects that shows how. "
                    "If not, see the learning roadmap for a focused plan.",
                )
            )

        listed_only = [m.name for m in required + preferred if m.evidence_strength == "listed"]
        if listed_only:
            recs.append(
                Recommendation(
                    priority="high" if len(listed_only) >= 3 else "medium",
                    category="evidence",
                    title="Back up listed skills with evidence",
                    detail=f"{_join(listed_only)} appear only in a skills list or summary, which earns less credit "
                    "than skills shown in real work.",
                    action="For each skill, add one Experience or Projects bullet: what you built with it and "
                    "the result (for example, 'Built a FastAPI service that cut report time by 40%').",
                )
            )

        partial = [m for m in required + preferred if m.status == "partial"]
        if partial:
            pairs = [f"{m.name} (you have {m.matched_via})" for m in partial]
            recs.append(
                Recommendation(
                    priority="medium",
                    category="skills",
                    title="Bridge from related skills",
                    detail=f"Related experience earned partial credit for {_join(pairs, 4)}.",
                    action="Mention any exposure to the exact tools, or note transferable experience explicitly; "
                    "a short side project with the exact tool closes the gap fastest.",
                )
            )

        missing_preferred = [m.name for m in preferred if m.status == "missing"]
        if missing_preferred:
            recs.append(
                Recommendation(
                    priority="low",
                    category="skills",
                    title="Consider preferred skills",
                    detail=f"Nice-to-have skills not found: {_join(missing_preferred)}.",
                    action="These are optional, but adding one or two (if you genuinely have them) can set your "
                    "application apart.",
                )
            )

        if similarity.score < self.config.low_similarity_threshold and similarity.missing_terms:
            recs.append(
                Recommendation(
                    priority="medium",
                    category="keywords",
                    title="Align wording with the job description",
                    detail="The resume's wording overlaps little with the job description. Frequent job terms not "
                    f"found in the resume: {_join(similarity.missing_terms, 6)}.",
                    action="Where it is accurate, describe your work using the same terms the employer uses. "
                    "Do not add keywords you cannot back up in an interview.",
                )
            )

        if (
            job.min_years_experience
            and profile.years_experience is not None
            and profile.years_experience < job.min_years_experience
        ):
            recs.append(
                Recommendation(
                    priority="medium",
                    category="experience",
                    title="Make all relevant experience visible",
                    detail=f"About {profile.years_experience:g} years detected versus {job.min_years_experience:g}+ "
                    "requested.",
                    action="Include internships, freelance work, research and substantial projects with dates "
                    "(e.g. 'Jun 2023 - Present') so all relevant experience is counted.",
                )
            )
        elif job.min_years_experience and profile.years_experience is None:
            recs.append(
                Recommendation(
                    priority="medium",
                    category="experience",
                    title="Add dates to your experience",
                    detail="No employment dates were detected, so experience could not be measured.",
                    action="Add start and end dates to each role, for example 'Jan 2022 - Present'.",
                )
            )

        if job.education_requirement and profile.education_level is None:
            recs.append(
                Recommendation(
                    priority="low",
                    category="structure",
                    title="Make your education explicit",
                    detail="The role specifies a degree, but no degree was detected in the resume.",
                    action="Add an Education section with the degree name written out (e.g. 'Bachelor of Science in "
                    "Computer Science').",
                )
            )

        if profile.quantified_lines < self.config.min_quantified_lines:
            recs.append(
                Recommendation(
                    priority="medium",
                    category="impact",
                    title="Quantify your achievements",
                    detail=f"Only {profile.quantified_lines} line(s) contain measurable results.",
                    action="Add numbers to your strongest bullets: users served, time saved, accuracy improved, "
                    "data volume handled.",
                )
            )

        missing_sections = [s for s in ("experience", "skills", "education") if s not in profile.sections_found]
        if "projects" not in profile.sections_found and "experience" not in profile.sections_found:
            missing_sections.append("projects")
        if missing_sections:
            recs.append(
                Recommendation(
                    priority="low",
                    category="structure",
                    title="Use clear section headings",
                    detail=f"No recognisable heading found for: {_join(missing_sections)}.",
                    action="Use standard headings (Experience, Projects, Skills, Education) so readers and "
                    "applicant tracking systems can find information quickly.",
                )
            )

        if profile.word_count < self.config.min_resume_words:
            recs.append(
                Recommendation(
                    priority="low",
                    category="structure",
                    title="Add more detail",
                    detail=f"The resume has about {profile.word_count} words, which is brief.",
                    action="Expand your most relevant roles and projects with two to four specific bullets each.",
                )
            )
        elif profile.word_count > self.config.max_resume_words:
            recs.append(
                Recommendation(
                    priority="low",
                    category="structure",
                    title="Tighten the resume",
                    detail=f"The resume has about {profile.word_count} words, which is long for most roles.",
                    action="Remove older or unrelated content and keep the bullets most relevant to this job.",
                )
            )

        recs.sort(key=lambda r: _PRIORITY_ORDER[r.priority])
        return recs[: self.config.max_items]

    # ------------------------------------------------------------------ roadmap

    def _step(self, phase: int, title: str, skill: str, reason: str, effort_factor: float = 1.0) -> RoadmapStep:
        definition = self._skills[skill]
        category = self.taxonomy.categories[definition.category]
        weeks = (definition.learning_weeks or category.learning_weeks) * effort_factor
        return RoadmapStep(
            phase=phase,
            phase_title=title,
            skill=skill,
            category_label=category.label,
            reason=reason,
            estimated_weeks=round(weeks, 1),
            practice_project=category.practice_project,
            resources=[Resource(title=r.title, url=r.url) for r in definition.resources],
        )

    def roadmap(self, required: list[SkillMatch], preferred: list[SkillMatch]) -> list[RoadmapStep]:
        """Phase 1: missing required skills. Phase 2: bridge related skills. Phase 3: preferred skills.
        Phase 4: turn listed-only required skills into demonstrated ones."""
        steps: list[RoadmapStep] = []
        p1 = "Close required-skill gaps"
        for m in required:
            if m.status == "missing":
                steps.append(self._step(1, p1, m.name, "Required by the job and not found in the resume."))
        p2 = "Bridge from related skills"
        for m in required + preferred:
            if m.status == "partial":
                steps.append(
                    self._step(
                        2,
                        p2,
                        m.name,
                        f"You already know {m.matched_via}; learning {m.name} should be faster.",
                        effort_factor=0.5,
                    )
                )
        p3 = "Add preferred skills"
        for m in preferred:
            if m.status == "missing":
                steps.append(self._step(3, p3, m.name, "Listed as a nice-to-have in the job description."))
        p4 = "Show evidence for listed skills"
        for m in required:
            if m.evidence_strength == "listed":
                steps.append(
                    self._step(
                        4,
                        p4,
                        m.name,
                        "Listed on the resume but not shown in experience or projects; a small project provides proof.",
                        effort_factor=0.25,
                    )
                )
        return steps
