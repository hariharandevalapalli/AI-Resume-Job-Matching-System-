"""MatchEngine: orchestrates parsing, analysis, scoring, explanations and recommendations."""

from __future__ import annotations

import time
from datetime import date
from functools import lru_cache

from resume_matcher.config import AppConfig, get_config
from resume_matcher.exceptions import InvalidInputError, ResumeMatcherError
from resume_matcher.jd_analyzer import JobDescriptionAnalyzer
from resume_matcher.logging_config import get_logger
from resume_matcher.pdf_parser import extract_text
from resume_matcher.recommendations import RecommendationEngine
from resume_matcher.resume_analyzer import ResumeAnalyzer
from resume_matcher.schemas import (
    ComparisonResult,
    FailedDocument,
    JobAnalysis,
    MatchResult,
    RankedResume,
    ScoreComponent,
    SkillMatch,
)
from resume_matcher.scoring import (
    SkillScorer,
    build_components,
    compute_text_similarity,
    describe_skill_coverage,
    score_education,
    score_experience,
    skill_coverage,
)
from resume_matcher.skills import SkillExtractor
from resume_matcher.versioning import model_info

logger = get_logger(__name__)


def _clean_name(name: str) -> str:
    name = name.replace("\\", "/").rsplit("/", 1)[-1].strip() or "resume"
    return name[:120]


class MatchEngine:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        scoring = self.config.scoring
        self.extractor = SkillExtractor(self.config.taxonomy, scoring.skill_matching)
        self.jd_analyzer = JobDescriptionAnalyzer(self.extractor, scoring.jd_parsing, scoring.education.levels)
        self.resume_analyzer = ResumeAnalyzer(self.extractor, scoring)
        self.skill_scorer = SkillScorer(scoring, self.config.taxonomy)
        self.recommender = RecommendationEngine(self.config.taxonomy, scoring.recommendations)
        self.model_info = model_info(self.config)

    # ------------------------------------------------------------------ validation

    def validate_job_description(self, text: str) -> str:
        limits = self.config.scoring.limits
        cleaned = (text or "").strip()
        if len(cleaned) < limits.min_jd_chars:
            raise InvalidInputError(
                f"The job description is too short ({len(cleaned)} characters). "
                f"Please paste at least {limits.min_jd_chars} characters."
            )
        if len(cleaned) > limits.max_jd_chars:
            raise InvalidInputError(
                f"The job description is too long ({len(cleaned)} characters); the limit is {limits.max_jd_chars}."
            )
        return cleaned

    def validate_resume_text(self, text: str) -> str:
        limits = self.config.scoring.limits
        cleaned = (text or "").strip()
        if len(cleaned) < limits.min_resume_chars:
            raise InvalidInputError(
                f"The resume text is too short ({len(cleaned)} characters); at least {limits.min_resume_chars} "
                "are needed for a meaningful analysis."
            )
        return cleaned[: limits.max_resume_chars]

    # ------------------------------------------------------------------ public API

    def analyze_job(self, jd_text: str) -> JobAnalysis:
        return self.jd_analyzer.analyze(self.validate_job_description(jd_text))

    def match_text(
        self,
        resume_text: str,
        jd_text: str,
        resume_name: str = "resume",
        job: JobAnalysis | None = None,
        page_count: int | None = None,
        today: date | None = None,
    ) -> MatchResult:
        started = time.perf_counter()
        resume_text = self.validate_resume_text(resume_text)
        jd_text = self.validate_job_description(jd_text)
        job = job or self.jd_analyzer.analyze(jd_text)
        analyzed = self.resume_analyzer.analyze(resume_text, page_count=page_count, today=today)
        profile = analyzed.profile
        scoring = self.config.scoring

        required = self.skill_scorer.match_skills(job.required_skills, "required", profile.skills, job.alternatives)
        preferred = self.skill_scorer.match_skills(job.preferred_skills, "preferred", profile.skills, job.alternatives)
        similarity = compute_text_similarity(self.jd_analyzer.relevant_text(jd_text), analyzed.text, scoring)

        sim_expl = (
            f"TF-IDF cosine similarity is {similarity.cosine:.2f}; values of {scoring.tfidf.saturation:g} or more "
            "count as full overlap."
        )
        if similarity.matched_terms:
            sim_expl += " Top shared terms: " + ", ".join(t.term for t in similarity.matched_terms[:5]) + "."

        raw = {
            "required_skills": (
                bool(required),
                skill_coverage(required),
                describe_skill_coverage(required, "required")
                if required
                else "No required skills from the taxonomy were detected in the job description.",
            ),
            "preferred_skills": (
                bool(preferred),
                skill_coverage(preferred),
                describe_skill_coverage(preferred, "preferred")
                if preferred
                else "The job description lists no separate preferred skills.",
            ),
            "text_similarity": (True, similarity.score, sim_expl),
            "experience": score_experience(job, profile),
            "education": score_education(job, profile, scoring),
        }
        components, overall = build_components(raw, scoring)
        band = scoring.band_for(overall)

        job_skill_names = set(job.required_skills) | set(job.preferred_skills)
        additional = [s.name for s in profile.skills if s.name not in job_skill_names]

        result = MatchResult(
            resume_name=_clean_name(resume_name),
            overall_score=overall,
            band=band,
            summary=self._summary(overall, band, required, preferred, components),
            components=components,
            required_skills=required,
            preferred_skills=preferred,
            additional_skills=additional,
            text_similarity=similarity,
            recommendations=self.recommender.recommend(required, preferred, similarity, profile, job),
            roadmap=self.recommender.roadmap(required, preferred),
            resume_profile=profile,
            job_analysis=job,
            model_info=self.model_info,
        )
        logger.info(
            "Match computed: score=%.1f band=%s required=%d preferred=%d elapsed_ms=%.0f",
            overall,
            band,
            len(required),
            len(preferred),
            (time.perf_counter() - started) * 1000,
        )
        return result

    def match_pdf(self, data: bytes, filename: str, jd_text: str, job: JobAnalysis | None = None) -> MatchResult:
        parsed = extract_text(data, _clean_name(filename), self.config.scoring.limits)
        return self.match_text(parsed.text, jd_text, resume_name=filename, job=job, page_count=parsed.page_count)

    def compare_pdfs(
        self,
        documents: list[tuple[str, bytes]],
        jd_text: str,
        rejected: list[FailedDocument] | None = None,
    ) -> ComparisonResult:
        """Rank several PDF resumes against one job description. Unreadable files are reported, not fatal.

        `rejected` carries files that were refused before parsing (e.g. too large at upload time).
        """
        limits = self.config.scoring.limits
        failed: list[FailedDocument] = list(rejected or [])
        total = len(documents) + len(failed)
        if total < 2:
            raise InvalidInputError("Please provide at least two resumes to compare.")
        if total > limits.max_resumes_compare:
            raise InvalidInputError(f"At most {limits.max_resumes_compare} resumes can be compared at once.")
        jd_text = self.validate_job_description(jd_text)
        job = self.jd_analyzer.analyze(jd_text)

        results: list[MatchResult] = []
        for filename, data in documents:
            try:
                results.append(self.match_pdf(data, filename, jd_text, job=job))
            except ResumeMatcherError as exc:
                logger.warning("Comparison skipped a document: %s", type(exc).__name__)
                failed.append(FailedDocument(resume_name=_clean_name(filename), error=exc.message))
        return self._build_comparison(job, results, failed)

    def compare_texts(self, documents: list[tuple[str, str]], jd_text: str) -> ComparisonResult:
        jd_text = self.validate_job_description(jd_text)
        job = self.jd_analyzer.analyze(jd_text)
        results: list[MatchResult] = []
        failed: list[FailedDocument] = []
        for name, text in documents:
            try:
                results.append(self.match_text(text, jd_text, resume_name=name, job=job))
            except ResumeMatcherError as exc:
                failed.append(FailedDocument(resume_name=_clean_name(name), error=exc.message))
        return self._build_comparison(job, results, failed)

    # ------------------------------------------------------------------ helpers

    def _build_comparison(
        self, job: JobAnalysis, results: list[MatchResult], failed: list[FailedDocument]
    ) -> ComparisonResult:
        if not results:
            raise InvalidInputError("None of the resumes could be analysed: " + "; ".join(f.error for f in failed))
        # Make duplicate file names distinguishable in tables and charts.
        seen: dict[str, int] = {}
        for result in results:
            count = seen.get(result.resume_name, 0)
            seen[result.resume_name] = count + 1
            if count:
                result.resume_name = f"{result.resume_name} ({count + 1})"

        ordered = sorted(results, key=lambda r: (-r.overall_score, r.resume_name))
        rankings = []
        for rank, result in enumerate(ordered, start=1):
            matched = [m for m in result.required_skills if m.status == "matched"]
            strengths = [m.name for m in matched if m.evidence_strength == "demonstrated"][:5]
            rankings.append(
                RankedResume(
                    rank=rank,
                    resume_name=result.resume_name,
                    overall_score=result.overall_score,
                    band=result.band,
                    component_scores={c.key: c.contribution for c in result.components if c.included},
                    required_matched=len(matched),
                    required_total=len(result.required_skills),
                    missing_required=[m.name for m in result.required_skills if m.status == "missing"],
                    strengths=strengths,
                )
            )
        matrix = {r.resume_name: {m.name: m.status for m in r.required_skills + r.preferred_skills} for r in ordered}
        return ComparisonResult(
            job_analysis=job,
            rankings=rankings,
            results=ordered,
            skill_matrix=matrix,
            failed=failed,
            model_info=self.model_info,
        )

    @staticmethod
    def _summary(
        overall: float,
        band: str,
        required: list[SkillMatch],
        preferred: list[SkillMatch],
        components: list[ScoreComponent],
    ) -> str:
        parts = [f"Overall score {overall:.0f}/100 ({band.lower()})."]
        if required:
            matched = sum(1 for m in required if m.status == "matched")
            partial = sum(1 for m in required if m.status == "partial")
            text = f"The resume covers {matched} of {len(required)} required skills"
            if partial:
                text += f", with partial credit for {partial} more through related skills"
            parts.append(text + ".")
        if preferred:
            matched = sum(1 for m in preferred if m.status == "matched")
            parts.append(f"It covers {matched} of {len(preferred)} preferred skills.")
        included = [c for c in components if c.included and c.effective_weight > 0]
        if included:
            strongest = max(included, key=lambda c: c.raw_score)
            weakest = min(included, key=lambda c: c.raw_score)
            if strongest.key != weakest.key:
                parts.append(
                    f"Strongest area: {strongest.label.lower()} ({strongest.raw_score:.0%}). "
                    f"Biggest gap: {weakest.label.lower()} ({weakest.raw_score:.0%})."
                )
        missing = [m.name for m in required if m.status == "missing"]
        if missing:
            parts.append(
                "Missing required skills: " + ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "") + "."
            )
        return " ".join(parts)


@lru_cache(maxsize=1)
def get_engine() -> MatchEngine:
    return MatchEngine()
