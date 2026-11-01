"""Resume analysis: sections, skills with evidence, experience, education and quality signals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from resume_matcher.config import ScoringConfig
from resume_matcher.patterns import (
    EMAIL_PATTERN,
    PHONE_PATTERN,
    find_date_ranges,
    find_education_levels,
    find_years_requirements,
    is_quantified,
    looks_like_phone,
    redact_pii,
    total_months,
)
from resume_matcher.schemas import ResumeProfile
from resume_matcher.skills import SkillExtractor
from resume_matcher.text_processing import GENERAL_SECTION, SectionDetector, Sections, normalize_text, word_count


@dataclass
class AnalyzedResume:
    text: str
    sections: Sections
    profile: ResumeProfile


def _redact_name_line(text: str) -> str:
    """Resumes usually start with the candidate's name; hide it in previews."""
    lines = text.split("\n")
    if lines:
        first = lines[0].strip()
        words = first.split()
        if 1 < len(words) <= 4 and all(w[:1].isupper() for w in words) and not any(c.isdigit() for c in first):
            lines[0] = "[name]"
    return "\n".join(lines)


class ResumeAnalyzer:
    def __init__(self, extractor: SkillExtractor, config: ScoringConfig) -> None:
        self.extractor = extractor
        self.config = config
        self.detector = SectionDetector(config.resume_sections)

    def _experience_years(
        self, sections: Sections, full_text: str, today: date
    ) -> tuple[float | None, str, float | None]:
        if "experience" in sections.lines:
            source_text = sections.text("experience")
        else:
            # No experience heading: use everything except education, where dates are degrees.
            source_text = "\n".join(
                "\n".join(lines) for name, lines in sections.items() if name not in ("education", "certifications")
            )
        months = total_months(find_date_ranges(source_text, today))
        max_years = self.config.experience.max_reasonable_years
        claims = [c for c in find_years_requirements(full_text) if c <= max_years]
        claimed = max(claims) if claims else None
        if months > 0:
            return round(min(months / 12, max_years), 1), "employment dates", claimed
        if claimed is not None:
            return claimed, "stated in resume", claimed
        return None, "not detected", None

    def _education_level(self, sections: Sections, full_text: str) -> str | None:
        text = sections.text("education") or full_text
        levels = find_education_levels(text)
        if not levels:
            return None
        ranking = self.config.education.levels
        return max(levels, key=lambda lvl: ranking.get(lvl, 0))

    def analyze(
        self,
        raw_text: str,
        page_count: int | None = None,
        today: date | None = None,
    ) -> AnalyzedResume:
        today = today or date.today()
        text = normalize_text(raw_text)
        sections = self.detector.split(text)
        skills = self.extractor.extract_from_sections(sections)
        years, years_source, claimed = self._experience_years(sections, text, today)

        evidence_sections = {"experience", "projects", "achievements", GENERAL_SECTION}
        quantified = sum(
            1 for name, lines in sections.items() if name in evidence_sections for line in lines if is_quantified(line)
        )

        preview_source = _redact_name_line(text) if self.config.privacy.redact_pii_in_preview else text
        if self.config.privacy.redact_pii_in_preview:
            preview_source = redact_pii(preview_source)
        preview = preview_source[: self.config.privacy.preview_chars]

        profile = ResumeProfile(
            word_count=word_count(text),
            page_count=page_count,
            sections_found=sections.found(),
            skills=skills,
            years_experience=years,
            years_experience_source=years_source,
            claimed_years_experience=claimed,
            education_level=self._education_level(sections, text),
            has_email=bool(EMAIL_PATTERN.search(text)),
            has_phone=any(looks_like_phone(m.group(0)) for m in PHONE_PATTERN.finditer(text)),
            quantified_lines=quantified,
            preview=preview,
        )
        return AnalyzedResume(text=text, sections=sections, profile=profile)
