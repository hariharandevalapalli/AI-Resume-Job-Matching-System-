"""Job description analysis: required vs preferred skills, experience and education."""

from __future__ import annotations

import re
from collections import Counter
from typing import Literal

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from resume_matcher.config import JDParsingConfig
from resume_matcher.patterns import find_education_levels, find_years_requirements
from resume_matcher.schemas import JobAnalysis
from resume_matcher.skills import SkillExtractor
from resume_matcher.text_processing import normalize_text, word_count

Mode = Literal["required", "preferred", "ignored"]

_DEGREE_WORDS = re.compile(r"\b(?:degree|graduate|diploma)\b", re.IGNORECASE)
_CLAUSE_SPLIT = re.compile(r"(?<=[.;!?])\s+|;\s*|\s+\|\s+")
_JD_NOISE = {
    "experience",
    "work",
    "team",
    "role",
    "years",
    "year",
    "ability",
    "strong",
    "good",
    "knowledge",
    "skills",
    "including",
    "using",
    "join",
    "looking",
    "candidate",
    "will",
    "required",
    "preferred",
    "plus",
    "requirements",
    "responsibilities",
    "etc",
    "new",
}


def _norm(text: str) -> str:
    text = text.lower().replace("&", " and ").replace("'", "")
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text).split())


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


class JobDescriptionAnalyzer:
    def __init__(self, extractor: SkillExtractor, parsing: JDParsingConfig, education_levels: dict[str, float]):
        self.extractor = extractor
        self.parsing = parsing
        self.education_levels = education_levels
        self._preferred_headers = [_norm(h) for h in parsing.preferred_headers]
        self._ignored_headers = [_norm(h) for h in parsing.ignored_headers]
        self._required_headers = [_norm(h) for h in parsing.required_headers]
        self._preferred_cues = [_norm(c) for c in parsing.preferred_cues]
        self._required_cues = [_norm(c) for c in parsing.required_cues]

    def _header_mode(self, line: str) -> Mode | None:
        """Classify a short heading line; None if the line is not a heading."""
        stripped = line.strip()
        words = stripped.rstrip(":").split()
        if not words or len(words) > 7:
            return None
        if not stripped.endswith(":") and (stripped.endswith(".") or len(words) > 5):
            return None
        key = _norm(stripped)
        # Preferred is checked first: "Preferred Qualifications" also contains "qualifications".
        for headers, mode in (
            (self._preferred_headers, "preferred"),
            (self._ignored_headers, "ignored"),
            (self._required_headers, "required"),
        ):
            if key in headers:
                return mode  # type: ignore[return-value]
        if stripped.endswith(":") or stripped.isupper():
            for headers, mode in (
                (self._preferred_headers, "preferred"),
                (self._ignored_headers, "ignored"),
                (self._required_headers, "required"),
            ):
                if any(_contains_phrase(key, h) for h in headers):
                    return mode  # type: ignore[return-value]
        return None

    def _clause_is_preferred(self, clause: str) -> bool:
        key = _norm(clause)
        return any(_contains_phrase(key, cue) for cue in self._preferred_cues)

    def _clause_is_required(self, clause: str) -> bool:
        key = _norm(clause)
        return any(_contains_phrase(key, cue) for cue in self._required_cues)

    def classify_clauses(self, text: str) -> list[tuple[Mode, str]]:
        """Split the JD into clauses labelled required / preferred / ignored."""
        mode: Mode = "required"
        labelled: list[tuple[Mode, str]] = []
        for line in normalize_text(text).split("\n"):
            if not line.strip():
                continue
            header_mode = self._header_mode(line)
            if header_mode is not None:
                mode = header_mode
                # Inline content after "Nice to have: Docker, AWS".
                _, _, rest = line.partition(":")
                if rest.strip():
                    labelled.append((mode, rest.strip()))
                continue
            for clause in _CLAUSE_SPLIT.split(line):
                clause = clause.strip()
                if not clause:
                    continue
                if mode != "ignored" and self._clause_is_preferred(clause):
                    labelled.append(("preferred", clause))
                elif mode == "preferred" and self._clause_is_required(clause):
                    labelled.append(("required", clause))
                else:
                    labelled.append((mode, clause))
        return labelled

    def _min_education(self, text: str) -> str | None:
        levels = find_education_levels(text)
        if not levels:
            return None
        return min(levels, key=lambda lvl: self.education_levels.get(lvl, 0))

    def key_terms(self, text: str, limit: int = 15) -> list[str]:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z+#.\-]*[a-zA-Z+#]", text.lower())
        counts = Counter(t for t in tokens if t not in ENGLISH_STOP_WORDS and t not in _JD_NOISE and len(t) > 2)
        return [term for term, _ in counts.most_common(limit)]

    def analyze(self, text: str) -> JobAnalysis:
        clauses = self.classify_clauses(text)
        required: dict[str, None] = {}
        preferred: dict[str, None] = {}
        alternatives: dict[str, list[str]] = {}
        for mode, clause in clauses:
            if mode == "ignored":
                continue
            # "Bachelor's in Computer Science, Statistics" names fields of study, not skills.
            if _DEGREE_WORDS.search(clause) or find_education_levels(clause):
                continue
            target = required if mode == "required" else preferred
            for skill in self.extractor.find_skills(clause):
                target.setdefault(skill, None)
            for group in self.extractor.find_alternative_groups(clause):
                for skill in group:
                    others = alternatives.setdefault(skill, [])
                    others.extend(s for s in group if s != skill and s not in others)
        preferred_only = [s for s in preferred if s not in required]

        required_text = "\n".join(c for m, c in clauses if m == "required")
        preferred_text = "\n".join(c for m, c in clauses if m == "preferred")
        years = find_years_requirements(required_text) or find_years_requirements(preferred_text)
        education = self._min_education(required_text)
        preferred_education = self._min_education(preferred_text) if education is None else None

        return JobAnalysis(
            required_skills=list(required),
            preferred_skills=preferred_only,
            alternatives=alternatives,
            min_years_experience=years[0] if years else None,
            education_requirement=education,
            preferred_education=preferred_education,
            key_terms=self.key_terms(self.relevant_text(text, clauses)),
            word_count=word_count(text),
        )

    def relevant_text(self, text: str, clauses: list[tuple[Mode, str]] | None = None) -> str:
        """The JD without ignored sections (benefits, company blurb), used for text similarity."""
        clauses = clauses if clauses is not None else self.classify_clauses(text)
        kept = "\n".join(c for m, c in clauses if m != "ignored")
        return kept or normalize_text(text)
