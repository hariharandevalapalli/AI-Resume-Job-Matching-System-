"""Rule-based skill extraction driven by the configurable taxonomy."""

from __future__ import annotations

import re
from dataclasses import dataclass

from resume_matcher.config import SkillMatchingConfig, Taxonomy
from resume_matcher.schemas import ExtractedSkill, SkillEvidence
from resume_matcher.text_processing import GENERAL_SECTION, Sections, snippet_around

# Characters that make a match part of a bigger token: "Java" in "JavaScript",
# "C" in "C++" / "C#", "SQL" in "MySQL".
_LEFT = r"(?<![A-Za-z0-9+#])"
_RIGHT = r"(?![A-Za-z0-9+#])"
# Extra guards for one/two-letter aliases such as "C", "R", "Go", "ML":
# reject "R&D", "C-level", "C." (initials) and "Go-to".
_SHORT_LEFT = r"(?<![A-Za-z0-9+#&'\-./])"
_SHORT_RIGHT = r"(?![A-Za-z0-9+#&'\-])(?!\.\w)"
# Text allowed between two skills that are offered as alternatives.
_ALT_SEPARATOR = re.compile(r",|/|or|,\s*or|and/or|,?\s*or\s+similar\s*,?")


def _alias_pattern(alias: str) -> str:
    """Regex for one alias; spaces and hyphens inside multi-word aliases are interchangeable."""
    tokens = [t for t in re.split(r"[\s\-]+", alias.strip()) if t]
    body = r"[\s\-]+".join(re.escape(t) for t in tokens)
    if len(alias.strip()) <= 2:
        return f"{_SHORT_LEFT}{body}{_SHORT_RIGHT}"
    return f"{_LEFT}{body}{_RIGHT}"


def _compile(aliases: list[str], flags: int) -> re.Pattern[str] | None:
    unique = sorted({a for a in aliases if a.strip()}, key=len, reverse=True)
    if not unique:
        return None
    return re.compile("|".join(f"(?:{_alias_pattern(a)})" for a in unique), flags)


@dataclass(frozen=True)
class _CompiledSkill:
    name: str
    category: str
    insensitive: re.Pattern[str] | None
    sensitive: re.Pattern[str] | None


@dataclass(frozen=True)
class SkillMention:
    skill: str
    start: int
    end: int
    surface: str


class SkillExtractor:
    """Finds taxonomy skills in text using alias regexes with longest-match-wins resolution."""

    def __init__(self, taxonomy: Taxonomy, matching: SkillMatchingConfig) -> None:
        self.taxonomy = taxonomy
        self.matching = matching
        self._skills = [
            _CompiledSkill(
                name=s.name,
                category=s.category,
                insensitive=_compile(s.aliases, re.IGNORECASE),
                sensitive=_compile(s.case_sensitive_aliases, 0),
            )
            for s in taxonomy.skills
        ]
        self._category_of = {s.name: s.category for s in taxonomy.skills}

    def category_of(self, skill: str) -> str:
        return self._category_of[skill]

    def category_label(self, skill: str) -> str:
        return self.taxonomy.categories[self._category_of[skill]].label

    def find_mentions(self, line: str) -> list[SkillMention]:
        """All non-overlapping skill mentions in a line. Longer matches win overlaps,
        so "GitHub Actions" is not also counted as "GitHub"."""
        candidates: list[SkillMention] = []
        for skill in self._skills:
            for pattern in (skill.insensitive, skill.sensitive):
                if pattern is None:
                    continue
                for m in pattern.finditer(line):
                    candidates.append(SkillMention(skill.name, m.start(), m.end(), m.group(0)))
        candidates.sort(key=lambda c: (-(c.end - c.start), c.start))
        taken: list[SkillMention] = []
        for cand in candidates:
            if all(cand.end <= t.start or cand.start >= t.end for t in taken):
                taken.append(cand)
        return sorted(taken, key=lambda c: c.start)

    def find_alternative_groups(self, clause: str) -> list[list[str]]:
        """Groups of skills offered as alternatives, e.g. "PyTorch or TensorFlow",
        "Tableau / Power BI", "PostgreSQL, MySQL or MongoDB"."""
        mentions = self.find_mentions(clause)
        groups: list[list[str]] = []

        def close(chain: list[str], has_or: bool) -> None:
            if not has_or:
                return
            # Only skills of the same kind as the last option are alternatives:
            # in "JavaScript, React or Vue" JavaScript stays a separate requirement.
            category = self.category_of(chain[-1])
            group = list(dict.fromkeys(s for s in chain if self.category_of(s) == category))
            if len(group) > 1:
                groups.append(group)

        chain: list[str] = []
        has_or = False
        for prev, cur in zip(mentions, mentions[1:], strict=False):
            between = clause[prev.end : cur.start].strip().lower()
            if _ALT_SEPARATOR.fullmatch(between):
                if not chain:
                    chain = [prev.skill]
                chain.append(cur.skill)
                has_or = has_or or between != ","
                continue
            if chain:
                close(chain, has_or)
            chain, has_or = [], False
        if chain:
            close(chain, has_or)
        return groups

    def find_skills(self, text: str) -> list[str]:
        """Unique skill names in order of first appearance."""
        seen: dict[str, None] = {}
        for line in text.split("\n"):
            for mention in self.find_mentions(line):
                seen.setdefault(mention.skill, None)
        return list(seen)

    def _is_demonstrated(self, section: str, line: str) -> bool:
        if section in self.matching.demonstrated_sections:
            return True
        if section == GENERAL_SECTION:
            return len(line.split()) >= self.matching.min_words_for_sentence_evidence
        return False

    def extract_from_sections(self, sections: Sections) -> list[ExtractedSkill]:
        """Extract skills with evidence (section + snippet) from a sectioned resume."""
        found: dict[str, dict] = {}
        for section, lines in sections.items():
            for line in lines:
                for mention in self.find_mentions(line):
                    entry = found.setdefault(
                        mention.skill,
                        {"mentions": 0, "sections": [], "demonstrated": False, "evidence": []},
                    )
                    entry["mentions"] += 1
                    if section not in entry["sections"]:
                        entry["sections"].append(section)
                    demonstrated = self._is_demonstrated(section, line)
                    entry["demonstrated"] = entry["demonstrated"] or demonstrated
                    snippet = snippet_around(line, mention.start, mention.end)
                    if all(e[1].snippet != snippet for e in entry["evidence"]):
                        entry["evidence"].append((demonstrated, SkillEvidence(section=section, snippet=snippet)))

        limit = self.matching.max_evidence_per_skill
        results: list[ExtractedSkill] = []
        for name, entry in found.items():
            # Show demonstrated evidence first, then listings.
            ordered = sorted(entry["evidence"], key=lambda e: not e[0])
            results.append(
                ExtractedSkill(
                    name=name,
                    category=self.category_of(name),
                    category_label=self.category_label(name),
                    mentions=entry["mentions"],
                    sections=entry["sections"],
                    demonstrated=entry["demonstrated"],
                    evidence=[e[1] for e in ordered[:limit]],
                )
            )
        results.sort(key=lambda s: (not s.demonstrated, -s.mentions, s.name))
        return results
