from __future__ import annotations

from datetime import date

from resume_matcher.engine import MatchEngine
from resume_matcher.patterns import find_date_ranges, find_education_levels, redact_pii, total_months
from tests.conftest import STRONG_RESUME, TODAY


def test_date_ranges_and_overlap_merge() -> None:
    text = "Engineer  Jan 2020 - Dec 2021\nConsultant  Jun 2021 - Jun 2022\nIntern 06/2018 - 08/2018"
    intervals = find_date_ranges(text, TODAY)
    assert len(intervals) == 3
    # Jan 2020 - Jun 2022 merged (30 months) + Jun-Aug 2018 (3 months)
    assert total_months(intervals) == 33


def test_present_uses_reference_date() -> None:
    intervals = find_date_ranges("Engineer, Mar 2024 - Present", date(2026, 3, 15))
    assert total_months(intervals) == 25


def test_year_only_ranges_and_invalid_ranges() -> None:
    assert total_months(find_date_ranges("Analyst 2019 - 2021", TODAY)) == 24
    assert find_date_ranges("Engineer 2022 - 2019", TODAY) == []
    assert find_date_ranges("Engineer 2030 - Present", TODAY) == []


def test_profile_from_strong_resume(engine: MatchEngine) -> None:
    analyzed = engine.resume_analyzer.analyze(STRONG_RESUME, today=TODAY)
    profile = analyzed.profile
    assert profile.sections_found == ["summary", "experience", "skills", "education"]
    assert profile.years_experience == 5.9  # Jan 2021 - Nov 2026 inclusive (71 months)
    assert profile.years_experience_source == "employment dates"
    assert profile.claimed_years_experience == 4
    assert profile.education_level == "bachelor"
    assert profile.has_email and profile.has_phone
    assert profile.quantified_lines >= 1


def test_claimed_years_used_without_dates(engine: MatchEngine) -> None:
    text = "Summary\nBackend developer with 6+ years of experience in Python and Django.\n" * 2
    profile = engine.resume_analyzer.analyze(text, today=TODAY).profile
    assert profile.years_experience == 6
    assert profile.years_experience_source == "stated in resume"


def test_education_dates_not_counted_as_experience(engine: MatchEngine) -> None:
    text = "Education\nBSc Computer Science 2016 - 2020\nSkills\nPython, SQL and lots of other things here"
    profile = engine.resume_analyzer.analyze(text, today=TODAY).profile
    assert profile.years_experience is None


def test_preview_redacts_pii(engine: MatchEngine) -> None:
    preview = engine.resume_analyzer.analyze(STRONG_RESUME, today=TODAY).profile.preview
    assert "Jane Doe" not in preview and "[name]" in preview
    assert "jane.doe@example.com" not in preview and "[email]" in preview
    assert "555 123 4567" not in preview and "[phone]" in preview


def test_redact_pii_keeps_date_ranges() -> None:
    assert redact_pii("Worked 2019 - 2021, see github.com/someone") == "Worked 2019 - 2021, see [link]"


def test_education_level_patterns() -> None:
    assert "master" in find_education_levels("M.Sc. in Data Science")
    assert "phd" in find_education_levels("Ph.D. candidate")
    assert "bachelor" in find_education_levels("B.Tech, Computer Engineering")
    assert find_education_levels("Certified Scrum Master, MS Excel expert") == []
