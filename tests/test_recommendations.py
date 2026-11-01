from __future__ import annotations

from resume_matcher.engine import MatchEngine
from tests.conftest import JD_TEXT, STRONG_RESUME, TODAY, WEAK_RESUME


def test_missing_required_skills_are_high_priority(engine: MatchEngine) -> None:
    result = engine.match_text(WEAK_RESUME, JD_TEXT, today=TODAY)
    first = result.recommendations[0]
    assert first.priority == "high"
    assert first.title == "Address missing required skills"
    assert "Python" in first.detail


def test_recommendations_sorted_and_capped(engine: MatchEngine) -> None:
    result = engine.match_text(WEAK_RESUME, JD_TEXT, today=TODAY)
    order = {"high": 0, "medium": 1, "low": 2}
    priorities = [order[r.priority] for r in result.recommendations]
    assert priorities == sorted(priorities)
    assert len(result.recommendations) <= engine.config.scoring.recommendations.max_items


def test_listed_only_skills_get_evidence_advice(engine: MatchEngine) -> None:
    result = engine.match_text(STRONG_RESUME, "Requirements\n- SQL and Git are essential for this role", today=TODAY)
    titles = [r.title for r in result.recommendations]
    assert "Back up listed skills with evidence" in titles


def test_roadmap_phases(engine: MatchEngine) -> None:
    result = engine.match_text(WEAK_RESUME, JD_TEXT, today=TODAY)
    phases = [s.phase for s in result.roadmap]
    assert phases == sorted(phases)
    phase1 = [s.skill for s in result.roadmap if s.phase == 1]
    assert "Python" in phase1 and "Docker" in phase1
    phase3 = [s.skill for s in result.roadmap if s.phase == 3]
    assert "NLP" in phase3
    python_step = next(s for s in result.roadmap if s.skill == "Python")
    assert python_step.resources and python_step.resources[0].url.startswith("https://")
    assert python_step.estimated_weeks > 0


def test_strong_resume_has_no_gap_roadmap(engine: MatchEngine) -> None:
    result = engine.match_text(STRONG_RESUME, JD_TEXT, today=TODAY)
    assert all(s.phase != 1 for s in result.roadmap)


def test_missing_dates_recommendation(engine: MatchEngine) -> None:
    resume = "Experience\nBuilt machine learning models with Python and scikit-learn for retail clients\n" * 3
    result = engine.match_text(resume, JD_TEXT, today=TODAY)
    assert "Add dates to your experience" in [r.title for r in result.recommendations]
