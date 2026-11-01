from __future__ import annotations

from resume_matcher.engine import MatchEngine
from tests.conftest import JD_TEXT


def test_required_and_preferred_sections(engine: MatchEngine) -> None:
    job = engine.analyze_job(JD_TEXT)
    assert {"Python", "scikit-learn", "PyTorch", "TensorFlow", "Docker", "AWS"} <= set(job.required_skills)
    assert {"MLflow", "Airflow", "NLP"} <= set(job.preferred_skills)
    assert not set(job.required_skills) & set(job.preferred_skills)


def test_benefits_section_ignored(engine: MatchEngine) -> None:
    job = engine.analyze_job(JD_TEXT)
    assert "Kubernetes" not in job.required_skills + job.preferred_skills


def test_inline_preferred_cue_overrides_section(engine: MatchEngine) -> None:
    jd = "Requirements\n- Python and SQL\n- Docker experience is a plus\n- Familiarity with AWS"
    job = engine.analyze_job(jd)
    assert job.required_skills == ["Python", "SQL"]
    assert job.preferred_skills == ["Docker", "AWS"]


def test_required_cue_inside_preferred_section(engine: MatchEngine) -> None:
    jd = "Nice to have\n- Kafka experience\n- Docker is required for all engineers on this team"
    job = engine.analyze_job(jd)
    assert job.required_skills == ["Docker"]
    assert job.preferred_skills == ["Kafka"]


def test_years_and_education(engine: MatchEngine) -> None:
    job = engine.analyze_job(JD_TEXT)
    assert job.min_years_experience == 3
    assert job.education_requirement == "bachelor"


def test_years_range_word_numbers_and_preferred_degree(engine: MatchEngine) -> None:
    job = engine.analyze_job(
        "Requirements\n- 2-4 years of professional experience with Java\n- Master's degree preferred"
    )
    assert job.min_years_experience == 2
    assert job.education_requirement is None
    assert job.preferred_education == "master"
    job = engine.analyze_job("We need someone with five years of industry experience in Python development work.")
    assert job.min_years_experience == 5


def test_minimum_degree_is_lowest_listed(engine: MatchEngine) -> None:
    job = engine.analyze_job("Requirements\n- Bachelor's or Master's degree in Computer Science\n- Python skills")
    assert job.education_requirement == "bachelor"


def test_degree_fields_are_not_skills(engine: MatchEngine) -> None:
    job = engine.analyze_job(
        "Requirements\n- Bachelor's degree in Statistics or Computer Science\n- Python for analysis"
    )
    assert "Statistics" not in job.required_skills
    assert job.required_skills == ["Python"]


def test_alternatives_recorded(engine: MatchEngine) -> None:
    job = engine.analyze_job(JD_TEXT)
    assert job.alternatives["PyTorch"] == ["TensorFlow"]
    assert job.alternatives["MLflow"] == ["Airflow"]


def test_relevant_text_drops_ignored_sections(engine: MatchEngine) -> None:
    text = engine.jd_analyzer.relevant_text(JD_TEXT)
    assert "Free lunch" not in text
    assert "scikit-learn" in text
