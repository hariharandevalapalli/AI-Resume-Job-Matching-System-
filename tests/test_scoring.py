from __future__ import annotations

import pytest

from resume_matcher.config import AppConfig, Weights
from resume_matcher.engine import MatchEngine
from resume_matcher.exceptions import InvalidInputError
from resume_matcher.scoring import compute_text_similarity
from tests.conftest import JD_TEXT, STRONG_RESUME, TODAY, WEAK_RESUME


def test_strong_resume_beats_weak_resume(engine: MatchEngine) -> None:
    strong = engine.match_text(STRONG_RESUME, JD_TEXT, today=TODAY)
    weak = engine.match_text(WEAK_RESUME, JD_TEXT, today=TODAY)
    assert strong.overall_score >= 75
    assert weak.overall_score < 40
    assert strong.band == "Strong match"
    assert weak.band == "Low match"


def test_contributions_sum_to_overall(engine: MatchEngine) -> None:
    result = engine.match_text(STRONG_RESUME, JD_TEXT, today=TODAY)
    total = sum(c.contribution for c in result.components)
    assert total == pytest.approx(result.overall_score, abs=0.2)
    included = [c for c in result.components if c.included]
    assert sum(c.effective_weight for c in included) == pytest.approx(1.0, abs=1e-3)


def test_components_excluded_and_renormalised(engine: MatchEngine) -> None:
    jd = "Requirements\n- Python and SQL for building data tools\n- Docker for deployment of services"
    result = engine.match_text(STRONG_RESUME, jd, today=TODAY)
    by_key = {c.key: c for c in result.components}
    assert not by_key["preferred_skills"].included
    assert not by_key["experience"].included
    assert not by_key["education"].included
    assert by_key["required_skills"].effective_weight == pytest.approx(0.45 / 0.70, abs=1e-3)


def test_alternative_satisfies_requirement_once(engine: MatchEngine) -> None:
    result = engine.match_text(STRONG_RESUME, JD_TEXT, today=TODAY)
    req = {m.name: m for m in result.required_skills}
    assert req["PyTorch"].status == "matched" and req["PyTorch"].matched_via is None
    assert req["TensorFlow"].status == "matched" and req["TensorFlow"].matched_via == "PyTorch"
    assert req["PyTorch"].weight == pytest.approx(0.5)
    assert req["TensorFlow"].weight == pytest.approx(0.5)


def test_related_skill_gives_partial_credit(engine: MatchEngine, config: AppConfig) -> None:
    jd = "Requirements\n- Experience building dashboards in Power BI for business users"
    resume = "Experience\nBuilt sales dashboards in Tableau used by 40 managers across regions\n" * 2
    result = engine.match_text(resume, jd, today=TODAY)
    match = next(m for m in result.required_skills if m.name == "Power BI")
    assert match.status == "partial"
    assert match.matched_via == "Tableau"
    assert match.credit == config.scoring.skill_matching.related_skill_credit


def test_listed_only_skill_earns_less(engine: MatchEngine, config: AppConfig) -> None:
    result = engine.match_text(STRONG_RESUME, "Requirements\n- SQL and Python are both essential skills", today=TODAY)
    req = {m.name: m for m in result.required_skills}
    assert req["SQL"].evidence_strength == "listed"
    assert req["SQL"].credit == config.scoring.skill_matching.listed_only_credit
    assert req["Python"].evidence_strength == "listed"


def test_experience_and_education_scores(engine: MatchEngine) -> None:
    jd = "Requirements\n- 10+ years of experience with Python\n- PhD in Computer Science required"
    result = engine.match_text(STRONG_RESUME, jd, today=TODAY)
    by_key = {c.key: c for c in result.components}
    assert by_key["experience"].raw_score == pytest.approx(0.59, abs=0.01)
    assert by_key["education"].raw_score == 0.0  # bachelor is two levels below PhD


def test_text_similarity_terms(config: AppConfig) -> None:
    sim = compute_text_similarity(
        "Python developer building data pipelines", "Python data pipelines expert", config.scoring
    )
    assert 0 < sim.cosine <= 1
    assert sim.matched_terms[0].term in {"python", "data", "pipelines", "data pipelines"}
    assert sum(t.share for t in sim.matched_terms) == pytest.approx(1.0, abs=0.01)
    assert "developer" in sim.missing_terms


def test_text_similarity_handles_empty_vocabulary(config: AppConfig) -> None:
    sim = compute_text_similarity("the and of", "is a the", config.scoring)
    assert sim.cosine == 0 and sim.score == 0


def test_custom_weights_change_score(config: AppConfig) -> None:
    custom = config.model_copy(deep=True)
    custom.scoring.weights = Weights(
        required_skills=0, preferred_skills=0, text_similarity=1, experience=0, education=0
    )
    result = MatchEngine(custom).match_text(STRONG_RESUME, JD_TEXT, today=TODAY)
    assert result.overall_score == pytest.approx(result.text_similarity.score * 100, abs=0.1)


@pytest.mark.parametrize(
    ("resume", "jd", "message"),
    [
        ("too short", JD_TEXT, "too short"),
        (STRONG_RESUME, "short jd", "too short"),
        (STRONG_RESUME, "x" * 20001, "too long"),
        ("   ", JD_TEXT, "too short"),
    ],
)
def test_input_validation(engine: MatchEngine, resume: str, jd: str, message: str) -> None:
    with pytest.raises(InvalidInputError, match=message):
        engine.match_text(resume, jd)


def test_jd_without_known_skills_still_scores(engine: MatchEngine) -> None:
    jd = "We want a friendly person to help customers in our bakery and keep the shop tidy every day."
    result = engine.match_text(STRONG_RESUME, jd, today=TODAY)
    assert 0 <= result.overall_score <= 100
    assert result.required_skills == []
    assert [c.key for c in result.components if c.included] == ["text_similarity"]


def test_summary_and_disclaimer(engine: MatchEngine) -> None:
    result = engine.match_text(WEAK_RESUME, JD_TEXT, today=TODAY)
    assert "Missing required skills" in result.summary
    assert "not a hiring decision" in result.disclaimer
    assert result.model_info.config_fingerprint == engine.config.fingerprint
