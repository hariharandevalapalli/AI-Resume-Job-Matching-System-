from __future__ import annotations

import pytest

from resume_matcher.config import AppConfig
from resume_matcher.skills import SkillExtractor
from resume_matcher.text_processing import SectionDetector


@pytest.fixture(scope="module")
def extractor(config: AppConfig) -> SkillExtractor:
    return SkillExtractor(config.taxonomy, config.scoring.skill_matching)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Deployed on k8s with Helm", "Kubernetes"),
        ("Models built with sklearn", "scikit-learn"),
        ("Used scikit learn daily", "scikit-learn"),
        ("Strong in machine-learning", "Machine Learning"),
        ("Wrote services in Golang", "Go"),
        ("APIs with Node.js", "Node.js"),
        ("Experience with C# and .NET", "C#"),
        ("CI/CD pipelines", "CI/CD"),
    ],
)
def test_aliases(extractor: SkillExtractor, text: str, expected: str) -> None:
    assert expected in extractor.find_skills(text)


def test_technology_aware_boundaries(extractor: SkillExtractor) -> None:
    skills = extractor.find_skills("JavaScript and C++ developer using MySQL")
    assert {"JavaScript", "C++", "MySQL"} <= set(skills)
    assert "Java" not in skills
    assert "C" not in skills
    assert "SQL" not in skills


def test_ambiguous_short_words_need_exact_case(extractor: SkillExtractor) -> None:
    skills = extractor.find_skills("I will go to the R&D lab and react quickly to excel in my role")
    assert not {"Go", "R", "React", "Excel"} & set(skills)
    skills = extractor.find_skills("Built dashboards in R and services in Go with React frontends")
    assert {"R", "Go", "React"} <= set(skills)


def test_longest_match_wins(extractor: SkillExtractor) -> None:
    skills = extractor.find_skills("Pipelines on GitHub Actions; styled with Tailwind CSS; mobile in React Native")
    assert "GitHub Actions" in skills and "GitHub" not in skills
    assert "Tailwind CSS" in skills and "CSS" not in skills
    assert "React Native" in skills and "React" not in skills


def test_case_insensitive_matching(extractor: SkillExtractor) -> None:
    assert extractor.find_skills("PYTHON, docker, PostGreSQL") == ["Python", "Docker", "PostgreSQL"]


def test_no_skills_in_plain_text(extractor: SkillExtractor) -> None:
    assert extractor.find_skills("I enjoy hiking and cooking with friends.") == []


def test_alternative_groups(extractor: SkillExtractor) -> None:
    assert extractor.find_alternative_groups("Experience with PyTorch or TensorFlow") == [["PyTorch", "TensorFlow"]]
    assert extractor.find_alternative_groups("Tableau / Power BI dashboards") == [["Tableau", "Power BI"]]
    # JavaScript is a different kind of skill from the frameworks, so it is not optional.
    assert extractor.find_alternative_groups("JavaScript, React or Vue") == [["React", "Vue.js"]]
    assert extractor.find_alternative_groups("Python, SQL and Docker") == []


def test_evidence_strength_by_section(config: AppConfig, extractor: SkillExtractor) -> None:
    detector = SectionDetector(config.scoring.resume_sections)
    sections = detector.split("Experience\nBuilt ETL jobs in Python on AWS\nSkills\nPython, Docker, AWS\n")
    skills = {s.name: s for s in extractor.extract_from_sections(sections)}
    assert skills["Python"].demonstrated is True
    assert skills["Docker"].demonstrated is False
    assert skills["Python"].mentions == 2
    assert skills["Python"].evidence[0].section == "experience"
    assert skills["Docker"].evidence[0].section == "skills"
