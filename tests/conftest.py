from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from resume_matcher.config import AppConfig, load_config
from resume_matcher.engine import MatchEngine
from resume_matcher.sample_pdf import render_text_pdf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TODAY = date(2026, 11, 1)

JD_TEXT = """Machine Learning Engineer

Requirements:
- 3+ years of experience building machine learning models in Python
- Strong knowledge of scikit-learn and PyTorch or TensorFlow
- Experience deploying models with Docker on AWS
- Bachelor's degree in Computer Science or a related field

Nice to have:
- MLflow or Airflow
- Experience with NLP is a plus

Benefits:
- Free lunch and a Kubernetes-themed office party
"""

STRONG_RESUME = """Jane Doe
jane.doe@example.com | +1 555 123 4567

Summary
Machine learning engineer with 4 years of experience.

Experience
ML Engineer, Acme Corp    Jan 2021 - Present
- Built churn prediction models with scikit-learn and PyTorch, improving retention by 12%
- Deployed models as REST APIs using Docker and FastAPI on AWS
- Tracked experiments with MLflow and built NLP text classifiers

Skills
Python, SQL, Git

Education
B.Tech in Computer Science, 2020
"""

WEAK_RESUME = """John Roe
Summary
Marketing specialist focused on social media campaigns and brand storytelling.

Experience
Marketing Specialist, Brandly    Mar 2022 - Present
- Ran social media campaigns that grew followers by 35%
- Wrote newsletters and blog posts for brand awareness

Education
Bachelor of Arts in Communication, 2021
"""


@pytest.fixture(scope="session")
def config() -> AppConfig:
    return load_config()


@pytest.fixture(scope="session")
def engine(config: AppConfig) -> MatchEngine:
    return MatchEngine(config)


@pytest.fixture(scope="session")
def strong_pdf() -> bytes:
    return render_text_pdf(STRONG_RESUME)


@pytest.fixture(scope="session")
def weak_pdf() -> bytes:
    return render_text_pdf(WEAK_RESUME)
