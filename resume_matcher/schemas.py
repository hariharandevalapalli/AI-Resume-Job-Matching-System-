"""Pydantic models shared by the engine and the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Requirement = Literal["required", "preferred"]
SkillStatus = Literal["matched", "partial", "missing"]
EvidenceStrength = Literal["demonstrated", "listed", "related", "none"]
Priority = Literal["high", "medium", "low"]

DISCLAIMER = (
    "This score is an indicative, explainable estimate of how well a resume's text matches a job "
    "description. It is not a hiring decision and should always be reviewed by a person."
)


class Resource(BaseModel):
    title: str
    url: str


class SkillEvidence(BaseModel):
    section: str = Field(description="Resume section where the skill was found.")
    snippet: str = Field(description="Text around the mention (PII-free excerpt).")


class ExtractedSkill(BaseModel):
    """A skill detected in a resume."""

    name: str
    category: str
    category_label: str
    mentions: int
    sections: list[str]
    demonstrated: bool = Field(description="Found in experience/projects or a descriptive sentence.")
    evidence: list[SkillEvidence]


class SkillMatch(BaseModel):
    """How a single job-description skill is covered by the resume."""

    name: str
    category: str
    category_label: str
    requirement: Requirement
    status: SkillStatus
    evidence_strength: EvidenceStrength
    credit: float = Field(ge=0, le=1, description="Credit awarded (0-1) before category weighting.")
    weight: float = Field(ge=0, description="Category weight used when averaging coverage.")
    matched_via: str | None = Field(
        default=None,
        description="Resume skill used instead: an accepted alternative (status matched) or a related skill "
        "(status partial).",
    )
    evidence: list[SkillEvidence] = Field(default_factory=list)


class JobAnalysis(BaseModel):
    required_skills: list[str]
    preferred_skills: list[str]
    alternatives: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Skill -> other skills accepted instead (from phrases like 'PyTorch or TensorFlow').",
    )
    min_years_experience: float | None
    education_requirement: str | None
    preferred_education: str | None
    key_terms: list[str]
    word_count: int


class ResumeProfile(BaseModel):
    word_count: int
    page_count: int | None = None
    sections_found: list[str]
    skills: list[ExtractedSkill]
    years_experience: float | None
    years_experience_source: str
    claimed_years_experience: float | None
    education_level: str | None
    has_email: bool
    has_phone: bool
    quantified_lines: int
    preview: str = Field(description="Short PII-redacted preview of the extracted text.")


class ScoreComponent(BaseModel):
    key: str
    label: str
    included: bool = Field(description="False when the job description gives nothing to compare against.")
    raw_score: float = Field(ge=0, le=1)
    configured_weight: float
    effective_weight: float = Field(description="Weight after re-normalising over included components.")
    contribution: float = Field(description="Points contributed to the 0-100 overall score.")
    explanation: str


class TermContribution(BaseModel):
    term: str
    share: float = Field(description="Share of the cosine similarity contributed by this term (0-1).")


class TextSimilarity(BaseModel):
    cosine: float
    score: float
    matched_terms: list[TermContribution]
    missing_terms: list[str]


class Recommendation(BaseModel):
    priority: Priority
    category: str
    title: str
    detail: str
    action: str


class RoadmapStep(BaseModel):
    phase: int
    phase_title: str
    skill: str
    category_label: str
    reason: str
    estimated_weeks: float
    practice_project: str
    resources: list[Resource]


class ScoreBand(BaseModel):
    min_score: float
    label: str


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    taxonomy_version: str
    config_fingerprint: str
    app_version: str
    skills_in_taxonomy: int
    weights: dict[str, float]
    skill_credits: dict[str, float] = Field(description="Credit for demonstrated, listed-only and related skills.")
    category_weights: dict[str, float]
    tfidf_saturation: float
    bands: list[ScoreBand]


class MatchResult(BaseModel):
    resume_name: str
    overall_score: float = Field(ge=0, le=100)
    band: str
    summary: str
    components: list[ScoreComponent]
    required_skills: list[SkillMatch]
    preferred_skills: list[SkillMatch]
    additional_skills: list[str] = Field(description="Resume skills not mentioned in the job description.")
    text_similarity: TextSimilarity
    recommendations: list[Recommendation]
    roadmap: list[RoadmapStep]
    resume_profile: ResumeProfile
    job_analysis: JobAnalysis
    model_info: ModelInfo
    disclaimer: str = DISCLAIMER


class RankedResume(BaseModel):
    rank: int
    resume_name: str
    overall_score: float
    band: str
    component_scores: dict[str, float]
    required_matched: int
    required_total: int
    missing_required: list[str]
    strengths: list[str]


class FailedDocument(BaseModel):
    resume_name: str
    error: str


class ComparisonResult(BaseModel):
    job_analysis: JobAnalysis
    rankings: list[RankedResume]
    results: list[MatchResult]
    skill_matrix: dict[str, dict[str, SkillStatus]] = Field(
        description="resume name -> job skill -> status, for required and preferred skills."
    )
    failed: list[FailedDocument]
    model_info: ModelInfo
    disclaimer: str = DISCLAIMER


# ----------------------------------------------------------------------- API requests


class JobDescriptionRequest(BaseModel):
    job_description: str = Field(min_length=1, examples=["We are hiring a Python developer with 3+ years ..."])


class TextMatchRequest(BaseModel):
    resume_text: str = Field(min_length=1)
    job_description: str = Field(min_length=1)
    resume_name: str = Field(default="resume", max_length=120)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    app_version: str
    model_version: str
    config_fingerprint: str
