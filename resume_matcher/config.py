"""Load and validate the YAML configuration (scoring rules + skill taxonomy)."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from resume_matcher.exceptions import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_DIR_ENV = "RESUME_MATCHER_CONFIG_DIR"


# --------------------------------------------------------------------------- scoring


class Weights(BaseModel):
    required_skills: float = Field(ge=0)
    preferred_skills: float = Field(ge=0)
    text_similarity: float = Field(ge=0)
    experience: float = Field(ge=0)
    education: float = Field(ge=0)

    @model_validator(mode="after")
    def _non_zero(self) -> Weights:
        if sum(self.model_dump().values()) <= 0:
            raise ValueError("at least one scoring weight must be positive")
        return self


class SkillMatchingConfig(BaseModel):
    demonstrated_credit: float = Field(ge=0, le=1)
    listed_only_credit: float = Field(ge=0, le=1)
    related_skill_credit: float = Field(ge=0, le=1)
    category_weights: dict[str, float] = Field(default_factory=dict)
    demonstrated_sections: list[str]
    min_words_for_sentence_evidence: int = Field(ge=1)
    max_evidence_per_skill: int = Field(ge=1)


class TfidfConfig(BaseModel):
    ngram_range: tuple[int, int] = (1, 2)
    sublinear_tf: bool = True
    max_features: int = Field(default=5000, ge=100)
    saturation: float = Field(gt=0, le=1)
    top_terms: int = Field(default=12, ge=1)
    extra_stop_words: list[str] = Field(default_factory=list)

    @field_validator("ngram_range")
    @classmethod
    def _valid_range(cls, value: tuple[int, int]) -> tuple[int, int]:
        low, high = value
        if low < 1 or high < low:
            raise ValueError("ngram_range must be [min, max] with 1 <= min <= max")
        return value


class ExperienceConfig(BaseModel):
    max_reasonable_years: float = Field(gt=0)


class EducationConfig(BaseModel):
    levels: dict[str, float]
    one_level_below_credit: float = Field(ge=0, le=1)


class Band(BaseModel):
    min_score: float = Field(ge=0, le=100)
    label: str


class JDParsingConfig(BaseModel):
    required_headers: list[str]
    preferred_headers: list[str]
    ignored_headers: list[str]
    preferred_cues: list[str]
    required_cues: list[str]


class LimitsConfig(BaseModel):
    max_file_size_mb: float = Field(gt=0)
    max_pdf_pages: int = Field(ge=1)
    max_resumes_compare: int = Field(ge=2)
    min_resume_chars: int = Field(ge=1)
    min_jd_chars: int = Field(ge=1)
    max_jd_chars: int = Field(ge=100)
    max_resume_chars: int = Field(ge=1000)

    @property
    def max_file_size_bytes(self) -> int:
        return int(self.max_file_size_mb * 1024 * 1024)


class PrivacyConfig(BaseModel):
    store_resumes: bool = False
    redact_pii_in_preview: bool = True
    preview_chars: int = Field(default=600, ge=0)


class RecommendationConfig(BaseModel):
    max_items: int = Field(ge=1)
    low_similarity_threshold: float = Field(ge=0, le=1)
    min_resume_words: int = Field(ge=0)
    max_resume_words: int = Field(ge=1)
    min_quantified_lines: int = Field(ge=0)


class ScoringConfig(BaseModel):
    version: str
    model_name: str
    weights: Weights
    skill_matching: SkillMatchingConfig
    tfidf: TfidfConfig
    experience: ExperienceConfig
    education: EducationConfig
    bands: list[Band]
    resume_sections: dict[str, list[str]]
    jd_parsing: JDParsingConfig
    limits: LimitsConfig
    privacy: PrivacyConfig
    recommendations: RecommendationConfig

    @field_validator("bands")
    @classmethod
    def _sorted_bands(cls, bands: list[Band]) -> list[Band]:
        if not bands:
            raise ValueError("at least one score band is required")
        ordered = sorted(bands, key=lambda b: b.min_score, reverse=True)
        if ordered[-1].min_score != 0:
            raise ValueError("the lowest band must start at 0")
        return ordered

    def band_for(self, score: float) -> str:
        for band in self.bands:
            if score >= band.min_score:
                return band.label
        return self.bands[-1].label


# --------------------------------------------------------------------------- taxonomy


class Resource(BaseModel):
    title: str
    url: str


class SkillCategory(BaseModel):
    label: str
    learning_weeks: float = Field(gt=0)
    practice_project: str


class SkillDefinition(BaseModel):
    name: str
    category: str
    aliases: list[str] = Field(default_factory=list)
    case_sensitive_aliases: list[str] = Field(default_factory=list)
    related: list[str] = Field(default_factory=list)
    resources: list[Resource] = Field(default_factory=list)
    learning_weeks: float | None = None

    @model_validator(mode="after")
    def _has_alias(self) -> SkillDefinition:
        if not self.aliases and not self.case_sensitive_aliases:
            raise ValueError(f"skill '{self.name}' needs at least one alias")
        return self


class Taxonomy(BaseModel):
    version: str
    categories: dict[str, SkillCategory]
    skills: list[SkillDefinition]

    @model_validator(mode="after")
    def _consistent(self) -> Taxonomy:
        names = [s.name for s in self.skills]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"duplicate skill names: {sorted(duplicates)}")
        known = set(names)
        for skill in self.skills:
            if skill.category not in self.categories:
                raise ValueError(f"skill '{skill.name}' uses unknown category '{skill.category}'")
            unknown = [r for r in skill.related if r not in known]
            if unknown:
                raise ValueError(f"skill '{skill.name}' has unknown related skills {unknown}")
        return self

    def by_name(self) -> dict[str, SkillDefinition]:
        return {s.name: s for s in self.skills}

    def related_map(self) -> dict[str, set[str]]:
        """Symmetric relation: if A lists B as related, B is related to A too."""
        relations: dict[str, set[str]] = {s.name: set() for s in self.skills}
        for skill in self.skills:
            for other in skill.related:
                relations[skill.name].add(other)
                relations[other].add(skill.name)
        return relations


# --------------------------------------------------------------------------- loading


class AppConfig(BaseModel):
    scoring: ScoringConfig
    taxonomy: Taxonomy

    @property
    def fingerprint(self) -> str:
        """Short, stable hash of the full configuration, used for model versioning."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"configuration file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"{path} must contain a mapping at the top level")
    return data


def resolve_config_dir(config_dir: str | Path | None = None) -> Path:
    if config_dir is not None:
        return Path(config_dir)
    env_value = os.getenv(CONFIG_DIR_ENV)
    return Path(env_value) if env_value else DEFAULT_CONFIG_DIR


def load_config(config_dir: str | Path | None = None) -> AppConfig:
    """Load scoring.yaml and skills_taxonomy.yaml from the config directory."""
    directory = resolve_config_dir(config_dir)
    try:
        scoring = ScoringConfig.model_validate(_read_yaml(directory / "scoring.yaml"))
        taxonomy = Taxonomy.model_validate(_read_yaml(directory / "skills_taxonomy.yaml"))
    except ValidationError as exc:
        raise ConfigurationError(f"invalid configuration in {directory}:\n{exc}") from exc
    return AppConfig(scoring=scoring, taxonomy=taxonomy)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Process-wide cached configuration."""
    return load_config()
