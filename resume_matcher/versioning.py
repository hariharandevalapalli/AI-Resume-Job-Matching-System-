"""Model versioning: every result carries the scoring version and config fingerprint."""

from __future__ import annotations

from resume_matcher import __version__
from resume_matcher.config import AppConfig
from resume_matcher.schemas import ModelInfo, ScoreBand


def model_info(config: AppConfig) -> ModelInfo:
    scoring = config.scoring
    matching = scoring.skill_matching
    return ModelInfo(
        model_name=scoring.model_name,
        model_version=scoring.version,
        taxonomy_version=config.taxonomy.version,
        config_fingerprint=config.fingerprint,
        app_version=__version__,
        skills_in_taxonomy=len(config.taxonomy.skills),
        weights=scoring.weights.model_dump(),
        skill_credits={
            "demonstrated": matching.demonstrated_credit,
            "listed_only": matching.listed_only_credit,
            "related": matching.related_skill_credit,
        },
        category_weights={
            config.taxonomy.categories[key].label if key in config.taxonomy.categories else key: value
            for key, value in matching.category_weights.items()
        },
        tfidf_saturation=scoring.tfidf.saturation,
        bands=[ScoreBand(min_score=b.min_score, label=b.label) for b in scoring.bands],
    )
