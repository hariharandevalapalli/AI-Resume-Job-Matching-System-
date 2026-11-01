from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from resume_matcher.config import DEFAULT_CONFIG_DIR, AppConfig, load_config
from resume_matcher.exceptions import ConfigurationError


def _copy_config(tmp_path: Path) -> Path:
    target = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, target)
    return target


def test_default_config_loads(config: AppConfig) -> None:
    assert config.scoring.version
    assert len(config.taxonomy.skills) > 80
    assert config.scoring.bands[-1].min_score == 0


def test_fingerprint_is_stable_and_sensitive(tmp_path: Path, config: AppConfig) -> None:
    assert load_config().fingerprint == config.fingerprint
    directory = _copy_config(tmp_path)
    scoring_path = directory / "scoring.yaml"
    data = yaml.safe_load(scoring_path.read_text(encoding="utf-8"))
    data["weights"]["education"] = 0.2
    scoring_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert load_config(directory).fingerprint != config.fingerprint


def test_all_zero_weights_rejected(tmp_path: Path) -> None:
    directory = _copy_config(tmp_path)
    scoring_path = directory / "scoring.yaml"
    data = yaml.safe_load(scoring_path.read_text(encoding="utf-8"))
    data["weights"] = {k: 0 for k in data["weights"]}
    scoring_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="weight"):
        load_config(directory)


def test_unknown_related_skill_rejected(tmp_path: Path) -> None:
    directory = _copy_config(tmp_path)
    taxonomy_path = directory / "skills_taxonomy.yaml"
    data = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))
    data["skills"][0]["related"] = ["Definitely Not A Skill"]
    taxonomy_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown related"):
        load_config(directory)


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        load_config(tmp_path)


def test_band_lookup(config: AppConfig) -> None:
    assert config.scoring.band_for(95) == "Strong match"
    assert config.scoring.band_for(0) == "Low match"


def test_related_map_is_symmetric(config: AppConfig) -> None:
    related = config.taxonomy.related_map()
    for skill, others in related.items():
        for other in others:
            assert skill in related[other]
