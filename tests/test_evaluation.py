from __future__ import annotations

import json
from pathlib import Path

import pytest

from resume_matcher.evaluation.metrics import (
    average_precision,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    spearman,
)
from resume_matcher.evaluation.runner import evaluate, load_evaluation_set, save_report


def test_ndcg_perfect_and_reversed() -> None:
    assert ndcg_at_k([3, 2, 1, 0], 4) == pytest.approx(1.0)
    assert ndcg_at_k([0, 1, 2, 3], 4) < 0.6
    assert ndcg_at_k([0, 0, 0], 3) == 0.0


def test_dcg_known_value() -> None:
    # (2^3-1)/log2(2) + (2^2-1)/log2(3) = 7 + 1.8928
    assert dcg_at_k([3, 2], 2) == pytest.approx(8.8928, abs=1e-3)


def test_binary_metrics() -> None:
    grades = [0, 3, 0, 2]
    assert precision_at_k(grades, 2, threshold=2) == 0.5
    assert reciprocal_rank(grades, threshold=2) == 0.5
    assert average_precision(grades, threshold=2) == pytest.approx((1 / 2 + 2 / 4) / 2)
    assert average_precision([0, 0], threshold=2) == 0.0
    assert reciprocal_rank([0, 1], threshold=2) == 0.0


def test_spearman() -> None:
    assert spearman([3, 2, 1], [3, 2, 1]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [3, 2, 1]) == 0.0


def test_dataset_loads() -> None:
    dataset = load_evaluation_set()
    assert len(dataset.queries) == 3
    assert len(dataset.resumes) == 9


def test_full_evaluation_quality_gate() -> None:
    report = evaluate(variants={"full_model": None})
    mean = report["variants"]["full_model"]["mean"]
    assert mean["ndcg@3"] >= 0.9
    assert mean["mrr"] == 1.0  # the best candidate is ranked first for every job


def test_evaluation_via_pdf_matches_text() -> None:
    text_report = evaluate(variants={"full_model": None})
    pdf_report = evaluate(variants={"full_model": None}, via_pdf=True)
    assert pdf_report["variants"]["full_model"]["mean"]["ndcg@3"] == pytest.approx(
        text_report["variants"]["full_model"]["mean"]["ndcg@3"], abs=0.05
    )


def test_save_report_writes_history_and_registry(tmp_path: Path) -> None:
    report = evaluate(variants={"full_model": None})
    path = save_report(report, tmp_path / "evaluation")
    assert path.exists()
    assert (tmp_path / "evaluation" / "history.csv").read_text(encoding="utf-8").count("\n") == 2
    registry = json.loads((tmp_path / "model_registry.json").read_text(encoding="utf-8"))
    key = f"{report['model_version']}+{report['config_fingerprint']}"
    assert registry["models"][key]["metrics"]["ndcg@3"] == report["variants"]["full_model"]["mean"]["ndcg@3"]
