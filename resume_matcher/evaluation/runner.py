"""Run the labelled evaluation set, compare against baselines and record results.

Outputs (under reports/evaluation/ by default):
  * run_<timestamp>_<fingerprint>.json  full per-query results
  * history.csv                         one row per run and variant
  * ../model_registry.json              latest metrics per model version + fingerprint
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from resume_matcher.config import PROJECT_ROOT, AppConfig, Weights, load_config
from resume_matcher.engine import MatchEngine
from resume_matcher.evaluation.metrics import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    spearman,
)
from resume_matcher.logging_config import get_logger
from resume_matcher.pdf_parser import extract_text
from resume_matcher.sample_pdf import render_text_pdf

logger = get_logger(__name__)

DEFAULT_LABELS = PROJECT_ROOT / "data" / "evaluation" / "labels.yaml"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "evaluation"

# Baselines show what each part of the model contributes.
VARIANTS: dict[str, dict[str, float] | None] = {
    "full_model": None,  # weights from config/scoring.yaml
    "tfidf_only": {"required_skills": 0, "preferred_skills": 0, "text_similarity": 1, "experience": 0, "education": 0},
    "skills_only": {
        "required_skills": 0.75,
        "preferred_skills": 0.25,
        "text_similarity": 0,
        "experience": 0,
        "education": 0,
    },
}
METRIC_KEYS = ["ndcg@3", "ndcg@5", "map", "mrr", "precision@3", "spearman"]


@dataclass
class EvaluationSet:
    relevant_threshold: float
    reference_date: date
    resumes: dict[str, str]
    queries: dict[str, tuple[str, dict[str, float]]]  # query -> (jd_text, {resume: grade})


def load_evaluation_set(labels_path: Path = DEFAULT_LABELS) -> EvaluationSet:
    spec = yaml.safe_load(labels_path.read_text(encoding="utf-8"))
    # Relative directories in labels.yaml are resolved against the project root.
    resume_dir = PROJECT_ROOT / spec["resume_dir"]
    job_dir = PROJECT_ROOT / spec["job_dir"]
    resumes: dict[str, str] = {}
    queries: dict[str, tuple[str, dict[str, float]]] = {}
    for name, query in spec["queries"].items():
        labels = {k: float(v) for k, v in query["labels"].items()}
        for resume_id in labels:
            if resume_id not in resumes:
                resumes[resume_id] = (resume_dir / f"{resume_id}.txt").read_text(encoding="utf-8")
        queries[name] = ((job_dir / query["job_file"]).read_text(encoding="utf-8"), labels)
    return EvaluationSet(
        relevant_threshold=float(spec.get("relevant_threshold", 2)),
        reference_date=date.fromisoformat(str(spec.get("reference_date", date.today().isoformat()))),
        resumes=resumes,
        queries=queries,
    )


def _variant_config(base: AppConfig, weights: dict[str, float] | None) -> AppConfig:
    if weights is None:
        return base
    config = base.model_copy(deep=True)
    config.scoring.weights = Weights(**weights)
    return config


def _query_metrics(scores: dict[str, float], labels: dict[str, float], threshold: float) -> dict[str, Any]:
    ranking = sorted(scores, key=lambda r: (-scores[r], r))
    grades = [labels[r] for r in ranking]
    return {
        "ranking": [{"resume": r, "score": scores[r], "grade": labels[r]} for r in ranking],
        "ndcg@3": ndcg_at_k(grades, 3),
        "ndcg@5": ndcg_at_k(grades, 5),
        "map": average_precision(grades, threshold),
        "mrr": reciprocal_rank(grades, threshold),
        "precision@3": precision_at_k(grades, 3, threshold),
        "spearman": spearman([scores[r] for r in ranking], grades),
    }


def evaluate(
    config: AppConfig | None = None,
    labels_path: Path = DEFAULT_LABELS,
    via_pdf: bool = False,
    variants: dict[str, dict[str, float] | None] | None = None,
) -> dict[str, Any]:
    """Score every (job, resume) pair for each variant and compute ranking metrics."""
    base = config or load_config()
    dataset = load_evaluation_set(labels_path)
    texts = dict(dataset.resumes)
    if via_pdf:
        # Exercise the full PDF path: render each resume to PDF and extract it again.
        texts = {
            rid: extract_text(render_text_pdf(text), f"{rid}.pdf", base.scoring.limits).text
            for rid, text in texts.items()
        }

    report: dict[str, Any] = {"variants": {}}
    for variant, weights in (variants or VARIANTS).items():
        engine = MatchEngine(_variant_config(base, weights))
        per_query: dict[str, Any] = {}
        for query, (jd_text, labels) in dataset.queries.items():
            job = engine.analyze_job(jd_text)
            scores = {
                rid: engine.match_text(texts[rid], jd_text, rid, job=job, today=dataset.reference_date).overall_score
                for rid in labels
            }
            per_query[query] = _query_metrics(scores, labels, dataset.relevant_threshold)
        mean = {k: round(sum(q[k] for q in per_query.values()) / len(per_query), 4) for k in METRIC_KEYS}
        report["variants"][variant] = {"mean": mean, "queries": per_query}
        logger.info("Evaluated variant=%s %s", variant, mean)

    report.update(
        model_name=base.scoring.model_name,
        model_version=base.scoring.version,
        config_fingerprint=base.fingerprint,
        taxonomy_version=base.taxonomy.version,
        weights=base.scoring.weights.model_dump(),
        via_pdf=via_pdf,
        queries=len(dataset.queries),
        judgments=sum(len(labels) for _, labels in dataset.queries.values()),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return report


def save_report(report: dict[str, Any], report_dir: Path = DEFAULT_REPORT_DIR) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = report["timestamp"].replace(":", "").replace("-", "").replace("+0000", "Z")
    path = report_dir / f"run_{stamp}_{report['config_fingerprint']}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    history = report_dir / "history.csv"
    new_file = not history.exists()
    with history.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["timestamp", "model_version", "config_fingerprint", "variant", "via_pdf", *METRIC_KEYS])
        for variant, data in report["variants"].items():
            writer.writerow(
                [
                    report["timestamp"],
                    report["model_version"],
                    report["config_fingerprint"],
                    variant,
                    report["via_pdf"],
                    *[data["mean"][k] for k in METRIC_KEYS],
                ]
            )

    registry_path = report_dir.parent / "model_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {"models": {}}
    key = f"{report['model_version']}+{report['config_fingerprint']}"
    registry["models"][key] = {
        "model_name": report["model_name"],
        "model_version": report["model_version"],
        "config_fingerprint": report["config_fingerprint"],
        "taxonomy_version": report["taxonomy_version"],
        "weights": report["weights"],
        "last_evaluated": report["timestamp"],
        "metrics": report["variants"]["full_model"]["mean"] if "full_model" in report["variants"] else {},
        "report": path.name,
    }
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return path


def format_summary(report: dict[str, Any]) -> str:
    header = f"{'variant':<14}" + "".join(f"{k:>13}" for k in METRIC_KEYS)
    lines = [
        f"Model {report['model_name']} v{report['model_version']} (config {report['config_fingerprint']}), "
        f"{report['queries']} queries / {report['judgments']} judgments, via_pdf={report['via_pdf']}",
        header,
        "-" * len(header),
    ]
    for variant, data in report["variants"].items():
        lines.append(f"{variant:<14}" + "".join(f"{data['mean'][k]:>13.3f}" for k in METRIC_KEYS))
    return "\n".join(lines)
