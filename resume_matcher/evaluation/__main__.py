"""CLI: python -m resume_matcher.evaluation [--via-pdf] [--no-save] [--min-ndcg 0.8]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from resume_matcher.evaluation.runner import DEFAULT_LABELS, DEFAULT_REPORT_DIR, evaluate, format_summary, save_report
from resume_matcher.logging_config import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate ranking quality on the labelled dataset.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="Path to labels.yaml")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Where to write reports")
    parser.add_argument("--via-pdf", action="store_true", help="Render resumes to PDF and parse them back first")
    parser.add_argument("--no-save", action="store_true", help="Print metrics without writing reports")
    parser.add_argument(
        "--min-ndcg",
        type=float,
        default=None,
        help="Exit with code 1 if the full model's mean NDCG@3 is below this value (quality gate)",
    )
    args = parser.parse_args(argv)

    setup_logging("WARNING")
    report = evaluate(labels_path=args.labels, via_pdf=args.via_pdf)
    print(format_summary(report))
    if not args.no_save:
        path = save_report(report, args.report_dir)
        print(f"\nReport saved to {path}")

    if args.min_ndcg is not None:
        ndcg = report["variants"]["full_model"]["mean"]["ndcg@3"]
        if ndcg < args.min_ndcg:
            print(f"Quality gate FAILED: NDCG@3 {ndcg:.3f} < {args.min_ndcg:.3f}")
            return 1
        print(f"Quality gate passed: NDCG@3 {ndcg:.3f} >= {args.min_ndcg:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
