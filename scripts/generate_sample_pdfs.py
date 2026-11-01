"""Render the sample resumes in data/resumes/*.txt to PDFs in samples/resumes/.

Usage:  python scripts/generate_sample_pdfs.py
"""

from __future__ import annotations

from pathlib import Path

from resume_matcher.config import PROJECT_ROOT
from resume_matcher.sample_pdf import render_text_pdf

SOURCE_DIR = PROJECT_ROOT / "data" / "resumes"
OUTPUT_DIR = PROJECT_ROOT / "samples" / "resumes"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = sorted(SOURCE_DIR.glob("*.txt"))
    for source in sources:
        target: Path = OUTPUT_DIR / f"{source.stem}.pdf"
        target.write_bytes(render_text_pdf(source.read_text(encoding="utf-8"), title=source.stem))
        print(f"wrote {target.relative_to(PROJECT_ROOT)}")
    print(f"{len(sources)} sample PDFs generated (all candidates are fictional).")


if __name__ == "__main__":
    main()
