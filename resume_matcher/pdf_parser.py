"""In-memory PDF text extraction with PyMuPDF. Nothing is written to disk."""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf

from resume_matcher.config import LimitsConfig
from resume_matcher.exceptions import DocumentParsingError, FileTooLargeError, UnsupportedFileTypeError
from resume_matcher.logging_config import get_logger

logger = get_logger(__name__)
PDF_MAGIC = b"%PDF-"


@dataclass(frozen=True)
class ParsedDocument:
    text: str
    page_count: int


def validate_pdf_bytes(data: bytes, filename: str, limits: LimitsConfig) -> None:
    if not data:
        raise DocumentParsingError(f"'{filename}' is empty.")
    if len(data) > limits.max_file_size_bytes:
        raise FileTooLargeError(
            f"'{filename}' is {len(data) / 1024 / 1024:.1f} MB; the limit is {limits.max_file_size_mb:g} MB."
        )
    # PDF headers may be preceded by a little junk; the spec allows it within the first 1 KB.
    if PDF_MAGIC not in data[:1024]:
        raise UnsupportedFileTypeError(f"'{filename}' is not a PDF file. Please upload a PDF resume.")


def extract_text(data: bytes, filename: str, limits: LimitsConfig) -> ParsedDocument:
    """Extract plain text from PDF bytes, validating size, type, encryption and page count."""
    validate_pdf_bytes(data, filename, limits)
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several exception types for corrupt files
        raise DocumentParsingError(f"'{filename}' could not be opened as a PDF (the file may be corrupted).") from exc

    try:
        if doc.needs_pass:
            raise DocumentParsingError(f"'{filename}' is password-protected. Please upload an unlocked PDF.")
        if doc.page_count == 0:
            raise DocumentParsingError(f"'{filename}' has no pages.")
        if doc.page_count > limits.max_pdf_pages:
            raise DocumentParsingError(
                f"'{filename}' has {doc.page_count} pages; resumes longer than {limits.max_pdf_pages} pages "
                "are not supported."
            )
        pages = [page.get_text("text", sort=True) for page in doc]
        page_count = doc.page_count
    finally:
        doc.close()

    text = "\n".join(pages).strip()
    if len(text) < limits.min_resume_chars:
        raise DocumentParsingError(
            f"'{filename}' contains little or no selectable text. Scanned/image-only PDFs are not supported "
            "(no OCR); please export the resume as a text-based PDF."
        )
    text = text[: limits.max_resume_chars]
    logger.info("Parsed PDF: pages=%d chars=%d", page_count, len(text))
    return ParsedDocument(text=text, page_count=page_count)
