from __future__ import annotations

import pymupdf
import pytest

from resume_matcher.config import AppConfig
from resume_matcher.exceptions import DocumentParsingError, FileTooLargeError, UnsupportedFileTypeError
from resume_matcher.pdf_parser import extract_text
from resume_matcher.sample_pdf import render_text_pdf
from tests.conftest import STRONG_RESUME


def test_extracts_text(config: AppConfig, strong_pdf: bytes) -> None:
    parsed = extract_text(strong_pdf, "cv.pdf", config.scoring.limits)
    assert parsed.page_count == 1
    assert "scikit-learn" in parsed.text
    assert "Jan 2021 - Present" in parsed.text


def test_multi_page_pdf(config: AppConfig) -> None:
    long_text = "\n".join(f"Line {i}: built Python services for analytics" for i in range(120))
    parsed = extract_text(render_text_pdf(long_text), "long.pdf", config.scoring.limits)
    assert parsed.page_count >= 2
    assert "Line 119" in parsed.text


def test_rejects_non_pdf(config: AppConfig) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        extract_text(b"Hello, I am a text file pretending to be a resume", "cv.pdf", config.scoring.limits)


def test_rejects_empty_file(config: AppConfig) -> None:
    with pytest.raises(DocumentParsingError, match="empty"):
        extract_text(b"", "cv.pdf", config.scoring.limits)


def test_rejects_oversized_file(config: AppConfig) -> None:
    data = b"%PDF-1.7\n" + b"0" * (config.scoring.limits.max_file_size_bytes + 1)
    with pytest.raises(FileTooLargeError):
        extract_text(data, "big.pdf", config.scoring.limits)


def test_rejects_corrupted_pdf(config: AppConfig) -> None:
    with pytest.raises(DocumentParsingError, match="could not be opened"):
        extract_text(b"%PDF-1.7\nthis is not really a pdf body", "broken.pdf", config.scoring.limits)


def test_rejects_image_only_pdf(config: AppConfig) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(50, 50, 200, 200), color=(0, 0, 0), fill=(0.5, 0.5, 0.5))
    data = doc.tobytes()
    with pytest.raises(DocumentParsingError, match="no selectable text"):
        extract_text(data, "scan.pdf", config.scoring.limits)


def test_rejects_password_protected_pdf(config: AppConfig) -> None:
    doc = pymupdf.open(stream=render_text_pdf(STRONG_RESUME), filetype="pdf")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="secret")
    with pytest.raises(DocumentParsingError, match="password"):
        extract_text(data, "locked.pdf", config.scoring.limits)


def test_rejects_too_many_pages(config: AppConfig) -> None:
    doc = pymupdf.open()
    for i in range(config.scoring.limits.max_pdf_pages + 1):
        doc.new_page().insert_text((50, 72), f"Page {i} with enough words to be extracted as text content")
    with pytest.raises(DocumentParsingError, match="pages"):
        extract_text(doc.tobytes(), "huge.pdf", config.scoring.limits)
