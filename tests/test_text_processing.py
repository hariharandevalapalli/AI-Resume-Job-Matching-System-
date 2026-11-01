from __future__ import annotations

from resume_matcher.config import AppConfig
from resume_matcher.text_processing import (
    SectionDetector,
    merge_wrapped_lines,
    normalize_text,
    snippet_around,
    word_count,
)


def test_normalize_removes_bullets_and_unifies_dashes() -> None:
    text = "• Built APIs\r\n▪  Led team\n2019 – 2021\n\n\n\nNext"
    result = normalize_text(text)
    assert result.split("\n")[:3] == ["Built APIs", "Led team", "2019 - 2021"]
    assert "\n\n\n" not in result


def test_normalize_handles_ligatures() -> None:
    assert normalize_text("ﬁnance ofﬁce") == "finance office"


def test_merge_wrapped_lines() -> None:
    lines = ["Built a recommendation engine that served", "millions of users daily.", "Skills"]
    assert merge_wrapped_lines(lines) == [
        "Built a recommendation engine that served millions of users daily.",
        "Skills",
    ]


def test_section_detection_with_inline_and_caps_headers(config: AppConfig) -> None:
    detector = SectionDetector(config.scoring.resume_sections)
    text = "Jane Doe\nPROFESSIONAL EXPERIENCE\nEngineer at X\nTechnical Skills: Python, SQL\nEducation\nBSc"
    sections = detector.split(text)
    assert sections.found() == ["experience", "skills", "education"]
    assert sections.lines["skills"] == ["Python, SQL"]
    assert sections.lines["general"] == ["Jane Doe"]


def test_sentence_is_not_a_header(config: AppConfig) -> None:
    detector = SectionDetector(config.scoring.resume_sections)
    assert detector.match_header("I have experience leading projects.") is None
    assert detector.match_header("Projects") == ("projects", "")


def test_snippet_trimming() -> None:
    line = "x" * 300 + " Python " + "y" * 300
    start = line.index("Python")
    snippet = snippet_around(line, start, start + 6, width=100)
    assert "Python" in snippet and snippet.startswith("...") and snippet.endswith("...")
    assert len(snippet) <= 110


def test_word_count() -> None:
    assert word_count("Hello, world! 3 times.") == 4
