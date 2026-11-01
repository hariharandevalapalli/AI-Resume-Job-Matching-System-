"""Frontend tests: the HTTP client and headless rendering of every Streamlit page."""

from __future__ import annotations

import socket
import sys
from pathlib import Path
from typing import Any

import pytest
import requests
from streamlit.testing.v1 import AppTest

from resume_matcher.engine import MatchEngine
from tests.conftest import JD_TEXT, STRONG_RESUME, TODAY, WEAK_RESUME

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
sys.path.insert(0, str(FRONTEND))

from api_client import ApiClient, ApiError  # noqa: E402


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_client_reports_unreachable_backend() -> None:
    client = ApiClient(f"http://127.0.0.1:{_closed_port()}")
    with pytest.raises(ApiError, match="Cannot reach the analysis service"):
        client.health()


def test_client_surfaces_backend_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    response = requests.Response()
    response.status_code = 415
    response._content = b'{"detail": "not a PDF file"}'
    monkeypatch.setattr(requests, "request", lambda *args, **kwargs: response)
    with pytest.raises(ApiError, match="not a PDF file"):
        ApiClient("http://example.invalid").match("cv.docx", b"data", JD_TEXT)


def _page(name: str) -> AppTest:
    return AppTest.from_file(str(FRONTEND / "views" / f"{name}.py"), default_timeout=30)


@pytest.fixture(autouse=True)
def _backend_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the frontend at a closed port so page tests never depend on a running API."""
    monkeypatch.setenv("API_URL", f"http://127.0.0.1:{_closed_port()}")


@pytest.mark.parametrize("name", ["home", "analyze", "about", "results", "compare"])
def test_pages_render_without_backend(name: str) -> None:
    at = _page(name).run()
    assert not at.exception, at.exception


def _result(engine: MatchEngine, resume: str, name: str) -> dict[str, Any]:
    return engine.match_text(resume, JD_TEXT, resume_name=name, today=TODAY).model_dump(mode="json")


def test_results_page_renders_full_analysis(engine: MatchEngine) -> None:
    at = _page("results")
    at.session_state["result"] = _result(engine, STRONG_RESUME, "jane.pdf")
    at.run()
    assert not at.exception, at.exception
    assert any("jane.pdf" in t.value for t in at.title)
    assert [t.label for t in at.tabs][:2] == ["Score breakdown", "Skills & evidence"]


def test_compare_page_renders_rankings(engine: MatchEngine) -> None:
    comparison = engine.compare_texts([("strong.pdf", STRONG_RESUME), ("weak.pdf", WEAK_RESUME)], JD_TEXT)
    at = _page("compare")
    at.session_state["comparison"] = comparison.model_dump(mode="json")
    at.run()
    assert not at.exception, at.exception
    assert at.dataframe[0].value["Resume"].tolist() == ["strong.pdf", "weak.pdf"]


def test_markdown_report(engine: MatchEngine) -> None:
    from ui import markdown_report

    report = markdown_report(_result(engine, WEAK_RESUME, "john.pdf"))
    assert report.startswith("# Resume match report: john.pdf")
    assert "## Your results, in plain language" in report
    assert "Required skills not found in the resume" in report
    assert "build a small practice project" in report
    assert "## Recommendations" in report and "not a hiring decision" in report
