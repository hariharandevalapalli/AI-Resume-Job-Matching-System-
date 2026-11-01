from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from resume_matcher.api.main import app
from tests.conftest import JD_TEXT, STRONG_RESUME


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _pdf(name: str, data: bytes) -> tuple[str, tuple[str, bytes, str]]:
    return ("resumes", (name, data, "application/pdf"))


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert len(body["config_fingerprint"]) == 12
    assert "X-Request-ID" in response.headers


def test_model_info_and_skills(client: TestClient) -> None:
    info = client.get("/api/v1/model-info").json()
    assert set(info["weights"]) == {"required_skills", "preferred_skills", "text_similarity", "experience", "education"}
    skills = client.get("/api/v1/skills").json()
    assert "Python" in skills["Programming Languages"]


def test_openapi_docs_available(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/api/v1/match", "/api/v1/compare", "/api/v1/analyze-job", "/api/v1/match-text"} <= set(paths)


def test_analyze_job(client: TestClient) -> None:
    response = client.post("/api/v1/analyze-job", json={"job_description": JD_TEXT})
    assert response.status_code == 200
    assert "Python" in response.json()["required_skills"]


def test_match_pdf(client: TestClient, strong_pdf: bytes) -> None:
    response = client.post(
        "/api/v1/match",
        files={"resume": ("jane.pdf", strong_pdf, "application/pdf")},
        data={"job_description": JD_TEXT},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["resume_name"] == "jane.pdf"
    assert 0 <= body["overall_score"] <= 100
    assert body["resume_profile"]["page_count"] == 1
    assert body["components"] and body["recommendations"] is not None


def test_match_text(client: TestClient) -> None:
    response = client.post(
        "/api/v1/match-text",
        json={"resume_text": STRONG_RESUME, "job_description": JD_TEXT, "resume_name": "jane"},
    )
    assert response.status_code == 200
    assert response.json()["band"] == "Strong match"


def test_non_pdf_rejected_with_415(client: TestClient) -> None:
    response = client.post(
        "/api/v1/match",
        files={"resume": ("cv.docx", b"PK\x03\x04 not a pdf", "application/octet-stream")},
        data={"job_description": JD_TEXT},
    )
    assert response.status_code == 415
    assert "not a PDF" in response.json()["detail"]


def test_oversized_upload_rejected_with_413(client: TestClient) -> None:
    big = b"%PDF-1.7\n" + b"0" * (5 * 1024 * 1024 + 10)
    response = client.post(
        "/api/v1/match",
        files={"resume": ("big.pdf", big, "application/pdf")},
        data={"job_description": JD_TEXT},
    )
    assert response.status_code == 413


def test_short_job_description_rejected(client: TestClient, strong_pdf: bytes) -> None:
    response = client.post(
        "/api/v1/match",
        files={"resume": ("jane.pdf", strong_pdf, "application/pdf")},
        data={"job_description": "too short"},
    )
    assert response.status_code == 422
    assert "too short" in response.json()["detail"]


def test_missing_fields_give_readable_422(client: TestClient) -> None:
    response = client.post("/api/v1/match", data={"job_description": JD_TEXT})
    assert response.status_code == 422
    assert "resume" in response.json()["detail"]


def test_compare_ranks_and_reports_failures(client: TestClient, strong_pdf: bytes, weak_pdf: bytes) -> None:
    response = client.post(
        "/api/v1/compare",
        files=[_pdf("weak.pdf", weak_pdf), _pdf("strong.pdf", strong_pdf), _pdf("notes.pdf", b"plain text")],
        data={"job_description": JD_TEXT},
    )
    assert response.status_code == 200
    body = response.json()
    assert [r["resume_name"] for r in body["rankings"]] == ["strong.pdf", "weak.pdf"]
    assert body["rankings"][0]["rank"] == 1
    assert body["failed"][0]["resume_name"] == "notes.pdf"
    assert set(body["skill_matrix"]) == {"strong.pdf", "weak.pdf"}
    assert body["skill_matrix"]["strong.pdf"]["Python"] == "matched"


def test_compare_scores_match_single_analysis(client: TestClient, strong_pdf: bytes, weak_pdf: bytes) -> None:
    single = client.post(
        "/api/v1/match",
        files={"resume": ("strong.pdf", strong_pdf, "application/pdf")},
        data={"job_description": JD_TEXT},
    ).json()
    compared = client.post(
        "/api/v1/compare",
        files=[_pdf("strong.pdf", strong_pdf), _pdf("weak.pdf", weak_pdf)],
        data={"job_description": JD_TEXT},
    ).json()
    assert compared["rankings"][0]["overall_score"] == single["overall_score"]


def test_compare_duplicate_names_are_disambiguated(client: TestClient, strong_pdf: bytes) -> None:
    response = client.post(
        "/api/v1/compare",
        files=[_pdf("cv.pdf", strong_pdf), _pdf("cv.pdf", strong_pdf)],
        data={"job_description": JD_TEXT},
    )
    assert sorted(r["resume_name"] for r in response.json()["rankings"]) == ["cv.pdf", "cv.pdf (2)"]


def test_compare_needs_two_resumes(client: TestClient, strong_pdf: bytes) -> None:
    response = client.post("/api/v1/compare", files=[_pdf("one.pdf", strong_pdf)], data={"job_description": JD_TEXT})
    assert response.status_code == 422
    assert "at least two" in response.json()["detail"]


def test_compare_rejects_too_many(client: TestClient, strong_pdf: bytes) -> None:
    files = [_pdf(f"cv{i}.pdf", strong_pdf) for i in range(11)]
    response = client.post("/api/v1/compare", files=files, data={"job_description": JD_TEXT})
    assert response.status_code == 422


def test_compare_all_failed(client: TestClient) -> None:
    response = client.post(
        "/api/v1/compare",
        files=[_pdf("a.pdf", b"nope"), _pdf("b.pdf", b"nope")],
        data={"job_description": JD_TEXT},
    )
    assert response.status_code == 422
    assert "None of the resumes" in response.json()["detail"]
