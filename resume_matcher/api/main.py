"""FastAPI application exposing the matcher over HTTP (Swagger UI at /docs)."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from resume_matcher import __version__
from resume_matcher.engine import MatchEngine, get_engine
from resume_matcher.exceptions import FileTooLargeError, InvalidInputError, ResumeMatcherError
from resume_matcher.logging_config import get_logger, setup_logging
from resume_matcher.schemas import (
    ComparisonResult,
    FailedDocument,
    HealthResponse,
    JobAnalysis,
    JobDescriptionRequest,
    MatchResult,
    ModelInfo,
    TextMatchRequest,
)

setup_logging()
logger = get_logger("resume_matcher.api")

DESCRIPTION = """
Explainable resume-to-job matching using TF-IDF cosine similarity, rule-based skill extraction
and transparent, configuration-driven scoring.

* Resumes are processed **in memory** and are not stored.
* Scores are **indicative** and must not be used as automated hiring decisions.
"""

app = FastAPI(
    title="AI Resume-Job Matcher API",
    version=__version__,
    description=DESCRIPTION,
    openapi_tags=[
        {"name": "system", "description": "Health and model information."},
        {"name": "analysis", "description": "Job description analysis and resume matching."},
    ],
)

EngineDep = Annotated[MatchEngine, Depends(get_engine)]


@app.middleware("http")
async def request_logging(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled error request_id=%s path=%s", request_id, request.url.path)
        response = JSONResponse(status_code=500, content={"detail": "Internal server error.", "request_id": request_id})
    elapsed = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_id=%s method=%s path=%s status=%d elapsed_ms=%.0f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


@app.exception_handler(ResumeMatcherError)
async def handle_domain_error(_: Request, exc: ResumeMatcherError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    messages = []
    for err in exc.errors():
        location = ".".join(str(part) for part in err.get("loc", []) if part != "body")
        messages.append(f"{location}: {err.get('msg')}" if location else str(err.get("msg")))
    return JSONResponse(status_code=422, content={"detail": "; ".join(messages) or "Invalid request."})


async def _read_upload(upload: UploadFile, engine: MatchEngine) -> tuple[str, bytes]:
    """Read an upload fully into memory, refusing anything above the size limit."""
    limit = engine.config.scoring.limits.max_file_size_bytes
    name = upload.filename or "resume.pdf"
    data = await upload.read(limit + 1)
    await upload.close()
    if len(data) > limit:
        raise FileTooLargeError(
            f"'{name}' exceeds the {engine.config.scoring.limits.max_file_size_mb:g} MB upload limit."
        )
    return name, data


# ---------------------------------------------------------------------------- system


@app.get("/health", response_model=HealthResponse, tags=["system"], summary="Service health check")
def health(engine: EngineDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_version=__version__,
        model_version=engine.model_info.model_version,
        config_fingerprint=engine.model_info.config_fingerprint,
    )


@app.get("/api/v1/model-info", response_model=ModelInfo, tags=["system"], summary="Scoring model version and weights")
def get_model_info(engine: EngineDep) -> ModelInfo:
    return engine.model_info


@app.get("/api/v1/skills", tags=["system"], summary="Skill taxonomy grouped by category")
def list_skills(engine: EngineDep) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    taxonomy = engine.config.taxonomy
    for skill in taxonomy.skills:
        grouped.setdefault(taxonomy.categories[skill.category].label, []).append(skill.name)
    return grouped


# ---------------------------------------------------------------------------- analysis


@app.post(
    "/api/v1/analyze-job",
    response_model=JobAnalysis,
    tags=["analysis"],
    summary="Extract required/preferred skills, experience and education from a job description",
)
def analyze_job(payload: JobDescriptionRequest, engine: EngineDep) -> JobAnalysis:
    return engine.analyze_job(payload.job_description)


@app.post(
    "/api/v1/match",
    response_model=MatchResult,
    tags=["analysis"],
    summary="Match one PDF resume against a job description",
    responses={413: {"description": "File too large"}, 415: {"description": "Not a PDF"}},
)
async def match_resume(
    engine: EngineDep,
    resume: Annotated[UploadFile, File(description="Resume in PDF format")],
    job_description: Annotated[str, Form(description="Job description text")],
) -> MatchResult:
    name, data = await _read_upload(resume, engine)
    return await run_in_threadpool(engine.match_pdf, data, name, job_description)


@app.post(
    "/api/v1/match-text",
    response_model=MatchResult,
    tags=["analysis"],
    summary="Match plain resume text against a job description",
)
def match_text(payload: TextMatchRequest, engine: EngineDep) -> MatchResult:
    return engine.match_text(payload.resume_text, payload.job_description, resume_name=payload.resume_name)


@app.post(
    "/api/v1/compare",
    response_model=ComparisonResult,
    tags=["analysis"],
    summary="Rank several PDF resumes against one job description",
)
async def compare_resumes(
    engine: EngineDep,
    resumes: Annotated[list[UploadFile], File(description="Two or more PDF resumes")],
    job_description: Annotated[str, Form(description="Job description text")],
) -> ComparisonResult:
    max_count = engine.config.scoring.limits.max_resumes_compare
    if len(resumes) > max_count:
        raise InvalidInputError(f"At most {max_count} resumes can be compared at once.")
    documents: list[tuple[str, bytes]] = []
    rejected: list[FailedDocument] = []
    for upload in resumes:
        try:
            documents.append(await _read_upload(upload, engine))
        except FileTooLargeError as exc:
            rejected.append(FailedDocument(resume_name=upload.filename or "resume.pdf", error=exc.message))
    return await run_in_threadpool(engine.compare_pdfs, documents, job_description, rejected)
