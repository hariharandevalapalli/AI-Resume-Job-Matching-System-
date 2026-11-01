"""HTTP client for the FastAPI backend. The frontend never imports backend code."""

from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_API_URL = "http://localhost:8000"


class ApiError(Exception):
    """A user-presentable error from the backend or the network."""


class ApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = 90.0) -> None:
        self.base_url = (base_url or os.getenv("API_URL", DEFAULT_API_URL)).rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = requests.request(method, url, timeout=kwargs.pop("timeout", self.timeout), **kwargs)
        except requests.ConnectionError as exc:
            raise ApiError(
                f"Cannot reach the analysis service at {self.base_url}. Make sure the backend is running."
            ) from exc
        except requests.Timeout as exc:
            raise ApiError("The analysis service took too long to respond. Please try again.") from exc
        if response.ok:
            return response.json()
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text or response.reason
        raise ApiError(str(detail))

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health", timeout=5)

    def model_info(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/model-info", timeout=10)

    def skills(self) -> dict[str, list[str]]:
        return self._request("GET", "/api/v1/skills", timeout=10)

    def analyze_job(self, job_description: str) -> dict[str, Any]:
        return self._request("POST", "/api/v1/analyze-job", json={"job_description": job_description})

    def match(self, filename: str, data: bytes, job_description: str) -> dict[str, Any]:
        files = {"resume": (filename, data, "application/pdf")}
        return self._request("POST", "/api/v1/match", files=files, data={"job_description": job_description})

    def compare(self, documents: list[tuple[str, bytes]], job_description: str) -> dict[str, Any]:
        files = [("resumes", (name, data, "application/pdf")) for name, data in documents]
        return self._request("POST", "/api/v1/compare", files=files, data={"job_description": job_description})
