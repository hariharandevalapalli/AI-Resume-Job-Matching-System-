"""Domain exceptions. The API maps each one to an HTTP status code."""

from __future__ import annotations


class ResumeMatcherError(Exception):
    """Base class for all expected, user-facing errors."""

    status_code: int = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidInputError(ResumeMatcherError):
    """Input failed validation (too short, too long, empty...)."""

    status_code = 422


class UnsupportedFileTypeError(ResumeMatcherError):
    """The uploaded file is not a PDF."""

    status_code = 415


class FileTooLargeError(ResumeMatcherError):
    """The uploaded file exceeds the configured size limit."""

    status_code = 413


class DocumentParsingError(ResumeMatcherError):
    """The PDF could not be read or contains no extractable text."""

    status_code = 422


class ConfigurationError(Exception):
    """The YAML configuration is invalid. Not user-facing."""
