# One image serves both the API and the Streamlit UI (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY resume_matcher ./resume_matcher
COPY config ./config
COPY frontend ./frontend
COPY data ./data
COPY samples ./samples
COPY .streamlit ./.streamlit
RUN pip install --no-deps . \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
CMD ["uvicorn", "resume_matcher.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
