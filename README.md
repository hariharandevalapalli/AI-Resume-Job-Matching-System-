# AI Resume–Job Matcher

**An explainable NLP and MLOps-based resume ranking system.**

Upload a PDF resume and a job description, and the system returns a 0–100 match score that you can audit.
Every point is traced back to a weighted component, and every skill is backed by the exact resume line that
supports it. It also lists missing skills, gives prioritised resume suggestions and builds a phased learning
roadmap. Several resumes can be ranked against one job.

The system runs on a CPU-only student laptop: no GPU, no large pretrained models and no external API calls.
It is built on TF-IDF, cosine similarity, a configurable skill taxonomy and transparent rules.

> **Responsible use.** Scores are indicative estimates of document overlap. They are **not** hiring decisions
> and must always be reviewed by a person. Resumes are processed in memory and are not stored.

---

## Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [How the score works](#how-the-score-works)
4. [Windows setup](#windows-setup)
5. [Running the application](#running-the-application)
6. [Using the API](#using-the-api)
7. [Tests, linting and evaluation](#tests-linting-and-evaluation)
8. [MLOps practices](#mlops-practices)
9. [Docker](#docker)
10. [Configuration guide](#configuration-guide)
11. [Privacy](#privacy)
12. [Verification report](#verification-report)
13. [Limitations](#limitations)

---

## Features

| Area | What it does |
|---|---|
| PDF processing | In-memory text extraction with PyMuPDF; validates type, size, page count, encryption and scanned/empty PDFs |
| Resume analysis | Section detection (Experience, Projects, Skills, Education…), years of experience from date ranges, highest degree, quantified achievements, PII-redacted preview |
| Job description analysis | Required vs preferred skills (from headings and phrases like "is a plus"), "A or B" alternatives, minimum years, minimum degree; ignores Benefits/About-us sections |
| Skill extraction | 115-skill taxonomy with aliases (`k8s` → Kubernetes); technology-aware boundaries (Java ≠ JavaScript, C ≠ C++); longest match wins (GitHub Actions ≠ GitHub) |
| Explainable scoring | Five weighted components with re-normalisation, per-skill evidence (demonstrated / listed / related), TF-IDF term contributions |
| Recommendations | Prioritised, actionable resume suggestions plus a 4-phase learning roadmap with effort estimates and official resources |
| Comparison | Rank 2–10 resumes; stacked score breakdown, skill coverage matrix, CSV export, per-resume drill-down |
| Dashboard | Streamlit UI with Home, Analyze, Results, Compare and About & Methodology pages |
| API | FastAPI with Swagger docs, validation, structured errors, request IDs and timing logs |
| MLOps | Config-driven scoring, config fingerprint versioning, evaluation framework with baselines, model registry, logging, Docker |

## Architecture

```
┌─────────────────────────┐   HTTP (multipart / JSON)   ┌──────────────────────────────────────┐
│  Streamlit frontend      │ ─────────────────────────▶ │  FastAPI backend (resume_matcher.api) │
│  frontend/app.py         │ ◀───────────────────────── │  /api/v1/match, /compare, ...         │
│  (never imports backend) │        JSON results         └──────────────────┬───────────────────┘
└─────────────────────────┘                                                 │
                                                                MatchEngine (engine.py)
        ┌──────────────┬───────────────┬──────────────────┬──────────────────┼───────────────────┐
   pdf_parser     resume_analyzer   jd_analyzer         scoring          recommendations     versioning
   (PyMuPDF)      sections, dates,  required/preferred, TF-IDF cosine,  suggestions,        model version +
                  degrees, PII      alternatives        skill coverage,  learning roadmap    config fingerprint
                         └──────── skills.py (taxonomy-driven extractor) ┘
                                              │
                         config/scoring.yaml  +  config/skills_taxonomy.yaml
```

```
mlops/
├── config/
│   ├── scoring.yaml            # weights, credits, thresholds, parsing rules, limits
│   └── skills_taxonomy.yaml    # skills, aliases, related skills, learning resources
├── resume_matcher/             # backend package
│   ├── api/main.py             # FastAPI app
│   ├── engine.py               # orchestration (single match + comparison)
│   ├── pdf_parser.py           # in-memory PDF extraction + validation
│   ├── text_processing.py      # normalisation, section detection
│   ├── skills.py               # rule-based skill extraction with evidence
│   ├── jd_analyzer.py          # job description analysis
│   ├── resume_analyzer.py      # experience, education, quality signals
│   ├── patterns.py             # regexes: years, date ranges, degrees, PII
│   ├── scoring.py              # TF-IDF similarity, coverage, aggregation
│   ├── recommendations.py      # suggestions + learning roadmap
│   ├── versioning.py           # model info / fingerprint
│   ├── schemas.py              # Pydantic models shared by engine and API
│   ├── sample_pdf.py           # text → PDF renderer (samples, tests, evaluation)
│   └── evaluation/             # ranking metrics, runner, CLI
├── frontend/                   # Streamlit app (app.py, ui.py, views/, styles.css)
├── data/                       # sample resumes, job descriptions, evaluation labels
├── samples/resumes/            # generated sample PDFs (fictional candidates)
├── scripts/generate_sample_pdfs.py
├── reports/                    # evaluation runs, history.csv, model_registry.json
├── tests/                      # 113 pytest tests
├── Dockerfile, docker-compose.yml
├── requirements.txt, requirements-dev.txt, pyproject.toml
└── .streamlit/config.toml
```

## How the score works

```
overall = 100 × Σ (w_c / Σ w_included) × s_c      for each included component c
```

| Component | Default weight | Raw score s (0–1) |
|---|---|---|
| Required skills | 0.45 | Weighted average credit over the job's required skills |
| Preferred skills | 0.15 | Same, for nice-to-have skills |
| Content similarity | 0.25 | TF-IDF cosine (unigrams + bigrams) ÷ 0.5, capped at 1 |
| Experience | 0.10 | min(1, years detected ÷ years required) |
| Education | 0.05 | 1 if degree meets the requirement, 0.5 if one level below, else 0 |

**A component is excluded** when the job description gives nothing to compare against (for example, no degree
is requested). The remaining weights are then re-normalised, so a missing requirement is never counted as a
failure.

**Skill credit** depends on the evidence:

| Evidence | Credit |
|---|---|
| Shown in Experience / Projects / Achievements (or a descriptive sentence) | 1.00 |
| Listed only (Skills section or summary) | 0.85 |
| Missing, but a related skill is present (e.g. TensorFlow for PyTorch) | 0.50 |
| Not found | 0 |

* **Alternatives:** "PyTorch or TensorFlow" counts as **one** requirement. Either skill satisfies it, and the
  group's weight is shared so one skill is not rewarded twice.
* **Soft skills** are weighted ×0.5, because resumes rarely state them explicitly.
* **Consistency:** TF-IDF is fitted on the resume/job pair only, so a resume gets the same score alone or in a
  comparison.
* **Score bands:** Strong ≥ 75, Good ≥ 60, Partial ≥ 40, Low < 40.

Every number above lives in [config/scoring.yaml](config/scoring.yaml).

## Windows setup

Requirements: **Python 3.10 or newer** (developed and verified on Python 3.11.9, Windows 11). Open
**PowerShell** in the project folder:

```powershell
cd D:\Web-dev\College-Projects\mlops
```

Create a virtual environment and install dependencies:

```powershell
py -3.11 -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

Optionally activate the environment so you can type `python` instead of `.\.venv\Scripts\python.exe`:

```powershell
.\.venv\Scripts\Activate.ps1
```

> If PowerShell refuses to run `Activate.ps1` because of its execution policy, skip activation and keep using
> the full `.\.venv\Scripts\python.exe` path. Every command below works without activation.

Generate the sample PDF resumes (they are already included; this regenerates them):

```powershell
.\.venv\Scripts\python.exe scripts\generate_sample_pdfs.py
```

## Running the application

The backend and frontend run as two processes. Use **two PowerShell windows**, both in the project folder.

**Window 1: API (FastAPI + Uvicorn)**

```powershell
.\.venv\Scripts\python.exe -m uvicorn resume_matcher.api.main:app --host 127.0.0.1 --port 8000
```

* Swagger UI: <http://127.0.0.1:8000/docs>
* Health check: <http://127.0.0.1:8000/health>

**Window 2: dashboard (Streamlit)**

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend\app.py --server.port 8501
```

* Dashboard: <http://localhost:8501>

The frontend reaches the API at `http://localhost:8000` by default. To point it elsewhere, set `API_URL` first:

```powershell
$env:API_URL = "http://127.0.0.1:8000"
```

**Quick tour:** open the dashboard, choose **Analyze resume**, then **Use a sample resume**. Pick *ML Engineer
Senior* and the *ML Engineer* sample job description, then click **Run analysis**. For rankings, open
**Compare resumes**, turn on **Add sample resumes** and pick a sample job.

## Using the API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Service status, model version, config fingerprint |
| GET | `/api/v1/model-info` | Weights, credits, bands, versions |
| GET | `/api/v1/skills` | Skill taxonomy grouped by category |
| POST | `/api/v1/analyze-job` | JSON `{job_description}` → required/preferred skills, years, degree |
| POST | `/api/v1/match` | Multipart `resume` (PDF) + `job_description` → full explainable result |
| POST | `/api/v1/match-text` | JSON `{resume_text, job_description, resume_name}` → same result for plain text |
| POST | `/api/v1/compare` | Multipart `resumes` (2–10 PDFs) + `job_description` → ranking, matrix, per-resume results |

Error codes: `413` file too large, `415` not a PDF, `422` invalid input (short JD, unreadable/scanned PDF,
too many files…). Every error returns `{"detail": "<human-readable message>"}`, and every response carries an
`X-Request-ID` header.

PowerShell examples (use `curl.exe`, because plain `curl` is an alias for `Invoke-WebRequest`):

```powershell
curl.exe http://127.0.0.1:8000/health
```

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/match -F "resume=@samples/resumes/data_analyst.pdf;type=application/pdf" -F "job_description=<data/job_descriptions/data_analyst.txt"
```

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/compare -F "resumes=@samples/resumes/data_analyst.pdf" -F "resumes=@samples/resumes/devops_engineer.pdf" -F "job_description=<data/job_descriptions/data_analyst.txt"
```

## Tests, linting and evaluation

Run the test suite (113 tests: parsing, extraction, JD analysis, scoring, recommendations, API, evaluation
metrics and headless rendering of every Streamlit page):

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Lint and format check with Ruff:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
```

```powershell
.\.venv\Scripts\python.exe -m ruff format --check .
```

Run the ranking evaluation. It writes a report, appends to `reports/evaluation/history.csv` and updates
`reports/model_registry.json`:

```powershell
.\.venv\Scripts\python.exe -m resume_matcher.evaluation
```

Useful flags: `--via-pdf` renders each resume to PDF and parses it back, testing the full pipeline.
`--no-save` prints metrics only. `--min-ndcg 0.9` exits with code 1 if NDCG@3 falls below the threshold,
which works as a quality gate.

### Evaluation results (measured)

Dataset: 3 job descriptions × 9 fictional resumes = 27 hand-labelled judgments (grades 0–3, "relevant" = grade
≥ 2). See [data/evaluation/labels.yaml](data/evaluation/labels.yaml). Model v1.0.0, config `918e58022923`:

| Variant | NDCG@3 | NDCG@5 | MAP | MRR | Precision@3 | Spearman ρ |
|---|---|---|---|---|---|---|
| **Full model** | 1.000 | 0.987 | 1.000 | 1.000 | 0.778 | 0.861 |
| TF-IDF only (baseline) | 1.000 | 0.987 | 1.000 | 1.000 | 0.778 | 0.861 |
| Skills only (baseline) | 1.000 | 0.974 | 1.000 | 1.000 | 0.778 | 0.828 |

How to read this honestly:

* Precision@3 of 0.778 is the **maximum possible** here, because two of the three jobs have only two relevant
  resumes.
* On this small, clearly separated dataset, the full model **ranks as well as the TF-IDF baseline, not
  better**. Its advantage is explainability: it can say *which* skills match, *where* the evidence is, and
  *what* is missing, which raw cosine similarity cannot. It beats the skills-only baseline on NDCG@5 and
  Spearman.
* The `--via-pdf` run produces identical metrics, so PDF extraction does not degrade quality on these samples.
* The labels were written by the project author, and the dataset is too small to support claims about
  real-world accuracy. It is a regression check, not a validation study.

## MLOps practices

* **Configuration-driven scoring:** weights, credits, thresholds, section names, JD cues and limits live in YAML
  and are validated with Pydantic at startup. Invalid config fails fast with a clear message.
* **Model versioning:** each result carries `model_version` (from `scoring.yaml`), `taxonomy_version` and a
  12-character **config fingerprint** (SHA-256 of the full configuration). Any weight change produces a new
  fingerprint.
* **Model registry:** `reports/model_registry.json` stores the latest metrics per `version+fingerprint`.
  `reports/evaluation/history.csv` keeps one row per run and variant, so changes can be compared over time.
* **Reproducible evaluation:** fixed labelled dataset, fixed reference date for "Present", deterministic
  algorithms, and baseline/ablation variants in every run.
* **Quality gate:** `--min-ndcg` returns a non-zero exit code on regression, and a test asserts NDCG@3 ≥ 0.9
  and MRR = 1.
* **Logging:** timestamped, levelled logs (`LOG_LEVEL` env var). The API logs request ID, path, status and
  latency, and the engine logs scores and timings. **Resume content is never logged.**
* **Containerisation:** a single Docker image serves both the API and the UI through `docker-compose.yml`.

## Docker

With Docker Desktop running, from the project folder:

```powershell
docker compose up --build
```

Then open <http://localhost:8501> (UI) and <http://localhost:8000/docs> (API). The UI container reaches the API
at `http://api:8000`. The UI waits until the API health check passes. To stop:

```powershell
docker compose down
```

## Configuration guide

**Change weights:** edit `weights:` in [config/scoring.yaml](config/scoring.yaml), bump `version:`, restart the
API and re-run the evaluation to record the new fingerprint.

**Add a skill:** append an entry to [config/skills_taxonomy.yaml](config/skills_taxonomy.yaml):

```yaml
  - name: Svelte
    category: web_development
    aliases: [svelte, sveltekit]
    related: [React, Vue.js]
    resources:
      - {title: Svelte tutorial, url: "https://svelte.dev/tutorial"}
```

Put ambiguous aliases (common English words, single letters) in `case_sensitive_aliases`. The taxonomy is
validated on load: duplicate names, unknown categories and unknown related skills are rejected.

**Use a different config folder:** set `$env:RESUME_MATCHER_CONFIG_DIR = "D:\path\to\config"` before starting
the API.

## Privacy

* PDFs are read from memory (`pymupdf.open(stream=...)`), and the application never writes resumes to disk.
  Note: for uploads larger than about 1 MB, the web framework (Starlette) may buffer the upload in a temporary
  file that is deleted when the request finishes.
* Nothing is persisted between requests. Results live only in the browser session.
* Logs contain sizes, counts, scores and timings, never resume text.
* Previews redact the candidate's name line, emails, phone numbers and links. Contact details are reported only
  as "present / not present".

## Verification report

The following were **actually executed** during development on Windows 11 with Python 3.11.9:

| Check | Result |
|---|---|
| `pytest` | **113 passed** |
| `ruff check .` and `ruff format --check .` | All checks passed |
| `python -m resume_matcher.evaluation --min-ndcg 0.9` | Passed (NDCG@3 = 1.000); report and registry written |
| `python -m resume_matcher.evaluation --via-pdf --no-save` | Same metrics as the text run |
| API via Uvicorn + `curl` | `/health`, `/model-info`, `/match` (PDF), `/compare` (with a non-PDF reported as a per-file failure) and a 415 for a non-PDF all behaved as expected |
| Streamlit UI in a browser | Home, Analyze → Results (sample resume + sample JD), Compare (5 sample resumes), and About were exercised; no horizontal overflow at desktop width or at a ~440 px narrow width |
| `docker compose config` | Valid |

**Not verified:** `docker compose up --build` was **not run**, because the Docker engine was not running on the
development machine. The Dockerfile and compose file are written and syntax-checked, but the image build and
containers have not been tested. Uploading a file through the browser's file picker was not automated (the
same upload path is covered by API tests and `curl`). Learning-resource links point to official documentation
but are not checked automatically.

## Limitations

* **Rule-based extraction:** skills not in the taxonomy, or phrased unusually ("built neural nets" for Deep
  Learning), may be missed. Common words used as aliases can occasionally cause false positives.
* **"Go" and "R"** are matched only when capitalised, so "Go" at the start of a sentence can be a false
  positive.
* **Degree lines:** to avoid treating fields of study as skills, JD clauses that mention a degree are not used
  for skill extraction. A skill written in the same sentence as the degree requirement is therefore ignored.
* **Experience estimation** relies on date ranges in the Experience section (or stated "X years"). It merges
  overlaps but cannot tell part-time from full-time work, and year-only ranges are approximate.
* **No OCR:** scanned or image-only PDFs are rejected with an explanation. Multi-column layouts may extract in
  a mixed order.
* **English and technical roles:** the taxonomy and parsing cues are tuned for English-language tech, data
  and engineering jobs.
* **TF-IDF** captures word overlap, not meaning: synonyms do not match unless they are taxonomy aliases.
* **Evaluation:** the labelled dataset is small and synthetic (see above), so the metrics show the system
  behaves sensibly. They do not prove fairness or predictive validity for hiring.
* **Fairness:** the system ignores names, gender, age and photos, but text-overlap scoring can still favour
  candidates who write in a particular style. Treat scores as one input to human judgement, never as a
  decision.
#   A I - R e s u m e - M l o p s  
 # AI-Resume-Job-Matching-System-
# AI-Resume-Job-Matching-System-
