# FlowForge

[![CI](https://github.com/amitworx/flowforge/actions/workflows/ci.yml/badge.svg)](https://github.com/amitworx/flowforge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FlowForge is an open-source, self-hostable Python document-processing and workflow automation platform. Its core is designed to run reliably without requiring any LLM or paid cloud AI services.

---

> [!WARNING]
> **Development vs. Production Notice:**
> The provided `docker-compose.yml`, `Dockerfile`, Django `runserver`, `.env.example`, and sample credentials (`SECRET_KEY`, `flowforge` database password, and sample credentials like `devuser:devpass123`) are **strictly for local development and testing only**.
> - **Authentication Security:** HTTP Basic Authentication (`Authorization: Basic ...`) transmits credentials encoded in Base64 (unencrypted). In local development over `localhost` / `127.0.0.1`, this is acceptable for rapid testing. Outside local development, **Basic Authentication strictly requires TLS/HTTPS** or a secure session/token infrastructure to prevent credential interception.
> - **Environment Validation:** In `production` or `staging` environments, the application **strictly rejects** the sample/insecure secret key, refuses to start if `DEBUG=True`, and rejects default database credentials.
> - **Port Hardening:** PostgreSQL in `docker-compose.yml` is hardened by default to bind only to `127.0.0.1` (localhost), preventing accidental exposure to the host network or public internet.
> - Never deploy to production using development configurations or sample secrets.

---

## Architectural Overview

- **Core Engine:** Django 5.2 (LTS) & Django REST Framework (DRF)
- **Database:** PostgreSQL 17 (via `psycopg` 3)
- **Configuration:** Typed configuration via `pydantic-settings` with strict environment-mode validation
- **Ingestion Engine:** Synchronous, deterministic CSV/JSON processing with safe multi-layer bounding and normalization
- **Linting & Formatting:** Ruff
- **Test Suite:** pytest with `pytest-django`

### API Layout
- Versioned API base: `/api/v1/`
- Health check endpoint: `GET /api/v1/health/`
- Ingestion jobs endpoint: `POST /api/v1/jobs/` (Authenticated)
- List user jobs: `GET /api/v1/jobs/` (Authenticated)
- Retrieve job metadata: `GET /api/v1/jobs/{uuid}/` (Authenticated)
- Download normalized JSON: `GET /api/v1/jobs/{uuid}/download/` (Authenticated)
- Create workflow definition: `POST /api/v1/workflows/` (Authenticated)
- List workflow definitions: `GET /api/v1/workflows/` (Authenticated)
- Retrieve workflow definition: `GET /api/v1/workflows/{uuid}/` (Authenticated)
- Execute workflow run: `POST /api/v1/workflows/{uuid}/runs/` (Authenticated, Synchronous)
- Enqueue async workflow run: `POST /api/v1/workflows/{uuid}/runs/async/` (Authenticated, Asynchronous)
- List workflow runs: `GET /api/v1/workflow-runs/` (Authenticated)
- Retrieve workflow run details: `GET /api/v1/workflow-runs/{uuid}/` (Authenticated)

---

## Quick Start with Docker Compose (Local Development)

Docker Compose provides a reproducible environment that starts both the PostgreSQL database and the web application without requiring PostgreSQL or Python installed on the host.

### Prerequisites
- Docker Engine & Docker Compose (v2+)

### Run Application and Database

```bash
docker compose up -d --build
```

The application will be accessible at:
- **Health Check:** `http://127.0.0.1:8001/api/v1/health/` (or host port configured via `PORT` variable)

### Running Database Migrations
Migrations do **not** run automatically on container startup to prevent unexpected schema mutations. To apply migrations inside Docker Compose:

```bash
docker compose exec web python manage.py migrate
```

To view running containers and logs:
```bash
docker compose ps
docker compose logs -f web
```

To stop the containers (preserving persistent database volume):
```bash
docker compose down
```

---

## Ingestion Core API (M0-002 & M0-002A)

FlowForge provides deterministic ingestion for tabular data (CSV and JSON) without relying on external cloud AI or queues.

### Multi-Layer Upload Bounding Architecture
Uploads are bounded through defense-in-depth layers to guarantee application memory safety without reading unbounded payloads:
1. **Layer 1 (Request Boundary):** `Content-Length` header is checked on the incoming request before reading multipart data; requests exceeding the limit (1 MiB + multipart boundary overhead) are rejected early with `413 Request Entity Too Large`.
2. **Layer 2 (Upload Object Boundary):** Uploaded file metadata (`uploaded_file.size`) is inspected; files exceeding 1 MiB are rejected immediately with `413 Request Entity Too Large`.
3. **Layer 3 (Streaming Read Boundary):** `parse_and_normalize_stream` reads only up to `MAX_UPLOAD_SIZE_BYTES + 1` bytes from the stream, preventing unconstrained memory consumption regardless of client headers.
4. **Layer 4 (Record Count Boundary):** Parsers enforce a strict cap of **1,000 records/rows maximum**. Payloads with 1,001 or more rows are rejected cleanly with `400 Bad Request`.

### Policy & Validation Guardrails
- **CSV Handling:** Strict parsing (`csv.reader(..., strict=True)`) rejects malformed quoting, unterminated quotes, and syntax errors as clean 400 responses. UTF-8 encoded (UTF-8 BOM automatically handled), required non-empty headers, duplicate header detection, cell string preservation (e.g., leading zeros like `"00142"`).
- **JSON Handling:** UTF-8 encoded top-level array of uniform scalar objects. Nested dictionaries and lists are rejected to protect against unbounded recursion depth. Column ordering is deterministically unified across objects.
- **Extension & Content Disagreement Policy:** File extension and content must agree:
  - Files named `.json` must contain valid JSON syntax (`[` or `{`); uploading CSV with a `.json` extension is rejected with `400 Bad Request`.
  - Files named `.csv` must not contain JSON syntax; uploading JSON with a `.csv` extension is rejected with `400 Bad Request`.
  - Unsupported file extensions (e.g., `.exe`, `.py`, `.xml`) are cleanly rejected with `400 Bad Request`.
- **Authentication:** All `/api/v1/jobs/` endpoints require authentication. Basic authentication in local examples is for development testing only; production deployments require HTTPS.
- **Data Isolation:** Object-level ownership is strictly enforced—users can only list, view, and download their own jobs.

### HTTP Usage Examples

Sample test files are available in the [`examples/`](examples/) directory:
- [`examples/sample.csv`](examples/sample.csv)
- [`examples/sample.json`](examples/sample.json)

#### 1. Ingest a CSV File
```bash
curl -u devuser:devpass123 \
  -F "file=@examples/sample.csv" \
  http://127.0.0.1:8001/api/v1/jobs/
```

**Response (`201 Created`):**
```json
{
  "id": "0811b650-4c2e-4c1d-8059-a0ea2707a483",
  "file_type": "csv",
  "state": "completed",
  "original_filename": "sample.csv",
  "input_row_count": 3,
  "output_row_count": 3,
  "validation_errors": null,
  "column_names": ["id", "name", "department", "badge_number"],
  "download_url": "http://127.0.0.1:8001/api/v1/jobs/0811b650-4c2e-4c1d-8059-a0ea2707a483/download/",
  "created_at": "2026-11-09T05:58:33.272298Z",
  "updated_at": "2026-11-09T05:58:33.272321Z"
}
```

#### 2. Ingest a JSON Dataset
```bash
curl -u devuser:devpass123 \
  -F "file=@examples/sample.json" \
  http://127.0.0.1:8001/api/v1/jobs/
```

#### 3. List Ingestion Jobs
```bash
curl -u devuser:devpass123 \
  http://127.0.0.1:8001/api/v1/jobs/
```

#### 4. Download Normalized Output
```bash
curl -u devuser:devpass123 \
  -O -J \
  http://127.0.0.1:8001/api/v1/jobs/<job-id>/download/
```

---

## Local Development Setup (Host Python + Docker PostgreSQL)

This workflow runs PostgreSQL in Docker while running Python/Django directly on your Windows or Unix host.

### Prerequisites
- Python 3.12+
- Docker (for local PostgreSQL container)

### 1. Environment Setup

Copy the example environment configuration:

**Windows PowerShell:**
```powershell
Copy-Item .env.example .env
```

**Linux / macOS / Bash:**
```bash
cp .env.example .env
```

### 2. Start PostgreSQL Container

Run the dedicated database service. The database port is bound to `127.0.0.1:5432` by default:
```bash
docker compose up -d db
```

### 3. Create Virtual Environment and Install Dependencies

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

**Linux / macOS / Bash:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
```

### 4. Run Database Migrations

**Windows PowerShell:**
```powershell
$env:PYTHONPATH="src"
python manage.py migrate
```

**Linux / macOS / Bash:**
```bash
PYTHONPATH=src python manage.py migrate
```

### 5. Run Local Development Server

**Windows PowerShell:**
```powershell
$env:PYTHONPATH="src"
python manage.py runserver 127.0.0.1:8000
```

**Linux / macOS / Bash:**
```bash
PYTHONPATH=src python manage.py runserver 127.0.0.1:8000
```

---

## Workflow Engine API (M0-003)

FlowForge provides a synchronous, deterministic workflow execution engine that runs directly against completed ingestion jobs (`ProcessingJob`).

### Core Design & Guarantees
- **Pure Python Core:** The pipeline transformation engine (`flowforge.workflows.engine`) contains zero database or HTTP dependencies and is independently testable.
- **Strict Determinism:** Transformations operate predictably without arbitrary eval/exec code or unconstrained expressions.
- **Snapshot Immutability:** When a workflow is executed, an immutable snapshot of the definition's steps is stored directly in the `WorkflowRun`. Subsequent mutations to the definition do not retroactively alter execution history.
- **Audit Durability:** Each step execution produces an ordered audit record in `WorkflowStepRun` tracking step index, operation, row counts, and status (`completed` or `failed`). If a step fails, the failure is recorded, previous step audits are preserved, and partial results are never marked as completed.
- **Resource Bounds:** Maximum 10 steps per workflow, maximum 1,000 input/output rows, and maximum 128 characters for names and columns.
- **Source Protection:** Running a workflow never alters or mutates the source `ProcessingJob.output_data`.

### Supported Transformations
1. `select_columns`: Filters and orders columns in output rows.
   ```json
   {"op": "select_columns", "columns": ["id", "username", "role"]}
   ```
2. `rename_columns`: Renames existing columns while checking for collisions and preserving defined order.
   ```json
   {"op": "rename_columns", "mapping": {"username": "handle"}}
   ```
3. `filter_equals`: Performs strict, type-sensitive scalar comparison (`string`, `int`, `float`, `bool`, or `null`).
   ```json
   {"op": "filter_equals", "column": "role", "value": "admin"}
   ```
   - **Type Sensitivity Policy:** In standard Python, `True == 1` and `False == 0` evaluate to True. FlowForge's engine strictly prevents this: boolean `True`/`False` values never match numeric `1`/`0`. Numeric comparison between `int` and `float` is supported (e.g. `1 == 1.0`). `null` (None) strictly matches only `null`. String values preserve exact formatting, leading zeros, and casing.
   - **Empty Result & Schema Propagation Policy:** If an intermediate step filters out all records (yielding 0 rows), subsequent steps **continue to validate and transform their column schemas deterministically**. For example, referencing a nonexistent column in a `select_columns` or `rename_columns` step after an empty filter will cleanly fail the workflow run and record step audit history through that failure point. Non-empty and empty pipelines both maintain deterministic column tracking.

### Workflow Execution Example (cURL)

**PowerShell:**
```powershell
# 1. Define a workflow
$wfBody = @{
    name = "Filter Admins"
    steps = @(
        @{ op = "select_columns"; columns = @("id", "username", "role") },
        @{ op = "rename_columns"; mapping = @{ username = "handle" } },
        @{ op = "filter_equals"; column = "role"; value = "admin" }
    )
} | ConvertTo-Json -Depth 5

$wf = Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/v1/workflows/" `
    -Method Post -Headers @{ Authorization = "Basic " + [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("devuser:devpass123")) } `
    -ContentType "application/json" -Body $wfBody

# 2. Execute against completed job
$runBody = @{ source_job_id = "<job_uuid>" } | ConvertTo-Json
$run = Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/v1/workflows/$($wf.id)/runs/" `
    -Method Post -Headers @{ Authorization = "Basic " + [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("devuser:devpass123")) } `
    -ContentType "application/json" -Body $runBody

# 3. Retrieve run details & step history
Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/v1/workflow-runs/$($run.id)/" `
    -Method Get -Headers @{ Authorization = "Basic " + [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("devuser:devpass123")) }
```

**Bash / Linux / macOS:**
```bash
# 1. Define a workflow
WF_ID=$(curl -s -X POST http://127.0.0.1:8001/api/v1/workflows/ \
  -u devuser:devpass123 \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Filter Admins",
    "steps": [
      {"op": "select_columns", "columns": ["id", "username", "role"]},
      {"op": "rename_columns", "mapping": {"username": "handle"}},
      {"op": "filter_equals", "column": "role", "value": "admin"}
    ]
  }' | jq -r .id)

# 2. Execute against completed job
RUN_ID=$(curl -s -X POST http://127.0.0.1:8001/api/v1/workflows/${WF_ID}/runs/ \
  -u devuser:devpass123 \
  -H "Content-Type: application/json" \
  -d '{"source_job_id": "<job_uuid>"}' | jq -r .id)

# 3. Retrieve run details & step history
curl -s -X GET http://127.0.0.1:8001/api/v1/workflow-runs/${RUN_ID}/ \
  -u devuser:devpass123 | jq .
```

---

## Asynchronous Workflow Execution with Celery & Redis (M0-004)

FlowForge provides opt-in asynchronous execution of workflows using Celery and Redis while strictly preserving the synchronous execution endpoint unchanged.

```mermaid
flowchart LR
    Client["Client / API Caller"] -->|"POST /workflows/{id}/runs/async/"| API["FlowForge Web API"]
    API -->|"1. Validate ownership & status"| DB[("PostgreSQL")]
    API -->|"2. Create WorkflowRun (queued)"| DB
    API -->|"3. transaction.on_commit(delay)"| Broker[("Redis Broker (Internal)")]
    API -->|"4. HTTP 202 Accepted"| Client
    Worker["Celery Worker"] -->|"5. Consume task"| Broker
    Worker -->|"6. Atomic claim (queued -> running)"| DB
    Worker -->|"7. Execute step_snapshot"| Engine["Workflow Pipeline Engine"]
    Worker -->|"8. Persist completed/failed & audits"| DB
    Client -->|"9. GET /workflow-runs/{id}/"| API
```

### Architecture & Safety Invariants
- **Opt-in Endpoint:** Asynchronous execution is requested via `POST /api/v1/workflows/{uuid}/runs/async/`. It returns HTTP `202 Accepted` with state `"queued"` and status polling URL.
- **Transactional Enqueue:** Tasks are dispatched only after transaction commit using `django.db.transaction.on_commit`.
- **Minimal Celery Payloads:** Celery messages carry strictly the database `run_id` (never file content or user payloads).
- **Worker Claim Atomicity:** Workers conditionally claim runs (`UPDATE ... WHERE state='queued'`). If already claimed or processed, redundant delivery does not duplicate work.
- **Snapshot Execution:** Workers execute against the immutable `step_snapshot` persisted on `WorkflowRun`, ensuring subsequent mutations to `WorkflowDefinition` do not affect in-flight runs.
- **Isolated Redis Network:** Redis in `docker-compose.yml` does **not** expose host ports; it is accessible solely across the internal Docker Compose bridge network.

### cURL / PowerShell Examples

#### Enqueue Asynchronous Workflow Run
```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/workflows/<workflow_uuid>/runs/async/ \
  -u devuser:devpass123 \
  -H "Content-Type: application/json" \
  -d '{"source_job_id": "<source_job_uuid>"}' | jq .
```

**Response (`202 Accepted`):**
```json
{
  "id": "c1f77d54-159e-4c57-814c-e83cb2b6a506",
  "workflow_definition": "8eab8562-99b7-4b66-ba65-71694096fe00",
  "source_job": "6e1cf1d0-8695-494b-ba84-f6cbcd2f040f",
  "state": "queued",
  "status_url": "/api/v1/workflow-runs/c1f77d54-159e-4c57-814c-e83cb2b6a506/",
  "created_at": "2026-11-09T09:40:00.123456Z"
}
```

#### Poll Workflow Run Status
```bash
curl -s -X GET http://127.0.0.1:8001/api/v1/workflow-runs/<run_uuid>/ \
  -u devuser:devpass123 | jq .
```

```powershell
# PowerShell
Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/v1/workflow-runs/<run_uuid>/" `
  -Headers @{ Authorization = "Basic " + [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("devuser:devpass123")) }
```

---

## Testing & Quality Gates

Run the test suite:
```powershell
# Windows PowerShell
$env:PYTHONPATH="src"
pytest -v

# Linux / Bash
PYTHONPATH=src pytest -v
```

Run code formatting and lint checks:
```bash
ruff check .
ruff format --check .
```

Run dependency security audit:
```bash
pip-audit -r requirements.txt
```

---

## API Documentation

### Health Check Endpoint
- **URL:** `/api/v1/health/`
- **Method:** `GET`
- **Authentication:** None required

#### Success Response (`200 OK`):
```json
{
  "status": "ok",
  "version": "0.1.0",
  "timestamp": "2026-11-08T21:23:31.633401+00:00",
  "checks": {
    "database": "healthy"
  }
}
```

#### Failure Response (`503 Service Unavailable`):
```json
{
  "status": "unhealthy",
  "version": "0.1.0",
  "timestamp": "2026-11-08T21:23:31.633401+00:00",
  "checks": {
    "database": "unhealthy"
  }
}
```

---

## Troubleshooting

- **Port Conflict (`listen tcp 127.0.0.1:8000: bind`):**
  If port 8000 is occupied on your host, `docker-compose.yml` defaults host port mapping to `127.0.0.1:8001:8000`. You can customize the exposed port using `PORT=8005 docker compose up -d`.
- **PostgreSQL Port Binding:**
  By default, `docker-compose.yml` binds PostgreSQL to `127.0.0.1:5432` (`${DB_HOST_BIND:-127.0.0.1}`). If port 5432 is occupied on the host, set `HOST_DB_PORT=5433` in your `.env`.
- **Database Connection Refused:**
  Ensure the `db` service is healthy using `docker compose ps`. When running tests locally outside Docker, ensure `db` is running on port 5432.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
