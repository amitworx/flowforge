# FlowForge

[![CI](https://github.com/amitworx/flowforge/actions/workflows/ci.yml/badge.svg)](https://github.com/amitworx/flowforge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FlowForge is an open-source, self-hostable Python document-processing and workflow automation platform. Its core is designed to run reliably without requiring any LLM or paid cloud AI services.

---

> [!WARNING]
> **Development vs. Production Notice:**
> The provided `docker-compose.yml`, `Dockerfile`, Django `runserver`, `.env.example`, and sample credentials (`SECRET_KEY`, `flowforge` database password) are **strictly for local development and testing only**.
> - In `production` or `staging` environments, the application **strictly rejects** the sample/insecure secret key, refuses to start if `DEBUG=True`, and rejects the default database password.
> - PostgreSQL in `docker-compose.yml` is hardened by default to bind only to `127.0.0.1` (localhost), preventing accidental exposure to the host network or public internet.
> - Never deploy to production using development configurations or sample secrets.

---

## Architectural Overview

- **Core Engine:** Django 5.2 (LTS) & Django REST Framework (DRF)
- **Database:** PostgreSQL 17 (via `psycopg` 3)
- **Configuration:** Typed configuration via `pydantic-settings` with strict environment-mode validation
- **Ingestion Engine:** Synchronous, deterministic CSV/JSON processing with safe bounding and normalization
- **Linting & Formatting:** Ruff
- **Test Suite:** pytest with `pytest-django`

### API Layout
- Versioned API base: `/api/v1/`
- Health check endpoint: `GET /api/v1/health/`
- Ingestion jobs endpoint: `POST /api/v1/jobs/` (Authenticated)
- List user jobs: `GET /api/v1/jobs/` (Authenticated)
- Retrieve job metadata: `GET /api/v1/jobs/{uuid}/` (Authenticated)
- Download normalized JSON: `GET /api/v1/jobs/{uuid}/download/` (Authenticated)

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

## Ingestion Core API (M0-002)

FlowForge provides deterministic ingestion for tabular data (CSV and JSON) without relying on external cloud AI or queues.

### Policy & Security Guardrails
- **File size cap:** 1 MiB maximum upload size (bounded stream reading).
- **Row count cap:** 1,000 records maximum per dataset.
- **Authentication:** All `/api/v1/jobs/` endpoints require authentication (Session or Basic Auth).
- **Data Isolation:** Enforced object-level ownership—users can only list, view, and download their own jobs.
- **CSV Handling:** UTF-8 encoded (UTF-8 BOM handled), mandatory header row, duplicate header detection, cell string preservation (including leading zeros).
- **JSON Handling:** UTF-8 array of uniform objects with scalar values (nested complex objects are rejected to avoid unconstrained depth/memory usage).

### HTTP Usage Examples

Sample test files are available in the [`examples/`](examples/) directory:
- [`examples/sample.csv`](examples/sample.csv)
- [`examples/sample.json`](examples/sample.json)

#### 1. Ingest a CSV File
```bash
curl -u <username>:<password> \
  -F "file=@examples/sample.csv" \
  http://127.0.0.1:8001/api/v1/jobs/
```

**Response (`201 Created`):**
```json
{
  "id": "af164e84-db1a-48e7-be0a-049a324ec3b8",
  "file_type": "csv",
  "state": "completed",
  "original_filename": "sample.csv",
  "input_row_count": 3,
  "output_row_count": 3,
  "validation_errors": null,
  "column_names": ["id", "name", "department", "badge_number"],
  "download_url": "http://127.0.0.1:8001/api/v1/jobs/af164e84-db1a-48e7-be0a-049a324ec3b8/download/",
  "created_at": "2026-11-08T21:26:23.632263Z",
  "updated_at": "2026-11-08T21:26:23.632280Z"
}
```

#### 2. Ingest a JSON Dataset
```bash
curl -u <username>:<password> \
  -F "file=@examples/sample.json" \
  http://127.0.0.1:8001/api/v1/jobs/
```

#### 3. List Ingestion Jobs
```bash
curl -u <username>:<password> \
  http://127.0.0.1:8001/api/v1/jobs/
```

#### 4. Download Normalized Output
```bash
curl -u <username>:<password> \
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
