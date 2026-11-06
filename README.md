# FlowForge

[![CI](https://github.com/amitworx/flowforge/actions/workflows/ci.yml/badge.svg)](https://github.com/amitworx/flowforge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FlowForge is an open-source, self-hostable Python document-processing and workflow automation platform. Its core is designed to run reliably without requiring any LLM or paid cloud AI services.

---

## Architectural Overview

- **Core Engine:** Django 5.2 (LTS) & Django REST Framework (DRF)
- **Database:** PostgreSQL 17 (via `psycopg` 3)
- **Configuration:** Typed configuration via `pydantic-settings`
- **Linting & Formatting:** Ruff
- **Test Suite:** pytest with `pytest-django`

### API Layout
- Versioned API base: `/api/v1/`
- Health check endpoint: `GET /api/v1/health/`

---

## Quick Start with Docker Compose (Recommended)

Docker Compose provides a reproducible environment that starts both the PostgreSQL database and the web application without requiring PostgreSQL or Python installed on the host.

### Prerequisites
- Docker Engine & Docker Compose (v2+)

### Run Application and Database

```bash
docker compose up -d --build
```

The application will be accessible at:
- **Health Check:** `http://localhost:8001/api/v1/health/` (or port configured via `PORT` variable)

To view running containers and logs:
```bash
docker compose ps
docker compose logs -f web
```

To stop the containers:
```bash
docker compose down
```

---

## Local Development Setup

### Prerequisites
- Python 3.12+
- Docker (for local PostgreSQL container)

### 1. Environment Setup

Copy the example environment configuration:
```bash
# On Linux / macOS / Bash
cp .env.example .env

# On Windows PowerShell
Copy-Item .env.example .env
```

### 2. Start PostgreSQL Container

Run the dedicated database service with persistent volume:
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
python manage.py runserver 0.0.0.0:8000
```

**Linux / macOS / Bash:**
```bash
PYTHONPATH=src python manage.py runserver 0.0.0.0:8000
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
  "timestamp": "2026-11-06T06:09:16.347132+00:00",
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
  "timestamp": "2026-11-06T06:09:16.347132+00:00",
  "checks": {
    "database": "unhealthy"
  }
}
```

> **Note on Health Check Scope:** The endpoint checks database query responsiveness (`SELECT 1;`). It does not verify external networks or asynchronous message queues as those are outside the foundation scope.

---

## Troubleshooting

- **Port Conflict (`listen tcp 0.0.0.0:8000: bind`):**
  If port 8000 is occupied on your host machine, `docker-compose.yml` defaults host port mapping to `8001:8000`. You can customize the exposed port using `PORT=<number> docker compose up -d`.
- **Database Connection Refused:**
  Ensure the `db` service is healthy using `docker compose ps`. When running tests locally outside Docker, ensure `flowforge-db` is running on port 5432.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
