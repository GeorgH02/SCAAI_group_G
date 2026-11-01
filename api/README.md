# Vienna Stay Assistant — FastAPI Backend

REST API providing HTTP access to the Vienna Stay Assistant's existing `EvidenceWorkflow`.

The API handles request validation, workflow initialization, database connections, error handling, and structured evidence responses.

## Endpoints

### GET /health

Checks whether the API server is running.

Response:
```json
{"status": "ok"}
```

### POST /evidence

Accepts an accommodation-related question and executes the evidence-gathering workflow.

Request:
```json
{
  "question": "Find apartments for 2 guests in Vienna"
}
```

The response contains the workflow status and a restricted set of evidence fields. Internal SQL statements, query parameters, and ingestion identifiers are excluded.

The endpoint returns structured evidence rather than a generated conversational answer.

## Setup

Install the project and API dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-api.txt
```

PostgreSQL and Ollama must be running, and the accommodation data must be loaded before using the evidence endpoint.

## Running the API

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Interactive API documentation is available at:

http://127.0.0.1:8000/docs

## Testing

```bash
python -m unittest discover -s tests -p "test_api.py" -v
```

The API tests use mocked workflows and do not require a running language model.

## Current limitations

- Accommodation evidence is based on the September 17, 2026 Inside Airbnb snapshot, not live booking availability.
- Initial model loading may require substantial system memory.
- The API is currently configured for local development.
- Additional security validation is required before public deployment.