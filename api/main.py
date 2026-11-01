
import json
import os
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from api.dependencies import get_answerer, get_workflow
from api.public_evidence import public_evidence


TELEMETRY_LOG = Path(os.getenv("TELEMETRY_LOG", Path(__file__).resolve().parents[1] / "derived" / "telemetry.jsonl"))

app = FastAPI(
    title="Vienna Stay Assistant API",
    version="0.1.0",
)


class EvidenceRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value


class EvidenceResponse(BaseModel):
    status: str
    results: list[Any]


class AnswerRequest(EvidenceRequest):
    new_years_eve_year: int | None = Field(
        default=None, ge=2026, le=2030,
        description="Year meant by 'New Year's Eve' when the question gives none",
    )


class CheckedStatement(BaseModel):
    sentence: str
    citations: list[str]
    evidence: list[str]


class RemovedStatement(BaseModel):
    sentence: str
    reason: str | None


class ReviewEvidence(BaseModel):
    review_id: int
    listing_id: int
    review_date: str
    language: str | None = None
    scope: str
    text: str


class AnswerResponse(BaseModel):
    status: str
    answer: str | None = None
    verified: bool = False
    pii_redacted: bool = False
    citations: list[CheckedStatement] = []
    removed: list[RemovedStatement] = []
    reviews: list[ReviewEvidence] = []
    evidence: list[Any] = []
    notices: list[str] = []
    message: str | None = None
    guard_category: str | None = Field(default=None, description="Input guard that blocked the question")
    guard_reason: str | None = None
    masked_question: str | None = Field(default=None, description="The question as the model saw it")
    timings: dict[str, float | int | None] = Field(
        default={}, description="Seconds per stage, time to first answer token (ttft_s) and tokens_per_s")



@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/evidence", response_model=EvidenceResponse)
def gather_evidence(
    request: EvidenceRequest,
    workflow=Depends(get_workflow),
):
    try:
        result = workflow.invoke(request.question)
        result["results"] = public_evidence(result["results"])
        return EvidenceResponse.model_validate(result)
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Evidence workflow failed",
        ) from None


def to_response(result):
    """Turn a pipeline result into the public AnswerResponse (no SQL, internal IDs or raw rows)."""
    if "clarification" in result:
        return AnswerResponse(status="clarification_needed", message=result["clarification"])
    results = result.get("results", [])
    common = {
        "evidence": public_evidence(results),
        "notices": list(dict.fromkeys(
            n for item in results if isinstance(item, dict) for n in item.get("notices", []) if isinstance(n, str))),
        "masked_question": result.get("masked_question"),
        "timings": result.get("timings", {}),
    }
    if "answer" not in result:
        return AnswerResponse(status=result["status"], message=result.get("message"),
                              guard_category=result.get("category"), guard_reason=result.get("reason"), **common)
    answer = result["answer"]
    return AnswerResponse(
        **common,
        status=result["status"],
        answer=answer["answer"],
        verified=answer["verified"],
        pii_redacted=answer["pii_redacted"],
        citations=answer["citations"],
        removed=answer["removed"],
        reviews=[{key: review.get(key) for key in ReviewEvidence.model_fields} for review in result["reviews"]],
    )


def log_request(response, mode):
    """Append one telemetry line per answered request: status, guards and timings (no question text)."""
    record = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "mode": mode,
        "status": response.status, "guard_category": response.guard_category, "verified": response.verified,
        "statements_kept": len(response.citations), "statements_removed": len(response.removed),
        "pii_redacted": response.pii_redacted, "model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        **response.timings,
    }
    try:
        TELEMETRY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with TELEMETRY_LOG.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record) + "\n")
    except OSError:
        pass  # telemetry must never break an answer


def year_text(request):
    return str(request.new_years_eve_year) if request.new_years_eve_year else None


@app.post("/answer", response_model=AnswerResponse)
def answer_question(request: AnswerRequest, run_question=Depends(get_answerer)):
    """Answer a question from guarded evidence; every sentence is checked by the output guard.

    Slow on CPU (minutes): evidence search, answer generation and the output guard all run locally.
    """
    try:
        response = to_response(run_question(request.question, year_text(request)))
    except Exception:
        raise HTTPException(status_code=502, detail="Answer pipeline failed") from None
    log_request(response, "answer")
    return response


@app.post("/answer/stream")
def answer_question_stream(request: AnswerRequest, run_question=Depends(get_answerer)):
    """Same as /answer, as server-sent events while it runs.

    Events: "stage" {name, label} when a pipeline stage starts, "token" {text} for each piece of the draft
    answer (not yet checked by the output guard), then "final" (the AnswerResponse) or "error" {detail}.
    """
    events = queue.Queue()

    def work():
        try:
            result = run_question(request.question, year_text(request),
                                  on_event=lambda kind, **data: events.put((kind, data)))
            response = to_response(result)
            log_request(response, "stream")
            events.put(("final", response.model_dump()))
        except Exception:
            events.put(("error", {"detail": "Answer pipeline failed"}))
        events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while (item := events.get()) is not None:
            kind, data = item
            yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
