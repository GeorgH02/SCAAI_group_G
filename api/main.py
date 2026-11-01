
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from api.dependencies import get_answerer, get_workflow
from api.public_evidence import public_evidence


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
    message: str | None = None



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


@app.post("/answer", response_model=AnswerResponse)
def answer_question(request: AnswerRequest, run_question=Depends(get_answerer)):
    """Answer a question from guarded evidence; every sentence is checked by the output guard.

    Slow on CPU (minutes): evidence search, answer generation and the output guard all run locally.
    """
    try:
        result = run_question(
            request.question,
            str(request.new_years_eve_year) if request.new_years_eve_year else None,
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Answer pipeline failed") from None
    if "clarification" in result:
        return AnswerResponse(status="clarification_needed", message=result["clarification"])
    evidence = public_evidence(result.get("results", []))
    if "answer" not in result:
        return AnswerResponse(status=result["status"], message=result.get("message"), evidence=evidence)
    answer = result["answer"]
    return AnswerResponse(
        status=result["status"],
        answer=answer["answer"],
        verified=answer["verified"],
        pii_redacted=answer["pii_redacted"],
        citations=answer["citations"],
        removed=answer["removed"],
        reviews=[{key: review.get(key) for key in ReviewEvidence.model_fields} for review in result["reviews"]],
        evidence=evidence,
    )
