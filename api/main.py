
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from api.dependencies import get_workflow


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
        return EvidenceResponse.model_validate(result)
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Evidence workflow failed",
        ) from None
