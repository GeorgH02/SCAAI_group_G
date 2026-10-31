
"""Dependencies for the FastAPI evidence endpoint."""

import os
from functools import lru_cache

from fastapi import HTTPException


@lru_cache(maxsize=1)
def get_resources():
    """Load shared AI resources once per API process."""
    from data_setup.load_vectors import Masker
    from guardrails.input_guard import InputGuard, bge_m3_embedder, llama_guard
    from langchain_agent.workflow import local_model

    masker = Masker()
    guard = InputGuard(
        embed=bge_m3_embedder(),
        safety=llama_guard() if os.getenv("LLAMA_GUARD") == "1" else None,
    )
    model = local_model(timeout=400)

    return masker, guard, model


def get_workflow():
    """Create a request-scoped workflow and close its database connection."""
    connection = None

    try:
        from data_setup.load_vectors import clean_text
        from guardrails.evidence_guard import find_injection, remove_hostile_rows
        from langchain_agent.workflow import EvidenceWorkflow, shield_names
        from try_agent_db import connect

        masker, guard, model = get_resources()
        connection = connect()

        landmarks = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM vienna_stay_assistant.landmarks"
            )
        ]

        workflow = EvidenceWorkflow(
            connection=connection,
            model=model,
            mask=shield_names(
                lambda text: masker.mask(clean_text(text)),
                landmarks,
            ),
            input_guard=guard,
            evidence_guard=lambda text: find_injection(text) is None,
            clean_evidence=remove_hostile_rows,
        )

    except Exception:
        if connection is not None:
            connection.close()
        raise HTTPException(
            status_code=503,
            detail="Evidence service unavailable",
        ) from None

    try:
        yield workflow
    finally:
        connection.close()
