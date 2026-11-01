"""Streamlit chat for the Vienna Stay Assistant: a thin client of the FastAPI backend.

All models run behind the API, so this process loads none of them. Answers stream from
POST /answer/stream: pipeline stages and the draft answer appear live, then the guarded answer replaces it.
Start the API first:   python -m uvicorn api.main:app --port 8000
Then the chat:         python -m streamlit run app1.py
Set ANSWER_API_URL if the API runs elsewhere.
"""

import json
import os

import httpx
import streamlit as st

ANSWER_API_URL = os.getenv("ANSWER_API_URL", "http://127.0.0.1:8000/answer")
SNAPSHOT_YEAR = 2026
TIMEOUT_SECONDS = 1800  # CPU-only runs can take several minutes
PIPELINE_FAILED = ("The answer pipeline failed. Check that PostgreSQL, Qdrant and Ollama are running "
                   "and see the API log.")

STATUS_TEXT = {
    "no_evidence": "No matching evidence was found in the loaded data. Try a broader question or other dates.",
    "blocked_evidence": "Potentially unsafe text was found in the retrieved evidence, so it was not used.",
    "budget_exhausted": "The search reached its tool-call limit before collecting enough evidence. "
                        "Try a narrower question.",
    "context_limit": "The search returned too much evidence to process safely. Try a narrower question.",
}


def ask(question, new_years_eve_year, on_stage=None, on_token=None):
    """Stream the answer from the API; returns (final response body, error text).

    on_stage(data) is called when a pipeline stage starts, on_token(text) for each draft answer piece.
    """
    body = {"question": question}
    if new_years_eve_year:
        body["new_years_eve_year"] = int(new_years_eve_year)
    try:
        with httpx.stream("POST", ANSWER_API_URL.rstrip("/") + "/stream", json=body,
                          timeout=httpx.Timeout(TIMEOUT_SECONDS, connect=10)) as response:
            if response.status_code == 422:
                return None, "The question was not accepted (it must contain 1–4,000 characters)."
            if response.status_code != 200:
                return None, PIPELINE_FAILED
            event = None
            for line in response.iter_lines():
                if line.startswith("event: "):
                    event = line[len("event: "):]
                elif line.startswith("data: "):
                    data = json.loads(line[len("data: "):])
                    if event == "stage" and on_stage:
                        on_stage(data)
                    elif event == "token" and on_token:
                        on_token(data["text"])
                    elif event == "final":
                        return data, None
                    elif event == "error":
                        return None, PIPELINE_FAILED
    except httpx.ConnectError:
        return None, (f"The answer API is not reachable at {ANSWER_API_URL}. "
                      "Start it with `python -m uvicorn api.main:app --port 8000`.")
    except httpx.TimeoutException:
        return None, "The answer API did not respond in time."
    return None, "The answer stream ended without a result."


def timing_line(timings):
    """Latency summary shown under each answer (also logged by the API)."""
    parts = []
    for key, label in (("total_s", "total"), ("ttft_s", "first token"), ("evidence_s", "evidence search"),
                       ("reviews_s", "review search"), ("output_guard_s", "output guard")):
        if timings.get(key) is not None:
            parts.append(f"{label} {timings[key]:.1f} s")
    if timings.get("tokens_per_s"):
        parts.append(f"{timings['tokens_per_s']:.1f} tokens/s")
    return "⏱ " + " · ".join(parts) if parts else ""


def guard_status(result, question):
    """One line per guard, so it is visible which protection acted."""
    lines = []
    status = result.get("status")
    if status == "blocked":
        lines.append(f"⛔ **Input guard blocked the question** — {result.get('guard_reason') or 'policy'} "
                     f"(`{result.get('guard_category') or 'blocked'}`)")
    elif status != "clarification_needed":
        lines.append("✅ Input guard: passed (injection rules, topic gate)")
    masked = result.get("masked_question")
    if masked and masked.strip() != question.strip():
        lines.append(f"🔒 PII masked before the model saw the question: _{masked}_")
    if status == "blocked_evidence":
        lines.append("⛔ Evidence guard: retrieved text looked like instructions to the model")
    if result.get("answer") is not None:
        if result.get("verified"):
            lines.append(f"✅ Output guard: {len(result.get('citations', []))} statement(s) verified against "
                         "their evidence (NLI + numbers)")
        else:
            lines.append("⛔ Output guard: no statement could be verified, so the answer was withheld")
        if result.get("removed"):
            lines.append(f"✂️ Output guard removed {len(result['removed'])} unsupported statement(s)")
        if result.get("pii_redacted"):
            lines.append("🔒 Output guard masked personal data in the answer")
    return lines


def show_details(result):
    for notice in result.get("notices", []):
        if any(word in notice.casefold() for word in ("sample", "quietest", "limit")):
            st.warning(notice)
    if result.get("citations"):
        with st.expander(f"Citations ({len(result['citations'])} verified statements)"):
            for item in result["citations"]:
                st.markdown(f"**{item['sentence']}**")
                for premise in item["evidence"]:
                    st.caption(premise)
    if result.get("removed"):
        with st.expander(f"Removed by output guard ({len(result['removed'])})"):
            st.dataframe(result["removed"], width="stretch", hide_index=True)
    if result.get("reviews"):
        with st.expander(f"Review evidence ({len(result['reviews'])})"):
            st.dataframe(result["reviews"], width="stretch", hide_index=True)
    if result.get("evidence"):
        with st.expander("Database evidence"):
            st.json(result["evidence"], expanded=False)


def render(message):
    st.markdown(message["content"])
    result = message.get("result")
    if result:
        for line in message.get("guards", []):
            st.markdown(line)
        if timing_line(result.get("timings") or {}):
            st.caption(timing_line(result["timings"]))
        show_details(result)


st.set_page_config(page_title="Vienna Stay Assistant", page_icon="🏠")
st.title("Vienna Stay Assistant")
st.caption("Ask about listings, snapshot prices, calendar evidence, and guest reviews. "
           "Data: Inside Airbnb Vienna snapshot (CC BY 4.0) and the Austrian 2022 strategic noise map.")
st.warning(
    "Prototype: answers are generated locally from retrieved evidence. Each sentence is checked against "
    "the evidence it cites (NLI and numbers) and removed if unsupported; the checks can still miss errors."
)

with st.sidebar:
    st.subheader("New Year's Eve year")
    new_years_eve_year = st.selectbox(
        "New Year's Eve year",
        options=["Choose a year", *(str(year) for year in range(SNAPSHOT_YEAR, SNAPSHOT_YEAR + 5))],
    )
    new_years_eve_year = None if new_years_eve_year == "Choose a year" else new_years_eve_year
    st.caption(
        "This selector applies only to New Year's Eve questions without a year. "
        "Availability and prices come from a historical snapshot, not live booking data."
    )
    st.caption(f"API: {ANSWER_API_URL}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render(message)

st.caption("Each message is searched as a separate question; earlier chat turns are displayed but not used as context.")
question = st.chat_input("Ask a complete question about Vienna stays")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        progress = st.status("Sending the question to the API...", expanded=True)
        draft_box, draft = st.empty(), []

        def on_stage(data):
            progress.update(label=data["label"] + "...")
            progress.write("• " + data["label"])

        def on_token(text):
            draft.append(text)
            draft_box.markdown("_Draft, not yet checked:_\n\n" + "".join(draft) + " ▌")

        result, error = ask(question, new_years_eve_year, on_stage, on_token)
        draft_box.empty()
        if error:
            progress.update(label="Failed", state="error", expanded=False)
            st.error(error)
        else:
            total = (result.get("timings") or {}).get("total_s")
            progress.update(label=f"Done in {total:.0f} s" if total else "Done", state="complete", expanded=False)
            if result.get("answer") is not None:
                content = result["answer"]
            elif result["status"] in ("blocked", "clarification_needed"):
                content = result.get("message") or "The question was declined by an input guard."
            else:
                content = STATUS_TEXT.get(result["status"], f"The search stopped with status: {result['status']}.")
            message = {"role": "assistant", "content": content, "result": result,
                       "guards": guard_status(result, question)}
            render(message)
            st.session_state.messages.append(message)
