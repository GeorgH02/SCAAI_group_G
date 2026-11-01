# start docker compose
# start venv311
# install requirements
# run run_pipeline.ipynb (adjust limit!)
# docker compose up -d --wait --wait-timeout 120 ollama
# docker compose exec -T ollama ollama pull qwen2.5:7b
# .\.venv311\Scripts\python.exe -m streamlit run app1.py

import psycopg
import streamlit as st
from openai import APIError
from qdrant_client.http.exceptions import UnexpectedResponse

from assistant.pipeline import SNAPSHOT_DATE, run_question
from langchain_agent.workflow import WorkflowError


def show_output_guard(answer):
    if answer.get("pii_redacted"):
        st.warning("Personal information was found in the generated answer and has been masked.")
    if answer.get("citations"):
        with st.expander(f"Citations ({len(answer['citations'])} verified statements)"):
            for item in answer["citations"]:
                st.markdown(f"**{item['sentence']}**")
                for premise in item["evidence"]:
                    st.caption(premise)
    if answer.get("removed"):
        with st.expander(f"Removed by output guard ({len(answer['removed'])})"):
            st.dataframe(answer["removed"], use_container_width=True, hide_index=True)


st.set_page_config(page_title="Vienna Stay Assistant", page_icon="🏠")
st.title("Vienna Stay Assistant")
st.caption("Ask about listings, snapshot prices, calendar evidence, and guest reviews.")
st.warning(
    "Prototype: answers are generated locally from retrieved evidence. Each sentence is checked against "
    "the evidence it cites (NLI and numbers) and removed if unsupported; the checks can still miss errors."
)

with st.sidebar:
    st.subheader("New Year's Eve year")
    new_years_eve_year = st.selectbox(
        "New Year's Eve year",
        options=["Choose a year", *(str(year) for year in range(SNAPSHOT_DATE.year, SNAPSHOT_DATE.year + 5))],
    )
    new_years_eve_year = (
        None if new_years_eve_year == "Choose a year" else new_years_eve_year
    )
    st.caption(
        "This selector applies only to New Year's Eve questions without a year. "
        "Availability and prices come from a historical snapshot, not live booking data."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("answer"):
            show_output_guard(message["answer"])
        if message.get("reviews"):
            with st.expander("Review evidence"):
                st.dataframe(message["reviews"], use_container_width=True, hide_index=True)
        if message.get("status") and message["status"] != "evidence_ready":
            st.caption(f"Evidence workflow status: {message['status']}")
        if message.get("details"):
            st.info(message["details"])

st.caption("Each message is searched as a separate question; earlier chat turns are displayed but not used as context.")
question = st.chat_input("Ask a complete question about Vienna stays")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Checking guarded evidence and preparing a grounded summary..."):
                result = run_question(question, new_years_eve_year)
            if "clarification" in result:
                st.info(result["clarification"])
                response = result["clarification"]
                st.session_state.messages.append({"role": "assistant", "content": response})
            elif "answer" in result:
                st.markdown(result["answer"]["answer"])
                st.caption("AI-generated summary. Each sentence was checked against the evidence it cites.")
                show_output_guard(result["answer"])
                sample_notices = sorted({
                    notice
                    for item in result["results"]
                    for notice in item.get("notices", [])
                    if "sample" in notice.casefold()
                })
                for notice in sample_notices:
                    st.warning(notice)
                if result["reviews"]:
                    with st.expander("Review evidence"):
                        st.dataframe(result["reviews"], use_container_width=True, hide_index=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"]["answer"],
                    "answer": result["answer"],
                    "reviews": result["reviews"],
                    "status": result["status"],
                })
            else:
                status = result["status"]
                if status == "blocked":
                    response = result.get("message") or "The question was declined by an input guard."
                    if result.get("reason"):
                        response += f" (Guard: {result['reason']})"
                elif status == "no_evidence":
                    response = (
                        "No matching evidence was found in the loaded data. The database/vector store "
                        "may contain only a sample; check the import scope and requested dates."
                    )
                elif status == "blocked_evidence":
                    response = "Potentially unsafe text was found in retrieved evidence, so it was not used."
                elif status == "budget_exhausted":
                    response = "The search reached its tool-call limit before collecting enough evidence. Try a narrower question."
                elif status == "context_limit":
                    response = "The search returned too much evidence to process safely. Try narrowing the question."
                else:
                    response = f"The evidence workflow stopped with status: {status}."
                sample_notices = sorted({
                    notice
                    for item in result.get("results", [])
                    for notice in item.get("notices", [])
                    if "sample" in notice.casefold()
                })
                if sample_notices:
                    response += "\n\n" + " ".join(sample_notices)
                st.info(response)
                st.session_state.messages.append({
                    "role": "assistant", "content": response, "status": status, "details": response,
                })
        except psycopg.Error:
            st.error("PostgreSQL could not complete the request. Check that the database is running and the pipeline has loaded its schema.")
        except WorkflowError as error:
            st.error(str(error))
        except APIError:
            st.error("Ollama did not complete the model request. Check the Ollama service and that qwen2.5:7b is available.")
        except UnexpectedResponse:
            st.error("Qdrant rejected the review search. Check that the review collection exists and matches the current pipeline.")
        except ValueError as error:
            st.error(str(error))
        except (OSError, RuntimeError) as error:
            st.error(str(error))

# start venv311 (not venv!)
# .\.venv311\Scripts\python.exe -m pip install --default-timeout=600 --retries 10 -r requirements.txt
# .\.venv311\Scripts\python.exe -m streamlit run app1.py