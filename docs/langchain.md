# LangChain integration

Install the project dependencies, including LangChain, in a Python 3.11 environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Typed SQL tools (usable without a model)

```python
import os
import psycopg
from langchain_tools import build_tools

# Set DATABASE_URL to a dedicated SELECT-only login. Do not print it.
with psycopg.connect(os.environ['DATABASE_URL']) as connection:
    tools = {tool.name: tool for tool in build_tools(connection)}
    evidence = tools['search_listings'].invoke({
        'district': 'Neubau', 'accommodates': 4, 'max_price': '150',
    })
```

All five tools retain the [query contract](query_tools.md): fixed parameterized
SQL, read-only transactions, statement timeouts, exact source keys, provenance,
missing-price notices and sample warnings. Model schemas forbid extra arguments
and validate IDs, booleans and pagination. Monetary bounds are decimal strings.
SQL itself still validates dates, price ranges and stay length. The connection
belongs to the caller and must be idle; use a separate connection per request.
Execute tools sequentially on that connection. No arbitrary SQL tool is exposed.

## Local model evidence workflow

Use the Windows/Linux launchers in [local LLM setup](llama_server.md) to start
the inference server with model alias `vienna-stay`.

`EvidenceWorkflow` binds these tools to a LangChain chat model and executes
multi-step tool calls. `local_model(model=..., base_url=...)` uses ChatOpenAI's
compatible HTTP transport against an independently running **local llama.cpp**
server (`http://127.0.0.1:8080/v1` by default). The server/model must support tool
calling. No server, model weights, GPU configuration or cloud subscription is
installed by this integration. The helper rejects remote URLs and disables
environment proxies/redirects; configure the server's model alias explicitly.

Application wiring, once guard implementations exist:

```python
from load_vectors import Masker, clean_text
from langchain_workflow import EvidenceWorkflow, local_model

def gather_evidence(connection, question, *, model_alias, input_guard, evidence_guard):
    masker = Masker()  # Requires both existing spaCy models; failure aborts.
    workflow = EvidenceWorkflow(
        connection,
        local_model(model=model_alias),
        mask=lambda text: masker.mask(clean_text(text)),
        input_guard=input_guard,
        evidence_guard=evidence_guard,
    )
    return workflow.invoke(question)
```

Both guards are mandatory callables returning **exactly `True`** to permit
continuation. `input_guard` receives the locally masked question and must enforce
topic, injection and identity-request policy before the first model invocation.
`evidence_guard` receives redacted JSON and must detect indirect instructions
before evidence enters the next model invocation. These interfaces are integration
points, **not implemented classifiers**. Do not substitute unconditional approval
in an application. Masking errors or guard/model/tool exceptions abort with a safe
`WorkflowError`. Trusted injected components must not log or export their inputs,
register callbacks or perform remote inference. Masking remains statistical;
neither these interfaces nor the prompt establish zero PII leakage or jailbreak resistance.

Defaults: four model rounds, eight total tool calls, 60,000 context characters,
4,000 input characters, and 60 seconds per local model request with no retries.
Tools run sequentially. Request history is local to each invocation. LangSmith
tracing is disabled for the request and no callbacks or persistence are configured.
Tool argument strings and evidence copies are masked before model reuse. Original
SQL evidence artifacts retain exact keys/parameters; handle them as private
application data and do not log or send them directly to models. Injected models
and guards remain trusted application code.

The return shape is `{status, results}`. Status is `evidence_ready`, `no_evidence`,
`blocked`, `blocked_evidence`, `budget_exhausted`, or `context_limit`. Budget/context
stops can contain partial results and do not establish complete retrieval. Blocks
return no evidence. Model-authored prose is always discarded, including requests
for clarification; an application must handle ambiguous/unsupported requests.
There is no factual-answer generation, NLI, citation mapping, output PII check,
dense/BM25 search, text-to-SQL, public CLI/chat API or streaming in this layer.
The caller must supply those remaining pieces before public use.

Tests use actual LangChain schemas/messages and fake SQL/model boundaries. They
do not prove local server compatibility, live ingestion, guard efficacy, model
quality or end-to-end application operation. Optional tests skip if LangChain is
not installed; ingestion/query tests remain independently runnable.

API references: [StructuredTool](https://reference.langchain.com/python/langchain-core/tools/structured/StructuredTool/from_function)
and [ChatOpenAI tool calling](https://docs.langchain.com/oss/python/integrations/chat/openai).
