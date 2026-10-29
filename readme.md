# Vienna Stay Assistant

Vienna accommodation project using the **2026-09-17** snapshot from
[Inside Airbnb](https://insideairbnb.com/get-the-data/) ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
SQL/vector ingestion and [read-only query tools](docs/query_tools.md) are implemented;
a [LangChain evidence workflow](docs/langchain.md) is available.
The chat application is still planned.

## Setup and run

Requires **Python 3.11** and **Docker Desktop with Compose**. From the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start Docker Desktop, open [run_pipeline.ipynb](run_pipeline.ipynb), select the
`.venv` kernel, and choose **Run All**. It starts databases, downloads data, and
loads Qdrant vectors and PostgreSQL tables. First-time model downloads and full
imports can take time.

- `LIMIT = 10`: sample imports; use `None` for full imports.
- `SQL_REPLACE = False`: protects existing SQL data; set `True` to replace it.
- If vectors are already loaded, run configuration/startup cells, then the SQL section.

Start Ollama and download the model with `.\scripts\start_llm.ps1` (Windows)
or `bash scripts/start_llm.sh` (Linux). See [local LLM setup](docs/ollama.md).

With the server ready, open [try_agent.ipynb](try_agent.ipynb), select the
project kernel, and run its cells for a LangChain demo using fictional listings.

## Ollama with Docker Compose

From the project folder, start the container and download the model once:

```bash
docker compose up -d --wait --wait-timeout 120 ollama
docker compose exec -T ollama ollama pull qwen2.5:7b
```

The model download is about 4.7 GB and persists in the `ollama_data` volume.
This uses the [official Ollama image](https://docs.ollama.com/docker) with CPU
inference, a 16,384-token context and one parallel request. GPU acceleration
requires additional Docker/driver configuration. The image uses the `latest` tag;
pin a tested version or digest for reproducible deployments.

The Python client and demo notebook default to this model and endpoint:

```python
from langchain_agent.workflow import local_model
model = local_model()  # qwen2.5:7b at http://127.0.0.1:21434/v1
```

Check downloaded models with `docker compose exec ollama ollama list`, or try
an interactive chat with `docker compose exec ollama ollama run qwen2.5:7b`.
A healthy container means the API is running; model loading happens on the first
inference request. Stop it with `docker compose stop ollama`. The launch scripts
automate these same Compose commands; no host Ollama installation is needed.

For an NVIDIA GPU, start Ollama with the override file instead:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up -d ollama
```

## Guardrails

`guardrails/input_guard.py` checks each question (rules, topic gate, optional Llama Guard) and
`guardrails/evidence_guard.py` removes tool-result rows that contain injected instructions.
The topic gate uses `BAAI/bge-m3`, which downloads from Hugging Face on first use (about 2.3 GB).
Llama Guard is optional (Meta Llama 3.2 license, about 1.6 GB) and is switched on with `LLAMA_GUARD=1`:

```bash
docker compose exec -T ollama ollama pull llama-guard3:1b
python try_agent_db.py "Which apartments are within 500 m of Wien Westbahnhof?"
```

## Hybrid review search

`retrieval/hybrid.py` searches masked reviews with two signals and merges them:

1. **Dense:** bge-m3 query embedding → Qdrant nearest chunks (filtered by `listing_id` / `language`)
2. **Lexical:** PostgreSQL full text (`to_tsvector('simple', ...)`, OR over query words, stopwords removed)
3. **Fusion:** Reciprocal Rank Fusion, `score = Σ 1 / (60 + rank)`, keyed by `review_id`
4. **Hydration:** rows are loaded from PostgreSQL, which also applies the date window
5. **Rerank:** `BAAI/bge-reranker-v2-m3` cross-encoder on the top 30 (about 2.3 GB, downloads on first use)

Every row keeps `review_id`, `listing_id` and `chunk_id` for citations. If Qdrant or the reranker
is down, the search falls back to the remaining signals and says so in `notices`.

```bash
python try_hybrid.py "quiet at night"                    # search only, no LLM
python try_hybrid.py "sehr sauber" --listing-ids 123 456 --no-rerank
HYBRID=1 python try_agent_db.py                           # the LLM gets a hybrid_search_reviews tool
```

The collection is the newest complete manifest in `derived/vectors/` (override with `QDRANT_COLLECTION`).

## Database access (HeidiSQL or similar)

The default host ports avoid the existing services on this Docker host:

| Service | Host port | Container port |
| --- | --- | --- |
| PostgreSQL / PostGIS | `5433` | `5432` |
| Qdrant HTTP / dashboard | `6353` | `6333` |
| Qdrant gRPC | `6354` | `6334` |
| Ollama HTTP / OpenAI-compatible API | `21434` | `11434` |

All project services bind to `127.0.0.1`. Host clients and the notebooks use
these host ports; connections between Compose services use `postgres:5432`
and `qdrant:6333` (HTTP) or `qdrant:6334` (gRPC). Ollama is reachable by other
containers at `http://ollama:11434`; the current `local_model()` helper is intended
for notebooks on the host and accepts only loopback URLs.
If you previously set `DATABASE_URL` or `PGPORT`, update them to use port `5433`.


With Docker services running, create a **PostgreSQL (TCP/IP)** connection:

| Setting | Value |
| --- | --- |
| Host | `127.0.0.1` |
| User | `postgres` |
| Password | `postgres` |
| Port | `5433` |
| Database | `vienna_stay` |
| Schema | `vienna_stay_assistant` |

Expand `vienna_stay_assistant` to see the project tables. Qdrant has a separate
[dashboard](http://localhost:6353/dashboard).

## Tests and shutdown

```powershell
py -3.11 -m unittest discover -s tests -v
docker compose down
```

Shutdown preserves stored data. Adding `-v` deletes the database and model volumes.
