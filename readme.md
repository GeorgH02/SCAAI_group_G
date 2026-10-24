# Vienna Stay Assistant

Vienna accommodation project using the **2026-09-17** snapshot from
[Inside Airbnb](https://insideairbnb.com/get-the-data/) ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
SQL/vector ingestion and [read-only query tools](Docs/query_tools.md) are implemented;
a [LangChain evidence workflow](Docs/langchain.md) is available.
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

Start an installed llama.cpp server with `.\scripts\start_llm.ps1` (Windows)
or `bash scripts/start_llm.sh` (Linux). See [local LLM setup](Docs/llama_server.md)
for installation and options. Stop the server with Ctrl+C.

## Database access (HeidiSQL or similar)

With Docker services running, create a **PostgreSQL (TCP/IP)** connection:

| Setting | Value |
| --- | --- |
| Host | `127.0.0.1` |
| User | `postgres` |
| Password | `postgres` |
| Port | `5432` |
| Database | `vienna_stay` |
| Schema | `vienna_stay_assistant` |

Expand `vienna_stay_assistant` to see the project tables. Qdrant has a separate
[dashboard](http://localhost:6333/dashboard).

## Tests and shutdown

```powershell
py -3.11 -m unittest discover -s tests -v
docker compose down
```

Shutdown preserves stored data. Adding `-v` deletes the database volumes.
