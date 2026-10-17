# Vienna Stay Assistant

Vienna accommodation assistant using the **2026-09-17** snapshot from
[Inside Airbnb](https://insideairbnb.com/get-the-data/), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Requires **Python 3.11** and **Docker with Docker Compose**. Run from the project directory in PowerShell.

After setup, open [run_pipeline.ipynb](run_pipeline.ipynb) in VS Code or Jupyter,
select the `.venv` Python kernel, start Docker Desktop, and choose **Run All**.
It starts the databases, downloads data, and loads vectors in order.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
docker compose up -d
```

## Download and load data

```powershell
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe load_vectors.py
```

The loader masks PII and embeds listing descriptions and reviews from the last
24 months using BGE-M3. The first run downloads model weights. Successful loads
write the collection name and summary to `derived/vectors/`.
Use `load_vectors.py --help` for options.
Embedding automatically uses CUDA when available, otherwise CPU. Override with
`--device cpu` or `--device cuda`; the selected device is printed at startup.

## Local databases

- PostgreSQL/PostGIS: `localhost:5432`, database `vienna_stay`, user/password `postgres`.
- Qdrant: [dashboard](http://localhost:6333/dashboard).

The chat application and SQL ingestion are not implemented yet.

Stop services while keeping stored data:

```powershell
docker compose down
```

Stop services and deleting stored data:
```powershell
docker compose down -v
```

```powershell
docker compose up -d
```
