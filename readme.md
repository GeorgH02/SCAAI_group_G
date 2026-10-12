# Vienna Stay Assistant

Install Docker with Docker Compose, then run from the project directory:

```sh
docker compose up -d
docker compose ps
```

- PostgreSQL/PostGIS: `localhost:5432`, database `vienna_stay`, user `postgres`, password `postgres`.
- Qdrant: [dashboard](http://localhost:6333/dashboard), HTTP port `6333`, gRPC port `6334`; no API key required.

This starts the databases and creates an empty SQL schema. The chat application
and data ingestion are not implemented yet.

To download the source data:

```sh
python3 download_data.py
```

To stop the services while retaining database data:

```sh
docker compose down
```

Existing PostgreSQL volumes keep their original credentials; the defaults above
apply when initializing a new volume.
