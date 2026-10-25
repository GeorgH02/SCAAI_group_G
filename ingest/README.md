# Ingestion

Run from the project root with the Docker services up (`docker compose up -d postgres`) and the
`.venv` active. Raw files live in `data/` (gitignored).

## Run order

1. `python data_setup/download_data.py` - downloads the raw CSV/GeoJSON files to `data/`.
2. `python -m data_setup.load_sql [--replace]` - loads neighbourhoods, hosts, listings, reviews and
   calendar in one transaction (applies `docker/postgres/init/20-schema.sql` first). Masks PII with
   Presidio + spaCy (de/en), so the first run downloads/loads models. Use `--limit 50` for a sample.
3. `python -m ingest.noise --gpkg <path> --indicator road --layer <layer> --band-column <column>`, then the same
   with `--indicator rail`. Loads `noise_zones` and writes the worst band within 150 m to
   `listings.lnight_road_band` / `lnight_rail_band`. **Rerun after every `--replace` of step 2.**
   The GeoPackage path, layer and band column are not confirmed yet.

`noise.py` logs row counts before and after each step. `load_sql.py` reports final counts in
`derived/sql/<load_id>.json` and the `ingestion_runs` table. Both can be rerun safely.

`landmarks` (20 stations and sights, coordinates from OpenStreetMap) is seeded by `20-schema.sql`, which
`load_sql.py` applies on every run.

## Dropped or excluded rows

| Where | Rule | Counter |
| --- | --- | --- |
| calendar | `listing_id` has no parent listing (FK) | `orphan_calendar_rows_excluded` |
| calendar | listing not active (`number_of_reviews_ltm == 0`) | `calendar_inactive_listing_rows_excluded` |
| reviews | older than `--reviews-since` (default 2024-09-17) | `review_selection_*` |
| reviews | shorter than `--min-review-chars` (default 30) after masking | `short_reviews` |
| reviews | Airbnb auto-generated cancellation texts, conflicting duplicate IDs | `review_selection_*` |
| listings | missing price stays NULL (not dropped) | `listings_missing_price` |

## Known deviations from the project description

The local snapshot is 2026-09-17, not the 2026-06-20 snapshot behind the spec's counts. Measured on the
local files: 12,982 listings (spec 13,283), 650,325 reviews of which 267,439 in the window (spec 268,517),
4,738,430 calendar rows with no orphans (spec: 23 orphans). No listing field reproduces the spec's 8,542
active listings; "active" is defined as `number_of_reviews_ltm > 0` (8,394 listings, closest in share). Change
`is_active` in `data_setup/load_sql.py` if the group agrees on another definition.

## Tests

`python -m unittest discover -s tests -v`. Set `VSA_TEST_DATABASE_URL` to an isolated `vsa_test_*` PostGIS
database to include the SQL integration test. The noise spatial join has no automated database test yet.
